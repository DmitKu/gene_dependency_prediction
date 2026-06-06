# -*- coding: utf-8 -*-
"""
utils_RNAbased_crispr_model.py
===============
CRISPR Sensitivity Model v3 — Cross-Attention + Linear Bypass

All dataset, model, loss, and evaluation components.
Import this module from training scripts.

Architecture summary
--------------------
  Gene features  →  gene_encoder  →  gene_res         →  gene_emb     [B, gene_hidden]
  Cell features  →  cell_tokenizer                     →  cell_tokens  [B, n_slots, gene_hidden]

  1st cross-attention  (query=gene_emb)                →  cell_context  [B, gene_hidden]
  2nd cross-attention  (query=cell_context)            →  cell_context2 [B, gene_hidden]

  cell_summary  = 0.5 * attn1_weighted + 0.5 * attn2_weighted
  bypass_logit  = LinearBypass(gene_emb, cell_summary)

  x        = merge(cat[cell_context2, gene_emb])   →  [B, hidden_dim]
  gene_cond = cond_proj(gene_emb)                  →  [B, gene_hidden]

  x = trunk_res1/2/3(x, cond=gene_cond)
  output = head(x) + bypass_logit

Variance-Informed Loss Weighting
---------------------------------
  Each sample carries a per-gene biological variance weight computed at
  dataset load time from the FULL crispr array (all splits).

  For gene g:
      raw_var(g)  = variance of crispr scores for gene g across all cell lines
      weight(g)   = clip(raw_var(g), min=floor, max=cap) / mean_weight

  This weight is the 8th element of each __getitem__ tuple and is passed
  into combined_loss(gene_var_weight=...).  It is multiplied with the
  existing per-sample deviation weight:

      final_weight = deviation_weight * gene_var_weight  (then renormalised)

  Effect:
    - High-variance genes (KRAS, BRAF, context-dependent essentials) get
      stronger gradient signal.
    - Low-variance / flat genes get less gradient — less noise.
    - Zero-variance genes are floored at var_weight_floor (default 0.1).

  Caller change (training script)
  ---------------------------------
      gene_feat, cell_feat, target, cl_idx, _, _, _, gene_var_w = batch
      loss, ... = combined_loss(pred, target, cl_idx,
                                gene_var_weight=gene_var_w, ...)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.amp import autocast
from collections import defaultdict
import h5py
import numpy as np


# ============================================================
# Dataset
# ============================================================

class GeneDataset(Dataset):
    """
    Loads gene-level CRISPR sensitivity data from an HDF5 file.

    Expected HDF5 layout
    --------------------
    index/splits/{train,val,test}  : 1-D integer index array
    cell_lines/features            : float32 array [n_cell_lines, F]
    cell_lines/model_id            : bytes array   [n_cell_lines]
    genes/features                 : float32 array [n_genes, G]
    genes/model_id                 : bytes array   [n_genes]
    genes/gene_id                  : bytes array   [n_genes]
    genes/crispr                   : float32 array [n_genes]

    Returns (per __getitem__)
    -------------------------
    [0] gene_feat       : Tensor [G]
    [1] cell_feat       : Tensor [F]
    [2] crispr          : Tensor [1]
    [3] cl_idx          : LongTensor scalar
    [4] gene_id         : str
    [5] model_id        : str
    [6] idx             : int
    [7] gene_var_weight : float32 scalar Tensor  ← NEW
                          Biological variance weight for this gene.
                          Pass to combined_loss(gene_var_weight=...).

    Parameters
    ----------
    var_weight_floor : float
        Minimum variance weight (before normalisation).  Default 0.1.
    var_weight_cap : float
        Maximum variance weight (before normalisation).  Default 5.0.
        Prevents a handful of extreme outlier genes from dominating.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        var_weight_floor: float = 0.1,
        var_weight_cap:   float = 5.0,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        print(f"Loading {split} data ...")
        t0 = time.time()

        with h5py.File(h5_path, "r") as f:
            gene_indices      = f[f"index/splits/{split}"][:]
            self.cl_features  = torch.tensor(f["cell_lines/features"][:], dtype=torch.float32)
            cl_model_ids      = f["cell_lines/model_id"][:]
            all_gene_feat     = f["genes/features"][:]
            all_model_ids     = f["genes/model_id"][:]
            all_gene_ids      = f["genes/gene_id"][:]
            all_crispr        = f["genes/crispr"][:]   # full array — needed for variance

            self.gene_feat = torch.tensor(all_gene_feat[gene_indices], dtype=torch.float32)
            self.crispr    = torch.tensor(all_crispr[gene_indices],    dtype=torch.float32)

            self.gene_ids  = [all_gene_ids[i].decode()  for i in gene_indices]
            self.model_ids = [all_model_ids[i].decode() for i in gene_indices]

        # ── Cell-line lookup ─────────────────────────────────────────────────
        self.cl_model_id_to_index = {mid.decode(): i for i, mid in enumerate(cl_model_ids)}
        self.cl_indices = torch.tensor(
            [self.cl_model_id_to_index[mid] for mid in self.model_ids],
            dtype=torch.long,
        )
        self.cl_index_to_model_id = {v: k for k, v in self.cl_model_id_to_index.items()}

        # ── Gene index lookup ────────────────────────────────────────────────
        # Each unique gene_id gets a stable integer index.
        # This is used by demeaned_pearson (now per-gene) and combined_loss.
        # Sorted for determinism across splits.
        unique_gene_ids        = sorted(set(self.gene_ids))
        self.gene_id_to_index  = {gid: i for i, gid in enumerate(unique_gene_ids)}
        self.n_genes           = len(unique_gene_ids)
        self.gene_indices      = torch.tensor(
            [self.gene_id_to_index[gid] for gid in self.gene_ids],
            dtype=torch.long,
        )

        # ── Variance-informed weights ────────────────────────────────────────
        # Computed from the FULL crispr array (all splits) so every split gets
        # the same biological-signal estimate for each gene.  This is not label
        # leakage: knowing "KRAS is highly variable" is biological prior
        # knowledge, not information about specific held-out target values.
        self.gene_var_weights = self._compute_gene_var_weights(
            all_gene_ids   = all_gene_ids,
            all_crispr     = all_crispr,
            split_gene_ids = self.gene_ids,
            floor          = var_weight_floor,
            cap            = var_weight_cap,
        )

        # ── Duplicate check ──────────────────────────────────────────────────
        pairs    = list(zip(self.gene_ids, self.model_ids))
        n_unique = len(set(pairs))
        if n_unique != len(pairs):
            import warnings
            warnings.warn(
                f"Split '{split}': {len(pairs) - n_unique} duplicate "
                f"(gene_id, model_id) pairs found — check data pipeline"
            )

        w = self.gene_var_weights
        print(
            f"  -> {len(self.gene_feat):,} samples | "
            f"{len(set(self.gene_ids)):,} genes | "
            f"{len(set(self.model_ids)):,} cell lines | "
            f"loaded in {time.time() - t0:.2f}s\n"
            f"  -> gene_var_weight  min={w.min():.3f}  "
            f"mean={w.mean():.3f}  max={w.max():.3f}  "
            f"(floor={var_weight_floor}  cap={var_weight_cap})"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_gene_var_weights(
        all_gene_ids:   np.ndarray,
        all_crispr:     np.ndarray,
        split_gene_ids: list,
        floor:          float,
        cap:            float,
    ) -> torch.Tensor:
        """
        Compute per-sample gene variance weights for this split using
        rank-based weighting.

        Why rank-based rather than raw-variance:
        -----------------------------------------
        Raw variance has two problems:
          1. The cap (e.g. 5.0) was applied to raw variance values before
             normalisation, so after dividing by the mean the post-normalisation
             max could exceed cap (observed: cap=5.0 → max=8.376).
          2. The absolute scale of variance values is arbitrary and dataset-
             dependent, making floor/cap hard to reason about intuitively.

        Rank-based weighting fixes both:
          - floor and cap are weights in the final normalised space, not raw
            variance thresholds. floor=0.1, cap=5.0 means the flattest gene
            always gets weight 0.1/mean and the most variable always gets
            weight 5.0/mean. Guaranteed, regardless of variance distribution.
          - The ranking is computed on UNIQUE genes, then mapped to samples.
            This is critical: ranking the per-sample array directly would give
            genes with more cell-line measurements multiple rank entries and
            corrupt the ordering.

        Steps
        -----
        1. Group all crispr scores by gene_id (full HDF5 array, all splits).
        2. Compute variance per unique gene (ddof=1).
        3. Rank unique genes by variance (lowest = rank 1).
        4. Map rank → weight linearly: weight = floor + (cap - floor) * rank_norm
           where rank_norm = (rank - 1) / (n_unique_genes - 1)  ∈ [0, 1].
        5. Build a gene_id → weight lookup and map to split samples.
        6. Normalise so mean = 1.0 (keeps loss magnitude stable).
        """
        from scipy.stats import rankdata

        # ── Step 1: group scores by gene_id ─────────────────────────────────
        gene_scores: dict = defaultdict(list)
        for gid_bytes, score in zip(all_gene_ids, all_crispr):
            gene_scores[gid_bytes.decode()].append(score)

        # ── Step 2: variance per unique gene ────────────────────────────────
        unique_genes = list(gene_scores.keys())
        gene_variances = np.array(
            [
                float(np.array(gene_scores[g], dtype=np.float32).var(ddof=1))
                if len(gene_scores[g]) > 1 else 0.0
                for g in unique_genes
            ],
            dtype=np.float32,
        )

        # ── Step 3: rank unique genes by variance ────────────────────────────
        # rankdata assigns 1 to the smallest, n to the largest.
        # method='average' handles ties (same variance → same weight).
        ranks = rankdata(gene_variances, method='average')  # [n_unique_genes]

        # ── Step 4: map rank → weight linearly in [floor, cap] ───────────────
        # rank_norm ∈ [0, 1]:  0 = flattest gene,  1 = most variable gene
        n_unique = len(unique_genes)
        if n_unique > 1:
            rank_norm = (ranks - 1.0) / (n_unique - 1.0)
        else:
            rank_norm = np.ones(n_unique, dtype=np.float32) * 0.5

        weights_unique = floor + (cap - floor) * rank_norm  # [n_unique_genes]

        # ── Step 5: build lookup and map to split samples ────────────────────
        gene_weight_lookup = dict(zip(unique_genes, weights_unique))
        # Fallback to floor for any gene not seen in the full array
        sample_weights = np.array(
            [gene_weight_lookup.get(gid, floor) for gid in split_gene_ids],
            dtype=np.float32,
        )

        # ── Step 6: normalise so mean = 1.0 ─────────────────────────────────
        mean_w = sample_weights.mean()
        if mean_w > 1e-8:
            sample_weights = sample_weights / mean_w

        return torch.tensor(sample_weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gene_feat)

    def __getitem__(self, idx):
        return (
            self.gene_feat[idx],
            self.cl_features[self.cl_indices[idx]],
            self.crispr[idx].unsqueeze(0),
            self.cl_indices[idx],           # [3] LongTensor scalar — cell line index
            self.gene_ids[idx],             # [4] str — gene identifier
            self.model_ids[idx],            # [5] str — cell-line model_id
            idx,                            # [6] int — position within split
            self.gene_var_weights[idx],     # [7] float32 scalar — variance weight
            self.gene_indices[idx],         # [8] LongTensor scalar — gene index ← NEW
        )


# ============================================================
# Building blocks
# ============================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: out = (1 + gamma) * x + beta"""

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class GELUResidualBlock(nn.Module):
    """Pre-LayerNorm residual block (optionally FiLM-conditioned)."""

    def __init__(self, dim: int, dropout: float = 0.2, cond_dim: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.film = FiLMLayer(cond_dim, dim) if cond_dim > 0 else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        out = self.net(self.norm(x))
        if self.film is not None and cond is not None:
            out = self.film(out, cond)
        return out + x


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization (no mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)


class CellTokenizer(nn.Module):
    def __init__(
        self,
        cell_feat_dim: int,
        n_slots:       int,
        d_model:       int,
        compress_dim:  int   = 1024,
        gene_dim:      int   = 128,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_model = d_model

        self.compress = nn.Sequential(
            nn.Linear(cell_feat_dim, compress_dim),
            nn.LayerNorm(compress_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.film       = FiLMLayer(cond_dim=gene_dim, feature_dim=compress_dim)
        self.tokenize   = nn.Linear(compress_dim, n_slots * d_model)
        self.token_norm = RMSNorm(d_model)
        self.slot_embed = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)

    def forward(self, x: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z = self.compress(x)
        z = self.film(z, gene_emb)
        t = self.tokenize(z).reshape(B, self.n_slots, self.d_model)
        return self.token_norm(t) + self.slot_embed


class LinearBypass(nn.Module):
    def __init__(self, gene_dim: int, cell_dim: int, rank: int = 32, reg_scale: float = 1e-3):
        super().__init__()
        self.gene_proj = nn.Linear(gene_dim, rank, bias=False)
        self.cell_proj = nn.Linear(cell_dim, rank, bias=False)
        self.scale     = nn.Parameter(torch.ones(1) * 0.5)
        self.gate      = nn.Parameter(torch.tensor(-2.0))
        self.reg_scale = reg_scale
        nn.init.normal_(self.gene_proj.weight, 0, 0.01)
        nn.init.normal_(self.cell_proj.weight, 0, 0.01)

    def forward(self, gene_emb, cell_summary):
        g    = F.normalize(self.gene_proj(gene_emb), dim=-1)
        c    = F.normalize(self.cell_proj(cell_summary), dim=-1)
        raw  = self.scale * (g * c).sum(-1, keepdim=True)
        gate = torch.sigmoid(self.gate)
        reg  = raw.pow(2).mean() * self.reg_scale
        return gate * raw, reg


# ============================================================
# Main model  (unchanged from original v3)
# ============================================================

class CRISPRSensitivityModelV3(nn.Module):
    """CRISPR sensitivity predictor — v3."""

    def __init__(
        self,
        cell_features_size: int   = 2388,
        gene_features_size: int   = 26,
        hidden_dim:         int   = 128,
        gene_hidden:        int   = 64,
        n_attn_slots:       int   = 64,
        n_attn_heads:       int   = 4,
        bypass_rank:        int   = 32,
        compress_dim:       int   = 512,
        dropout:            float = 0.2,
    ):
        super().__init__()

        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_features_size, gene_hidden),
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(gene_hidden, gene_hidden),
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
        )
        self.gene_res = GELUResidualBlock(gene_hidden, dropout=0.1)

        self.cell_tokenizer = CellTokenizer(
            cell_feat_dim = cell_features_size,
            n_slots       = n_attn_slots,
            d_model       = gene_hidden,
            compress_dim  = compress_dim,
            gene_dim      = gene_hidden,
            dropout       = dropout,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm  = nn.LayerNorm(gene_hidden)

        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm2  = nn.LayerNorm(gene_hidden)

        self.linear_bypass = LinearBypass(
            gene_dim=gene_hidden, cell_dim=gene_hidden, rank=bypass_rank,
        )

        self.merge = nn.Sequential(
            nn.Linear(gene_hidden * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(gene_hidden * 2, gene_hidden),
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
        )

        self.trunk_res1 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)
        self.trunk_res2 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)
        self.trunk_res3 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self.out_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.out_shift = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def forward(
        self,
        cell_features: torch.Tensor,
        gene_features: torch.Tensor,
        ablate_bypass: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gene_emb    = self.gene_res(self.gene_encoder(gene_features))
        cell_tokens = self.cell_tokenizer(cell_features, gene_emb)

        attn_out1, attn_weights1 = self.cross_attn(
            gene_emb.unsqueeze(1), cell_tokens, cell_tokens,
            need_weights=True, average_attn_weights=True,
        )
        cell_context = self.attn_norm(attn_out1.squeeze(1) + gene_emb)

        attn_out2, attn_weights2 = self.cross_attn2(
            cell_context.unsqueeze(1), cell_tokens, cell_tokens,
            need_weights=True, average_attn_weights=True,
        )
        cell_context2 = self.attn_norm2(attn_out2.squeeze(1) + cell_context)

        cell_summary1 = (attn_weights1.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
        cell_summary2 = (attn_weights2.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
        cell_summary  = 0.5 * cell_summary1 + 0.5 * cell_summary2

        bypass_logit, bypass_reg = self.linear_bypass(gene_emb, cell_summary)
        if ablate_bypass:
            bypass_logit = torch.zeros_like(bypass_logit)
            bypass_reg   = torch.tensor(0.0, device=gene_emb.device)

        x_input   = torch.cat([cell_context2, gene_emb], dim=-1)
        x         = self.merge(x_input)
        gene_cond = self.cond_proj(torch.cat([gene_emb, cell_context2], dim=-1))
        x = self.trunk_res1(x, cond=gene_cond)
        x = self.trunk_res2(x, cond=gene_cond)
        x = self.trunk_res3(x, cond=gene_cond)
        trunk_pred = self.head(x)

        raw = trunk_pred + bypass_logit
        return raw * self.out_scale + self.out_shift, bypass_reg

    def _init_weights(self):
        bypass_ids = {
            id(self.linear_bypass.gene_proj.weight),
            id(self.linear_bypass.cell_proj.weight),
        }
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if id(m.weight) in bypass_ids:
                    continue
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ============================================================
# Loss functions
# ============================================================

def differentiable_pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Raw Pearson — dominated by gene main effect."""
    pred   = pred.view(-1)
    target = target.view(-1)
    pm = pred   - pred.mean()
    pt = target - target.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def demeaned_pearson(
    pred:     torch.Tensor,
    target:   torch.Tensor,
    gene_idx: torch.Tensor,
) -> torch.Tensor:
    # Force float32 regardless of autocast — prevents float16 overflow
    pred     = pred.view(-1).float()
    target   = target.view(-1).float()
    gene_idx = gene_idx.view(-1)

    n_genes = int(gene_idx.max().item()) + 1

    counts = torch.zeros(n_genes, device=pred.device, dtype=torch.float32)
    counts.scatter_add_(0, gene_idx, torch.ones_like(pred))

    pred_sum   = torch.zeros(n_genes, device=pred.device, dtype=torch.float32)
    target_sum = torch.zeros(n_genes, device=pred.device, dtype=torch.float32)
    pred_sum.scatter_add_(0, gene_idx, pred)
    target_sum.scatter_add_(0, gene_idx, target)

    gene_pred_mean   = (pred_sum   / counts.clamp(min=1))[gene_idx]
    gene_target_mean = (target_sum / counts.clamp(min=1))[gene_idx]

    pred_dm   = pred   - gene_pred_mean
    target_dm = target - gene_target_mean

    cross     = torch.zeros(n_genes, device=pred.device, dtype=torch.float32)
    pred_sq   = torch.zeros(n_genes, device=pred.device, dtype=torch.float32)
    target_sq = torch.zeros(n_genes, device=pred.device, dtype=torch.float32)
    cross.scatter_add_(0,     gene_idx, pred_dm * target_dm)
    pred_sq.scatter_add_(0,   gene_idx, pred_dm.pow(2))
    target_sq.scatter_add_(0, gene_idx, target_dm.pow(2))

    valid = counts >= 2
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

    # clamp(min=0) before sqrt prevents NaN from float16 rounding artifacts
    denom      = (pred_sq[valid].clamp(min=0).sqrt() *
                  target_sq[valid].clamp(min=0).sqrt()).clamp(min=1e-8)
    per_gene_r = cross[valid] / denom

    # Final NaN guard — if anything slipped through, replace with 0
    per_gene_r = torch.nan_to_num(per_gene_r, nan=0.0, posinf=0.0, neginf=0.0)

    return per_gene_r.mean()


def combined_loss(
    pred:            torch.Tensor,
    target:          torch.Tensor,
    cl_idx:          torch.Tensor,
    gene_idx:        torch.Tensor,            # ← NEW: integer gene index per sample
    alpha:           float             = 0.5,
    beta:            float             = 0.4,
    gene_var_weight: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined MSE + global Pearson + mean per-gene Pearson loss.

    gene_idx : Tensor [B]
        Integer gene index for each sample (GeneDataset.gene_indices, batch pos 8).
        Passed to demeaned_pearson which now computes mean per-gene Pearson
        across cell lines — the same metric as the R evaluation script.

    gene_var_weight : Tensor [B], optional
        Per-sample biological variance weight (batch pos 7).
        Multiplied with the deviation-based weight, then renormalised.
    """
    pred_s   = pred.view(-1)
    target_s = target.view(-1)

    with torch.no_grad():
        target_mean = target_s.mean()
        deviation   = (target_s - target_mean).abs()
        deviation_w = 1.0 + 2.0 * (deviation / (deviation.max() + 1e-8))

        if gene_var_weight is not None:
            gvw     = gene_var_weight.view(-1).to(pred_s.device, dtype=pred_s.dtype)
            weights = deviation_w * gvw
        else:
            weights = deviation_w

        weights = weights / (weights.mean() + 1e-8)

    mse_term         = (weights * (pred_s - target_s) ** 2).mean()
    pearson_r        = differentiable_pearson(pred_s, target_s)
    pearson_demeaned = demeaned_pearson(pred_s, target_s, gene_idx)   # ← gene_idx not cl_idx

    loss = (
        alpha                * mse_term
        + (1 - alpha - beta) * (1 - pearson_r)
        + beta               * (1 - pearson_demeaned)
    )
    return loss, mse_term, pearson_r, pearson_demeaned


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model:         nn.Module,
    loader,
    device:        str,
    qt=None,
    alpha:         float = 0.5,
    ablate_bypass: bool  = False,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    High-performance evaluation using GPU-native vectorization.

    gene_var_weight is unpacked from batch position 7 and forwarded to
    combined_loss so the validation loss uses the same weighted objective
    as training.
    """
    model.eval()
    all_pred, all_target, all_cl_idx = [], [], []
    total_loss = total_mse = total_pearson = 0.0

    device_type = torch.device(device).type

    with torch.no_grad():
        for gene_feat, cell_feat, target, cl_idx, _, _, _, gene_var_w, gene_idx in loader:
            gene_feat  = gene_feat.to(device, non_blocking=True)
            cell_feat  = cell_feat.to(device, non_blocking=True)
            target_d   = target.to(device, non_blocking=True)
            cl_idx     = cl_idx.to(device, non_blocking=True)
            gene_var_w = gene_var_w.to(device, non_blocking=True)
            gene_idx   = gene_idx.to(device, non_blocking=True)

            with autocast(device_type=device_type):
                pred, _ = model(cell_feat, gene_feat, ablate_bypass=ablate_bypass)
                loss, mse_term, pearson_r, _ = combined_loss(
                    pred, target_d, cl_idx, gene_idx,
                    alpha=alpha,
                    gene_var_weight=gene_var_w,
                )

            all_pred.append(pred.view(-1))
            all_target.append(target_d.view(-1))
            all_cl_idx.append(cl_idx.view(-1))

            total_loss    += loss.item()
            total_mse     += mse_term.item()
            total_pearson += pearson_r.item()

    n = len(loader)
    val_loss    = total_loss    / n
    val_mse     = total_mse     / n
    val_pearson = total_pearson / n

    eval_pred   = torch.cat(all_pred)
    eval_target = torch.cat(all_target)
    cl_idx      = torch.cat(all_cl_idx)

    if qt is not None:
        pred_np   = qt.inverse_transform(eval_pred.cpu().numpy().reshape(-1, 1)).squeeze()
        target_np = qt.inverse_transform(eval_target.cpu().numpy().reshape(-1, 1)).squeeze()
        eval_pred   = torch.tensor(pred_np,   device=device, dtype=torch.float32)
        eval_target = torch.tensor(target_np, device=device, dtype=torch.float32)

    mae  = (eval_pred - eval_target).abs().mean().item()
    rmse = ((eval_pred - eval_target) ** 2).mean().sqrt().item()

    pm = eval_pred   - eval_pred.mean()
    pt = eval_target - eval_target.mean()
    pearson_full = ((pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)).item()

    n_cl   = cl_idx.max() + 1
    counts = torch.zeros(n_cl, device=device).scatter_add_(
        0, cl_idx, torch.ones_like(eval_pred)
    ).clamp(min=1)
    p_sums = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, eval_pred)
    t_sums = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, eval_target)

    pred_dm   = eval_pred   - (p_sums / counts)[cl_idx]
    target_dm = eval_target - (t_sums / counts)[cl_idx]

    pm_d = pred_dm   - pred_dm.mean()
    pt_d = target_dm - target_dm.mean()
    pearson_full_demeaned = (
        (pm_d * pt_d).sum() / (pm_d.norm() * pt_d.norm() + 1e-8)
    ).item()

    mask         = counts >= 10
    valid_cl_ids = torch.nonzero(mask).view(-1)

    cl_pearsons = []
    if valid_cl_ids.numel() > 0:
        for cid in valid_cl_ids:
            cl_mask = (cl_idx == cid)
            p_cl    = eval_pred[cl_mask]
            t_cl    = eval_target[cl_mask]
            pm_cl   = p_cl - p_cl.mean()
            pt_cl   = t_cl - t_cl.mean()
            denom   = pm_cl.norm() * pt_cl.norm()
            if denom > 1e-8:
                cl_pearsons.append((pm_cl * pt_cl).sum() / denom)

    if cl_pearsons:
        cl_t    = torch.stack(cl_pearsons)
        cl_mean = cl_t.mean().item()
        cl_std  = cl_t.std().item() if cl_t.numel() > 1 else 0.0
    else:
        cl_mean, cl_std = 0.0, 0.0

    return (
        val_loss, val_mse, val_pearson,
        mae, rmse,
        pearson_full,
        pearson_full_demeaned,
        cl_mean, cl_std,
    )


# ============================================================
# Diagnostics
# ============================================================

def diagnose_bypass(
    model:     nn.Module,
    loader,
    device:    str,
    n_batches: int = 5,
) -> None:
    """Print bypass vs trunk head contribution.

    Healthy model: bypass/head ratio < 1.0, attn entropy > 2.0
    Collapse risk: bypass/head ratio > 2.0, attn entropy < 0.5
    """
    model.eval()
    bypass_mags, head_mags, attn_entropies = [], [], []

    with torch.no_grad():
        for i, (gene_feat, cell_feat, target, cl_idx, gene_id, model_id, _, _, _) in enumerate(loader):
            if i >= n_batches:
                break
            gene_feat = gene_feat.to(device)
            cell_feat = cell_feat.to(device)

            gene_emb    = model.gene_res(model.gene_encoder(gene_feat))
            cell_tokens = model.cell_tokenizer(cell_feat, gene_emb)

            attn_out1, weights1 = model.cross_attn(
                gene_emb.unsqueeze(1), cell_tokens, cell_tokens,
                need_weights=True, average_attn_weights=True,
            )
            cell_context = model.attn_norm(attn_out1.squeeze(1) + gene_emb)

            attn_out2, weights2 = model.cross_attn2(
                cell_context.unsqueeze(1), cell_tokens, cell_tokens,
                need_weights=True, average_attn_weights=True,
            )
            cell_context2 = model.attn_norm2(attn_out2.squeeze(1) + cell_context)

            cell_summary1 = (weights1.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
            cell_summary2 = (weights2.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
            cell_summary  = 0.5 * cell_summary1 + 0.5 * cell_summary2

            bypass, _ = model.linear_bypass(gene_emb, cell_summary)

            x         = model.merge(torch.cat([cell_context2, gene_emb], dim=-1))
            gene_cond = model.cond_proj(torch.cat([gene_emb, cell_context2], dim=-1))
            x         = model.trunk_res1(x, cond=gene_cond)
            x         = model.trunk_res2(x, cond=gene_cond)
            x         = model.trunk_res3(x, cond=gene_cond)
            head_out  = model.head(x)

            bypass_mags.append((bypass   * model.out_scale).abs().mean().item())
            head_mags.append(  (head_out * model.out_scale).abs().mean().item())

            w1      = weights1.squeeze(1)
            entropy = -(w1 * (w1 + 1e-8).log()).sum(-1).mean()
            attn_entropies.append(entropy.item())

    bypass_mean  = sum(bypass_mags)    / len(bypass_mags)
    head_mean    = sum(head_mags)      / len(head_mags)
    entropy_mean = sum(attn_entropies) / len(attn_entropies)
    ratio        = bypass_mean / (head_mean + 1e-8)

    print(f"\n-- Bypass Diagnostic -------------------------------------------")
    print(f"  Bypass magnitude  : {bypass_mean:.4f}")
    print(f"  Head magnitude    : {head_mean:.4f}")
    print(f"  Bypass/Head ratio : {ratio:.3f}  (healthy: <1.0  collapse: >2.0)")
    print(f"  Attn entropy      : {entropy_mean:.4f}  (healthy: >2.0  collapsed: <0.5)")
    print(f"----------------------------------------------------------------\n")