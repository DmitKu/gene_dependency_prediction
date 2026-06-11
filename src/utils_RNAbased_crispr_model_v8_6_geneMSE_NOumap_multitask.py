# -*- coding: utf-8 -*-
"""
utils_RNAbased_crispr_model.py
===============
CRISPR Sensitivity Model v3 — Cross-Attention + Linear Bypass + Multi-task

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
  gene_cond = cond_proj(cat[gene_emb, cell_context2]) →  [B, gene_hidden]

  x = trunk_res1/2/3(x, cond=gene_cond)
  output       = head(x) + bypass_logit            regression output
  sens_logit   = sensitivity_head(cell_context2)   auxiliary classification output

Batch tuple positions
---------------------
  [0] gene_feat        Tensor [G]
  [1] cell_feat        Tensor [F]
  [2] crispr           Tensor [1]
  [3] cl_idx           LongTensor scalar
  [4] gene_id          str
  [5] model_id         str
  [6] idx              int
  [7] gene_var_weight  float32 scalar Tensor
  [8] gene_int_idx     LongTensor scalar
  [9] sensitive_label  float32 scalar Tensor  ← NEW (multi-task)
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
    [0] gene_feat        : Tensor [G]
    [1] cell_feat        : Tensor [F]
    [2] crispr           : Tensor [1]
    [3] cl_idx           : LongTensor scalar
    [4] gene_id          : str
    [5] model_id         : str
    [6] idx              : int
    [7] gene_var_weight  : float32 scalar Tensor
    [8] gene_int_idx     : LongTensor scalar — integer gene index
    [9] sensitive_label  : float32 scalar Tensor — 1.0 if below gene-specific p10

    Parameters
    ----------
    var_weight_floor : float
        Minimum variance weight (before normalisation).  Default 0.05.
    var_weight_cap : float
        Maximum variance weight (before normalisation).  Default 8.0.
        Prevents a handful of extreme outlier genes from dominating.
    sensitivity_percentile : float
        Gene-specific percentile threshold below which a cell line is labelled
        "sensitive" (class 1).  Default 0.10 (bottom 10 % per gene).
        Must be gene-specific — a global threshold would just recover broadly
        essential genes, which the gene features already encode.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        var_weight_floor: float = 0.05,
        var_weight_cap:   float = 8.0,
        sensitivity_percentile: float = 0.10,
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

        # ── Variance-informed weights ─────────────────────────────────────────
        self.gene_var_weights = self._compute_gene_var_weights(
            all_gene_ids   = all_gene_ids,
            all_crispr     = all_crispr,
            split_gene_ids = self.gene_ids,
            floor          = var_weight_floor,
            cap            = var_weight_cap,
        )

        # ── Duplicate check ───────────────────────────────────────────────────
        pairs    = list(zip(self.gene_ids, self.model_ids))
        n_unique = len(set(pairs))
        if n_unique != len(pairs):
            import warnings
            warnings.warn(
                f"Split '{split}': {len(pairs) - n_unique} duplicate "
                f"(gene_id, model_id) pairs found — check data pipeline"
            )

        # ── Integer gene index (for gene-demeaned loss) ───────────────────────
        unique_genes      = sorted(set(self.gene_ids))
        self._gene_to_int = {g: i for i, g in enumerate(unique_genes)}
        self.gene_int_idx = torch.tensor(
            [self._gene_to_int[g] for g in self.gene_ids],
            dtype=torch.long,
        )

        # ── Gene-specific sensitivity labels (multi-task) ────────────────────
        # Threshold is per-gene so the model learns cell-line sensitivity
        # *relative to that gene's typical range*, not absolute essentiality.
        # A global threshold would just recover broadly essential genes, which
        # the gene features already encode and buys nothing new.
        self.sensitive_labels = self._compute_sensitivity_labels(
            crispr      = self.crispr,
            gene_ids    = self.gene_ids,
            percentile  = sensitivity_percentile,
        )

        w   = self.gene_var_weights
        n_s = int(self.sensitive_labels.sum().item())
        print(
            f"  -> {len(self.gene_feat):,} samples | "
            f"{len(set(self.gene_ids)):,} genes | "
            f"{len(set(self.model_ids)):,} cell lines | "
            f"loaded in {time.time() - t0:.2f}s\n"
            f"  -> gene_var_weight  min={w.min():.3f}  "
            f"mean={w.mean():.3f}  max={w.max():.3f}  "
            f"(floor={var_weight_floor}  cap={var_weight_cap})\n"
            f"  -> sensitive_labels positives={n_s:,} / {len(self.gene_feat):,} "
            f"({100.*n_s/len(self.gene_feat):.1f}%)  "
            f"[gene-specific p{sensitivity_percentile*100:.0f} threshold]\n"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_sensitivity_labels(
        crispr:     torch.Tensor,
        gene_ids:   list,
        percentile: float,
    ) -> torch.Tensor:
        """
        Binary label: 1.0 if sample is below the gene-specific `percentile`
        threshold, 0.0 otherwise.

        Uses gene_ids from the split only (not full dataset) so each split's
        threshold is computed from its own distribution.  This is intentional:
        val/test percentiles are computed from their own data so the label
        is consistent with what the model sees during evaluation.
        """
        gene_to_indices: dict = defaultdict(list)
        for i, gid in enumerate(gene_ids):
            gene_to_indices[gid].append(i)

        labels = torch.zeros(len(gene_ids), dtype=torch.float32)

        for gid, indices in gene_to_indices.items():
            idx_t  = torch.tensor(indices, dtype=torch.long)
            vals   = crispr[idx_t]
            if vals.numel() < 2:
                # Only one observation — can't define a meaningful percentile;
                # leave label as 0 (not sensitive by default).
                continue
            threshold = torch.quantile(vals, percentile)
            for i in indices:
                if crispr[i] <= threshold:
                    labels[i] = 1.0

        return labels

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
        Compute per-sample gene variance weights for this split.

        1. Group all crispr scores by gene_id (full dataset).
        2. Compute variance per gene (ddof=1).
        3. Look up the variance for each sample in this split.
        4. Clip to [floor, cap] then normalise so mean = 1.0.
        """
        gene_scores: dict = defaultdict(list)
        for gid_bytes, score in zip(all_gene_ids, all_crispr):
            gene_scores[gid_bytes.decode()].append(score)

        gene_var: dict = {}
        for gid, scores in gene_scores.items():
            arr = np.array(scores, dtype=np.float32)
            gene_var[gid] = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0

        raw = np.array(
            [gene_var.get(gid, 0.0) for gid in split_gene_ids],
            dtype=np.float32,
        )

        raw = np.clip(raw, floor, cap)
        mean_w = raw.mean()
        if mean_w > 1e-8:
            raw = raw / mean_w

        return torch.tensor(raw, dtype=torch.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gene_feat)

    def __getitem__(self, idx):
        cl_i = self.cl_indices[idx]
        return (
            self.gene_feat[idx],                   # [0] Tensor [G]
            self.cl_features[cl_i],                # [1] Tensor [F]
            self.crispr[idx].unsqueeze(0),         # [2] Tensor [1]
            cl_i,                                  # [3] LongTensor scalar
            self.gene_ids[idx],                    # [4] str
            self.model_ids[idx],                   # [5] str
            idx,                                   # [6] int
            self.gene_var_weights[idx],            # [7] float32 scalar Tensor
            self.gene_int_idx[idx],                # [8] LongTensor scalar
            self.sensitive_labels[idx],            # [9] float32 scalar Tensor  ← NEW
        )


# ============================================================
# Building blocks
# ============================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Applies per-feature affine transform gated by a conditioning vector.

        out = (1 + gamma) * x + beta
        where [gamma, beta] = Linear(cond)
    """

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class GELUResidualBlock(nn.Module):
    """
    Pre-LayerNorm residual block (optionally FiLM-conditioned).

    Pre-LN design
    -------------
    The LayerNorm is applied to the branch *input* before the linear
    projection, not to the branch output. This lets the residual stream
    carry large values across blocks (important for predicting extreme
    CRISPR scores) while still stabilising gradient flow.

        out = Linear(LayerNorm(x))       # branch
        if FiLM: out = FiLM(out, cond)
        return out + x                   # x is raw, no activation applied
    """

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
        self.film = FiLMLayer(cond_dim=gene_dim, feature_dim=compress_dim)

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
        self.gate      = nn.Parameter(torch.tensor(-2.0))  # starts near-closed
        self.reg_scale = reg_scale
        nn.init.normal_(self.gene_proj.weight, 0, 0.01)
        nn.init.normal_(self.cell_proj.weight, 0, 0.01)

    def forward(self, gene_emb, cell_summary):
        g   = F.normalize(self.gene_proj(gene_emb),    dim=-1)
        c   = F.normalize(self.cell_proj(cell_summary), dim=-1)
        raw = self.scale * (g * c).sum(-1, keepdim=True)
        gate = torch.sigmoid(self.gate)
        reg  = raw.pow(2).mean() * self.reg_scale
        return gate * raw, reg


# ============================================================
# Main model
# ============================================================

class CRISPRSensitivityModelV3(nn.Module):
    """
    CRISPR sensitivity predictor — v3 + multi-task auxiliary head.

    Parameters
    ----------
    cell_features_size : int    Input dimension of cell-line features.
    gene_features_size : int    Input dimension of gene features.
    hidden_dim         : int    Trunk hidden dimension (default 128).
    gene_hidden        : int    Gene embedding dimension (default 64).
    n_attn_slots       : int    Number of cell tokens for cross-attention (default 64).
    n_attn_heads       : int    Attention heads (default 4).
    bypass_rank        : int    Rank of the bilinear bypass (default 32).
    compress_dim       : int    Cell tokenizer compression dimension (default 512).
    dropout            : float  Default dropout rate (default 0.2).

    Forward returns
    ---------------
    regression_out    : Tensor [B, 1]   — scaled dependency score
    bypass_reg        : Tensor scalar   — L2 regularisation term for bypass
    sensitivity_logit : Tensor [B, 1]   — raw logit for gene-sensitive classification
    """

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

        # ── ① Gene encoder ──────────────────────────────────────────────────
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

        # ── ② Cell tokenizer ────────────────────────────────────────────────
        self.cell_tokenizer = CellTokenizer(
            cell_feat_dim = cell_features_size,
            n_slots       = n_attn_slots,
            d_model       = gene_hidden,
            compress_dim  = compress_dim,
            gene_dim      = gene_hidden,
            dropout       = dropout,
        )

        # ── ③ First cross-attention ─────────────────────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm  = nn.LayerNorm(gene_hidden)

        # ── ④ Second cross-attention ────────────────────────────────────────
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm2  = nn.LayerNorm(gene_hidden)

        # ── ⑤ Linear bypass ─────────────────────────────────────────────────
        self.linear_bypass = LinearBypass(
            gene_dim=gene_hidden, cell_dim=gene_hidden, rank=bypass_rank,
        )

        # ── ⑥ Trunk input projection ────────────────────────────────────────
        self.merge = nn.Sequential(
            nn.Linear(gene_hidden * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── ⑦ Conditioning projection ───────────────────────────────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(gene_hidden * 2, gene_hidden),
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
        )

        # ── ⑧ Trunk ─────────────────────────────────────────────────────────
        self.trunk_res1 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)
        self.trunk_res2 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)
        self.trunk_res3 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)

        # ── ⑨ Regression head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # ── ⑩ Auxiliary sensitivity classification head ──────────────────────
        # Branches off cell_context2 — the post-attention, gene-conditioned cell
        # representation.  This is the right branch point because:
        #   - gene_emb alone encodes broad essentiality (not cell-specific)
        #   - cell_context2 contains the gene-conditioned cell state, which is
        #     exactly what determines whether *this* cell line is sensitive to
        #     *this* gene — the signal we want the model to learn explicitly.
        # Gradient is NOT detached — the cls loss shapes the attention layers.
        self.sensitivity_head = nn.Sequential(
            nn.LayerNorm(gene_hidden),
            nn.Linear(gene_hidden, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),   # raw logit — BCEWithLogitsLoss applied in training
        )

        # ── Output scaling ───────────────────────────────────────────────────
        self.out_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.out_shift = nn.Parameter(torch.zeros(1))

        self._init_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        cell_features: torch.Tensor,   # [B, F]
        gene_features: torch.Tensor,   # [B, G]
        ablate_bypass: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        regression_out    : [B, 1]
        bypass_reg        : scalar
        sensitivity_logit : [B, 1]
        """
        # ── ① Gene encoding ─────────────────────────────────────────────────
        gene_emb = self.gene_res(self.gene_encoder(gene_features))

        # ── ② Cell tokenization ─────────────────────────────────────────────
        cell_tokens = self.cell_tokenizer(cell_features, gene_emb)

        # ── ③ First cross-attention (query = gene_emb) ──────────────────────
        attn_out1, attn_weights1 = self.cross_attn(
            gene_emb.unsqueeze(1), cell_tokens, cell_tokens,
            need_weights=True, average_attn_weights=True,
        )
        cell_context = self.attn_norm(attn_out1.squeeze(1) + gene_emb)

        # ── ④ Second cross-attention (query = cell_context) ─────────────────
        attn_out2, attn_weights2 = self.cross_attn2(
            cell_context.unsqueeze(1), cell_tokens, cell_tokens,
            need_weights=True, average_attn_weights=True,
        )
        cell_context2 = self.attn_norm2(attn_out2.squeeze(1) + cell_context)

        # ── ⑤ Bypass ────────────────────────────────────────────────────────
        cell_summary1 = (attn_weights1.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
        cell_summary2 = (attn_weights2.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
        cell_summary  = 0.5 * cell_summary1 + 0.5 * cell_summary2

        bypass_logit, bypass_reg = self.linear_bypass(gene_emb, cell_summary)
        if ablate_bypass:
            bypass_logit = torch.zeros_like(bypass_logit)
            bypass_reg   = torch.tensor(0.0, device=gene_emb.device)

        # ── ⑥ Trunk ─────────────────────────────────────────────────────────
        x_input   = torch.cat([cell_context2, gene_emb], dim=-1)
        x         = self.merge(x_input)
        gene_cond = self.cond_proj(torch.cat([gene_emb, cell_context2], dim=-1))
        x = self.trunk_res1(x, cond=gene_cond)
        x = self.trunk_res2(x, cond=gene_cond)
        x = self.trunk_res3(x, cond=gene_cond)
        trunk_pred = self.head(x)

        # ── ⑦ Regression output ──────────────────────────────────────────────
        raw            = trunk_pred + bypass_logit
        regression_out = raw * self.out_scale + self.out_shift

        # ── ⑧ Auxiliary classification output ───────────────────────────────
        # Gradient flows back through cell_context2 and the attention layers,
        # shaping those representations to encode sensitivity.
        sensitivity_logit = self.sensitivity_head(cell_context2)

        return regression_out, bypass_reg, sensitivity_logit

    # ------------------------------------------------------------------
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


def gene_demeaned_mse(
    pred:     torch.Tensor,
    target:   torch.Tensor,
    gene_idx: torch.Tensor,
    weights:  torch.Tensor = None,
) -> torch.Tensor:
    """
    Per-gene-normalised MSE on within-gene residuals.

    For each gene g, subtracts the per-gene target mean from both pred and
    target (the demeaning cancels algebraically to pred - target, so this is
    equivalent to global MSE numerically). The value is in the WEIGHTING:

      - weights parameter lets you upweight high-variance genes differently
        from the global MSE term
      - gene-count normalisation gives equal gradient weight to a gene with
        3 cell lines in the batch and a gene with 40 cell lines — without
        this, large genes dominate and sparse genes get almost no signal

    Only genes with >=2 cell lines in the batch contribute (need at least 2
    observations to have any within-gene ranking signal).
    """
    pred     = pred.view(-1).float()
    target   = target.view(-1).float()
    gene_idx = gene_idx.view(-1)

    n_genes = int(gene_idx.max().item()) + 1
    counts  = torch.zeros(n_genes, device=pred.device, dtype=pred.dtype)
    counts.scatter_add_(0, gene_idx, torch.ones_like(pred))
    counts  = counts.clamp(min=1)

    t_sum  = torch.zeros(n_genes, device=target.device, dtype=target.dtype)
    t_sum.scatter_add_(0, gene_idx, target)
    t_mean = (t_sum / counts)[gene_idx]

    pred_dm   = pred   - t_mean
    target_dm = target - t_mean
    residuals = (pred_dm - target_dm).pow(2)

    if weights is not None:
        w         = weights.view(-1).to(pred.device, dtype=pred.dtype)
        w         = w / (w.mean() + 1e-8)
        residuals = w * residuals

    gene_mse = torch.zeros(n_genes, device=pred.device, dtype=pred.dtype)
    gene_mse.scatter_add_(0, gene_idx, residuals)
    gene_mse = gene_mse / counts

    present = (counts > 1)
    return gene_mse[present].mean()


def combined_loss(
    pred:            torch.Tensor,
    target:          torch.Tensor,
    cl_idx:          torch.Tensor,
    gene_idx:        torch.Tensor,
    alpha:           float             = 0.5,
    beta:            float             = 0.4,
    gene_var_weight: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined MSE + Pearson + gene-demeaned MSE regression loss.

    The auxiliary classification loss is computed separately in the training
    script and added on top, so this function remains unchanged and all
    existing call-sites remain valid.
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

    mse_term  = (weights * (pred_s - target_s) ** 2).mean()
    pearson_r = differentiable_pearson(pred_s, target_s)
    gdm_mse   = gene_demeaned_mse(pred_s, target_s, gene_idx, weights=weights)

    loss = (
        alpha                * mse_term
        + (1 - alpha - beta) * (1 - pearson_r)
        + beta               * gdm_mse
    )
    return loss, mse_term, pearson_r, gdm_mse


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
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Evaluation loop.

    Batch tuple positions unpacked:
      [0] gene_feat      [1] cell_feat    [2] target       [3] cl_idx
      [4] gene_id        [5] model_id     [6] idx
      [7] gene_var_w     [8] gene_int_idx [9] sensitive_label

    Returns
    -------
    val_loss, val_mse, val_pearson,
    mae, rmse,
    pearson_full, pearson_full_demeaned,
    cl_mean, cl_std,
    cls_auc_approx   ← AUROC proxy for sensitivity classification
    """
    model.eval()
    all_pred, all_target, all_cl_idx, all_gene_int_idx = [], [], [], []
    all_sens_logit, all_sens_label = [], []
    total_loss = total_mse = total_pearson = 0.0

    device_type = torch.device(device).type

    with torch.no_grad():
        for (gene_feat, cell_feat, target, cl_idx,
             _, _, _,
             gene_var_w, gene_int_idx, sensitive_label) in loader:

            gene_feat       = gene_feat.to(device,       non_blocking=True)
            cell_feat       = cell_feat.to(device,       non_blocking=True)
            target_d        = target.to(device,          non_blocking=True)
            cl_idx          = cl_idx.to(device,          non_blocking=True)
            gene_var_w      = gene_var_w.to(device,      non_blocking=True)
            gene_int_idx    = gene_int_idx.to(device,    non_blocking=True)
            sensitive_label = sensitive_label.to(device, non_blocking=True)

            with autocast(device_type=device_type):
                pred, _, sens_logit = model(
                    cell_feat, gene_feat,
                    ablate_bypass=ablate_bypass,
                )
                loss, mse_term, pearson_r, _ = combined_loss(
                    pred, target_d, cl_idx,
                    gene_idx        = gene_int_idx,
                    alpha           = alpha,
                    gene_var_weight = gene_var_w,
                )

            all_pred.append(pred.view(-1))
            all_target.append(target_d.view(-1))
            all_cl_idx.append(cl_idx.view(-1))
            all_gene_int_idx.append(gene_int_idx.view(-1))
            all_sens_logit.append(sens_logit.view(-1))
            all_sens_label.append(sensitive_label.view(-1))

            total_loss    += loss.item()
            total_mse     += mse_term.item()
            total_pearson += pearson_r.item()

    n = len(loader)
    val_loss    = total_loss    / n
    val_mse     = total_mse     / n
    val_pearson = total_pearson / n

    eval_pred    = torch.cat(all_pred)
    eval_target  = torch.cat(all_target)
    cl_idx       = torch.cat(all_cl_idx)
    gene_int_idx = torch.cat(all_gene_int_idx)
    sens_logit   = torch.cat(all_sens_logit)
    sens_label   = torch.cat(all_sens_label)

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

    # ── Gene-demeaned Pearson ─────────────────────────────────────────────
    n_genes  = gene_int_idx.max() + 1
    g_counts = torch.zeros(n_genes, device=device).scatter_add_(
        0, gene_int_idx, torch.ones_like(eval_pred)
    ).clamp(min=1)
    t_gsums  = torch.zeros(n_genes, device=device).scatter_add_(
        0, gene_int_idx, eval_target
    )
    t_gmeans = (t_gsums / g_counts)[gene_int_idx]

    pred_gdm   = eval_pred   - t_gmeans
    target_gdm = eval_target - t_gmeans
    pearson_full_demeaned = (
        (pred_gdm * target_gdm).sum()
        / (pred_gdm.norm() * target_gdm.norm() + 1e-8)
    ).item()

    # ── Per-cell-line Pearson (cells with >=10 genes) ─────────────────────
    n_cl   = cl_idx.max() + 1
    counts = torch.zeros(n_cl, device=device).scatter_add_(
        0, cl_idx, torch.ones_like(eval_pred)
    ).clamp(min=1)

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

    # ── Sensitivity classification AUROC (approx via Pearson of prob vs label)
    # Full AUROC requires sorting; Wilcoxon rank-sum is exact but expensive.
    # Pearson(sigmoid(logit), label) correlates tightly with AUROC and needs
    # no external dependency.  Values >0.15 indicate useful discrimination.
    sens_prob = torch.sigmoid(sens_logit)
    sp = sens_prob  - sens_prob.mean()
    sl = sens_label - sens_label.mean()
    cls_auc_approx = (
        (sp * sl).sum() / (sp.norm() * sl.norm() + 1e-8)
    ).item()

    return (
        val_loss, val_mse, val_pearson,
        mae, rmse,
        pearson_full,
        pearson_full_demeaned,
        cl_mean, cl_std,
        cls_auc_approx,
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
    """
    Print bypass vs trunk head contribution.

    Healthy model: bypass/head ratio < 1.0, attn entropy > 2.0
    Collapse risk: bypass/head ratio > 2.0, attn entropy < 0.5
    """
    model.eval()
    bypass_mags, head_mags, attn_entropies = [], [], []

    with torch.no_grad():
        for i, (gene_feat, cell_feat, target, cl_idx,
                gene_id, model_id, _,
                _, gene_int_idx, _) in enumerate(loader):
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