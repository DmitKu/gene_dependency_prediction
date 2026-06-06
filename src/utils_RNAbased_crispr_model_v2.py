# -*- coding: utf-8 -*-
"""
utils_RNAbased_crispr_model.py
===============
CRISPR Sensitivity Model v4 — Cross-Attention + MoE + OutlierGate + QuantileHead

Changes from v3
---------------
  FIX 1 — MixtureOfExpertsCellSummary replaces the two sequential cross-attentions.
           A gene-conditioned router picks which expert view of the cell matters,
           so KRAS-dependency patterns are kept separate from EIF1AX patterns.

  FIX 2 — QuantileHead replaces the single-value head.
           Pinball loss prevents mean-collapse by forcing calibrated spread.

  FIX 3 — OutlierGate added between merge and trunk.
           Learns a per-(gene,cell) scalar that amplifies cell context for
           the minority of highly-sensitive cell lines (KRAS ~20%).

  FIX 4 — sensitivity_weighted_loss replaces combined_loss.
           Weights are computed relative to per-cell-line baselines, not
           the batch mean, so rare strong dependencies are not drowned out.

Architecture summary (v4)
--------------------------
  Gene features  ->  gene_encoder  ->  gene_res         ->  gene_emb     [B, gene_hidden]
  Cell features  ->  cell_tokenizer(gene_emb)            ->  cell_tokens  [B, n_slots, gene_hidden]

  MoE cross-attention (K experts, gene-routed)           ->  cell_context [B, gene_hidden]

  bypass_logit  = LinearBypass(gene_emb, cell_context)

  gated         = OutlierGate(gene_emb, cell_context)    ->  [B, gene_hidden]
  x             = merge(cat[gated, gene_emb])            ->  [B, hidden_dim]
  gene_cond     = cond_proj(gene_emb)                    ->  [B, gene_hidden]

  x = trunk_res1/2/3(x, cond=gene_cond)
  quantile_preds = QuantileHead(x)                       ->  [B, 5]
  output (point) = quantile_preds[:, 2] + bypass_logit  ->  [B, 1]
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.amp import autocast
import h5py


# ============================================================
# Dataset  (unchanged from v3)
# ============================================================

class GeneDataset(Dataset):
    """
    Loads gene-level CRISPR sensitivity data from an HDF5 file.

    Expected HDF5 layout
    --------------------
    index/splits/{train,val,test}  : 1-D integer index array
    cell_lines/features            : float32 array [n_cell_lines, F]
    cell_lines/model_id           : bytes array   [n_cell_lines]
    genes/features                 : float32 array [n_genes, G]
    genes/model_id                 : bytes array   [n_genes]
    genes/crispr                   : float32 array [n_genes]

    Returns (per __getitem__)
    -------------------------
    gene_feat   : Tensor [G]
    cell_feat   : Tensor [F]
    crispr      : Tensor [1]
    cl_idx      : LongTensor scalar
    gene_id     : str
    model_id    : str
    idx         : int
    """

    def __init__(self, h5_path: str, split: str = "train"):
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
            all_crispr        = f["genes/crispr"][:]

            self.gene_feat  = torch.tensor(all_gene_feat[gene_indices], dtype=torch.float32)
            self.crispr     = torch.tensor(all_crispr[gene_indices],    dtype=torch.float32)

            self.gene_ids   = [all_gene_ids[i].decode()  for i in gene_indices]
            self.model_ids  = [all_model_ids[i].decode() for i in gene_indices]

        self.cl_model_id_to_index = {mid.decode(): i for i, mid in enumerate(cl_model_ids)}

        self.cl_indices = torch.tensor(
            [self.cl_model_id_to_index[mid] for mid in self.model_ids],
            dtype=torch.long,
        )

        self.cl_index_to_model_id = {v: k for k, v in self.cl_model_id_to_index.items()}

        pairs    = list(zip(self.gene_ids, self.model_ids))
        n_unique = len(set(pairs))
        if n_unique != len(pairs):
            import warnings
            warnings.warn(
                f"Split '{split}': {len(pairs) - n_unique} duplicate "
                f"(gene_id, model_id) pairs found"
            )

        print(f"  -> {len(self.gene_feat):,} samples | "
              f"{len(set(self.gene_ids)):,} genes | "
              f"{len(set(self.model_ids)):,} cell lines | "
              f"loaded in {time.time() - t0:.2f}s")

    def __len__(self) -> int:
        return len(self.gene_feat)

    def __getitem__(self, idx):
        return (
            self.gene_feat[idx],
            self.cl_features[self.cl_indices[idx]],
            self.crispr[idx].unsqueeze(0),
            self.cl_indices[idx],
            self.gene_ids[idx],
            self.model_ids[idx],
            idx,
        )


# ============================================================
# Building blocks  (unchanged from v3)
# ============================================================

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class GELUResidualBlock(nn.Module):
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
        self.gate      = nn.Parameter(torch.tensor(-2.0))
        self.reg_scale = reg_scale
        nn.init.normal_(self.gene_proj.weight, 0, 0.01)
        nn.init.normal_(self.cell_proj.weight, 0, 0.01)

    def forward(self, gene_emb, cell_summary):
        g   = F.normalize(self.gene_proj(gene_emb), dim=-1)
        c   = F.normalize(self.cell_proj(cell_summary), dim=-1)
        raw = self.scale * (g * c).sum(-1, keepdim=True)
        gate = torch.sigmoid(self.gate)
        reg  = raw.pow(2).mean() * self.reg_scale
        return gate * raw, reg


# ============================================================
# FIX 1 — MixtureOfExpertsCellSummary
# Replaces: self.cross_attn, self.attn_norm, self.cross_attn2, self.attn_norm2
# Location: CRISPRSensitivityModelV3.__init__  blocks ③ and ④
# ============================================================

class MixtureOfExpertsCellSummary(nn.Module):
    """
    K independent cross-attention experts over cell tokens.
    A gene-conditioned router (softmax) decides how much weight
    each expert contributes — so KRAS-pathway signatures are
    handled by different experts than housekeeping-gene signatures.

    Replaces the two sequential cross-attentions in v3.
    Output shape matches v3: [B, d_model]
    """
    def __init__(
        self,
        gene_dim:    int,
        d_model:     int,
        n_experts:   int   = 4,
        n_attn_heads: int  = 4,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, n_attn_heads,
                dropout=dropout, batch_first=True,
            )
            for _ in range(n_experts)
        ])

        # Gene-conditioned router: which expert view matters for this gene
        self.router = nn.Sequential(
            nn.Linear(gene_dim, n_experts),
            nn.Softmax(dim=-1),          # [B, n_experts]
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        gene_emb:    torch.Tensor,   # [B, gene_dim]
        cell_tokens: torch.Tensor,   # [B, n_slots, d_model]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mixed        : [B, d_model]   — MoE-weighted cell context
        router_weights: [B, n_experts] — for diagnostics / entropy tracking
        """
        router_weights = self.router(gene_emb)   # [B, n_experts]

        expert_outputs = []
        for expert in self.experts:
            out, _ = expert(
                gene_emb.unsqueeze(1),   # query:  [B, 1, d_model]
                cell_tokens,             # key:    [B, n_slots, d_model]
                cell_tokens,             # value:  [B, n_slots, d_model]
            )
            expert_outputs.append(out.squeeze(1))   # [B, d_model]

        stack = torch.stack(expert_outputs, dim=1)             # [B, n_experts, d_model]
        mixed = (router_weights.unsqueeze(-1) * stack).sum(1)  # [B, d_model]
        return self.norm(mixed + gene_emb), router_weights     # residual connection kept


# ============================================================
# FIX 3 — OutlierGate
# Inserted between merge and trunk_res1 in forward()
# ============================================================

class OutlierGate(nn.Module):
    """
    Learns a scalar gate per (gene, cell_line) pair.
    gate ~ 0  ->  use gene_emb as anchor (most cell lines)
    gate ~ 1  ->  cell context drives prediction (outlier cell lines)

    This directly addresses sparse dependencies like KRAS (~20% sensitive).
    Bias initialised to -2 so the gate starts nearly closed and opens
    only when the cell context strongly justifies it.
    """
    def __init__(self, gene_dim: int, cell_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim + cell_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # Start conservative: most cell lines are NOT outliers
        nn.init.constant_(self.net[2].bias, -2.0)

    def forward(
        self,
        gene_emb:     torch.Tensor,   # [B, gene_dim]
        cell_context: torch.Tensor,   # [B, cell_dim]
    ) -> torch.Tensor:                # [B, gene_dim]  (same shape as gene_emb)
        gate = self.net(torch.cat([gene_emb, cell_context], dim=-1))   # [B, 1]
        return gene_emb * (1 - gate) + cell_context * gate


# ============================================================
# FIX 2 — QuantileHead
# Replaces: self.head  in CRISPRSensitivityModelV3
# ============================================================

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]   # index 2 = median = point prediction

class QuantileHead(nn.Module):
    """
    Predicts 5 quantiles [0.1, 0.25, 0.5, 0.75, 0.9].
    The median (index 2) is used as the point prediction.
    Pinball loss penalises miscalibrated spread, preventing the
    model from collapsing to the conditional mean.
    """
    def __init__(self, hidden_dim: int, n_quantiles: int = 5):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # [B, n_quantiles]


def pinball_loss(
    pred_quantiles: torch.Tensor,   # [B, n_quantiles]
    target:         torch.Tensor,   # [B] or [B, 1]
) -> torch.Tensor:
    """Pinball (quantile regression) loss."""
    q = torch.tensor(QUANTILES, device=pred_quantiles.device, dtype=pred_quantiles.dtype)
    target = target.view(-1, 1).expand_as(pred_quantiles)
    err    = target - pred_quantiles
    loss   = torch.max((q - 1) * err, q * err)
    return loss.mean()


# ============================================================
# Main model  — v4
# ============================================================

class CRISPRSensitivityModelV3(nn.Module):
    """
    CRISPR sensitivity predictor v4.
    Drop-in replacement for v3 — same constructor signature.

    Key changes
    -----------
    cross_attn / cross_attn2  ->  moe_attn  (MixtureOfExpertsCellSummary)
    self.head                 ->  self.head  (QuantileHead, 5 outputs)
    OutlierGate added between merge and trunk
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
        n_moe_experts:      int   = 4,    # NEW: number of MoE experts
    ):
        super().__init__()

        # ── ① Gene encoder (unchanged) ──────────────────────────────────────
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

        # ── ② Cell tokenizer (unchanged) ────────────────────────────────────
        self.cell_tokenizer = CellTokenizer(
            cell_feat_dim = cell_features_size,
            n_slots       = n_attn_slots,
            d_model       = gene_hidden,
            compress_dim  = compress_dim,
            gene_dim      = gene_hidden,
            dropout       = dropout,
        )

        # ── ③+④ FIX 1: MoE cross-attention replaces cross_attn + cross_attn2 ─
        # REMOVED: self.cross_attn, self.attn_norm, self.cross_attn2, self.attn_norm2
        # ADDED:   self.moe_attn
        self.moe_attn = MixtureOfExpertsCellSummary(
            gene_dim    = gene_hidden,
            d_model     = gene_hidden,
            n_experts   = n_moe_experts,
            n_attn_heads = n_attn_heads,
            dropout     = dropout,
        )

        # ── ⑤ Linear bypass (unchanged) ─────────────────────────────────────
        self.linear_bypass = LinearBypass(
            gene_dim=gene_hidden, cell_dim=gene_hidden, rank=bypass_rank,
        )

        # ── ⑥ FIX 3: OutlierGate (NEW) ──────────────────────────────────────
        # Inserted between MoE output and merge projection
        self.outlier_gate = OutlierGate(
            gene_dim  = gene_hidden,
            cell_dim  = gene_hidden,
            hidden    = 64,
        )

        # ── ⑦ Trunk input projection (unchanged) ────────────────────────────
        self.merge = nn.Sequential(
            nn.Linear(gene_hidden * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── ⑧ Conditioning projection (unchanged) ───────────────────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(gene_hidden * 2, gene_hidden),
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
        )

        # ── ⑨ Trunk (unchanged) ─────────────────────────────────────────────
        self.trunk_res1 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)
        self.trunk_res2 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)
        self.trunk_res3 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)

        # ── ⑩ FIX 2: QuantileHead replaces single-value head ────────────────
        # REMOVED: self.head (Sequential -> Linear -> ... -> Linear(64,1))
        # ADDED:   self.head (QuantileHead -> Linear(64,5))
        self.head = QuantileHead(hidden_dim, n_quantiles=len(QUANTILES))

        # Output scale/shift applied only to the median quantile for bypass sum
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
        point_pred      : [B, 1]           median prediction (for inference / Pearson)
        quantile_preds  : [B, n_quantiles] all quantile predictions (for pinball loss)
        bypass_reg      : scalar           bypass regularisation term
        """

        # ── ① Gene encoding ─────────────────────────────────────────────────
        gene_emb = self.gene_res(self.gene_encoder(gene_features))

        # ── ② Cell tokenization ─────────────────────────────────────────────
        cell_tokens = self.cell_tokenizer(cell_features, gene_emb)

        # ── ③ FIX 1: MoE cross-attention ────────────────────────────────────
        # REPLACES: two sequential cross-attentions + manual cell_summary calc
        cell_context, router_weights = self.moe_attn(gene_emb, cell_tokens)
        # cell_context: [B, gene_hidden]

        # ── ④ Linear bypass ─────────────────────────────────────────────────
        bypass_logit, bypass_reg = self.linear_bypass(gene_emb, cell_context)
        if ablate_bypass:
            bypass_logit = torch.zeros_like(bypass_logit)
            bypass_reg   = torch.tensor(0.0, device=gene_emb.device)

        # ── ⑤ FIX 3: OutlierGate ────────────────────────────────────────────
        # INSERTED: before merge, after MoE
        gated = self.outlier_gate(gene_emb, cell_context)   # [B, gene_hidden]

        # ── ⑥ Trunk ─────────────────────────────────────────────────────────
        x_input  = torch.cat([gated, gene_emb], dim=-1)     # [B, gene_hidden*2]
        x        = self.merge(x_input)
        gene_cond = self.cond_proj(torch.cat([gene_emb, cell_context], dim=-1))
        x = self.trunk_res1(x, cond=gene_cond)
        x = self.trunk_res2(x, cond=gene_cond)
        x = self.trunk_res3(x, cond=gene_cond)

        # ── ⑦ FIX 2: QuantileHead ───────────────────────────────────────────
        quantile_preds = self.head(x)                        # [B, 5]
        median_pred    = quantile_preds[:, 2:3]              # index 2 = q0.5

        point_pred = median_pred * self.out_scale + self.out_shift + bypass_logit
        return point_pred, quantile_preds, bypass_reg

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
# FIX 4 — sensitivity_weighted_loss
# Replaces: combined_loss  everywhere it is called
# ============================================================

def differentiable_pearson(pred, target):
    pred   = pred.view(-1)
    target = target.view(-1)
    pm = pred   - pred.mean()
    pt = target - target.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def demeaned_pearson(pred, target, cl_idx):
    pred   = pred.view(-1).float()
    target = target.view(-1).float()
    cl_idx = cl_idx.view(-1)
    n_cl   = int(cl_idx.max().item()) + 1

    counts = torch.zeros(n_cl, device=pred.device, dtype=pred.dtype)
    counts.scatter_add_(0, cl_idx, torch.ones_like(pred))
    counts = counts.clamp(min=1)

    pred_sum   = torch.zeros(n_cl, device=pred.device,   dtype=pred.dtype).scatter_add_(0, cl_idx, pred)
    target_sum = torch.zeros(n_cl, device=target.device, dtype=target.dtype).scatter_add_(0, cl_idx, target)

    pred_dm   = pred   - (pred_sum   / counts)[cl_idx]
    target_dm = target - (target_sum / counts)[cl_idx]

    pm = pred_dm   - pred_dm.mean()
    pt = target_dm - target_dm.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def sensitivity_weighted_loss(
    point_pred:     torch.Tensor,   # [B, 1]  — median prediction
    quantile_preds: torch.Tensor,   # [B, 5]  — all quantile predictions
    target:         torch.Tensor,   # [B, 1]
    cl_idx:         torch.Tensor,   # [B]
    alpha:          float = 0.2,
    beta:           float = 0.4,
    pinball_weight: float = 0.2,    # weight for quantile calibration loss
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FIX 4: weights relative to per-cell-line baseline, not batch mean.
    Rare strong dependencies (KRAS in ~20% of cell lines) are no longer
    drowned out by the majority of non-sensitive samples.

    Also adds pinball loss on all quantiles (FIX 2 companion term).

    Returns: loss, mse_term, pearson_r, pearson_demeaned
    """
    pred_s   = point_pred.view(-1)
    target_s = target.view(-1)
    cl_idx_s = cl_idx.view(-1)

    # ── Per-cell-line mean target (cell-line baseline) ───────────────────────
    n_cl   = cl_idx_s.max() + 1
    counts = torch.zeros(n_cl, device=pred_s.device).scatter_add_(
                 0, cl_idx_s, torch.ones_like(pred_s)).clamp(min=1)
    cl_target_sum = torch.zeros(n_cl, device=target_s.device).scatter_add_(
                        0, cl_idx_s, target_s)
    cell_baseline = (cl_target_sum / counts)[cl_idx_s]   # [B]

    # ── Weight = how much does this gene deviate from cell-line baseline ──────
    # (instead of deviation from batch mean used in v3)
    with torch.no_grad():
        gene_deviation = (target_s - cell_baseline).abs()
        weights = 1.0 + 3.0 * (gene_deviation / (gene_deviation.max() + 1e-8))
        weights = weights / weights.mean()

    mse_term         = (weights * (pred_s - target_s) ** 2).mean()
    pearson_r        = differentiable_pearson(pred_s, target_s)
    pearson_demeaned = demeaned_pearson(pred_s, target_s, cl_idx_s)

    # ── Pinball loss on all quantiles (FIX 2 companion) ──────────────────────
    pb_loss = pinball_loss(quantile_preds, target_s)

    loss = (
        alpha          * mse_term
        + (1 - alpha - beta - pinball_weight) * (1 - pearson_r)
        + beta         * (1 - pearson_demeaned)
        + pinball_weight * pb_loss
    )
    return loss, mse_term, pearson_r, pearson_demeaned


# Keep combined_loss as an alias so old code doesn't break immediately
def combined_loss(pred, target, cl_idx, alpha=0.5, beta=0.4):
    raise RuntimeError(
        "combined_loss removed in v4. Use sensitivity_weighted_loss instead.\n"
        "Signature: sensitivity_weighted_loss(point_pred, quantile_preds, target, cl_idx, ...)"
    )


# ============================================================
# Evaluation  — updated for v4 forward() signature
# ============================================================

def evaluate(
    model: nn.Module,
    loader,
    device: str,
    qt=None,
    alpha: float = 0.5,
    ablate_bypass: bool = False,
) -> tuple:
    model.eval()
    all_pred, all_target, all_cl_idx = [], [], []
    total_loss = total_mse = total_pearson = 0.0

    device_type = torch.device(device).type

    with torch.no_grad():
        for gene_feat, cell_feat, target, cl_idx, _, _, _ in loader:
            gene_feat = gene_feat.to(device, non_blocking=True)
            cell_feat = cell_feat.to(device, non_blocking=True)
            target_d  = target.to(device, non_blocking=True)
            cl_idx    = cl_idx.to(device, non_blocking=True)

            with autocast(device_type=device_type):
                # FIX: unpack 3 values (point_pred, quantile_preds, bypass_reg)
                point_pred, quantile_preds, _ = model(
                    cell_feat, gene_feat, ablate_bypass=ablate_bypass
                )
                loss, mse_term, pearson_r, _ = sensitivity_weighted_loss(
                    point_pred, quantile_preds, target_d, cl_idx, alpha=alpha
                )

            all_pred.append(point_pred.view(-1))
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
        import numpy as np
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
                 0, cl_idx, torch.ones_like(eval_pred)).clamp(min=1)
    p_sums = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, eval_pred)
    t_sums = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, eval_target)

    pred_dm   = eval_pred   - (p_sums / counts)[cl_idx]
    target_dm = eval_target - (t_sums / counts)[cl_idx]

    pm_d = pred_dm   - pred_dm.mean()
    pt_d = target_dm - target_dm.mean()
    pearson_full_demeaned = ((pm_d * pt_d).sum() / (pm_d.norm() * pt_d.norm() + 1e-8)).item()

    mask         = counts >= 10
    valid_cl_ids = torch.nonzero(mask).view(-1)
    cl_pearsons  = []

    if valid_cl_ids.numel() > 0:
        for cid in valid_cl_ids:
            cl_mask = (cl_idx == cid)
            p_cl = eval_pred[cl_mask]
            t_cl = eval_target[cl_mask]
            pm_cl = p_cl - p_cl.mean()
            pt_cl = t_cl - t_cl.mean()
            denom = pm_cl.norm() * pt_cl.norm()
            if denom > 1e-8:
                cl_pearsons.append(((pm_cl * pt_cl).sum() / denom))

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
# Diagnostics  — updated for v4 forward() + MoE router entropy
# ============================================================

def diagnose_bypass(
    model:     nn.Module,
    loader,
    device:    str,
    n_batches: int = 5,
) -> None:
    """
    Print bypass vs trunk head contribution, plus MoE router entropy.

    Healthy model : bypass/head ratio < 1.0
                    attn entropy > 1.5  (experts specialising but not collapsed)
    Collapse risk : bypass/head ratio > 2.0
                    attn entropy < 0.3  (one expert dominates everything)
    """
    model.eval()
    bypass_mags, head_mags, router_entropies = [], [], []

    with torch.no_grad():
        for i, (gene_feat, cell_feat, *_) in enumerate(loader):
            if i >= n_batches:
                break
            gene_feat = gene_feat.to(device)
            cell_feat = cell_feat.to(device)

            gene_emb    = model.gene_res(model.gene_encoder(gene_feat))
            cell_tokens = model.cell_tokenizer(cell_feat, gene_emb)

            # MoE attention
            cell_context, router_weights = model.moe_attn(gene_emb, cell_tokens)

            # Bypass
            bypass, _ = model.linear_bypass(gene_emb, cell_context)

            # Trunk
            gated     = model.outlier_gate(gene_emb, cell_context)
            x         = model.merge(torch.cat([gated, gene_emb], dim=-1))
            gene_cond = model.cond_proj(torch.cat([gene_emb, cell_context], dim=-1))
            x         = model.trunk_res1(x, cond=gene_cond)
            x         = model.trunk_res2(x, cond=gene_cond)
            x         = model.trunk_res3(x, cond=gene_cond)
            q_preds   = model.head(x)
            head_out  = q_preds[:, 2:3]   # median

            bypass_mags.append((bypass * model.out_scale).abs().mean().item())
            head_mags.append((head_out * model.out_scale).abs().mean().item())

            # Router entropy: high = experts share load; low = collapse to one expert
            w       = router_weights.clamp(min=1e-8)
            entropy = -(w * w.log()).sum(-1).mean()
            router_entropies.append(entropy.item())

    bypass_mean  = sum(bypass_mags)      / len(bypass_mags)
    head_mean    = sum(head_mags)        / len(head_mags)
    entropy_mean = sum(router_entropies) / len(router_entropies)
    ratio        = bypass_mean / (head_mean + 1e-8)

    print(f"\n-- Bypass Diagnostic (v4) ------------------------------------------")
    print(f"  Bypass magnitude   : {bypass_mean:.4f}")
    print(f"  Head magnitude     : {head_mean:.4f}")
    print(f"  Bypass/Head ratio  : {ratio:.3f}  (healthy: <1.0  collapse: >2.0)")
    print(f"  MoE router entropy : {entropy_mean:.4f}  (healthy: >1.5  collapsed: <0.3)")
    print(f"--------------------------------------------------------------------\n")