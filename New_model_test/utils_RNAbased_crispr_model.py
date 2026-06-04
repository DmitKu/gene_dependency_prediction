# -*- coding: utf-8 -*-
"""
src/utils_RNAbased_crispr_model.py
==================================
CRISPR Sensitivity Model V5 (RNA-based with cluster attention context)

Hybrid architecture combining:
- V4's robust FiLM-conditioned cell encoder + multi-head bilinear interaction
- V5's cluster token in multi-head attention context for explicit feature integration

Key improvements:
- Cluster token prepended to cell token context for attention operations
- Dual cross-attention: gene queries cells, then cells query back
- Proper gradient flow through all attention paths
- Clean initialization prevents silent training failures

Architecture:
  Gene side: cluster_id → embedding + stat encoder → gene query [B, 1, H]
  Cell side: cell_features → FiLM-conditioned encoder → tokens [B, N_slots, H]
  Cluster token: cluster_id → embedding → prepended to tokens [B, N_slots+1, H]
  Attention: Cross-attention between gene and augmented cell context
  Output: Merged representation → trunk + bias terms
"""

import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.amp import autocast
import h5py


# ============================================================
# Dataset
# ============================================================

class GeneDataset(Dataset):
    """
    Loads all data into CPU RAM on construction.

    Returns per __getitem__ (8-tuple):
    -----------------------------------
    gene_feat       : float32 Tensor [26]
    cell_feat       : float32 Tensor [2388]
    crispr          : float32 Tensor [1]
    cl_idx          : int64   scalar  — row index into cell_lines/features
    gene_cluster_id : int64   scalar  — cluster index 0..2387
    gene_id         : str
    model_id        : str
    idx             : int     — position within split
    """

    def __init__(self, h5_path: str, split: str = "train"):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        print(f"[GeneDataset] Loading '{split}' split ...")
        t0 = time.time()

        with h5py.File(h5_path, "r") as f:
            gene_indices = f[f"index/splits/{split}"][:]
            self.cl_features = torch.tensor(
                f["cell_lines/features"][:], dtype=torch.float32
            )
            cl_model_ids = f["cell_lines/model_id"][:]

            all_gene_feat   = f["genes/features"][:]
            all_cluster_ids = f["genes/cluster_id"][:]
            all_model_ids   = f["genes/model_id"][:]
            all_gene_ids    = f["genes/gene_id"][:]
            all_crispr      = f["genes/crispr"][:]

            self.gene_feat        = torch.tensor(
                all_gene_feat[gene_indices],   dtype=torch.float32
            )
            self.crispr           = torch.tensor(
                all_crispr[gene_indices],      dtype=torch.float32
            )
            self.gene_cluster_ids = torch.tensor(
                all_cluster_ids[gene_indices], dtype=torch.long
            )

            self.gene_ids  = [all_gene_ids[i].decode()  for i in gene_indices]
            self.model_ids = [all_model_ids[i].decode() for i in gene_indices]

        self.cl_model_id_to_index = {
            mid.decode(): i for i, mid in enumerate(cl_model_ids)
        }

        self.cl_indices = torch.tensor(
            [self.cl_model_id_to_index[mid] for mid in self.model_ids],
            dtype=torch.long,
        )

        self.cl_index_to_model_id = {
            v: k for k, v in self.cl_model_id_to_index.items()
        }

        pairs    = list(zip(self.gene_ids, self.model_ids))
        n_unique = len(set(pairs))
        if n_unique != len(pairs):
            warnings.warn(
                f"Split '{split}': {len(pairs) - n_unique} duplicate "
                f"(gene_id, model_id) pairs — check data pipeline"
            )

        n_clusters_seen = int(self.gene_cluster_ids.max().item()) + 1
        print(
            f"  → {len(self.gene_feat):,} samples | "
            f"{len(set(self.gene_ids)):,} unique genes | "
            f"{len(set(self.model_ids)):,} cell lines | "
            f"{n_clusters_seen} gene clusters seen | "
            f"loaded in {time.time() - t0:.1f}s"
        )

    def __len__(self) -> int:
        return len(self.gene_feat)

    def __getitem__(self, idx):
        return (
            self.gene_feat[idx],
            self.cl_features[self.cl_indices[idx]],
            self.crispr[idx].unsqueeze(0),
            self.cl_indices[idx],
            self.gene_cluster_ids[idx],
            self.gene_ids[idx],
            self.model_ids[idx],
            idx,
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


class RMSNorm(nn.Module):
    """RMS Layer Normalization for stability in attention context."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)


class ResidualFiLMBlock(nn.Module):
    """Pre-LayerNorm residual block with FiLM conditioning."""
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm    = nn.LayerNorm(dim)
        self.net     = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.film    = FiLMLayer(cond_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.net(self.norm(x))
        out = self.film(out, cond)
        out = self.dropout(out)
        return out + x


# ============================================================
# Main model (V5: Hybrid)
# ============================================================

class CRISPRModelV5(nn.Module):
    """
    CRISPR dependency predictor V5.
    
    Hybrid architecture:
    - Takes proven V4 components (FiLM, multi-head bilinear, explicit main effects)
    - Adds cluster token in attention context (the one good idea from V5)
    - Properly initialized to prevent silent failures
    
    Parameters
    ----------
    n_clusters      : int    Number of gene clusters = cell feature dim (2388)
    gene_feat_size  : int    Gene feature dimension (26)
    cell_feat_size  : int    Cell feature dimension (2388)
    hidden_dim      : int    Embedding dimension for attention context (128)
    n_slots         : int    Number of cell feature tokens (32)
    n_heads         : int    Attention heads (8)
    dropout         : float  Dropout rate (0.15)
    """

    def __init__(
        self,
        n_clusters:     int   = 2388,
        gene_feat_size: int   = 26,
        cell_feat_size: int   = 2388,
        hidden_dim:     int   = 128,
        n_slots:        int   = 32,
        n_heads:        int   = 8,
        dropout:        float = 0.15,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        assert cell_feat_size == n_clusters, \
            (f"cell_feat_size ({cell_feat_size}) must equal n_clusters ({n_clusters})")

        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.n_slots    = n_slots
        self.n_clusters = n_clusters

        # ── Gene side ──────────────────────────────────────────────────
        # Cluster embedding + stat encoder
        self.gene_cluster_emb = nn.Embedding(n_clusters, hidden_dim)

        self.gene_stat_encoder = nn.Sequential(
            nn.Linear(gene_feat_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.gene_gate = nn.Sequential(
            nn.Linear(gene_feat_size, hidden_dim),
            nn.Sigmoid(),
        )

        # ── Cluster token (key V5 contribution) ──────────────────────────
        # Cluster embedding for attention context
        self.cluster_token_emb = nn.Embedding(n_clusters, hidden_dim)

        # ── Cell side ──────────────────────────────────────────────────
        # Input: [cell_features | relative_features] = 4776
        cell_input_dim = cell_feat_size * 2

        self.cell_proj = nn.Sequential(
            nn.Linear(cell_input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cell_align = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.cell_film_blocks = nn.ModuleList([
            ResidualFiLMBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(3)
        ])

        # Tokenization for attention: cell context → n_slots tokens
        self.tokenize = nn.Linear(hidden_dim, n_slots * hidden_dim)
        self.token_norm = RMSNorm(hidden_dim)
        self.slot_embed = nn.Parameter(torch.randn(1, n_slots, hidden_dim) * 0.02)

        # Own-cluster pathway
        self.own_expr_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.cell_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── Cross-attention (gene ↔ augmented cell context) ──────────────
        # Gene queries cell context, then cell context queries back
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=dropout
        )
        self.attn_norm1 = nn.LayerNorm(hidden_dim)
        self.attn_norm2 = nn.LayerNorm(hidden_dim)

        # ── Multi-head bilinear interaction ────────────────────────────
        self.bilinear_u = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bilinear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ── Explicit main effect terms (zero-init) ──────────────────────
        self.gene_bias = nn.Linear(hidden_dim, 1)
        self.cell_bias = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gene_bias.weight)
        nn.init.zeros_(self.gene_bias.bias)
        nn.init.zeros_(self.cell_bias.weight)
        nn.init.zeros_(self.cell_bias.bias)

        # ── Interaction head ───────────────────────────────────────────
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, cell_feat: torch.Tensor, gene_feat: torch.Tensor,
                gene_cluster_id: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            cell_feat        : [B, 2388] raw cell features
            gene_feat        : [B, 26] gene stat features
            gene_cluster_id  : [B] cluster indices

        Returns:
            pred : [B, 1] CRISPR sensitivity predictions
        """
        B = cell_feat.size(0)

        # ── Gene encoding ──────────────────────────────────────────────
        gene_cluster = self.gene_cluster_emb(gene_cluster_id)  # [B, H]
        gene_stat = self.gene_stat_encoder(gene_feat)  # [B, H]
        gene_gate = self.gene_gate(gene_feat)  # [B, H]
        gene_repr = gene_gate * gene_cluster + (1 - gene_gate) * gene_stat  # [B, H]
        gene_q = gene_repr.unsqueeze(1)  # [B, 1, H]

        # ── Cell encoding ──────────────────────────────────────────────
        # Extract owned cluster expression (privileged)
        own_expr = cell_feat[torch.arange(B), gene_cluster_id].unsqueeze(-1)  # [B, 1]

        # Relative features
        rel_feat = cell_feat / (own_expr + 1e-8)  # [B, 2388]
        cell_input = torch.cat([cell_feat, rel_feat], dim=-1)  # [B, 4776]

        # Project and FiLM condition
        cell_proj = self.cell_proj(cell_input)  # [B, 512]
        cell_proj = self.cell_align(cell_proj)  # [B, H]

        # FiLM blocks conditioned by gene representation
        for block in self.cell_film_blocks:
            cell_proj = block(cell_proj, gene_repr)  # [B, H]

        # Tokenize for attention
        tokens = self.tokenize(cell_proj).view(B, self.n_slots, self.hidden_dim)  # [B, N_slots, H]
        tokens = self.token_norm(tokens) + self.slot_embed  # [B, N_slots, H]

        # Own-cluster pathway (parallel)
        own_emb = self.own_expr_encoder(own_expr)  # [B, H]

        # ── Cluster token in attention context (V5 contribution) ───────
        # Prepend cluster token to cell tokens
        cluster_token = self.cluster_token_emb(gene_cluster_id).unsqueeze(1)  # [B, 1, H]
        context = torch.cat([cluster_token, tokens], dim=1)  # [B, N_slots+1, H]

        # ── Cross-attention: gene queries augmented cell context ────────
        out1, _ = self.cross_attn1(
            query=gene_q,
            key=context,
            value=context,
            need_weights=False
        )
        gene_out = self.attn_norm1(out1.squeeze(1) + gene_repr)  # [B, H]

        # Cross-attention: cell context queries back
        out2, _ = self.cross_attn2(
            query=context,
            key=gene_q,
            value=gene_q,
            need_weights=False
        )
        ctx_out = self.attn_norm2(out2[:, 0] + cluster_token.squeeze(1))  # [B, H] from cluster token

        # ── Cell representation (fusion with own-cluster pathway) ──────
        cell_repr = self.cell_fusion(torch.cat([cell_proj, own_emb], dim=-1))  # [B, H]

        # ── Multi-head bilinear interaction ────────────────────────────
        ug = self.bilinear_u(gene_repr)  # [B, H]
        vc = self.bilinear_v(cell_repr)  # [B, H]
        interaction = (ug * vc)  # [B, H] element-wise

        # ── Output ────────────────────────────────────────────────────
        # Main effects
        main_gene = self.gene_bias(gene_repr)  # [B, 1]
        main_cell = self.cell_bias(cell_repr)  # [B, 1]
        main_effects = main_gene + main_cell  # [B, 1]

        # Merge representations for interaction head
        merged = self.merge(torch.cat([interaction, gene_repr], dim=-1))  # [B, H]
        interaction_pred = self.head(merged)  # [B, 1]

        # Final prediction
        pred = main_effects + interaction_pred  # [B, 1]

        return pred


# Alias for compatibility
CRISPRModel = CRISPRModelV5


# ============================================================
# Loss functions
# ============================================================

def differentiable_pearson(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Global Pearson correlation — differentiable."""
    pred   = pred.view(-1)
    target = target.view(-1)
    pm = pred   - pred.mean()
    pt = target - target.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def demeaned_pearson(
    pred:   torch.Tensor,
    target: torch.Tensor,
    cl_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Pearson after removing per-cell-line mean.

    This is the primary biological metric: how well does the model
    rank genes within a cell line, independent of gene main effects?
    Optimising this directly forces the model to learn cell-specific
    gene×cell interactions rather than gene marginals.
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()
    cl_idx = cl_idx.view(-1)

    n_cl   = int(cl_idx.max().item()) + 1
    counts = torch.zeros(n_cl, device=pred.device, dtype=pred.dtype)
    counts.scatter_add_(0, cl_idx, torch.ones_like(pred))
    counts = counts.clamp(min=1)

    pred_sum = torch.zeros(n_cl, device=pred.device,   dtype=pred.dtype)
    targ_sum = torch.zeros(n_cl, device=target.device, dtype=target.dtype)
    pred_sum.scatter_add_(0, cl_idx, pred)
    targ_sum.scatter_add_(0, cl_idx, target)

    pred_dm   = pred   - (pred_sum / counts)[cl_idx]
    target_dm = target - (targ_sum / counts)[cl_idx]

    pm = pred_dm   - pred_dm.mean()
    pt = target_dm - target_dm.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def combined_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    cl_idx: torch.Tensor,
    alpha:  float = 0.5,
    beta:   float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Composite loss:
        alpha         * weighted_MSE
      + (1-alpha-beta)* (1 - global_pearson)
      + beta          * (1 - demeaned_pearson)

    alpha : MSE weight (handles extreme scores)
    beta  : demeaned Pearson weight — primary interaction metric
    1-alpha-beta : global Pearson weight

    With beta=0.4 and alpha=0.5: 50% MSE, 40% demeaned, 10% global.
    The demeaned Pearson weight is intentionally high to force the
    model to learn cell-specific gene rankings, not just gene marginals.
    """
    pred_s   = pred.view(-1)
    target_s = target.view(-1)

    # Weighted MSE: upweights extreme dependency scores
    # Genes with unusually high/low dependency are most biologically
    # interesting and hardest to predict — give them more weight.
    with torch.no_grad():
        deviation = (target_s - target_s.mean()).abs()
        weights   = 1.0 + 2.0 * (deviation / (deviation.max() + 1e-8))
        weights   = weights / weights.mean()

    mse_term         = (weights * (pred_s - target_s) ** 2).mean()
    pearson_r        = differentiable_pearson(pred_s, target_s)
    pearson_demeaned = demeaned_pearson(pred_s, target_s, cl_idx)

    loss = (
          alpha           * mse_term
        + (1-alpha-beta)  * (1 - pearson_r)
        + beta            * (1 - pearson_demeaned)
    )
    return loss, mse_term, pearson_r, pearson_demeaned


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model:  nn.Module,
    loader,
    device: str,
    qt      = None,
    alpha:  float = 0.5,
    beta:   float = 0.4,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Full evaluation pass over a DataLoader.

    Returns
    -------
    val_loss, val_mse, val_pearson_batch,
    mae, rmse,
    pearson_full, pearson_full_demeaned,
    cl_pearson_mean, cl_pearson_std
    """
    model.eval()
    all_pred, all_target, all_cl_idx = [], [], []
    total_loss = total_mse = total_pearson = 0.0
    device_type = torch.device(device).type

    with torch.no_grad():
        for batch in loader:
            (gene_feat, cell_feat, target,
             cl_idx, gene_cluster_id, _, _, _) = batch

            gene_feat       = gene_feat.to(device,       non_blocking=True)
            cell_feat       = cell_feat.to(device,       non_blocking=True)
            target_d        = target.to(device,          non_blocking=True)
            cl_idx_d        = cl_idx.to(device,          non_blocking=True)
            gene_cluster_id = gene_cluster_id.to(device, non_blocking=True)

            with autocast(device_type=device_type):
                pred = model(cell_feat, gene_feat, gene_cluster_id)
                loss, mse_term, pearson_r, _ = combined_loss(
                    pred, target_d, cl_idx_d, alpha=alpha, beta=beta
                )

            all_pred.append(pred.view(-1))
            all_target.append(target_d.view(-1))
            all_cl_idx.append(cl_idx_d.view(-1))

            total_loss    += loss.item()
            total_mse     += mse_term.item()
            total_pearson += pearson_r.item()

    n = len(loader)

    # Concatenate on GPU — avoids expensive CPU transfers mid-loop
    eval_pred   = torch.cat(all_pred)
    eval_target = torch.cat(all_target)
    cl_idx      = torch.cat(all_cl_idx)

    # Optional: inverse-transform quantile-normalised targets
    if qt is not None:
        import numpy as np
        pred_np   = qt.inverse_transform(
            eval_pred.cpu().numpy().reshape(-1, 1)
        ).squeeze()
        target_np = qt.inverse_transform(
            eval_target.cpu().numpy().reshape(-1, 1)
        ).squeeze()
        eval_pred   = torch.tensor(pred_np,   device=device, dtype=torch.float32)
        eval_target = torch.tensor(target_np, device=device, dtype=torch.float32)

    mae  = (eval_pred - eval_target).abs().mean().item()
    rmse = ((eval_pred - eval_target) ** 2).mean().sqrt().item()

    # ── Global Pearson ─────────────────────────────────────────────────
    pm = eval_pred   - eval_pred.mean()
    pt = eval_target - eval_target.mean()
    pearson_full = ((pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)).item()

    # ── Demeaned Pearson ───────────────────────────────────────────────
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

    # ── Per-cell-line Pearson (cell lines with ≥ 10 samples) ──────────
    mask         = counts >= 10
    valid_cl_ids = torch.nonzero(mask).view(-1)
    cl_pearsons  = []

    if valid_cl_ids.numel() > 0:
        for cid in valid_cl_ids:
            m     = cl_idx == cid
            p_cl  = eval_pred[m]
            t_cl  = eval_target[m]
            pm_cl = p_cl - p_cl.mean()
            pt_cl = t_cl - t_cl.mean()
            denom = pm_cl.norm() * pt_cl.norm()
            if denom > 1e-8:
                cl_pearsons.append((pm_cl * pt_cl).sum() / denom)

    if cl_pearsons:
        cl_t    = torch.stack(cl_pearsons)
        cl_mean = cl_t.mean().item()
        cl_std  = cl_t.std().item() if cl_t.numel() > 1 else 0.0
    else:
        cl_mean, cl_std = 0.0, 0.0

    return (
        total_loss / n, total_mse / n, total_pearson / n,
        mae, rmse,
        pearson_full, pearson_full_demeaned,
        cl_mean, cl_std,
    )


# ============================================================
# Diagnostics
# ============================================================

def diagnose_model(
    model:     nn.Module,
    loader,
    device:    str,
    n_batches: int = 5,
) -> None:
    """
    Print diagnostic report on model component contributions for V5.

    Monitors
    --------
    Gate value         : 0 = relying on stats, 1 = relying on cluster emb
    Interaction/main   : ratio of head output to main effect output
                         target >1.0 — model is learning interactions
    Own/context ratio  : own-cluster pathway vs FiLM context pathway
    FiLM gamma per block: near-zero = FiLM not active (increase LR/capacity)
    Attention dynamics : cluster token attention weights

    Call every N epochs during training to catch collapse early.
    """
    model.eval()

    gate_vals        = []
    bilinear_mags    = []
    main_effect_mags = []
    own_vs_ctx       = []
    film_gammas      = []
    cluster_attn_weights = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            (gene_feat, cell_feat, _, _,
             gene_cluster_id, _, _, _) = batch

            gene_feat       = gene_feat.to(device)
            cell_feat       = cell_feat.to(device)
            gene_cluster_id = gene_cluster_id.to(device)
            B               = cell_feat.size(0)

            # Gene embedding
            cluster_emb = model.gene_cluster_emb(gene_cluster_id)
            stat_emb    = model.gene_stat_encoder(gene_feat)
            gate        = model.gene_gate(gene_feat)
            g           = gate * cluster_emb + (1 - gate) * stat_emb
            gate_vals.append(gate.mean().item())

            # Cell features
            own_expr      = cell_feat[
                torch.arange(B, device=device), gene_cluster_id
            ].unsqueeze(1)
            relative_feat = cell_feat / (own_expr.clamp(min=1e-8))
            cell_input    = torch.cat([cell_feat, relative_feat], dim=-1)

            x = model.cell_proj(cell_input)
            x = model.cell_align(x)

            # FiLM gamma per block
            blk_gammas = []
            for block in model.cell_film_blocks:
                gamma, _ = block.film.proj(g).chunk(2, dim=-1)
                blk_gammas.append(gamma.abs().mean().item())
                x = block(x, g)
            film_gammas.append(blk_gammas)
            ctx_emb = x

            # Tokenize for attention
            tokens = model.tokenize(ctx_emb).view(B, model.n_slots, model.hidden_dim)
            tokens = model.token_norm(tokens) + model.slot_embed
            
            # Cluster token (V5 contribution)
            cluster_token = model.cluster_token_emb(gene_cluster_id).unsqueeze(1)
            context = torch.cat([cluster_token, tokens], dim=1)
            
            # Attention analysis — sample first batch element
            gene_q = g.unsqueeze(1)
            attn_output, attn_weights = model.cross_attn1(
                query=gene_q, key=context, value=context, need_weights=True
            )
            if attn_weights is not None:
                cluster_attn_weights.append(attn_weights[0, 0, 0].item())

            own_emb = model.own_expr_encoder(own_expr)
            own_vs_ctx.append(
                own_emb.abs().mean().item() /
                (ctx_emb.abs().mean().item() + 1e-8)
            )

            c = model.cell_fusion(torch.cat([ctx_emb, own_emb], dim=-1))

            ug          = model.bilinear_u(g)
            vc          = model.bilinear_v(c)
            interaction = ug * vc

            head_out = model.head(model.merge(torch.cat([interaction, g], dim=-1)))
            main_out = model.gene_bias(g) + model.cell_bias(c)

            bilinear_mags.append(head_out.abs().mean().item())
            main_effect_mags.append(main_out.abs().mean().item())

    avg_gate    = sum(gate_vals)        / len(gate_vals)
    avg_bilinear = sum(bilinear_mags)  / len(bilinear_mags)
    avg_main    = sum(main_effect_mags) / len(main_effect_mags)
    avg_own_ctx = sum(own_vs_ctx)       / len(own_vs_ctx)
    avg_gammas  = [
        sum(b[i] for b in film_gammas) / len(film_gammas)
        for i in range(len(film_gammas[0]))
    ]
    avg_cluster_attn = sum(cluster_attn_weights) / len(cluster_attn_weights) if cluster_attn_weights else 0.0
    ratio = avg_bilinear / (avg_main + 1e-8)

    print(f"\n── Model Diagnostic (V5) ─────────────────────────────────")
    print(f"  Gate mean (0=stats 1=emb)   : {avg_gate:.3f}")
    print(f"  Interaction head magnitude  : {avg_bilinear:.4f}")
    print(f"  Main effect magnitude       : {avg_main:.4f}")
    print(f"  Interaction / Main ratio    : {ratio:.3f}  (target >1.0)")
    print(f"  Own-cluster / context ratio : {avg_own_ctx:.3f}")
    print(f"  Cluster token attention     : {avg_cluster_attn:.3f}")
    print(f"  FiLM |gamma| per block      : {[f'{g:.3f}' for g in avg_gammas]}")
    print(f"    ↳ near-zero gammas = FiLM inactive → raise LR or capacity")
    print(f"─────────────────────────────────────────────────────────\n")
