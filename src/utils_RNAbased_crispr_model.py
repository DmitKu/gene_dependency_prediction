# -*- coding: utf-8 -*-
"""
crispr_model.py
===============
CRISPR Sensitivity Model v3 — Cross-Attention + Linear Bypass

All dataset, model, loss, and evaluation components.
Import this module from training scripts.

Architecture summary
--------------------
  Gene features  →  gene_encoder  →  gene_res  →  gene_emb [B, 128]
  Cell features  →  cell_tokenizer              →  cell_tokens [B, 64, 128]

  1st cross-attention  (query=gene_emb)         →  cell_context [B, 128]
  2nd cross-attention  (query=cell_context)      →  cell_context2 [B, 128]

  bypass_logit  = LinearBypass(gene_emb, attn_weighted_summary2)

  x = merge(cat[gene_emb, cell_context2])
  combined_cond = cond_proj(cat[gene_emb, cell_context2])

  x = trunk_res1/2/3(x, cond=combined_cond)
  output = head(x) + bypass_logit

Fixes vs v2
-----------
  FIX 1  cell_context = norm(attn_out)          — no gene_emb residual
  FIX 2  bypass uses attn-weighted slot summary  — not flat mean
  FIX 3  gene_emb appears exactly once in merge
  FIX 4  FiLM conditioned on gene + cell combined
"""

import time
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
    Loads gene-level CRISPR sensitivity data from an HDF5 file.

    Expected HDF5 layout
    --------------------
    index/splits/{train,val,test}  : 1-D integer index array
    cell_lines/features            : float32 array [n_cell_lines, F]
    cell_lines/model_ids           : bytes array   [n_cell_lines]
    genes/features                 : float32 array [n_genes, G]
    genes/model_id                 : bytes array   [n_genes]
    genes/crispr                   : float32 array [n_genes]

    Returns (per __getitem__)
    -------------------------
    gene_feat   : Tensor [G]
    cell_feat   : Tensor [F]
    crispr      : Tensor [1]
    cl_idx      : LongTensor scalar — index into cell_lines/features
    """

    def __init__(self, h5_path: str, split: str = "train"):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        print(f"Loading {split} data …")
        t0 = time.time()

        with h5py.File(h5_path, "r") as f:
            gene_indices      = f[f"index/splits/{split}"][:]
            self.cl_features  = torch.tensor(f["cell_lines/features"][:], dtype=torch.float32)
            cl_model_ids      = f["cell_lines/model_ids"][:]
            all_gene_feat     = f["genes/features"][:]
            all_model_ids     = f["genes/model_id"][:]
            all_crispr        = f["genes/crispr"][:]

            self.gene_feat = torch.tensor(all_gene_feat[gene_indices], dtype=torch.float32)
            self.model_ids = [all_model_ids[i].decode() for i in gene_indices]
            self.crispr    = torch.tensor(all_crispr[gene_indices],    dtype=torch.float32)

        self.cl_model_id_to_index = {mid.decode(): i for i, mid in enumerate(cl_model_ids)}
        self.cl_indices = torch.tensor(
            [self.cl_model_id_to_index[mid] for mid in self.model_ids],
            dtype=torch.long,
        )
        print(f"  → {len(self.gene_feat):,} samples loaded in {time.time() - t0:.2f}s")

    def __len__(self) -> int:
        return len(self.gene_feat)

    def __getitem__(self, idx):
        return (
            self.gene_feat[idx],
            self.cl_features[self.cl_indices[idx]],
            self.crispr[idx].unsqueeze(0),
            self.cl_indices[idx],
            idx, 
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
        # Removed self.act = nn.GELU() here

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        out = self.net(self.norm(x))
        if self.film is not None and cond is not None:
            out = self.film(out, cond)
        
        # FIX: Return pure addition to maintain the identity path
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
    """
    Projects a flat cell-line feature vector into a sequence of tokens
    suitable for cross-attention. Adds learnable slot embeddings for 
    permutation variance.
    """

    def __init__(
        self,
        cell_feat_dim: int,
        n_slots:       int,
        d_model:       int,
        compress_dim:  int   = 1024,  # Increased to prevent rank collapse
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
        self.tokenize   = nn.Linear(compress_dim, n_slots * d_model)
        self.token_norm = RMSNorm(d_model)
        
        # Learnable positional/slot embeddings
        self.slot_embed = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        z = self.compress(x)
        t = self.tokenize(z).reshape(B, self.n_slots, self.d_model)
        
        # Broadcast and add slot embeddings to the normalized tokens
        return self.token_norm(t) + self.slot_embed


class LinearBypass(nn.Module):
    """
    Low-rank bilinear bypass path.

    Computes a scalar logit from gene and cell representations without
    passing through the deep trunk, providing a direct gradient path.

        logit = sum( W_g @ gene_emb  *  W_c @ cell_summary )
    """

    def __init__(self, gene_dim: int, cell_dim: int, rank: int = 32):
        super().__init__()
        self.gene_proj = nn.Linear(gene_dim, rank, bias=False)
        self.cell_proj = nn.Linear(cell_dim, rank, bias=False)
        nn.init.normal_(self.gene_proj.weight, 0, 0.1)
        nn.init.normal_(self.cell_proj.weight, 0, 0.1)

    def forward(self, gene_emb: torch.Tensor, cell_summary: torch.Tensor) -> torch.Tensor:
        return (self.gene_proj(gene_emb) * self.cell_proj(cell_summary)).sum(-1, keepdim=True)


# ============================================================
# Main model
# ============================================================

class CRISPRSensitivityModelV3(nn.Module):
    """
    CRISPR sensitivity predictor — v3.

    Parameters
    ----------
    cell_features_size : int    Input dimension of cell-line features.
    gene_features_size : int    Input dimension of gene features.
    hidden_dim         : int    Trunk hidden dimension (default 256).
    n_attn_slots       : int    Number of cell tokens for cross-attention (default 64).
    n_attn_heads       : int    Attention heads (default 4).
    bypass_rank        : int    Rank of the bilinear bypass (default 32).
    compress_dim       : int    Cell tokenizer compression dimension (default 256).
    dropout            : float  Default dropout rate (default 0.2).
    """

    def __init__(
        self,
        cell_features_size: int   = 2388,
        gene_features_size: int   = 27,
        hidden_dim:         int   = 128,
        n_attn_slots:       int   = 64,
        n_attn_heads:       int   = 4,
        bypass_rank:        int   = 32,
        compress_dim:       int   = 512,
        dropout:            float = 0.2,
    ):
        super().__init__()
        gene_hidden = 64

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

        # ── ② Cell tokenizer (64 slots) ─────────────────────────────────────
        self.cell_tokenizer = CellTokenizer(
            cell_feat_dim = cell_features_size,
            n_slots       = n_attn_slots,
            d_model       = gene_hidden,
            compress_dim  = 1024,
            dropout       = dropout,
        )

        # ── ③ First cross-attention ─────────────────────────────────────────
        # Query: gene_emb — "which cell slots are relevant for this gene?"
        # FIX 1: norm applied to attn_out only (no gene_emb residual)
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm   = nn.LayerNorm(gene_hidden)

        # ── ④ Second cross-attention ────────────────────────────────────────
        # Query: cell_context — "given what I found, what else is relevant?"
        # Refines the cell representation using the first-pass output as query.
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm2  = nn.LayerNorm(gene_hidden)

        # ── ⑤ Linear bypass (FIX 2: uses second-pass attn weights) ─────────
        self.linear_bypass = LinearBypass(
            gene_dim=gene_hidden, cell_dim=gene_hidden, rank=bypass_rank,
        )

        # ── ⑥ Trunk Input Projection ────────────────────────────────────────
        # Takes only the refined cell context
        self.merge = nn.Sequential(
            nn.Linear(gene_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ── ⑦ Conditioning Projection ───────────────────────────────────────
        # Takes only the gene embedding
        self.cond_proj = nn.Sequential(
            nn.Linear(gene_hidden, gene_hidden),
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
        )


        # ── ⑦ Trunk (3 Pre-LN FiLM-conditioned residual blocks) ─────────────
        self.trunk_res1 = GELUResidualBlock(hidden_dim, dropout=0.25, cond_dim=gene_hidden)
        self.trunk_res2 = GELUResidualBlock(hidden_dim, dropout=0.25, cond_dim=gene_hidden)
        self.trunk_res3 = GELUResidualBlock(hidden_dim, dropout=0.25, cond_dim=gene_hidden)

        # ── ⑧ Head (LayerNorm removed to allow large output magnitude) ───────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        cell_features: torch.Tensor,   # [B, F]
        gene_features: torch.Tensor,   # [B, G]
        ablate_bypass: bool = False,
    ) -> torch.Tensor:                 # [B, 1]

        # ── ① Gene encoding ─────────────────────────────────────────────────
        gene_emb = self.gene_res(self.gene_encoder(gene_features))   # [B, 128]

        # ── ② Cell tokenization ─────────────────────────────────────────────
        cell_tokens = self.cell_tokenizer(cell_features)             # [B, 64, 128]

        # ── ③ First cross-attention (query = gene_emb) ──────────────────────
        attn_out1, _ = self.cross_attn(
            gene_emb.unsqueeze(1), cell_tokens, cell_tokens,
            need_weights=True, average_attn_weights=True,
        )
        cell_context = self.attn_norm(attn_out1.squeeze(1))          # [B, 128]

        # ── ④ Second cross-attention (query = cell_context) ─────────────────
        attn_out2, attn_weights2 = self.cross_attn2(
            cell_context.unsqueeze(1), cell_tokens, cell_tokens,
            need_weights=True, average_attn_weights=True,
        )
        # Residual: refined context + first-pass context
        cell_context2 = self.attn_norm2(
            attn_out2.squeeze(1) + cell_context                      # [B, 128]
        )

        # ── ⑤ Bypass (FIX 2: attention-weighted slot summary) ───────────────
        cell_summary = (
            attn_weights2.squeeze(1).unsqueeze(-1) * cell_tokens
        ).sum(dim=1)                                                  # [B, 128]

        bypass_logit = self.linear_bypass(gene_emb, cell_summary)    # [B, 1]
        if ablate_bypass:
            bypass_logit = torch.zeros_like(bypass_logit)

        # ── ⑥ Trunk Input ───────────────────────────────────────────────────
        # Pass only the refined cell state into the trunk
        x = self.merge(cell_context2)                                # [B, 256]

        # ── ⑦ Combined Conditioning ─────────────────────────────────────────
        # Derive FiLM conditioning purely from the gene representation
        gene_cond = self.cond_proj(gene_emb)                         # [B, 128]

        # ── ⑧ Trunk ─────────────────────────────────────────────────────────
        # Residual blocks apply gene-specific modulation to the cell state
        x = self.trunk_res1(x, cond=gene_cond)
        x = self.trunk_res2(x, cond=gene_cond)
        x = self.trunk_res3(x, cond=gene_cond)

        # ── ⑨ Head ──────────────────────────────────────────────────────────
        return self.head(x) + bypass_logit                   # [B, 1]

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

def differentiable_pearson(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Pearson correlation for use as a training objective."""
    pred   = pred.squeeze()
    target = target.squeeze()
    pm = pred   - pred.mean()
    pt = target - target.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def combined_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    alpha:  float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted combination of MSE, Pearson, and standard-deviation matching.

    Loss = alpha * weighted_MSE  +  (1-alpha) * (1 - Pearson)  +  0.1 * |std_pred - std_target|

    The MSE term up-weights extreme negative values (sensitive dependencies)
    to counteract label imbalance:
        weight = 1.0   (default)
        weight = 2.0   if target < -0.7
        weight = 4.0   if target < -1.5

    Returns
    -------
    loss      : scalar Tensor — total loss
    mse_term  : scalar Tensor — weighted MSE component
    pearson_r : scalar Tensor — Pearson correlation
    """
    pred_s   = pred.squeeze()
    target_s = target.squeeze()

    weights = torch.ones_like(target_s)
    weights[target_s < -0.7] = 2.0
    weights[target_s < -1.5] = 4.0
    mse_term = (weights * (pred_s - target_s) ** 2).mean()

    pearson_r = differentiable_pearson(pred_s, target_s)
    std_loss  = (pred_s.std() - target_s.std()).abs()

    loss = alpha * mse_term + (1 - alpha) * (1 - pearson_r) + 0.1 * std_loss
    return loss, mse_term, pearson_r


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model:  nn.Module,
    loader,
    device: str,
    qt=None,
) -> tuple[float, float, float, float, float]:
    """
    Evaluate *model* on *loader* and return metrics in the original
    (Chronos) label space when a QuantileTransformer *qt* is provided.

    Parameters
    ----------
    model  : trained CRISPRSensitivityModelV3
    loader : DataLoader yielding (gene_feat, cell_feat, target, cl_idx)
    device : 'cuda' or 'cpu'
    qt     : fitted sklearn QuantileTransformer or None

    Returns
    -------
    mae             : float — mean absolute error
    rmse            : float — root-mean-squared error
    pearson_full    : float — global Pearson correlation
    pearson_pcl     : float — mean per-cell-line Pearson
    pearson_pcl_sd  : float — std of per-cell-line Pearson values
    """
    model.eval()
    all_pred, all_target, all_cl = [], [], []

    with torch.no_grad():
        for gene_feat, cell_feat, target, cl_idx, _ in loader:
            gene_feat = gene_feat.to(device, non_blocking=True)
            cell_feat = cell_feat.to(device, non_blocking=True)
            with autocast(device):
                pred = model(cell_feat, gene_feat)
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_cl.append(cl_idx.cpu())

    all_pred   = torch.cat(all_pred).squeeze()
    all_target = torch.cat(all_target).squeeze()
    all_cl     = torch.cat(all_cl)

    if qt is not None:
        import numpy as np
        pred_np   = qt.inverse_transform(all_pred.numpy().reshape(-1, 1)).squeeze()
        target_np = qt.inverse_transform(all_target.numpy().reshape(-1, 1)).squeeze()
        eval_pred   = torch.tensor(pred_np,   dtype=torch.float32)
        eval_target = torch.tensor(target_np, dtype=torch.float32)
    else:
        eval_pred, eval_target = all_pred, all_target

    mae  = (eval_pred - eval_target).abs().mean().item()
    rmse = ((eval_pred - eval_target) ** 2).mean().sqrt().item()

    pm = eval_pred   - eval_pred.mean()
    pt = eval_target - eval_target.mean()
    pearson_full = ((pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)).item()

    cl_pearsons = []
    for cl_id in all_cl.unique():
        mask = all_cl == cl_id
        if mask.sum() < 10:
            continue
        p, t  = eval_pred[mask], eval_target[mask]
        pm_cl = p - p.mean()
        pt_cl = t - t.mean()
        denom = pm_cl.norm() * pt_cl.norm()
        if denom < 1e-8:
            continue
        cl_pearsons.append(((pm_cl * pt_cl).sum() / denom).item())

    cl_t = torch.tensor(cl_pearsons)
    return mae, rmse, pearson_full, cl_t.mean().item(), cl_t.std().item()