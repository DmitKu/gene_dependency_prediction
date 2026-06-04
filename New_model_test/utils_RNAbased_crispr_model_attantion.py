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
    cell_lines/model_id           : bytes array   [n_cell_lines]
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
            cl_model_ids      = f["cell_lines/model_id"][:]
            all_gene_feat     = f["genes/features"][:]
            all_model_ids     = f["genes/model_id"][:]      # cell-line ID per gene row
            all_gene_ids      = f["genes/gene_id"][:]       # gene ID per gene row
            all_crispr        = f["genes/crispr"][:]
    
            self.gene_feat  = torch.tensor(all_gene_feat[gene_indices], dtype=torch.float32)
            self.crispr     = torch.tensor(all_crispr[gene_indices],    dtype=torch.float32)
    
            # String IDs for every sample in this split
            self.gene_ids   = [all_gene_ids[i].decode()  for i in gene_indices]  # which gene
            self.model_ids  = [all_model_ids[i].decode() for i in gene_indices]  # which cell line
    
        # Build cell-line lookup: model_id string → integer row index in cl_features
        self.cl_model_id_to_index = {mid.decode(): i for i, mid in enumerate(cl_model_ids)}
    
        # Integer cell-line index per sample — used to fetch cl_features in __getitem__
        self.cl_indices = torch.tensor(
            [self.cl_model_id_to_index[mid] for mid in self.model_ids],
            dtype=torch.long,
        )
    
        # Build reverse lookup: integer index → model_id string (for reporting)
        self.cl_index_to_model_id = {v: k for k, v in self.cl_model_id_to_index.items()}
    
        # Validate: no duplicate (gene_id, model_id) pairs in this split
        pairs    = list(zip(self.gene_ids, self.model_ids))
        n_unique = len(set(pairs))
        if n_unique != len(pairs):
            import warnings
            warnings.warn(
                f"Split '{split}': {len(pairs) - n_unique} duplicate "
                f"(gene_id, model_id) pairs found — check data pipeline"
            )
    
        print(f"  → {len(self.gene_feat):,} samples | "
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
            self.cl_indices[idx],      # int tensor — cell-line row index
            self.gene_ids[idx],        # str — gene identifier
            self.model_ids[idx],       # str — cell-line model_id
            idx,                       # int — position within split
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
    Interpretable tokenizer.

    Each biological cluster becomes one token.

    Token i:
        token_i = value_i * embedding_i

    Attention weights can be directly mapped back to
    cluster IDs.
    """

    def __init__(
        self,
        cell_feat_dim: int,
        d_model: int,
        gene_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.cell_feat_dim = cell_feat_dim
        self.d_model = d_model

        # one learnable embedding per cluster
        self.cluster_embed = nn.Parameter(
            torch.randn(cell_feat_dim, d_model) * 0.02
        )

        # optional gene conditioning
        self.film = FiLMLayer(
            cond_dim=gene_dim,
            feature_dim=cell_feat_dim,
        )

        self.norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cell_features,
        gene_emb,
    ):
        """
        cell_features:
            [B, 2388]

        returns:
            [B, 2388, d_model]
        """

        values = self.film(
            cell_features,
            gene_emb,
        )

        tokens = (
            values.unsqueeze(-1)
            * self.cluster_embed.unsqueeze(0)
        )

        tokens = self.norm(tokens)

        return self.dropout(tokens)


class LinearBypass(nn.Module):
    def __init__(self, gene_dim: int, cell_dim: int, rank: int = 32, reg_scale: float = 1e-3):
        super().__init__()
        self.gene_proj = nn.Linear(gene_dim, rank, bias=False)
        self.cell_proj = nn.Linear(cell_dim, rank, bias=False)
        self.scale     = nn.Parameter(torch.ones(1) * 0.5)
        self.gate      = nn.Parameter(torch.tensor(-2.0))  # starts near-closed (sigmoid(-2) ≈ 0.12)
        self.reg_scale = reg_scale
        nn.init.normal_(self.gene_proj.weight, 0, 0.01)
        nn.init.normal_(self.cell_proj.weight, 0, 0.01)

    def forward(self, gene_emb, cell_summary):
        g = F.normalize(self.gene_proj(gene_emb), dim=-1)
        c = F.normalize(self.cell_proj(cell_summary), dim=-1)
        raw = self.scale * (g * c).sum(-1, keepdim=True)
        gate = torch.sigmoid(self.gate)
        reg  = raw.pow(2).mean() * self.reg_scale
        return gate * raw, reg

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
    compress_dim       : int    Cell tokenizer compression dimension (default 1024).
    dropout            : float  Default dropout rate (default 0.2).
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
        # gene_hidden = 64   ← REMOVE this hardcoded line

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
            cell_feat_dim=cell_features_size,
            d_model=gene_hidden,
            gene_dim=gene_hidden,
            dropout=dropout,
        )

        # ── ③ First cross-attention ─────────────────────────────────────────
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=gene_hidden, num_heads=n_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm   = nn.LayerNorm(gene_hidden)

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
            nn.Linear(gene_hidden * 2, hidden_dim),  # ← gene_hidden*2 because we cat gene_emb
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── ⑦ Conditioning projection ───────────────────────────────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(gene_hidden * 2, gene_hidden),  # ← concat gene + cell
            nn.LayerNorm(gene_hidden),
            nn.GELU(),
        )
        # ── ⑧ Trunk ─────────────────────────────────────────────────────────
        self.trunk_res1 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)  # ← was 0.25
        self.trunk_res2 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)  # ← was 0.25
        self.trunk_res3 = GELUResidualBlock(hidden_dim, dropout=0.15, cond_dim=gene_hidden)  # ← was 0.25
        

        # ── ⑨ Head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),        # ← was ReLU (fixed separately)
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        
        # scaling, importatnt strong but rear cases
        self.out_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.out_shift = nn.Parameter(torch.zeros(1))

        self._init_weights()
    # ------------------------------------------------------------------
    def forward(
        self,
        cell_features: torch.Tensor,   # [B, F]
        gene_features: torch.Tensor,   # [B, G]
        ablate_bypass: bool = False,
    ) -> torch.Tensor:                 # [B, 1]

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

        # ── ⑥ Head & Trunk (Merged and Fixed) ───────────────────────────────
        # gene_mean_pred: absorbs gene main effect explicitly
        x_input = torch.cat([cell_context2, gene_emb], dim=-1)
        x = self.merge(x_input)
        gene_cond = self.cond_proj(torch.cat([gene_emb, cell_context2], dim=-1))
        x = self.trunk_res1(x, cond=gene_cond)
        x = self.trunk_res2(x, cond=gene_cond)
        x = self.trunk_res3(x, cond=gene_cond)
        trunk_pred = self.head(x)
        raw = trunk_pred + bypass_logit
        return raw * self.out_scale + self.out_shift, bypass_reg

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
    """Raw Pearson — dominated by gene main effect."""
    # Use view(-1) to safely handle batch_size=1
    pred   = pred.view(-1)
    target = target.view(-1)
    pm = pred   - pred.mean()
    pt = target - target.mean()
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)


def demeaned_pearson(
    pred: torch.Tensor,
    target: torch.Tensor,
    cl_idx: torch.Tensor,
) -> torch.Tensor:
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    cl_idx = cl_idx.view(-1)
    
    n_cl = int(cl_idx.max().item()) + 1
    
    # ⚡ Optimized: scatter_add
    counts = torch.zeros(
        n_cl,
        device=pred.device,
        dtype=pred.dtype,
    )
    
    counts.scatter_add_(
        0,
        cl_idx,
        torch.ones_like(pred),
    )
    
    counts = counts.clamp(min=1)
    
    pred_sum = torch.zeros(
        n_cl,
        device=pred.device,
        dtype=pred.dtype,
    )
    
    pred_sum.scatter_add_(
        0,
        cl_idx,
        pred,
    )
    
    target_sum = torch.zeros(
        n_cl,
        device=target.device,
        dtype=target.dtype,
    )
    
    target_sum.scatter_add_(
        0,
        cl_idx,
        target,
    )
    
    pred_dm = pred - (pred_sum / counts)[cl_idx]
    target_dm = target - (target_sum / counts)[cl_idx]
    
    pm = pred_dm - pred_dm.mean()
    pt = target_dm - target_dm.mean()
    
    return (pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)




def combined_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    cl_idx: torch.Tensor,
    alpha:  float = 0.5,
    beta:   float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # Use view(-1) to safely handle batch_size=1
    pred_s   = pred.view(-1)
    target_s = target.view(-1)

    with torch.no_grad():
        target_mean = target_s.mean()
        deviation   = (target_s - target_mean).abs()
        weights     = 1.0 + 2.0 * (deviation / (deviation.max() + 1e-8))
        weights     = weights / weights.mean()

    mse_term         = (weights * (pred_s - target_s) ** 2).mean()
    pearson_r        = differentiable_pearson(pred_s, target_s)
    pearson_demeaned = demeaned_pearson(pred_s, target_s, cl_idx)

    loss = (alpha * mse_term
            + (1 - alpha - beta) * (1 - pearson_r)
            + beta * (1 - pearson_demeaned))

    return loss, mse_term, pearson_r, pearson_demeaned

# ============================================================
# Evaluation
# ============================================================
def evaluate(
    model: nn.Module,
    loader,
    device: str,
    qt=None,
    alpha: float = 0.5,
    ablate_bypass: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    High-performance evaluation using GPU-native vectorization.
    """
    model.eval()
    all_pred, all_target, all_cl_idx = [], [], []
    total_loss = total_mse = total_pearson = 0.0

    # Safely parse device type string for autocast context manager
    device_type = torch.device(device).type

    with torch.no_grad():
        for gene_feat, cell_feat, target, cl_idx, _, _, _ in loader:
            gene_feat = gene_feat.to(device, non_blocking=True)
            cell_feat = cell_feat.to(device, non_blocking=True)
            target_d  = target.to(device, non_blocking=True)
            cl_idx    = cl_idx.to(device, non_blocking=True)

            with autocast(device_type=device_type):
                pred, _ = model(cell_feat, gene_feat, ablate_bypass=ablate_bypass)
                loss, mse_term, pearson_r, _ = combined_loss(pred, target_d, cl_idx, alpha=alpha)

            # Accumulate on GPU to prevent expensive synchronization bottlenecks
            # Using view(-1) prevents flattening bugs if the final batch size is 1
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

    # Concatenate directly on the GPU
    eval_pred   = torch.cat(all_pred)
    eval_target = torch.cat(all_target)
    cl_idx      = torch.cat(all_cl_idx)

    # Handle Quantile Transformation tracking if present
    if qt is not None:
        import numpy as np
        pred_np   = qt.inverse_transform(eval_pred.cpu().numpy().reshape(-1, 1)).squeeze()
        target_np = qt.inverse_transform(eval_target.cpu().numpy().reshape(-1, 1)).squeeze()
        eval_pred   = torch.tensor(pred_np,   device=device, dtype=torch.float32)
        eval_target = torch.tensor(target_np, device=device, dtype=torch.float32)

    # Basic error tracking
    mae  = (eval_pred - eval_target).abs().mean().item()
    rmse = ((eval_pred - eval_target) ** 2).mean().sqrt().item()

    # ── Global Pearson ───────────────────────────────────────────────────────
    pm = eval_pred   - eval_pred.mean()
    pt = eval_target - eval_target.mean()
    pearson_full = ((pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)).item()

    # ── Vectorized Demeaning ─────────────────────────────────────────────────
    # We map directly to unique integers using cl_idx instead of string group loops
    n_cl   = cl_idx.max() + 1
    counts = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, torch.ones_like(eval_pred)).clamp(min=1)
    p_sums = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, eval_pred)
    t_sums = torch.zeros(n_cl, device=device).scatter_add_(0, cl_idx, eval_target)

    pred_dm   = eval_pred   - (p_sums / counts)[cl_idx]
    target_dm = eval_target - (t_sums / counts)[cl_idx]

    pm_d = pred_dm   - pred_dm.mean()
    pt_d = target_dm - target_dm.mean()
    pearson_full_demeaned = ((pm_d * pt_d).sum() / (pm_d.norm() * pt_d.norm() + 1e-8)).item()

    # ── Per-Cell-Line Pearson ────────────────────────────────────────────────
    # Vectorized filtering for cell lines with 10 or more samples
    mask = counts >= 10
    valid_cl_ids = torch.nonzero(mask).view(-1) # view(-1) prevents loop crashes on single items

    cl_pearsons = []
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

    # Convert results to tracking tensors cleanly
    if cl_pearsons:
        cl_t = torch.stack(cl_pearsons)
        cl_mean = cl_t.mean().item()
        cl_std  = cl_t.std().item() if cl_t.numel() > 1 else 0.0
    else:
        cl_mean, cl_std = 0.0, 0.0

    return (
        val_loss, val_mse, val_pearson,
        mae, rmse,
        pearson_full,
        pearson_full_demeaned,
        cl_mean, cl_std
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
    """Print bypass vs trunk head contribution to output.
    
    Call every N epochs to detect shortcut learning collapse.
    Healthy model: bypass/head ratio < 1.0, attn entropy > 2.0
    Collapse risk: bypass/head ratio > 2.0, attn entropy < 0.5
    """
    model.eval()
    bypass_mags, head_mags, attn_entropies = [], [], []

    with torch.no_grad():
        for i, (gene_feat, cell_feat, target, cl_idx, gene_id, model_id, _) in enumerate(loader):
            if i >= n_batches:
                break
            gene_feat = gene_feat.to(device)
            cell_feat = cell_feat.to(device)

            gene_emb    = model.gene_res(model.gene_encoder(gene_feat))
            cell_tokens = model.cell_tokenizer(cell_feat, gene_emb)

            attn_out1, weights1 = model.cross_attn(
                gene_emb.unsqueeze(1),
                cell_tokens,
                cell_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            cell_context = model.attn_norm(attn_out1.squeeze(1) + gene_emb)

            attn_out2, weights2 = model.cross_attn2(
                cell_context.unsqueeze(1),
                cell_tokens,
                cell_tokens,
                need_weights=True,
                average_attn_weights=True,
            )
            cell_context2 = model.attn_norm2(
                attn_out2.squeeze(1) + cell_context
            )

            
            cell_summary1 = (
                weights1.squeeze(1).unsqueeze(-1) * cell_tokens
            ).sum(dim=1)
            
            cell_summary2 = (
                weights2.squeeze(1).unsqueeze(-1) * cell_tokens
            ).sum(dim=1)
            
            cell_summary = 0.5 * cell_summary1 + 0.5 * cell_summary2

            bypass, _ = model.linear_bypass(gene_emb, cell_summary)
            
            x = model.merge(
                torch.cat([cell_context2, gene_emb], dim=-1)
            )
            gene_cond = model.cond_proj(
                torch.cat([gene_emb, cell_context2], dim=-1)
            )
            
            x        = model.trunk_res1(x, cond=gene_cond)
            x        = model.trunk_res2(x, cond=gene_cond)
            x        = model.trunk_res3(x, cond=gene_cond)
            
            head_out = model.head(x)

            scaled_bypass = (bypass * model.out_scale).abs().mean().item()
            scaled_head   = (head_out * model.out_scale).abs().mean().item()
            bypass_mags.append(scaled_bypass)
            head_mags.append(scaled_head)

            w1      = weights1.squeeze(1)
            entropy = -(w1 * (w1 + 1e-8).log()).sum(-1).mean()

            attn_entropies.append(entropy.item())

    bypass_mean  = sum(bypass_mags)     / len(bypass_mags)
    head_mean    = sum(head_mags)       / len(head_mags)
    entropy_mean = sum(attn_entropies)  / len(attn_entropies)
    ratio        = bypass_mean / (head_mean + 1e-8)

    print(f"\n── Bypass Diagnostic ───────────────────────────────")
    print(f"  Bypass magnitude  : {bypass_mean:.4f}")
    print(f"  Head magnitude    : {head_mean:.4f}")
    print(f"  Bypass/Head ratio : {ratio:.3f}  (healthy: <1.0  collapse: >2.0)")
    print(f"  Attn entropy      : {entropy_mean:.4f}  (healthy: >2.0  collapsed: <0.5)")
    print(f"────────────────────────────────────────────────────\n")