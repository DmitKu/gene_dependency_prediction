# -*- coding: utf-8 -*-
"""
src/utils_crispr_model_v4.py
============================
CRISPR Sensitivity Model v4 — Deep Cell Encoder + FiLM + Multi-head Bilinear

Architecture summary
--------------------

GENE SIDE (26 weak cluster-stat features + discrete cluster identity)
  cluster_id  →  Embedding(n_clusters, rank)     →  cluster_emb  [B, rank]
  gene_stats  →  gene_stat_encoder (MLP)         →  stat_emb     [B, rank]
  gene_stats  →  gene_gate (sigmoid)             →  gate         [B, rank]
  g = gate * cluster_emb + (1 - gate) * stat_emb                 [B, rank]

  gate is learned per-dimension: well-populated clusters shift weight
  toward the embedding; sparse clusters rely more on the 26 stats.

CELL SIDE (2388 RNA cluster-sum features, semantically indexed by cluster)

  Privileged extraction (explicitly hand the model the most important feature):
    own_expr   = cell_feat[:, gene_cluster_id]                   [B, 1]
    rel_feat   = cell_feat / (own_expr + 1e-8)                   [B, 2388]
    cell_input = cat[cell_feat, rel_feat]                        [B, 4776]

  FiLM-conditioned deep encoder (gene cluster steers how cell features are read):
    cell_input → cell_proj (4776→1024) → cell_align (1024→rank)
               → ResidualFiLMBlock × n_cell_layers, cond=g
               → ctx_emb                                         [B, rank]

  Own-cluster pathway (dedicated pathway for the privileged scalar):
    own_expr → own_expr_encoder                  → own_emb       [B, rank]

  Cell fusion:
    cat[ctx_emb, own_emb] → cell_fusion          → c             [B, rank]

INTERACTION (multi-head bilinear — matches known low-rank structure of
             CRISPR dependency matrices, rank ~10-30)
  ug          = bilinear_u(g).view(B, n_heads, head_dim)
  vc          = bilinear_v(c).view(B, n_heads, head_dim)
  interaction = (ug * vc).view(B, rank)                          [B, rank]

OUTPUT
  main_effects = gene_bias(g) + cell_bias(c)   # zero-init → clean split
  x            = cat[g, c, interaction]         [B, rank*3]
  out          = main_effects + head(x)         [B, 1]

HDF5 layout expected
--------------------
  index/splits/{train,val,test}  : int64  [n_split_samples]
  cell_lines/features            : float32[n_cell_lines, 2388]
  cell_lines/model_id            : bytes  [n_cell_lines]
  genes/features                 : float32[n_genes, 26]
  genes/model_id                 : bytes  [n_genes]   ← cell-line id per gene row
  genes/gene_id                  : bytes  [n_genes]
  genes/cluster_id               : int32  [n_genes]   ← gene cluster 0..2387
  genes/crispr                   : float32[n_genes]
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
    Loads all data into CPU RAM on construction to eliminate HDF5 I/O
    bottlenecks during training. With 6M samples this is the correct
    strategy — random HDF5 access at batch time is extremely slow.

    Returns per __getitem__ (8-tuple):
    -----------------------------------
    gene_feat       : float32 Tensor [26]
    cell_feat       : float32 Tensor [2388]
    crispr          : float32 Tensor [1]
    cl_idx          : int64   scalar  — row index into cell_lines/features
                                        used for demeaned Pearson in loss
    gene_cluster_id : int64   scalar  — cluster index 0..2387
                                        used for Embedding lookup + own_expr
    gene_id         : str
    model_id        : str
    idx             : int     — position within split (for debugging)
    """

    def __init__(self, h5_path: str, split: str = "train"):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"
        print(f"[GeneDataset] Loading '{split}' split ...")
        t0 = time.time()

        with h5py.File(h5_path, "r") as f:

            # ── Split index ──────────────────────────────────────────────
            gene_indices = f[f"index/splits/{split}"][:]   # [N_split]

            # ── Cell-line features (full matrix, shared across splits) ───
            # Stored once and indexed per sample via cl_indices.
            self.cl_features = torch.tensor(
                f["cell_lines/features"][:], dtype=torch.float32
            )                                              # [n_cl, 2388]
            cl_model_ids = f["cell_lines/model_id"][:]     # bytes [n_cl]

            # ── Gene-level arrays (all genes, sliced by split index) ─────
            all_gene_feat   = f["genes/features"][:]       # [n_genes, 26]
            all_cluster_ids = f["genes/cluster_id"][:]     # [n_genes] int
            all_model_ids   = f["genes/model_id"][:]       # [n_genes] bytes = cell-line id
            all_gene_ids    = f["genes/gene_id"][:]        # [n_genes] bytes
            all_crispr      = f["genes/crispr"][:]         # [n_genes] float

            # ── Slice to this split ──────────────────────────────────────
            self.gene_feat        = torch.tensor(
                all_gene_feat[gene_indices],   dtype=torch.float32
            )
            self.crispr           = torch.tensor(
                all_crispr[gene_indices],      dtype=torch.float32
            )
            self.gene_cluster_ids = torch.tensor(
                all_cluster_ids[gene_indices], dtype=torch.long
            )

            # String IDs — used for duplicate checking and reporting only
            self.gene_ids  = [all_gene_ids[i].decode()  for i in gene_indices]
            self.model_ids = [all_model_ids[i].decode() for i in gene_indices]

        # ── Cell-line lookup: model_id string → integer row in cl_features
        self.cl_model_id_to_index = {
            mid.decode(): i for i, mid in enumerate(cl_model_ids)
        }

        # Integer cl_idx per sample — used in __getitem__ and demeaned Pearson
        self.cl_indices = torch.tensor(
            [self.cl_model_id_to_index[mid] for mid in self.model_ids],
            dtype=torch.long,
        )

        # Reverse lookup for reporting
        self.cl_index_to_model_id = {
            v: k for k, v in self.cl_model_id_to_index.items()
        }

        # ── Sanity check: no duplicate (gene_id, model_id) pairs ────────
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
            self.gene_feat[idx],                          # [26]
            self.cl_features[self.cl_indices[idx]],       # [2388]
            self.crispr[idx].unsqueeze(0),                # [1]
            self.cl_indices[idx],                         # scalar long
            self.gene_cluster_ids[idx],                   # scalar long
            self.gene_ids[idx],                           # str
            self.model_ids[idx],                          # str
            idx,                                          # int
        )


# ============================================================
# Building blocks
# ============================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

        out = (1 + gamma) * x + beta
        [gamma, beta] = Linear(cond)    cond = gene embedding

    Small init on weight so FiLM starts near identity (gamma≈0, beta≈0)
    and only grows if the gradient signal demands it.
    """
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feature_dim)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class ResidualFiLMBlock(nn.Module):
    """
    Pre-LayerNorm residual block with FiLM conditioning.

    Forward:
        branch = net(LayerNorm(x))     [dim → dim*2 → dim]
        branch = FiLM(branch, cond)    gene cluster rescales branch
        out    = branch + x            clean residual stream

    FiLM is applied to the branch BEFORE adding back to residual,
    so the residual stream is not directly scaled by gene conditioning
    (which could cause instability at early training).
    """
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
# Main model
# ============================================================

class CRISPRModel(nn.Module):
    """
    CRISPR dependency predictor v4.

    Parameters
    ----------
    n_clusters      : int    Number of gene clusters = cell feature dim (2388).
    gene_feat_size  : int    Gene feature dimension (26).
    cell_feat_size  : int    Cell feature dimension (must equal n_clusters).
    rank            : int    Shared embedding dimension (default 128).
    n_heads         : int    Multi-head bilinear heads (default 8).
    n_cell_layers   : int    FiLM residual blocks in cell encoder (default 4).
    dropout         : float  Dropout rate (default 0.2).
    """

    def __init__(
        self,
        n_clusters:     int   = 2388,
        gene_feat_size: int   = 26,
        cell_feat_size: int   = 2388,
        rank:           int   = 128,
        n_heads:        int   = 8,
        n_cell_layers:  int   = 4,
        dropout:        float = 0.2,
    ):
        super().__init__()
        assert rank % n_heads == 0, \
            f"rank ({rank}) must be divisible by n_heads ({n_heads})"
        assert cell_feat_size == n_clusters, \
            (f"cell_feat_size ({cell_feat_size}) must equal n_clusters ({n_clusters}) "
             f"— one RNA sum per cluster")

        self.rank      = rank
        self.n_heads   = n_heads
        self.head_dim  = rank // n_heads
        self.n_clusters = n_clusters

        # ── ① Gene side ───────────────────────────────────────────────────
        #
        # cluster_emb: WHO this gene is — discrete cluster identity.
        #   Each of 2388 clusters gets a learned dense vector trained
        #   end-to-end. Starts small, differentiates during training.
        #
        # gene_stat_encoder: WHERE in the cluster — the 26 continuous
        #   stats refine the cluster embedding with within-cluster detail.
        #
        # gene_gate: learned per-dimension blend. For well-populated
        #   clusters the embedding is reliable (gate→1). For sparse
        #   clusters the stats are safer (gate→0). Learned automatically.

        self.gene_cluster_emb = nn.Embedding(n_clusters, rank)

        self.gene_stat_encoder = nn.Sequential(
            nn.Linear(gene_feat_size, rank),
            nn.LayerNorm(rank),
            nn.GELU(),
            nn.Linear(rank, rank),
            nn.LayerNorm(rank),
        )

        self.gene_gate = nn.Sequential(
            nn.Linear(gene_feat_size, rank),
            nn.Sigmoid(),
        )

        # ── ② Cell side ───────────────────────────────────────────────────
        #
        # Input = [raw cell features | relative features] = 4776-dim.
        #
        # raw      = cell_feat                  absolute RNA sums
        # relative = cell_feat / own_expr       co-expression ratios
        #            own_expr = cell_feat[gene_cluster_id]
        #
        # The relative features are biologically meaningful:
        # "cluster 312 is 3× more expressed than this gene's cluster"
        # tells the model about co-expression structure without requiring
        # it to learn the reference point implicitly.
        #
        # The FiLM conditioning on gene embedding g means the cell encoder
        # learns DIFFERENT feature weightings for different gene cluster
        # types — a kinase gene reads the cell profile differently than
        # a transcription factor gene.

        cell_input_dim = cell_feat_size * 2   # 4776

        self.cell_proj = nn.Sequential(
            nn.Linear(cell_input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cell_align = nn.Sequential(
            nn.Linear(1024, rank),
            nn.LayerNorm(rank),
        )

        self.cell_film_blocks = nn.ModuleList([
            ResidualFiLMBlock(rank, rank, dropout)
            for _ in range(n_cell_layers)
        ])

        # ── ③ Own-cluster expression pathway ──────────────────────────────
        #
        # cell_feat[gene_cluster_id] is extracted explicitly so the model
        # never needs to discover which of 2388 dimensions is "home".
        # It has its own encoder pathway with dedicated gradient flow.

        self.own_expr_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, rank),
            nn.LayerNorm(rank),
        )

        self.cell_fusion = nn.Sequential(
            nn.Linear(rank * 2, rank),
            nn.LayerNorm(rank),
            nn.GELU(),
        )

        # ── ④ Multi-head bilinear interaction ─────────────────────────────
        #
        # score = (Ug) ⊙ (Vc) summed over head_dim
        # n_heads independent interaction axes match the known low-rank
        # (~10-30) structure of CRISPR dependency matrices.
        # Small init: interaction starts near zero, grows only as needed.

        self.bilinear_u = nn.Linear(rank, rank, bias=False)
        self.bilinear_v = nn.Linear(rank, rank, bias=False)

        # ── ⑤ Explicit main effect terms ──────────────────────────────────
        #
        # Zero-initialised: at epoch 0 the model outputs pure interaction
        # predictions. Main effects grow only as training demands them.
        # This is the opposite of the previous architecture which collapsed
        # to gene main effects immediately due to detach() bugs.

        self.gene_bias = nn.Linear(rank, 1)
        self.cell_bias = nn.Linear(rank, 1)

        # ── ⑥ Interaction head ────────────────────────────────────────────
        #
        # Sees all three views simultaneously:
        #   g           — gene identity (cluster + stats)
        #   c           — gene-conditioned cell context + own expression
        #   interaction — multi-head bilinear gene×cell signal
        #
        # Because main effects are absorbed by gene_bias + cell_bias,
        # this head is structurally encouraged to model only joint signal.

        self.head = nn.Sequential(
            nn.Linear(rank * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def forward(
        self,
        cell_feat:       torch.Tensor,   # [B, 2388]
        gene_feat:       torch.Tensor,   # [B, 26]
        gene_cluster_id: torch.Tensor,   # [B]  LongTensor, values 0..2387
    ) -> torch.Tensor:                   # [B, 1]

        B      = cell_feat.size(0)
        device = cell_feat.device

        # ── ① Gene embedding ──────────────────────────────────────────────
        cluster_emb = self.gene_cluster_emb(gene_cluster_id)  # [B, rank]
        stat_emb    = self.gene_stat_encoder(gene_feat)        # [B, rank]
        gate        = self.gene_gate(gene_feat)                # [B, rank]
        g = gate * cluster_emb + (1 - gate) * stat_emb        # [B, rank]

        # ── ② Own-cluster expression (privileged explicit feature) ────────
        # Direct index lookup: no gradient needed to discover this.
        own_expr = cell_feat[
            torch.arange(B, device=device), gene_cluster_id
        ].unsqueeze(1)                                         # [B, 1]

        # ── ③ Cell input: raw + relative ──────────────────────────────────
        # relative_feat[i, j] = "how expressed is cluster j relative
        #                         to this gene's own cluster"
        relative_feat = cell_feat / (own_expr.clamp(min=1e-8))  # [B, 2388]
        cell_input    = torch.cat([cell_feat, relative_feat], dim=-1)  # [B, 4776]

        # ── ④ FiLM-conditioned cell encoder ───────────────────────────────
        # Gene cluster g steers which cell dimensions matter at every layer.
        x = self.cell_proj(cell_input)                         # [B, 1024]
        x = self.cell_align(x)                                 # [B, rank]
        for block in self.cell_film_blocks:
            x = block(x, g)
        ctx_emb = x                                            # [B, rank]

        # ── ⑤ Own-cluster dedicated pathway ───────────────────────────────
        own_emb = self.own_expr_encoder(own_expr)              # [B, rank]

        # ── ⑥ Cell fusion ─────────────────────────────────────────────────
        c = self.cell_fusion(
            torch.cat([ctx_emb, own_emb], dim=-1)
        )                                                      # [B, rank]

        # ── ⑦ Multi-head bilinear interaction ─────────────────────────────
        ug = self.bilinear_u(g).view(B, self.n_heads, self.head_dim)
        vc = self.bilinear_v(c).view(B, self.n_heads, self.head_dim)
        interaction = (ug * vc).view(B, self.rank)             # [B, rank]

        # ── ⑧ Final prediction ────────────────────────────────────────────
        x   = torch.cat([g, c, interaction], dim=-1)           # [B, rank*3]
        out = self.gene_bias(g) + self.cell_bias(c) + self.head(x)
        return out                                             # [B, 1]

    # ------------------------------------------------------------------
    def _init_weights(self):
        # Zero-init main effect heads — forces interaction learning first.
        # These grow only as training demands them.
        for m in [self.gene_bias, self.cell_bias]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Small bilinear init — interaction starts near zero
        nn.init.normal_(self.bilinear_u.weight, 0, 0.01)
        nn.init.normal_(self.bilinear_v.weight, 0, 0.01)

        # Small cluster embedding init — clusters differentiate during training
        nn.init.normal_(self.gene_cluster_emb.weight, 0, 0.01)

        excluded = {
            id(self.gene_bias.weight),
            id(self.cell_bias.weight),
            id(self.bilinear_u.weight),
            id(self.bilinear_v.weight),
            id(self.gene_cluster_emb.weight),
        }
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if id(m.weight) in excluded:
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
    Print diagnostic report on model component contributions.

    Monitors
    --------
    Gate value         : 0 = relying on stats, 1 = relying on cluster emb
    Interaction/main   : ratio of head output to main effect output
                         target >1.0 — model is learning interactions
    Own/context ratio  : own-cluster pathway vs FiLM context pathway
    FiLM gamma per block: near-zero = FiLM not active (increase LR/capacity)

    Call every N epochs during training to catch collapse early.
    """
    model.eval()

    gate_vals        = []
    bilinear_mags    = []
    main_effect_mags = []
    own_vs_ctx       = []
    film_gammas      = []

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

            own_emb = model.own_expr_encoder(own_expr)
            own_vs_ctx.append(
                own_emb.abs().mean().item() /
                (ctx_emb.abs().mean().item() + 1e-8)
            )

            c = model.cell_fusion(torch.cat([ctx_emb, own_emb], dim=-1))

            ug          = model.bilinear_u(g).view(B, model.n_heads, model.head_dim)
            vc          = model.bilinear_v(c).view(B, model.n_heads, model.head_dim)
            interaction = (ug * vc).view(B, model.rank)

            head_out = model.head(torch.cat([g, c, interaction], dim=-1))
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
    ratio = avg_bilinear / (avg_main + 1e-8)

    print(f"\n── Model Diagnostic ─────────────────────────────────────")
    print(f"  Gate mean (0=stats 1=emb)   : {avg_gate:.3f}")
    print(f"  Interaction head magnitude  : {avg_bilinear:.4f}")
    print(f"  Main effect magnitude       : {avg_main:.4f}")
    print(f"  Interaction / Main ratio    : {ratio:.3f}  (target >1.0)")
    print(f"  Own-cluster / context ratio : {avg_own_ctx:.3f}")
    print(f"  FiLM |gamma| per block      : {[f'{g:.3f}' for g in avg_gammas]}")
    print(f"    ↳ near-zero gammas = FiLM inactive → raise LR or capacity")
    print(f"─────────────────────────────────────────────────────────\n")