# -*- coding: utf-8 -*-
"""
utils_feature_importance.py
============================
Reusable attribution utilities for CRISPRSensitivityModelV3.

Functions
---------
attribute_cell_inputs       — gradient × input attribution per batch
aggregate_per_gene          — chunked mean |attribution| per target gene
aggregate_global            — chunked mean |attribution| across all samples
"""

import csv
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Attribution
# ============================================================

def attribute_cell_inputs(
    model:         torch.nn.Module,
    cell_features: torch.Tensor,    # [B, F]  on device
    gene_features: torch.Tensor,    # [B, G]  on device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient × input attribution scoped to cell_features only.

    gene_features has no grad (we don't need its gradient), which
    reduces activation memory vs enabling grads on the full input.

    Parameters
    ----------
    model         : trained CRISPRSensitivityModelV3 in eval mode
    cell_features : [B, F]  float32 tensor on device
    gene_features : [B, G]  float32 tensor on device

    Returns
    -------
    importance  : cpu float32 [B, F]
                  signed attribution (grad × input)
                  positive → cluster pushes prediction UP  (more essential)
                  negative → cluster pushes prediction DOWN (less essential)
    predictions : cpu float32 [B]
    """
    cell_feat_grad = cell_features.detach().requires_grad_(True)
    gene_feat      = gene_features.detach()   # grad not needed for gene path

    pred = model(cell_feat_grad, gene_feat)   # [B, 1]
    pred.sum().backward()

    importance = (cell_feat_grad.grad * cell_feat_grad).detach().cpu()
    return importance, pred.detach().cpu().squeeze()


# ============================================================
# Per-gene aggregation (chunked, memmap-safe)
# ============================================================

def aggregate_per_gene(
    imp_path:   Path,
    meta_path:  Path,
    n_samples:  int,
    n_features: int,
    chunk_size: int,
    save_path:  Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean |attribution| per gene without loading the full matrix.

    Reads importance_matrix.npy through its memmap in chunk_size row
    chunks so peak RAM stays bounded regardless of dataset size.

    Parameters
    ----------
    imp_path   : path to importance_matrix.npy (memmap, float32 [N, F])
    meta_path  : path to sample_meta.csv (must contain column 'ds_idx')
    n_samples  : total number of rows N in the matrix
    n_features : number of cell features F (e.g. 2388)
    chunk_size : rows processed per iteration
    save_path  : directory where output .npy files are written

    Returns
    -------
    gene_imp     : float32 [n_unique_genes, F]  mean |attribution| per gene
    unique_genes : int64  [n_unique_genes]       gene ds_idx values (row order)

    Saves
    -----
    cluster_importance_per_gene.npy
    cluster_importance_per_gene_ids.npy
    """
    import pandas as pd

    print("\nAggregating per-gene importances (chunked) …")

    meta     = pd.read_csv(meta_path)
    gene_ids = meta["ds_idx"].values                   # [N]

    imp_mm = np.memmap(imp_path, dtype="float32", mode="r",
                       shape=(n_samples, n_features))

    unique_genes   = np.unique(gene_ids)
    n_genes        = len(unique_genes)
    gene_sum       = np.zeros((n_genes, n_features), dtype=np.float64)
    gene_count     = np.zeros(n_genes, dtype=np.int64)
    gene_id_to_pos = {gid: i for i, gid in enumerate(unique_genes)}

    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end  = min(chunk_start + chunk_size, n_samples)
        chunk      = np.abs(imp_mm[chunk_start:chunk_end])
        chunk_gids = gene_ids[chunk_start:chunk_end]

        for local_i, gid in enumerate(chunk_gids):
            pos = gene_id_to_pos[gid]
            gene_sum[pos]   += chunk[local_i]
            gene_count[pos] += 1

        print(f"  chunked {chunk_end:>8,}/{n_samples:,}", end="\r")

    print()

    gene_imp = (gene_sum / np.maximum(gene_count[:, None], 1)).astype(np.float32)

    np.save(save_path / "cluster_importance_per_gene.npy",     gene_imp)
    np.save(save_path / "cluster_importance_per_gene_ids.npy", unique_genes)
    print(f"Saved: cluster_importance_per_gene.npy      shape={gene_imp.shape}")

    return gene_imp, unique_genes


# ============================================================
# Global aggregation (chunked)
# ============================================================

def aggregate_global(
    imp_path:   Path,
    n_samples:  int,
    n_features: int,
    chunk_size: int,
    save_path:  Path,
) -> np.ndarray:
    """
    Compute mean |attribution| across all samples, chunked through memmap.

    Parameters
    ----------
    imp_path   : path to importance_matrix.npy (memmap, float32 [N, F])
    n_samples  : total number of rows N
    n_features : number of cell features F
    chunk_size : rows processed per iteration
    save_path  : directory where output .npy file is written

    Returns
    -------
    global_imp : float32 [F]  mean |attribution| per cluster

    Saves
    -----
    cluster_importance_global.npy
    """
    print("\nAggregating global importances (chunked) …")

    imp_mm     = np.memmap(imp_path, dtype="float32", mode="r",
                           shape=(n_samples, n_features))
    global_sum = np.zeros(n_features, dtype=np.float64)

    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end   = min(chunk_start + chunk_size, n_samples)
        global_sum += np.abs(imp_mm[chunk_start:chunk_end]).sum(axis=0)
        print(f"  chunked {chunk_end:>8,}/{n_samples:,}", end="\r")

    print()
    global_imp = (global_sum / n_samples).astype(np.float32)

    out = save_path / "cluster_importance_global.npy"
    np.save(out, global_imp)
    print(f"Saved: cluster_importance_global.npy         shape={global_imp.shape}")

    return global_imp