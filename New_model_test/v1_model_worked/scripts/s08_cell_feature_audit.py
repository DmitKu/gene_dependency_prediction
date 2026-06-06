# -*- coding: utf-8 -*-
"""
s07_cell_feature_audit.py
=======================================
Run script — extracts gradient × input attribution scores for each of
the 2388 RNA cluster features for every (cell_line × gene) sample.

Runs over all three splits (train / val / test) and writes each to its
own subfolder to avoid overwrites.

Functions live in src/utils_feature_importance.py.

Outputs written to outputs/feature_importance/{split}/:
    importance_matrix.npy            — float32 [N, 2388]  memory-mapped
                                       signed importance per sample
    sample_meta.csv                  — row → (ds_idx, cl_idx, target, pred)
    cluster_importance_per_gene.npy  — float32 [n_genes, 2388]
    cluster_importance_per_gene_ids.npy
    cluster_importance_global.npy    — float32 [2388]

Usage
-----
    python scripts/s07_cell_feature_audit.py
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.utils_RNAbased_crispr_model import GeneDataset, CRISPRSensitivityModelV3
from src.utils_feature_importance import (
    attribute_cell_inputs,
    aggregate_per_gene,
    aggregate_global,
)

# ============================================================
# CONFIG
# ============================================================

H5_PATH      = _root / "outputs" / "H5_model_data" / "model_H5_data.h5"
WEIGHTS_PATH = _root / "outputs" / "model_training" / "crispr_best_pearson_model.pt"
BASE_SAVE    = _root / "outputs" / "feature_importance"   # split subfolder added below

SPLITS     = ["train", "val", "test"]
BATCH_SIZE = 128
CHUNK_SIZE = 50_000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_KWARGS = dict(
    hidden_dim   = 128,
    n_attn_slots = 64,
    n_attn_heads = 4,
    bypass_rank  = 32,
    compress_dim = 512,
    dropout      = 0.2,
)


# ============================================================
# Per-split extraction
# ============================================================

def run_split(model, split: str):
    save_path = BASE_SAVE / split          # e.g. outputs/feature_importance/val/
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Split: {split.upper()}")
    print(f"{'='*60}")
    dataset     = GeneDataset(H5_PATH, split=split)
    n_cell_feat = dataset.cl_features.shape[1]
    n_samples   = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size         = BATCH_SIZE,
        shuffle            = False,
        num_workers        = 4,
        pin_memory         = (DEVICE == "cuda"),
        persistent_workers = False,
    )

    print(f"  → {n_samples:,} samples | saving to {save_path}")

    # ── Memory-mapped output ──────────────────────────────────────────────────
    imp_path = save_path / "importance_matrix.npy"
    imp_mm   = np.memmap(imp_path, dtype="float32", mode="w+",
                         shape=(n_samples, n_cell_feat))

    # ── Meta CSV ──────────────────────────────────────────────────────────────
    meta_path   = save_path / "sample_meta.csv"
    meta_file   = open(meta_path, "w", newline="")
    meta_writer = csv.writer(meta_file)
    meta_writer.writerow(["sample_idx", "ds_idx", "cl_idx",
                           "crispr_target", "prediction"])

    # ── Attribution loop ──────────────────────────────────────────────────────
    print("\nExtracting attributions …")
    t0         = time.time()
    sample_ptr = 0

    for batch_idx, (gene_feat, cell_feat, target, cl_idx, ds_idx) in enumerate(loader):
        B = gene_feat.size(0)

        gene_feat = gene_feat.to(DEVICE, non_blocking=True)
        cell_feat = cell_feat.to(DEVICE, non_blocking=True)

        imp, preds = attribute_cell_inputs(model, cell_feat, gene_feat)

        imp_mm[sample_ptr : sample_ptr + B] = imp.numpy()

        for i in range(B):
            meta_writer.writerow([
                sample_ptr + i,
                ds_idx[i].item(),
                cl_idx[i].item(),
                round(target[i].item(), 6),
                round(preds[i].item(),  6),
            ])

        sample_ptr += B

        if (batch_idx + 1) % 20 == 0 or sample_ptr == n_samples:
            elapsed  = time.time() - t0
            progress = sample_ptr / n_samples
            eta      = elapsed / progress * (1 - progress) if progress > 0 else 0
            mem_gb   = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            print(
                f"  {sample_ptr:>8,}/{n_samples:,}  |  "
                f"{elapsed:.0f}s elapsed  |  ETA {eta:.0f}s  |  "
                f"GPU {mem_gb:.2f} GB"
            )

    meta_file.close()
    imp_mm.flush()
    print(f"Attribution complete in {time.time()-t0:.1f}s")

    # ── Aggregations ──────────────────────────────────────────────────────────
    aggregate_per_gene(imp_path, meta_path, n_samples, n_cell_feat, CHUNK_SIZE, save_path)
    global_imp = aggregate_global(imp_path, n_samples, n_cell_feat, CHUNK_SIZE, save_path)

    # ── Sanity report ─────────────────────────────────────────────────────────
    top10 = global_imp.argsort()[::-1][:10]
    print(f"\nTop 10 RNA clusters [{split}] (global mean |attribution|):")
    for rank, idx in enumerate(top10):
        print(f"  #{rank+1:>2}  cluster {idx:>4}  importance = {global_imp[idx]:.6f}")


# ============================================================
# Main
# ============================================================

def main():
    BASE_SAVE.mkdir(parents=True, exist_ok=True)

    # Load model once — reused across all splits
    print(f"Loading model weights from {WEIGHTS_PATH} …")

    # Peek at dataset to get feature dims without loading all splits twice
    _ds = GeneDataset(H5_PATH, split="val")
    n_cell_feat = _ds.cl_features.shape[1]
    n_gene_feat = _ds.gene_feat.shape[1]
    del _ds

    model = CRISPRSensitivityModelV3(
        cell_features_size = n_cell_feat,
        gene_features_size = n_gene_feat,
        **MODEL_KWARGS,
    ).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"  → Model loaded on {DEVICE}")

    total_t0 = time.time()
    for split in SPLITS:
        run_split(model, split)

    total_min = (time.time() - total_t0) / 60
    print(f"\n{'='*60}")
    print(f"All splits complete in {total_min:.1f} min")
    print(f"Outputs in: {BASE_SAVE}/{{train,val,test}}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()