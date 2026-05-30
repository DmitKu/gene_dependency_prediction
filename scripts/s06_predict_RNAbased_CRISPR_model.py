# -*- coding: utf-8 -*-
"""
s06_predict_RNAbased_CRISPR_model.py
=====================================
Inference / evaluation entry-point for the CRISPR Sensitivity Model.

Usage
-----
    python scripts/s06_predict_RNAbased_CRISPR_model.py

All configuration is defined in the CONFIG section below.
Outputs written to SAVE_PATH:
    predictions_{SPLIT}.csv   — per-sample predictions + residuals
    metrics_{SPLIT}.txt       — summary metrics
"""

import sys
import csv
import time
import h5py
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

# ── Local imports ─────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.utils_RNAbased_crispr_model import (
    GeneDataset,
    CRISPRSensitivityModelV3,
    evaluate,
)


# ============================================================
# CONFIG
# ============================================================

H5_PATH          = _root / "outputs" / "H5_model_data" / "model_H5_data.h5"
TRANSFORMER_PATH = _root / "outputs" / "RNA_fetures" / "chronos_quantile_transformer.pkl"
MODEL_PATH       = _root / "outputs" / "model_training" / "crispr_best_pearson_model.pt"
SAVE_PATH        = _root / "outputs" / "model_predictions"

SPLIT         = "test"   # "train" | "val" | "test"
ABLATE_BYPASS = False    # True → zero out linear bypass at inference
BATCH_SIZE    = 20_000

# Must match the architecture used during training
MODEL_KWARGS = dict(
    hidden_dim   = 128,
    n_attn_slots = 64,
    n_attn_heads = 4,
    bypass_rank  = 32,
    compress_dim = 512,
    dropout      = 0.2,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Helpers
# ============================================================

def pearson(p: torch.Tensor, t: torch.Tensor) -> float:
    pm = p - p.mean()
    pt = t - t.mean()
    return ((pm * pt).sum() / (pm.norm() * pt.norm() + 1e-8)).item()


# ============================================================
# Main
# ============================================================

def main():
    # ── Validate paths ────────────────────────────────────────────────────────
    for path, label in [
        (H5_PATH,          "HDF5 data file"),
        (MODEL_PATH,       "Model weights"),
        (TRANSFORMER_PATH, "QuantileTransformer"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    OUT_CSV     = SAVE_PATH / f"predictions_{SPLIT}.csv"
    OUT_METRICS = SAVE_PATH / f"metrics_{SPLIT}.txt"

    print(f"Device  : {DEVICE}")
    print(f"Model   : {MODEL_PATH.name}")
    print(f"Split   : {SPLIT}")
    print(f"Bypass  : {'ABLATED' if ABLATE_BYPASS else 'active'}\n")

    # ── QuantileTransformer (for Chronos-space metrics) ───────────────────────
    qt = joblib.load(TRANSFORMER_PATH)
    print(f"QuantileTransformer loaded from {TRANSFORMER_PATH.name}")
    print("  → Evaluation metrics will be reported in Chronos space\n")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    ds = GeneDataset(H5_PATH, split=SPLIT)

    # Load gene IDs directly from HDF5 using split indices (same pattern as
    # utils_attention_audit.py) — GeneDataset does not expose gene_names
    with h5py.File(H5_PATH, "r") as f:
        all_gene_ids  = f["genes/gene_id"][:]
        split_indices = f[f"index/splits/{SPLIT}"][:]
        gene_ids      = [all_gene_ids[i].decode() for i in split_indices]

    loader = DataLoader(
        ds,
        batch_size         = BATCH_SIZE,
        shuffle            = False,
        num_workers        = 4,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRISPRSensitivityModelV3(
        cell_features_size = ds.cl_features.shape[1],
        gene_features_size = ds.gene_feat.shape[1],
        **MODEL_KWARGS,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    print(f"Trainable params : {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    # GeneDataset.__getitem__ returns: gene_feat, cell_feat, target, cl_idx, sample_idx
    all_pred, all_target, all_cl_idx, all_sample_idx = [], [], [], []
    t0 = time.time()

    with torch.no_grad():
        for gene_feat, cell_feat, target, cl_idx, sample_idx in loader:
            gene_feat = gene_feat.to(DEVICE, non_blocking=True)
            cell_feat = cell_feat.to(DEVICE, non_blocking=True)
            with autocast(DEVICE):
                pred = model(cell_feat, gene_feat, ablate_bypass=ABLATE_BYPASS)
            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_cl_idx.append(cl_idx.cpu())
            all_sample_idx.append(sample_idx.cpu())

    print(f"Inference done in {time.time() - t0:.1f}s")

    all_pred       = torch.cat(all_pred).squeeze()
    all_target     = torch.cat(all_target).squeeze()
    all_cl_idx     = torch.cat(all_cl_idx)
    all_sample_idx = torch.cat(all_sample_idx)

    # ── Metrics (Chronos space) ───────────────────────────────────────────────
    mae, rmse, pearson_full, pearson_pcl, pearson_pcl_sd = evaluate(
        model, loader, DEVICE, qt=qt
    )

    # Per-cell-line Pearson count (raw space, for n_cl reporting)
    cl_pearsons_raw = []
    for cl_id in all_cl_idx.unique():
        mask = all_cl_idx == cl_id
        if mask.sum() < 10:
            continue
        cl_pearsons_raw.append(pearson(all_pred[mask], all_target[mask]))
    n_cl = len(cl_pearsons_raw)

    print(f"\n{'=' * 55}")
    print(f"  MAE                : {mae:.4f}  (Chronos space)")
    print(f"  RMSE               : {rmse:.4f}  (Chronos space)")
    print(f"  Pearson (global)   : {pearson_full:.4f}  (Chronos space)")
    print(f"  Pearson (per-CL)   : {pearson_pcl:.4f} ± {pearson_pcl_sd:.4f}  (n={n_cl} cell lines)")
    print(f"{'=' * 55}\n")

    with open(OUT_METRICS, "w") as f:
        f.write(
            f"Model          : {MODEL_PATH.name}\n"
            f"Split          : {SPLIT}\n"
            f"Bypass         : {'ablated' if ABLATE_BYPASS else 'active'}\n"
            f"N samples      : {len(all_pred):,}\n"
            f"N cell lines   : {n_cl}\n"
            f"MAE            : {mae:.6f}  (Chronos space)\n"
            f"RMSE           : {rmse:.6f}  (Chronos space)\n"
            f"Pearson global : {pearson_full:.6f}  (Chronos space)\n"
            f"Pearson per-CL : {pearson_pcl:.6f} ± {pearson_pcl_sd:.6f}  (Chronos space)\n"
        )
    print(f"Metrics saved → {OUT_METRICS}")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    all_pred_np   = all_pred.numpy()
    all_target_np = all_target.numpy()
    all_cl_np     = all_cl_idx.numpy()
    all_sample_np = all_sample_idx.numpy()

    # Build integer index → cell-line model-ID lookup
    cl_idx_to_model_id = {v: k for k, v in ds.cl_model_id_to_index.items()}

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "cell_line_model_id", "gene_id",
                         "crispr_actual", "crispr_predicted", "residual"])

        for i in range(len(all_pred_np)):
            pred_val = float(all_pred_np[i])
            true_val = float(all_target_np[i])
            writer.writerow([
                int(all_sample_np[i]),
                cl_idx_to_model_id[int(all_cl_np[i])],
                gene_ids[int(all_sample_np[i])],        # HDF5 positional index
                f"{true_val:.6f}",
                f"{pred_val:.6f}",
                f"{pred_val - true_val:.6f}",
            ])

    print(f"Predictions saved → {OUT_CSV}  ({len(all_pred_np):,} rows)")


if __name__ == "__main__":
    main()