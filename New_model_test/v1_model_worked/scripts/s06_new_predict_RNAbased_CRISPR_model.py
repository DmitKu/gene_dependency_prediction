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
    gene_pearson_{SPLIT}.csv  — per-gene Pearson across cell lines
    metrics_{SPLIT}.txt       — summary metrics
"""

import sys
import csv
import time
from pathlib import Path
from collections import defaultdict

import joblib
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

import numpy as np
import pandas as pd

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
    hidden_dim    = 128,
    gene_hidden   = 64,
    n_attn_slots  = 64,
    n_attn_heads  = 4,
    bypass_rank   = 8,
    compress_dim  = 1024,
    dropout       = 0.2,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    # GeneDataset.__getitem__ returns:
    #   gene_feat, cell_feat, target, cl_idx, gene_id, model_id, sample_idx
    # model.forward() returns: (pred, bypass_reg)  ← must unpack both
    all_pred, all_target, all_cl_idx, all_sample_idx = [], [], [], []
    all_gene_ids_list, all_model_ids_list = [], []
    t0 = time.time()

    # Parse device type string for autocast
    device_type = torch.device(DEVICE).type

    with torch.no_grad():
        for gene_feat, cell_feat, target, cl_idx, gene_id, model_id, sample_idx in loader:
            gene_feat = gene_feat.to(DEVICE, non_blocking=True)
            cell_feat = cell_feat.to(DEVICE, non_blocking=True)

            with autocast(device_type=device_type):
                # FIX: model returns (pred, bypass_reg) — unpack both
                pred, _ = model(cell_feat, gene_feat, ablate_bypass=ABLATE_BYPASS)

            all_pred.append(pred.cpu())
            all_target.append(target.cpu())
            all_cl_idx.append(cl_idx.cpu())
            all_sample_idx.append(sample_idx.cpu())
            all_gene_ids_list.extend(gene_id)
            all_model_ids_list.extend(model_id)

    print(f"Inference done in {time.time() - t0:.1f}s")

    all_pred       = torch.cat(all_pred).squeeze()
    all_target     = torch.cat(all_target).squeeze()
    all_cl_idx     = torch.cat(all_cl_idx)
    all_sample_idx = torch.cat(all_sample_idx)

    # ── Inverse-transform both pred and target to Chronos space ──────────────
    all_pred_np   = qt.inverse_transform(all_pred.numpy().reshape(-1, 1)).squeeze()
    all_target_np = qt.inverse_transform(all_target.numpy().reshape(-1, 1)).squeeze()

    # ── Summary metrics from evaluate() ──────────────────────────────────────
    # evaluate() returns 9 values:
    #   val_loss, val_mse, val_pearson,
    #   mae, rmse,
    #   pearson_full,
    #   pearson_full_demeaned,
    #   cl_mean, cl_std
    (_, _, _,
     mae, rmse,
     pearson_full,
     pearson_full_demeaned,
     cl_mean, cl_std) = evaluate(
        model, loader, DEVICE,
        qt=qt, alpha=0.1,
        ablate_bypass=ABLATE_BYPASS,
    )

    # ── n_cl: count cell lines with ≥10 samples, consistent with evaluate() ──
    cl_groups = defaultdict(list)
    for i, mid in enumerate(all_model_ids_list):
        cl_groups[mid].append(i)
    n_cl = sum(1 for indices in cl_groups.values() if len(indices) >= 10)

    # ── Per-gene Pearson across cell lines ────────────────────────────────────
    df_gene = pd.DataFrame({
        "gene_id": all_gene_ids_list,
        "pred":    all_pred_np,
        "true":    all_target_np,
    })

    gene_results = []
    for gene, g in df_gene.groupby("gene_id", sort=False):
        if len(g) < 10:
            continue
        r = np.corrcoef(g["pred"].values, g["true"].values)[0, 1]
        if np.isnan(r):
            continue
        gene_results.append({
            "gene_id":      gene,
            "n_cell_lines": len(g),
            "pearson":      r,
        })

    gene_results  = pd.DataFrame(gene_results)
    pearson_pg    = gene_results["pearson"].mean()
    pearson_pg_sd = gene_results["pearson"].std(ddof=1)
    n_genes       = len(gene_results)

    gene_results.to_csv(SAVE_PATH / f"gene_pearson_{SPLIT}.csv", index=False)

    # ── Print metrics ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  MAE                     : {mae:.4f}  (Chronos space)")
    print(f"  RMSE                    : {rmse:.4f}  (Chronos space)")
    print(f"  Pearson (global)        : {pearson_full:.4f}  (Chronos space)")
    print(f"  Pearson (demeaned)      : {pearson_full_demeaned:.4f}  (Chronos space)")
    print(f"  Pearson (per-CL)        : {cl_mean:.4f} ± {cl_std:.4f}  (n={n_cl})")
    print(f"  Pearson (per-Gene)      : {pearson_pg:.4f} ± {pearson_pg_sd:.4f}  (n={n_genes})")
    print(f"{'=' * 55}\n")

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        f.write(
            f"Model                : {MODEL_PATH.name}\n"
            f"Split                : {SPLIT}\n"
            f"Bypass               : {'ablated' if ABLATE_BYPASS else 'active'}\n"
            f"N samples            : {len(all_pred):,}\n"
            f"N cell lines (>=10)  : {n_cl}\n"
            f"N genes (>=10)       : {n_genes}\n"
            f"MAE                  : {mae:.6f}  (Chronos space)\n"
            f"RMSE                 : {rmse:.6f}  (Chronos space)\n"
            f"Pearson global       : {pearson_full:.6f}  (Chronos space)\n"
            f"Pearson demeaned     : {pearson_full_demeaned:.6f}  (Chronos space)\n"
            f"Pearson per-CL       : {cl_mean:.6f} +/- {cl_std:.6f}\n"
            f"Pearson per-Gene     : {pearson_pg:.6f} +/- {pearson_pg_sd:.6f}\n"
        )
    print(f"Metrics saved      -> {OUT_METRICS}")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    all_sample_np = all_sample_idx.numpy()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "cell_line_model_id", "gene_id",
            "crispr_actual", "crispr_predicted", "residual",
        ])
        for i in range(len(all_pred_np)):
            pred_val = float(all_pred_np[i])
            true_val = float(all_target_np[i])
            writer.writerow([
                int(all_sample_np[i]),
                all_model_ids_list[i],
                all_gene_ids_list[i],
                f"{true_val:.6f}",
                f"{pred_val:.6f}",
                f"{pred_val - true_val:.6f}",
            ])

    print(f"Predictions saved  -> {OUT_CSV}  ({len(all_pred_np):,} rows)")


if __name__ == "__main__":
    main()