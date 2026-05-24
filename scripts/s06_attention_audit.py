# -*- coding: utf-8 -*-
"""
s06_attention_audit.py
======================
Extracts and saves cross-attention weights from CRISPRSensitivityModelV3
for every gene / cell-line sample across all dataset splits.

Output
------
outputs/attention_audit_full.parquet
    Columns:
        Slot_0 … Slot_63  — attention weight over each cell-token slot
        Gene_ID           — gene model-ID string
        Cell_ID           — cell-line model-ID string
        Cell_Index        — integer index into cell_lines/features
        Split             — "train" | "val" | "test"

Usage
-----
    python scripts/s06_attention_audit.py
"""

import sys
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

import torch
from src.utils_attention_audit import extract_attention_weights

# ── Config (must match s05) ───────────────────────────────────────────────────

H5_PATH      = _root / "outputs" / "H5_model_data" / "model_H5_data.h5"
WEIGHTS_PATH = _root / "outputs" / "model_training" / "crispr_best_pearson_model.pt"
SAVE_PATH  = _root / "outputs" / "attention_audit" 
OUTPUT_PATH  = _root / "outputs" / "attention_audit" / "attention_audit_full.parquet"

MODEL_KWARGS = dict(
    hidden_dim   = 128,
    n_attn_slots = 64,
    n_attn_heads = 4,
    bypass_rank  = 32,
    compress_dim = 512,
    dropout      = 0.2,
)

BATCH_SIZE = 4096
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS     = ("train", "val", "test")


# ── Main ──────────────────────────────────────────────────────────────────────
SAVE_PATH.mkdir(parents=True, exist_ok=True)

def main():
    for p in (H5_PATH, WEIGHTS_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    print(f"Device       : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"Output path  : {OUTPUT_PATH}")

    total = extract_attention_weights(
        h5_path      = H5_PATH,
        weights_path = WEIGHTS_PATH,
        output_path  = OUTPUT_PATH,
        model_kwargs = MODEL_KWARGS,
        batch_size   = BATCH_SIZE,
        device       = DEVICE,
        splits       = SPLITS,
    )

    print(f"\nExtraction complete. {total:,} total rows → {OUTPUT_PATH}")

    # Quick sanity check
    df = pd.read_parquet(OUTPUT_PATH, columns=["Gene_ID", "Cell_ID", "Split", "Slot_0"])
    print(f"\nRow counts per split:\n{df['Split'].value_counts().to_string()}")
    print(f"\nSample rows:\n{df.head().to_string()}")


if __name__ == "__main__":
    main()


# ── Analysis stub (run separately after extraction) ───────────────────────────
#
# import pandas as pd
# import numpy as np
#
# df = pd.read_parquet("outputs/attention_audit_full.parquet")
# slot_cols = [c for c in df.columns if c.startswith("Slot_")]
#
# # Which slots does the model rely on most globally?
# mean_per_slot = df[slot_cols].mean().sort_values(ascending=False)
# print("Top-10 most-attended slots:\n", mean_per_slot.head(10))
#
# # Mean attention profile per gene
# gene_attn = df.groupby("Gene_ID")[slot_cols].mean()
#
# # Mean attention profile per cell line
# cl_attn = df.groupby("Cell_ID")[slot_cols].mean()
#
# # Attention entropy per sample (low = focused, high = diffuse)
# attn_mat = df[slot_cols].values
# entropy = -(attn_mat * np.log(attn_mat + 1e-12)).sum(axis=1)
# df["attn_entropy"] = entropy
# print(df.groupby("Split")["attn_entropy"].describe())