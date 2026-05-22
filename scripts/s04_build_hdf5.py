"""
run_build_hdf5.py
=================
Entry-point for building model_H5_data.h5.

Orchestrates the full pipeline by calling functions from hdf5_builder.py.
This script contains *no* logic of its own — only configuration and the
call sequence.

Usage
-----
    python scripts/s04_build_hdf5.py

Prerequisites
-------------
    s03_feature_engineering.py must have been run first so that
    feature data are available and both CSVs contain a 'split'
    column and the CRISPR values are already
    QuantileTransformed.
"""

import numpy as np
from pathlib import Path
import sys


# Make src/ importable — works from CLI and Spyder
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:                    # __file__ undefined in Spyder
    _root = Path.cwd()               # assumes Spyder cwd = project root
sys.path.insert(0, str(_root / "src"))

from utils_hdf5_builder import (
    load_csvs,
    identify_feature_columns,
    validate_model_ids,
    extract_split_indices,
    compute_normalization_stats,
    normalize_and_impute,
    prepare_crispr_target,
    write_hdf5,
    verify_hdf5,
    print_summary,
)


# ── Configuration ─────────────────────────────────────────────────────────────


# _root = Path(
#     r"C:\Users\dkuch\Documents\Blog_ideas_data\Computational"
#     r"\MOA_Prediction_based_on_CETSA\20251122_Model_development"
#     r"\GitHub_GeneDependancy_prediction"
# )

_root = Path(__file__).resolve().parents[1]
SAVE_DIR = _root/"outputs/H5_model_data"
CELL_LINE_CSV   = "outputs/RNA_fetures/Cell_line_based_features.csv"
GENE_CSV        = "outputs/RNA_fetures/RNA_based_features_CRISPR.csv"
OUTPUT_H5       = "outputs/H5_model_data/model_H5_data.h5"
GENE_CHUNK_ROWS = 20_000


# ── Pipeline ──────────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:

    # 1. Load data
    df_cl, df_gene = load_csvs(_root, CELL_LINE_CSV, GENE_CSV)

    # 2. Identify feature columns
    cl_feat_cols, gene_feat_cols = identify_feature_columns(df_cl, df_gene)

    # 3. Validate ModelID consistency
    validate_model_ids(df_cl, df_gene)

    # 4. Read pre-computed split labels
    train_idx, val_idx, test_idx, train_cls, val_cls, test_cls = (
        extract_split_indices(df_gene)
    )
    train_mask = df_gene["split"].values == "train"

    # 5. Compute normalization statistics (training rows only)
    cl_mean, cl_std, gene_mean, gene_std = compute_normalization_stats(
        df_cl, cl_feat_cols, train_cls,
        df_gene, gene_feat_cols, train_mask,
    )

    # 6. Normalize and impute
    cl_feats, gene_feats = normalize_and_impute(
        df_cl, cl_feat_cols, cl_mean, cl_std,
        df_gene, gene_feat_cols, gene_mean, gene_std,
    )

    # 7. Prepare CRISPR target (already QuantileTransformed)
    crispr_vals, train_idx, val_idx, test_idx = prepare_crispr_target(
        df_gene, train_idx, val_idx, test_idx
    )

    # 8. Write HDF5
    write_hdf5(
        output_path    = _root / OUTPUT_H5,
        cl_feats       = cl_feats,
        gene_feats     = gene_feats,
        crispr_vals    = crispr_vals,
        df_cl          = df_cl,
        df_gene        = df_gene,
        cl_feat_cols   = cl_feat_cols,
        gene_feat_cols = gene_feat_cols,
        cl_mean        = cl_mean,
        cl_std         = cl_std,
        gene_mean      = gene_mean,
        gene_std       = gene_std,
        train_idx      = train_idx,
        val_idx        = val_idx,
        test_idx       = test_idx,
        train_cls      = train_cls,
        val_cls        = val_cls,
        test_cls       = test_cls,
        gene_chunk_rows = GENE_CHUNK_ROWS,
    )

    # 9. Verify and report
    verify_hdf5(_root / OUTPUT_H5)
    print_summary(
        cl_feats    = cl_feats,
        gene_feats  = gene_feats,
        crispr_vals = crispr_vals,
        train_idx   = train_idx,
        val_idx     = val_idx,
        test_idx    = test_idx,
        train_cls   = train_cls,
        val_cls     = val_cls,
        test_cls    = test_cls,
    )


if __name__ == "__main__":
    main()