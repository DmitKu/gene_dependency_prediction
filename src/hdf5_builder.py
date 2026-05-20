"""
hdf5_builder.py
===============
Library of functions for building model_data_v3.h5.

Called by run_build_hdf5.py — do not run directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h5py
from pathlib import Path


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csvs(
    base: Path,
    cell_line_csv: str,
    gene_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and lightly validate the two input CSVs.

    Returns
    -------
    df_cl : pd.DataFrame
        Cell-line features with 'ModelID' and 'split' columns.
    df_gene : pd.DataFrame
        Gene features with 'ModelID', 'gene', 'CRISPR', and 'split' columns,
        sorted by gene name for deterministic row order.
    """
    print("Loading CSVs …")
    df_cl   = pd.read_csv(base / cell_line_csv)
    df_gene = pd.read_csv(base / gene_csv)

    # Deterministic row order
    df_gene = df_gene.sort_values(by="gene").reset_index(drop=True)

    for label, df in [("gene CSV", df_gene), ("cell-line CSV", df_cl)]:
        if "split" not in df.columns:
            raise AssertionError(
                f"'split' column missing from {label}.\n"
                "Re-run s3_feature_engineering.py first."
            )

    return df_cl, df_gene


def identify_feature_columns(
    df_cl: pd.DataFrame,
    df_gene: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return the feature column names for each DataFrame.

    Excludes metadata / identifier columns that should never be used as model
    inputs.
    """
    cl_feat_cols = [
        c for c in df_cl.columns
        if c not in ("ModelID", "split")
    ]
    gene_feat_cols = [
        c for c in df_gene.columns
        if c not in ("ModelID", "gene", "CRISPR", "re-asigned", "split")
    ]

    print(f"  Cell lines : {len(df_cl):>8,} rows  ×  {len(cl_feat_cols):>5} features")
    print(f"  Genes      : {len(df_gene):>8,} rows  ×  {len(gene_feat_cols):>5} features")

    return cl_feat_cols, gene_feat_cols


# ── Validation ────────────────────────────────────────────────────────────────

def validate_model_ids(df_cl: pd.DataFrame, df_gene: pd.DataFrame) -> None:
    """Raise if any ModelID present in genes is absent from cell lines."""
    print("Validating ModelIDs …")
    cl_ids   = set(df_cl["ModelID"].unique())
    gene_ids = set(df_gene["ModelID"].unique())
    missing  = gene_ids - cl_ids
    if missing:
        raise ValueError(
            f"{len(missing)} ModelID(s) in genes not found in cell_lines.\n"
            f"  Examples: {list(missing)[:5]}"
        )
    print(f"  OK — {len(gene_ids)} unique ModelIDs matched.")


# ── Split handling ────────────────────────────────────────────────────────────

def extract_split_indices(
    df_gene: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set, set, set]:
    """Read pre-computed split labels and return row indices + cell-line sets.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Integer row indices into df_gene for each partition.
    train_cls, val_cls, test_cls : set[str]
        Unique ModelIDs belonging to each partition.
    """
    print("Reading split assignments …")
    gene_splits = df_gene["split"].values

    train_mask = gene_splits == "train"
    val_mask   = gene_splits == "val"
    test_mask  = gene_splits == "test"

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    train_cls = set(df_gene.loc[train_mask, "ModelID"].unique())
    val_cls   = set(df_gene.loc[val_mask,   "ModelID"].unique())
    test_cls  = set(df_gene.loc[test_mask,  "ModelID"].unique())

    print(
        f"  Cell lines — train: {len(train_cls):>4}  "
        f"val: {len(val_cls):>4}  test: {len(test_cls):>4}"
    )
    print(
        f"  Gene rows  — train: {len(train_idx):>8,}  "
        f"val: {len(val_idx):>8,}  test: {len(test_idx):>8,}"
    )

    return train_idx, val_idx, test_idx, train_cls, val_cls, test_cls


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_normalization_stats(
    df_cl: pd.DataFrame,
    cl_feat_cols: list[str],
    train_cls: set,
    df_gene: pd.DataFrame,
    gene_feat_cols: list[str],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean/std from *training* rows only.

    Parameters
    ----------
    train_mask : np.ndarray of bool
        Boolean mask over df_gene rows selecting the training partition.

    Returns
    -------
    cl_mean, cl_std, gene_mean, gene_std : np.ndarray (float32)
    """
    print("Computing normalization statistics from training data …")

    train_cl_mask  = df_cl["ModelID"].isin(train_cls).values
    train_cl_arr   = df_cl.loc[train_cl_mask, cl_feat_cols].values.astype(np.float32)

    cl_mean = np.nanmean(train_cl_arr, axis=0)
    cl_std  = np.nanstd (train_cl_arr, axis=0)
    cl_std[cl_std < 1e-6] = 1.0

    train_gene_arr = df_gene.loc[train_mask, gene_feat_cols].values.astype(np.float32)

    gene_mean = np.nanmean(train_gene_arr, axis=0)
    gene_std  = np.nanstd (train_gene_arr, axis=0)
    gene_std[gene_std < 1e-6] = 1.0

    print(f"  Cell-line scaler fit on {train_cl_mask.sum()} cell lines.")
    print(f"  Gene scaler fit on {train_mask.sum():,} gene rows.")

    return cl_mean, cl_std, gene_mean, gene_std


def normalize_and_impute(
    df_cl: pd.DataFrame,
    cl_feat_cols: list[str],
    cl_mean: np.ndarray,
    cl_std: np.ndarray,
    df_gene: pd.DataFrame,
    gene_feat_cols: list[str],
    gene_mean: np.ndarray,
    gene_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-score normalize features and impute any remaining NaNs with 0.

    Returns
    -------
    cl_feats, gene_feats : np.ndarray (float32)
    """
    print("Normalizing …")
    cl_feats   = (df_cl[cl_feat_cols].values.astype(np.float32)     - cl_mean)   / cl_std
    gene_feats = (df_gene[gene_feat_cols].values.astype(np.float32) - gene_mean) / gene_std

    print("Imputing NaN values …")
    cl_feats  [np.isnan(cl_feats)]   = 0.0
    gene_feats[np.isnan(gene_feats)] = 0.0

    assert not np.isnan(cl_feats).any(),   "NaNs remain in cell-line features."
    assert not np.isnan(gene_feats).any(), "NaNs remain in gene features."
    print("  No NaNs remaining. ✓")

    return cl_feats, gene_feats


# ── CRISPR target ─────────────────────────────────────────────────────────────

def prepare_crispr_target(
    df_gene: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the QuantileTransformed CRISPR target and drop any NaN rows.

    Returns
    -------
    crispr_vals : np.ndarray (float32)
        Full-length target array (NaNs are preserved in the array itself;
        the split indices exclude them).
    train_idx, val_idx, test_idx : np.ndarray
        Possibly reduced indices after removing NaN CRISPR rows.
    """
    crispr_vals     = df_gene["CRISPR"].values.astype(np.float32)
    crispr_nan_mask = np.isnan(crispr_vals)

    if crispr_nan_mask.any():
        n_drop = crispr_nan_mask.sum()
        print(f"  WARNING: {n_drop} rows with NaN CRISPR — removed from all splits.")
        valid     = ~crispr_nan_mask
        train_idx = train_idx[valid[train_idx]]
        val_idx   = val_idx  [valid[val_idx]]
        test_idx  = test_idx [valid[test_idx]]

    return crispr_vals, train_idx, val_idx, test_idx


# ── HDF5 I/O ──────────────────────────────────────────────────────────────────

def write_hdf5(
    output_path: Path,
    cl_feats: np.ndarray,
    gene_feats: np.ndarray,
    crispr_vals: np.ndarray,
    df_cl: pd.DataFrame,
    df_gene: pd.DataFrame,
    cl_feat_cols: list[str],
    gene_feat_cols: list[str],
    cl_mean: np.ndarray,
    cl_std: np.ndarray,
    gene_mean: np.ndarray,
    gene_std: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    train_cls: set,
    val_cls: set,
    test_cls: set,
    gene_chunk_rows: int = 20_000,
) -> None:
    """Write the fully preprocessed dataset to an HDF5 file.

    The file layout is::

        /cell_lines/features          float32 (n_cl, n_cl_feats)
        /cell_lines/model_ids         bytes   (n_cl,)
        /genes/features               float32 (n_gene, n_gene_feats)  gzip-4
        /genes/crispr                 float32 (n_gene,)               gzip-4
        /genes/gene_id                bytes   (n_gene,)
        /genes/model_id               bytes   (n_gene,)
        /index/splits/train|val|test  int64   row indices into /genes/*
        /index/split_model_ids/…      bytes   ModelIDs per split
        /normalization/…              scalars & feature name arrays

    Global attrs record the CRISPR transform name and split strategy.
    """
    print(f"\nWriting {output_path} …")

    with h5py.File(output_path, "w") as f:

        # Cell lines
        grp_cl = f.create_group("cell_lines")
        grp_cl.create_dataset("features",  data=cl_feats)
        grp_cl.create_dataset("model_ids", data=df_cl["ModelID"].values.astype("S"))

        # Genes
        grp_g = f.create_group("genes")
        grp_g.create_dataset(
            "features",
            data=gene_feats,
            chunks=(gene_chunk_rows, gene_feats.shape[1]),
            compression="gzip", compression_opts=4,
        )
        grp_g.create_dataset(
            "crispr",
            data=crispr_vals,
            chunks=(gene_chunk_rows,),
            compression="gzip", compression_opts=4,
        )
        grp_g.create_dataset("gene_id",  data=df_gene["gene"].values.astype("S"))
        grp_g.create_dataset("model_id", data=df_gene["ModelID"].values.astype("S"))

        # Split indices (gene-row level)
        grp_splits = f.create_group("index/splits")
        grp_splits.create_dataset("train", data=train_idx)
        grp_splits.create_dataset("val",   data=val_idx)
        grp_splits.create_dataset("test",  data=test_idx)

        # Split membership (cell-line level)
        grp_cl_ids = f.create_group("index/split_model_ids")
        grp_cl_ids.create_dataset("train", data=np.array(sorted(train_cls), dtype="S"))
        grp_cl_ids.create_dataset("val",   data=np.array(sorted(val_cls),   dtype="S"))
        grp_cl_ids.create_dataset("test",  data=np.array(sorted(test_cls),  dtype="S"))

        # Normalization statistics
        grp_norm = f.create_group("normalization")
        grp_norm.create_dataset("cl_mean",         data=cl_mean.astype(np.float32))
        grp_norm.create_dataset("cl_std",          data=cl_std.astype(np.float32))
        grp_norm.create_dataset("cl_feat_names",   data=np.array(cl_feat_cols,   dtype="S"))
        grp_norm.create_dataset("gene_mean",       data=gene_mean.astype(np.float32))
        grp_norm.create_dataset("gene_std",        data=gene_std.astype(np.float32))
        grp_norm.create_dataset("gene_feat_names", data=np.array(gene_feat_cols, dtype="S"))

        # Global metadata
        f.attrs["crispr_transform"]        = "QuantileTransformer(output_distribution='normal')"
        f.attrs["crispr_transformer_path"] = "chronos_quantile_transformer.pkl"
        f.attrs["split_strategy"]          = "cell_line"

    print("  HDF5 written successfully. ✓")


def verify_hdf5(output_path: Path) -> None:
    """Print the HDF5 tree to stdout for a quick sanity check."""
    print("\nHDF5 structure:")
    with h5py.File(output_path, "r") as f:
        f.visititems(
            lambda name, obj: print(
                f"  /{name:<55s} {str(getattr(obj, 'shape', '')):>20s}"
            )
        )


# ── Summary reporting ─────────────────────────────────────────────────────────

def print_summary(
    cl_feats: np.ndarray,
    gene_feats: np.ndarray,
    crispr_vals: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    train_cls: set,
    val_cls: set,
    test_cls: set,
) -> None:
    """Print split counts, feature dimensions, and CRISPR target statistics."""
    print("\nSplit summary:")
    print(
        f"  Cell lines — train: {len(train_cls):>4}  "
        f"val: {len(val_cls):>4}  test: {len(test_cls):>4}"
    )
    print(
        f"  Gene rows  — train: {len(train_idx):>8,}  "
        f"val: {len(val_idx):>8,}  test: {len(test_idx):>8,}"
    )

    print("\nFeature dimensions:")
    print(f"  Cell-line features : {cl_feats.shape[1]}")
    print(f"  Gene features      : {gene_feats.shape[1]}")

    print("\nCRISPR target statistics (QuantileTransformed) — training set:")
    tr_crispr = crispr_vals[train_idx]
    print(f"  mean  : {tr_crispr.mean():.4f}  (expect ≈ 0.0)")
    print(f"  std   : {tr_crispr.std():.4f}   (expect ≈ 1.0)")
    print(f"  min   : {tr_crispr.min():.4f}")
    print(f"  max   : {tr_crispr.max():.4f}")

    print(
        "\nNOTE: CRISPR values are QuantileTransformed. To report metrics in\n"
        "  Chronos space, load chronos_quantile_transformer.pkl and call\n"
        "  qt.inverse_transform(predictions)."
    )