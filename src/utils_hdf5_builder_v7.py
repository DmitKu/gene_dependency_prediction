# -*- coding: utf-8 -*-
"""
utils_hdf5_builder.py
===============
Library of functions for building model_H5_data.h5.

Called by s04_build_hdf5.py — do not run directly.
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
    cl_umap_csv: str,
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
    df_cl_umap = pd.read_csv(base / cl_umap_csv)

    # Deterministic row order
    df_gene = df_gene.sort_values(by="gene").reset_index(drop=True)

    for label, df in [("gene CSV", df_gene), ("cell-line CSV", df_cl)]:
        if "split" not in df.columns:
            raise AssertionError(
                f"'split' column missing from {label}.\n"
                "Re-run s3_feature_engineering.py first."
            )

    return df_cl, df_gene, df_cl_umap


def identify_feature_columns(
    df_cl: pd.DataFrame,
    df_gene: pd.DataFrame,
    df_cl_umap: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return the feature column names for each DataFrame.

    Excludes metadata / identifier columns that should never be used as
    model inputs.
    """
    cl_feat_cols = [
        c for c in df_cl.columns
        if c not in ("ModelID", "split")
    ]
    gene_feat_cols = [
        c for c in df_gene.columns
        if c not in ("ModelID", "gene", "CRISPR", "split", "cluster",
                     )
    ]
    
    cl_umap_feat_cols = [
        c for c in df_cl_umap.columns
        if c not in ("ModelID", "gene", "CRISPR", "split", "cluster",
                     )
    ]

    print(f"  Cell lines : {len(df_cl):>8,} rows  ×  {len(cl_feat_cols):>5} features")
    print(f"  Genes      : {len(df_gene):>8,} rows  ×  {len(gene_feat_cols):>5} features")
    print(f"  cl_umap      : {len(df_cl_umap):>8,} rows  ×  {len(cl_umap_feat_cols):>5} features")


    return cl_feat_cols, gene_feat_cols, cl_umap_feat_cols


# ── Validation ────────────────────────────────────────────────────────────────

def validate_model_ids(df_cl: pd.DataFrame, df_gene: pd.DataFrame, df_cl_umap: pd.DataFrame) -> None:
    """Raise if any ModelID present in genes is absent from cell lines."""
    print("Validating ModelIDs …")
    cl_ids   = set(df_cl["ModelID"].unique())
    gene_ids = set(df_gene["ModelID"].unique())
    cl_umap_ids = set(df_cl_umap["ModelID"].unique())
    missing  = gene_ids - cl_ids
    missing  = missing - cl_umap_ids
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


# ── Feature extraction & validation ──────────────────────────────────────────

def load_and_validate_features(
    df_cl: pd.DataFrame,
    cl_feat_cols: list[str],
    df_gene: pd.DataFrame,
    gene_feat_cols: list[str],
    df_cl_umap: pd.DataFrame,
    cl_umap_feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature arrays and assert no NaNs remain.

    Features are already fully normalized by s03_feature_engineering.py
    (sign_log1p → QuantileTransformer → MinMaxScaler), so no further
    scaling is applied here.

    Returns
    -------
    cl_feats, gene_feats : np.ndarray (float32)
    """
    print("Extracting feature arrays …")
    cl_feats   = df_cl[cl_feat_cols].values.astype(np.float32)
    gene_feats = df_gene[gene_feat_cols].values.astype(np.float32)
    cl_umap_feats = df_cl_umap[cl_umap_feat_cols].values.astype(np.float32)

    # Sanity check — upstream pipeline should have removed all NaNs
    nan_cl   = np.isnan(cl_feats).sum()
    nan_gene = np.isnan(gene_feats).sum()
    nan_cl_umap = np.isnan(cl_umap_feats).sum()
    

    if nan_cl > 0:
        raise ValueError(
            f"{nan_cl} NaNs found in cell-line features. "
            "Re-run s03_feature_engineering.py."
        )
    if nan_gene > 0:
        raise ValueError(
            f"{nan_gene} NaNs found in gene features. "
            "Re-run s03_feature_engineering.py."
        )
    if nan_cl_umap > 0:
        raise ValueError(
            f"{nan_cl_umap} NaNs found in cl umap features. "
            "Re-run s03_feature_engineering.py."
        )

    print("  No NaNs found in features. ✓")
    print(f"  cl_feats   shape: {cl_feats.shape}")
    print(f"  gene_feats shape: {gene_feats.shape}")
    print(f"  cl_umap_feats shape: {cl_umap_feats.shape}")

    return cl_feats, gene_feats, cl_umap_feats


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
    else:
        print("  No NaN CRISPR values. ✓")

    return crispr_vals, train_idx, val_idx, test_idx


# ── HDF5 I/O ──────────────────────────────────────────────────────────────────

def write_hdf5(
    output_path: Path,
    cl_feats: np.ndarray,
    gene_feats: np.ndarray,
    cl_umap_feats: np.ndarray,
    crispr_vals: np.ndarray,
    df_cl: pd.DataFrame,
    df_gene: pd.DataFrame,
    df_cl_umap: pd.DataFrame,
    cl_feat_cols: list[str],
    gene_feat_cols: list[str],
    cl_umap_feat_cols: list[str],
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
        /cell_lines/model_id          bytes   (n_cl,)
        /cl_umap/features             float32 (n_cl, n_cl_feats)
        /cl_umap/model_id             bytes   (n_cl,)
        /genes/features               float32 (n_gene, n_gene_feats)  gzip-4
        /genes/crispr                 float32 (n_gene,)               gzip-4
        /genes/gene_id                bytes   (n_gene,)
        /genes/model_id               bytes   (n_gene,)
        /index/splits/train|val|test  int64   row indices into /genes/*
        /index/split_model_ids/…      bytes   ModelIDs per split
        /metadata/feature_names/…     bytes   feature name arrays

    Global attrs record the CRISPR transform name and split strategy.
    """
    print(f"\nWriting {output_path} …")

    with h5py.File(output_path, "w") as f:

        # Cell lines
        grp_cl = f.create_group("cell_lines")
        grp_cl.create_dataset("features",  data=cl_feats)
        grp_cl.create_dataset("model_id", data=df_cl["ModelID"].values.astype("S"))
        
        # cl umap
        grp_clumap = f.create_group("cl_umap")
        grp_clumap.create_dataset("features",  data=cl_umap_feats)
        grp_clumap.create_dataset("model_id", data=df_cl_umap["ModelID"].values.astype("S"))

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
        grp_g.create_dataset("gene_id", data=df_gene["gene"].values.astype("S"))
        grp_g.create_dataset("model_id", data=df_gene["ModelID"].values.astype("S"))
        grp_g.create_dataset("cluster_id", data=df_gene["cluster"].values.astype(int)) ###added

        # Split indices (gene-row level)
        grp_splits = f.create_group("index/splits")
        grp_splits.create_dataset("train", data=train_idx)
        grp_splits.create_dataset("val",   data=val_idx)
        grp_splits.create_dataset("test",  data=test_idx)

        # Split membership (cell-line level)
        grp_cl_ids = f.create_group("index/split_model_id")
        grp_cl_ids.create_dataset("train", data=np.array(sorted(train_cls), dtype="S"))
        grp_cl_ids.create_dataset("val",   data=np.array(sorted(val_cls),   dtype="S"))
        grp_cl_ids.create_dataset("test",  data=np.array(sorted(test_cls),  dtype="S"))

        # Feature names (replaces normalization group)
        grp_meta = f.create_group("metadata/feature_names")
        grp_meta.create_dataset("cl_feat_names",
                                data=np.array(cl_feat_cols,   dtype="S"))
        grp_meta.create_dataset("gene_feat_names",
                                data=np.array(gene_feat_cols, dtype="S"))
        grp_meta.create_dataset("cl_umap_feat_names",
                                data=np.array(cl_umap_feat_cols, dtype="S"))


        # Global metadata
        f.attrs["crispr_transform"]        = "QuantileTransformer(output_distribution='normal')"
        f.attrs["crispr_transformer_path"] = "chronos_quantile_transformer.pkl"
        f.attrs["split_strategy"]          = "cell_line"
        f.attrs["normalization"]           = "applied_in_s03_feature_engineering"

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