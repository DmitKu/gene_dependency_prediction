# -*- coding: utf-8 -*-
"""
utils_manifold_clustering.py
────────
All utilities for s3_prepare_features.py:

  - Paths & constants
  - Gene selection
  - Data loading & alignment
  - RNA melting & leave-one-out cluster statistics
  - Train / val / test split on cell lines
  - QuantileTransformer fit & apply
  - Feature engineering
  - sign_log1p & inf-replacement transforms
  - Cluster-sum wide table
"""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


# ══════════════════════════════════════════════════════════════════════════════
# Gene selection
# ══════════════════════════════════════════════════════════════════════════════

def select_genes_by_variance(crispr_path: Path, min_sd: float = 0.01) -> list[str]:
    """Return gene names whose cross-cell-line CRISPR std >= *min_sd*."""
    df = pd.read_csv(crispr_path)
    df = df.rename(columns={"Unnamed: 0": "ModelID"}).set_index("ModelID")
    var_per_gene = df.std(axis=0, numeric_only=True)
    return var_per_gene[var_per_gene >= min_sd].index.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Data loading & alignment
# ══════════════════════════════════════════════════════════════════════════════

def load_cluster_info(cluster_csv: Path) -> pl.DataFrame:
    """Load gene → cluster mapping; keeps ['gene', 'cluster']."""
    return pl.read_csv(cluster_csv).select(["gene", "cluster"])


def load_rna(rna_csv: Path,
             selected_genes: list) -> pl.DataFrame:
    """Load RNA expression wide table; renames the index column to 'ModelID'."""
    df = pl.read_csv(rna_csv).rename({"": "ModelID"})
    common_gene = set(df.columns) & set(selected_genes)
    df = df.select(['ModelID',*common_gene])
    return df

def load_crispr(crispr_csv: Path,
                selected_genes: list) -> pl.DataFrame:
    """
    Load CRISPR Chronos-score wide table.
    Strips the trailing '..ENSG…' suffix DepMap appends to some column names.
    """
    header_cols = pl.read_csv(crispr_csv, n_rows=0).columns
    rename_map  = {c: c.split("..")[0] for c in header_cols if ".." in c}
    df = pl.read_csv(crispr_csv).rename({"": "ModelID"}).rename(rename_map)
    common_gene = set(df.columns) & set(selected_genes)
    df = df.select(['ModelID',*common_gene])
    return df



def common_CellLine_alignment(df_rna_CL: pl.DataFrame,
                              df_rna_GENE: pl.DataFrame,
                              df_crispr: pl.DataFrame) -> pl.DataFrames:
    """remove uncommon cell lines
       returns filtered RNA dataframe and filtered CRISPR dataframe
    """
    common_cell_lines = set(df_rna_CL['ModelID'])&set(df_rna_GENE['ModelID'])&set(df_crispr['ModelID'])
    df_rna_CL = df_rna_CL.filter(pl.col("ModelID").is_in(common_cell_lines))
    df_rna_GENE = df_rna_GENE.filter(pl.col("ModelID").is_in(common_cell_lines))
    df_crispr = df_crispr.filter(pl.col("ModelID").is_in(common_cell_lines))
 
    return df_rna_CL, df_rna_GENE, df_crispr


    
def sanity_check_data(df_wide: pl.DataFrame, common_genes: list[str], label: str, n: int = 10) -> None:
    """Print basic stats on the first *n* genes as a data-quality check.
    
    Args:
        df_wide: The Polars DataFrame containing the gene data.
        common_genes: List of gene names to select from.
        label: The dataset label for the print statement (e.g., "RNA" or "CRISPR").
        n: Number of genes to sample. Defaults to 10.
    """
    sample_vals = df_wide.select(common_genes[:n]).to_numpy().flatten()
    sample_vals = sample_vals[~np.isnan(sample_vals)]
    
    if sample_vals.size == 0:
        print(f"\n{label} sanity check: No valid numerical data found.")
        return

    print(
        f"\n{label} sanity check: min={sample_vals.min():.2f}  "
        f"median={np.median(sample_vals):.2f}  "
        f"max={sample_vals.max():.2f}"
    )    
# ══════════════════════════════════════════════════════════════════════════════
# RNA melt & leave-one-out cluster statistics
# ══════════════════════════════════════════════════════════════════════════════

def melt_rna_with_clusters(
    rna_wide: pl.DataFrame,
    cluster_info: pl.DataFrame,
    common_genes:list,
) -> pl.DataFrame:
    """Unpivot RNA from wide to long and attach cluster labels."""
    rna_lng = rna_wide.unpivot(index="ModelID", variable_name="gene", value_name="RNA")
    rna_lng = rna_lng.filter(pl.col("gene").is_in(common_genes))

    return rna_lng.join(cluster_info, on="gene", how="left")


def compute_loo_cluster_stats(rna_lng: pl.DataFrame) -> pl.DataFrame:
    """
    Add leave-one-out (LOO) cluster statistics to *rna_lng*.

    Each gene's cluster context is computed excluding itself so the model
    cannot trivially learn from its own signal reflected in the cluster summary.
    The incremental Welford variance update avoids a second groupby pass:
        Var_excl = (Var_all*(N-1) - (x - mean_all)*(x - mean_excl)) / (N_excl - 1)
    """
    cluster_stats = (
        rna_lng
        .group_by(["ModelID", "cluster"])
        .agg([
            pl.col("RNA").sum()                    .alias("clust_sum_all"),
            pl.col("RNA").count().cast(pl.Float64) .alias("clust_N_all"),
            pl.col("RNA").mean()                   .alias("clust_mean_all"),
            pl.col("RNA").var()                    .alias("clust_var_all"),
            pl.col("RNA").median()                 .alias("clust_median_all"),
            pl.col("RNA").max()                    .alias("clust_max_all"),
            pl.col("RNA").min()                    .alias("clust_min_all"),
        ])
    )
    rna_lng = rna_lng.join(cluster_stats, on=["ModelID", "cluster"], how="left")

    rna_lng = rna_lng.with_columns([
        (pl.col("clust_sum_all") - pl.col("RNA")).alias("clust_sum_excl"),
        (pl.col("clust_N_all")   - 1.0)          .alias("clust_N_excl"),
    ])

    rna_lng = rna_lng.with_columns([
        pl.when(pl.col("clust_N_excl") > 0)
          .then(pl.col("clust_sum_excl") / pl.col("clust_N_excl"))
          .otherwise(pl.col("clust_mean_all"))
          .alias("clust_mean_excl"),
    ])

    rna_lng = rna_lng.with_columns([
        pl.when(pl.col("clust_N_excl") > 1)
          .then(
              (
                  (pl.col("clust_var_all") * (pl.col("clust_N_all") - 1)
                   - (pl.col("RNA") - pl.col("clust_mean_all"))
                     * (pl.col("RNA") - pl.col("clust_mean_excl")))
                  / (pl.col("clust_N_excl") - 1)
              ).clip(lower_bound=0.0).sqrt()
          )
          .otherwise(pl.lit(0.0))
          .alias("clust_sd_excl"),
    ])

    return rna_lng


# ══════════════════════════════════════════════════════════════════════════════
# Train / val / test split on cell lines
# ══════════════════════════════════════════════════════════════════════════════

def split_cell_lines(
    model_ids: set[str],
    val_frac: float,
    test_frac: float,
    random_seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """
    Randomly partition cell lines into train / val / test.

    Splitting on cell lines (not rows) prevents leakage: a single cell
    line contributes one row per gene, so row-level splitting would leak
    the same cell line across train and eval sets.
    """
    rng     = np.random.default_rng(random_seed)
    all_cls = np.array(sorted(model_ids))
    rng.shuffle(all_cls)

    n_cl    = len(all_cls)
    n_test  = int(n_cl * test_frac)
    n_val   = int(n_cl * val_frac)
    n_train = n_cl - n_val - n_test

    train_cls = set(all_cls[:n_train])
    val_cls   = set(all_cls[n_train : n_train + n_val])
    test_cls  = set(all_cls[n_train + n_val :])

    return train_cls, val_cls, test_cls


def add_split_column(
    data: pl.DataFrame,
    train_cls: set[str],
    val_cls: set[str],
) -> pl.DataFrame:
    """Append a 'split' column ('train' | 'val' | 'test') to *data*."""
    def _label(model_id: str) -> str:
        if model_id in train_cls:
            return "train"
        if model_id in val_cls:
            return "val"
        return "test"

    return data.with_columns(
        pl.Series("split", [_label(m) for m in data["ModelID"].to_list()])
    )


def print_split_stats(
    data: pl.DataFrame,
    train_cls: set[str],
    val_cls: set[str],
    test_cls: set[str],
) -> None:
    """Print cell-line and gene-row counts per split."""
    print(
        f"  Cell lines — train: {len(train_cls):>4}  "
        f"val: {len(val_cls):>4}  test: {len(test_cls):>4}"
    )
    print(
        f"  Gene rows  — train: {(data['split'] == 'train').sum():>8,}  "
        f"val: {(data['split'] == 'val').sum():>8,}  "
        f"test: {(data['split'] == 'test').sum():>8,}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# QuantileTransformer (CRISPR target)
# ══════════════════════════════════════════════════════════════════════════════

def fit_quantile_transformer(
    data: pl.DataFrame,
    random_seed: int,
    out_path: Path,
) -> QuantileTransformer:
    """
    Fit a QuantileTransformer → N(0,1) on the TRAIN rows of *data*,
    save it to *out_path*, and return it.

    Raw Chronos is ~80 % near 0 (nonessential) with a long left tail.
    Mapping to N(0,1) removes the implicit class imbalance without
    requiring focal weights on the target.

    CRITICAL: fit on training data only to prevent leakage into val/test.
    """
    train_scores = (
        data.filter(pl.col("split") == "train")["CRISPR"]
        .to_numpy().reshape(-1, 1)
    )
    qt = QuantileTransformer(
        output_distribution = "normal",
        n_quantiles         = min(1000, len(train_scores)),
        random_state        = random_seed,
    )
    qt.fit(train_scores)
    joblib.dump(qt, out_path)
    print(f"  Transformer saved → {out_path.name}")

    tr_tfm = qt.transform(train_scores).squeeze()
    print(
        f"  Raw Chronos (train) : mean={train_scores.mean():.3f}  "
        f"std={train_scores.std():.3f}  "
        f"min={train_scores.min():.3f}  max={train_scores.max():.3f}"
    )
    print(
        f"  Transformed (train) : mean={tr_tfm.mean():.3f}  "
        f"std={tr_tfm.std():.3f}  "
        f"min={tr_tfm.min():.3f}  max={tr_tfm.max():.3f}"
    )
    print("  (should be approx. mean≈0, std≈1)")
    return qt


def apply_quantile_transformer(
    data: pl.DataFrame,
    qt: QuantileTransformer,
) -> pl.DataFrame:
    """Apply *qt* to the 'CRISPR' column of *data* (all splits)."""
    crispr_raw         = data["CRISPR"].to_numpy().reshape(-1, 1)
    crispr_transformed = qt.transform(crispr_raw).squeeze().astype(np.float32)
    return data.with_columns(pl.Series("CRISPR", crispr_transformed))


# ══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════════════

def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Derive all gene-level features from RNA and LOO cluster statistics.

    Feature groups
    ──────────────
    Cluster-context (LOO):
      clust_mean / sd / sum / N / median / max / min
      gene_rank_in_clust, gene_z_score_in_clust
      gene_percentile_in_cluster, gene_vs_cluster_mean_ratio
      gene_fraction_of_cluster_total, is_highest_in_cluster

    Global (all genes for this cell line):
      z_score_glob, rank_value_glob
    """
    clust_scope = ["ModelID", "cluster"]

    df = df.with_columns([
        pl.col("clust_mean_excl")  .alias("clust_mean"),
        pl.col("clust_sd_excl")    .alias("clust_sd"),
        pl.col("clust_sum_excl")   .alias("clust_sum"),
        pl.col("clust_N_excl")     .alias("clust_N"),
        pl.col("clust_median_all") .alias("clust_median"),
        pl.col("clust_max_all")    .alias("clust_max"),
        pl.col("clust_min_all")    .alias("clust_min"),
    ])

    df = df.with_columns([
        pl.col("RNA").rank(method="average").over(clust_scope).alias("gene_rank_in_clust"),
    ])

    df = df.with_columns([
        (pl.col("RNA") == pl.col("clust_max"))
            .cast(pl.Int8).alias("is_highest_in_cluster"),
        ((pl.col("RNA") - pl.col("clust_mean"))
         / pl.col("clust_sd").clip(lower_bound=1e-8))
            .alias("gene_z_score_in_clust"),
        (pl.col("gene_rank_in_clust") / pl.col("clust_N").clip(lower_bound=1.0))
            .alias("gene_percentile_in_cluster"),
        (pl.col("RNA") / pl.col("clust_mean").clip(lower_bound=1e-8))
            .alias("gene_vs_cluster_mean_ratio"),
        (pl.col("RNA") / pl.col("clust_sum").clip(lower_bound=1e-8))
            .alias("gene_fraction_of_cluster_total"),
    ])

    df = (
        df.with_columns([
            pl.col("RNA").count().over("ModelID").cast(pl.Float64).alias("_global_N"),
            pl.col("RNA").rank(method="average").over("ModelID").alias("_raw_rank_glob"),
            ((pl.col("RNA") - pl.col("RNA").mean().over("ModelID"))
             / pl.col("RNA").std().over("ModelID").clip(lower_bound=1e-8))
                .alias("z_score_glob"),
        ])
        .with_columns(
            (pl.col("_raw_rank_glob") / pl.col("_global_N")).alias("rank_value_glob")
        )
        .drop(["_global_N", "_raw_rank_glob"])
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════════════

def sign_log1p(s: pl.Series) -> pl.Series:
    """sign(x) * log(1 + |x|) — compresses skewed features, preserves sign."""
    return s.sign() * (s.abs() + 1).log(base=math.e)


def quantile_transform_sel_cols(df_wide: pl.DataFrame,
                                minmax_cols: list):
    # 1. Set up the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            # Pass the entire list of columns here
            ('quantile', QuantileTransformer(n_quantiles=1000,
                                             output_distribution='normal',
                                             random_state=42), minmax_cols)
        ],
        remainder='passthrough' # Leaves all other columns (like binary flags or raw scales) completely untouched
    )

    # 23. Ensure the output stays as a clean DataFrame with column names intact
    preprocessor.set_output(transform="pandas")

    # 3. Fit and transform your feature matrix
    X_train_transformed = preprocessor.fit_transform(df_wide.to_pandas())
    # 4. rename columns
    X_train_transformed.columns = X_train_transformed.columns.str.replace('^.+__','',regex=True)
    return X_train_transformed

def minmax_transform_sel_cols(df_wide: pl.DataFrame,
                                minmax_cols: list):
    # 1. Set up the ColumnTransformer
    preprocessor_minmax = ColumnTransformer(
        transformers=[
            ('minmax', MinMaxScaler(), minmax_cols)
        ],
        remainder='passthrough' # Leaves all your other features completely raw
    )

    # 23. Ensure the output stays as a clean DataFrame with column names intact
    preprocessor_minmax.set_output(transform="pandas")

    # 3. Fit and transform your feature matrix
    X_train_transformed = preprocessor_minmax.fit_transform(df_wide)
    # 4. rename columns
    X_train_transformed.columns = X_train_transformed.columns.str.replace('^.+__','',regex=True)
    X_train_transformed = pl.from_pandas(X_train_transformed)
    return X_train_transformed



def robust_transform_sel_cols(df_wide: pl.DataFrame, robust_cols: list) -> pl.DataFrame:
    # 1. Set up the ColumnTransformer with RobustScaler
    preprocessor_robust = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), robust_cols)
        ],
        remainder='passthrough' # Leaves all your other features completely raw
    )

    # 2. Ensure the output stays as a clean Polars DataFrame natively
    preprocessor_robust.set_output(transform="pandas")

    # 3. Fit and transform your feature matrix
    X_train_transformed = preprocessor_robust.fit_transform(df_wide)
    # 4. rename columns
    X_train_transformed.columns = X_train_transformed.columns.str.replace('^.+__','',regex=True)
    X_train_transformed = pl.from_pandas(X_train_transformed)
    
    return X_train_transformed


def replace_inf_with_null(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Replace ±Inf with null in specified float columns."""
    exprs = []
    for c in cols:
        if df[c].dtype in (pl.Float32, pl.Float64):
            exprs.append(
                pl.when(pl.col(c).is_infinite())
                  .then(None)
                  .otherwise(pl.col(c))
                  .alias(c)
            )
        else:
            exprs.append(pl.col(c))
    return df.with_columns(exprs) if exprs else df


def apply_log_transform(
    data: pl.DataFrame,
    skip_cols: set[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Apply sign_log1p to all numeric columns not in *skip_cols*.
    Returns the transformed DataFrame and the list of columns transformed.
    """

    log_cols = [
        c for c in data.columns
        if c not in skip_cols
        and data[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8)
    ]

    data_out = data.with_columns([sign_log1p(data[c]).alias(c) for c in log_cols])

    float_cols = [
        c for c in data_out.columns
        if data_out[c].dtype in (pl.Float32, pl.Float64) and c != "CRISPR"
    ]
    data_out = replace_inf_with_null(data_out, float_cols)

    return data_out, log_cols


# ══════════════════════════════════════════════════════════════════════════════
# Cluster-sum wide table
# ══════════════════════════════════════════════════════════════════════════════

def build_cluster_sum_wide(
    rna_lng_valid: pl.DataFrame,
    split_map: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build a per-cell-line wide table of per-cluster RNA sums.

    Parameters
    ----------
    rna_lng_valid : Long RNA table filtered to cell lines present in the final data.
    split_map     : DataFrame with ['ModelID', 'split'] for downstream reference.
    """
    cluster_sum_wide = (
        rna_lng_valid
        .group_by(["ModelID", "cluster"])
        .agg(pl.col("RNA").sum().alias("cluster_sum"))
        .pivot(on="cluster", index="ModelID", values="cluster_sum")
        .sort("ModelID")
    )
    return cluster_sum_wide.join(split_map, on="ModelID", how="left")