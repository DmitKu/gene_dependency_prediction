# -*- coding: utf-8 -*-
"""
scripts/s3_prepare_features.py
──────────────────────────────
Pipeline step 3 — build the gene-level feature matrix and cluster-sum table.

Run from the project root:
    python scripts/s03_feature_engineering.py

Outputs (all paths defined in src/utils.py):
    RNA_CRISPR_all_for_model_extended.csv   ← gene feature matrix + 'split' column
    Mean_cluster_data_all_for_model.csv     ← cluster-sum wide table + 'split' column
    chronos_quantile_transformer.pkl        ← saved for inverse-transform at eval

Next step: s4_build_hdf5.py
"""

from __future__ import annotations
from pathlib import Path
import polars as pl
import logging, sys
import pickle

# Make src/ importable — works from CLI and Spyder
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:                    # __file__ undefined in Spyder
    _root = Path.cwd()               # assumes Spyder cwd = project root
sys.path.insert(0, str(_root / "src"))

from utils_feature_engineering import (
    # Data loading
    load_cluster_info, load_rna, load_crispr,
    common_CellLine_alignment, sanity_check_data,
    # Cluster stats
    melt_rna_with_clusters, compute_loo_cluster_stats, build_cluster_sum_wide,
    # Split
    split_cell_lines, add_split_column, print_split_stats,
    # Transforms
    fit_quantile_transformer, apply_quantile_transformer,
    quantile_transform_sel_cols, minmax_transform_sel_cols,
    sign_log1p, replace_inf_with_null, apply_log_transform,
    # Features
    create_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ── 1. Paths & constants ──────────────────────────────────────────────────────────────
#_root = Path(__file__).resolve().parents[1]
#RNA_FILE    = Path("data/Expression_Public_25Q3_subsetted.csv"
#CRISPR_FILE = Path("data/CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv"


_root = Path(
    r"C:\Users\dkuch\Documents\Blog_ideas_data\Computational"
    r"\MOA_Prediction_based_on_CETSA\20251122_Model_development"
    r"\GitHub_GeneDependancy_prediction"
)


DEPMAP_BASE = Path(
    r"C:\Users\dkuch\Documents\Blog_ideas_data\Computational"
    r"\MOA_Prediction_based_on_CETSA\public_data\DepMap"
)

CLUSTER_CSV      = _root       / "outputs/clustering/UMAP_with_clusters.csv"
RNA_FILE         = DEPMAP_BASE / "Expression" / "Expression_Public_25Q3_subsetted.csv"
CRISPR_FILE      = DEPMAP_BASE / "CRISPR"     / "CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv"
SEL_GENES_FILE   = _root       / "outputs/clustering/Selected_RNA_CRISPR.pkl"
SAVE_DIR = _root / "outputs" / "RNA_fetures"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLUSTER     = SAVE_DIR / "Cell_line_based_features.csv"
OUT_MAIN        = SAVE_DIR / "RNA_based_features_CRISPR.csv"
OUT_TRANSFORMER = SAVE_DIR / "chronos_quantile_transformer.pkl"   # used for inverse-transform at eval

VAL_FRAC      = 0.10    # fraction of cell lines → validation
TEST_FRAC     = 0.10    # fraction of cell lines → test
RANDOM_SEED   = 42

# Columns that should NOT receive the sign_log1p transform —
# either non-numeric identifiers, already in [0, 1], or binary flags.
SKIP_LOG: set[str] = {
    "ModelID", "gene", "cluster", "CRISPR", "split",
    "gene_percentile_in_cluster",
    "gene_fraction_of_cluster_total",
    "rank_value_glob",
    "is_highest_in_cluster",
    "clust_N",
}


# ── 2. Load data ──────────────────────────────────────────────────────────────
print("Loading selected genes for RNA & CRISPR …")
with open(SEL_GENES_FILE, "rb") as f:
    data_gene_sel = pickle.load(f)

selected_crispr_genes = data_gene_sel["crispr_gene"]
selected_rna_genes = data_gene_sel["rna_gene"]

print("Loading cluster info, RNA, CRISPR …")
cluster_info = load_cluster_info(CLUSTER_CSV)
rna_wide_CL     = load_rna(RNA_FILE,
                               selected_rna_genes)
rna_wide_GENE     = load_rna(RNA_FILE,
                                  selected_crispr_genes)
crispr_wide  = load_crispr(CRISPR_FILE,
                           selected_crispr_genes)


rna_wide_CL, rna_wide_GENE, crispr_wide = common_CellLine_alignment(df_rna_CL=rna_wide_CL,
                                                  df_rna_GENE=rna_wide_GENE,
                                                  df_crispr=crispr_wide)

sanity_check_data(rna_wide_CL, selected_rna_genes, label="RNA")

sanity_check_data(crispr_wide, selected_crispr_genes, label="CRISPR")


# ── 3. Build RNA long table with cluster labels ───────────────────────────────

common_genes = set(rna_wide_GENE.columns)&set(cluster_info['gene'])
print("Melting RNA …")
rna_lng_GENE = melt_rna_with_clusters(rna_wide_GENE,
                                      cluster_info,
                                      common_genes)
print(f"  RNA long shape: {rna_lng_GENE.shape}")


# ── 4. Leave-one-out cluster statistics ───────────────────────────────────────

print("Computing leave-one-out cluster statistics …")
rna_lng_GENE = compute_loo_cluster_stats(rna_lng_GENE)


# ── 5. Melt CRISPR and join ───────────────────────────────────────────────────

print("Melting CRISPR and joining …")
crispr_lng = crispr_wide.unpivot(
    index="ModelID", variable_name="gene", value_name="CRISPR"
)

crispr_lng = crispr_lng.drop_nulls(subset=["CRISPR"])


data = (
    rna_lng_GENE
    .join(crispr_lng, on=["ModelID", "gene"], how="inner")
    .drop_nulls(["RNA", "CRISPR"])
)


# ── 6. Train / val / test split on cell lines ─────────────────────────────────

print("\nSplitting cell lines into train / val / test …")
train_cls, val_cls, test_cls = split_cell_lines(
    model_ids=crispr_wide['ModelID'].unique(), val_frac=VAL_FRAC,
    test_frac=TEST_FRAC, random_seed=RANDOM_SEED,
)
data = add_split_column(data, train_cls, val_cls)
print_split_stats(data, train_cls, val_cls, test_cls)


# ── 7. QuantileTransform CRISPR — fit on train only ──────────────────────────

print("\nFitting QuantileTransformer on train CRISPR scores …")
qt   = fit_quantile_transformer(data, RANDOM_SEED, OUT_TRANSFORMER)
data = apply_quantile_transformer(data, qt)


# ── 8. Feature engineering ────────────────────────────────────────────────────

print("\nEngineering features …")
data = create_features(data)


# ── 9. sign_log1p transform ───────────────────────────────────────────────────

data_out, log_cols = apply_log_transform(data = data, 
                                         skip_cols= SKIP_LOG)


# ── 10. quantile transformation ───────────────────────────────────────────────────

quantile_cols = [
    'RNA', 'clust_sum_all', 'clust_N_all', 
    'clust_mean_all','clust_var_all', 'clust_median_all',
    'clust_max_all','clust_min_all', 'clust_sum_excl',
    'clust_N_excl', 'clust_mean_excl', 'clust_sd_excl',
    'clust_mean', 'clust_sd', 'clust_sum', 'clust_N',
    'clust_median', 'clust_max', 'clust_min',
    'gene_rank_in_clust','gene_vs_cluster_mean_ratio',
    'z_score_glob','rank_value_glob','gene_fraction_of_cluster_total'
    ]

data_out = quantile_transform_sel_cols(data_out,
                                   quantile_cols)

# ── 11. minmax transformation ───────────────────────────────────────────────────

minmax_cols = ['gene_percentile_in_cluster',
                  'gene_z_score_in_clust'
                  ]

data_out = minmax_transform_sel_cols(data_out,
                                     minmax_cols)

# ── 12. Cluster-sum table ─────────────────────────────────────────────────────

print("\nBuilding cluster-sum cell-line table …")
split_map   = data_out.select(["ModelID", "split"]).unique("ModelID")

print("Melting RNA …")
rna_wide_CL_melt = melt_rna_with_clusters(rna_wide_CL, cluster_info, selected_rna_genes)
print(f"  RNA long shape: {rna_wide_CL.shape}")

cluster_out = build_cluster_sum_wide(rna_wide_CL_melt, split_map)
cluster_out = cluster_out.drop("null")


cluster_num_cols = [col for col in cluster_out.columns if col not in ['ModelID','split']]
cluster_out = quantile_transform_sel_cols(cluster_out, cluster_num_cols)


# ── 13. remove columns with many NULLs ─────────────────────────────────────────────────────

null_columns = [
    col for col in data_out.columns 
    if data_out[col].null_count() > 0
]

data_out = data_out.drop(null_columns)

# ── 14. Save ──────────────────────────────────────────────────────────────────

print("\nSaving …")
cluster_out.to_csv(OUT_CLUSTER, index=False)
data_out.write_csv(OUT_MAIN)
print(f"  → {OUT_CLUSTER}  {cluster_out.shape}")
print(f"  → {OUT_MAIN}  {data_out.shape}")


# ── 15. summary ────────────────────────────────────────────────────────

gene_feat_cols = [
    c for c in data_out.columns
    if c not in {"ModelID", "gene", "cluster", "CRISPR", "split"}
]
print(f"\nGene feature columns:  {len(gene_feat_cols)} total")
print(f"\nCell-line cluster-sum columns: {len(cluster_num_cols)}")
print(f"\nOutputs:")
print(f"  {OUT_MAIN.name}  ← includes 'split' column and transformed CRISPR")
print(f"  {OUT_CLUSTER.name}  ← includes 'split' column")
print(f"  {OUT_TRANSFORMER.name}  ← QuantileTransformer for inverse-transform at eval")
print("\nDone. Run s4_build_hdf5.py next.")
