"""
scripts/02_Manifold_Clustering.py
----------------------------------
Driver for the UMAP + DBSCAN gene clustering step.

Usage
-----
    python scripts/02_Manifold_Clustering.py

Adjust the CONFIG block below before running.
"""

import logging, sys
from pathlib import Path
import dask.dataframe as dd
import pandas as pd

# Make src/ importable — works from CLI and Spyder
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:                    # __file__ undefined in Spyder
    _root = Path.cwd()               # assumes Spyder cwd = project root
sys.path.insert(0, str(_root / "src"))

from utils_manifold_clustering import (
    select_genes_by_variance,
    select_active_inactive_genes,
    load_gene_set,
    load_correlations,
    run_pca,
    run_umap,
    run_dbscan,
    save_clusters,
    plot_cluster_histogram,
    plot_umap,
    plot_gene_highlight,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

#COR_DIR    = Path("data/cor_results")
#GENES_CSV  = Path("data/selected_genes.csv")
#RNA_FILE    = Path("data/Expression_Public_25Q3_subsetted.csv"
#CRISPR_FILE = Path("data/CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv"


COR_DIR    = Path(r'C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\Analysis_data\cor_results')
selected_genes_path =  'C:/Users/dkuch/Documents/Blog_ideas_data/Computational/MOA_Prediction_based_on_CETSA/Analysis_data/Selected_genes_based_on_RNA_CRISPR.csv'
BASE_PATH   = Path("C:/Users/dkuch/Documents/Blog_ideas_data/Computational/MOA_Prediction_based_on_CETSA")
RNA_FILE    = BASE_PATH/ "public_data/DepMap/Expression/Expression_Public_25Q3_subsetted.csv"
CRISPR_FILE = BASE_PATH/ "public_data/DepMap/CRISPR/CRISPR_(DepMap_Public_25Q3+Score,_Chronos)_subsetted.csv"

GENES_CSV  = Path(selected_genes_path)
#SAVE_DIR   = Path(r"C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\20251122_Model_development\GitHub_GeneDependancy_prediction\outputs\clustering")

MIN_RNA_SD = 0.7
N_JOBS             = 11
PCA_COMPONENTS     = 200
UMAP_NEIGHBORS     = 2
UMAP_MIN_DIST      = 0.005
UMAP_METRIC        = "euclidean"
DBSCAN_EPS         = 0.005
DBSCAN_MIN_SAMPLES = 1
HIGHLIGHT_GENES    = ["MET", "EGFR", 'MYC', 'TP53']

# ── PIPELINE ──────────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
SAVE_DIR = _root / "outputs" / "clustering"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# gene_set loaded first — passed into loader so filtering happens per file
gene_set    = load_gene_set(GENES_CSV)
corr        = load_correlations(COR_DIR, gene_set=gene_set, n_jobs=N_JOBS)
RNA_data    = pd.read_csv(RNA_FILE)
CRISPR_data = pd.read_csv(CRISPR_FILE)



###################################################
# select RNAs
###################################################
print("Selecting genes by RNA variance …")
selected_rna_genes = select_genes_by_variance(RNA_data, min_sd=MIN_RNA_SD)
print(f"  Genes passing RNA variance filter: {len(selected_rna_genes):,}\n")

###################################################
# select CRISPR genes
###################################################
print("Selecting genes by CRISPR dyversity\n(at least 5 cell lines active and 5 cell lines inactiv) …")
selected_crispr_genes = select_active_inactive_genes(data_crispr = CRISPR_data,
                                            activity_threshhold = -0.5,
                                            non_activity_threshhold = -0.3)
print(f"  Genes passing dyversity filter: {len(selected_crispr_genes):,}\n")

del CRISPR_data
del RNA_data


###################################################
# remove corealtions that are not informative
###################################################
selected_crispr_genes = set(selected_crispr_genes)
selected_rna_genes = set(selected_rna_genes)


corr = corr.loc[
    corr['crispr_gene'].isin(selected_crispr_genes) &
    corr['rna_gene'].isin(selected_rna_genes)
]


###################################################
# shape from long to wide format
# columns: CRISPR genes and rows: RNA genes
# Writen usig DASK to save RAM, very RAM heavy analysis
###################################################

# Convert pandas DataFrame to Dask
ddf = dd.from_pandas(corr, npartitions=1)

# Cast to category
ddf["rna_gene"] = ddf["rna_gene"].astype("category")
ddf["crispr_gene"] = ddf["crispr_gene"].astype("category")

# Make categories known
ddf["rna_gene"] = ddf["rna_gene"].cat.as_known()
ddf["crispr_gene"] = ddf["crispr_gene"].cat.as_known()

# Pivot table
wide = ddf.pivot_table(
    index="rna_gene",
    columns="crispr_gene",
    values="sperman",
    aggfunc="first"
)

# Trigger computation
wide = wide.compute()

del corr, gene_set

X_pca     = run_pca(wide, n_components=PCA_COMPONENTS)
X_umap    = run_umap(X_pca, n_neighbors=UMAP_NEIGHBORS,
                     min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC)
del X_pca

labels, _ = run_dbscan(X_umap, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

save_clusters(wide.index, X_umap, labels, SAVE_DIR)

tag = f"PCA{PCA_COMPONENTS}_{UMAP_METRIC}_NN{UMAP_NEIGHBORS}_dist{UMAP_MIN_DIST}"
plot_cluster_histogram(labels, SAVE_DIR)
plot_umap(X_umap, wide.index, labels, SAVE_DIR)

for gene in HIGHLIGHT_GENES:
    plot_gene_highlight(X_umap, wide.index, labels, gene, SAVE_DIR)