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

# Make src/ importable — works from CLI and Spyder
try:
    _root = Path(__file__).resolve().parents[1]
except NameError:                    # __file__ undefined in Spyder
    _root = Path.cwd()               # assumes Spyder cwd = project root
sys.path.insert(0, str(_root / "src"))

from utils_manifold_clustering import (
    load_gene_set,
    load_correlations,
    filter_and_pivot,
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
SAVE_DIR   = Path("outputs/clustering")

COR_DIR    = Path(r'C:\Users\dkuch\Documents\Blog_ideas_data\Computational\MOA_Prediction_based_on_CETSA\Analysis_data\cor_results')
selected_genes_path =  'C:/Users/dkuch/Documents/Blog_ideas_data/Computational/MOA_Prediction_based_on_CETSA/Analysis_data/Selected_genes_based_on_RNA_CRISPR.csv'
GENES_CSV  = Path(selected_genes_path)

N_JOBS             = 11
PCA_COMPONENTS     = 200
UMAP_NEIGHBORS     = 2
UMAP_MIN_DIST      = 0.005
UMAP_METRIC        = "euclidean"
DBSCAN_EPS         = 0.05
DBSCAN_MIN_SAMPLES = 1
HIGHLIGHT_GENES    = ["MET"]

# ── PIPELINE ──────────────────────────────────────────────────────────────────

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# gene_set loaded first — passed into loader so filtering happens per file
gene_set  = load_gene_set(GENES_CSV)
corr      = load_correlations(COR_DIR, gene_set=gene_set, n_jobs=N_JOBS)
wide      = filter_and_pivot(corr, gene_set)
del corr, gene_set

X_pca     = run_pca(wide, n_components=PCA_COMPONENTS)
X_umap    = run_umap(X_pca, n_neighbors=UMAP_NEIGHBORS,
                     min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC)
del X_pca

labels, _ = run_dbscan(X_umap, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)

save_clusters(wide.index, X_umap, labels, SAVE_DIR)

tag = f"PCA{PCA_COMPONENTS}_{UMAP_METRIC}_NN{UMAP_NEIGHBORS}_dist{UMAP_MIN_DIST}"
plot_cluster_histogram(labels, SAVE_DIR)
plot_umap(X_umap, wide.index, labels, SAVE_DIR, tag=tag)

for gene in HIGHLIGHT_GENES:
    plot_gene_highlight(X_umap, wide.index, labels, gene, SAVE_DIR)