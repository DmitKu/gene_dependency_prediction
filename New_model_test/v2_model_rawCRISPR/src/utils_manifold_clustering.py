"""
src/utils_manifold_clustering.py
---------------------------------
Reusable functions for manifold-based gene clustering.
Called by scripts/02_Manifold_Clustering.py.
"""

from __future__ import annotations
import glob, logging
from pathlib import Path

import pandas as pd
import pyreadr
import umap
import plotly.express as px
import plotly.io as pio
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

# Genes with near-all-NA rows/cols — extend as needed
_BAD_GENES: frozenset[str] = frozenset({"DEFB113", "KRTAP23.1"})


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_gene_set(genes_csv: Path, col: str = "gene_name") -> set[str]:
    """Return the curated gene set produced by the RNA/CRISPR selection step."""
    df = pd.read_csv(genes_csv)
    genes = set(df[col].dropna())
    log.info("Gene set: %d genes.", len(genes))
    return genes


def _read_and_filter(file_path: str, gene_set: frozenset[str]) -> pd.DataFrame:
    """Read one RDS file and immediately filter to *gene_set*.

    Filtering inside each worker means the full unfiltered data
    never accumulates in RAM before concatenation.
    """
    df = next(iter(pyreadr.read_r(file_path).values()))
    return df.loc[df["crispr_gene"].isin(gene_set) & df["rna_gene"].isin(gene_set)]


def load_correlations(cor_dir: Path, gene_set: set[str], n_jobs: int = 4) -> pd.DataFrame:
    """Load all *.rds Spearman files, filtering to *gene_set* per file.

    Each parallel worker reads one file and drops rows outside the gene set
    before returning, so peak RAM equals one raw file + the filtered results,
    not the entire corpus.
    """
    files = sorted(glob.glob(str(cor_dir / "*.rds")))
    if not files:
        raise FileNotFoundError(f"No *.rds files in {cor_dir}")
    log.info("Loading %d RDS files (%d workers)…", len(files), n_jobs)
    frozen = frozenset(gene_set)   # frozenset is hashable — safe for joblib
    frames = Parallel(n_jobs=n_jobs)(
        delayed(_read_and_filter)(fp, frozen) for fp in files
    )
    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d rows after per-file filtering.", len(combined))
    return combined

# ---------------------------------------------------------------------------
# RNA and CRISPR gene selection
# ---------------------------------------------------------------------------

def select_genes_by_variance(df, min_sd: float = 0.01) -> list[str]:
    df = df.rename(columns={"Unnamed: 0": "ModelID"}).set_index("ModelID")
    var_per_gene = df.std(axis=0, numeric_only=True)
    return var_per_gene[var_per_gene >= min_sd].index.tolist()

def select_active_inactive_genes(data_crispr,
                                  activity_threshhold,
                                  non_activity_threshhold):
    data_crispr = data_crispr.rename(columns={"Unnamed: 0": "ModelID"})
    data_crispr = data_crispr.set_index('ModelID')

    # Boolean masks
    dependent_mask = data_crispr < activity_threshhold
    nondependent_mask = data_crispr > non_activity_threshhold

    # Count per gene
    num_dependent = dependent_mask.sum(axis=0)
    num_nondependent = nondependent_mask.sum(axis=0)

    # Genes with at least 5 dependent AND 5 nondependent cell lines
    eligible_genes = (num_dependent >= 5) & (num_nondependent >= 5)

    # Extract list of eligible genes
    eligible_gene_list = eligible_genes[eligible_genes].index.tolist()
    return eligible_gene_list


# ---------------------------------------------------------------------------
# Dimensionality reduction + clustering
# ---------------------------------------------------------------------------

def run_pca(matrix: pd.DataFrame, n_components: int = 200):
    log.info("PCA → %d components…", n_components)
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(matrix.values)
    log.info("Variance explained: %.1f%%", pca.explained_variance_ratio_.cumsum()[-1] * 100)
    return X


def run_umap(X, n_neighbors=2, min_dist=0.005, metric="euclidean",
             n_components=2, random_state=42):
    log.info("UMAP: neighbors=%d  min_dist=%s  metric=%s…", n_neighbors, min_dist, metric)
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     n_components=n_components, metric=metric,
                     random_state=random_state).fit_transform(X)


def run_dbscan(X_umap, eps=0.05, min_samples=1):
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X_umap).labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log.info("DBSCAN: %d clusters | %d noise points.", n_clusters, (labels == -1).sum())
    return labels, n_clusters


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_clusters(gene_names, X_umap, labels, save_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame({
        "gene": gene_names,
        "UMAP1": X_umap[:, 0],
        "UMAP2": X_umap[:, 1],
        "cluster": labels.astype(str),
    })
    path = save_dir / "UMAP_with_clusters.csv"
    df.to_csv(path, index=False)
    log.info("Saved → %s", path)
    return df


def plot_cluster_histogram(labels, save_dir: Path) -> None:
    counts = pd.Series(labels).value_counts()
    fig = px.histogram(x=counts.values, nbins=50,
                       title="Cluster size distribution",
                       labels={"x": "Genes per cluster", "y": "Count"})
    pio.write_html(fig, file=str(save_dir / "hist_cluster_size.html"), auto_open=False)


def plot_umap(X_umap, gene_names, labels, save_dir: Path, tag: str = "") -> None:
    fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1],
                     color=labels.astype(str), hover_name=gene_names,
                     title="UMAP – all genes",
                     labels={"x": "UMAP1", "y": "UMAP2"})
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(showlegend=False)
    fname = f"umap_all_genes{'_' + tag if tag else ''}.html"
    pio.write_html(fig, file=str(save_dir / fname), auto_open=False)


def plot_gene_highlight(X_umap, gene_names, labels, gene: str, save_dir: Path) -> None:
    names = list(gene_names)
    if gene not in names:
        log.warning("Gene '%s' not found – skipping highlight plot.", gene)
        return
    target = labels[names.index(gene)]
    color = (labels == target).astype(str)
    fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1],
                     color=color, hover_name=names,
                     color_discrete_map={"True": "#E63946", "False": "#ADB5BD"},
                     title=f"Cluster containing {gene}")
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    pio.write_html(fig, file=str(save_dir / f"umap_cluster_of_{gene}.html"), auto_open=False)