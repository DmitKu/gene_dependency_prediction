# -*- coding: utf-8 -*-
"""
embedding_analysis.py
=====================
Cell-line embedding extraction and biological validation utilities.

Extracts intermediate representations from CRISPRSensitivityModelV3
for downstream dimensionality reduction and multi-sample analysis.

Typical use
-----------
    from embedding_analysis import (
        extract_gene_cellline_embeddings,
        reduce_embeddings,
        validate_with_metadata,
        integrated_gradients,
        cluster_attribution_report,
    )

Layer guide
-----------
    cell_context2  [B, gene_hidden]   — cell features filtered through gene query.
                                        Best for cell-line clustering per gene.
    x_input        [B, gene_hidden*2] — joint gene × cell representation.
                                        Best for (gene, cell) pair analysis.
    trunk          [B, hidden_dim]    — pre-head, most predictive.
                                        Best for sensitivity-stratified clustering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional

# ── optional heavy deps (warn instead of hard-fail) ──────────────────────────
try:
    from umap import UMAP
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP reduction unavailable. "
          "Install with: pip install umap-learn")

try:
    from sklearn.decomposition import PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. PCA unavailable. "
          "Install with: pip install scikit-learn")

try:
    from scipy.stats import kruskal, pearsonr, spearmanr
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Statistical tests unavailable. "
          "Install with: pip install scipy")


# ============================================================
# Internal forward helper
# ============================================================

def _forward_to_layers(
    model:      nn.Module,
    gene_feat:  torch.Tensor,   # [B, G]
    cell_feat:  torch.Tensor,   # [B, F]
) -> dict[str, torch.Tensor]:
    """
    Run the model forward pass and return all intermediate tensors.
    No gradients — inference only.

    Returns
    -------
    dict with keys:
        gene_emb      : [B, gene_hidden]
        cell_tokens   : [B, n_slots, gene_hidden]
        cell_context  : [B, gene_hidden]   after 1st cross-attn
        cell_context2 : [B, gene_hidden]   after 2nd cross-attn
        attn_w1       : [B, n_slots]       1st attention weights
        attn_w2       : [B, n_slots]       2nd attention weights
        x_input       : [B, gene_hidden*2] cat[cell_context2, gene_emb]
        trunk_out     : [B, hidden_dim]    after trunk res blocks
        prediction    : [B, 1]            final output
    """
    gene_emb    = model.gene_res(model.gene_encoder(gene_feat))
    cell_tokens = model.cell_tokenizer(cell_feat, gene_emb)

    attn_out1, attn_w1 = model.cross_attn(
        gene_emb.unsqueeze(1), cell_tokens, cell_tokens,
        need_weights=True, average_attn_weights=True,
    )
    cell_context = model.attn_norm(attn_out1.squeeze(1) + gene_emb)

    attn_out2, attn_w2 = model.cross_attn2(
        cell_context.unsqueeze(1), cell_tokens, cell_tokens,
        need_weights=True, average_attn_weights=True,
    )
    cell_context2 = model.attn_norm2(attn_out2.squeeze(1) + cell_context)

    x_input   = torch.cat([cell_context2, gene_emb], dim=-1)
    x         = model.merge(x_input)
    gene_cond = model.cond_proj(torch.cat([gene_emb, cell_context2], dim=-1))
    x         = model.trunk_res1(x, cond=gene_cond)
    x         = model.trunk_res2(x, cond=gene_cond)
    x         = model.trunk_res3(x, cond=gene_cond)

    # bypass for full prediction (not stored separately — just for output)
    cell_summary1 = (attn_w1.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
    cell_summary2 = (attn_w2.squeeze(1).unsqueeze(-1) * cell_tokens).sum(dim=1)
    cell_summary  = 0.5 * cell_summary1 + 0.5 * cell_summary2
    bypass_logit, _ = model.linear_bypass(gene_emb, cell_summary)

    trunk_pred  = model.head(x)
    prediction  = (trunk_pred + bypass_logit) * model.out_scale + model.out_shift

    return {
        "gene_emb":      gene_emb,
        "cell_tokens":   cell_tokens,
        "cell_context":  cell_context,
        "cell_context2": cell_context2,
        "attn_w1":       attn_w1.squeeze(1),    # [B, n_slots]
        "attn_w2":       attn_w2.squeeze(1),    # [B, n_slots]
        "x_input":       x_input,
        "trunk_out":     x,
        "prediction":    prediction,
    }


# ============================================================
# Single-gene / single-cell-line embedding extraction
# ============================================================

def extract_gene_cellline_embeddings(
    model:        nn.Module,
    dataset,                        # GeneDataset instance
    gene_id:      str,              # e.g. "KRAS"
    device:       str = "cpu",
    layer:        str = "cell_context2",
) -> pd.DataFrame:
    """
    For a single gene, extract embeddings for every cell line
    that has a prediction for that gene.

    Parameters
    ----------
    model      : trained CRISPRSensitivityModelV3
    dataset    : GeneDataset (any split)
    gene_id    : gene identifier string matching dataset.gene_ids
    device     : torch device string
    layer      : one of 'cell_context2', 'x_input', 'trunk'

    Returns
    -------
    pd.DataFrame with columns:
        model_id, gene_id, prediction, target, emb_0 … emb_N
    """
    assert layer in ("cell_context2", "x_input", "trunk"), \
        f"layer must be one of 'cell_context2', 'x_input', 'trunk'; got '{layer}'"

    indices = [i for i, g in enumerate(dataset.gene_ids) if g == gene_id]
    if not indices:
        raise ValueError(
            f"Gene '{gene_id}' not found in dataset. "
            f"Sample of available genes: {dataset.gene_ids[:5]}"
        )

    print(f"Extracting '{layer}' embeddings for {gene_id} "
          f"across {len(indices)} cell lines …")

    model.eval()
    records = []

    with torch.no_grad():
        for idx in indices:
            gene_feat, cell_feat, target, cl_idx, gid, mid, _ = dataset[idx]

            gene_feat = gene_feat.to(device).unsqueeze(0)
            cell_feat = cell_feat.to(device).unsqueeze(0)

            layers = _forward_to_layers(model, gene_feat, cell_feat)

            layer_map = {
                "cell_context2": "cell_context2",
                "x_input":       "x_input",
                "trunk":         "trunk_out",
            }
            emb = layers[layer_map[layer]].squeeze(0).cpu().numpy()

            records.append({
                "model_id":   mid,
                "gene_id":    gid,
                "prediction": layers["prediction"].item(),
                "target":     target.item(),
                **{f"emb_{i}": float(emb[i]) for i in range(len(emb))},
            })

    df = pd.DataFrame(records)
    print(f"  → {len(df)} samples | embedding dim: {len(emb)}")
    return df


# ============================================================
# Multi-gene extraction
# ============================================================

def extract_multi_gene_embeddings(
    model:    nn.Module,
    dataset,
    gene_ids: list[str],
    device:   str = "cpu",
    layer:    str = "cell_context2",
) -> pd.DataFrame:
    """
    Extract embeddings for multiple genes and stack into one DataFrame.
    Each row is one (gene_id, model_id) pair.

    Useful for asking: do the same cell lines cluster together
    regardless of which gene is queried?
    """
    dfs = []
    for gid in gene_ids:
        try:
            df = extract_gene_cellline_embeddings(
                model, dataset, gid, device, layer
            )
            dfs.append(df)
        except ValueError as e:
            print(f"  Skipping {gid}: {e}")

    if not dfs:
        raise RuntimeError("No valid gene embeddings extracted.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal samples extracted: {len(combined)} "
          f"| Genes: {combined['gene_id'].nunique()} "
          f"| Cell lines: {combined['model_id'].nunique()}")
    return combined


# ============================================================
# Aggregation
# ============================================================

def aggregate_embeddings(
    df:    pd.DataFrame,
    by:    str = "model_id",
    agg:   str = "mean",
) -> tuple[np.ndarray, list[str]]:
    """
    Aggregate sample-level embeddings to cell-line or gene level.

    Parameters
    ----------
    df  : output of extract_gene_cellline_embeddings
    by  : 'model_id' → one vector per cell line
          'gene_id'  → one vector per gene
    agg : 'mean' | 'median' | 'max'

    Returns
    -------
    matrix : [n_unique, dim]  numpy array
    keys   : list of group labels aligned with matrix rows
    """
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")],
                      key=lambda c: int(c.split("_")[1]))

    agg_fn = {"mean": "mean", "median": "median", "max": "max"}[agg]
    grouped = df.groupby(by)[emb_cols].agg(agg_fn)

    return grouped.values.astype(np.float32), list(grouped.index)


# ============================================================
# Dimensionality reduction
# ============================================================

def reduce_embeddings(
    matrix:     np.ndarray,
    method:     str   = "umap",
    pca_first:  bool  = True,
    pca_dim:    int   = 50,
    umap_neighbors: int   = 15,
    umap_min_dist:  float = 0.1,
    random_state:   int   = 42,
) -> np.ndarray:
    """
    Reduce embedding matrix to 2D for plotting.

    Parameters
    ----------
    matrix        : [N, dim] embedding array
    method        : 'umap' | 'pca'
    pca_first     : run PCA to pca_dim before UMAP (recommended for dim > 50)
    pca_dim       : intermediate PCA dimension
    umap_neighbors: UMAP n_neighbors parameter
    umap_min_dist : UMAP min_dist parameter

    Returns
    -------
    coords : [N, 2] 2D coordinates
    """
    assert _SKLEARN_AVAILABLE, "scikit-learn required for PCA"

    X = matrix.copy()

    if pca_first and X.shape[1] > pca_dim:
        X = PCA(n_components=min(pca_dim, X.shape[0] - 1),
                random_state=random_state).fit_transform(X)

    if method == "umap":
        assert _UMAP_AVAILABLE, "umap-learn required for UMAP reduction"
        coords = UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=min(umap_neighbors, len(X) - 1),
            min_dist=umap_min_dist,
            random_state=random_state,
        ).fit_transform(X)

    elif method == "pca":
        coords = PCA(n_components=2,
                     random_state=random_state).fit_transform(X)

    else:
        raise ValueError(f"method must be 'umap' or 'pca', got '{method}'")

    return coords.astype(np.float32)


# ============================================================
# Biological validation
# ============================================================

def validate_with_metadata(
    df:         pd.DataFrame,
    metadata:   pd.DataFrame,
    color_by:   list[str],
    merge_on:   str = "model_id",
) -> pd.DataFrame:
    """
    Merge embedding DataFrame with cell-line metadata and report
    how well known biological stratifiers align with the embedding.

    Parameters
    ----------
    df        : output of extract_gene_cellline_embeddings (with umap_x, umap_y)
    metadata  : cell-line annotation table with merge_on column
    color_by  : list of metadata columns to test (e.g. ['KRAS_mutation', 'tissue'])
    merge_on  : join key — must exist in both df and metadata

    Returns
    -------
    merged DataFrame ready for plotting
    """
    assert _SCIPY_AVAILABLE, "scipy required for statistical tests"

    merged = df.merge(metadata, on=merge_on, how="left")

    print("\n── Biological alignment ────────────────────────────────────")

    for col in color_by:
        if col not in merged.columns:
            print(f"  {col:<25} not found in metadata — skipping")
            continue

        valid = merged[["umap_x", col]].dropna()
        if valid.empty:
            print(f"  {col:<25} no valid data after dropna")
            continue

        groups = [
            valid.loc[valid[col] == val, "umap_x"].values
            for val in valid[col].unique()
            if (valid[col] == val).sum() > 1
        ]

        if len(groups) < 2:
            print(f"  {col:<25} insufficient groups for test")
            continue

        stat, p = kruskal(*groups)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {col:<25} Kruskal-Wallis p={p:.4f}  {sig}")

    # prediction vs target Pearson
    if "target" in merged.columns and "prediction" in merged.columns:
        valid = merged[["prediction", "target"]].dropna()
        if not valid.empty:
            r, p = pearsonr(valid["prediction"], valid["target"])
            print(f"\n  Pred vs target Pearson  r={r:.3f}  p={p:.4f}")
            r_s, p_s = spearmanr(valid["prediction"], valid["target"])
            print(f"  Pred vs target Spearman r={r_s:.3f}  p={p_s:.4f}")

    print("────────────────────────────────────────────────────────────")
    return merged


# ============================================================
# Integrated Gradients — feature-level attribution
# ============================================================

def integrated_gradients(
    model:      nn.Module,
    gene_feat:  torch.Tensor,   # [G]
    cell_feat:  torch.Tensor,   # [F]
    baseline:   torch.Tensor,   # [F]  e.g. training mean cell line
    device:     str,
    n_steps:    int = 50,
) -> torch.Tensor:              # [F]
    """
    Integrated Gradients attribution over cell features.

    Returns a signed [F] vector:
        positive → cluster pushed prediction toward higher dependency
        negative → cluster pushed prediction toward lower dependency

    Completeness axiom: attributions.sum() ≈ pred(cell_feat) - pred(baseline)

    Parameters
    ----------
    baseline : meaningful reference point.
               Recommended: training set mean cell-line vector.
               Zero vector is acceptable but less informative biologically.
    n_steps  : integration steps. 50 is usually sufficient; increase to 100
               if completeness check fails.
    """
    model.eval()

    gene_feat = gene_feat.to(device).unsqueeze(0)
    cell_feat = cell_feat.to(device).unsqueeze(0)
    baseline  = baseline.to(device).unsqueeze(0)

    alphas = torch.linspace(0, 1, n_steps, device=device)         # [n_steps]
    interp = baseline + alphas[:, None] * (cell_feat - baseline)  # [n_steps, F]
    interp = interp.detach().requires_grad_(True)

    gene_exp  = gene_feat.expand(n_steps, -1)
    preds, _  = model(interp, gene_exp)

    grads = torch.autograd.grad(preds.sum(), interp)[0]            # [n_steps, F]

    avg_grads    = grads.mean(dim=0)                               # [F]
    attributions = avg_grads * (cell_feat - baseline).squeeze(0)   # [F]

    return attributions.detach()


def check_completeness(
    model:        nn.Module,
    gene_feat:    torch.Tensor,
    cell_feat:    torch.Tensor,
    baseline:     torch.Tensor,
    attributions: torch.Tensor,
    device:       str,
    tol:          float = 0.05,
) -> float:
    """
    Verify the completeness axiom: sum(attrs) ≈ pred(input) - pred(baseline).
    Returns the absolute gap; prints a pass/fail message.
    """
    model.eval()
    with torch.no_grad():
        pred_in, _  = model(
            cell_feat.to(device).unsqueeze(0),
            gene_feat.to(device).unsqueeze(0),
        )
        pred_bl, _  = model(
            baseline.to(device).unsqueeze(0),
            gene_feat.to(device).unsqueeze(0),
        )

    attr_sum = attributions.sum().item()
    expected = (pred_in - pred_bl).item()
    gap      = abs(attr_sum - expected)

    status = "✓ OK" if gap < tol else f"✗ gap={gap:.4f} — increase n_steps"
    print(f"Completeness  attr_sum={attr_sum:.4f}  "
          f"pred_diff={expected:.4f}  {status}")
    return gap


def cluster_attribution_report(
    model:         nn.Module,
    gene_feat:     torch.Tensor,
    cell_feat:     torch.Tensor,
    baseline:      torch.Tensor,
    cluster_names: list[str],
    device:        str,
    top_k:         int = 20,
    n_steps:       int = 50,
) -> dict:
    """
    Full per-sample attribution report over cell clusters.

    Returns
    -------
    dict with keys:
        attributions  : [F] full attribution vector
        top_positive  : list of (cluster_name, score) — drove sensitivity up
        top_negative  : list of (cluster_name, score) — drove sensitivity down
        attr_sum      : float — should ≈ pred - baseline_pred
    """
    attrs = integrated_gradients(
        model, gene_feat, cell_feat, baseline, device, n_steps
    )

    top_pos = attrs.topk(top_k)
    top_neg = (-attrs).topk(top_k)

    return {
        "attributions": attrs,
        "top_positive": [
            (cluster_names[i], attrs[i].item())
            for i in top_pos.indices
        ],
        "top_negative": [
            (cluster_names[i], attrs[i].item())
            for i in top_neg.indices
        ],
        "attr_sum": attrs.sum().item(),
    }


def dataset_attributions(
    model:         nn.Module,
    loader,
    baseline:      torch.Tensor,
    device:        str,
    n_steps:       int = 20,
) -> tuple[torch.Tensor, list[str], list[str]]:
    """
    Compute Integrated Gradients for every sample in a DataLoader.

    Parameters
    ----------
    baseline : [F] reference cell-line vector (training mean recommended)
    n_steps  : integration steps (20 is fast; use 50 for publication figures)

    Returns
    -------
    all_attrs : [N, F]  attribution matrix
    gene_ids  : [N]     gene identifier per row
    model_ids : [N]     cell-line identifier per row
    """
    model.eval()
    all_attrs, all_genes, all_cells = [], [], []

    for gene_feat, cell_feat, _, _, gene_ids, model_ids, _ in loader:
        for i in range(len(gene_feat)):
            attr = integrated_gradients(
                model, gene_feat[i], cell_feat[i],
                baseline, device, n_steps,
            )
            all_attrs.append(attr)
            all_genes.append(gene_ids[i])
            all_cells.append(model_ids[i])

    return torch.stack(all_attrs), all_genes, all_cells


# ============================================================
# Convenience: add UMAP coords to DataFrame
# ============================================================

def add_umap_coords(
    df:     pd.DataFrame,
    method: str = "umap",
    by:     str | None = None,
    **reduce_kwargs,
) -> pd.DataFrame:
    """
    Reduce embedding columns in df to 2D and add umap_x / umap_y columns.

    Parameters
    ----------
    df     : DataFrame with emb_0 … emb_N columns
    method : 'umap' | 'pca'
    by     : if provided, aggregate by this column before reduction
             then merge coords back to original df rows
    """
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")],
                      key=lambda c: int(c.split("_")[1]))

    if by is not None:
        matrix, keys = aggregate_embeddings(df, by=by)
        coords = reduce_embeddings(matrix, method=method, **reduce_kwargs)
        coord_df = pd.DataFrame({
            by:       keys,
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
        })
        return df.merge(coord_df, on=by, how="left")

    matrix = df[emb_cols].values.astype(np.float32)
    coords = reduce_embeddings(matrix, method=method, **reduce_kwargs)
    df = df.copy()
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]
    return df