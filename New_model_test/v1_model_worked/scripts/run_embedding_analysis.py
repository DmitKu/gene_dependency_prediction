# -*- coding: utf-8 -*-
"""
run_embedding_analysis.py
=========================
End-to-end analysis script for CRISPRSensitivityModelV3.

Workflows
---------
1. KRAS-across-cell-lines embedding + UMAP
2. Multi-gene embedding comparison
3. Integrated Gradients cluster attribution
4. Biological validation against metadata

Usage
-----
    python run_embedding_analysis.py \
        --h5_path      data/crispr_data.h5 \
        --checkpoint   checkpoints/model_best.pt \
        --metadata     data/cell_line_metadata.csv \
        --gene_id      KRAS \
        --output_dir   results/embeddings

Metadata CSV format (optional but recommended for validation)
-------------------------------------------------------------
    model_id, tissue, KRAS_mutation, TP53_status, STK11_status, ...
    ACH-000001, lung, G12C, mutant, WT, ...
"""

import argparse
import os
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — change to "TkAgg" if running interactively
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from torch.utils.data import DataLoader

# ── project imports ───────────────────────────────────────────────────────────
from utils_RNAbased_crispr_model import GeneDataset, CRISPRSensitivityModelV3
from embedding_analysis import (
    extract_gene_cellline_embeddings,
    extract_multi_gene_embeddings,
    add_umap_coords,
    validate_with_metadata,
    integrated_gradients,
    check_completeness,
    cluster_attribution_report,
    dataset_attributions,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Embedding extraction and biological validation"
    )
    p.add_argument("--h5_path",     required=True,
                   help="Path to HDF5 data file")
    p.add_argument("--checkpoint",  required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--metadata",    default=None,
                   help="CSV with cell-line annotations (optional)")
    p.add_argument("--gene_id",     default="KRAS",
                   help="Primary gene to analyse (default: KRAS)")
    p.add_argument("--extra_genes", nargs="*", default=[],
                   help="Additional genes for multi-gene comparison")
    p.add_argument("--layer",       default="cell_context2",
                   choices=["cell_context2", "x_input", "trunk"],
                   help="Model layer to extract (default: cell_context2)")
    p.add_argument("--split",       default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to analyse (default: test)")
    p.add_argument("--reduction",   default="umap",
                   choices=["umap", "pca"],
                   help="Dimensionality reduction method (default: umap)")
    p.add_argument("--output_dir",  default="results/embeddings",
                   help="Directory to save outputs")
    p.add_argument("--device",      default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto'")
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--ig_steps",    type=int, default=50,
                   help="Integrated Gradients interpolation steps")
    p.add_argument("--top_k",       type=int, default=20,
                   help="Number of top clusters to report in attribution")
    p.add_argument("--color_by",    nargs="*",
                   default=["KRAS_mutation", "tissue", "TP53_status",
                            "STK11_status"],
                   help="Metadata columns to colour UMAP by")
    return p.parse_args()


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path: str, device: str) -> CRISPRSensitivityModelV3:
    """
    Load model from checkpoint. Supports checkpoints saved as:
        - full model state dict under key 'model_state_dict'
        - raw state dict (top-level)
        - full torch.save(model) object
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, CRISPRSensitivityModelV3):
        model = ckpt
    else:
        # Extract hyperparameters if saved, else use defaults
        cfg = ckpt.get("model_config", {})
        model = CRISPRSensitivityModelV3(**cfg)

        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)

    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded  |  {n_params:,} parameters  |  device: {device}")
    return model


# ============================================================
# Plotting helpers
# ============================================================

def _scatter(ax, x, y, labels, title, cmap="tab20"):
    unique = sorted(set(str(l) for l in labels))
    colors = cm.get_cmap(cmap, len(unique))
    color_map = {l: colors(i) for i, l in enumerate(unique)}

    for label in unique:
        mask = [str(l) == label for l in labels]
        ax.scatter(
            x[mask], y[mask],
            c=[color_map[label]],
            label=label,
            s=30, alpha=0.75, edgecolors="none",
        )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.legend(
        fontsize=7, markerscale=1.5,
        bbox_to_anchor=(1.02, 1), loc="upper left",
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)


def plot_umap(
    df:       pd.DataFrame,
    color_by: list[str],
    title:    str,
    out_path: str,
):
    """
    Plot UMAP coloured by each column in color_by.
    Also always plots coloured by 'prediction' (continuous).
    """
    n_cols   = len(color_by) + 1   # +1 for prediction
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(5 * n_cols, 4.5),
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = [axes]

    x = df["umap_x"].values
    y = df["umap_y"].values

    # continuous: prediction score
    sc = axes[0].scatter(
        x, y, c=df["prediction"].values,
        cmap="coolwarm", s=30, alpha=0.8, edgecolors="none",
    )
    plt.colorbar(sc, ax=axes[0], shrink=0.8, label="Predicted score")
    axes[0].set_title(f"{title}\nPredicted CRISPR score",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("UMAP 1", fontsize=9)
    axes[0].set_ylabel("UMAP 2", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # categorical: metadata columns
    for ax, col in zip(axes[1:], color_by):
        if col in df.columns and df[col].notna().any():
            _scatter(ax, x, y, df[col].fillna("NA").values,
                     title=col, cmap="tab20")
        else:
            ax.text(0.5, 0.5, f"'{col}' not available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
            ax.set_title(col, fontsize=11)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_attribution_bar(
    report:        dict,
    gene_id:       str,
    model_id:      str,
    out_path:      str,
    top_k:         int = 20,
):
    """Horizontal bar chart of top positive and negative cluster attributions."""
    pos = report["top_positive"][:top_k]
    neg = report["top_negative"][:top_k]

    names  = [n for n, _ in pos] + [n for n, _ in neg]
    scores = [s for _, s in pos] + [s for _, s in neg]
    colors = ["#e05c5c" if s > 0 else "#5c8ae0" for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, scores, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Integrated Gradient attribution", fontsize=10)
    ax.set_title(
        f"Cluster attribution\nGene: {gene_id}  |  Cell line: {model_id}",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # ── device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── data & model ─────────────────────────────────────────────────────────
    dataset = GeneDataset(args.h5_path, split=args.split)
    model   = load_model(args.checkpoint, device)

    # ── metadata (optional) ──────────────────────────────────────────────────
    metadata = None
    if args.metadata and os.path.isfile(args.metadata):
        metadata = pd.read_csv(args.metadata)
        print(f"Metadata loaded: {metadata.shape[0]} cell lines, "
              f"columns: {list(metadata.columns)}")
    else:
        print("No metadata provided — skipping biological validation.")

    # ── training mean baseline for Integrated Gradients ──────────────────────
    # We load the training split briefly just to compute the mean cell-line vector
    print("\nComputing baseline (training mean cell-line vector) …")
    train_ds = GeneDataset(args.h5_path, split="train")
    baseline = train_ds.cl_features.mean(dim=0)   # [F]
    print(f"  Baseline shape: {tuple(baseline.shape)}")

    # ── cluster names ─────────────────────────────────────────────────────────
    # If you have named cluster labels, replace this with your list
    n_clusters    = dataset.cl_features.shape[1]
    cluster_names = [f"cluster_{i}" for i in range(n_clusters)]

    # ══════════════════════════════════════════════════════════════════════════
    # Workflow 1: single-gene cell-line embedding (e.g. KRAS)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Workflow 1 — {args.gene_id} embedding across cell lines")
    print(f"{'='*60}")

    df_gene = extract_gene_cellline_embeddings(
        model, dataset, args.gene_id, device, layer=args.layer
    )

    # add UMAP
    df_gene = add_umap_coords(df_gene, method=args.reduction)

    # merge metadata if available
    if metadata is not None:
        df_plot = validate_with_metadata(
            df_gene, metadata,
            color_by=args.color_by,
        )
    else:
        df_plot = df_gene.copy()

    # save embeddings
    emb_csv = os.path.join(args.output_dir,
                           f"{args.gene_id}_cellline_embeddings.csv")
    df_plot.to_csv(emb_csv, index=False)
    print(f"  Embeddings saved: {emb_csv}")

    # plot
    plot_umap(
        df_plot,
        color_by=[c for c in args.color_by if c in df_plot.columns],
        title=f"{args.gene_id} — cell-line embedding ({args.layer})",
        out_path=os.path.join(
            args.output_dir, f"{args.gene_id}_umap.png"
        ),
    )

    # ── prediction vs target scatter ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_gene["target"], df_gene["prediction"],
               alpha=0.5, s=20, edgecolors="none", color="#3a7abf")
    ax.set_xlabel("True CRISPR score")
    ax.set_ylabel("Predicted CRISPR score")
    ax.set_title(f"{args.gene_id} — prediction vs target")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.output_dir, f"{args.gene_id}_pred_vs_target.png"),
        dpi=150,
    )
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Workflow 2: multi-gene embedding comparison
    # ══════════════════════════════════════════════════════════════════════════
    if args.extra_genes:
        print(f"\n{'='*60}")
        print(f"Workflow 2 — multi-gene embedding comparison")
        print(f"{'='*60}")

        all_genes = [args.gene_id] + args.extra_genes
        df_multi  = extract_multi_gene_embeddings(
            model, dataset, all_genes, device, layer=args.layer
        )
        df_multi  = add_umap_coords(df_multi, method=args.reduction)

        multi_csv = os.path.join(args.output_dir, "multi_gene_embeddings.csv")
        df_multi.to_csv(multi_csv, index=False)
        print(f"  Multi-gene embeddings saved: {multi_csv}")

        # colour by gene_id to see if cell lines cluster consistently
        fig, ax = plt.subplots(figsize=(7, 5))
        for gid in df_multi["gene_id"].unique():
            sub = df_multi[df_multi["gene_id"] == gid]
            ax.scatter(sub["umap_x"], sub["umap_y"],
                       label=gid, s=20, alpha=0.6, edgecolors="none")
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1),
                  loc="upper left", frameon=False)
        ax.set_title("Multi-gene embedding — coloured by gene")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(
            os.path.join(args.output_dir, "multi_gene_umap.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved: multi_gene_umap.png")

    # ══════════════════════════════════════════════════════════════════════════
    # Workflow 3: Integrated Gradients — attribution for example samples
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Workflow 3 — Integrated Gradients cluster attribution")
    print(f"{'='*60}")

    # Find a few example samples for the target gene
    gene_indices = [
        i for i, g in enumerate(dataset.gene_ids)
        if g == args.gene_id
    ][:5]   # first 5 cell lines

    for sample_idx in gene_indices:
        gene_feat, cell_feat, target, _, gid, mid, _ = dataset[sample_idx]

        report = cluster_attribution_report(
            model, gene_feat, cell_feat, baseline,
            cluster_names, device,
            top_k=args.top_k,
            n_steps=args.ig_steps,
        )

        # completeness check
        check_completeness(
            model, gene_feat, cell_feat, baseline,
            report["attributions"], device,
        )

        print(f"\n  {gid} | {mid}")
        print(f"  Prediction attr sum: {report['attr_sum']:.4f}")
        print(f"  Top positive clusters (drove sensitivity UP):")
        for name, score in report["top_positive"][:5]:
            print(f"    {name:<20}  {score:+.4f}")
        print(f"  Top negative clusters (drove sensitivity DOWN):")
        for name, score in report["top_negative"][:5]:
            print(f"    {name:<20}  {score:+.4f}")

        # bar chart
        plot_attribution_bar(
            report, gid, mid,
            out_path=os.path.join(
                args.output_dir,
                f"attribution_{gid}_{mid.replace('/', '_')}.png"
            ),
            top_k=args.top_k,
        )

    # ── mean attribution across all samples for this gene ────────────────────
    print(f"\nComputing mean attribution across all {args.gene_id} samples …")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
    )

    # filter to target gene samples only for speed
    gene_subset_indices = [
        i for i, g in enumerate(dataset.gene_ids)
        if g == args.gene_id
    ]
    subset = torch.utils.data.Subset(dataset, gene_subset_indices)
    subset_loader = DataLoader(subset, batch_size=1, shuffle=False)

    all_attrs, all_genes, all_cells = dataset_attributions(
        model, subset_loader, baseline, device, n_steps=20,
    )

    mean_attr = all_attrs.mean(0).numpy()              # [F]
    top_idx   = np.abs(mean_attr).argsort()[::-1][:args.top_k]

    print(f"\n  Mean attribution — top {args.top_k} clusters for {args.gene_id}:")
    for i in top_idx:
        print(f"    {cluster_names[i]:<20}  {mean_attr[i]:+.4f}")

    # save mean attribution
    attr_df = pd.DataFrame({
        "cluster":     cluster_names,
        "attribution": mean_attr,
    }).sort_values("attribution", key=abs, ascending=False)

    attr_csv = os.path.join(
        args.output_dir, f"{args.gene_id}_mean_attribution.csv"
    )
    attr_df.to_csv(attr_csv, index=False)
    print(f"  Mean attribution saved: {attr_csv}")

    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()