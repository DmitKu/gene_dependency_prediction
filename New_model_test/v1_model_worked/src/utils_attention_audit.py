# -*- coding: utf-8 -*-
"""
utils_attention_audit.py
========================
Helper functions for extracting and saving cross-attention weights
from CRISPRSensitivityModelV3.

Imported by scripts/s06_attention_audit.py.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

from src.utils_RNAbased_crispr_model import (
    GeneDataset,
    CRISPRSensitivityModelV3,
)

# ── Module-level capture buffer (populated by forward hook) ──────────────────
# Shape after hook fires: [B, n_attn_slots]  (float32, NumPy, on CPU)
_captured_attn: np.ndarray | None = None


# ── Hook ─────────────────────────────────────────────────────────────────────

def make_attn_hook():
    """
    Returns a forward hook for nn.MultiheadAttention.

    The model calls cross_attn2 with:
        need_weights=True, average_attn_weights=True   ← set in model forward()

    So output[1] has shape [B, tgt_len=1, src_len=n_slots].
    We squeeze the query dimension (dim 1) → [B, n_slots].

    Note: do NOT .mean(dim=1) here — the model already averaged over heads
    via average_attn_weights=True.  A second mean would operate on the query
    dimension instead, silently producing wrong numbers.
    """
    def _hook_fn(module, input, output):
        global _captured_attn
        _captured_attn = (
            output[1]           # [B, 1, n_slots]
            .squeeze(1)         # [B, n_slots]
            .detach()
            .float()            # guard against bf16 / fp16 training dtype
            .cpu()
            .numpy()
        )
    return _hook_fn


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_cl_index_to_id(dataset: GeneDataset) -> dict[int, str]:
    """Reverse the cell-line model-ID → integer-index mapping on the dataset."""
    return {idx: mid for mid, idx in dataset.cl_model_id_to_index.items()}


def load_model(
    h5_path: Path,
    weights_path: Path,
    model_kwargs: dict,
    device: str,
) -> tuple[CRISPRSensitivityModelV3, GeneDataset, dict[int, str]]:
    """
    Initialise and return:
        model        — loaded, eval-mode CRISPRSensitivityModelV3
        ref_dataset  — train-split GeneDataset (used for feature-dim inference)
        cl_idx_to_id — int index → cell-line model-ID string
    """
    ref_ds = GeneDataset(h5_path, split="train")
    cl_idx_to_id = build_cl_index_to_id(ref_ds)

    model = CRISPRSensitivityModelV3(
        cell_features_size=ref_ds.cl_features.shape[1],
        gene_features_size=ref_ds.gene_feat.shape[1],
        **model_kwargs,
    ).to(device)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, ref_ds, cl_idx_to_id


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_attention_weights(
    h5_path:      Path,
    weights_path: Path,
    output_path:  Path,
    model_kwargs: dict,
    batch_size:   int,
    device:       str,
    splits:       tuple[str, ...] = ("train", "val", "test"),
) -> int:
    """
    Extract cross-attention weights from cross_attn2 for all requested splits
    and write them to a single Parquet file at *output_path*.

    The ParquetWriter is opened lazily on the first batch so the schema is
    inferred automatically; rows are streamed to disk one batch at a time,
    keeping memory usage flat regardless of dataset size.

    Parameters
    ----------
    h5_path      : Path to the HDF5 data file.
    weights_path : Path to the saved model state-dict (.pt).
    output_path  : Destination Parquet file.
    model_kwargs : Architecture kwargs forwarded to CRISPRSensitivityModelV3.
    batch_size   : DataLoader batch size.
    device       : 'cuda' or 'cpu'.
    splits       : Which dataset splits to process.

    Returns
    -------
    total : int — total number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, _, cl_idx_to_id = load_model(h5_path, weights_path, model_kwargs, device)

    hook_handle = model.cross_attn2.register_forward_hook(make_attn_hook())
    slot_cols   = [f"Slot_{j}" for j in range(model_kwargs["n_attn_slots"])]
    pq_writer   = None
    total       = 0

    try:
        for split in splits:
            print(f"\nSplit: {split}")
            dataset = GeneDataset(h5_path, split=split)
            with h5py.File(h5_path, "r") as f:
                all_gene_ids  = f["genes/gene_id"][:]
                split_indices = f[f"index/splits/{split}"][:]
                gene_ids      = [all_gene_ids[i].decode() for i in split_indices]

            
            loader  = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = False,        # preserve order for index slicing
                num_workers = 4,
                pin_memory  = True,
            )

            with torch.no_grad():
                for batch_idx, (g_feat, c_feat, _target, cl_idx, sample_idx) in enumerate(   # ← 4 values
                    tqdm(loader, desc=f"  {split}")
                ):
                    model(c_feat.to(device), g_feat.to(device))

                    cl_ints = cl_idx.numpy()

                    df = pd.DataFrame(_captured_attn, columns=slot_cols)
                    df["Gene_ID"]    = [gene_ids[int(i)] for i in sample_idx.numpy()]      # ← positional slice
                    df["Cell_ID"]    = [cl_idx_to_id.get(int(i), f"unk_{i}") for i in cl_ints]
                    df["Cell_Index"] = cl_ints.astype(np.int32)
                    df["Split"]      = split

                    table = pa.Table.from_pandas(df, preserve_index=False)

                    if pq_writer is None:
                        pq_writer = pq.ParquetWriter(output_path, table.schema)
                    pq_writer.write_table(table)
                    total += len(df)

            print(f"  → {len(dataset):,} rows written")

    finally:
        hook_handle.remove()
        if pq_writer is not None:
            pq_writer.close()

    return total