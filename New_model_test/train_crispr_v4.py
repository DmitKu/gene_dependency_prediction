# -*- coding: utf-8 -*-
"""
scripts/train_crispr_v4.py
==========================
Training entry-point for CRISPR Sensitivity Model v4.

Usage
-----
    python scripts/train_crispr_v4.py
"""

import sys
import time
import math
import shutil
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

# ── Local imports ─────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.utils_crispr_model_v4 import (
    GeneDataset,
    CRISPRModel,
    combined_loss,
    evaluate,
    diagnose_model,
)

# ============================================================
# CONFIG  — edit here, nowhere else
# ============================================================

H5_PATH          = _root / "outputs" / "H5_model_data"   / "model_H5_data.h5"
TRANSFORMER_PATH = _root / "outputs" / "RNA_features"    / "chronos_quantile_transformer.pkl"
SAVE_PATH        = _root / "outputs" / "model_training_v4"

EPOCHS     = 200
BATCH_SIZE = 8_192
LR         = 1e-3
PATIENCE   = 30          # increased — v4 takes longer to converge initially

# Model hyperparameters — passed directly to CRISPRModel
MODEL_KWARGS = dict(
    rank          = 128,
    n_heads       = 8,
    n_cell_layers = 4,
    dropout       = 0.2,
    # n_clusters, gene_feat_size, cell_feat_size derived from data at runtime
)

# Ablate bypass for first N epochs: forces trunk+FiLM to learn
# interactions before the bilinear shortcut can dominate.
# Not needed in v4 (no bypass), but kept as a training philosophy note.
# v4 has no bypass — bilinear is part of the main path.

RESUME_FROM = None  # set to CHECKPOINT_PATH to resume

CHECKPOINT_PATH    = SAVE_PATH / "crispr_v4_checkpoint.pt"
BEST_PEARSON_PATH  = SAVE_PATH / "crispr_v4_best_pearson.pt"
FINAL_WEIGHTS_PATH = SAVE_PATH / "crispr_v4_final.pt"
LOG_PATH           = SAVE_PATH / "training_history_v4.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data loaders
# ============================================================

def build_loaders(h5_path: Path, batch_size: int):
    train_ds = GeneDataset(h5_path, split="train")
    val_ds   = GeneDataset(h5_path, split="val")

    # num_workers=4 with persistent_workers and prefetch_factor=2
    # gives near-zero data loading overhead for in-RAM datasets.
    loader_kwargs = dict(
        batch_size         = batch_size,
        num_workers        = 4,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 2,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_ds, val_ds, train_loader, val_loader


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Parameter groups with differentiated learning rates:

    cluster_emb : lower LR + higher weight decay
        Embeddings for 2388 clusters. Many clusters may be sparse.
        Slower learning rate prevents sparse cluster embeddings from
        overfitting to few observations.

    bilinear    : lower LR
        Interaction term. We want it to grow slowly so the FiLM-
        conditioned cell encoder learns first.

    gene_bias / cell_bias : very low LR
        Main effect terms. Zero-initialised. Let them grow slowly
        so the interaction head gets the signal first.

    everything else : base LR
        Cell encoder, gene stat encoder, gate, FiLM blocks, head.
    """
    cluster_emb_params = list(model.gene_cluster_emb.parameters())
    bilinear_params    = list(model.bilinear_u.parameters()) + \
                         list(model.bilinear_v.parameters())
    main_eff_params    = list(model.gene_bias.parameters()) + \
                         list(model.cell_bias.parameters())

    cluster_emb_ids = {id(p) for p in cluster_emb_params}
    bilinear_ids    = {id(p) for p in bilinear_params}
    main_eff_ids    = {id(p) for p in main_eff_params}

    other_params = [
        p for p in model.parameters()
        if id(p) not in cluster_emb_ids
        and id(p) not in bilinear_ids
        and id(p) not in main_eff_ids
    ]

    return torch.optim.AdamW(
        [
            {"params": other_params,       "lr": lr,         "weight_decay": 1e-4},
            {"params": cluster_emb_params, "lr": lr * 0.3,   "weight_decay": 1e-3},
            {"params": bilinear_params,    "lr": lr * 0.3,   "weight_decay": 1e-4},
            {"params": main_eff_params,    "lr": lr * 0.1,   "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(optimizer, train_loader, epochs: int):
    steps_per_epoch = len(train_loader)
    warmup_steps    = steps_per_epoch * 5       # 5-epoch linear warmup
    restart_epochs  = 50                         # cosine restart period
    T_0             = steps_per_epoch * restart_epochs

    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


# ============================================================
# Dynamic alpha schedule
# ============================================================

def get_dynamic_alpha(
    epoch:         int,
    warmup_epochs: int   = 30,
    start_alpha:   float = 0.7,   # early: weight MSE heavily (stable gradients)
    end_alpha:     float = 0.2,   # late:  shift weight to Pearson terms
) -> float:
    """
    Cosine decay of MSE weight over warmup_epochs, then holds at end_alpha.

    Early training: high MSE weight keeps gradients stable while the
    model learns to predict the right scale of CRISPR scores.
    Later training: lower MSE weight, higher Pearson weight, which
    encourages learning gene ranking within cell lines (interactions).
    """
    if epoch >= warmup_epochs:
        return end_alpha
    progress = epoch / warmup_epochs
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end_alpha + (start_alpha - end_alpha) * cosine_decay


# ============================================================
# Training step
# ============================================================

def train_one_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    alpha:     float,
    beta:      float,
    device:    str,
) -> tuple[float, float, float]:
    model.train()
    total_loss = total_mse = total_pearson = 0.0
    device_type = torch.device(device).type

    for batch in loader:
        (gene_feat, cell_feat, target,
         cl_idx, gene_cluster_id, _, _, _) = batch

        gene_feat       = gene_feat.to(device,       non_blocking=True)
        cell_feat       = cell_feat.to(device,       non_blocking=True)
        target          = target.to(device,          non_blocking=True)
        cl_idx          = cl_idx.to(device,          non_blocking=True)
        gene_cluster_id = gene_cluster_id.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type):
            pred = model(cell_feat, gene_feat, gene_cluster_id)
            loss, mse_term, pearson_r, _ = combined_loss(
                pred, target, cl_idx, alpha=alpha, beta=beta
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pre_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only step scheduler if scaler didn't skip the update
            # (skipped updates happen when gradients overflow in fp16)
            if scaler.get_scale() == pre_scale:
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss    += loss.item()
        total_mse     += mse_term.item()
        total_pearson += pearson_r.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_pearson / n


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_pearson, pearson_full, pearson_full_demeaned,
                    patience_counter, alpha, beta):
    ckpt = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "best_pearson":     best_pearson,
        "pearson":          pearson_full,
        "pearson_demeaned": pearson_full_demeaned,
        "patience_counter": patience_counter,
        "loss_config":      {"alpha": alpha, "beta": beta},
    }
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt.get("best_pearson",     -1.0),
        ckpt.get("patience_counter",  0),
    )


# ============================================================
# Logging
# ============================================================

def init_log(path: Path, resume: bool):
    with open(path, "a" if resume else "w") as f:
        if not resume:
            f.write(
                "epoch,lr,"
                "train_loss,train_mse,train_pearson_batch,"
                "val_loss,val_mse,val_pearson_batch,"
                "mae,rmse,pearson_full,pearson_demeaned,"
                "cl_pearson_mean,cl_pearson_std,"
                "alpha,beta,time_sec\n"
            )
        else:
            f.write(f"# resumed — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def log_epoch(path, epoch, lr,
              train_loss, train_mse, train_pearson,
              val_loss, val_mse, val_pearson,
              mae, rmse, pearson_full, pearson_demeaned,
              cl_mean, cl_std, alpha, beta, epoch_time):
    with open(path, "a") as f:
        f.write(
            f"{epoch},{lr:.6e},"
            f"{train_loss:.6f},{train_mse:.6f},{train_pearson:.4f},"
            f"{val_loss:.6f},{val_mse:.6f},{val_pearson:.4f},"
            f"{mae:.4f},{rmse:.4f},{pearson_full:.4f},{pearson_demeaned:.4f},"
            f"{cl_mean:.4f},{cl_std:.4f},"
            f"{alpha:.3f},{beta:.3f},{epoch_time:.2f}\n"
        )


# ============================================================
# Main
# ============================================================

def main():
    if not H5_PATH.exists():
        raise FileNotFoundError(f"HDF5 not found: {H5_PATH}")
    if not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(f"QuantileTransformer not found: {TRANSFORMER_PATH}")

    qt = joblib.load(TRANSFORMER_PATH)
    print(f"QuantileTransformer loaded from {TRANSFORMER_PATH.name}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, train_loader, val_loader = build_loaders(H5_PATH, BATCH_SIZE)

    # Derive dimensions from data — never hardcode
    n_clusters     = int(train_ds.gene_cluster_ids.max().item()) + 1
    gene_feat_size = train_ds.gene_feat.shape[1]
    cell_feat_size = train_ds.cl_features.shape[1]

    print(f"\nModel config:")
    print(f"  n_clusters    = {n_clusters}")
    print(f"  gene_feat_size = {gene_feat_size}")
    print(f"  cell_feat_size = {cell_feat_size}")
    print(f"  rank           = {MODEL_KWARGS['rank']}")
    print(f"  n_cell_layers  = {MODEL_KWARGS['n_cell_layers']}")

    assert cell_feat_size == n_clusters, (
        f"cell_feat_size ({cell_feat_size}) != n_clusters ({n_clusters}). "
        f"Cell features must be one RNA sum per cluster."
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = CRISPRModel(
        n_clusters     = n_clusters,
        gene_feat_size = gene_feat_size,
        cell_feat_size = cell_feat_size,
        **MODEL_KWARGS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params = {n_params:,}")

    # ── Optimizer and scheduler ───────────────────────────────────────────
    optimizer = build_optimizer(model, LR)
    scheduler = build_scheduler(optimizer, train_loader, EPOCHS)
    scaler    = GradScaler("cuda") if DEVICE == "cuda" else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch      = 0
    best_pearson     = -1.0
    patience_counter = 0

    if RESUME_FROM is not None and Path(RESUME_FROM).exists():
        print(f"\nResuming from: {RESUME_FROM}")
        start_epoch, best_pearson, patience_counter = load_checkpoint(
            RESUME_FROM, model, optimizer, scheduler, scaler, DEVICE
        )
        print(f"  → epoch {start_epoch} | best demeaned Pearson {best_pearson:.4f}")
    else:
        print("\nTraining from scratch.")

    init_log(LOG_PATH, resume=(RESUME_FROM is not None and Path(str(RESUME_FROM)).exists()))

    # ── Training loop ─────────────────────────────────────────────────────
    BETA = 0.4   # fixed — demeaned Pearson weight

    for epoch in range(start_epoch, EPOCHS):
        epoch_start  = time.time()
        current_alpha = get_dynamic_alpha(epoch, warmup_epochs=30,
                                          start_alpha=0.7, end_alpha=0.2)

        # Train
        train_loss, train_mse, train_pearson = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            alpha=current_alpha, beta=BETA, device=DEVICE,
        )

        # Validate
        (val_loss, val_mse, val_pearson,
         mae, rmse,
         pearson_full, pearson_full_demeaned,
         cl_mean, cl_std) = evaluate(
            model, val_loader, DEVICE,
            qt=qt, alpha=current_alpha, beta=BETA,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        remaining  = (EPOCHS - epoch - 1) * epoch_time

        mem_alloc  = torch.cuda.memory_allocated(0)  / 1024**3 if torch.cuda.is_available() else 0
        mem_reserv = torch.cuda.memory_reserved(0)   / 1024**3 if torch.cuda.is_available() else 0

        print(
            f"Epoch {epoch+1:03d} | "
            f"α={current_alpha:.2f} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"mse {train_mse:.5f}/{val_mse:.5f} | "
            f"pearson(batch) {train_pearson:.4f}/{val_pearson:.4f} | "
            f"MAE {mae:.4f} RMSE {rmse:.4f} | "
            f"Pearson(global) {pearson_full:.4f} | "
            f"Pearson(demeaned) {pearson_full_demeaned:.4f} | "
            f"cl_t {cl_mean:.4f}±{cl_std:.4f} | "
            f"lr {current_lr:.2e} | "
            f"GPU {mem_alloc:.1f}/{mem_reserv:.1f}GB | "
            f"{epoch_time:.1f}s ETA {remaining/60:.0f}min"
        )

        log_epoch(
            LOG_PATH, epoch + 1, current_lr,
            train_loss, train_mse, train_pearson,
            val_loss,   val_mse,   val_pearson,
            mae, rmse, pearson_full, pearson_full_demeaned,
            cl_mean, cl_std, current_alpha, BETA, epoch_time,
        )

        # ── Diagnostics every 10 epochs ───────────────────────────────────
        if (epoch + 1) % 10 == 0:
            diagnose_model(model, val_loader, DEVICE, n_batches=5)

        # ── Checkpoint on best demeaned Pearson ───────────────────────────
        if pearson_full_demeaned > best_pearson:
            best_pearson     = pearson_full_demeaned
            patience_counter = 0

            save_checkpoint(
                CHECKPOINT_PATH, epoch, model, optimizer, scheduler, scaler,
                best_pearson, pearson_full, pearson_full_demeaned,
                patience_counter, current_alpha, BETA,
            )
            torch.save(model.state_dict(), BEST_PEARSON_PATH)

            print(
                f"  ✓ New best demeaned Pearson {best_pearson:.4f} | "
                f"global {pearson_full:.4f} | "
                f"cl_t {cl_mean:.4f}±{cl_std:.4f} | "
                f"MAE {mae:.4f} — saved"
            )
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    # ── Final save ────────────────────────────────────────────────────────
    shutil.copy(BEST_PEARSON_PATH, FINAL_WEIGHTS_PATH)
    print(f"\nTraining complete. Best demeaned Pearson: {best_pearson:.4f}")
    print(f"Final weights: {FINAL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()