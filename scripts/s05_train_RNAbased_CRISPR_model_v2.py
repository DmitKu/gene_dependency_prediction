# -*- coding: utf-8 -*-
"""
train.py
========
Training entry-point for CRISPR Sensitivity Model v3.

Usage
-----
    python scripts/train.py

All hyper-parameters are defined in the CONFIG section below.
Outputs written to the working directory:
    crispr_checkpoint_v3.pt          — full checkpoint (resume-compatible)
    crispr_best_model_v3.pt          — best val-loss model weights
    crispr_best_pearson_model_v3.pt  — best per-CL Pearson model weights
    crispr_model_weights_v3_final.pt — final weights after training
    training_history_v3.csv          — per-epoch metrics log
"""

import sys
import time
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import shutil
import math

# ── Local imports ────────────────────────────────────────────────────────────
# Ensure the project root (parent of scripts/) is on the path so that
# `src.crispr_model` resolves regardless of where the script is launched from.
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))


from src.RNAbased_crispr_model import (
    GeneDataset,
    CRISPRSensitivityModelV3,
    combined_loss,
    evaluate,
)


# ============================================================
# CONFIG
# ============================================================

_root = Path(
    r"C:\Users\dkuch\Documents\Blog_ideas_data\Computational"
    r"\MOA_Prediction_based_on_CETSA\20251122_Model_development"
    r"\GitHub_GeneDependancy_prediction"
)

H5_PATH          = _root/"outputs"/"H5_model_data"/"model_H5_data.h5"
TRANSFORMER_PATH = _root/"outputs"/"RNA_fetures"/"chronos_quantile_transformer.pkl"
SAVE_PATH = _root /"outputs"/"model_training" 

# Training
EPOCHS     = 200
BATCH_SIZE = 8_192
LR         = 3e-3
#ALPHA      = 0.5       # weight between MSE (ALPHA) and Pearson (1-ALPHA)
PATIENCE   = 10

# Model architecture
MODEL_KWARGS = dict(
    hidden_dim    = 128,
    n_attn_slots  = 64,
    n_attn_heads  = 4,
    bypass_rank   = 32,
    compress_dim  = 512,
    dropout       = 0.2,
)

# Set to a checkpoint path (str/Path) to resume training, else None
RESUME_FROM = None

# Output paths
CHECKPOINT_PATH     = SAVE_PATH/"crispr_checkpoint.pt"
BEST_LOSS_PATH      = SAVE_PATH/"crispr_best_model.pt"
BEST_PEARSON_PATH   = SAVE_PATH/"crispr_best_pearson_model.pt"
FINAL_WEIGHTS_PATH  = SAVE_PATH/"crispr_model_weights_final.pt"
LOG_PATH            = SAVE_PATH/"training_history .csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data loaders
# ============================================================
SAVE_PATH.mkdir(parents=True, exist_ok=True)



def build_loaders(h5_path: Path, batch_size: int):
    """Construct train and validation DataLoaders."""
    train_ds = GeneDataset(h5_path, split="train")
    val_ds   = GeneDataset(h5_path, split="val")

    loader_kwargs = dict(
        batch_size       = batch_size,
        num_workers      = 4,
        pin_memory       = True,
        persistent_workers = True,
        prefetch_factor  = 2,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_ds, val_ds, train_loader, val_loader


# ============================================================
# Optimizer / scheduler
# ============================================================

def build_optimizer(model: nn.Module, lr: float):
    """
    AdamW with differential weight decay.

    head and cond_proj use a much lower weight decay so they remain free
    to produce large outputs for extreme CRISPR dependencies.
    """
    head_and_cond_params = (
        list(model.head.parameters()) +
        list(model.cond_proj.parameters())
    )
    head_and_cond_ids = {id(p) for p in head_and_cond_params}
    other_params      = [p for p in model.parameters()
                         if id(p) not in head_and_cond_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,         "weight_decay": 1e-4},
            {"params": head_and_cond_params,  "weight_decay": 1e-6},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )
    return optimizer


# def build_scheduler(optimizer, train_loader, epochs: int, lr: float):
#     return torch.optim.lr_scheduler.OneCycleLR(
#         optimizer,
#         max_lr          = lr,
#         steps_per_epoch = len(train_loader),
#         epochs          = epochs,
#         pct_start       = 0.05,
#         anneal_strategy = "cos",
#     )

def build_scheduler(optimizer, train_loader, epochs: int, lr: float):
    steps_per_epoch = len(train_loader)
    
    # Restart the learning rate every 50 epochs
    restart_epochs = 75
    T_0 = steps_per_epoch * restart_epochs
    
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,         # Steps until the first restart
        T_mult=1,        # Keep the restart interval constant (1) or double it each time (2)
        eta_min=1e-6     # Minimum learning rate at the bottom of the cycle
    )

# ============================================================
# Training / validation steps
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, scaler, alpha, device):
    """Run one full training epoch. Returns (loss, mse, pearson) averages."""
    model.train()
    total_loss = total_mse = total_pearson = 0.0

    for gene_feat, cell_feat, target, _ in loader:
        gene_feat = gene_feat.to(device, non_blocking=True)
        cell_feat = cell_feat.to(device, non_blocking=True)
        target    = target.to(device,    non_blocking=True)

        optimizer.zero_grad()
        with autocast(device):
            pred = model(cell_feat, gene_feat)
            loss, mse_term, pearson_r = combined_loss(pred, target, alpha=alpha)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == scale_before:   # step was not skipped
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


@torch.no_grad()
def validate_one_epoch(model, loader, alpha, device):
    """Run one full validation epoch. Returns (loss, mse, pearson) averages."""
    model.eval()
    total_loss = total_mse = total_pearson = 0.0

    for gene_feat, cell_feat, target, _ in loader:
        gene_feat = gene_feat.to(device, non_blocking=True)
        cell_feat = cell_feat.to(device, non_blocking=True)
        target    = target.to(device,    non_blocking=True)
        with autocast(device):
            pred = model(cell_feat, gene_feat)
            loss, mse_term, pearson_r = combined_loss(pred, target, alpha=alpha)
        total_loss    += loss.item()
        total_mse     += mse_term.item()
        total_pearson += pearson_r.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_pearson / n


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, best_pearson, pearson_full, pearson_pcl,
                    patience_counter, alpha, transformer_path):
    ckpt = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "best_val_loss":    best_val_loss,
        "best_pearson":     best_pearson,
        "pearson":          pearson_full,
        "pearson_per_cl":   pearson_pcl,
        "patience_counter": patience_counter,
        "loss_config":      {"alpha": alpha},
        "transformer_path": str(transformer_path),
    }
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_loss"],
        ckpt.get("best_pearson", -1.0),
        ckpt.get("patience_counter", 0),
    )


# ============================================================
# Logging helpers
# ============================================================

def init_log(path: Path, resume: bool):
    mode = "a" if resume else "w"
    with open(path, mode) as f:
        if not resume:
            f.write(
                "epoch,lr,"
                "train_loss,train_mse,train_pearson_batch,"
                "val_loss,val_mse,val_pearson_batch,"
                "mae_chronos,rmse_chronos,pearson_full,"
                "pearson_per_cl_mean,pearson_per_cl_sd,"
                "time_sec\n"
            )
        else:
            f.write(f"# resumed — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def log_epoch(path, epoch, lr,
              train_loss, train_mse, train_pearson,
              val_loss, val_mse, val_pearson,
              mae, rmse, pearson_full, pearson_pcl, pearson_pcl_sd,
              epoch_time):
    with open(path, "a") as f:
        f.write(
            f"{epoch},{lr:.6e},"
            f"{train_loss:.6f},{train_mse:.6f},{train_pearson:.4f},"
            f"{val_loss:.6f},{val_mse:.6f},{val_pearson:.4f},"
            f"{mae:.4f},{rmse:.4f},{pearson_full:.4f},"
            f"{pearson_pcl:.4f},{pearson_pcl_sd:.4f},"
            f"{epoch_time:.2f}\n"
        )


# ============================================================
# Main
# ============================================================
# def get_dynamic_alpha(epoch: int, warmup_epochs: int = 30, start_alpha: float = 0.8, end_alpha: float = 0.2) -> float:
#     """Calculates the current alpha based on the epoch."""
#     if epoch >= warmup_epochs:
#         return end_alpha
#     decay_rate = (start_alpha - end_alpha) / warmup_epochs
#     return start_alpha - (decay_rate * epoch)


def get_dynamic_alpha(epoch: int, warmup_epochs: int = 30, start_alpha: float = 0.8, end_alpha: float = 0.2) -> float:
    if epoch >= warmup_epochs:
        return end_alpha
    # Cosine transition
    progress = epoch / warmup_epochs
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return end_alpha + (start_alpha - end_alpha) * cosine_decay

def main():
    # ── Validate paths ───────────────────────────────────────────────────────
    if not H5_PATH.exists():
        raise FileNotFoundError(f"HDF5 data file not found: {H5_PATH}")
    if not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(
            f"QuantileTransformer not found: {TRANSFORMER_PATH}\n"
            "Re-run s3_feature_engineering.py to generate it."
        )

    # ── Load QuantileTransformer ─────────────────────────────────────────────
    qt = joblib.load(TRANSFORMER_PATH)
    print(f"QuantileTransformer loaded from {TRANSFORMER_PATH.name}")
    print("  → Evaluation metrics will be reported in Chronos space")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, train_loader, val_loader = build_loaders(H5_PATH, BATCH_SIZE)

    # ── Model ────────────────────────────────────────────────────────────────
    model = CRISPRSensitivityModelV3(
        cell_features_size = train_ds.cl_features.shape[1],
        gene_features_size = train_ds.gene_feat.shape[1],
        **MODEL_KWARGS,
    ).to(DEVICE)

    # ── Optimizer / scheduler / scaler ───────────────────────────────────────
    optimizer = build_optimizer(model, LR)
    scheduler = build_scheduler(optimizer, train_loader, EPOCHS, LR)
    scaler    = GradScaler("cuda") if DEVICE == "cuda" else None

    # ── Resume or initialise fresh ───────────────────────────────────────────
    start_epoch      = 0
    best_val_loss    = float("inf")
    best_pearson     = -1.0
    best_pearson_pcl = -float('inf')
    patience_counter = 0

    if RESUME_FROM is not None:
        print(f"\nResuming from: {RESUME_FROM}")
        start_epoch, best_val_loss, best_pearson_pcl, patience_counter = load_checkpoint(
            RESUME_FROM, model, optimizer, scheduler, scaler, DEVICE
        )
        print(f"  → Resuming at epoch {start_epoch} | best val loss {best_val_loss:.6f}")
    else:
        train_mean = train_ds.crispr.mean().item()
        model.head[-1].bias.data.fill_(train_mean)
        print(f"Output bias initialised to train mean: {train_mean:.4f}")
        print("Training from scratch.")

    # ── Print run summary ────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nCUDA available   : {torch.cuda.is_available()}")
    print(f"Device           : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Trainable params : {total_params:,}")
    #print(f"Loss             : {ALPHA:.1f} × MSE  +  {1-ALPHA:.1f} × (1-Pearson)  +  0.1 × std_match")
    print(
        f"Architecture     : cross-attention "
        f"({model.cell_tokenizer.n_slots} slots, "
        f"attn-weighted bypass, combined FiLM cond) "
        f"+ bilinear bypass (rank {model.linear_bypass.gene_proj.out_features})"
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    init_log(LOG_PATH, resume=(RESUME_FROM is not None))

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        # 1. Calculate the dynamic alpha for this epoch
        current_alpha = get_dynamic_alpha(
            epoch=epoch, 
            warmup_epochs=30, # Adjust based on when you want the transition to end
            start_alpha=0.8, 
            end_alpha=0.2
        )

        train_loss, train_mse, train_pearson = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, current_alpha, DEVICE
        )
        val_loss, val_mse, val_pearson = validate_one_epoch(
            model, val_loader, current_alpha, DEVICE
        )
        mae, rmse, pearson_full, pearson_pcl, pearson_pcl_sd = evaluate(
            model, val_loader, DEVICE, qt=qt
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        remaining  = (EPOCHS - epoch - 1) * epoch_time

        mem_alloc  = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        mem_reserv = torch.cuda.memory_reserved(0)  / 1024**3 if torch.cuda.is_available() else 0

        print(
            f"Epoch {epoch+1:03d} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"mse {train_mse:.5f}/{val_mse:.5f} | "
            f"pearson(batch) {train_pearson:.4f}/{val_pearson:.4f} | "
            f"MAE {mae:.4f} | RMSE {rmse:.4f} | "
            f"Pearson(global) {pearson_full:.4f} | "
            f"Pearson(per-CL) {pearson_pcl:.4f} ± {pearson_pcl_sd:.4f} | "
            f"lr {current_lr:.2e} | "
            f"GPU {mem_alloc:.2f}/{mem_reserv:.2f} GB | "
            f"time {epoch_time:.1f}s | ETA {remaining/60:.1f}min"
        )

        log_epoch(
            LOG_PATH, epoch + 1, current_lr,
            train_loss, train_mse, train_pearson,
            val_loss,   val_mse,   val_pearson,
            mae, rmse, pearson_full, pearson_pcl, pearson_pcl_sd,
            epoch_time,
        )

        # ── Checkpoint on best val loss ──────────────────────────────────────
        if pearson_pcl > best_pearson_pcl:
            best_pearson_pcl = pearson_pcl
            patience_counter = 0
            
            save_checkpoint(
                CHECKPOINT_PATH, epoch, model, optimizer, scheduler, scaler,
                val_loss, best_pearson_pcl, pearson_full, pearson_pcl,
                patience_counter, current_alpha, TRANSFORMER_PATH,
            )
            torch.save(model.state_dict(), BEST_PEARSON_PATH)
            
            print(
                f"  ✓ Best per-CL Pearson {best_pearson_pcl:.4f} | "
                f"MAE {mae:.4f} | RMSE {rmse:.4f} | "
                f"Pearson(global) {pearson_full:.4f} | "
                f"Val Loss {val_loss:.6f} (Alpha: {current_alpha:.2f}) — saved"
            )
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

       
    # ── Final save ───────────────────────────────────────────────────────────
    shutil.copy(BEST_PEARSON_PATH, FINAL_WEIGHTS_PATH)
    print(f"Training complete. Final weights saved to {FINAL_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()