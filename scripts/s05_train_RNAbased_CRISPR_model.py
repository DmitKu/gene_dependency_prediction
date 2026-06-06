# -*- coding: utf-8 -*-
"""
s05_train_RNAbased_CRISPR_model.py
========
Training entry-point for CRISPR Sensitivity Model.

Usage
-----
    python scripts/s05_train_RNAbased_CRISPR_model.py
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

# Enable cuDNN benchmarking for static input sizes (speed boost)
torch.backends.cudnn.benchmark = True

# ── Local imports ────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.utils_RNAbased_crispr_model import (
    GeneDataset,
    CRISPRSensitivityModelV3,
    combined_loss,
    evaluate,
    diagnose_bypass,
)

# ============================================================
# CONFIG
# ============================================================

H5_PATH          = _root / "outputs" / "H5_model_data" / "model_H5_data.h5"
TRANSFORMER_PATH = _root / "outputs" / "RNA_fetures" / "chronos_quantile_transformer.pkl"
SAVE_PATH        = _root / "outputs" / "model_training" 

EPOCHS     = 200
BATCH_SIZE = 8_192
LR         = 1e-3
PATIENCE   = 25

MODEL_KWARGS = dict(
    hidden_dim    = 128,
    gene_hidden   = 64,
    n_attn_slots  = 64,
    n_attn_heads  = 4,
    bypass_rank   = 8,
    compress_dim  = 1024, 
    dropout       = 0.2,
)

RESUME_FROM = None

CHECKPOINT_PATH     = SAVE_PATH / "crispr_checkpoint.pt"
BEST_LOSS_PATH      = SAVE_PATH / "crispr_best_model.pt"
BEST_PEARSON_PATH   = SAVE_PATH / "crispr_best_pearson_model.pt"
FINAL_WEIGHTS_PATH  = SAVE_PATH / "crispr_model_weights_final.pt"
LOG_PATH            = SAVE_PATH / "training_history.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_PATH.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data loaders
# ============================================================
def build_loaders(h5_path: Path, batch_size: int):
    train_ds = GeneDataset(h5_path, split="train")
    val_ds   = GeneDataset(h5_path, split="val")

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
# Optimizer / scheduler
# ============================================================
def build_optimizer(model: nn.Module, lr: float):
    head_and_cond_params = list(model.head.parameters()) + list(model.cond_proj.parameters())
    bypass_params = list(model.linear_bypass.parameters())
    out_scale_params = [model.out_scale, model.out_shift]

    head_and_cond_ids  = {id(p) for p in head_and_cond_params}
    bypass_ids         = {id(p) for p in bypass_params}
    out_scale_ids      = {id(p) for p in out_scale_params}

    other_params = [
        p for p in model.parameters()
        if id(p) not in head_and_cond_ids
        and id(p) not in bypass_ids
        and id(p) not in out_scale_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params,         "weight_decay": 1e-4, "lr": lr},
            {"params": head_and_cond_params, "weight_decay": 1e-6, "lr": lr},
            {"params": bypass_params,        "weight_decay": 1e-4, "lr": lr * 0.5},
            {"params": out_scale_params,     "weight_decay": 0.0,  "lr": lr * 0.5},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )
    return optimizer

def build_scheduler(optimizer, train_loader, epochs: int, lr: float):
    steps_per_epoch = len(train_loader)
    restart_epochs  = 75
    T_0             = steps_per_epoch * restart_epochs
    warmup_steps    = steps_per_epoch * 3

    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_steps],
    )

# ============================================================
# Training / validation steps
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, alpha, device, ablate_bypass=False):
    model.train()
    total_loss = total_mse = total_pearson = 0.0

    for gene_feat, cell_feat, target, cl_idx, _, _, _ in loader:
        gene_feat = gene_feat.to(device, non_blocking=True)
        cell_feat = cell_feat.to(device, non_blocking=True)
        target    = target.to(device, non_blocking=True)
        cl_idx    = cl_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) # Speed improvement
        with autocast(device_type=device):
            pred, bypass_reg = model(cell_feat, gene_feat, ablate_bypass=ablate_bypass)
            loss, mse_term, pearson_r, _ = combined_loss(
                pred, target, cl_idx, alpha=alpha, beta=0.4
            )
            loss = loss + bypass_reg

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale == scaler.get_scale():  # step was not skipped
                scheduler.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        with torch.no_grad():
            model.out_scale.clamp_(0.5, 5.0)

        total_loss    += loss.item()
        total_mse     += mse_term.item()
        total_pearson += pearson_r.item()

    n = len(loader)
    return total_loss / n, total_mse / n, total_pearson / n

# ============================================================
# Checkpoint & Logging helpers
# ============================================================
def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, best_pearson, pearson_full, pearson_full_demeaned,
                    patience_counter, alpha, transformer_path):
    ckpt = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "best_val_loss":    best_val_loss,
        "best_pearson":     best_pearson,
        "pearson":          pearson_full,
        "pearson_demeaned": pearson_full_demeaned,
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

def init_log(path: Path, resume: bool):
    mode = "a" if resume else "w"
    with open(path, mode) as f:
        if not resume:
            f.write(
                "epoch,lr,"
                "train_loss,train_mse,train_pearson_batch,"
                "val_loss,val_mse,val_pearson_batch,"
                "mae_chronos,rmse_chronos,pearson_full,pearson_full_demeaned,"
                "cl_t_mean,cl_t_sd,"
                "time_sec\n"
            )
        else:
            f.write(f"# resumed — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def log_epoch(path, epoch, lr,
              train_loss, train_mse, train_pearson,
              val_loss, val_mse, val_pearson,
              mae, rmse, pearson_full, pearson_full_demeaned,
              cl_t_mean, cl_t_std,
              epoch_time):
    with open(path, "a") as f:
        f.write(
            f"{epoch},{lr:.6e},"
            f"{train_loss:.6f},{train_mse:.6f},{train_pearson:.4f},"
            f"{val_loss:.6f},{val_mse:.6f},{val_pearson:.4f},"
            f"{mae:.4f},{rmse:.4f},{pearson_full:.4f},{pearson_full_demeaned:.4f},"
            f"{cl_t_mean:.4f},{cl_t_std:.4f},"
            f"{epoch_time:.2f}\n"
        )

def get_dynamic_alpha(epoch: int, warmup_epochs: int = 15,
                      start_alpha: float = 0.6,
                      end_alpha: float = 0.1) -> float:
    if epoch >= warmup_epochs:
        return end_alpha
    progress = epoch / warmup_epochs
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return end_alpha + (start_alpha - end_alpha) * cosine_decay

# ============================================================
# Main
# ============================================================
def main():
    ABLATE_BYPASS_EPOCHS = 10
    if not H5_PATH.exists():
        raise FileNotFoundError(f"HDF5 data file not found: {H5_PATH}")
    if not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(f"QuantileTransformer not found: {TRANSFORMER_PATH}")

    qt = joblib.load(TRANSFORMER_PATH)
    qt = None
    print(f"QuantileTransformer loaded from {TRANSFORMER_PATH.name}")
    print("  → Evaluation metrics will be reported in Chronos space")

    train_ds, val_ds, train_loader, val_loader = build_loaders(H5_PATH, BATCH_SIZE)

    model = CRISPRSensitivityModelV3(
        cell_features_size = train_ds.cl_features.shape[1],
        gene_features_size = train_ds.gene_feat.shape[1],
        **MODEL_KWARGS,
    ).to(DEVICE)

    # Initialize optimizer BEFORE torch.compile to retain standard attribute access
    optimizer = build_optimizer(model, LR)
    scheduler = build_scheduler(optimizer, train_loader, EPOCHS, LR)
    scaler    = GradScaler("cuda") if DEVICE == "cuda" else None

    # if DEVICE == "cuda":
    #     model = torch.compile(model)

    start_epoch      = 0
    best_val_loss    = float("inf")
    best_pearson     = -1.0
    patience_counter = 0

    if RESUME_FROM is not None:
        print(f"\nResuming from: {RESUME_FROM}")
        start_epoch, best_val_loss, best_pearson, patience_counter = load_checkpoint(
            RESUME_FROM, model, optimizer, scheduler, scaler, DEVICE
        )
        print(f"  → Resuming at epoch {start_epoch} | best val loss {best_val_loss:.6f}")
    else:
        print("Training from scratch.")

    init_log(LOG_PATH, resume=(RESUME_FROM is not None))

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        
        current_alpha = get_dynamic_alpha(
            epoch=epoch, 
            warmup_epochs=40,
            start_alpha=0.6, 
            end_alpha=0.2
        )

        train_loss, train_mse, train_pearson = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, current_alpha, DEVICE,
            ablate_bypass=(epoch < ABLATE_BYPASS_EPOCHS),   
        )
        
        (val_loss, val_mse, val_pearson,
         mae, rmse, pearson_full, pearson_full_demeaned,
         cl_t_mean, cl_t_std) = evaluate(
             model, val_loader, DEVICE,
             qt=qt, alpha=current_alpha,
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
            f"Pearson(demeaned) {pearson_full_demeaned:.4f} | " 
            f"cl_t {cl_t_mean:.4f} ± {cl_t_std:.4f} | "
            f"lr {current_lr:.2e} | "
            f"GPU {mem_alloc:.2f}/{mem_reserv:.2f} GB | "
            f"time {epoch_time:.1f}s | ETA {remaining/60:.1f}min"
        )

        log_epoch(
            LOG_PATH, epoch + 1, current_lr,
            train_loss, train_mse, train_pearson,
            val_loss,   val_mse,   val_pearson,
            mae, rmse, pearson_full, pearson_full_demeaned,
            cl_t_mean, cl_t_std,
            epoch_time,
        )
        
        if (epoch + 1) % 10 == 0:
            diagnose_bypass(model, val_loader, DEVICE, n_batches=5)
 
            (_, _, _,
             mae_no_bypass, _, pearson_no_bypass, pearson_full_demeaned_no_bypass,
             _, _) = evaluate(
                model, val_loader, DEVICE,
                qt=qt, alpha=current_alpha,
                ablate_bypass=True,
            )
            print(
                f"  Trunk-only (no bypass): "
                f"Pearson(demeaned) {pearson_full_demeaned_no_bypass:.4f} | "
                f"MAE {mae_no_bypass:.4f}"
            )
            print(
                f"  Bypass contribution   : "
                f"{pearson_full_demeaned - pearson_full_demeaned_no_bypass:+.4f} Pearson points"
            )
            if pearson_full_demeaned_no_bypass < 0.1:
                print("  ⚠️  WARNING: trunk learning collapsed — bypass carrying all signal")

        # ── Checkpoint on best val pearson (demeaned) ────────────────────────
        if pearson_full_demeaned > best_pearson:
            best_pearson = pearson_full_demeaned
            patience_counter = 0
            
            save_checkpoint(
                CHECKPOINT_PATH, epoch, model, optimizer, scheduler, scaler,
                val_loss, best_pearson, pearson_full, pearson_full_demeaned,
                patience_counter, current_alpha, TRANSFORMER_PATH,
            )
            torch.save(model.state_dict(), BEST_PEARSON_PATH)
            
            print(
                f"  ✓ Best Global Pearson (demeaned) {best_pearson:.4f} | "
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

    shutil.copy(BEST_PEARSON_PATH, FINAL_WEIGHTS_PATH)
    print(f"Training complete. Final weights saved to {FINAL_WEIGHTS_PATH}")

if __name__ == "__main__":
    main()