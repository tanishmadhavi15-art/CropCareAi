"""
CropCare AI — Training engine with GPU support, mixed precision,
early stopping, checkpointing, and resumable training.
"""

import os
import sys
import json
import time
import math
import random
import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODELS_DIR, FULL_DATASET_MODELS_DIR, REPORTS_DIR, DATA_RAW_DIR, TRAINING_LOG_PATH,
    NUM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX, TARGET_CLASSES,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    RANDOM_SEED, NUM_WORKERS, IMG_SIZE, EARLY_STOP_PATIENCE,
    BEST_MODEL_PATH,
)
from src.models import build_model, count_parameters, ARCHITECTURES
from src.dataset import get_dataloaders, compute_class_weights


# ──────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Slight speed trade-off for reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True   # faster on fixed input size


# ──────────────────────────────────────────────────────
# Device selection — always print selected device
# ──────────────────────────────────────────────────────

def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] Selected: {device}", end="")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f" | GPU: {props.name} | VRAM: {vram_gb:.2f} GB")
    else:
        print(" | (CPU mode — GPU not available or forced off)")
    return device


# ──────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────

def compute_macro_f1(preds: torch.Tensor, targets: torch.Tensor,
                     num_classes: int = NUM_CLASSES) -> float:
    """Compute macro-averaged F1 from prediction/target tensors."""
    f1_scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        f1_scores.append(f1)
    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


# ──────────────────────────────────────────────────────
# One epoch — train
# ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                    use_amp: bool, scheduler=None) -> Dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        try:
            if use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("\n[OOM] CUDA out of memory — skipping batch. "
                      "Consider reducing batch size.")
                continue
            raise

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1 = compute_macro_f1(all_preds, all_targets)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_f1": f1,
    }


# ──────────────────────────────────────────────────────
# One epoch — evaluate
# ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp: bool) -> Dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    f1 = compute_macro_f1(all_preds, all_targets)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "macro_f1": f1,
        "all_preds": all_preds,
        "all_targets": all_targets,
    }


# ──────────────────────────────────────────────────────
# GPU memory report
# ──────────────────────────────────────────────────────

def gpu_memory_stats() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return f"alloc={alloc:.0f}MB / reserved={reserved:.0f}MB / peak={peak:.0f}MB"


# ──────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────

def train(
    arch_name: str = "CropCareNet_M",
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    use_class_weights: bool = True,
    use_weighted_sampler: bool = False,
    data_dir: str = DATA_RAW_DIR,
    checkpoint_dir: str = FULL_DATASET_MODELS_DIR,
    resume_from: Optional[str] = None,
    force_cpu: bool = False,
    num_workers: int = NUM_WORKERS,
    smoke_test: bool = False,
) -> Dict:
    """
    Full training pipeline. Returns a results dict.

    Args:
        arch_name: One of CropCareNet_S, CropCareNet_M, CropCareNet_L
        smoke_test: If True, run only 2 mini-epochs with 2 batches each (CI/CD)
    """
    set_seed(RANDOM_SEED)
    device = select_device(force_cpu)
    use_amp = (device.type == "cuda")  # mixed precision on GPU only

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Data ──
    print(f"\n[Data] Loading dataset from {data_dir} ...")
    train_loader, val_loader, test_loader, data_info = get_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train: {data_info['n_train']} | Val: {data_info['n_val']} "
          f"| Test: {data_info['n_test']}")
    print(f"  Class imbalance ratio: {data_info['imbalance_ratio']}")
    weight_samples = train_loader.dataset.samples

    if smoke_test:
        # Subset to 2 batches for quick iteration
        from itertools import islice
        print("[Smoke] Running in smoke-test mode (2 batches).")
        train_loader = list(islice(train_loader, 2))
        val_loader   = list(islice(val_loader,   2))
        smoke_train_count = sum(batch_y.numel() for _, batch_y in train_loader)
        smoke_val_count = sum(batch_y.numel() for _, batch_y in val_loader)
        print(
            f"[Smoke] Metrics use {smoke_train_count} training and "
            f"{smoke_val_count} validation examples, not the full splits."
        )
        num_epochs = 2
        early_stop_patience = 99

    # ── Model ──
    print(f"\n[Model] Building {arch_name} ...")
    model = build_model(arch_name, num_classes=NUM_CLASSES).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")
    if device.type == "cuda":
        print(f"  GPU memory after model load: {gpu_memory_stats()}")

    # ── Loss ──
    if use_class_weights:
        cw = compute_class_weights(weight_samples, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        print(f"  Class weights: {cw.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Optimizer & Scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1,         # 10% warmup
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = GradScaler(enabled=use_amp)

    # ── Resume ──
    start_epoch = 0
    best_val_f1 = -1.0
    best_epoch = 0
    history = []

    if resume_from and os.path.isfile(resume_from):
        print(f"\n[Resume] Loading checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        checkpoint_classes = ckpt.get("run_config", {}).get("target_classes")
        if checkpoint_classes != TARGET_CLASSES:
            raise ValueError(
                "Checkpoint class mapping does not match the current verified dataset. "
                "Start a new full-dataset run instead of resuming this checkpoint."
            )
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"  Resumed from epoch {start_epoch}, best val F1: {best_val_f1:.4f}")

    # ── Config snapshot ──
    run_config = {
        "arch": arch_name,
        "img_size": img_size,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stop_patience": early_stop_patience,
        "use_class_weights": use_class_weights,
        "use_weighted_sampler": use_weighted_sampler,
        "use_amp": use_amp,
        "device": str(device),
        "n_params": n_params,
        "data_info": data_info,
        "random_seed": RANDOM_SEED,
        "target_classes": TARGET_CLASSES,
    }
    print(f"\n[Config] {json.dumps({k: v for k, v in run_config.items() if k != 'data_info'}, indent=2)}")

    # ── Training Loop ──
    best_model_path = os.path.join(checkpoint_dir, f"best_{arch_name}.pth")
    last_model_path = os.path.join(checkpoint_dir, f"last_{arch_name}.pth")
    patience_counter = 0
    training_start = time.time()

    print(f"\n{'='*60}")
    print(f"Training {arch_name} | {num_epochs} epochs | device={device} | AMP={use_amp}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp,
            scheduler=scheduler,
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)

        epoch_time = time.time() - epoch_start
        val_f1 = val_metrics["macro_f1"]

        # Current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Memory stats
        mem_str = gpu_memory_stats() if device.type == "cuda" else "N/A"

        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.4f} F1={train_metrics['macro_f1']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.4f} F1={val_f1:.4f} | "
            f"LR={current_lr:.6f} | {epoch_time:.1f}s | {mem_str}",
            flush=True
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_metrics['loss'], 5),
            "train_acc":  round(train_metrics['acc'], 5),
            "train_f1":   round(train_metrics['macro_f1'], 5),
            "val_loss":   round(val_metrics['loss'], 5),
            "val_acc":    round(val_metrics['acc'], 5),
            "val_f1":     round(val_f1, 5),
            "lr":         round(current_lr, 8),
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(epoch_record)

        # Checkpoint (every epoch for resumability)
        ckpt_state = {
            "epoch": epoch,
            "arch": arch_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
            "run_config": run_config,
        }
        torch.save(ckpt_state, last_model_path)

        # Best model tracking (based on val macro-F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            ckpt_state["best_val_f1"] = best_val_f1
            torch.save(ckpt_state, best_model_path)
            print(f"  *** New best val F1: {best_val_f1:.4f} - checkpoint saved ***", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n[EarlyStop] No improvement for {early_stop_patience} epochs. "
                      f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}.")
                break

    total_time = time.time() - training_start

    # ── Save training log ──
    run_config["total_training_time_s"] = round(total_time, 1)
    run_config["best_val_f1"] = round(best_val_f1, 5)
    run_config["best_epoch"] = best_epoch
    run_config["history"] = history

    log_path = os.path.join(REPORTS_DIR, f"training_{arch_name}.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2)
    print(f"\n[Done] Training time: {total_time:.1f}s | Best val F1: {best_val_f1:.4f}")
    print(f"[Saved] Training log: {log_path}")
    print(f"[Saved] Best model: {best_model_path}")

    return run_config


# ──────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CropCare AI CNN")
    parser.add_argument("--arch", default="CropCareNet_M",
                        choices=list(ARCHITECTURES.keys()))
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--no-class-weights", action="store_false", dest="class_weights",
                        help="Disable train-derived square-root inverse-frequency loss weights")
    parser.set_defaults(class_weights=True)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--data-dir", default=DATA_RAW_DIR)
    parser.add_argument("--checkpoint-dir", default=FULL_DATASET_MODELS_DIR)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick 2-epoch test with 2 batches")
    args = parser.parse_args()

    train(
        arch_name=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.patience,
        use_class_weights=args.class_weights,
        use_weighted_sampler=args.weighted_sampler,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        force_cpu=args.cpu,
        num_workers=args.workers,
        smoke_test=args.smoke_test,
    )
