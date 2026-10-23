"""Train torchvision MobileNetV3-Large on the verified PlantVillage mapping."""

import argparse
import json
import os
import random
import sys
import time
from itertools import islice
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    BATCH_SIZE,
    DATA_RAW_DIR,
    EARLY_STOP_PATIENCE,
    MOBILENET_BATCH_SIZE,
    MOBILENET_EPOCHS,
    MOBILENET_FINETUNE_LR,
    MOBILENET_HEAD_EPOCHS,
    MOBILENET_HEAD_LR,
    MOBILENET_IMG_SIZE,
    MOBILENET_MODELS_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_CLASSES,
    WEIGHT_DECAY,
)
from src.dataset import compute_class_weights, get_dataloaders
from src.mobilenet import MODEL_NAME, build_mobilenet_v3_large


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _set_trainable_stage(model: nn.Module, stage: str) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = stage == "finetune"
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def _make_optimizer(model: nn.Module, stage: str, weight_decay: float):
    if stage == "head":
        return AdamW(model.classifier.parameters(), lr=MOBILENET_HEAD_LR, weight_decay=weight_decay)
    return AdamW(
        [
            {"params": model.features.parameters(), "lr": MOBILENET_FINETUNE_LR},
            {"params": model.classifier.parameters(), "lr": MOBILENET_FINETUNE_LR * 5},
        ],
        weight_decay=weight_decay,
    )


def _run_batches(
    model: nn.Module,
    batches: Iterable,
    optimizer,
    criterion,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    training: bool,
) -> Dict:
    model.train(training)
    if training and not any(parameter.requires_grad for parameter in model.features.parameters()):
        model.features.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for images, labels in batches:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

        loss_sum += loss.detach().float().item() * images.size(0)
        predictions = logits.detach().argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)
        all_preds.append(predictions.cpu())
        all_targets.append(labels.cpu())

    if total == 0:
        raise RuntimeError("No batches were processed; check the selected split and batch size.")

    predictions = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    f1_scores = []
    for class_idx in range(NUM_CLASSES):
        true_positive = ((predictions == class_idx) & (targets == class_idx)).sum().item()
        false_positive = ((predictions == class_idx) & (targets != class_idx)).sum().item()
        false_negative = ((predictions != class_idx) & (targets == class_idx)).sum().item()
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1_scores.append(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "loss": loss_sum / total,
        "acc": correct / total,
        "macro_f1": sum(f1_scores) / NUM_CLASSES,
        "samples_seen": total,
    }


def train(
    data_dir: str = DATA_RAW_DIR,
    batch_size: int = MOBILENET_BATCH_SIZE,
    num_epochs: int = MOBILENET_EPOCHS,
    head_epochs: int = MOBILENET_HEAD_EPOCHS,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_dir: str = MOBILENET_MODELS_DIR,
    report_path: Optional[str] = None,
    resume_from: Optional[str] = None,
    num_workers: int = 0,
    smoke_test: bool = False,
) -> Dict:
    """Train head first, then fine-tune all layers; select by validation macro-F1."""
    if num_epochs < 1 or not 0 <= head_epochs < num_epochs:
        raise ValueError("Require num_epochs >= 1 and 0 <= head_epochs < num_epochs.")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        print(f"[Device] {torch.cuda.get_device_name(0)} | AMP enabled")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_loader, val_loader, _, data_info = get_dataloaders(
        data_dir,
        img_size=MOBILENET_IMG_SIZE,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_amp,
        transform_profile="mobilenet_v3",
    )
    train_samples = train_loader.dataset.samples

    if smoke_test:
        train_loader = list(islice(train_loader, 2))
        val_loader = list(islice(val_loader, 2))
        train_seen = sum(labels.numel() for _, labels in train_loader)
        val_seen = sum(labels.numel() for _, labels in val_loader)
        print(f"[Smoke] Limited diagnostic: {train_seen} train and {val_seen} validation samples.")
        num_epochs = 2
        head_epochs = 1

    model = build_mobilenet_v3_large(NUM_CLASSES, pretrained=True).to(device)
    class_weights = compute_class_weights(train_samples, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_f1 = -1.0
    best_epoch = 0
    history = []
    resume_checkpoint = None
    if resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from}")
        resume_checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        saved_classes = resume_checkpoint.get("class_mapping", {}).get("classes")
        if saved_classes != TARGET_CLASSES:
            raise ValueError("Resume checkpoint class mapping does not match the current mapping.")
        model.load_state_dict(resume_checkpoint["model_state"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state", {}))
        start_epoch = resume_checkpoint["epoch"] + 1
        best_val_f1 = resume_checkpoint.get("best_val_f1", -1.0)
        best_epoch = resume_checkpoint.get("best_epoch", 0)
        history = resume_checkpoint.get("history", [])

    run_config = {
        "model": MODEL_NAME,
        "weights": "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        "pretrained": True,
        "fine_tuning": "classifier head first, then full backbone and head",
        "num_classes": NUM_CLASSES,
        "target_classes": TARGET_CLASSES,
        "img_size": MOBILENET_IMG_SIZE,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "head_epochs": head_epochs,
        "head_lr": MOBILENET_HEAD_LR,
        "finetune_backbone_lr": MOBILENET_FINETUNE_LR,
        "finetune_classifier_lr": MOBILENET_FINETUNE_LR * 5,
        "weight_decay": weight_decay,
        "use_amp": use_amp,
        "device": str(device),
        "class_weights": class_weights.detach().cpu().tolist(),
        "data_info": data_info,
        "random_seed": RANDOM_SEED,
        "smoke_test": smoke_test,
    }

    best_path = os.path.join(checkpoint_dir, f"best_{MODEL_NAME}.pth")
    last_path = os.path.join(checkpoint_dir, f"last_{MODEL_NAME}.pth")
    stage = "head" if start_epoch < head_epochs else "finetune"
    _set_trainable_stage(model, stage)
    optimizer = _make_optimizer(model, stage, weight_decay)
    stage_epochs = max(1, head_epochs if stage == "head" else num_epochs - head_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=stage_epochs, eta_min=1e-6)

    if resume_checkpoint and resume_checkpoint.get("stage") == stage:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if resume_checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(resume_checkpoint["scheduler_state"])

    stale_epochs = 0
    training_start = time.time()
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        current_stage = "head" if epoch < head_epochs else "finetune"
        if current_stage != stage:
            stage = current_stage
            _set_trainable_stage(model, stage)
            optimizer = _make_optimizer(model, stage, weight_decay)
            stage_epochs = max(1, num_epochs - head_epochs)
            scheduler = CosineAnnealingLR(optimizer, T_max=stage_epochs, eta_min=1e-6)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        epoch_lrs = [group["lr"] for group in optimizer.param_groups]
        train_metrics = _run_batches(
            model, train_loader, optimizer, criterion, device, scaler, use_amp, training=True
        )
        val_metrics = _run_batches(
            model, val_loader, optimizer, criterion, device, scaler, use_amp, training=False
        )
        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = max(epoch_lrs)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6 if use_amp else None

        record = {
            "epoch": epoch + 1,
            "stage": stage,
            "train_loss": round(train_metrics["loss"], 6),
            "train_acc": round(train_metrics["acc"], 6),
            "train_macro_f1": round(train_metrics["macro_f1"], 6),
            "train_samples_seen": train_metrics["samples_seen"],
            "val_loss": round(val_metrics["loss"], 6),
            "val_acc": round(val_metrics["acc"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_f1": round(val_metrics["macro_f1"], 6),
            "val_samples_seen": val_metrics["samples_seen"],
            "lr_max": current_lr,
            "lr_max_next_epoch": max(group["lr"] for group in optimizer.param_groups),
            "elapsed_seconds": round(epoch_time, 2),
            "peak_cuda_memory_mb": round(peak_memory_mb, 1) if peak_memory_mb is not None else None,
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1}/{num_epochs} [{stage}] "
            f"train loss={record['train_loss']:.4f} acc={record['train_acc']:.4f} "
            f"macro-F1={record['train_macro_f1']:.4f} ({record['train_samples_seen']} samples) | "
            f"val loss={record['val_loss']:.4f} acc={record['val_acc']:.4f} "
            f"macro-F1={record['val_macro_f1']:.4f} ({record['val_samples_seen']} samples) | "
            f"lr-max={current_lr:.2g} time={epoch_time:.1f}s peak-VRAM={peak_memory_mb}MB"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1

        checkpoint = {
            "arch": MODEL_NAME,
            "epoch": epoch,
            "stage": stage,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "history": history,
            "class_mapping": {"num_classes": NUM_CLASSES, "classes": TARGET_CLASSES},
            "run_config": run_config,
        }
        torch.save(checkpoint, last_path)
        if best_epoch == epoch + 1:
            torch.save(checkpoint, best_path)
        if stale_epochs >= EARLY_STOP_PATIENCE and epoch + 1 >= head_epochs + 1:
            print(f"[EarlyStop] Validation macro-F1 did not improve for {stale_epochs} epochs.")
            break

    result = {
        **run_config,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "best_val_macro_f1": best_val_f1,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "total_training_time_s": round(time.time() - training_start, 2),
        "history": history,
    }
    report_path = report_path or os.path.join(REPORTS_DIR, f"training_{MODEL_NAME}.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(result, report_file, indent=2)
    print(f"[Saved] Best checkpoint: {best_path}")
    print(f"[Saved] Training report: {report_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained MobileNetV3-Large on PlantVillage")
    parser.add_argument("--data-dir", default=DATA_RAW_DIR)
    parser.add_argument("--batch-size", type=int, default=MOBILENET_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=MOBILENET_EPOCHS)
    parser.add_argument("--head-epochs", type=int, default=MOBILENET_HEAD_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--checkpoint-dir", default=MOBILENET_MODELS_DIR)
    parser.add_argument("--report", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true", help="Run two batches per split for pipeline validation")
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        head_epochs=args.head_epochs,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        report_path=args.report,
        resume_from=args.resume,
        num_workers=args.workers,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
