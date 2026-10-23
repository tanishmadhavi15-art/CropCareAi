"""
CropCare AI — Evaluation script.

Loads the best checkpoint and evaluates on the held-out test set.
Produces: accuracy, macro precision/recall/F1, per-class metrics,
confusion matrix, and saves a full JSON evaluation report.
Never fabricates results.
"""

import os
import sys
import json
import time

import torch
import torch.nn as nn
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    REPORTS_DIR, DATA_RAW_DIR, NUM_CLASSES,
    IDX_TO_CLASS, TARGET_CLASSES, IMG_SIZE, MOBILENET_IMG_SIZE, BATCH_SIZE,
    MOBILENET_BATCH_SIZE, NUM_WORKERS,
)
from src.models import build_model
from src.mobilenet import MODEL_NAME, build_mobilenet_v3_large
from src.dataset import get_dataloaders
from src.train import select_device


def compute_full_metrics(preds: torch.Tensor,
                         targets: torch.Tensor,
                         num_classes: int = NUM_CLASSES) -> dict:
    """Compute accuracy, per-class and macro precision/recall/F1."""
    total = len(targets)
    correct = (preds == targets).sum().item()
    overall_acc = correct / total if total > 0 else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        tn = ((preds != c) & (targets != c)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        support   = tp + fn

        class_name = IDX_TO_CLASS.get(c, str(c))
        per_class[class_name] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   int(support),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall    = sum(recalls)    / len(recalls)
    macro_f1        = sum(f1s)        / len(f1s)

    # Weighted averages (by support)
    supports = [per_class[IDX_TO_CLASS[c]]["support"] for c in range(num_classes)]
    total_support = sum(supports) + 1e-8
    weighted_precision = sum(p * s for p, s in zip(precisions, supports)) / total_support
    weighted_recall    = sum(r * s for r, s in zip(recalls, supports))    / total_support
    weighted_f1        = sum(f * s for f, s in zip(f1s, supports))        / total_support

    return {
        "overall_accuracy": round(overall_acc, 4),
        "macro_precision":  round(macro_precision, 4),
        "macro_recall":     round(macro_recall, 4),
        "macro_f1":         round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall":    round(weighted_recall, 4),
        "weighted_f1":        round(weighted_f1, 4),
        "per_class": per_class,
        "n_total": total,
        "n_correct": correct,
    }


def compute_confusion_matrix(preds: torch.Tensor,
                             targets: torch.Tensor,
                             num_classes: int = NUM_CLASSES) -> list:
    """Return confusion matrix as list-of-lists (rows=true, cols=pred)."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t][p] += 1
    return cm


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str = DATA_RAW_DIR,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    force_cpu: bool = False,
    report_path: str = None,
) -> dict:
    """
    Load checkpoint, run inference on the test split,
    and return/save a full evaluation report.
    """
    device = select_device(force_cpu)
    use_amp = device.type == "cuda"

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[Eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_classes = ckpt.get("run_config", {}).get("target_classes")
    if checkpoint_classes != TARGET_CLASSES:
        raise ValueError(
            "Checkpoint class mapping does not match the current verified dataset. "
            "Evaluate only a checkpoint trained with the current class mapping."
        )
    arch_name = ckpt.get("arch", "CropCareNet_M")
    print(f"  Architecture: {arch_name}")
    print(f"  Trained to epoch: {ckpt.get('epoch', '?') + 1}")
    print(f"  Best val F1 (training): {ckpt.get('best_val_f1', '?')}")

    is_mobilenet = arch_name == MODEL_NAME
    if is_mobilenet:
        checkpoint_mapping = ckpt.get("class_mapping", {}).get("classes")
        if checkpoint_mapping != TARGET_CLASSES:
            raise ValueError("MobileNet checkpoint mapping does not match the configured 38 classes.")
        model = build_mobilenet_v3_large(NUM_CLASSES, pretrained=False).to(device)
        img_size = MOBILENET_IMG_SIZE
    else:
        model = build_model(arch_name, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"\n[Data] Loading test split ...")
    _, _, test_loader, data_info = get_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        transform_profile="mobilenet_v3" if is_mobilenet else "custom",
    )
    print(f"  Test samples: {data_info['n_test']}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    all_preds = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0
    start = time.time()

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y.cpu())
        total_samples += batch_x.size(0)

    eval_time = time.time() - start
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_full_metrics(all_preds, all_targets)
    cm = compute_confusion_matrix(all_preds, all_targets)

    # Pretty print results
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION RESULTS - {arch_name}")
    print(f"{'='*60}")
    print(f"  Samples:           {total_samples}")
    print(f"  Overall Accuracy:  {metrics['overall_accuracy']:.4f} "
          f"({metrics['n_correct']}/{metrics['n_total']})")
    print(f"  Macro Precision:   {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:      {metrics['macro_recall']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"  Test Loss:         {total_loss/max(total_samples,1):.4f}")
    print(f"  Eval Time:         {eval_time:.2f}s")
    print()
    print("  Per-class metrics:")
    print(f"  {'Class':<30} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    print(f"  {'-'*54}")
    for cls_name, m in metrics['per_class'].items():
        print(f"  {cls_name:<30} {m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['f1']:>6.4f} {m['support']:>6d}")

    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = [IDX_TO_CLASS[i][:10] for i in range(NUM_CLASSES)]
    print(f"  {'':>12} " + " ".join(f"{h:>12}" for h in header))
    for i, row in enumerate(cm):
        row_label = IDX_TO_CLASS[i][:12]
        print(f"  {row_label:>12} " + " ".join(f"{v:>12d}" for v in row))

    # Build full report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path,
        "arch": arch_name,
        "device": str(device),
        "metrics": metrics,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)],
        "test_loss": round(total_loss / max(total_samples, 1), 5),
        "eval_time_s": round(eval_time, 2),
        "data_info": data_info,
        "checkpoint_metadata": {
            "epoch": ckpt.get("epoch"),
            "best_val_f1": ckpt.get("best_val_f1"),
            "run_config": ckpt.get("run_config", {}),
        },
    }

    if report_path is None:
        report_path = os.path.join(REPORTS_DIR, f"eval_{arch_name}.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n[Saved] Evaluation report: {report_path}")

    summary_path = os.path.splitext(report_path)[0] + ".txt"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Test evaluation: {arch_name}\n")
        summary_file.write(f"Checkpoint: {checkpoint_path}\n")
        summary_file.write(f"Samples: {metrics['n_total']}\n")
        summary_file.write(f"Accuracy: {metrics['overall_accuracy']:.4f}\n")
        summary_file.write(f"Macro-F1: {metrics['macro_f1']:.4f}\n")
        summary_file.write("\nPer-class precision / recall / F1 / support\n")
        for class_name, class_metrics in metrics["per_class"].items():
            summary_file.write(
                f"{class_name}: {class_metrics['precision']:.4f} / "
                f"{class_metrics['recall']:.4f} / {class_metrics['f1']:.4f} / "
                f"{class_metrics['support']}\n"
            )
    print(f"[Saved] Readable evaluation summary: {summary_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CropCare AI model on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data-dir", default=DATA_RAW_DIR)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--report", default=None, help="Output report path")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    selected_batch_size = args.batch_size or (
        MOBILENET_BATCH_SIZE if checkpoint.get("arch") == MODEL_NAME else BATCH_SIZE
    )
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=selected_batch_size,
        num_workers=args.workers,
        force_cpu=args.cpu,
        report_path=args.report,
    )
