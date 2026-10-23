"""
CropCare AI — Architecture comparison script.

Runs a short training (5 epochs, no early-stopping) on each of the three
architectures under identical conditions, then prints a comparison table.
Use this to decide which architecture to train fully.
"""

import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DATA_RAW_DIR, MODELS_DIR, REPORTS_DIR, NUM_CLASSES,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS, IMG_SIZE,
)
from src.models import ARCHITECTURES, count_parameters
from src.train import train

COMPARE_EPOCHS = 5   # quick comparison — not final training


def compare_architectures(data_dir=DATA_RAW_DIR):
    print("=" * 70)
    print("CropCare AI — Architecture Comparison")
    print(f"Epochs per architecture: {COMPARE_EPOCHS}")
    print("=" * 70)
    print()

    results = {}
    for arch_name in ARCHITECTURES:
        print(f"\n{'-' * 60}")
        print(f"Architecture: {arch_name}")
        print(f"{'-' * 60}")
        try:
            result = train(
                arch_name=arch_name,
                img_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                num_epochs=COMPARE_EPOCHS,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                early_stop_patience=999,   # no early stop in comparison
                data_dir=data_dir,
                checkpoint_dir=os.path.join(MODELS_DIR, "compare"),
                num_workers=NUM_WORKERS,
            )
            history = result.get("history", [])
            best_val_f1 = max(e["val_f1"] for e in history) if history else 0.0
            final_val_acc = history[-1]["val_acc"] if history else 0.0
            total_time = result.get("total_training_time_s", 0)
            n_params = result.get("n_params", 0)
            results[arch_name] = {
                "params": n_params,
                "best_val_f1": best_val_f1,
                "final_val_acc": final_val_acc,
                "total_time_s": total_time,
                "time_per_epoch_s": total_time / COMPARE_EPOCHS,
            }
        except Exception as e:
            print(f"[ERROR] {arch_name} failed: {e}")
            results[arch_name] = {"error": str(e)}

    # Print comparison table
    print("\n")
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Architecture':<20} {'Params':>10} {'Val Acc':>9} {'Val F1':>9} {'Time/epoch':>12}")
    print(f"{'-'*20} {'-'*10} {'-'*9} {'-'*9} {'-'*12}")
    for arch_name, r in results.items():
        if "error" in r:
            print(f"{arch_name:<20} ERROR: {r['error']}")
        else:
            print(
                f"{arch_name:<20} {r['params']:>10,} "
                f"{r['final_val_acc']:>8.4f} "
                f"{r['best_val_f1']:>8.4f} "
                f"{r['time_per_epoch_s']:>10.1f}s"
            )

    # Save comparison report
    report_path = os.path.join(REPORTS_DIR, "arch_comparison.json")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Comparison report: {report_path}")

    # Recommend best architecture
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        best = max(valid, key=lambda k: valid[k]["best_val_f1"])
        print(f"\n[Recommendation] Best val F1 after {COMPARE_EPOCHS} epochs: "
              f"{best} ({valid[best]['best_val_f1']:.4f})")
        print("  -> Use this architecture for full training.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare CNN architectures")
    parser.add_argument("--data-dir", default=DATA_RAW_DIR)
    parser.add_argument("--epochs", type=int, default=COMPARE_EPOCHS)
    args = parser.parse_args()
    COMPARE_EPOCHS = args.epochs
    compare_architectures(args.data_dir)
