"""
CropCare AI — Manifest generation and split leakage verification script.

Generates reproducible train, validation, and test manifests (CSV & JSON)
from the audited PlantVillage dataset.
Verifies:
1. Zero path overlap across splits (disjoint file paths).
2. Zero duplicate hash leakage across splits (atomic cluster assignment).
3. Conflict detection for identical hashes appearing under different classes.
4. Stratified class distribution across train/val/test splits.
"""

import os
import sys
import csv
import json
import random
import hashlib
from typing import Dict, List, Tuple
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, REPORTS_DIR,
    TARGET_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED,
)
from src.dataset import (
    build_file_list, group_samples_by_content, split_sample_clusters,
)


def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """Compute MD5 hash of a file to detect byte-identical duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_and_verify_manifests(
    data_dir: str = DATA_RAW_DIR,
    out_dir: str = DATA_PROCESSED_DIR,
    train_frac: float = TRAIN_SPLIT,
    val_frac: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> dict:
    print("=" * 70)
    print("CropCare AI - Data Manifest Generation & Leakage Verification")
    print("=" * 70)
    print(f"Data directory:      {data_dir}")
    print(f"Manifest output dir: {out_dir}")
    print(f"Target classes:      {len(TARGET_CLASSES)}")
    print(f"Splits ratio:        {train_frac*100:.0f}% train / {val_frac*100:.0f}% val / {(1 - train_frac - val_frac)*100:.0f}% test")
    print(f"Random seed:         {seed}")
    print()

    # 1. Discover samples
    samples = build_file_list(data_dir)
    print(f"[Info] Found {len(samples)} total target class images across {len(TARGET_CLASSES)} classes.")
    discovered_labels = {label for _, label in samples}
    expected_labels = set(range(len(TARGET_CLASSES)))
    if discovered_labels != expected_labels:
        missing_labels = expected_labels - discovered_labels
        unexpected_labels = discovered_labels - expected_labels
        raise RuntimeError(
            f"Dataset labels do not match the verified mapping; missing={missing_labels}, "
            f"unexpected={unexpected_labels}"
        )

    # 2. Content hash audit & conflict check prior to split
    print("[Info] Scanning image MD5 hashes across all samples...")
    hash_to_samples: Dict[str, List[Tuple[str, int]]] = {}
    for item in samples:
        fpath, l = item
        h = compute_file_hash(fpath)
        hash_to_samples.setdefault(h, []).append(item)

    conflicting_hashes = {}
    duplicate_groups = {}
    for h, items in hash_to_samples.items():
        if len(items) > 1:
            duplicate_groups[h] = [p for p, _ in items]
        labels = {l for _, l in items}
        if len(labels) > 1:
            conflicting_hashes[h] = {
                "classes": [IDX_TO_CLASS[l] for l in labels],
                "files": [p for p, _ in items]
            }

    print(f"[Audit] Unique image hashes: {len(hash_to_samples)} / {len(samples)} total files")
    print(f"[Audit] Hashes with identical duplicate images: {len(duplicate_groups)}")
    if conflicting_hashes:
        print(f"[WARN] Conflicting labels across identical image hashes: {len(conflicting_hashes)}")
        for h, info in list(conflicting_hashes.items())[:5]:
            print(f"  Hash {h[:12]}... labeled as both {info['classes']}")
    else:
        print("[Pass] Label consistency: 0 identical images appear with conflicting labels.")

    # 3. Stratified split grouped by content hash (Cluster-based splitting)
    clusters, duplicate_stats = group_samples_by_content(samples)
    train_s, val_s, test_s = split_sample_clusters(
        clusters, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    path_to_split = {
        path: split_name
        for split_name, split_data in (("train", train_s), ("val", val_s), ("test", test_s))
        for path, _ in split_data
    }
    split_crossing_clusters = [
        cluster for cluster in clusters
        if len({path_to_split[path] for path, _ in cluster}) > 1
    ]
    assert not split_crossing_clusters, (
        f"Perceptual duplicate leakage detected in {len(split_crossing_clusters)} groups"
    )
    print(f"\n[Info] Partitioned into:")
    print(f"  Train: {len(train_s)} samples ({len(train_s)/len(samples)*100:.2f}%)")
    print(f"  Val:   {len(val_s)} samples ({len(val_s)/len(samples)*100:.2f}%)")
    print(f"  Test:  {len(test_s)} samples ({len(test_s)/len(samples)*100:.2f}%)")
    print()

    # 4. Strict Path overlap verification
    train_paths = {p for p, _ in train_s}
    val_paths   = {p for p, _ in val_s}
    test_paths  = {p for p, _ in test_s}

    overlap_train_val  = train_paths & val_paths
    overlap_train_test = train_paths & test_paths
    overlap_val_test   = val_paths & test_paths

    assert len(overlap_train_val) == 0, f"Path leakage detected between train and val: {len(overlap_train_val)} files"
    assert len(overlap_train_test) == 0, f"Path leakage detected between train and test: {len(overlap_train_test)} files"
    assert len(overlap_val_test) == 0, f"Path leakage detected between val and test: {len(overlap_val_test)} files"
    print("[Pass] Verification 1: File paths across train/val/test are strictly disjoint (0 overlap).")

    # 5. Strict Hash overlap verification
    train_hashes = {compute_file_hash(p) for p, _ in train_s}
    val_hashes   = {compute_file_hash(p) for p, _ in val_s}
    test_hashes  = {compute_file_hash(p) for p, _ in test_s}

    hash_leak_train_val  = train_hashes & val_hashes
    hash_leak_train_test = train_hashes & test_hashes
    hash_leak_val_test   = val_hashes & test_hashes

    print(f"  Unique hashes in splits: Train={len(train_hashes)}, Val={len(val_hashes)}, Test={len(test_hashes)}")
    print(f"  Hash leakage Train-Val:  {len(hash_leak_train_val)}")
    print(f"  Hash leakage Train-Test: {len(hash_leak_train_test)}")
    print(f"  Hash leakage Val-Test:   {len(hash_leak_val_test)}")

    assert len(hash_leak_train_val) == 0, f"Critical: Hash leakage between train and val: {len(hash_leak_train_val)}"
    assert len(hash_leak_train_test) == 0, f"Critical: Hash leakage between train and test: {len(hash_leak_train_test)}"
    assert len(hash_leak_val_test) == 0, f"Critical: Hash leakage between val and test: {len(hash_leak_val_test)}"
    print("[Pass] Verification 2: EXACT-HASH LEAKAGE ACROSS SPLITS IS ZERO (0).")

    # Every perceptual cluster is assigned as one unit by the grouped splitter.
    split_path_sets = [train_paths, val_paths, test_paths]
    path_leakage = any(
        split_path_sets[left] & split_path_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    )
    assert not path_leakage

    # 6. Class balance breakdown
    train_dist = Counter(l for _, l in train_s)
    val_dist   = Counter(l for _, l in val_s)
    test_dist  = Counter(l for _, l in test_s)

    print("\n-- Stratified Split Breakdown --")
    print(f"{'Class Name':<45} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("-" * 77)
    per_class_summary = {}
    for idx in range(len(TARGET_CLASSES)):
        c_name = IDX_TO_CLASS[idx]
        tr_c = train_dist[idx]
        va_c = val_dist[idx]
        te_c = test_dist[idx]
        tot  = tr_c + va_c + te_c
        per_class_summary[c_name] = {
            "train": tr_c,
            "val": va_c,
            "test": te_c,
            "total": tot,
            "train_ratio": round(tr_c / tot, 4) if tot > 0 else 0,
        }
        print(f"{c_name:<45} {tr_c:>7d} {va_c:>7d} {te_c:>7d} {tot:>7d}")

    all_counts = [per_class_summary[IDX_TO_CLASS[idx]]["total"] for idx in range(len(TARGET_CLASSES))]
    min_class_count = min(all_counts)
    max_class_count = max(all_counts)
    overall_imbalance_ratio = max_class_count / min_class_count
    print(f"\n[Info] Overall class imbalance: {overall_imbalance_ratio:.2f}:1 "
          f"(max={max_class_count}, min={min_class_count})")
    print(f"[Info] Exact duplicate groups: {duplicate_stats['exact_duplicate_groups']}")
    print(f"[Info] Perceptual near-duplicate groups: {duplicate_stats['near_duplicate_groups']}")
    print(f"[Info] Perceptual near-duplicate pairs: {duplicate_stats['near_duplicate_pairs']}")

    # 7. Save Manifests (CSV & JSON)
    os.makedirs(out_dir, exist_ok=True)
    manifest_paths = {}

    for split_name, split_data in [("train", train_s), ("val", val_s), ("test", test_s)]:
        csv_path = os.path.join(out_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "class_idx", "class_name"])
            for p, l in split_data:
                writer.writerow([p, l, IDX_TO_CLASS[l]])
        manifest_paths[split_name] = csv_path
        print(f"[Saved] {split_name.capitalize()} manifest: {csv_path} ({len(split_data)} records)")

    summary_report = {
        "num_classes": len(TARGET_CLASSES),
        "seed": seed,
        "train_samples": len(train_s),
        "val_samples": len(val_s),
        "test_samples": len(test_s),
        "total_samples": len(samples),
        "class_imbalance_ratio": round(overall_imbalance_ratio, 4),
        "leakage_check": {
            "path_overlap_train_val": len(overlap_train_val),
            "path_overlap_train_test": len(overlap_train_test),
            "path_overlap_val_test": len(overlap_val_test),
            "hash_leak_train_val": len(hash_leak_train_val),
            "hash_leak_train_test": len(hash_leak_train_test),
            "hash_leak_val_test": len(hash_leak_val_test),
        },
        "duplicate_groups_count": len(duplicate_groups),
        "perceptual_duplicate_groups_count": duplicate_stats["near_duplicate_groups"],
        "perceptual_duplicate_pairs_count": duplicate_stats["near_duplicate_pairs"],
        "perceptual_split_crossing_groups": len(split_crossing_clusters),
        "conflicting_hashes_count": len(conflicting_hashes),
        "conflicting_hashes": conflicting_hashes,
        "per_class": per_class_summary,
        "target_classes": TARGET_CLASSES,
        "manifest_files": manifest_paths,
    }

    summary_path = os.path.join(out_dir, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Saved] Split summary report: {summary_path}")

    mapping_path = os.path.join(out_dir, "class_mapping.json")
    mapping_report = {
        "num_classes": len(TARGET_CLASSES),
        "classes": [
            {"class_idx": idx, "class_name": class_name,
             "usable_images": per_class_summary[class_name]["total"]}
            for idx, class_name in enumerate(TARGET_CLASSES)
        ],
        "seed": seed,
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_report, f, indent=2)
    print(f"[Saved] Verified class mapping: {mapping_path}")

    return summary_report


if __name__ == "__main__":
    generate_and_verify_manifests()
