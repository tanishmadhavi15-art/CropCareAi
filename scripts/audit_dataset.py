"""Audit every populated PlantVillage class directory and usable image."""

import os
import sys
import json
import time
from collections import defaultdict

from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DATA_RAW_DIR, TARGET_CLASSES, SUPPORTED_EXTENSIONS,
    AUDIT_REPORT_PATH, REPORTS_DIR
)


def audit_dataset(data_dir=None):
    """
    Perform a comprehensive audit of the PlantVillage dataset.

    Returns a dict with the audit results and saves it as JSON.
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_RAW_DIR, "PlantVillage")

    print("=" * 70)
    print("CropCare AI — Dataset Audit")
    print("=" * 70)
    print(f"Dataset directory: {data_dir}")
    print()

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Directory does not exist: {data_dir}")
        return None

    # ── Discover all class directories ──
    all_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    print(f"Found {len(all_dirs)} class directories total.")
    print()

    # ── Build mapping from actual directory names to target classes ──
    # The PlantVillage dataset may use different separators or capitalization.
    # We normalize by comparing lowercased versions.
    target_lower = {cls.lower().replace("___", "___"): cls for cls in TARGET_CLASSES}

    dir_to_target = {}
    matched_targets = set()
    for d in all_dirs:
        d_lower = d.lower()
        # Try exact match first
        if d in TARGET_CLASSES:
            dir_to_target[d] = d
            matched_targets.add(d)
        # Try case-insensitive match
        elif d_lower in target_lower:
            dir_to_target[d] = target_lower[d_lower]
            matched_targets.add(target_lower[d_lower])
        # Try with different separators
        else:
            d_normalized = d_lower.replace(" ", "_").replace("-", "_")
            for tl, tc in target_lower.items():
                tl_normalized = tl.replace(" ", "_").replace("-", "_")
                if d_normalized == tl_normalized:
                    dir_to_target[d] = tc
                    matched_targets.add(tc)
                    break

    # Report mapping
    print("-- Target Class Mapping --")
    for d, t in sorted(dir_to_target.items()):
        print(f"  {d} -> {t}")
    print()

    # Check for missing targets
    missing_targets = set(TARGET_CLASSES) - matched_targets
    if missing_targets:
        print(f"[ERROR] Missing target classes: {missing_targets}")
        print(f"[ERROR] Cannot proceed with training until all {len(TARGET_CLASSES)} classes are present.")
        # Don't return None — still produce the audit report
    else:
        print(f"[OK] All {len(TARGET_CLASSES)} target classes found.")
    print()

    # -- Identify excluded directories --
    excluded_dirs = [d for d in all_dirs if d not in dir_to_target]
    print(f"-- Excluded Classes ({len(excluded_dirs)}) --")
    for d in excluded_dirs:
        print(f"  {d}")
    print()

    # -- Audit each target class --
    audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_directory": data_dir,
        "total_classes_found": len(all_dirs),
        "target_classes_matched": len(matched_targets),
        "missing_target_classes": sorted(list(missing_targets)),
        "excluded_classes": excluded_dirs,
        "directory_to_target_mapping": dir_to_target,
        "class_details": {},
        "total_target_images": 0,
        "total_excluded_images": 0,
        "unreadable_images": [],
        "unsupported_files": [],
    }

    print("-- Per-Class Image Audit --")
    for actual_dir, target_class in sorted(dir_to_target.items(), key=lambda x: x[1]):
        class_dir = os.path.join(data_dir, actual_dir)
        class_info = _audit_class_directory(class_dir, actual_dir)
        audit["class_details"][target_class] = class_info
        audit["total_target_images"] += class_info["valid_count"]
        audit["unreadable_images"].extend(class_info["unreadable"])
        audit["unsupported_files"].extend(class_info["unsupported"])

        print(f"  {target_class}:")
        print(f"    Directory: {actual_dir}")
        print(f"    Valid images: {class_info['valid_count']}")
        print(f"    Unreadable:   {class_info['unreadable_count']}")
        print(f"    Unsupported:  {class_info['unsupported_count']}")
        print(f"    Dimensions:   min={class_info['min_dimensions']}, "
              f"max={class_info['max_dimensions']}")
        print(f"    File sizes:   min={class_info['min_file_size_kb']:.1f} KB, "
              f"max={class_info['max_file_size_kb']:.1f} KB, "
              f"avg={class_info['avg_file_size_kb']:.1f} KB")
        print()

    empty_classes = [
        class_name for class_name in TARGET_CLASSES
        if audit["class_details"].get(class_name, {}).get("valid_count", 0) == 0
    ]
    audit["empty_configured_classes"] = empty_classes

    # Audit other populated directories too, so an unconfigured usable class
    # cannot silently disappear from the verified mapping.
    populated_excluded = []
    for actual_dir in excluded_dirs:
        class_info = _audit_class_directory(os.path.join(data_dir, actual_dir), actual_dir)
        if class_info["valid_count"]:
            populated_excluded.append(actual_dir)
            audit["total_excluded_images"] += class_info["valid_count"]
        audit["unreadable_images"].extend(class_info["unreadable"])
        audit["unsupported_files"].extend(class_info["unsupported"])
    audit["populated_unmapped_classes"] = populated_excluded

    # -- Summary --
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Target classes:      {len(matched_targets)}/{len(TARGET_CLASSES)}")
    print(f"  Total target images: {audit['total_target_images']}")
    print(f"  Excluded classes:    {len(excluded_dirs)}")
    print(f"  Excluded images:     {audit['total_excluded_images']}")
    print(f"  Unreadable images:   {len(audit['unreadable_images'])}")
    print(f"  Unsupported files:   {len(audit['unsupported_files'])}")
    print()

    # Check class balance
    if audit["class_details"]:
        counts = [info["valid_count"] for info in audit["class_details"].values()]
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        audit["class_imbalance_ratio"] = round(imbalance_ratio, 2)
        print(f"  Class imbalance ratio: {imbalance_ratio:.2f} "
              f"(max/min = {max_count}/{min_count})")
        if imbalance_ratio > 2.0:
            print("  [WARN] Significant class imbalance detected. "
                  "Consider class weighting during training.")
        else:
            print("  [OK] Class distribution is reasonably balanced.")

    # Save audit report
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(AUDIT_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit, f, indent=2)
    mapping_path = os.path.join(REPORTS_DIR, "class_mapping.json")
    mapping = {
        "classes": [
            {"class_idx": idx, "class_name": class_name,
             "usable_images": audit["class_details"].get(class_name, {}).get("valid_count", 0)}
            for idx, class_name in enumerate(TARGET_CLASSES)
        ],
        "num_classes": len(TARGET_CLASSES),
        "data_directory": data_dir,
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n[INFO] Audit report saved to: {AUDIT_REPORT_PATH}")
    print(f"[INFO] Verified class mapping saved to: {mapping_path}")

    return audit


def _audit_class_directory(class_dir, dir_name):
    """Audit a single class directory."""
    info = {
        "directory_name": dir_name,
        "valid_count": 0,
        "unreadable_count": 0,
        "unsupported_count": 0,
        "unreadable": [],
        "unsupported": [],
        "widths": [],
        "heights": [],
        "file_sizes": [],
        "min_dimensions": None,
        "max_dimensions": None,
        "min_file_size_kb": 0,
        "max_file_size_kb": 0,
        "avg_file_size_kb": 0,
    }

    if not os.path.isdir(class_dir):
        return info

    files = sorted(os.listdir(class_dir))
    for fname in files:
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            info["unsupported_count"] += 1
            info["unsupported"].append(fpath)
            continue

        file_size_kb = os.path.getsize(fpath) / 1024.0
        info["file_sizes"].append(file_size_kb)

        try:
            with Image.open(fpath) as img:
                img.verify()  # verify it's a valid image
            # Re-open to get dimensions (verify() invalidates the object)
            with Image.open(fpath) as img:
                w, h = img.size
                info["widths"].append(w)
                info["heights"].append(h)
                info["valid_count"] += 1
        except Exception as e:
            info["unreadable_count"] += 1
            info["unreadable"].append({"file": fpath, "error": str(e)})

    # Compute dimension stats
    if info["widths"]:
        info["min_dimensions"] = (min(info["widths"]), min(info["heights"]))
        info["max_dimensions"] = (max(info["widths"]), max(info["heights"]))
    if info["file_sizes"]:
        info["min_file_size_kb"] = min(info["file_sizes"])
        info["max_file_size_kb"] = max(info["file_sizes"])
        info["avg_file_size_kb"] = sum(info["file_sizes"]) / len(info["file_sizes"])

    # Don't save raw lists in the report (too large)
    del info["widths"]
    del info["heights"]
    del info["file_sizes"]

    return info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit PlantVillage dataset")
    parser.add_argument("--data-dir", default=None,
                        help="Path to dataset directory (default: auto-detect)")
    args = parser.parse_args()
    result = audit_dataset(args.data_dir)
    if result is None:
        sys.exit(1)
    if result["missing_target_classes"] or result.get("populated_unmapped_classes"):
        print(f"\n[FAIL] Missing configured classes: {result['missing_target_classes']}")
        print(f"[FAIL] Populated unmapped classes: {result.get('populated_unmapped_classes', [])}")
        sys.exit(1)
    if result.get("empty_configured_classes"):
        print(f"\n[FAIL] Configured classes with no usable images: {result['empty_configured_classes']}")
        sys.exit(1)
    print("\n[PASS] Dataset audit completed successfully.")
