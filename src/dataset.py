"""
CropCare AI — Dataset loader with GPU-aware DataLoader.

Builds train/val/test splits from the audited PlantVillage subset,
applies training-only augmentations, and returns DataLoaders.
"""

import os
import sys
import json
import random
import hashlib
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TARGET_CLASSES, CLASS_TO_IDX, SUPPORTED_EXTENSIONS,
    IMG_SIZE, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
    RANDOM_SEED, NUM_WORKERS, DATA_RAW_DIR, AUDIT_REPORT_PATH,
)


# ──────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────

def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Training augmentations — applied only to training data."""
    return transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),   # slight oversize
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic transform for validation and test."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_mobilenet_train_transform() -> transforms.Compose:
    """ImageNet-normalized 224px augmentation for MobileNetV3 fine-tuning."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_mobilenet_eval_transform() -> transforms.Compose:
    """Deterministic ImageNet-normalized 224px preprocessing."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """
    PlantVillage multi-crop, multi-class image dataset.

    Only images belonging to the verified configured classes are included.
    A pre-built file list (list of (path, label_idx) tuples) is passed in.
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a black image on read failure — logged during audit
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────
# Build file lists from dataset directory
# ──────────────────────────────────────────────────────

def build_file_list(data_dir: str,
                    audit_report_path: str = AUDIT_REPORT_PATH
                    ) -> List[Tuple[str, int]]:
    """
    Scan the dataset directory and return a list of (filepath, class_idx) tuples
    for all valid images belonging to the verified target classes.

    Uses the audit report's directory-to-target mapping if available.
    """
    plant_dir = os.path.join(data_dir, "PlantVillage")
    if not os.path.isdir(plant_dir):
        raise FileNotFoundError(
            f"PlantVillage dataset not found at {plant_dir}. "
            "Run scripts/download_dataset.py first."
        )

    # Load directory mapping from audit report if it matches target classes
    dir_to_target: Dict[str, str] = {}
    if os.path.isfile(audit_report_path):
        try:
            with open(audit_report_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
            cached_map = audit.get("directory_to_target_mapping", {})
            if set(cached_map.values()) == set(TARGET_CLASSES):
                dir_to_target = cached_map
        except Exception:
            dir_to_target = {}

    # Build mapping dynamically if not matching
    if not dir_to_target:
        for d in os.listdir(plant_dir):
            full = os.path.join(plant_dir, d)
            if not os.path.isdir(full):
                continue
            if d in TARGET_CLASSES:
                dir_to_target[d] = d
            else:
                d_lower = d.lower()
                for tc in TARGET_CLASSES:
                    if tc.lower() == d_lower:
                        dir_to_target[d] = tc
                        break

    samples = []
    for actual_dir, target_class in dir_to_target.items():
        if target_class not in CLASS_TO_IDX:
            continue
        class_dir = os.path.join(plant_dir, actual_dir)
        label_idx = CLASS_TO_IDX[target_class]
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath):
                samples.append((fpath, label_idx))

    return samples


def compute_md5(filepath: str, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum for an image file, or fallback to hashing path if mock/non-existent."""
    if not os.path.isfile(filepath):
        return hashlib.md5(filepath.encode("utf-8")).hexdigest()
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_difference_hash(filepath: str) -> Optional[int]:
    """Return a 64-bit difference hash, or None when the image cannot be read."""
    if not os.path.isfile(filepath):
        return None
    try:
        with Image.open(filepath) as image:
            gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(gray.get_flattened_data())
        value = 0
        for row in range(8):
            offset = row * 9
            for column in range(8):
                value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
        return value
    except Exception:
        return None


def group_samples_by_content(
    samples: List[Tuple[str, int]],
    near_duplicate_distance: int = 4,
) -> Tuple[List[List[Tuple[str, int]]], Dict[str, int]]:
    """Group exact duplicates and close 64-bit dHashes using a BK-tree."""
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    exact_to_index: Dict[str, int] = {}
    exact_groups: Dict[str, List[int]] = {}
    digests = []
    for index, (filepath, _) in enumerate(samples):
        digest = compute_md5(filepath)
        digests.append(digest)
        exact_groups.setdefault(digest, []).append(index)
        if digest in exact_to_index:
            union(index, exact_to_index[digest])
        else:
            exact_to_index[digest] = index

    # Each tree node is a dHash integer with children keyed by Hamming distance.
    trees: Dict[int, Dict[int, Dict[int, int]]] = {}
    dhash_to_index: Dict[Tuple[int, int], int] = {}
    near_pairs = 0
    for index, (filepath, _) in enumerate(samples):
        label = samples[index][1]
        tree = trees.setdefault(label, {})
        image_hash = compute_difference_hash(filepath)
        if image_hash is None:
            continue
        if not tree:
            tree[image_hash] = {}
            dhash_to_index[(label, image_hash)] = index
            continue
        if image_hash in tree:
            previous_index = dhash_to_index[(label, image_hash)]
            union(index, previous_index)
            if digests[index] != digests[previous_index]:
                near_pairs += 1
            continue

        matches = []
        pending = [next(iter(tree))]
        while pending:
            node = pending.pop()
            distance = (image_hash ^ node).bit_count()
            if distance <= near_duplicate_distance:
                matches.append(node)
            low = distance - near_duplicate_distance
            high = distance + near_duplicate_distance
            pending.extend(
                child for edge, child in tree[node].items()
                if low <= edge <= high
            )

        for match in matches:
            union(index, dhash_to_index[(label, match)])
            near_pairs += 1

        node = next(iter(tree))
        while True:
            distance = (image_hash ^ node).bit_count()
            if distance in tree[node]:
                node = tree[node][distance]
            else:
                tree[node][distance] = image_hash
                tree[image_hash] = {}
                dhash_to_index[(label, image_hash)] = index
                break

    grouped_indices: Dict[int, List[int]] = {}
    for index in range(len(samples)):
        grouped_indices.setdefault(find(index), []).append(index)
    clusters = [
        [samples[index] for index in indices]
        for indices in grouped_indices.values()
    ]
    exact_duplicate_groups = sum(len(indices) > 1 for indices in exact_groups.values())
    near_duplicate_groups = sum(
        len({digests[index] for index in indices}) > 1
        for indices in grouped_indices.values()
    )
    stats = {
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_groups": near_duplicate_groups,
        "near_duplicate_pairs": near_pairs,
    }
    return clusters, stats


def split_sample_clusters(
    clusters: List[List[Tuple[str, int]]],
    train_frac: float = TRAIN_SPLIT,
    val_frac: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[List, List, List]:
    """Stratify pre-grouped samples while keeping each group in one split."""
    rng = random.Random(seed)
    by_class_clusters: Dict[int, List[List[Tuple[str, int]]]] = {}
    conflicts = []
    for cluster in clusters:
        distinct_labels = {label for _, label in cluster}
        if len(distinct_labels) > 1:
            conflicts.append(cluster)
        by_class_clusters.setdefault(cluster[0][1], []).append(cluster)

    if conflicts:
        print(f"[WARN] Found {len(conflicts)} exact-duplicate groups with conflicting labels!")

    train, val, test = [], [], []
    for cls_idx, class_clusters in sorted(by_class_clusters.items()):
        rng.shuffle(class_clusters)
        total_items = sum(len(cluster) for cluster in class_clusters)
        target_train = int(total_items * train_frac)
        target_val = int(total_items * val_frac)
        class_train, class_val, class_test = [], [], []
        train_count = val_count = 0

        for cluster in class_clusters:
            cluster_size = len(cluster)
            if train_count + cluster_size <= target_train or (
                train_count < target_train and val_count >= target_val
            ):
                class_train.extend(cluster)
                train_count += cluster_size
            elif val_count + cluster_size <= target_val or val_count < target_val:
                class_val.extend(cluster)
                val_count += cluster_size
            else:
                class_test.extend(cluster)

        train.extend(class_train)
        val.extend(class_val)
        test.extend(class_test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def split_samples_by_hash(samples: List[Tuple[str, int]],
                          train_frac: float = TRAIN_SPLIT,
                          val_frac: float = VAL_SPLIT,
                          seed: int = RANDOM_SEED,
                          near_duplicate_distance: int = 4,
                          ) -> Tuple[List, List, List]:
    """
    Stratified split grouped by exact content and perceptual difference hash.
    """
    clusters, _ = group_samples_by_content(
        samples, near_duplicate_distance=near_duplicate_distance
    )
    return split_sample_clusters(clusters, train_frac, val_frac, seed)


def split_samples(samples: List[Tuple[str, int]],
                  train_frac: float = TRAIN_SPLIT,
                  val_frac: float = VAL_SPLIT,
                  seed: int = RANDOM_SEED
                  ) -> Tuple[List, List, List]:
    """
    Stratified split of samples. Defaults to hash-grouped splitting to prevent leakage.
    """
    return split_samples_by_hash(samples, train_frac=train_frac, val_frac=val_frac, seed=seed)


def compute_class_weights(train_samples: List[Tuple[str, int]],
                          num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for loss weighting.
    Returns a tensor of shape (num_classes,) with float weights.
    """
    counts = torch.zeros(num_classes)
    for _, label in train_samples:
        if 0 <= label < num_classes:
            counts[label] += 1
    # Avoid division by zero
    counts = torch.clamp(counts, min=1)
    weights = torch.sqrt(counts.sum() / (num_classes * counts))
    weights = weights / weights.mean()
    return weights


def build_weighted_sampler(train_samples: List[Tuple[str, int]],
                           num_classes: int) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for imbalanced datasets."""
    class_weights = compute_class_weights(train_samples, num_classes)
    sample_weights = [class_weights[label].item() for _, label in train_samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ──────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────

def load_split_csv(csv_path: str,
                   num_classes: Optional[int] = None,
                   idx_to_class: Optional[Dict[int, str]] = None) -> List[Tuple[str, int]]:
    """
    Load sample list from manifest CSV with strict integrity validation.

    Validates:
      - File existence and non-empty content
      - Required CSV headers ('filepath', 'class_idx')
      - Non-empty, valid filepath on disk
      - Integer class_idx within [0, num_classes-1]
      - Consistent class_name if present
    """
    import csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Manifest file not found: {csv_path}")
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"Manifest file is empty: {csv_path}")

    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest {csv_path} contains no header row.")
        if "filepath" not in reader.fieldnames or "class_idx" not in reader.fieldnames:
            raise ValueError(
                f"Malformed manifest {csv_path}: missing required headers ('filepath', 'class_idx'). "
                f"Found headers: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader, start=2):
            fpath = (row.get("filepath") or "").strip()
            cls_str = (row.get("class_idx") or "").strip()

            if not fpath:
                raise ValueError(f"Missing filepath at row {row_idx} in {csv_path}")
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Image file in manifest row {row_idx} not found on disk: {fpath}")

            try:
                cls_idx = int(cls_str)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid non-integer class_idx '{cls_str}' at row {row_idx} in {csv_path}")

            if num_classes is not None:
                if cls_idx < 0 or cls_idx >= num_classes:
                    raise ValueError(
                        f"Out-of-range class_idx {cls_idx} at row {row_idx} in {csv_path}. "
                        f"Expected index in range [0, {num_classes - 1}]."
                    )

            if idx_to_class and "class_name" in row and row["class_name"]:
                cname = row["class_name"].strip()
                expected_name = idx_to_class.get(cls_idx)
                if expected_name and cname != expected_name:
                    raise ValueError(
                        f"Label inconsistency at row {row_idx} in {csv_path}: "
                        f"class_idx={cls_idx} corresponds to '{expected_name}', but found '{cname}'."
                    )

            samples.append((fpath, cls_idx))

    if not samples:
        raise ValueError(f"Manifest {csv_path} contains 0 valid sample rows.")

    return samples


def get_dataloaders(data_dir: str,
                    img_size: int = IMG_SIZE,
                    batch_size: int = BATCH_SIZE,
                    num_workers: int = NUM_WORKERS,
                    use_weighted_sampler: bool = False,
                    pin_memory: bool = True,
                    transform_profile: str = "custom",
                    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns (train_loader, val_loader, test_loader, info_dict).

    info_dict contains:
      - n_train, n_val, n_test: split sizes
      - class_counts: per-class counts in training set
      - class_weights: inverse-frequency weights
      - imbalance_ratio: max/min count ratio
    """
    from config.settings import DATA_PROCESSED_DIR, NUM_CLASSES, IDX_TO_CLASS
    train_csv = os.path.join(DATA_PROCESSED_DIR, "train.csv")
    val_csv   = os.path.join(DATA_PROCESSED_DIR, "val.csv")
    test_csv  = os.path.join(DATA_PROCESSED_DIR, "test.csv")

    all_samples: List[Tuple[str, int]] = []
    mapping_path = os.path.join(DATA_PROCESSED_DIR, "class_mapping.json")
    mapping_matches = True
    if os.path.isfile(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as mapping_file:
                saved_mapping = json.load(mapping_file)
            saved_classes = [item["class_name"] for item in saved_mapping.get("classes", [])]
            mapping_matches = saved_classes == TARGET_CLASSES
        except (OSError, ValueError, KeyError, TypeError):
            mapping_matches = False

    manifests_loaded = False
    if mapping_matches and all(os.path.isfile(path) for path in (train_csv, val_csv, test_csv)):
        try:
            train_s = load_split_csv(train_csv, num_classes=NUM_CLASSES, idx_to_class=IDX_TO_CLASS)
            val_s   = load_split_csv(val_csv, num_classes=NUM_CLASSES, idx_to_class=IDX_TO_CLASS)
            test_s  = load_split_csv(test_csv, num_classes=NUM_CLASSES, idx_to_class=IDX_TO_CLASS)
            all_samples = train_s + val_s + test_s
            manifests_loaded = True
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARN] Ignoring stale or invalid split manifests: {exc}")
    elif not mapping_matches:
        print("[WARN] Ignoring split manifests with a different class mapping.")

    if manifests_loaded:
        split_hashes = [
            {compute_md5(path) for path, _ in split}
            for split in (train_s, val_s, test_s)
        ]
        has_hash_leakage = any(
            split_hashes[left] & split_hashes[right]
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
        if has_hash_leakage:
            print("[WARN] Exact duplicate images cross manifest splits; rebuilding splits by hash.")
            train_s, val_s, test_s = split_samples(all_samples)
    else:
        all_samples = build_file_list(data_dir)
        if not all_samples:
            raise RuntimeError("No samples found. Verify dataset path and audit report.")
        train_s, val_s, test_s = split_samples(all_samples)

    if transform_profile == "mobilenet_v3":
        train_transform = get_mobilenet_train_transform()
        eval_transform = get_mobilenet_eval_transform()
    elif transform_profile == "custom":
        train_transform = get_train_transform(img_size)
        eval_transform = get_eval_transform(img_size)
    else:
        raise ValueError(f"Unknown transform profile: {transform_profile}")

    train_ds = PlantVillageDataset(train_s, transform=train_transform)
    val_ds   = PlantVillageDataset(val_s,   transform=eval_transform)
    test_ds  = PlantVillageDataset(test_s,  transform=eval_transform)

    # Class weight computation
    class_weights = compute_class_weights(train_s, NUM_CLASSES)
    counts = torch.zeros(NUM_CLASSES)
    for _, l in train_s:
        if 0 <= l < NUM_CLASSES:
            counts[l] += 1
    imbalance = (counts.max() / counts.clamp(min=1).min()).item()

    sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        sampler = build_weighted_sampler(train_s, NUM_CLASSES)
        shuffle_train = False  # mutually exclusive with sampler

    # Windows: num_workers > 0 requires __main__ guard in scripts
    # For safety keep pin_memory only when CUDA available
    _pin = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        sampler=sampler, num_workers=num_workers,
        pin_memory=_pin, drop_last=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
        persistent_workers=(num_workers > 0),
    )

    info = {
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
        "n_total": len(all_samples),
        "class_counts_train": {str(i): int(counts[i]) for i in range(NUM_CLASSES)},
        "class_weights": {str(i): float(class_weights[i]) for i in range(NUM_CLASSES)},
        "imbalance_ratio": round(imbalance, 3),
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    """Quick smoke-test for the dataset loader."""
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_RAW_DIR
    print(f"Loading dataset from: {data_dir}")
    try:
        tl, vl, tel, info = get_dataloaders(data_dir, num_workers=0)
        print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")
        print(f"Imbalance ratio: {info['imbalance_ratio']}")
        batch_x, batch_y = next(iter(tl))
        print(f"Batch shape: {batch_x.shape}, Labels: {batch_y}")
        print("[OK] Dataset loader verified.")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
