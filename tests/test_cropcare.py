"""
CropCare AI — Test suite.

Tests cover:
  - Model architecture (forward pass, parameter count, output shape)
  - Dataset loader (smoke, stratified split, transforms)
  - Inference engine (validation logic, predict flow)
  - Database (init, save, retrieve)
  - Config integrity

All tests run in CPU mode to avoid GPU requirement in CI.
"""

import os
import sys
import io
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    TARGET_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES,
    IMG_SIZE, DISEASE_INFO,
)
from src.models import (
    CropCareNet_S, CropCareNet_M, CropCareNet_L,
    build_model, count_parameters, ARCHITECTURES,
)
from src.inference import validate_image_bytes, ImageValidationError
from src.database import init_db, save_prediction, get_all_predictions, get_prediction_by_id
from src.evaluate import compute_full_metrics, compute_confusion_matrix
from src.dataset import split_samples, split_samples_by_hash, compute_class_weights


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def make_dummy_image_bytes(width=128, height=128, fmt="JPEG") -> bytes:
    """Create a valid in-memory image."""
    img = Image.fromarray(
        np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_tiny_image_bytes(width=5, height=5) -> bytes:
    img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ──────────────────────────────────────────────────────
# 1. Configuration tests
# ──────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def test_target_classes_count(self):
        self.assertEqual(len(TARGET_CLASSES), 38)

    def test_class_to_idx_consistency(self):
        for cls in TARGET_CLASSES:
            self.assertIn(cls, CLASS_TO_IDX)
            idx = CLASS_TO_IDX[cls]
            self.assertEqual(IDX_TO_CLASS[idx], cls)

    def test_idx_range(self):
        for idx in IDX_TO_CLASS:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, NUM_CLASSES)

    def test_disease_info_complete(self):
        for cls in TARGET_CLASSES:
            self.assertIn(cls, DISEASE_INFO,
                          f"DISEASE_INFO missing entry for {cls}")
            info = DISEASE_INFO[cls]
            self.assertIn("display_name", info)
            self.assertIn("symptoms", info)
            self.assertIn("management", info)
            self.assertTrue(len(info["symptoms"]) > 10)
            self.assertTrue(len(info["management"]) > 10)

    def test_num_classes_matches(self):
        self.assertEqual(NUM_CLASSES, len(TARGET_CLASSES))


# ──────────────────────────────────────────────────────
# 2. Model architecture tests
# ──────────────────────────────────────────────────────

class TestModelArchitectures(unittest.TestCase):

    def _run_forward(self, model, batch_size=2, img_size=128):
        model.eval()
        x = torch.randn(batch_size, 3, img_size, img_size)
        with torch.no_grad():
            out = model(x)
        return out

    def test_cropcarenet_s_output_shape(self):
        model = CropCareNet_S(num_classes=NUM_CLASSES)
        out = self._run_forward(model, batch_size=2)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_cropcarenet_m_output_shape(self):
        model = CropCareNet_M(num_classes=NUM_CLASSES)
        out = self._run_forward(model, batch_size=2)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_cropcarenet_l_output_shape(self):
        model = CropCareNet_L(num_classes=NUM_CLASSES)
        out = self._run_forward(model, batch_size=2)
        self.assertEqual(out.shape, (2, NUM_CLASSES))

    def test_cropcarenet_s_params_reasonable(self):
        model = CropCareNet_S(num_classes=NUM_CLASSES)
        n = count_parameters(model)
        self.assertGreater(n, 500_000)
        self.assertLess(n, 5_000_000)

    def test_cropcarenet_m_params_reasonable(self):
        model = CropCareNet_M(num_classes=NUM_CLASSES)
        n = count_parameters(model)
        self.assertGreater(n, 1_000_000)
        self.assertLess(n, 50_000_000)   # generous upper bound

    def test_cropcarenet_l_params_reasonable(self):
        model = CropCareNet_L(num_classes=NUM_CLASSES)
        n = count_parameters(model)
        self.assertGreater(n, 2_000_000)
        self.assertLess(n, 100_000_000)  # generous upper bound

    def test_build_model_factory(self):
        for arch_name in ARCHITECTURES:
            model = build_model(arch_name, num_classes=NUM_CLASSES)
            self.assertIsNotNone(model)

    def test_build_model_invalid_name(self):
        with self.assertRaises(ValueError):
            build_model("NonExistentArch")

    def test_no_pretrained_weights(self):
        """Verify weights are random (not all-zeros or all-ones)."""
        model = CropCareNet_M(num_classes=NUM_CLASSES)
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() > 1:
                std = param.data.std().item()
                self.assertGreater(std, 1e-6,
                    f"Weight '{name}' looks uninitialized (std={std})")
                break

    def test_batch_size_1(self):
        """BatchNorm should handle single-sample in eval mode."""
        model = CropCareNet_M(num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, NUM_CLASSES))

    def test_different_input_sizes(self):
        """Model should work with 128 and 160 px inputs."""
        model = CropCareNet_M(num_classes=NUM_CLASSES)
        model.eval()
        for sz in [128, 160]:
            x = torch.randn(1, 3, sz, sz)
            with torch.no_grad():
                out = model(x)
            self.assertEqual(out.shape[1], NUM_CLASSES)

    def test_gradients_flow(self):
        """Verify gradients propagate through the full network."""
        model = CropCareNet_M(num_classes=NUM_CLASSES)
        model.train()
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        self.assertTrue(has_grad, "No gradient found after backward pass")


# ──────────────────────────────────────────────────────
# 3. Dataset utility tests
# ──────────────────────────────────────────────────────

class TestDatasetUtils(unittest.TestCase):

    def _make_samples(self, n_per_class=20):
        samples = []
        for cls_idx in range(NUM_CLASSES):
            for i in range(n_per_class):
                samples.append((f"/fake/path/{cls_idx}/{i}.jpg", cls_idx))
        return samples

    def test_stratified_split_sizes(self):
        samples = self._make_samples(100)
        train, val, test = split_samples(samples, 0.7, 0.15)
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, len(samples))
        # Allow ±5% tolerance
        self.assertAlmostEqual(len(train) / len(samples), 0.70, delta=0.05)
        self.assertAlmostEqual(len(val)   / len(samples), 0.15, delta=0.05)

    def test_stratified_split_all_classes_represented(self):
        samples = self._make_samples(50)
        train, val, test = split_samples(samples, 0.7, 0.15)
        for split in [train, val, test]:
            classes_in_split = set(label for _, label in split)
            self.assertEqual(len(classes_in_split), NUM_CLASSES,
                             f"Not all classes in split of size {len(split)}")

    def test_split_reproducibility(self):
        samples = self._make_samples(50)
        t1, v1, e1 = split_samples(samples)
        t2, v2, e2 = split_samples(samples)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)

    def test_exact_duplicate_images_stay_in_one_split(self):
        from src.dataset import compute_md5

        with tempfile.TemporaryDirectory() as temp_dir:
            duplicate_bytes = make_dummy_image_bytes()
            duplicate_paths = [
                os.path.join(temp_dir, f"duplicate_{duplicate_index}.jpg")
                for duplicate_index in range(2)
            ]
            unique_path = os.path.join(temp_dir, "unique.jpg")
            for path in duplicate_paths:
                with open(path, "wb") as image_file:
                    image_file.write(duplicate_bytes)
            with open(unique_path, "wb") as image_file:
                image_file.write(make_dummy_image_bytes())

            samples = [(path, 0) for path in duplicate_paths + [unique_path]]
            splits = split_samples_by_hash(samples, train_frac=0.34, val_frac=0.33)
            duplicate_splits = [
                split_index
                for split_index, split in enumerate(splits)
                if any(path in duplicate_paths for path, _ in split)
            ]

            self.assertEqual(len(duplicate_splits), 1)

    def test_perceptual_near_duplicates_stay_in_one_split(self):
        from src.dataset import group_samples_by_content

        with tempfile.TemporaryDirectory() as temp_dir:
            base = np.zeros((96, 96, 3), dtype=np.uint8)
            base[20:75, 25:70] = (80, 140, 180)
            changed = base.copy()
            changed[0, 0] = (255, 255, 255)
            paths = [os.path.join(temp_dir, "base.png"), os.path.join(temp_dir, "changed.png")]
            Image.fromarray(base).save(paths[0])
            Image.fromarray(changed).save(paths[1])

            clusters, stats = group_samples_by_content([(paths[0], 0), (paths[1], 0)])

            self.assertEqual(len(clusters), 1)
            self.assertEqual(stats["near_duplicate_groups"], 1)

    def test_perceptual_near_duplicates_with_different_labels_stay_separate(self):
        from src.dataset import group_samples_by_content

        with tempfile.TemporaryDirectory() as temp_dir:
            base = np.zeros((96, 96, 3), dtype=np.uint8)
            base[20:75, 25:70] = (80, 140, 180)
            changed = base.copy()
            changed[0, 0] = (255, 255, 255)
            paths = [os.path.join(temp_dir, "base.png"), os.path.join(temp_dir, "changed.png")]
            Image.fromarray(base).save(paths[0])
            Image.fromarray(changed).save(paths[1])

            clusters, _ = group_samples_by_content([(paths[0], 0), (paths[1], 1)])

            self.assertEqual(len(clusters), 2)

    def test_exact_duplicates_with_different_labels_stay_together(self):
        from src.dataset import group_samples_by_content

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "same.png")
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)

            clusters, _ = group_samples_by_content([(image_path, 0), (image_path, 1)])

            self.assertEqual(len(clusters), 1)
            self.assertEqual({label for _, label in clusters[0]}, {0, 1})

    def test_manifest_hash_leakage_is_resplit(self):
        from config import settings
        from src import dataset
        from src.dataset import compute_md5

        with tempfile.TemporaryDirectory() as temp_dir:
            duplicate_bytes = make_dummy_image_bytes()
            duplicate_paths = [
                os.path.join(temp_dir, f"duplicate_{duplicate_index}.jpg")
                for duplicate_index in range(2)
            ]
            unique_path = os.path.join(temp_dir, "unique.jpg")
            for path in duplicate_paths:
                with open(path, "wb") as image_file:
                    image_file.write(duplicate_bytes)
            with open(unique_path, "wb") as image_file:
                image_file.write(make_dummy_image_bytes())

            manifest_rows = {
                "train": duplicate_paths[0],
                "val": duplicate_paths[1],
                "test": unique_path,
            }
            for split_name, image_path in manifest_rows.items():
                with open(os.path.join(temp_dir, f"{split_name}.csv"), "w", encoding="utf-8") as manifest:
                    manifest.write(f"filepath,class_idx\n{image_path},0\n")

            with (
                patch.object(settings, "DATA_PROCESSED_DIR", temp_dir),
                patch.object(dataset, "get_train_transform", return_value=None),
                patch.object(dataset, "get_eval_transform", return_value=None),
            ):
                train_loader, val_loader, test_loader, _ = dataset.get_dataloaders(
                    data_dir=temp_dir, batch_size=1, num_workers=0, pin_memory=False
                )

            split_hashes = [
                {compute_md5(path) for path, _ in loader.dataset.samples}
                for loader in (train_loader, val_loader, test_loader)
            ]
            self.assertFalse(split_hashes[0] & split_hashes[1])
            self.assertFalse(split_hashes[0] & split_hashes[2])
            self.assertFalse(split_hashes[1] & split_hashes[2])

    def test_stale_manifests_fall_back_to_dataset_samples(self):
        from config import settings
        from src import dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            for split_name in ("train", "val", "test"):
                with open(os.path.join(temp_dir, f"{split_name}.csv"), "w", encoding="utf-8") as manifest:
                    manifest.write("filepath,class_idx\nC:/missing/image.jpg,0\n")

            samples = self._make_samples(2)
            with (
                patch.object(settings, "DATA_PROCESSED_DIR", temp_dir),
                patch.object(dataset, "build_file_list", return_value=samples),
                patch.object(dataset, "get_train_transform", return_value=None),
                patch.object(dataset, "get_eval_transform", return_value=None),
            ):
                _, _, _, info = dataset.get_dataloaders(
                    data_dir=temp_dir, batch_size=1, num_workers=0, pin_memory=False
                )

            self.assertEqual(info["n_total"], len(samples))

    def test_manifest_mapping_mismatch_falls_back_to_raw_data(self):
        from config import settings
        from src import dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            for split_name in ("train", "val", "test"):
                with open(os.path.join(temp_dir, f"{split_name}.csv"), "w", encoding="utf-8") as manifest:
                    manifest.write("filepath,class_idx\nC:/unused/image.jpg,0\n")
            with open(os.path.join(temp_dir, "class_mapping.json"), "w", encoding="utf-8") as mapping_file:
                json.dump({"classes": [{"class_name": "Old___Mapping"}]}, mapping_file)

            samples = self._make_samples(2)
            with (
                patch.object(settings, "DATA_PROCESSED_DIR", temp_dir),
                patch.object(dataset, "build_file_list", return_value=samples),
                patch.object(dataset, "get_train_transform", return_value=None),
                patch.object(dataset, "get_eval_transform", return_value=None),
            ):
                _, _, _, info = dataset.get_dataloaders(
                    data_dir=temp_dir, batch_size=1, num_workers=0, pin_memory=False
                )

            self.assertEqual(info["n_total"], len(samples))

    def test_class_weights_shape(self):
        samples = self._make_samples(30)
        weights = compute_class_weights(samples, NUM_CLASSES)
        self.assertEqual(weights.shape[0], NUM_CLASSES)

    def test_class_weights_balanced(self):
        """For perfectly balanced data, all weights should be equal."""
        samples = self._make_samples(50)  # equal per class
        weights = compute_class_weights(samples, NUM_CLASSES)
        # All weights should be ≈ 1.0 for balanced data
        self.assertTrue(
            weights.std().item() < 0.01,
            f"Weights should be equal for balanced data, got std={weights.std()}"
        )

    def test_manifest_label_matches_class_mapping(self):
        import csv
        from src.dataset import load_split_csv

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "label-check.png")
            Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
            manifest_path = os.path.join(temp_dir, "labels.csv")
            with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.writer(manifest_file)
                writer.writerow(["filepath", "class_idx", "class_name"])
                writer.writerow([image_path, 24, "Soybean___healthy"])

            samples = load_split_csv(manifest_path, NUM_CLASSES, IDX_TO_CLASS)

            self.assertEqual(samples, [(image_path, CLASS_TO_IDX["Soybean___healthy"])])


# ──────────────────────────────────────────────────────
# 4. Image validation tests
# ──────────────────────────────────────────────────────

class TestImageValidation(unittest.TestCase):

    def test_valid_jpeg(self):
        img_bytes = make_dummy_image_bytes(fmt="JPEG")
        img = validate_image_bytes(img_bytes, "test.jpg")
        self.assertIsNotNone(img)

    def test_valid_png(self):
        img_bytes = make_dummy_image_bytes(fmt="PNG")
        img = validate_image_bytes(img_bytes, "test.png")
        self.assertIsNotNone(img)

    def test_unsupported_extension(self):
        img_bytes = make_dummy_image_bytes()
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(img_bytes, "test.tiff")
        self.assertIn("Unsupported", str(ctx.exception))

    def test_corrupt_bytes(self):
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(b"not an image at all", "test.jpg")

    def test_too_small_image(self):
        img_bytes = make_tiny_image_bytes(3, 3)
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(img_bytes, "tiny.jpg")
        self.assertIn("small", str(ctx.exception).lower())

    def test_oversized_file(self):
        # Create bytes that exceed 10 MB limit
        big_bytes = b"x" * (11 * 1024 * 1024)
        with self.assertRaises(ImageValidationError) as ctx:
            validate_image_bytes(big_bytes, "big.jpg")
        self.assertIn("large", str(ctx.exception).lower())

    def test_no_extension_check_when_no_filename(self):
        """Without filename, extension check is skipped."""
        img_bytes = make_dummy_image_bytes(fmt="JPEG")
        img = validate_image_bytes(img_bytes, "")  # no extension
        self.assertIsNotNone(img)


# ──────────────────────────────────────────────────────
# 5. Evaluation metric tests
# ──────────────────────────────────────────────────────

class TestEvaluationMetrics(unittest.TestCase):

    def test_metrics_for_small_batch_include_absent_classes(self):
        preds = torch.tensor([0, 0])
        targets = torch.tensor([0, 1])

        metrics = compute_full_metrics(preds, targets)

        self.assertEqual(metrics["n_total"], 2)
        self.assertEqual(metrics["overall_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / NUM_CLASSES, places=4)

    def test_perfect_predictions(self):
        preds   = torch.arange(NUM_CLASSES).repeat(10)
        targets = torch.arange(NUM_CLASSES).repeat(10)
        metrics = compute_full_metrics(preds, targets)
        self.assertAlmostEqual(metrics["overall_accuracy"], 1.0, places=4)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0, places=3)

    def test_zero_accuracy(self):
        # All predictions wrong — cycle by 1
        targets = torch.arange(NUM_CLASSES).repeat(10)
        preds   = (targets + 1) % NUM_CLASSES
        metrics = compute_full_metrics(preds, targets)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.0, places=4)

    def test_metrics_keys(self):
        preds   = torch.zeros(60, dtype=torch.long)
        targets = torch.zeros(60, dtype=torch.long)
        metrics = compute_full_metrics(preds, targets)
        for key in ["overall_accuracy", "macro_f1", "macro_precision",
                    "macro_recall", "per_class"]:
            self.assertIn(key, metrics)

    def test_confusion_matrix_shape(self):
        preds   = torch.randint(0, NUM_CLASSES, (100,))
        targets = torch.randint(0, NUM_CLASSES, (100,))
        cm = compute_confusion_matrix(preds, targets)
        self.assertEqual(len(cm), NUM_CLASSES)
        for row in cm:
            self.assertEqual(len(row), NUM_CLASSES)

    def test_confusion_matrix_diagonal_correct(self):
        """Diagonal entries must equal per-class correct count."""
        preds   = torch.arange(NUM_CLASSES).repeat(5)
        targets = torch.arange(NUM_CLASSES).repeat(5)
        cm = compute_confusion_matrix(preds, targets)
        for i in range(NUM_CLASSES):
            self.assertEqual(cm[i][i], 5)


# ──────────────────────────────────────────────────────
# 6. Database tests
# ──────────────────────────────────────────────────────

class TestDatabase(unittest.TestCase):

    def setUp(self):
        # Use a temporary DB for each test
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        # Patch DB_PATH
        import src.database as db_module
        self._orig_path = db_module.DB_PATH
        db_module.DB_PATH = self.tmp.name
        init_db()

    def tearDown(self):
        import src.database as db_module
        db_module.DB_PATH = self._orig_path
        os.unlink(self.tmp.name)

    def _dummy_result(self, cls="Tomato___healthy", conf=0.95):
        return {
            "predicted_class": cls,
            "display_name": DISEASE_INFO[cls]["display_name"],
            "confidence": conf,
            "top_predictions": [{"class": cls, "confidence": conf}],
            "arch": "CropCareNet_M",
            "img_size": 128,
            "disease_info": {
                "symptoms": DISEASE_INFO[cls]["symptoms"],
                "management": DISEASE_INFO[cls]["management"],
            },
        }

    def test_save_and_retrieve(self):
        result = self._dummy_result()
        row_id = save_prediction("leaf.jpg", result)
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

        record = get_prediction_by_id(row_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["predicted_class"], "Tomato___healthy")
        self.assertAlmostEqual(record["confidence"], 0.95, places=3)

    def test_multiple_saves(self):
        for cls in TARGET_CLASSES[:3]:
            save_prediction("img.jpg", self._dummy_result(cls))
        history = get_all_predictions(limit=10)
        self.assertEqual(len(history), 3)

    def test_get_all_predictions_newest_first(self):
        for cls in TARGET_CLASSES:
            save_prediction("img.jpg", self._dummy_result(cls))
        history = get_all_predictions()
        ids = [h["id"] for h in history]
        self.assertEqual(ids, sorted(ids, reverse=True))

    def test_top_predictions_deserialized(self):
        result = self._dummy_result()
        row_id = save_prediction("leaf.jpg", result)
        record = get_prediction_by_id(row_id)
        self.assertIsInstance(record["top_predictions"], list)

    def test_empty_history(self):
        history = get_all_predictions()
        self.assertEqual(history, [])


# ──────────────────────────────────────────────────────
# 7. Smoke test — training loop (CPU, 2 epochs)
# ──────────────────────────────────────────────────────

class TestTrainingSmokeTest(unittest.TestCase):
    """
    Creates a tiny in-memory dataset and runs 2 training epochs
    on CPU to verify the training loop executes without errors.
    """

    def test_training_loop_runs(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from src.models import CropCareNet_S
        from src.train import train_one_epoch, evaluate, compute_macro_f1
        from torch.utils.data import DataLoader, TensorDataset
        from torch.cuda.amp import GradScaler

        device = torch.device("cpu")
        model = CropCareNet_S(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        try:
            scaler = torch.amp.GradScaler('cpu', enabled=False)
        except Exception:
            scaler = GradScaler(enabled=False)  # fallback for older torch

        # 4 random batches
        X = torch.randn(32, 3, 128, 128)
        Y = torch.randint(0, NUM_CLASSES, (32,))
        ds = TensorDataset(X, Y)
        loader = DataLoader(ds, batch_size=8)

        for epoch in range(2):
            train_metrics = train_one_epoch(
                model, loader, optimizer, criterion, device, scaler, use_amp=False)
            val_metrics = evaluate(model, loader, criterion, device, use_amp=False)
            self.assertIn("loss", train_metrics)
            self.assertIn("acc", train_metrics)
            self.assertIn("macro_f1", train_metrics)
            self.assertGreaterEqual(train_metrics["loss"], 0)
            self.assertGreaterEqual(val_metrics["acc"], 0)
            self.assertLessEqual(val_metrics["acc"], 1)

    def test_one_batch_optimizer_update_changes_parameters(self):
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from src.train import train_one_epoch

        device = torch.device("cpu")
        model = nn.Linear(4, NUM_CLASSES)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.arange(8) % NUM_CLASSES),
            batch_size=8,
        )
        before = model.weight.detach().clone()

        metrics = train_one_epoch(
            model, loader, optimizer, nn.CrossEntropyLoss(), device, scaler,
            use_amp=False,
        )

        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
        self.assertGreater((model.weight.detach() - before).abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
