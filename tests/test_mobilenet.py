"""Focused tests for MobileNetV3-Large integration."""

import json
import io
import os
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from config.settings import (
    CLASS_TO_IDX,
    DATA_PROCESSED_DIR,
    NUM_CLASSES,
    TARGET_CLASSES,
)
from src.dataset import get_mobilenet_eval_transform, get_mobilenet_train_transform
from src.mobilenet import MODEL_NAME, build_mobilenet_v3_large
from src.train_mobilenet import _set_trainable_stage
from src.inference import predict


class TestMobileNetConfiguration(unittest.TestCase):
    def test_mapping_artifact_matches_configured_class_order(self):
        mapping_path = os.path.join(DATA_PROCESSED_DIR, "class_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as mapping_file:
            mapping = json.load(mapping_file)
        self.assertEqual(mapping["num_classes"], NUM_CLASSES)
        self.assertEqual(
            [item["class_name"] for item in mapping["classes"]],
            TARGET_CLASSES,
        )
        self.assertEqual(
            [item["class_idx"] for item in mapping["classes"]],
            list(range(NUM_CLASSES)),
        )
        self.assertEqual(len(CLASS_TO_IDX), NUM_CLASSES)

    def test_torchvision_model_has_configured_output_count(self):
        model = build_mobilenet_v3_large(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(output.shape, (1, NUM_CLASSES))
        self.assertEqual(MODEL_NAME, "MobileNetV3_Large")

    def test_transforms_use_224_imagenet_normalized_tensors(self):
        image = Image.new("RGB", (256, 256), color=(100, 120, 140))
        train_tensor = get_mobilenet_train_transform()(image)
        eval_tensor = get_mobilenet_eval_transform()(image)
        self.assertEqual(tuple(train_tensor.shape), (3, 224, 224))
        self.assertEqual(tuple(eval_tensor.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(train_tensor).all())
        self.assertTrue(torch.isfinite(eval_tensor).all())
        self.assertTrue(torch.equal(
            eval_tensor,
            get_mobilenet_eval_transform()(image),
        ))

    def test_fine_tuning_stages_toggle_backbone_only(self):
        model = build_mobilenet_v3_large(pretrained=False)
        _set_trainable_stage(model, "head")
        self.assertFalse(any(parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))
        _set_trainable_stage(model, "finetune")
        self.assertTrue(all(parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_prediction_returns_configured_class_and_probabilities(self):
        class FixedLogitModel(torch.nn.Module):
            def forward(self, images):
                self.input_shape = tuple(images.shape)
                logits = torch.zeros(images.shape[0], NUM_CLASSES)
                logits[:, 4] = 5
                return logits

        model = FixedLogitModel()
        image_file = io.BytesIO()
        Image.new("RGB", (64, 64), (60, 120, 180)).save(image_file, format="PNG")
        with (
            patch("src.inference.load_model", return_value=(model, torch.device("cpu"), MODEL_NAME)),
            patch("src.inference._is_mobilenet", True),
            patch("src.inference._img_size", 224),
        ):
            result = predict(image_file.getvalue(), filename="leaf.png")

        self.assertTrue(result["success"])
        self.assertEqual(result["predicted_class"], TARGET_CLASSES[4])
        self.assertEqual(result["top_predictions"][0]["class"], TARGET_CLASSES[4])
        self.assertGreater(result["confidence"], 0.0)
        self.assertAlmostEqual(result["confidence_pct"], result["confidence"] * 100, places=2)
        self.assertEqual(model.input_shape, (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
