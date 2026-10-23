"""Torchvision MobileNetV3-Large model factory for CropCare AI."""

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from config.settings import NUM_CLASSES


MODEL_NAME = "MobileNetV3_Large"


def build_mobilenet_v3_large(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Build MobileNetV3-Large and replace its final classifier layer."""
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    final_layer = model.classifier[-1]
    model.classifier[-1] = nn.Linear(final_layer.in_features, num_classes)
    return model