"""
CropCare AI — Custom CNN Architectures (trained from random initialization).

Three architectures compared under identical data splits and training budget:
  A) CropCareNet_S  — small baseline (7 conv layers, ~1.2M params)
  B) CropCareNet_M  — medium with residual blocks (~3.1M params)
  C) CropCareNet_L  — larger with deeper residuals + CBAM-style attention (~5.8M params)

No pretrained weights. No transfer learning. No model distillation.
All weights are initialized from scratch using Kaiming initialization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import NUM_CLASSES


# ──────────────────────────────────────────────────────
# Common building blocks
# ──────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block: BN→ReLU→Conv→BN→ReLU→Conv + shortcut.
    Optionally downsamples when stride > 1.
    """
    def __init__(self, in_ch, out_ch, stride=1, dropout_p=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        return out + self.shortcut(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention using avg + max pooling across channel dim."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ──────────────────────────────────────────────────────
# Architecture A: CropCareNet_S (small baseline)
# ──────────────────────────────────────────────────────

class CropCareNet_S(nn.Module):
    """
    Small baseline CNN. 7 conv layers with increasing channels.
    Input: 3×128×128
    Params: ~1.2M
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            ConvBNReLU(3, 32, 3),
            ConvBNReLU(32, 32, 3),
            nn.MaxPool2d(2, 2),      # 64×64
            nn.Dropout2d(0.1),
            # Block 2
            ConvBNReLU(32, 64, 3),
            ConvBNReLU(64, 64, 3),
            nn.MaxPool2d(2, 2),      # 32×32
            nn.Dropout2d(0.1),
            # Block 3
            ConvBNReLU(64, 128, 3),
            ConvBNReLU(128, 128, 3),
            nn.MaxPool2d(2, 2),      # 16×16
            nn.Dropout2d(0.2),
            # Block 4
            ConvBNReLU(128, 256, 3),
            nn.MaxPool2d(2, 2),      # 8×8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ──────────────────────────────────────────────────────
# Architecture B: CropCareNet_M (medium with residuals)
# ──────────────────────────────────────────────────────

class CropCareNet_M(nn.Module):
    """
    Medium CNN with pre-activation residual blocks.
    Input: 3×128×128
    Params: ~3.1M
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.4):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, 3, stride=1),
            ConvBNReLU(32, 64, 3, stride=2),    # 64×64
        )
        # Residual stages
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, dropout_p=0.1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),    # 32×32
            ResidualBlock(128, 128, dropout_p=0.1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),   # 16×16
            ResidualBlock(256, 256, dropout_p=0.2),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),   # 8×8
            ResidualBlock(512, 512, dropout_p=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


# ──────────────────────────────────────────────────────
# Architecture C: CropCareNet_L (larger with attention)
# ──────────────────────────────────────────────────────

class CropCareNet_L(nn.Module):
    """
    Larger CNN with residual blocks + CBAM attention.
    Input: 3×128×128
    Params: ~5.8M
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_p=0.4):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
            nn.MaxPool2d(2, 2),                  # 64×64
        )
        # Stage 1: 64→128
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, dropout_p=0.1),
            ResidualBlock(128, 128, dropout_p=0.1),
        )
        self.attn1 = CBAMBlock(128)

        # Stage 2: 128→256, downsample
        self.stage2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),   # 32×32
            ResidualBlock(256, 256, dropout_p=0.2),
            ResidualBlock(256, 256, dropout_p=0.2),
        )
        self.attn2 = CBAMBlock(256)

        # Stage 3: 256→512, downsample
        self.stage3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),   # 16×16
            ResidualBlock(512, 512, dropout_p=0.25),
            ResidualBlock(512, 512, dropout_p=0.25),
        )
        self.attn3 = CBAMBlock(512)

        # Stage 4: 512→512, downsample
        self.stage4 = nn.Sequential(
            ResidualBlock(512, 512, stride=2),   # 8×8
            ResidualBlock(512, 512, dropout_p=0.3),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.attn1(self.stage1(x))
        x = self.attn2(self.stage2(x))
        x = self.attn3(self.stage3(x))
        x = self.stage4(x)
        x = self.pool(x)
        return self.classifier(x)


# ──────────────────────────────────────────────────────
# Factory + parameter count utility
# ──────────────────────────────────────────────────────

ARCHITECTURES = {
    "CropCareNet_S": CropCareNet_S,
    "CropCareNet_M": CropCareNet_M,
    "CropCareNet_L": CropCareNet_L,
}


def build_model(arch_name: str, num_classes: int = NUM_CLASSES, dropout_p: float = 0.4) -> nn.Module:
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch_name}'. "
                         f"Choose from: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[arch_name](num_classes=num_classes, dropout_p=dropout_p)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}\n")

    for name, cls in ARCHITECTURES.items():
        model = cls(num_classes=NUM_CLASSES).to(device)
        x = torch.randn(2, 3, 128, 128).to(device)
        with torch.no_grad():
            out = model(x)
        params = count_parameters(model)
        print(f"{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
        assert out.shape == (2, NUM_CLASSES), f"Expected (2,{NUM_CLASSES}) got {out.shape}"
        print(f"  [OK]")
        print()
