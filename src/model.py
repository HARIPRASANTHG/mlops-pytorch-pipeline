"""Model factory for the CIFAR-10 image classifier.

Two architectures are supported:

* ``resnet18``   - torchvision ResNet-18 adapted for 32x32 inputs.
* ``simple_cnn`` - a small VGG-style CNN, useful for CPU-only smoke tests.

The architecture name is read from ``configs/training_config.yaml`` so the same
image can train either model without a rebuild.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class SimpleCNN(nn.Module):
    """Small CNN baseline (~0.55M params) for 32x32x3 inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _resnet18_for_cifar(num_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet-18 with the stem rewired for 32x32 images.

    The stock torchvision stem (7x7 stride-2 conv + maxpool) throws away most of
    a CIFAR image before the first residual block. Replacing it with a 3x3
    stride-1 conv and dropping the maxpool is the standard CIFAR adaptation and
    is worth ~8 accuracy points.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
) -> nn.Module:
    """Return an initialised model for the requested architecture."""
    architecture = architecture.lower().strip()
    if architecture == "resnet18":
        return _resnet18_for_cifar(num_classes, pretrained)
    if architecture in {"simple_cnn", "simplecnn", "cnn"}:
        return SimpleCNN(num_classes)
    raise ValueError(
        f"Unknown architecture '{architecture}'. Expected 'resnet18' or 'simple_cnn'."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
