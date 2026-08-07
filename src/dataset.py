"""CIFAR-10 data loading, transforms and inference preprocessing.

The same normalisation constants are used at training time and at serving time,
which is why the inference transform lives here rather than inside serve.py.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
IMAGE_SIZE = 32


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(IMAGE_SIZE, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def get_inference_transform() -> transforms.Compose:
    """Transform for arbitrary uploaded images.

    Uploads are any size, so they are resized to 32x32 before the same
    normalisation the model saw during training.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    pin_memory: bool | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build the CIFAR-10 train/val dataloaders."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def get_raw_test_dataset(data_dir: str, download: bool = True) -> datasets.CIFAR10:
    """Test split without transforms - used to export sample PNGs for the UI."""
    return datasets.CIFAR10(root=data_dir, train=False, download=download)
