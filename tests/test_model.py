"""Unit tests that run on CPU in a few seconds - safe for CI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import get_inference_transform, get_transforms  # noqa: E402
from model import CIFAR10_CLASSES, SimpleCNN, count_parameters, get_model  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml"


@pytest.mark.parametrize("architecture", ["resnet18", "simple_cnn"])
def test_forward_shape(architecture: str) -> None:
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_resnet_stem_is_cifar_adapted() -> None:
    model = get_model("resnet18")
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)


def test_num_classes_is_configurable() -> None:
    model = get_model("resnet18", num_classes=5)
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 5)


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError):
        get_model("mobilenet_v99")


def test_simple_cnn_is_small() -> None:
    assert count_parameters(SimpleCNN()) < 2_000_000


def test_train_transform_output_shape() -> None:
    from PIL import Image

    img = Image.new("RGB", (32, 32), (128, 64, 32))
    assert get_transforms(train=True)(img).shape == (3, 32, 32)
    assert get_transforms(train=False)(img).shape == (3, 32, 32)


def test_inference_transform_resizes_arbitrary_images() -> None:
    from PIL import Image

    img = Image.new("RGB", (640, 480), (10, 200, 90))
    assert get_inference_transform()(img).shape == (3, 32, 32)


def test_config_has_required_keys() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    assert cfg["model"]["num_classes"] == len(CIFAR10_CLASSES)
    for section, key in [
        ("training", "epochs"),
        ("training", "batch_size"),
        ("training", "learning_rate"),
        ("training", "early_stopping_patience"),
        ("data", "data_dir"),
        ("output", "checkpoint_dir"),
        ("output", "model_name"),
    ]:
        assert key in cfg[section], f"missing {section}.{key}"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = get_model("simple_cnn")
    path = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "num_classes": 10,
            "class_names": CIFAR10_CLASSES,
        },
        path,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = get_model(payload["architecture"], payload["num_classes"])
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()
    with torch.no_grad():
        assert restored(torch.randn(1, 3, 32, 32)).shape == (1, 10)
