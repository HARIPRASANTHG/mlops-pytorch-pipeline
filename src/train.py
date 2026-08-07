"""CIFAR-10 training entrypoint.

Config resolution order (first hit wins):
  1. ``--config <path>`` CLI argument
  2. ``TRAINING_CONFIG`` environment variable
  3. ``/app/configs/training_config.yaml``   (ConfigMap mount inside Kubernetes)
  4. ``configs/training_config.yaml``        (local checkout)

Every metric line is emitted as a single JSON object on stdout so that
``kubectl logs`` output can be piped straight into jq or a log collector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import CIFAR10_CLASSES, count_parameters, get_model

CONFIG_CANDIDATES = [
    Path("/app/configs/training_config.yaml"),
    Path("configs/training_config.yaml"),
    Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml",
]


def log(**payload: Any) -> None:
    """Emit one structured JSON line to stdout."""
    payload.setdefault("ts", round(time.time(), 3))
    print(json.dumps(payload), flush=True)


def resolve_config_path(cli_path: str | None = None) -> Path:
    if cli_path:
        return Path(cli_path)
    env_path = os.getenv("TRAINING_CONFIG")
    if env_path:
        return Path(env_path)
    for candidate in CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No training config found. Pass --config or set TRAINING_CONFIG."
    )


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        log(event="cuda_unavailable", message="CUDA requested but not available, using CPU")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    use_amp = scaler is not None and device.type == "cuda"

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier")
    parser.add_argument("--config", default=None, help="Path to training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    training_cfg = config["training"]
    output_cfg = config["output"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.device is not None:
        training_cfg["device"] = args.device

    set_seed(int(training_cfg.get("seed", 42)))
    device = select_device(training_cfg.get("device", "auto"))
    use_amp = bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"

    log(
        event="run_start",
        config_path=str(config_path),
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        mixed_precision=use_amp,
        architecture=model_cfg["architecture"],
        epochs=training_cfg["epochs"],
        batch_size=training_cfg["batch_size"],
    )

    model = get_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
        pretrained=bool(model_cfg.get("pretrained", False)),
    ).to(device)
    log(event="model_ready", trainable_parameters=count_parameters(model))

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=training_cfg["batch_size"],
        num_workers=int(training_cfg.get("num_workers", 2)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training_cfg["epochs"]))
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0))
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    patience = int(training_cfg["early_stopping_patience"])

    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / output_cfg["model_name"]
    history: list[dict] = []

    for epoch in range(int(training_cfg["epochs"])):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(entry)
        log(**entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "architecture": model_cfg["architecture"],
                    "num_classes": model_cfg["num_classes"],
                    "class_names": CIFAR10_CLASSES,
                },
                save_path,
            )
            log(event="checkpoint_saved", path=str(save_path), val_accuracy=round(val_acc, 4))
        else:
            patience_counter += 1
            log(event="no_improvement", patience_counter=patience_counter, patience=patience)
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch + 1)
                break

    metrics_path = checkpoint_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "best_val_loss": round(best_val_loss, 4),
                "best_val_accuracy": round(best_val_acc, 4),
                "device": str(device),
            },
            indent=2,
        )
    )
    log(
        event="training_complete",
        best_val_loss=round(best_val_loss, 4),
        best_val_accuracy=round(best_val_acc, 4),
        checkpoint=str(save_path),
        metrics=str(metrics_path),
    )


if __name__ == "__main__":
    main()
