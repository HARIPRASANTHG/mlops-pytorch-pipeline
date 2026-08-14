"""Flask inference server for the CIFAR-10 classifier.

Endpoints
---------
GET  /                    Browser UI: pick a random test image or upload your own
GET  /health              200 when the checkpoint is loaded, 503 otherwise
GET  /info                Model + runtime metadata
GET  /api/samples         Gallery of bundled CIFAR-10 test images (base64 thumbs)
POST /api/predict/random  Classify a randomly chosen test image
POST /predict             Classify an uploaded image (multipart field: "image")

The checkpoint path is resolved from MODEL_PATH, else from the training config,
else from /app/checkpoints/classifier_v1.pt. The model is loaded lazily and
re-checked on every /health call, so the serving pods can start before training
has finished and become ready the moment the checkpoint lands on the PVC.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from flask import Flask, jsonify, render_template, request
from PIL import Image

from dataset import get_inference_transform
from model import CIFAR10_CLASSES, get_model

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload ceiling

DEFAULT_CHECKPOINT = "/app/checkpoints/classifier_v1.pt"
SAMPLE_DIRS = [
    os.getenv("SAMPLE_DIR", ""),
    "/app/data/test_images",
    "data/test_images",
    str(Path(__file__).resolve().parent.parent / "data" / "test_images"),
]
SAMPLE_PATTERN = re.compile(r"^(\d+)_([a-z]+)\.png$")

_state: Dict[str, Any] = {
    "model": None,
    "class_names": CIFAR10_CLASSES,
    "checkpoint_path": None,
    "checkpoint_mtime": None,
    "metadata": {},
    "device": torch.device("cpu"),
    "error": None,
}
_transform = get_inference_transform()


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _config_checkpoint_path() -> str | None:
    for candidate in (
        os.getenv("TRAINING_CONFIG"),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
        str(Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml"),
    ):
        if candidate and Path(candidate).exists():
            cfg = yaml.safe_load(Path(candidate).read_text())
            out = cfg.get("output", {})
            if out.get("checkpoint_dir") and out.get("model_name"):
                return str(Path(out["checkpoint_dir"]) / out["model_name"])
    return None


def checkpoint_path() -> Path:
    return Path(os.getenv("MODEL_PATH") or _config_checkpoint_path() or DEFAULT_CHECKPOINT)


def load_model(force: bool = False) -> bool:
    """Load the checkpoint if present. Returns True when a model is ready."""
    path = checkpoint_path()
    if not path.exists():
        _state["model"] = None
        _state["error"] = f"checkpoint not found at {path}"
        return False

    mtime = path.stat().st_mtime
    if _state["model"] is not None and not force and _state["checkpoint_mtime"] == mtime:
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture", os.getenv("MODEL_ARCH", "resnet18"))
    num_classes = int(checkpoint.get("num_classes", 10))

    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
    _state.update(
        model=model,
        class_names=checkpoint.get("class_names", CIFAR10_CLASSES),
        checkpoint_path=str(path),
        checkpoint_mtime=mtime,
        device=device,
        error=None,
        metadata={
            "architecture": architecture,
            "num_classes": num_classes,
            "trained_epochs": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "val_loss": checkpoint.get("val_loss"),
            "device": str(device),
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    app.logger.info(json.dumps({"event": "model_loaded", **_state["metadata"]}))
    return True


# --------------------------------------------------------------------------- #
# Sample images
# --------------------------------------------------------------------------- #
def sample_dir() -> Path | None:
    for candidate in SAMPLE_DIRS:
        if candidate and Path(candidate).is_dir() and any(Path(candidate).glob("*.png")):
            return Path(candidate)
    return None


def list_samples() -> List[Path]:
    directory = sample_dir()
    return sorted(directory.glob("*.png")) if directory else []


def sample_true_label(path: Path) -> str | None:
    match = SAMPLE_PATTERN.match(path.name)
    return match.group(2) if match else None


def encode_png(image: Image.Image, size: int = 160) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").resize((size, size), Image.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_image(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    if _state["model"] is None and not load_model():
        raise RuntimeError(_state["error"] or "model not loaded")

    tensor = _transform(image.convert("RGB")).unsqueeze(0).to(_state["device"])
    started = time.perf_counter()
    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = F.softmax(logits, dim=1)[0]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    class_names = _state["class_names"]
    k = min(top_k, len(class_names))
    top_probs, top_idx = probs.topk(k)
    return {
        "predicted_class": class_names[int(top_idx[0])],
        "confidence": round(float(top_probs[0]), 4),
        "top_k": [
            {"class": class_names[int(i)], "probability": round(float(p), 4)}
            for p, i in zip(top_probs, top_idx)
        ],
        "probabilities": {
            name: round(float(probs[i]), 4) for i, name in enumerate(class_names)
        },
        "inference_ms": latency_ms,
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    ready = load_model()
    body = {
        "status": "ok" if ready else "model_unavailable",
        "model_loaded": ready,
        "checkpoint": str(checkpoint_path()),
        "samples_available": len(list_samples()),
    }
    if not ready:
        body["error"] = _state["error"]
    return jsonify(body), (200 if ready else 503)


@app.get("/info")
def info():
    load_model()
    return jsonify(
        {
            "model": _state["metadata"],
            "classes": _state["class_names"],
            "checkpoint": _state["checkpoint_path"],
            "sample_dir": str(sample_dir()) if sample_dir() else None,
            "torch_version": torch.__version__,
        }
    )


@app.get("/api/samples")
def api_samples():
    count = min(int(request.args.get("n", 12)), 40)
    files = list_samples()
    if not files:
        return jsonify({"samples": [], "message": "No sample images found. Run scripts/export_test_images.py."})
    chosen = random.sample(files, min(count, len(files)))
    return jsonify(
        {
            "samples": [
                {
                    "id": p.name,
                    "true_label": sample_true_label(p),
                    "thumbnail": encode_png(Image.open(p), 96),
                }
                for p in chosen
            ]
        }
    )


@app.post("/api/predict/random")
def api_predict_random():
    files = list_samples()
    if not files:
        return jsonify({"error": "No sample images available on this pod."}), 404

    requested = request.json.get("id") if request.is_json and request.json else None
    path = next((p for p in files if p.name == requested), None) if requested else None
    path = path or random.choice(files)

    image = Image.open(path)
    try:
        result = predict_image(image)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    true_label = sample_true_label(path)
    result.update(
        source="cifar10_test_set",
        image_id=path.name,
        true_label=true_label,
        correct=(true_label == result["predicted_class"]) if true_label else None,
        preview=encode_png(image),
    )
    return jsonify(result)


@app.post("/predict")
def predict():
    upload = request.files.get("image") or request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify({"error": "Attach an image with the form field 'image'."}), 400
    try:
        image = Image.open(io.BytesIO(upload.read()))
        image.load()
    except Exception:
        return jsonify({"error": "That file could not be read as an image."}), 400

    try:
        result = predict_image(image)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    result.update(source="upload", filename=upload.filename, resized_to="32x32")
    if request.form.get("preview") == "1":
        result["preview"] = encode_png(image)
    return jsonify(result)


load_model()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
