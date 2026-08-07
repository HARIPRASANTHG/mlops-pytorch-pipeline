"""Export a sample of CIFAR-10 test images as PNGs for the serving UI.

Filenames encode the ground-truth label (``0421_frog.png``) so the Flask app can
score its own prediction on the "Random test image" tab.

    python scripts/export_test_images.py --count 200 --data-dir ./data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataset import get_raw_test_dataset  # noqa: E402
from model import CIFAR10_CLASSES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default=None, help="defaults to <data-dir>/test_images")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or Path(args.data_dir) / "test_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_raw_test_dataset(args.data_dir, download=True)

    import random

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.count, len(dataset)))

    for idx in indices:
        image, label = dataset[idx]
        image.save(out_dir / f"{idx:05d}_{CIFAR10_CLASSES[label]}.png")

    print(f"Wrote {len(indices)} PNGs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
