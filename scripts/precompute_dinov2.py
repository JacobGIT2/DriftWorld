"""Precompute DINOv2 features for all CALVIN frames (for drifting model).

Computes DINOv2-S CLS token features on the original images (resized to 224x224)
and saves as .pt files alongside the original .npz files.

Usage:
    uv run python scripts/precompute_dinov2.py --data_dir data/calvin/task_D_D/training
    uv run python scripts/precompute_dinov2.py --data_dir data/calvin/task_D_D/validation
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from driftworld.models.dinov2 import IMAGENET_MEAN, IMAGENET_STD, FrozenDINOv2


def precompute(data_dir: str, batch_size: int = 128, device: str = "cuda"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all episode files
    npz_files = sorted(data_dir.glob("episode_*.npz"))
    # Filter out already-processed frames
    npz_files = [
        f for f in npz_files
        if not f.with_name(f.stem + "_dinov2.pt").exists()
    ]
    if not npz_files:
        print("All frames already processed.")
        return

    print(f"Processing {len(npz_files)} frames from {data_dir}")

    # Load DINOv2
    dinov2 = FrozenDINOv2().to(device)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    for batch_start in tqdm(range(0, len(npz_files), batch_size)):
        batch_files = npz_files[batch_start : batch_start + batch_size]
        images = []

        for npz_path in batch_files:
            data = np.load(npz_path)
            img = data["rgb_static"]  # (200, 200, 3) uint8
            img = Image.fromarray(img)
            img_tensor = TF.resize(TF.to_tensor(img), [224, 224])  # [0, 1]
            images.append(img_tensor)

        images = torch.stack(images).to(device)
        # Normalize with ImageNet stats
        images = (images - mean) / std

        with torch.no_grad():
            features = dinov2(images)  # (B, 384)

        for npz_path, feat in zip(batch_files, features):
            save_path = npz_path.with_name(npz_path.stem + "_dinov2.pt")
            torch.save(feat.cpu().half(), save_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute DINOv2 features")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    precompute(args.data_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
