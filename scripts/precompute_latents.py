"""Precompute SD-VAE latents for all CALVIN frames.

Encodes all episode images through the frozen SD-VAE and saves as fp16 .pt files
alongside the original .npz files. This avoids runtime encoding during training.

Usage:
    uv run python scripts/precompute_latents.py --data_dir data/calvin/task_D_D/training
    uv run python scripts/precompute_latents.py --data_dir data/calvin/task_D_D/validation
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from driftworld.models.sd_vae import FrozenSDVAE


def precompute(data_dir: str, batch_size: int = 64, device: str = "cuda"):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all episode files
    npz_files = sorted(data_dir.glob("episode_*.npz"))
    # Filter out already-processed frames
    npz_files = [
        f for f in npz_files
        if not f.with_name(f.stem + "_latent.pt").exists()
    ]
    if not npz_files:
        print("All frames already processed.")
        return

    print(f"Processing {len(npz_files)} frames from {data_dir}")

    # Load SD-VAE
    vae = FrozenSDVAE().to(device)

    # Process in batches
    for batch_start in tqdm(range(0, len(npz_files), batch_size)):
        batch_files = npz_files[batch_start : batch_start + batch_size]
        images = []

        for npz_path in batch_files:
            data = np.load(npz_path)
            img = data["rgb_static"]  # (200, 200, 3) uint8
            img = Image.fromarray(img)
            # Resize to 256x256 and normalize to [-1, 1]
            img_tensor = TF.resize(TF.to_tensor(img), [256, 256])  # [0, 1]
            img_tensor = img_tensor * 2 - 1  # [-1, 1]
            images.append(img_tensor)

        images = torch.stack(images).to(device)

        # Encode
        with torch.no_grad():
            latents = vae.encode(images)  # (B, 4, 32, 32)

        # Save each latent as fp16
        for npz_path, latent in zip(batch_files, latents):
            save_path = npz_path.with_name(npz_path.stem + "_latent.pt")
            torch.save(latent.cpu().half(), save_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute SD-VAE latents")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    precompute(args.data_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
