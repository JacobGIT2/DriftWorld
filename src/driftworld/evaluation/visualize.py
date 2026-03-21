"""Visualization utilities for world model predictions."""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..models.sd_vae import FrozenSDVAE


def tensor_to_numpy_image(img: Tensor) -> np.ndarray:
    """Convert (3, H, W) tensor in [-1, 1] to (H, W, 3) uint8 numpy array."""
    img = ((img + 1) / 2).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def make_comparison_grid(
    sd_vae: FrozenSDVAE,
    z_t: Tensor,
    z_gt: Tensor,
    predictions: dict[str, Tensor],
    save_path: str | Path,
    num_samples: int = 4,
):
    """Create side-by-side comparison grid: input | GT | model predictions.

    Args:
        sd_vae: SD-VAE decoder.
        z_t: (B, 4, 32, 32) input state latents.
        z_gt: (B, 4, 32, 32) ground truth next-state latents.
        predictions: Dict of model_name -> (B, 4, 32, 32) predicted latents.
        save_path: Where to save the figure.
        num_samples: Number of rows to show.
    """
    num_cols = 2 + len(predictions)  # input + GT + models
    num_samples = min(num_samples, z_t.shape[0])

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    col_names = ["Input (t)", "GT (t+1)"] + list(predictions.keys())

    for row in range(num_samples):
        # Decode input
        img_input = sd_vae.decode(z_t[row : row + 1])[0]
        axes[row, 0].imshow(tensor_to_numpy_image(img_input))

        # Decode GT
        img_gt = sd_vae.decode(z_gt[row : row + 1])[0]
        axes[row, 1].imshow(tensor_to_numpy_image(img_gt))

        # Decode predictions
        for col_idx, (name, z_pred) in enumerate(predictions.items()):
            img_pred = sd_vae.decode(z_pred[row : row + 1])[0]
            axes[row, 2 + col_idx].imshow(tensor_to_numpy_image(img_pred))

    for col_idx, name in enumerate(col_names):
        axes[0, col_idx].set_title(name, fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def make_rollout_video(
    sd_vae: FrozenSDVAE,
    model,
    z_0: Tensor,
    actions: Tensor,
    save_path: str | Path,
    fps: int = 10,
    gt_latents: Tensor | None = None,
):
    """Generate autoregressive rollout video.

    Args:
        sd_vae: SD-VAE decoder.
        model: World model with .predict(z_t, action) method.
        z_0: (1, 4, 32, 32) initial state latent.
        actions: (T, action_dim) action sequence.
        save_path: Output video path (.mp4 or .gif).
        fps: Frames per second.
        gt_latents: (T, 4, 32, 32) optional ground truth for side-by-side.
    """
    frames = []
    z = z_0

    # Initial frame
    img = sd_vae.decode(z)[0]
    frame = tensor_to_numpy_image(img)

    if gt_latents is not None:
        # Side-by-side: predicted | GT
        gt_img = sd_vae.decode(z_0)[0]
        gt_frame = tensor_to_numpy_image(gt_img)
        frame = np.concatenate([frame, gt_frame], axis=1)
    frames.append(frame)

    for t in range(actions.shape[0]):
        action = actions[t : t + 1].to(z.device)
        z = model.predict(z, action)

        img = sd_vae.decode(z)[0]
        frame = tensor_to_numpy_image(img)

        if gt_latents is not None:
            gt_img = sd_vae.decode(gt_latents[t : t + 1].to(z.device))[0]
            gt_frame = tensor_to_numpy_image(gt_img)
            frame = np.concatenate([frame, gt_frame], axis=1)

        frames.append(frame)

    save_path = Path(save_path)
    if save_path.suffix == ".gif":
        imageio.mimwrite(str(save_path), frames, fps=fps, loop=0)
    else:
        imageio.mimwrite(str(save_path), frames, fps=fps)
