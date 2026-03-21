"""Frozen Stable Diffusion VAE wrapper for encoding/decoding images to/from latent space."""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torch import Tensor


class FrozenSDVAE(nn.Module):
    """Wraps the SD-VAE (stabilityai/sd-vae-ft-mse) for encoding images to 32x32x4 latents."""

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.scaling_factor = self.vae.config.scaling_factor  # 0.18215

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        """Encode images to latent space.

        Args:
            images: (B, 3, 256, 256) in [-1, 1]

        Returns:
            Latents (B, 4, 32, 32) scaled by the VAE scaling factor.
        """
        posterior = self.vae.encode(images).latent_dist
        z = posterior.sample() * self.scaling_factor
        return z

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode latents back to images.

        Args:
            z: (B, 4, 32, 32) scaled latents

        Returns:
            Images (B, 3, 256, 256) in [-1, 1].
        """
        z = z / self.scaling_factor
        return self.vae.decode(z).sample
