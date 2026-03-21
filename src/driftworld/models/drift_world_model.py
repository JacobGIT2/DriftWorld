"""Drifting world model — one-step generation via kernel-based drifting field.

Based on "Drifting Models" (arxiv 2602.04770). The generator maps noise directly
to the next-state latent in a single forward pass. Training uses a drifting field
with anti-symmetric kernel to push generated samples toward real data distribution.

Key advantage: 1-NFE inference (single forward pass), ~50x faster than diffusion
for multi-step world model rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dinov2 import FrozenDINOv2, normalize_imagenet
from .dit_backbone import DiTBackbone
from .sd_vae import FrozenSDVAE


class DriftWorldModel(nn.Module):
    """Drifting world model with one-step generation.

    Generator: f(noise, z_t, action) -> z_{t+1} in a single forward pass.

    Training loss uses a kernelized drifting field:
      1. Generate samples from the generator
      2. Compute DINOv2 features for generated and real samples
      3. Compute kernel-weighted drift toward real samples
      4. Loss = MSE(x_gen, stopgrad(x_gen + drift))

    Args:
        backbone_cfg: DiT backbone configuration.
        tau: Kernel temperature for k(x,y) = exp(-||phi(x) - phi(y)|| / tau).
        num_particles: Number of generated samples per real sample for field estimation.
        dinov2_model: DINOv2 model variant name.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        tau: float = 0.1,
        num_particles: int = 1,
        dinov2_model: str = "dinov2_vits14",
    ):
        super().__init__()
        self.tau = tau
        self.num_particles = num_particles

        # Generator: noise + z_t -> z_{t+1} (no timestep needed)
        self.generator = DiTBackbone(
            in_channels=4,
            cond_channels=4,
            has_timestep=False,
            **backbone_cfg,
        )

        # Frozen feature extractor for kernel computation
        self.dinov2 = FrozenDINOv2(model_name=dinov2_model)

        # Frozen SD-VAE for decoding latents -> images (needed for DINOv2 input)
        self.sd_vae = FrozenSDVAE()

    def generate(
        self, z_t: Tensor, action: Tensor, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Single-step generation.

        Args:
            z_t: (B, 4, 32, 32) current state latent.
            action: (B, action_dim) action + robot_obs.
            noise: (B, 4, 32, 32) input noise. If None, sampled randomly.

        Returns:
            (z_gen, robot_obs_pred).
        """
        B = z_t.shape[0]
        if noise is None:
            noise = torch.randn(B, 4, 32, 32, device=z_t.device)
        out = self.generator(noise, z_t, action)
        if isinstance(out, tuple):
            return out
        return out, None

    def _get_dinov2_features(self, z: Tensor) -> Tensor:
        """Get DINOv2 features by decoding latents to pixel space."""
        with torch.no_grad():
            images = self.sd_vae.decode(z)  # (B, 3, 256, 256) in [-1, 1]
            images_224 = F.interpolate(images, size=224, mode="bilinear", align_corners=False)
            images_norm = normalize_imagenet(images_224)
            features = self.dinov2(images_norm)
        return features

    def compute_drift_loss(
        self,
        z_t: Tensor,
        action: Tensor,
        z_tp1: Tensor,
        dinov2_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute drifting field loss.

        Returns:
            Dict with 'loss' and optionally 'robot_obs_pred'.
        """
        B = z_t.shape[0]
        P = self.num_particles

        # Generate particles
        z_gen_list = []
        robot_obs_preds = []
        for _ in range(P):
            noise = torch.randn(B, 4, 32, 32, device=z_t.device)
            z_gen, rob_pred = self.generate(z_t, action, noise)
            z_gen_list.append(z_gen)
            if rob_pred is not None:
                robot_obs_preds.append(rob_pred)
        z_gen = torch.stack(z_gen_list, dim=1)  # (B, P, 4, 32, 32)

        # DINOv2 features for generated samples
        z_gen_flat = z_gen.reshape(B * P, 4, 32, 32)
        phi_gen = self._get_dinov2_features(z_gen_flat)  # (B*P, feat_dim)
        phi_gen = phi_gen.reshape(B, P, -1)  # (B, P, feat_dim)

        # DINOv2 features for real targets
        if dinov2_target is not None:
            phi_real = dinov2_target  # (B, feat_dim)
        else:
            phi_real = self._get_dinov2_features(z_tp1)  # (B, feat_dim)
        phi_real = phi_real.unsqueeze(1)  # (B, 1, feat_dim)

        # Compute kernel: k(gen, real) = exp(-||phi_gen - phi_real|| / tau)
        diff = phi_gen - phi_real  # (B, P, feat_dim)
        dist = diff.norm(dim=-1)  # (B, P) L2 distance
        k = torch.exp(-dist / self.tau)  # (B, P)

        # Drifting field: weighted direction toward real samples
        drift_direction = z_tp1.unsqueeze(1) - z_gen  # (B, P, 4, 32, 32)
        k_expanded = k[:, :, None, None, None]  # (B, P, 1, 1, 1)

        # Normalize kernel weights
        k_normalized = k_expanded / (k_expanded.sum(dim=1, keepdim=True) + 1e-8)
        drift = (k_normalized * drift_direction).sum(dim=1)  # (B, 4, 32, 32)

        # Expand drift back to per-particle
        drift_per_particle = drift.unsqueeze(1).expand_as(z_gen)

        # Target: x_gen + drift (stop gradient)
        target = (z_gen + drift_per_particle).detach()

        # Loss
        loss = F.mse_loss(z_gen, target)

        result = {"loss": loss}
        if robot_obs_preds:
            # Average robot_obs predictions across particles
            result["robot_obs_pred"] = torch.stack(robot_obs_preds, dim=0).mean(dim=0)
        return result

    @torch.no_grad()
    def predict(self, z_t: Tensor, action: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """One-step inference — single forward pass."""
        z_gen, robot_obs_pred = self.generate(z_t, action)
        if robot_obs_pred is not None:
            return z_gen, robot_obs_pred
        return z_gen
