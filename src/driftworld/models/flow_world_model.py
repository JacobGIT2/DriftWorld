"""Flow Matching world model using Conditional Flow Matching (CFM).

Learns a velocity field along straight-line (optimal transport) interpolation
paths between noise and target next-state latents.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .dit_backbone import DiTBackbone


class FlowMatchingWorldModel(nn.Module):
    """Flow Matching world model.

    Training: sample t ~ U[0,1], interpolate z_interp = (1-t)*noise + t*z_{t+1},
    predict velocity v = z_{t+1} - noise.

    Inference: Euler ODE integration from noise to generated sample.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        sigma_min: float = 1e-4,
        num_inference_steps: int = 50,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.num_inference_steps = num_inference_steps
        self.backbone = DiTBackbone(
            in_channels=4,
            cond_channels=4,
            has_timestep=True,
            **backbone_cfg,
        )

    def forward(
        self, z_t: Tensor, action: Tensor, z_tp1: Tensor
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            z_t: (B, 4, 32, 32) current state latent.
            action: (B, action_dim) action + robot_obs concatenated.
            z_tp1: (B, 4, 32, 32) next state latent (target).

        Returns:
            Dict with 'pred', 'target' velocities and 'robot_obs_pred'.
        """
        B = z_t.shape[0]
        t = torch.rand(B, device=z_t.device)
        z_0 = torch.randn_like(z_tp1)  # noise source
        z_1 = z_tp1  # target

        # Straight-line interpolation (OT path)
        t_expand = t[:, None, None, None]
        z_interp = (1 - t_expand) * z_0 + t_expand * z_1

        # Target velocity
        target_v = z_1 - z_0

        # Predict velocity
        out = self.backbone(z_interp, z_t, action, t)
        if isinstance(out, tuple):
            pred_v, robot_obs_pred = out
        else:
            pred_v, robot_obs_pred = out, None

        result = {"pred": pred_v, "target": target_v}
        if robot_obs_pred is not None:
            result["robot_obs_pred"] = robot_obs_pred
        return result

    @torch.no_grad()
    def predict(
        self,
        z_t: Tensor,
        action: Tensor,
        num_steps: int | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Generate next-state latent via Euler ODE integration.

        Args:
            z_t: (B, 4, 32, 32) current state latent.
            action: (B, action_dim) action + robot_obs.
            num_steps: Number of Euler steps.

        Returns:
            Predicted next-state latent (B, 4, 32, 32), and optionally robot_obs_pred.
        """
        num_steps = num_steps or self.num_inference_steps
        dt = 1.0 / num_steps
        z = torch.randn(z_t.shape[0], 4, 32, 32, device=z_t.device)
        robot_obs_pred = None

        for i in range(num_steps):
            t = torch.full((z_t.shape[0],), i * dt, device=z_t.device)
            out = self.backbone(z, z_t, action, t)
            if isinstance(out, tuple):
                v, robot_obs_pred = out
            else:
                v = out
            z = z + v * dt

        if robot_obs_pred is not None:
            return z, robot_obs_pred
        return z
