"""DDPM world model using denoising diffusion with DDIM fast sampling.

Predicts noise (epsilon) or velocity (v-prediction) conditioned on current state
and action. Uses diffusers schedulers for noise schedule management.
"""

import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler
from torch import Tensor

from .dit_backbone import DiTBackbone


class DDPMWorldModel(nn.Module):
    """DDPM world model with DDIM inference.

    Training: add noise to z_{t+1} at random timestep, predict noise/velocity.
    Inference: DDIM denoising from pure noise, conditioned on (z_t, action).
    """

    def __init__(
        self,
        backbone_cfg: dict,
        num_train_steps: int = 1000,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        num_inference_steps: int = 50,
    ):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.prediction_type = prediction_type
        self.num_inference_steps = num_inference_steps

        self.backbone = DiTBackbone(
            in_channels=4,
            cond_channels=4,
            has_timestep=True,
            **backbone_cfg,
        )

        # Training noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        # Fast inference scheduler
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

    def forward(
        self, z_t: Tensor, action: Tensor, z_tp1: Tensor
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            z_t: (B, 4, 32, 32) current state latent.
            action: (B, action_dim) action + robot_obs.
            z_tp1: (B, 4, 32, 32) next state latent (target).

        Returns:
            Dict with 'pred', 'target' and optionally 'robot_obs_pred'.
        """
        B = z_t.shape[0]
        noise = torch.randn_like(z_tp1)
        timesteps = torch.randint(
            0, self.num_train_steps, (B,), device=z_t.device, dtype=torch.long
        )

        # Add noise to target
        noisy_z = self.noise_scheduler.add_noise(z_tp1, noise, timesteps)

        # Predict noise/velocity
        out = self.backbone(noisy_z, z_t, action, timesteps.float())
        if isinstance(out, tuple):
            pred, robot_obs_pred = out
        else:
            pred, robot_obs_pred = out, None

        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(z_tp1, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        result = {"pred": pred, "target": target}
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
        """Generate next-state latent via DDIM sampling.

        Returns:
            Predicted next-state latent, and optionally robot_obs_pred.
        """
        num_steps = num_steps or self.num_inference_steps
        self.inference_scheduler.set_timesteps(num_steps, device=z_t.device)

        z = torch.randn(z_t.shape[0], 4, 32, 32, device=z_t.device)
        robot_obs_pred = None

        for t in self.inference_scheduler.timesteps:
            t_batch = t.expand(z_t.shape[0]).float()
            out = self.backbone(z, z_t, action, t_batch)
            if isinstance(out, tuple):
                noise_pred, robot_obs_pred = out
            else:
                noise_pred = out
            z = self.inference_scheduler.step(noise_pred, t, z).prev_sample

        if robot_obs_pred is not None:
            return z, robot_obs_pred
        return z
