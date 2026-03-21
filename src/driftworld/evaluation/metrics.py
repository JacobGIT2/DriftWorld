"""Evaluation metrics for world model quality assessment."""

import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

from ..models.sd_vae import FrozenSDVAE


class WorldModelEvaluator:
    """Evaluates world model predictions against ground truth.

    Computes metrics in both latent space and pixel space (after SD-VAE decoding).
    """

    def __init__(self, sd_vae: FrozenSDVAE | None = None, device: str = "cuda"):
        self.device = torch.device(device)
        self.sd_vae = sd_vae
        if sd_vae is not None:
            self.sd_vae = sd_vae.to(self.device)

        # Pixel-space metrics
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(self.device)

        # LPIPS (lazy import to avoid dependency issues)
        self._lpips = None

    @property
    def lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex").to(self.device)
        return self._lpips

    @torch.no_grad()
    def compute_metrics(
        self,
        z_pred: Tensor,
        z_gt: Tensor,
        decode: bool = True,
        robot_obs_pred: Tensor | None = None,
        robot_obs_gt: Tensor | None = None,
    ) -> dict[str, float]:
        """Compute evaluation metrics between predicted and ground truth.

        Args:
            z_pred: (B, 4, 32, 32) predicted latents.
            z_gt: (B, 4, 32, 32) ground truth latents.
            decode: Whether to decode to pixel space for visual metrics.
            robot_obs_pred: (B, 15) predicted robot obs.
            robot_obs_gt: (B, 15) ground truth robot obs.

        Returns:
            Dict of metric name -> value.
        """
        z_pred = z_pred.to(self.device)
        z_gt = z_gt.to(self.device)

        metrics = {}

        # Latent-space MSE
        metrics["latent_mse"] = F.mse_loss(z_pred, z_gt).item()

        if decode and self.sd_vae is not None:
            # Decode to pixel space
            img_pred = self.sd_vae.decode(z_pred)
            img_gt = self.sd_vae.decode(z_gt)

            # Normalize to [0, 1]
            img_pred_01 = (img_pred + 1) / 2
            img_gt_01 = (img_gt + 1) / 2

            # Pixel MSE
            metrics["pixel_mse"] = F.mse_loss(img_pred_01, img_gt_01).item()

            # SSIM
            metrics["ssim"] = self.ssim(img_pred_01, img_gt_01).item()

            # LPIPS (expects [-1, 1])
            metrics["lpips"] = self.lpips(img_pred, img_gt).mean().item()

        # Robot obs metrics
        if robot_obs_pred is not None and robot_obs_gt is not None:
            robot_obs_pred = robot_obs_pred.to(self.device)
            robot_obs_gt = robot_obs_gt.to(self.device)
            metrics["robot_obs_mse"] = F.mse_loss(robot_obs_pred, robot_obs_gt).item()

        return metrics

    @torch.no_grad()
    def evaluate_rollout(
        self,
        model,
        z_0: Tensor,
        actions: Tensor,
        z_gt_sequence: Tensor,
        robot_obs_gt_sequence: Tensor | None = None,
    ) -> dict[str, list[float]]:
        """Evaluate multi-step autoregressive rollout.

        Args:
            model: World model with .predict(z_t, action) method.
            z_0: (1, 4, 32, 32) initial state latent.
            actions: (T, action_dim) sequence of actions (with robot_obs concatenated).
            z_gt_sequence: (T, 4, 32, 32) ground truth latent sequence.
            robot_obs_gt_sequence: (T, 15) ground truth robot obs sequence.

        Returns:
            Dict mapping metric name -> list of values per timestep.
        """
        rollout_metrics = {"latent_mse": [], "ssim": [], "lpips": []}
        if robot_obs_gt_sequence is not None:
            rollout_metrics["robot_obs_mse"] = []

        z = z_0.to(self.device)
        for t in range(actions.shape[0]):
            action = actions[t : t + 1].to(self.device)
            out = model.predict(z, action)

            # Handle models that return (z, robot_obs_pred)
            if isinstance(out, tuple):
                z, robot_obs_pred = out
            else:
                z = out
                robot_obs_pred = None

            z_gt = z_gt_sequence[t : t + 1].to(self.device)
            rob_gt = None
            if robot_obs_gt_sequence is not None:
                rob_gt = robot_obs_gt_sequence[t : t + 1].to(self.device)

            step_metrics = self.compute_metrics(
                z, z_gt,
                robot_obs_pred=robot_obs_pred,
                robot_obs_gt=rob_gt,
            )

            for key in rollout_metrics:
                if key in step_metrics:
                    rollout_metrics[key].append(step_metrics[key])

        return rollout_metrics
