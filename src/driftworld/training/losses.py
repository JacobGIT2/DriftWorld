"""Per-model loss computation for world model training."""

import torch.nn.functional as F
from torch import Tensor


def compute_loss(
    model,
    z_t: Tensor,
    action: Tensor,
    z_tp1: Tensor,
    batch: dict,
    model_type: str,
    kl_weight: float = 1e-4,
    robot_obs_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Dispatch loss computation to the appropriate model type.

    Args:
        model: World model instance.
        z_t: (B, 4, 32, 32) current state latent.
        action: (B, action_dim) action + robot_obs concatenated.
        z_tp1: (B, 4, 32, 32) target next-state latent.
        batch: Full batch dict (may contain dinov2_tp1, robot_obs_tp1).
        model_type: One of 'flow', 'ddpm', 'vae', 'drift'.
        kl_weight: KL divergence weight for VAE.
        robot_obs_weight: Weight for robot_obs prediction loss.

    Returns:
        Dict with 'loss' key (scalar) and optional per-component losses.
    """
    robot_obs_tp1 = batch.get("robot_obs_tp1")
    if robot_obs_tp1 is not None:
        robot_obs_tp1 = robot_obs_tp1.to(z_t.device)

    if model_type == "flow":
        output = model(z_t, action, z_tp1)
        loss = F.mse_loss(output["pred"], output["target"])
        result = {"loss": loss, "gen_loss": loss}
        result = _add_robot_obs_loss(result, output, robot_obs_tp1, robot_obs_weight)
        return result

    elif model_type == "ddpm":
        output = model(z_t, action, z_tp1)
        loss = F.mse_loss(output["pred"], output["target"])
        result = {"loss": loss, "gen_loss": loss}
        result = _add_robot_obs_loss(result, output, robot_obs_tp1, robot_obs_weight)
        return result

    elif model_type == "vae":
        z_pred, mu, logvar, robot_obs_pred = model(z_t, action)
        recon_loss = F.mse_loss(z_pred, z_tp1)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = recon_loss + kl_weight * kl_loss
        result = {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
        output = {"robot_obs_pred": robot_obs_pred} if robot_obs_pred is not None else {}
        result = _add_robot_obs_loss(result, output, robot_obs_tp1, robot_obs_weight)
        return result

    elif model_type == "drift":
        dinov2_target = batch.get("dinov2_tp1")
        if dinov2_target is not None:
            dinov2_target = dinov2_target.to(z_t.device)
        output = model.compute_drift_loss(z_t, action, z_tp1, dinov2_target)
        result = {"loss": output["loss"], "gen_loss": output["loss"]}
        result = _add_robot_obs_loss(result, output, robot_obs_tp1, robot_obs_weight)
        return result

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _add_robot_obs_loss(
    result: dict,
    output: dict,
    robot_obs_tp1: Tensor | None,
    weight: float,
) -> dict:
    """Add robot_obs prediction loss if available."""
    robot_obs_pred = output.get("robot_obs_pred")
    if robot_obs_pred is not None and robot_obs_tp1 is not None:
        rob_loss = F.mse_loss(robot_obs_pred, robot_obs_tp1)
        result["robot_obs_loss"] = rob_loss
        result["loss"] = result["loss"] + weight * rob_loss
    return result
