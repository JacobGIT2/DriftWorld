"""World model variants and shared components."""

from .ddpm_world_model import DDPMWorldModel
from .dinov2 import FrozenDINOv2
from .dit_backbone import DiTBackbone
from .drift_world_model import DriftWorldModel
from .flow_world_model import FlowMatchingWorldModel
from .sd_vae import FrozenSDVAE
from .vae_world_model import VAEWorldModel


def build_world_model(cfg) -> DDPMWorldModel | FlowMatchingWorldModel | VAEWorldModel | DriftWorldModel:
    """Factory function to build a world model from config.

    Args:
        cfg: OmegaConf config with 'type' and model-specific parameters.

    Returns:
        Instantiated world model.
    """
    backbone_cfg = dict(cfg.backbone)
    model_type = cfg.type

    if model_type == "flow":
        return FlowMatchingWorldModel(
            backbone_cfg=backbone_cfg,
            sigma_min=cfg.flow.get("sigma_min", 1e-4),
            num_inference_steps=cfg.flow.get("num_inference_steps", 50),
        )
    elif model_type == "ddpm":
        return DDPMWorldModel(
            backbone_cfg=backbone_cfg,
            num_train_steps=cfg.ddpm.get("num_train_steps", 1000),
            beta_schedule=cfg.ddpm.get("beta_schedule", "linear"),
            prediction_type=cfg.ddpm.get("prediction_type", "epsilon"),
            num_inference_steps=cfg.ddpm.get("num_inference_steps", 50),
        )
    elif model_type == "vae":
        return VAEWorldModel(
            backbone_cfg=backbone_cfg,
            latent_dim=cfg.vae.get("latent_dim", 256),
        )
    elif model_type == "drift":
        return DriftWorldModel(
            backbone_cfg=backbone_cfg,
            tau=cfg.drift.get("tau", 0.1),
            num_particles=cfg.drift.get("num_particles", 1),
            dinov2_model=cfg.drift.get("dinov2_model", "dinov2_vits14"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
