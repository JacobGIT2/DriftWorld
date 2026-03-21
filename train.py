"""Training entry point for DriftWorld world models.

Usage:
    # Train flow matching (default)
    uv run python train.py

    # Train specific model
    uv run python train.py model.type=ddpm
    uv run python train.py model.type=vae
    uv run python train.py model.type=drift dataset.load_dinov2_features=true training.batch_size=32

    # Override training params
    uv run python train.py model.type=flow training.lr=5e-5 training.total_steps=100000
"""

import hydra
from omegaconf import DictConfig

from driftworld.training.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
