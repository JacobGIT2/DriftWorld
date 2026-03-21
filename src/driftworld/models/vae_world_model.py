"""VAE world model — encoder-decoder with latent bottleneck.

Encodes (z_t, action) to a Gaussian latent, decodes back to z_{t+1}.
Serves as a simple baseline; typically produces blurrier outputs than
diffusion/flow/drifting but trains fastest.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .dit_backbone import DiTBackbone


class VAEWorldModel(nn.Module):
    """VAE world model with DiT encoder and convolutional decoder.

    Encoder: DiT processes z_t with action conditioning, output is spatially pooled
             then projected to mu and logvar.
    Decoder: project sampled latent to spatial -> ConvTranspose layers -> z_{t+1}.
    """

    def __init__(
        self,
        backbone_cfg: dict,
        latent_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: DiT backbone outputs (B, out_channels, 32, 32)
        # Disable robot_obs head on encoder — we predict it from decoder instead
        enc_cfg = dict(backbone_cfg)
        enc_cfg["robot_obs_dim"] = 0
        out_channels = enc_cfg.get("out_channels", 4) or 4

        self.encoder = DiTBackbone(
            in_channels=4,
            cond_channels=0,
            has_timestep=False,
            **enc_cfg,
        )
        # Spatial pooling + projection to latent params
        enc_flat_dim = out_channels * 32 * 32  # 4096 for 4 channels
        self.enc_proj = nn.Linear(enc_flat_dim, backbone_cfg["hidden_dim"])
        self.to_mu = nn.Linear(backbone_cfg["hidden_dim"], latent_dim)
        self.to_logvar = nn.Linear(backbone_cfg["hidden_dim"], latent_dim)

        # Decoder: latent -> spatial -> upsample to (4, 32, 32)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.SiLU(),
            Reshape(-1, 256, 4, 4),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> 8x8
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 16x16
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x32
            nn.SiLU(),
            nn.Conv2d(32, 4, 3, padding=1),  # -> 4 channels
        )

        # Action embedding for decoder conditioning (added to latent)
        action_dim = backbone_cfg.get("action_dim", 22)
        self.action_to_latent = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Robot obs prediction head
        robot_obs_dim = backbone_cfg.get("robot_obs_dim", 15)
        self.robot_obs_dim = robot_obs_dim
        if robot_obs_dim > 0:
            self.robot_obs_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, robot_obs_dim),
            )

    def encode(self, z_t: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """Encode current state to posterior parameters.

        Returns:
            (mu, logvar) each (B, latent_dim).
        """
        out = self.encoder(z_t, None, action)  # (B, 4, 32, 32)
        flat = out.reshape(out.shape[0], -1)  # (B, 4096)
        h = self.enc_proj(flat)  # (B, hidden_dim)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, latent: Tensor, action: Tensor) -> tuple[Tensor, Tensor | None]:
        """Decode latent + action to next-state prediction.

        Returns:
            (z_pred, robot_obs_pred).
        """
        action_emb = self.action_to_latent(action)
        combined = latent + action_emb
        z_pred = self.decoder(combined)

        robot_obs_pred = None
        if self.robot_obs_dim > 0:
            robot_obs_pred = self.robot_obs_head(combined)

        return z_pred, robot_obs_pred

    def forward(
        self, z_t: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Full forward pass.

        Returns:
            (z_pred, mu, logvar, robot_obs_pred).
        """
        mu, logvar = self.encode(z_t, action)
        latent = self.reparameterize(mu, logvar)
        z_pred, robot_obs_pred = self.decode(latent, action)
        return z_pred, mu, logvar, robot_obs_pred

    @torch.no_grad()
    def predict(self, z_t: Tensor, action: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Inference: use mu directly (no sampling noise)."""
        mu, _ = self.encode(z_t, action)
        z_pred, robot_obs_pred = self.decode(mu, action)
        if robot_obs_pred is not None:
            return z_pred, robot_obs_pred
        return z_pred


class Reshape(nn.Module):
    """Reshape layer for use in nn.Sequential."""

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.shape)
