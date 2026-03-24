"""Policy and value networks for PPO training in imagination.

Image-conditioned: CNN encoder compresses z_t (4x32x32) into a feature vector,
concatenated with robot_obs (15D) before the MLP.
"""

import torch
import torch.nn as nn


class LatentEncoder(nn.Module):
    """CNN encoder for SD-VAE latents: (4, 32, 32) -> (feature_dim,)."""

    def __init__(self, in_channels: int = 4, feature_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),   # (32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # (128, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                               # (128, 1, 1)
            nn.Flatten(),                                          # (128,)
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Encode latent to feature vector.

        Args:
            z: (B, 4, 32, 32) SD-VAE latent

        Returns:
            features: (B, feature_dim)
        """
        return self.net(z)


class PolicyNetwork(nn.Module):
    """Image-conditioned actor: (z_t, robot_obs) -> action_chunk (K, 7D).

    Architecture:
        z_t (4, 32, 32) -> CNN encoder -> feature (128D)
        robot_obs (15D) ─────────────────────────────────┐
                                                         ├─ concat (143D) -> MLP -> action
        feature (128D) ──────────────────────────────────┘

    Outputs tanh-squashed actions matching CALVIN's rel_actions range.
    """

    def __init__(
        self,
        obs_dim: int = 15,
        action_dim: int = 7,
        chunk_size: int = 8,
        latent_channels: int = 4,
        latent_feature_dim: int = 128,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # CNN encoder for image latents
        self.latent_encoder = LatentEncoder(latent_channels, latent_feature_dim)

        # MLP on concatenated features
        combined_dim = latent_feature_dim + obs_dim
        layers = []
        in_dim = combined_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim * chunk_size)
        self.log_std = nn.Parameter(torch.zeros(action_dim * chunk_size))

    def _encode_obs(self, z_t: torch.Tensor, robot_obs: torch.Tensor) -> torch.Tensor:
        """Encode image latent + robot_obs into combined feature."""
        visual_feat = self.latent_encoder(z_t)  # (B, 128)
        return torch.cat([visual_feat, robot_obs], dim=-1)  # (B, 143)

    def forward(
        self, z_t: torch.Tensor, robot_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            z_t: (B, 4, 32, 32) image latent
            robot_obs: (B, 15) robot observations

        Returns:
            mean: (B, K, action_dim) action chunk means
            std: (B, K, action_dim) action chunk stds
        """
        B = z_t.shape[0]
        combined = self._encode_obs(z_t, robot_obs)
        h = self.backbone(combined)
        mean = self.mean_head(h).reshape(B, self.chunk_size, self.action_dim)
        std = self.log_std.exp().reshape(1, self.chunk_size, self.action_dim).expand(B, -1, -1)
        return mean, std

    def sample(
        self, z_t: torch.Tensor, robot_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities.

        Returns:
            actions: (B, K, action_dim) tanh-squashed actions
            log_probs: (B,) summed log probability across chunk
        """
        mean, std = self.forward(z_t, robot_obs)
        dist = torch.distributions.Normal(mean, std)
        raw_actions = dist.rsample()
        actions = torch.tanh(raw_actions)

        log_probs = dist.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=(-1, -2))
        return actions, log_probs

    def log_prob(
        self, z_t: torch.Tensor, robot_obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log prob of given actions."""
        mean, std = self.forward(z_t, robot_obs)
        raw_actions = torch.atanh(actions.clamp(-0.999, 0.999))
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(raw_actions) - torch.log(1 - actions.pow(2) + 1e-6)
        return log_probs.sum(dim=(-1, -2))

    def entropy(self, z_t: torch.Tensor, robot_obs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of action distribution."""
        mean, std = self.forward(z_t, robot_obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum(dim=(-1, -2))


class ValueNetwork(nn.Module):
    """Image-conditioned critic: (z_t, robot_obs) -> scalar value."""

    def __init__(
        self,
        obs_dim: int = 15,
        latent_channels: int = 4,
        latent_feature_dim: int = 128,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        self.latent_encoder = LatentEncoder(latent_channels, latent_feature_dim)

        combined_dim = latent_feature_dim + obs_dim
        layers = []
        in_dim = combined_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, robot_obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z_t: (B, 4, 32, 32) image latent
            robot_obs: (B, 15)

        Returns:
            value: (B,)
        """
        visual_feat = self.latent_encoder(z_t)
        combined = torch.cat([visual_feat, robot_obs], dim=-1)
        return self.net(combined).squeeze(-1)
