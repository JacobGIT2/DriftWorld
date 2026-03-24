"""Wrapper to deploy trained PPO policy in CALVIN simulator.

Handles:
- Encoding CALVIN RGB observations through SD-VAE to get z_t
- Action chunking (policy outputs K actions, execute one per step)
- Interface matching between our policy API and CALVIN's env API
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..models.sd_vae import FrozenSDVAE
from .networks import PolicyNetwork


class CalvinPolicyWrapper:
    """Wraps a trained image-conditioned PPO policy for CALVIN evaluation.

    CALVIN provides: obs["robot_obs"] (15D), obs["rgb_obs"]["rgb_static"] (200x200x3)
    Our policy expects: z_t (4, 32, 32) latent + robot_obs (15D)
    """

    def __init__(
        self,
        policy_checkpoint: str,
        device: str = "cuda",
        chunk_size: int = 8,
        action_dim: int = 7,
        obs_dim: int = 15,
    ):
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        # Load frozen SD-VAE for encoding observations
        self.sd_vae = FrozenSDVAE().to(self.device)

        # Load policy
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            latent_channels=4,
            latent_feature_dim=128,
        ).to(self.device)

        ckpt = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()

        # Action buffer for chunking
        self.action_buffer = None
        self.chunk_idx = 0

    def reset(self):
        """Reset action buffer between episodes."""
        self.action_buffer = None
        self.chunk_idx = 0

    @torch.no_grad()
    def _encode_image(self, rgb: np.ndarray) -> torch.Tensor:
        """Encode CALVIN RGB image to SD-VAE latent.

        Args:
            rgb: (200, 200, 3) uint8 numpy array from CALVIN

        Returns:
            z: (1, 4, 32, 32) latent tensor
        """
        img = Image.fromarray(rgb)
        img = img.resize((256, 256), Image.BILINEAR)
        img_tensor = TF.to_tensor(img).to(self.device)  # (3, 256, 256) in [0, 1]
        img_tensor = img_tensor * 2 - 1  # normalize to [-1, 1]
        z = self.sd_vae.encode(img_tensor.unsqueeze(0))  # (1, 4, 32, 32)
        return z

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        """Get action from observation.

        Args:
            obs: CALVIN observation dict with:
                - obs["robot_obs"]: (15,) numpy array
                - obs["rgb_obs"]["rgb_static"]: (200, 200, 3) uint8 numpy

        Returns:
            action: (7,) numpy array (rel_actions format)
        """
        # Refill action buffer if exhausted
        if self.action_buffer is None or self.chunk_idx >= self.chunk_size:
            # Encode observation
            rgb = obs["rgb_obs"]["rgb_static"]
            z_t = self._encode_image(rgb)  # (1, 4, 32, 32)

            robot_obs = torch.tensor(
                obs["robot_obs"], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, 15)

            # Get action chunk from policy (using mean, no sampling)
            mean, _ = self.policy(z_t, robot_obs)  # (1, K, 7)
            self.action_buffer = torch.tanh(mean).squeeze(0).cpu().numpy()  # (K, 7)
            self.chunk_idx = 0

        # Return next action from buffer
        action = self.action_buffer[self.chunk_idx]
        self.chunk_idx += 1
        return action
