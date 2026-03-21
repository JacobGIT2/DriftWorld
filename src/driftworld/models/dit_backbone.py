"""DiT (Diffusion Transformer) backbone with AdaLN-Zero conditioning.

Shared backbone for all world model variants. Processes 32x32x4 SD-VAE latents
as 16x16 patch tokens with action (and optional timestep) conditioning via AdaLN-Zero.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by MLP projection."""

    def __init__(self, hidden_dim: int, max_period: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed scalar timesteps.

        Args:
            t: (B,) integer or float timesteps.

        Returns:
            Embeddings (B, hidden_dim).
        """
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class AdaLNZeroBlock(nn.Module):
    """Transformer block with Adaptive Layer Norm Zero (AdaLN-Zero) modulation.

    Conditioning signal c is projected to 6 modulation parameters per block:
    (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        # AdaLN-Zero: project conditioning to 6 modulation vectors
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Forward pass with AdaLN-Zero modulation.

        Args:
            x: (B, N, D) token sequence.
            c: (B, D) conditioning embedding.

        Returns:
            Modulated output (B, N, D).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Modulated self-attention
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * h

        # Modulated MLP
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class DiTBackbone(nn.Module):
    """DiT-style transformer backbone for world models.

    Patchifies input latent (+ optional condition latent via channel concat),
    applies transformer blocks with AdaLN-Zero conditioning from action/timestep,
    then unpatchifies back to spatial output.

    Args:
        in_channels: Channels of the primary input (e.g., 4 for noisy latent).
        cond_channels: Channels of the condition input concatenated along channel dim.
            Set to 4 to condition on z_t via channel concat. Set to 0 for no concat.
        patch_size: Patch size for patchification of 32x32 latents.
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        mlp_ratio: MLP expansion ratio.
        action_dim: Dimension of action + robot_obs vector (default 22 = 7 + 15).
        out_channels: Output channels. If None, defaults to in_channels.
        has_timestep: Whether the model receives a diffusion/flow timestep.
        robot_obs_dim: Dimension of robot_obs to predict. 0 to disable prediction head.
    """

    def __init__(
        self,
        in_channels: int = 4,
        cond_channels: int = 4,
        patch_size: int = 2,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        action_dim: int = 22,
        out_channels: int | None = None,
        has_timestep: bool = False,
        robot_obs_dim: int = 15,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels or in_channels
        self.has_timestep = has_timestep
        self.robot_obs_dim = robot_obs_dim

        total_in_channels = in_channels + cond_channels
        num_patches = (32 // patch_size) ** 2  # 256 for patch_size=2

        # Patchify: Conv2d with kernel=stride=patch_size
        self.patch_embed = nn.Conv2d(
            total_in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))

        # Action embedding: 7D -> hidden_dim
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Optional timestep embedding
        if has_timestep:
            self.time_embed = TimestepEmbedder(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )

        # Final layer: AdaLN + linear projection
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.final_proj = nn.Linear(
            hidden_dim, patch_size * patch_size * self.out_channels
        )

        # Auxiliary head: predict next robot_obs from the CLS-like pooled representation
        if robot_obs_dim > 0:
            self.robot_obs_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, robot_obs_dim),
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following DiT conventions."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Zero-init final projection
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # Zero-init all AdaLN gate/modulation outputs
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def forward(
        self,
        x: Tensor,
        z_cond: Tensor | None,
        action: Tensor,
        timestep: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, in_channels, 32, 32) primary input (noise, noisy target, etc.)
            z_cond: (B, cond_channels, 32, 32) condition latent (z_t). None if cond_channels=0.
            action: (B, action_dim) action+robot_obs vector.
            timestep: (B,) diffusion/flow timestep. Required if has_timestep=True.

        Returns:
            If robot_obs_dim > 0: (spatial_output, robot_obs_pred)
                - spatial_output: (B, out_channels, 32, 32)
                - robot_obs_pred: (B, robot_obs_dim)
            Else: spatial_output only.
        """
        B = x.shape[0]

        # Channel-concatenate condition if provided
        if self.cond_channels > 0 and z_cond is not None:
            x = torch.cat([x, z_cond], dim=1)

        # Patchify: (B, C, 32, 32) -> (B, D, 16, 16) -> (B, 256, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Build conditioning vector
        c = self.action_embed(action)
        if self.has_timestep and timestep is not None:
            c = c + self.time_embed(timestep)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Robot obs prediction from mean-pooled token features
        robot_obs_pred = None
        if self.robot_obs_dim > 0:
            robot_obs_pred = self.robot_obs_head(x.mean(dim=1))  # (B, robot_obs_dim)

        x = self.final_proj(x)  # (B, 256, patch_size^2 * out_channels)

        # Unpatchify: (B, N, P*P*C) -> (B, C, 32, 32)
        p = self.patch_size
        h = w = 32 // p
        x = x.reshape(B, h, w, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.out_channels, 32, 32)

        if robot_obs_pred is not None:
            return x, robot_obs_pred
        return x
