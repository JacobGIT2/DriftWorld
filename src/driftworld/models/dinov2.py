"""Frozen DINOv2 feature extractor for drifting model kernel computation."""

import torch
import torch.nn as nn
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDINOv2(nn.Module):
    """Frozen DINOv2-S feature extractor returning CLS token features."""

    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # DINOv2-S: 384-dim, DINOv2-B: 768-dim
        self.output_dim = self.model.embed_dim

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Extract CLS token features.

        Args:
            images: (B, 3, 224, 224) normalized with ImageNet stats.

        Returns:
            Features (B, output_dim).
        """
        return self.model(images)


def normalize_imagenet(images: Tensor) -> Tensor:
    """Normalize images from [-1, 1] range to ImageNet normalization.

    Args:
        images: (B, 3, H, W) in [-1, 1]

    Returns:
        Images normalized with ImageNet mean/std.
    """
    # [-1, 1] -> [0, 1]
    images = (images + 1) / 2
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std
