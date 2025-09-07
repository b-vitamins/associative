"""Vision-specific modules for associative memory models.

This module provides specialized components for computer vision tasks using
associative memory transformers, including patch embedding utilities for
converting images into token sequences.

Classes:
    PatchEmbed: Image to patch embedding layer for vision transformers
"""

from collections.abc import Callable

import torch
from einops import rearrange
from torch import Tensor, nn

from .utils import Lambda


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding for vision transformers.

    Converts input images into sequences of flattened patches and projects them
    to embedding space. This is the standard approach for tokenizing images in
    vision transformer architectures.

    Attributes:
        img_size: Input image dimensions (height, width)
        patch_size: Patch dimensions (height, width)
        num_patches: Total number of patches per image
        to_patches: Function to extract and flatten patches
        proj: Linear projection layer
        norm: Optional normalization layer
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 256,
        norm_layer: Callable[..., nn.Module] | None = None,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize patch embedding layer.

        Args:
            img_size: Input image size (assumed square). Defaults to 32.
            patch_size: Size of each patch (assumed square). Defaults to 4.
            in_chans: Number of input channels. Defaults to 3.
            embed_dim: Embedding dimension for patches. Defaults to 256.
            norm_layer: Optional normalization layer constructor. Defaults to None.
            bias: Whether to use bias in linear projection. Defaults to True.
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2

        # Convert image to patches
        self.to_patches = Lambda(
            lambda x: rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                p1=patch_size,
                p2=patch_size,
            )
        )

        # Linear projection
        self.proj = nn.Linear(
            in_chans * patch_size * patch_size, embed_dim, bias=bias, **factory_kwargs
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Convert image to sequence of embedded patches.

        Args:
            x: Input image tensor of shape (batch, in_chans, height, width)

        Returns:
            Embedded patches of shape (batch, num_patches, embed_dim)
        """
        x = self.to_patches(x)
        x = self.proj(x)
        return self.norm(x)

    def from_patches(self, x: Tensor) -> Tensor:
        """Convert patches back to image format.

        Args:
            x: Patch sequence of shape (batch, num_patches, patch_features)

        Returns:
            Reconstructed image of shape (batch, channels, height, width)
        """
        h = w = int(self.num_patches**0.5)
        c = x.shape[-1] // (self.patch_size[0] * self.patch_size[1])

        return rearrange(
            x,
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=h,
            w=w,
            c=c,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
