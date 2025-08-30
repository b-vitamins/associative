"""Vision-specific modules for associative memory models."""

from collections.abc import Callable

import torch
from einops import rearrange
from torch import Tensor, nn

from .utils import Lambda


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding.

    Converts images into sequences of patches with learnable projection.
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
        """Convert image to sequence of embedded patches."""
        x = self.to_patches(x)
        x = self.proj(x)
        return self.norm(x)

    def from_patches(self, x: Tensor) -> Tensor:
        """Convert patches back to image format."""
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
