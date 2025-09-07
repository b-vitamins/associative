"""Spatial masking utilities for video reconstruction tasks.

This module provides spatial masking functions for video frames,
supporting various masking patterns including directional (upper, lower, left, right),
center, random, and block-based masking.
"""

import torch
from torch import Tensor

# Dimension constants
_DIM_4D = 4


def _create_directional_mask(
    mask: Tensor, mask_ratio: float, mask_type: str, height: int, width: int
) -> None:
    """Create directional masks (lower, upper, left, right)."""
    if mask_type == "lower":
        mask_start = int(height * (1 - mask_ratio))
        mask[:, mask_start:, :, :] = 0
    elif mask_type == "upper":
        mask_end = int(height * mask_ratio)
        mask[:, :mask_end, :, :] = 0
    elif mask_type == "left":
        mask_end = int(width * mask_ratio)
        mask[:, :, :mask_end, :] = 0
    elif mask_type == "right":
        mask_start = int(width * (1 - mask_ratio))
        mask[:, :, mask_start:, :] = 0


def _create_center_mask(
    mask: Tensor, mask_ratio: float, height: int, width: int
) -> None:
    """Create center region mask."""
    h_margin = int(height * (1 - mask_ratio) / 2)
    w_margin = int(width * (1 - mask_ratio) / 2)
    mask[:, h_margin : height - h_margin, w_margin : width - w_margin, :] = 0


def _create_random_mask(  # noqa: PLR0913
    mask: Tensor,
    mask_ratio: float,
    num_frames: int,
    height: int,
    width: int,
    channels: int,
    device: torch.device,
) -> None:
    """Create random pixel masking."""
    total_pixels = height * width
    num_masked = int(total_pixels * mask_ratio)

    for frame_idx in range(num_frames):
        indices = torch.randperm(total_pixels, device=device)[:num_masked]
        h_indices = indices // width
        w_indices = indices % width

        for c in range(channels):
            mask[frame_idx, h_indices, w_indices, c] = 0


def _create_block_mask(  # noqa: PLR0913
    mask: Tensor,
    mask_ratio: float,
    block_size: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> None:
    """Create block-based masking."""
    h_blocks = height // block_size
    w_blocks = width // block_size
    total_blocks = h_blocks * w_blocks
    num_masked_blocks = int(total_blocks * mask_ratio)

    for frame_idx in range(num_frames):
        block_indices = torch.randperm(total_blocks, device=device)[:num_masked_blocks]

        for idx in block_indices:
            block_idx_int = int(idx)
            h_block = block_idx_int // w_blocks
            w_block = block_idx_int % w_blocks

            h_start = h_block * block_size
            h_end = min(h_start + block_size, height)
            w_start = w_block * block_size
            w_end = min(w_start + block_size, width)

            mask[frame_idx, h_start:h_end, w_start:w_end, :] = 0


def apply_spatial_mask(
    video: Tensor,
    mask_ratio: float = 0.5,
    mask_type: str = "lower",
    block_size: int = 10,
) -> tuple[Tensor, Tensor]:
    """Apply spatial masking to video frames.

    Args:
        video: Video tensor of shape [num_frames, height, width, channels]
        mask_ratio: Fraction of spatial area to mask (0.0 to 1.0)
        mask_type: Type of masking pattern. Options:
            - "lower": Mask lower portion of frames
            - "upper": Mask upper portion of frames
            - "left": Mask left portion of frames
            - "right": Mask right portion of frames
            - "center": Mask center region of frames
            - "random": Random pixel masking
            - "block": Block-based masking with specified block_size
        block_size: Size of blocks for block masking (only used if mask_type="block")

    Returns:
        Tuple of (masked_video, mask) where:
            - masked_video: Video with mask applied (masked regions set to 0)
            - mask: Binary mask tensor (1 for unmasked, 0 for masked)

    Example:
        >>> video = torch.rand(512, 224, 224, 3)
        >>> masked_video, mask = apply_spatial_mask(video, 0.5, "lower")
        >>> # Lower half of each frame is now masked to 0
    """
    if video.dim() != _DIM_4D:
        raise ValueError(
            f"Video must be 4D [frames, height, width, channels], got {video.dim()}D"
        )

    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    num_frames, height, width, channels = video.shape
    device = video.device

    # Create mask based on type
    mask = torch.ones_like(video)

    if mask_type in ["lower", "upper", "left", "right"]:
        _create_directional_mask(mask, mask_ratio, mask_type, height, width)
    elif mask_type == "center":
        _create_center_mask(mask, mask_ratio, height, width)
    elif mask_type == "random":
        _create_random_mask(
            mask, mask_ratio, num_frames, height, width, channels, device
        )
    elif mask_type == "block":
        _create_block_mask(
            mask, mask_ratio, block_size, num_frames, height, width, device
        )
    else:
        raise ValueError(
            f"Unknown mask_type: {mask_type}. Must be one of: "
            "lower, upper, left, right, center, random, block"
        )

    # Apply mask to video
    masked_video = video * mask

    return masked_video, mask
