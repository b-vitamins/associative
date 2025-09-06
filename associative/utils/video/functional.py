"""Functional video processing operations.

This module provides functional (stateless) video processing operations
following the torch.nn.functional pattern. All functions are pure and
work directly on tensors without maintaining state.

These functions are used internally by the transform classes and can
also be used directly for custom processing pipelines.
"""

import os
from typing import Literal

import torch
from decord import VideoReader
from torch import Tensor
from torch.nn import functional

# Constants for tensor dimensions and channels
VIDEO_DIMS_4D = 4  # (N, C, H, W)
VIDEO_DIMS_5D = 5  # (B, N, C, H, W)
RGB_CHANNELS = 3


def _validate_video_input(frames: Tensor, expected_dims: int) -> None:
    """Validate video tensor dimensions.

    Args:
        frames: Input video tensor
        expected_dims: Expected number of dimensions (4 or 5)

    Raises:
        RuntimeError: If dimensions don't match expected
    """
    if frames.dim() != expected_dims:
        raise RuntimeError(
            f"Expected {expected_dims}D tensor, got {frames.dim()}D. "
            f"Shape should be {'(N, C, H, W)' if expected_dims == VIDEO_DIMS_4D else '(B, N, C, H, W)'}"
        )


def _validate_load_args(
    video_path: str, num_frames: int, resolution: int, sampling_strategy: str
) -> None:
    """Validate arguments for load_video."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if sampling_strategy not in ["uniform", "random", "sequential"]:
        raise ValueError(f"Invalid sampling_strategy: {sampling_strategy}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")


def _get_sampling_indices(
    sampling_strategy: str, total_frames: int, num_frames: int
) -> Tensor:
    """Get frame indices based on sampling strategy."""
    if sampling_strategy == "uniform":
        return uniform_sample_indices(total_frames, num_frames)
    if sampling_strategy == "random":
        if num_frames > total_frames:
            raise ValueError(
                f"Requested {num_frames} frames but video has {total_frames}"
            )
        return torch.randperm(total_frames)[:num_frames].sort().values
    # sequential
    if num_frames > total_frames:
        raise ValueError(f"Requested {num_frames} frames but video has {total_frames}")
    return torch.arange(num_frames)


def load_video(
    video_path: str,
    num_frames: int,
    resolution: int = 224,
    sampling_strategy: Literal["uniform", "random", "sequential"] = "uniform",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Load video frames from file with specified sampling strategy.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        resolution: Target resolution (assumes square frames)
        sampling_strategy: How to sample frames from video
        device: Device to place tensor on
        dtype: Data type for tensor

    Returns:
        Video tensor of shape (num_frames, 3, resolution, resolution)
        Values normalized to [-1, 1] range

    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If video cannot be loaded or decoded
        ValueError: If num_frames <= 0 or resolution <= 0

    Example:
        >>> frames = load_video("video.mp4", num_frames=512, resolution=224)
        >>> print(frames.shape)
        torch.Size([512, 3, 224, 224])
    """
    # Validate arguments
    _validate_load_args(video_path, num_frames, resolution, sampling_strategy)

    # Load video reader
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
    except Exception as e:
        raise RuntimeError(f"Failed to load video: {e!s}") from e

    # Get frame indices
    indices = _get_sampling_indices(sampling_strategy, total_frames, num_frames)

    # Load and process frames
    frames_list = []
    for idx in indices:
        np_frame = vr[int(idx)].asnumpy()
        frame = torch.from_numpy(np_frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # HWC -> CHW
        frame = resize_frames(frame.unsqueeze(0), size=resolution).squeeze(0)
        frames_list.append(frame)

    frames = torch.stack(frames_list)
    frames = normalize_frames(frames, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return frames.to(device=device, dtype=dtype)


def apply_mask(
    frames: Tensor,
    mask_ratio: float = 0.5,
    mask_type: Literal["bottom_half", "random", "none"] = "bottom_half",
    mask_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply masking to video frames.

    Args:
        frames: Input video frames of shape (N, C, H, W)
        mask_ratio: Fraction of content to mask (0.0 to 1.0)
        mask_type: Type of masking strategy to apply
        mask_value: Value to use for masked regions

    Returns:
        Tuple of (masked_frames, mask) where:
        - masked_frames: Frames with masking applied, same shape as input
        - mask: Boolean mask of shape (N, H, W) where True indicates masked regions

    Raises:
        ValueError: If mask_ratio not in [0, 1] or unsupported mask_type
        RuntimeError: If input tensor has wrong dimensions

    Example:
        >>> frames = torch.randn(100, 3, 224, 224)
        >>> masked, mask = apply_mask(frames, mask_ratio=0.5, mask_type="bottom_half")
        >>> print(masked.shape, mask.shape)
        torch.Size([100, 3, 224, 224]) torch.Size([100, 224, 224])
    """
    if not (0 <= mask_ratio <= 1):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if mask_type not in ["bottom_half", "random", "none"]:
        raise ValueError(f"Invalid mask_type: {mask_type}")
    _validate_video_input(frames, VIDEO_DIMS_4D)

    num_frames, channels, height, width = frames.shape
    mask = torch.zeros(
        num_frames, height, width, dtype=torch.bool, device=frames.device
    )

    if mask_type == "bottom_half":
        mask_height = int(height * mask_ratio)
        mask[:, -mask_height:, :] = True
    elif mask_type == "random":
        prob = torch.rand(num_frames, height, width, device=frames.device)
        mask = prob < mask_ratio
    # "none" leaves mask as False

    masked_frames = frames.clone()
    # Expand mask to cover all channels
    masked_frames[mask.unsqueeze(1).expand(-1, channels, -1, -1)] = mask_value
    return masked_frames, mask


def add_noise(
    frames: Tensor,
    noise_std: float = 0.1,
    noise_type: Literal["gaussian", "uniform"] = "gaussian",
) -> Tensor:
    """Add noise to video frames or embeddings.

    Args:
        frames: Input tensor to add noise to
        noise_std: Standard deviation of noise (for gaussian) or range (for uniform)
        noise_type: Type of noise distribution

    Returns:
        Noisy tensor of same shape as input

    Raises:
        ValueError: If noise_std < 0 or unsupported noise_type

    Example:
        >>> embeddings = torch.randn(100, 512)
        >>> noisy = add_noise(embeddings, noise_std=0.05)
        >>> print(noisy.shape)
        torch.Size([100, 512])
    """
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if noise_type not in ["gaussian", "uniform"]:
        raise ValueError(f"Invalid noise_type: {noise_type}")

    if noise_type == "gaussian":
        noise = torch.randn_like(frames) * noise_std
    else:  # uniform
        noise = (torch.rand_like(frames) * 2 - 1) * noise_std

    return frames + noise


def uniform_sample_indices(
    total_frames: int, num_frames: int, start_frame: int = 0
) -> Tensor:
    """Generate uniformly spaced frame indices for video sampling.

    Args:
        total_frames: Total number of frames in source video
        num_frames: Number of frames to sample
        start_frame: Starting frame index (default 0)

    Returns:
        Long tensor of shape (num_frames,) containing frame indices

    Raises:
        ValueError: If num_frames > total_frames or negative values

    Example:
        >>> indices = uniform_sample_indices(1000, 100)
        >>> print(indices[:5])
        tensor([5, 15, 25, 35, 45])
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {start_frame}")
    if start_frame >= total_frames:
        raise ValueError(
            f"start_frame ({start_frame}) must be less than total_frames ({total_frames})"
        )
    if num_frames > total_frames:
        raise ValueError(
            f"Cannot sample {num_frames} frames from video with {total_frames} frames"
        )

    return torch.round(torch.linspace(start_frame, total_frames - 1, num_frames)).long()


def resize_frames(
    frames: Tensor,
    size: tuple[int, int] | int,
    interpolation: str = "bilinear",
    align_corners: bool = False,
) -> Tensor:
    """Resize video frames to target size.

    Args:
        frames: Input frames of shape (N, C, H, W)
        size: Target size as (height, width) or single int for square
        interpolation: Interpolation method ('bilinear', 'nearest', 'bicubic')
        align_corners: Whether to align corners in interpolation

    Returns:
        Resized frames maintaining batch and channel dimensions

    Raises:
        ValueError: If size contains non-positive values or unsupported interpolation
        RuntimeError: If input has wrong number of dimensions

    Example:
        >>> frames = torch.randn(100, 3, 128, 128)
        >>> resized = resize_frames(frames, size=224)
        >>> print(resized.shape)
        torch.Size([100, 3, 224, 224])
    """
    _validate_video_input(frames, VIDEO_DIMS_4D)
    if isinstance(size, int):
        size = (size, size)
    if any(s <= 0 for s in size):
        raise ValueError(f"Size must be positive, got {size}")
    if interpolation not in ["bilinear", "nearest", "bicubic"]:
        raise ValueError(f"Unsupported interpolation: {interpolation}")

    # Nearest doesn't support align_corners
    kwargs = {"size": size, "mode": interpolation}
    if interpolation != "nearest":
        kwargs["align_corners"] = align_corners
    return functional.interpolate(frames, **kwargs)


def normalize_frames(
    frames: Tensor,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tensor:
    """Normalize video frames channel-wise.

    Args:
        frames: Input frames of shape (N, C, H, W)
        mean: Mean values for each channel
        std: Standard deviation values for each channel

    Returns:
        Normalized frames: (frames - mean) / std

    Raises:
        ValueError: If mean/std don't match number of channels
        RuntimeError: If any std value is zero

    Example:
        >>> frames = torch.rand(100, 3, 224, 224)  # [0, 1] range
        >>> normalized = normalize_frames(frames, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        >>> print(normalized.min(), normalized.max())
        tensor(-1.) tensor(1.)
    """
    if len(mean) != RGB_CHANNELS or len(std) != RGB_CHANNELS:
        raise ValueError(
            f"Mean and std must have 3 values for RGB channels, "
            f"got mean={len(mean)}, std={len(std)}"
        )
    if any(s == 0 for s in std):
        raise RuntimeError(f"Standard deviation cannot be zero, got std={std}")

    mean_t = torch.tensor(mean, dtype=frames.dtype, device=frames.device).view(
        1, 3, 1, 1
    )
    std_t = torch.tensor(std, dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
    return (frames - mean_t) / std_t


def compute_cosine_similarity(
    predictions: Tensor, targets: Tensor, dim: int = -1, eps: float = 1e-8
) -> Tensor:
    """Compute cosine similarity between prediction and target tensors.

    Args:
        predictions: Predicted tensor
        targets: Target tensor (same shape as predictions)
        dim: Dimension along which to compute similarity
        eps: Small epsilon for numerical stability

    Returns:
        Cosine similarity values, shape depends on input dimensions and dim parameter

    Raises:
        ValueError: If tensors have different shapes
        RuntimeError: If tensors are empty or contain only zeros

    Example:
        >>> pred = torch.randn(32, 512)
        >>> target = torch.randn(32, 512)
        >>> similarity = compute_cosine_similarity(pred, target, dim=-1)
        >>> print(similarity.shape)
        torch.Size([32])
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Predictions and targets must have same shape, "
            f"got {predictions.shape} vs {targets.shape}"
        )
    if predictions.numel() == 0:
        raise RuntimeError("Cannot compute similarity on empty tensors")

    norm_p = torch.norm(predictions, p=2, dim=dim, keepdim=True)
    norm_t = torch.norm(targets, p=2, dim=dim, keepdim=True)
    dot = torch.sum(predictions * targets, dim=dim, keepdim=True)
    sim = dot / (norm_p * norm_t + eps)
    return sim.squeeze(dim)


def stack_video_frames(frames_list: list[Tensor]) -> Tensor:
    """Stack list of video frame tensors into batch dimension.

    Args:
        frames_list: List of frame tensors, each of shape (N, C, H, W)

    Returns:
        Stacked tensor of shape (B, N, C, H, W) where B = len(frames_list)

    Raises:
        ValueError: If frames have inconsistent shapes
        RuntimeError: If frames_list is empty

    Example:
        >>> frames1 = torch.randn(100, 3, 224, 224)
        >>> frames2 = torch.randn(100, 3, 224, 224)
        >>> batch = stack_video_frames([frames1, frames2])
        >>> print(batch.shape)
        torch.Size([2, 100, 3, 224, 224])
    """
    if not frames_list:
        raise RuntimeError("Cannot stack empty list of frames")

    reference_shape = frames_list[0].shape
    for i, frame in enumerate(frames_list[1:], 1):
        if frame.shape != reference_shape:
            raise ValueError(
                f"All frames must have same shape. Frame 0 has shape {reference_shape}, "
                f"but frame {i} has shape {frame.shape}"
            )
    return torch.stack(frames_list, dim=0)


def flatten_video_frames(frames: Tensor) -> Tensor:
    """Flatten video frames for processing by associative memory models.

    Args:
        frames: Video frames of shape (N, C, H, W)

    Returns:
        Flattened frames of shape (N, C*H*W)

    Raises:
        RuntimeError: If input doesn't have 4 dimensions

    Example:
        >>> frames = torch.randn(100, 3, 224, 224)
        >>> flattened = flatten_video_frames(frames)
        >>> print(flattened.shape)
        torch.Size([100, 150528])  # 3*224*224
    """
    _validate_video_input(frames, VIDEO_DIMS_4D)
    return frames.view(frames.shape[0], -1)


def reshape_to_frames(
    flattened: Tensor, channels: int = 3, height: int = 224, width: int = 224
) -> Tensor:
    """Reshape flattened vectors back to video frame format.

    Args:
        flattened: Flattened tensor of shape (N, C*H*W)
        channels: Number of channels
        height: Frame height
        width: Frame width

    Returns:
        Video frames of shape (N, C, H, W)

    Raises:
        ValueError: If flattened dimension doesn't match C*H*W

    Example:
        >>> flattened = torch.randn(100, 150528)  # 3*224*224
        >>> frames = reshape_to_frames(flattened, 3, 224, 224)
        >>> print(frames.shape)
        torch.Size([100, 3, 224, 224])
    """
    batch_size, features = flattened.shape
    expected_features = channels * height * width
    if expected_features != features:
        raise ValueError(
            f"Feature dimension {features} doesn't match expected "
            f"{channels}*{height}*{width}={expected_features}"
        )
    return flattened.view(batch_size, channels, height, width)
