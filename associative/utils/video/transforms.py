"""Video transforms following torchvision.transforms pattern.

Composable video transforms that follow PyTorch's transform design patterns.
Each transform is a callable class that can be composed using Compose().
"""

from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor

from . import functional as F

# Constants for tensor dimensions and channels
VIDEO_DIMS_4D = 4  # (N, C, H, W)
RGB_CHANNELS = 3


class Compose:
    """Compose multiple transforms together.

    Args:
        transforms: List of callables to apply in sequence

    Example:
        >>> transform = Compose([
        ...     UniformSample(num_frames=100),
        ...     Resize(224),
        ...     Normalize()
        ... ])
    """

    def __init__(self, transforms: list[Callable]):
        if not transforms or not all(callable(t) for t in transforms):
            raise TypeError("All transforms must be callable")
        self.transforms = transforms

    def __call__(self, frames: Any) -> Any:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            result = transform(frames)
            # Extract tensor from tuple if needed
            frames = result[0] if isinstance(result, tuple) else result
        return frames

    def __repr__(self) -> str:
        lines = ["Compose("]
        lines.extend(f"    {t}" for t in self.transforms)
        lines.append(")")
        return "\n".join(lines)


class LoadVideo:
    """Load video from file path or pass through tensor.

    Args:
        num_frames: Number of frames to sample
        resolution: Target resolution (square)
        sampling_strategy: How to sample frames
        device: Device for tensor
        dtype: Data type for tensor
    """

    def __init__(
        self,
        num_frames: int,
        resolution: int = 224,
        sampling_strategy: Literal["uniform", "random", "sequential"] = "uniform",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        self.num_frames = num_frames
        self.resolution = resolution
        self.sampling_strategy = sampling_strategy
        self.device = device
        self.dtype = dtype

    def __call__(self, video_path: str | Tensor) -> Tensor:
        """Load video from path or pass through tensor."""
        if isinstance(video_path, str):
            return F.load_video(
                video_path=video_path,
                num_frames=self.num_frames,
                resolution=self.resolution,
                sampling_strategy=self.sampling_strategy,  # type: ignore[arg-type]
                device=self.device,
                dtype=self.dtype,
            )
        if isinstance(video_path, Tensor):
            return video_path
        raise TypeError(f"Expected str or Tensor, got {type(video_path)}")

    def __repr__(self) -> str:
        return (
            f"LoadVideo(num_frames={self.num_frames}, "
            f"resolution={self.resolution}, "
            f"sampling_strategy='{self.sampling_strategy}')"
        )


class UniformSample:
    """Sample frames uniformly from video.

    Args:
        num_frames: Number of frames to sample
        start_frame: Starting frame index
    """

    def __init__(self, num_frames: int, start_frame: int = 0):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {start_frame}")

        self.num_frames = num_frames
        self.start_frame = start_frame

    def __call__(self, frames: Tensor) -> Tensor:
        """Sample frames uniformly."""
        if frames.dim() != VIDEO_DIMS_4D:
            raise RuntimeError(
                f"Expected {VIDEO_DIMS_4D}D tensor (N,C,H,W), got {frames.dim()}D"
            )

        total_frames = frames.shape[0]
        if total_frames < self.num_frames:
            raise ValueError(
                f"Not enough frames: need {self.num_frames}, got {total_frames}"
            )

        indices = F.uniform_sample_indices(
            total_frames, self.num_frames, self.start_frame
        )
        return frames[indices]

    def __repr__(self) -> str:
        return f"UniformSample(num_frames={self.num_frames}, start_frame={self.start_frame})"


class Resize:
    """Resize video frames.

    Args:
        size: Target size as int (square) or (height, width) tuple
        interpolation: Interpolation method
        align_corners: Whether to align corners
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: str = "bilinear",
        align_corners: bool = False,
    ):
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"size must be positive, got {size}")
            self.size = (size, size)
        else:
            if any(s <= 0 for s in size):
                raise ValueError(f"All size values must be positive, got {size}")
            self.size = tuple(size)

        self.interpolation = interpolation
        self.align_corners = align_corners

    def __call__(self, frames: Tensor) -> Tensor:
        """Resize frames."""
        if frames.dim() != VIDEO_DIMS_4D:
            raise RuntimeError(
                f"Expected {VIDEO_DIMS_4D}D tensor (N,C,H,W), got {frames.dim()}D"
            )

        return F.resize_frames(
            frames=frames,
            size=self.size,  # type: ignore[arg-type]
            interpolation=self.interpolation,
            align_corners=self.align_corners,
        )

    def __repr__(self) -> str:
        return f"Resize(size={self.size}, interpolation='{self.interpolation}')"


class Normalize:
    """Normalize video frames channel-wise.

    Args:
        mean: Per-channel mean values
        std: Per-channel standard deviation values
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        if len(mean) != RGB_CHANNELS or len(std) != RGB_CHANNELS:
            raise ValueError(
                f"mean and std must have length {RGB_CHANNELS} for RGB channels"
            )
        if any(s == 0 for s in std):
            raise ValueError("std values cannot be zero")

        self.mean = mean
        self.std = std

    def __call__(self, frames: Tensor) -> Tensor:
        """Normalize frames."""
        if frames.dim() != VIDEO_DIMS_4D:
            raise RuntimeError(
                f"Expected {VIDEO_DIMS_4D}D tensor (N,C,H,W), got {frames.dim()}D"
            )
        if frames.shape[1] != RGB_CHANNELS:
            raise RuntimeError(
                f"Expected {RGB_CHANNELS} channels, got {frames.shape[1]}"
            )

        return F.normalize_frames(frames=frames, mean=self.mean, std=self.std)

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class ApplyMask:
    """Apply masking to video frames.

    Args:
        mask_ratio: Fraction to mask (0.0 to 1.0)
        mask_type: Masking strategy
        mask_value: Value for masked regions
        return_mask: Whether to return mask with frames
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        mask_type: Literal["bottom_half", "random", "none"] = "bottom_half",
        mask_value: float = 0.0,
        return_mask: bool = False,
    ):
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if mask_type not in ["bottom_half", "random", "none"]:
            raise ValueError(f"Invalid mask_type: {mask_type}")

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_value = mask_value
        self.return_mask = return_mask

    def __call__(self, frames: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Apply masking."""
        if frames.dim() != VIDEO_DIMS_4D:
            raise RuntimeError(
                f"Expected {VIDEO_DIMS_4D}D tensor (N,C,H,W), got {frames.dim()}D"
            )

        masked_frames, mask = F.apply_mask(
            frames=frames,
            mask_ratio=self.mask_ratio,
            mask_type=self.mask_type,  # type: ignore[arg-type]
            mask_value=self.mask_value,
        )

        return (masked_frames, mask) if self.return_mask else masked_frames

    def __repr__(self) -> str:
        return (
            f"ApplyMask(mask_ratio={self.mask_ratio}, "
            f"mask_type='{self.mask_type}', return_mask={self.return_mask})"
        )


class AddNoise:
    """Add noise to tensors.

    Args:
        noise_std: Standard deviation of noise
        noise_type: Type of noise distribution
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        noise_type: Literal["gaussian", "uniform"] = "gaussian",
    ):
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if noise_type not in ["gaussian", "uniform"]:
            raise ValueError(f"Invalid noise_type: {noise_type}")

        self.noise_std = noise_std
        self.noise_type = noise_type

    def __call__(self, frames: Tensor) -> Tensor:
        """Add noise to frames."""
        return F.add_noise(
            frames=frames,
            noise_std=self.noise_std,
            noise_type=self.noise_type,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:
        return f"AddNoise(noise_std={self.noise_std}, noise_type='{self.noise_type}')"


class ToTensor:
    """Convert data to PyTorch tensor.

    Args:
        dtype: Target data type
        device: Target device
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        self.dtype = dtype
        self.device = device

    def __call__(self, frames: Any) -> Tensor:
        """Convert to tensor."""
        try:
            if isinstance(frames, Tensor):
                result = frames.to(dtype=self.dtype)
            else:
                result = torch.tensor(frames, dtype=self.dtype)

            if self.device is not None:
                result = result.to(device=self.device)

            return result
        except Exception as e:
            raise RuntimeError(f"Failed to convert to tensor: {e}") from e

    def __repr__(self) -> str:
        return f"ToTensor(dtype={self.dtype}, device={self.device})"


class Lambda:
    """Apply user-defined function.

    Args:
        lambd: Function to apply
    """

    def __init__(self, lambd: Callable):
        if not callable(lambd):
            raise TypeError(f"lambd must be callable, got {type(lambd)}")
        self.lambd = lambd

    def __call__(self, frames: Any) -> Any:
        """Apply lambda function."""
        return self.lambd(frames)

    def __repr__(self) -> str:
        return "Lambda()"
