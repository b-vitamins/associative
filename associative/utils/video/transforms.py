"""Video transforms following torchvision.transforms pattern.

This module provides composable video transforms that follow PyTorch's
transform design patterns. Each transform is a callable class that
can be composed using Compose() and integrated with data loaders.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import torch
from torch import Tensor

from . import functional as video_functional

# Constants for tensor dimensions and channels
EXPECTED_VIDEO_DIMS = 4  # (N, C, H, W)
RGB_CHANNELS = 3
SIZE_TUPLE_LEN = 2  # (height, width)


class VideoTransform(ABC):
    """Abstract base class for video transforms.

    All video transforms should inherit from this class and implement
    the __call__ method. This follows PyTorch's transform pattern.
    """

    @abstractmethod
    def __call__(self, frames: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Apply transform to video frames.

        Args:
            frames: Input video tensor

        Returns:
            Transformed video tensor, or tuple of (transformed, auxiliary) data
        """
        pass

    def __repr__(self) -> str:
        """String representation of transform."""
        return f"{self.__class__.__name__}()"


class Compose:
    """Composes several video transforms together.

    This follows the same pattern as torchvision.transforms.Compose.

    Args:
        transforms: List of transforms to compose

    Example:
        >>> transform = Compose([
        ...     UniformSample(num_frames=512),
        ...     Resize(224),
        ...     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ... ])
        >>> frames = transform(input_frames)
    """

    def __init__(self, transforms: list[VideoTransform]):
        """Initialize composed transform.

        Args:
            transforms: List of transforms to apply in sequence

        Raises:
            TypeError: If any item in transforms is not callable
        """
        if not all(callable(t) for t in transforms):
            raise TypeError("All transforms must be callable")
        self.transforms = transforms

    def __call__(self, frames: Tensor) -> Tensor:
        """Apply all transforms in sequence.

        Args:
            frames: Input video tensor

        Returns:
            Transformed video tensor after applying all transforms

        Raises:
            RuntimeError: If any transform fails
        """
        for transform in self.transforms:
            result = transform(frames)
            # If transform returns a tuple, take only the first element (the tensor)
            frames = result[0] if isinstance(result, tuple) else result
        return frames

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class LoadVideo(VideoTransform):
    """Load video from file path.

    Args:
        num_frames: Number of frames to sample
        resolution: Target resolution (assumes square frames)
        sampling_strategy: How to sample frames
        device: Device to load tensor to
        dtype: Data type for tensor

    Example:
        >>> loader = LoadVideo(num_frames=512, resolution=224)
        >>> frames = loader("path/to/video.mp4")
    """

    def __init__(
        self,
        num_frames: int,
        resolution: int = 224,
        sampling_strategy: Literal["uniform", "random", "sequential"] = "uniform",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize video loader transform.

        Raises:
            ValueError: If num_frames <= 0 or resolution <= 0
        """
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
        """Load and process video.

        Args:
            video_path: Path to video file (str) or pre-loaded tensor

        Returns:
            Video frames of shape (num_frames, 3, resolution, resolution)

        Raises:
            TypeError: If video_path is neither string nor tensor
            FileNotFoundError: If video file doesn't exist
        """
        if isinstance(video_path, str):
            return video_functional.load_video(
                video_path=video_path,
                num_frames=self.num_frames,
                resolution=self.resolution,
                sampling_strategy=cast(
                    Literal["uniform", "random", "sequential"], self.sampling_strategy
                ),
                device=self.device,
                dtype=self.dtype,
            )
        if isinstance(video_path, Tensor):
            # Already loaded, just return (could add validation here)
            return video_path
        raise TypeError(f"Expected str or Tensor, got {type(video_path)}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_frames={self.num_frames}, "
            f"resolution={self.resolution}, "
            f"sampling_strategy='{self.sampling_strategy}')"
        )


class UniformSample(VideoTransform):
    """Sample frames uniformly from video sequence.

    Args:
        num_frames: Number of frames to sample
        start_frame: Starting frame index

    Example:
        >>> sampler = UniformSample(num_frames=100)
        >>> sampled_frames = sampler(input_frames)
    """

    def __init__(self, num_frames: int, start_frame: int = 0):
        """Initialize uniform sampler.

        Raises:
            ValueError: If num_frames <= 0 or start_frame < 0
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {start_frame}")

        self.num_frames = num_frames
        self.start_frame = start_frame

    def __call__(self, frames: Tensor) -> Tensor:
        """Sample frames uniformly.

        Args:
            frames: Input video tensor of shape (N, C, H, W)

        Returns:
            Sampled frames of shape (num_frames, C, H, W)

        Raises:
            ValueError: If not enough frames available
            RuntimeError: If input tensor has wrong dimensions
        """
        if frames.dim() != EXPECTED_VIDEO_DIMS:
            raise RuntimeError(f"Expected 4D tensor (N,C,H,W), got {frames.dim()}D")

        total_frames = frames.shape[0]
        if total_frames < self.num_frames:
            raise ValueError(
                f"Not enough frames: need {self.num_frames}, got {total_frames}"
            )

        indices = video_functional.uniform_sample_indices(
            total_frames, self.num_frames, self.start_frame
        )
        return frames[indices]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_frames={self.num_frames}, "
            f"start_frame={self.start_frame})"
        )


class Resize(VideoTransform):
    """Resize video frames to target size.

    Args:
        size: Target size as (height, width) or single int for square
        interpolation: Interpolation method
        align_corners: Whether to align corners

    Example:
        >>> resize = Resize(224)
        >>> resized_frames = resize(input_frames)
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        interpolation: str = "bilinear",
        align_corners: bool = False,
    ):
        """Initialize resize transform.

        Raises:
            ValueError: If size contains non-positive values
        """
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"size must be positive, got {size}")
            self.size: tuple[int, int] = (size, size)
        else:
            if len(size) != SIZE_TUPLE_LEN:
                raise ValueError(f"size must be a 2-tuple, got {len(size)} elements")
            if any(s <= 0 for s in size):
                raise ValueError(f"All size values must be positive, got {size}")
            self.size = (size[0], size[1])

        self.interpolation = interpolation
        self.align_corners = align_corners

    def __call__(self, frames: Tensor) -> Tensor:
        """Resize frames to target size.

        Args:
            frames: Input video tensor of shape (N, C, H, W)

        Returns:
            Resized frames maintaining batch and channel dimensions

        Raises:
            RuntimeError: If input tensor has wrong dimensions
        """
        if frames.dim() != EXPECTED_VIDEO_DIMS:
            raise RuntimeError(f"Expected 4D tensor (N,C,H,W), got {frames.dim()}D")

        return video_functional.resize_frames(
            frames=frames,
            size=self.size,
            interpolation=self.interpolation,
            align_corners=self.align_corners,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={self.size}, "
            f"interpolation='{self.interpolation}')"
        )


class Normalize(VideoTransform):
    """Normalize video frames channel-wise.

    Args:
        mean: Mean values for each channel
        std: Standard deviation values for each channel

    Example:
        >>> normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        >>> normalized = normalize(input_frames)
    """

    def __init__(
        self,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        """Initialize normalization transform.

        Raises:
            ValueError: If mean/std have wrong length or std contains zeros
        """
        if len(mean) != RGB_CHANNELS or len(std) != RGB_CHANNELS:
            raise ValueError("mean and std must have length 3 for RGB channels")
        if any(s == 0 for s in std):
            raise ValueError("std values cannot be zero")

        self.mean = mean
        self.std = std

    def __call__(self, frames: Tensor) -> Tensor:
        """Normalize frames channel-wise.

        Args:
            frames: Input video tensor of shape (N, C, H, W)

        Returns:
            Normalized frames: (frames - mean) / std

        Raises:
            RuntimeError: If input tensor has wrong dimensions
        """
        if frames.dim() != EXPECTED_VIDEO_DIMS:
            raise RuntimeError(f"Expected 4D tensor (N,C,H,W), got {frames.dim()}D")
        if frames.shape[1] != RGB_CHANNELS:
            raise RuntimeError(f"Expected 3 channels, got {frames.shape[1]}")

        return video_functional.normalize_frames(
            frames=frames, mean=self.mean, std=self.std
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ApplyMask(VideoTransform):
    """Apply masking to video frames.

    Args:
        mask_ratio: Fraction of content to mask (0.0 to 1.0)
        mask_type: Type of masking strategy
        mask_value: Value to use for masked regions
        return_mask: Whether to return mask along with frames

    Example:
        >>> masker = ApplyMask(mask_ratio=0.5, mask_type="bottom_half")
        >>> masked_frames = masker(input_frames)
    """

    def __init__(
        self,
        mask_ratio: float = 0.5,
        mask_type: Literal["bottom_half", "random", "none"] = "bottom_half",
        mask_value: float = 0.0,
        return_mask: bool = False,
    ):
        """Initialize masking transform.

        Raises:
            ValueError: If mask_ratio not in [0, 1] or invalid mask_type
        """
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if mask_type not in ["bottom_half", "random", "none"]:
            raise ValueError(f"Invalid mask_type: {mask_type}")

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_value = mask_value
        self.return_mask = return_mask

    def __call__(self, frames: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Apply masking to frames.

        Args:
            frames: Input video tensor of shape (N, C, H, W)

        Returns:
            If return_mask=False: Masked frames of same shape as input
            If return_mask=True: Tuple of (masked_frames, mask)

        Raises:
            RuntimeError: If input tensor has wrong dimensions
        """
        if frames.dim() != EXPECTED_VIDEO_DIMS:
            raise RuntimeError(f"Expected 4D tensor (N,C,H,W), got {frames.dim()}D")

        masked_frames, mask = video_functional.apply_mask(
            frames=frames,
            mask_ratio=self.mask_ratio,
            mask_type=cast(Literal["bottom_half", "random", "none"], self.mask_type),
            mask_value=self.mask_value,
        )

        if self.return_mask:
            return masked_frames, mask
        return masked_frames

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"mask_ratio={self.mask_ratio}, "
            f"mask_type='{self.mask_type}', "
            f"return_mask={self.return_mask})"
        )


class AddNoise(VideoTransform):
    """Add noise to video frames or embeddings.

    Args:
        noise_std: Standard deviation of noise
        noise_type: Type of noise distribution

    Example:
        >>> noiser = AddNoise(noise_std=0.05)
        >>> noisy_frames = noiser(input_frames)
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        noise_type: Literal["gaussian", "uniform"] = "gaussian",
    ):
        """Initialize noise transform.

        Raises:
            ValueError: If noise_std < 0 or invalid noise_type
        """
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if noise_type not in ["gaussian", "uniform"]:
            raise ValueError(f"Invalid noise_type: {noise_type}")

        self.noise_std = noise_std
        self.noise_type = noise_type

    def __call__(self, frames: Tensor) -> Tensor:
        """Add noise to frames.

        Args:
            frames: Input tensor (any shape)

        Returns:
            Noisy tensor of same shape as input
        """
        return video_functional.add_noise(
            frames=frames,
            noise_std=self.noise_std,
            noise_type=cast(Literal["gaussian", "uniform"], self.noise_type),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"noise_std={self.noise_std}, "
            f"noise_type='{self.noise_type}')"
        )


class ToTensor(VideoTransform):
    """Convert numpy arrays or other formats to PyTorch tensors.

    Args:
        dtype: Target data type
        device: Target device

    Example:
        >>> to_tensor = ToTensor(dtype=torch.float32)
        >>> tensor_frames = to_tensor(numpy_frames)
    """

    def __init__(
        self, dtype: torch.dtype = torch.float32, device: torch.device | None = None
    ):
        """Initialize tensor conversion transform."""
        self.dtype = dtype
        self.device = device

    def __call__(self, frames: Any) -> Tensor:
        """Convert to tensor.

        Args:
            frames: Input data (numpy array, list, etc.)

        Returns:
            PyTorch tensor

        Raises:
            RuntimeError: If conversion fails
        """
        try:
            if not isinstance(frames, Tensor):
                frames = torch.tensor(frames, dtype=self.dtype)
            else:
                frames = frames.to(dtype=self.dtype)

            if self.device is not None:
                frames = frames.to(device=self.device)

            return frames
        except Exception as e:
            raise RuntimeError(f"Failed to convert to tensor: {e}") from e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dtype={self.dtype}, device={self.device})"


class Lambda(VideoTransform):
    """Apply a user-defined lambda transform.

    Args:
        lambd: Lambda/function to apply

    Example:
        >>> # Clamp values to [-1, 1] range
        >>> clamp = Lambda(lambda x: torch.clamp(x, -1, 1))
        >>> clamped = clamp(input_frames)
    """

    def __init__(self, lambd):
        """Initialize lambda transform.

        Args:
            lambd: Callable to apply to input

        Raises:
            TypeError: If lambd is not callable
        """
        if not callable(lambd):
            raise TypeError(f"lambd must be callable, got {type(lambd)}")
        self.lambd = lambd

    def __call__(self, frames: Tensor) -> Any:
        """Apply lambda function.

        Args:
            frames: Input tensor

        Returns:
            Result of lambda function
        """
        return self.lambd(frames)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
