"""Video reconstruction metrics following PyTorch patterns.

This module provides metric calculation utilities for video reconstruction
tasks, following patterns similar to torchmetrics. All metrics are
implemented as nn.Module classes with update/compute patterns.
"""

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from ._registry import METRIC_REGISTRY, register_metric

# Constants for tensor dimensions and channels
EXPECTED_VIDEO_DIMS = 4  # (N, C, H, W)
RGB_CHANNELS = 3


class VideoMetric(nn.Module, ABC):
    """Abstract base class for video reconstruction metrics.

    Follows torchmetrics patterns with update/compute/reset methods.
    Maintains running statistics for batch-wise computation.

    Args:
        device: Device for metric computation
        dtype: Data type for internal computations
    """

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """Initialize metric with device and dtype."""
        super().__init__()
        self._device = device
        self._dtype = dtype or torch.float32

        # Subclasses should register metric state buffers
        self._setup_state()

    @abstractmethod
    def _setup_state(self) -> None:
        """Setup metric state buffers.

        Subclasses should register buffers for accumulating statistics:

        Example:
            >>> self.register_buffer("sum_metric", torch.tensor(0.0))
            >>> self.register_buffer("num_samples", torch.tensor(0))
        """
        pass

    @abstractmethod
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update metric state with new batch.

        Args:
            predictions: Predicted values
            targets: Target values

        Raises:
            ValueError: If prediction/target shapes don't match
        """
        pass

    @abstractmethod
    def compute(self) -> Tensor:
        """Compute final metric from accumulated state.

        Returns:
            Computed metric value

        Raises:
            RuntimeError: If no samples have been processed
        """
        pass

    def reset(self) -> None:
        """Reset metric state to initial values."""
        for _buffer_name, buffer in self.named_buffers():
            if buffer.dtype.is_floating_point:
                buffer.zero_()
            else:
                buffer.fill_(0)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute metric for single batch without updating state.

        Args:
            predictions: Predicted values
            targets: Target values

        Returns:
            Metric value for this batch only
        """
        # Create temporary metric instance for single computation
        temp_metric = self.__class__()
        temp_metric.update(predictions, targets)
        return temp_metric.compute()

    def to(self, *args, **kwargs) -> "VideoMetric":
        """Move metric to device."""
        super().to(*args, **kwargs)
        # Extract device from args if provided
        if args and hasattr(args[0], "type"):  # Check if first arg is a device
            self._device = args[0]
        elif "device" in kwargs:
            self._device = kwargs["device"]
        return self


class CosineSimilarity(VideoMetric):
    """Cosine similarity metric for video reconstruction.

    Computes cosine similarity between predicted and target tensors,
    maintaining running average across batches.

    Args:
        dim: Dimension along which to compute similarity
        eps: Small epsilon for numerical stability
        device: Device for computation
        dtype: Data type for computation

    Example:
        >>> metric = CosineSimilarity(dim=-1)
        >>> pred = torch.randn(32, 512)
        >>> target = torch.randn(32, 512)
        >>> similarity = metric(pred, target)
        >>> print(similarity.item())
        0.123

        >>> # Running computation
        >>> metric.update(pred, target)
        >>> metric.update(pred2, target2)
        >>> avg_similarity = metric.compute()
    """

    def __init__(
        self,
        dim: int = -1,
        eps: float = 1e-8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize cosine similarity metric."""
        self.dim = dim
        self.eps = eps
        # These will be registered as buffers in _setup_state
        self.sum_similarity: Tensor
        self.num_samples: Tensor
        super().__init__(device=device, dtype=dtype)

    def _setup_state(self) -> None:
        """Setup running statistics buffers."""
        self.register_buffer("sum_similarity", torch.tensor(0.0, dtype=self._dtype))
        self.register_buffer("num_samples", torch.tensor(0, dtype=torch.long))

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running cosine similarity.

        Args:
            predictions: Predicted tensor of shape (N, ...)
            targets: Target tensor of same shape as predictions

        Raises:
            ValueError: If tensors have different shapes
        """
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Compute cosine similarity for this batch
        dot = (predictions * targets).sum(dim=self.dim)
        p_norm = predictions.norm(p=2, dim=self.dim)
        t_norm = targets.norm(p=2, dim=self.dim)
        similarity = dot / (p_norm * t_norm + self.eps)

        # Update running statistics
        batch_sum = similarity.sum()
        batch_size = similarity.numel()

        self.sum_similarity += batch_sum
        self.num_samples += batch_size

    def compute(self) -> Tensor:
        """Compute average cosine similarity.

        Returns:
            Average cosine similarity across all processed samples

        Raises:
            RuntimeError: If no samples processed
        """
        if self.num_samples == 0:
            raise RuntimeError("No samples processed. Call update() first.")

        return self.sum_similarity / self.num_samples

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, eps={self.eps})"


class ReconstructionMetrics(nn.Module):
    """Comprehensive reconstruction metrics for video tasks.

    Combines multiple metrics commonly used in video reconstruction:
    - Cosine similarity
    - Mean Squared Error (MSE)
    - Peak Signal-to-Noise Ratio (PSNR) - for pixel-space
    - Structural Similarity Index (SSIM) - for pixel-space

    Args:
        include_pixel_metrics: Whether to include PSNR/SSIM (for pixel reconstruction)
        cosine_dim: Dimension for cosine similarity computation
        device: Device for computation
        dtype: Data type for computation

    Example:
        >>> metrics = ReconstructionMetrics(include_pixel_metrics=True)
        >>> pred_frames = torch.randn(32, 3, 224, 224)
        >>> target_frames = torch.randn(32, 3, 224, 224)
        >>>
        >>> results = metrics(pred_frames, target_frames)
        >>> print(results)
        {'cosine_similarity': 0.123, 'mse': 0.456, 'psnr': 23.45, 'ssim': 0.789}
    """

    def __init__(
        self,
        include_pixel_metrics: bool = False,
        cosine_dim: int = -1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize comprehensive metrics.

        Args:
            include_pixel_metrics: Whether to compute PSNR/SSIM for images
            cosine_dim: Dimension for cosine similarity (-1 for last dim)
            device: Device for computation
            dtype: Data type
        """
        super().__init__()
        self.include_pixel_metrics = include_pixel_metrics

        # Core metrics always included
        self.cosine_similarity = CosineSimilarity(
            dim=cosine_dim, device=device, dtype=dtype
        )
        self.mse = MeanSquaredError(device=device, dtype=dtype)

        # Pixel-space metrics for image reconstruction
        if include_pixel_metrics:
            self.psnr = PSNR(device=device, dtype=dtype)
            self.ssim = SSIM(device=device, dtype=dtype)

    def forward(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Compute all metrics for single batch.

        Args:
            predictions: Predicted tensor
            targets: Target tensor of same shape

        Returns:
            Dictionary mapping metric names to values

        Raises:
            ValueError: If tensors have incompatible shapes
        """
        results = {}

        # Always compute core metrics
        results["cosine_similarity"] = self.cosine_similarity(
            predictions, targets
        ).item()
        results["mse"] = self.mse(predictions, targets).item()

        # Pixel metrics for image data
        if (
            self.include_pixel_metrics
            and predictions.dim() == EXPECTED_VIDEO_DIMS
            and predictions.shape[1] == RGB_CHANNELS
        ):
            results["psnr"] = self.psnr(predictions, targets).item()
            results["ssim"] = self.ssim(predictions, targets).item()

        return results

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update all running metrics.

        Args:
            predictions: Predicted tensor
            targets: Target tensor
        """
        self.cosine_similarity.update(predictions, targets)
        self.mse.update(predictions, targets)

        if (
            self.include_pixel_metrics
            and predictions.dim() == EXPECTED_VIDEO_DIMS
            and predictions.shape[1] == RGB_CHANNELS
        ):
            self.psnr.update(predictions, targets)
            self.ssim.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Compute all accumulated metrics.

        Returns:
            Dictionary of accumulated metric values
        """
        results = {
            "cosine_similarity": self.cosine_similarity.compute().item(),
            "mse": self.mse.compute().item(),
        }

        if self.include_pixel_metrics:
            results["psnr"] = self.psnr.compute().item()
            results["ssim"] = self.ssim.compute().item()

        return results

    def reset(self) -> None:
        """Reset all metrics."""
        self.cosine_similarity.reset()
        self.mse.reset()

        if self.include_pixel_metrics:
            self.psnr.reset()
            self.ssim.reset()


class MeanSquaredError(VideoMetric):
    """Mean Squared Error metric.

    Example:
        >>> mse = MeanSquaredError()
        >>> pred = torch.randn(32, 100)
        >>> target = torch.randn(32, 100)
        >>> error = mse(pred, target)
    """

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """Initialize MSE metric."""
        # These will be registered as buffers in _setup_state
        self.sum_squared_error: Tensor
        self.num_samples: Tensor
        super().__init__(device=device, dtype=dtype)

    def _setup_state(self) -> None:
        """Setup MSE state buffers."""
        self.register_buffer("sum_squared_error", torch.tensor(0.0, dtype=self._dtype))
        self.register_buffer("num_samples", torch.tensor(0, dtype=torch.long))

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update MSE computation."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        squared_error = (predictions - targets).pow(2).sum()
        num_elements = predictions.numel()

        self.sum_squared_error += squared_error
        self.num_samples += num_elements

    def compute(self) -> Tensor:
        """Compute mean squared error."""
        if self.num_samples == 0:
            raise RuntimeError("No samples processed")
        return self.sum_squared_error / self.num_samples


class PSNR(VideoMetric):
    """Peak Signal-to-Noise Ratio for image reconstruction.

    Args:
        max_val: Maximum possible pixel value (1.0 for normalized images)

    Example:
        >>> psnr = PSNR(max_val=1.0)
        >>> pred_images = torch.randn(8, 3, 224, 224)
        >>> target_images = torch.randn(8, 3, 224, 224)
        >>> psnr_value = psnr(pred_images, target_images)
    """

    def __init__(
        self,
        max_val: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize PSNR metric."""
        self.max_val = max_val
        # These will be registered as buffers in _setup_state
        self.sum_psnr: Tensor
        self.num_samples: Tensor
        super().__init__(device=device, dtype=dtype)

    def _setup_state(self) -> None:
        """Setup PSNR state buffers."""
        self.register_buffer("sum_psnr", torch.tensor(0.0, dtype=self._dtype))
        self.register_buffer("num_samples", torch.tensor(0, dtype=torch.long))

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update PSNR computation."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        # Compute MSE per image
        mse_per_image = (predictions - targets).pow(2).mean(dim=[1, 2, 3])

        # PSNR = 20 * log10(max_val / sqrt(mse))
        psnr_per_image = 20 * torch.log10(
            self.max_val / torch.sqrt(mse_per_image + 1e-8)
        )

        self.sum_psnr += psnr_per_image.sum()
        self.num_samples += predictions.shape[0]

    def compute(self) -> Tensor:
        """Compute average PSNR."""
        if self.num_samples == 0:
            raise RuntimeError("No samples processed")
        return self.sum_psnr / self.num_samples


class SSIM(VideoMetric):
    """Structural Similarity Index for image reconstruction.

    Args:
        window_size: Size of sliding window (must be odd)
        k1, k2: Constants for SSIM computation

    Example:
        >>> ssim = SSIM(window_size=11)
        >>> pred_images = torch.randn(8, 3, 224, 224)
        >>> target_images = torch.randn(8, 3, 224, 224)
        >>> ssim_value = ssim(pred_images, target_images)
    """

    def __init__(
        self,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize SSIM metric."""
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")

        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        # These will be registered as buffers in _setup_state
        self.sum_ssim: Tensor
        self.num_samples: Tensor
        super().__init__(device=device, dtype=dtype)

    def _setup_state(self) -> None:
        """Setup SSIM state buffers."""
        self.register_buffer("sum_ssim", torch.tensor(0.0, dtype=self._dtype))
        self.register_buffer("num_samples", torch.tensor(0, dtype=torch.long))

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update SSIM computation.

        Note: This is a simplified placeholder. Full SSIM implementation
        requires careful handling of sliding windows and gaussian kernels.
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        channel = predictions.size(1)
        sigma = 1.5
        c1 = (self.k1) ** 2  # Assuming max_val=1.0
        c2 = (self.k2) ** 2

        # Gaussian window
        gauss = torch.tensor(
            [
                math.exp(-((x - self.window_size // 2) ** 2) / (2 * sigma**2))
                for x in range(self.window_size)
            ],
            dtype=predictions.dtype,
            device=predictions.device,
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(
            channel, 1, self.window_size, self.window_size
        ).contiguous()

        # Means
        mu1 = functional.conv2d(
            predictions, window, padding=self.window_size // 2, groups=channel
        )
        mu2 = functional.conv2d(
            targets, window, padding=self.window_size // 2, groups=channel
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Sigmas
        sigma1_sq = (
            functional.conv2d(
                predictions * predictions,
                window,
                padding=self.window_size // 2,
                groups=channel,
            )
            - mu1_sq
        )
        sigma2_sq = (
            functional.conv2d(
                targets * targets, window, padding=self.window_size // 2, groups=channel
            )
            - mu2_sq
        )
        sigma12 = (
            functional.conv2d(
                predictions * targets,
                window,
                padding=self.window_size // 2,
                groups=channel,
            )
            - mu1_mu2
        )

        # SSIM map
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * (
            (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        )

        # Mean SSIM per image
        ssim_per_image = ssim_map.mean(dim=(1, 2, 3))

        self.sum_ssim += ssim_per_image.sum()
        self.num_samples += predictions.shape[0]

    def compute(self) -> Tensor:
        """Compute average SSIM."""
        if self.num_samples == 0:
            raise RuntimeError("No samples processed")
        return self.sum_ssim / self.num_samples


# Register metrics
register_metric("cosine_similarity", CosineSimilarity)
register_metric("reconstruction_metrics", ReconstructionMetrics)
register_metric("mse", MeanSquaredError)
register_metric("psnr", PSNR)
register_metric("ssim", SSIM)


def get_metric(name: str, **kwargs: Any) -> VideoMetric:
    """Get metric by name from registry.

    Args:
        name: Metric name
        **kwargs: Arguments for metric constructor

    Returns:
        Initialized metric

    Raises:
        KeyError: If metric not found
    """
    if name not in METRIC_REGISTRY:
        available = list(METRIC_REGISTRY.keys())
        raise KeyError(f"Metric '{name}' not found. Available: {available}")

    metric_class = METRIC_REGISTRY[name]
    return metric_class(**kwargs)
