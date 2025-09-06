"""Video reconstruction metrics.

Provides metrics for evaluating video and feature reconstruction quality,
with a clean API following PyTorch conventions.
"""

from typing import Any, Literal

import torch
from torch import Tensor, nn
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from ._registry import METRIC_REGISTRY, register_metric

# Constants for tensor dimensions and channels
VIDEO_DIMS_4D = 4  # (N, C, H, W)
RGB_CHANNELS = 3


class CosineSimilarity(nn.Module):
    """Cosine similarity metric with accumulation support.

    Args:
        dim: Dimension along which to compute similarity (default: -1)
        eps: Small value for numerical stability (default: 1e-8)
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        dim: int = -1,
        eps: float = 1e-8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.device = device
        self.dtype = dtype or torch.float32

        # State for accumulation
        self.register_buffer("_sum", torch.tensor(0.0, dtype=self.dtype))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.long))

        # Type hints for registered buffers
        self._sum: Tensor
        self._count: Tensor

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute cosine similarity for single batch.

        Args:
            predictions: Predicted tensor
            targets: Target tensor of same shape

        Returns:
            Average cosine similarity
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        # Compute cosine similarity
        dot = (predictions * targets).sum(dim=self.dim)
        pred_norm = predictions.norm(p=2, dim=self.dim)
        target_norm = targets.norm(p=2, dim=self.dim)
        similarity = dot / (pred_norm * target_norm + self.eps)

        return similarity.mean()

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        similarity = self.forward(predictions, targets)
        batch_size = predictions.shape[0] if predictions.dim() > 1 else 1

        self._sum += similarity * batch_size
        self._count += batch_size

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        if self._count == 0:
            raise RuntimeError("No samples processed. Call update() first.")
        return self._sum / self._count

    def reset(self) -> None:
        """Reset accumulated state."""
        self._sum.zero_()
        self._count.zero_()


class MeanSquaredError(nn.Module):
    """Mean squared error metric with accumulation support.

    Args:
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype or torch.float32

        # State for accumulation
        self.register_buffer("_sum", torch.tensor(0.0, dtype=self.dtype))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.long))

        # Type hints for registered buffers
        self._sum: Tensor
        self._count: Tensor

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MSE for single batch."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        return (predictions - targets).pow(2).mean()

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        mse = (predictions - targets).pow(2).sum()
        count = predictions.numel()

        self._sum += mse
        self._count += count

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        if self._count == 0:
            raise RuntimeError("No samples processed. Call update() first.")
        return self._sum / self._count

    def reset(self) -> None:
        """Reset accumulated state."""
        self._sum.zero_()
        self._count.zero_()


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio metric.

    Wraps torchmetrics PSNR for clean interface.

    Args:
        data_range: Maximum pixel value (1.0 for normalized, 255 for uint8)
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        data_range: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.data_range = data_range
        self.device = device
        self.dtype = dtype or torch.float32

        self._metric = PeakSignalNoiseRatio(data_range=data_range)
        if device:
            self._metric = self._metric.to(device)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute PSNR for single batch."""
        result = self._metric(predictions, targets)
        # Clamp infinite values to reasonable maximum
        if torch.isinf(result):
            result = torch.tensor(100.0, dtype=result.dtype, device=result.device)
        return result

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        self._metric.update(predictions, targets)

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        result = self._metric.compute()
        # Clamp infinite values
        if torch.isinf(result):
            result = torch.tensor(100.0, dtype=result.dtype, device=result.device)
        return result

    def reset(self) -> None:
        """Reset accumulated state."""
        self._metric.reset()


class SSIM(nn.Module):
    """Structural Similarity Index metric.

    Wraps torchmetrics SSIM for clean interface.

    Args:
        window_size: Size of gaussian kernel (must be odd)
        data_range: Maximum pixel value
        sigma: Standard deviation of gaussian kernel
        k1, k2: SSIM formula constants
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        window_size: int = 11,
        data_range: float = 1.0,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(f"window_size must be odd, got {window_size}")

        self.window_size = window_size
        self.data_range = data_range
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.device = device
        self.dtype = dtype or torch.float32

        self._metric = StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=(window_size, window_size),
            sigma=(sigma, sigma),
            k1=k1,
            k2=k2,
        )
        if device:
            self._metric = self._metric.to(device)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute SSIM for single batch."""
        result = self._metric(predictions, targets)
        if isinstance(result, tuple):
            result = result[0]  # Extract tensor from tuple if needed
        return torch.clamp(result, 0, 1)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        self._metric.update(predictions, targets)

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        result = self._metric.compute()
        if isinstance(result, tuple):
            result = result[0]  # Extract tensor from tuple if needed
        return torch.clamp(result, 0, 1)

    def reset(self) -> None:
        """Reset accumulated state."""
        self._metric.reset()


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity.

    Uses pretrained networks to measure perceptual similarity.
    Lower values indicate higher similarity.

    Args:
        net_type: Network backbone ("alex", "vgg", "squeeze")
        normalize: Normalize inputs from [0,1] to [-1,1]
        reduction: Spatial reduction method ("mean" or "sum")
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        normalize: bool = True,
        reduction: Literal["sum", "mean"] = "mean",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,  # Ignore extra args for compatibility
    ):
        super().__init__()
        self.net_type = net_type
        self.normalize = normalize
        self.reduction = reduction
        self.device = device
        self.dtype = dtype or torch.float32

        # Lazy initialization for efficiency
        self._metric = None

        # State for accumulation
        self.register_buffer("_sum", torch.tensor(0.0, dtype=self.dtype))
        self.register_buffer("_count", torch.tensor(0, dtype=torch.long))

        # Type hints for registered buffers
        self._sum: Tensor
        self._count: Tensor

    def _ensure_initialized(self):
        """Initialize metric on first use."""
        if self._metric is None:
            self._metric = LearnedPerceptualImagePatchSimilarity(
                net_type=self.net_type,  # type: ignore[arg-type]
                normalize=self.normalize,
                reduction=self.reduction,  # type: ignore[arg-type]
            )
            if self.device:
                self._metric = self._metric.to(self.device)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute LPIPS for single batch."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        if predictions.shape[1] != RGB_CHANNELS:
            raise ValueError(
                f"LPIPS requires {RGB_CHANNELS}-channel RGB images, got {predictions.shape[1]} channels"
            )

        self._ensure_initialized()
        assert self._metric is not None  # For type checker
        return self._metric(predictions, targets)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        lpips = self.forward(predictions, targets)
        batch_size = predictions.shape[0]

        self._sum += lpips * batch_size
        self._count += batch_size

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        if self._count == 0:
            raise RuntimeError("No samples processed. Call update() first.")

        self._ensure_initialized()
        assert self._metric is not None  # For type checker
        return self._metric.compute()

    def reset(self) -> None:
        """Reset accumulated state."""
        if self._metric is not None:
            self._metric.reset()
        self._sum.zero_()
        self._count.zero_()


class ReconstructionMetrics(nn.Module):
    """Comprehensive metrics for reconstruction tasks.

    Combines multiple metrics:
    - Cosine similarity (always)
    - MSE (always)
    - PSNR (for images)
    - SSIM (for images)

    Args:
        include_pixel_metrics: Whether to compute PSNR/SSIM
        cosine_dim: Dimension for cosine similarity
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        include_pixel_metrics: bool = False,
        cosine_dim: int = -1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.include_pixel_metrics = include_pixel_metrics

        # Always include these metrics
        self.cosine_similarity = CosineSimilarity(
            dim=cosine_dim, device=device, dtype=dtype
        )
        self.mse = MeanSquaredError(device=device, dtype=dtype)

        # Optionally include pixel metrics
        if include_pixel_metrics:
            self.psnr = PSNR(device=device, dtype=dtype)
            self.ssim = SSIM(device=device, dtype=dtype)

    def forward(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Compute all metrics for single batch."""
        results = {
            "cosine_similarity": self.cosine_similarity(predictions, targets).item(),
            "mse": self.mse(predictions, targets).item(),
        }

        # Add pixel metrics if applicable (4D RGB tensors)
        if (
            self.include_pixel_metrics
            and predictions.dim() == VIDEO_DIMS_4D
            and predictions.shape[1] == RGB_CHANNELS
        ):
            results["psnr"] = self.psnr(predictions, targets).item()
            results["ssim"] = self.ssim(predictions, targets).item()

        return results

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update all metrics."""
        self.cosine_similarity.update(predictions, targets)
        self.mse.update(predictions, targets)

        if (
            self.include_pixel_metrics
            and predictions.dim() == VIDEO_DIMS_4D
            and predictions.shape[1] == RGB_CHANNELS
        ):
            self.psnr.update(predictions, targets)
            self.ssim.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Compute all accumulated metrics."""
        results = {
            "cosine_similarity": self.cosine_similarity.compute().item(),
            "mse": self.mse.compute().item(),
        }

        if self.include_pixel_metrics:
            # These will have been updated only if we had image data
            try:
                results["psnr"] = self.psnr.compute().item()
                results["ssim"] = self.ssim.compute().item()
            except RuntimeError:
                # No image data was processed
                pass

        return results

    def reset(self) -> None:
        """Reset all metrics."""
        self.cosine_similarity.reset()
        self.mse.reset()

        if self.include_pixel_metrics:
            self.psnr.reset()
            self.ssim.reset()


# Register metrics
register_metric("cosine_similarity", CosineSimilarity)
register_metric("mse", MeanSquaredError)
register_metric("psnr", PSNR)
register_metric("ssim", SSIM)
register_metric("lpips", LPIPS)
register_metric("reconstruction_metrics", ReconstructionMetrics)


def get_metric(name: str, **kwargs: Any) -> nn.Module:
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
