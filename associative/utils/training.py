"""Training utilities for consistent logging and monitoring."""

import math
import time

import torch
from torch.nn import functional


class MetricTracker:
    """Track and display training metrics consistently."""

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        self.epoch_start_time = None

    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()

    def update(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics.update(metrics)

    def get_epoch_time(self) -> float:
        """Get time elapsed in current epoch."""
        if self.epoch_start_time is None:
            return 0.0
        return time.time() - self.epoch_start_time

    def get_total_time(self) -> float:
        """Get total training time."""
        return time.time() - self.start_time

    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        seconds_per_minute = 60
        seconds_per_hour = 3600

        if seconds < seconds_per_minute:
            return f"{seconds:.1f}s"
        if seconds < seconds_per_hour:
            return f"{seconds / seconds_per_minute:.1f}m"
        return f"{seconds / seconds_per_hour:.1f}h"

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None = None,
        test_metrics: dict[str, float] | None = None,
    ):
        """Log epoch metrics in a consistent format."""
        epoch_time = self.get_epoch_time()
        total_time = self.get_total_time()

        # Build metrics string
        parts = [f"[{epoch:04d}/{total_epochs:04d}]"]
        parts.append(
            f"Time: {self.format_time(epoch_time)}/{self.format_time(total_time)}"
        )

        # Add train metrics
        train_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in sorted(train_metrics.items())]
        )
        parts.append(f"Train: {train_str}")

        # Add validation metrics if provided
        if val_metrics:
            val_str = " | ".join(
                [f"{k}: {v:.4f}" for k, v in sorted(val_metrics.items())]
            )
            parts.append(f"Val: {val_str}")

        # Add test metrics if provided
        if test_metrics:
            test_str = " | ".join(
                [f"{k}: {v:.4f}" for k, v in sorted(test_metrics.items())]
            )
            parts.append(f"Test: {test_str}")

        return " | ".join(parts)

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        metrics: dict[str, float],
        lr: float | None = None,
    ):
        """Log batch metrics in a consistent format."""
        # Only log every 10th batch to reduce noise
        if batch % 10 != 0 and batch != total_batches - 1:
            return None

        parts = [f"[{epoch:04d}][{batch:04d}/{total_batches:04d}]"]

        # Add metrics
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in sorted(metrics.items())])
        parts.append(metric_str)

        # Add learning rate if provided
        if lr is not None:
            parts.append(f"LR: {lr:.2e}")

        return " | ".join(parts)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_model_info(model: torch.nn.Module, name: str = "Model") -> str:
    """Format model information consistently."""
    num_params = count_parameters(model)
    return f"{name}: {num_params:,} parameters"


def calculate_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> torch.Tensor:
    """Calculate Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        max_val: Maximum possible pixel value (1.0 for normalized images)

    Returns:
        PSNR value in dB as a tensor
    """
    mse = functional.mse_loss(pred, target)
    # Handle zero MSE case
    return torch.where(
        mse > 0,
        20 * math.log10(max_val) - 10 * torch.log10(mse),
        torch.tensor(float("inf"), device=mse.device, dtype=mse.dtype),
    )


# Cache for SSIM windows to avoid recreating them
_ssim_window_cache = {}


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Calculate Structural Similarity Index (SSIM).

    Args:
        pred: Predicted images [B, C, H, W]
        target: Target images [B, C, H, W]
        window_size: Size of sliding window (must be odd)
        max_val: Maximum possible pixel value

    Returns:
        SSIM value between -1 and 1 (1 = identical) as a tensor
    """

    def create_window(
        window_size: int, channel: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create 2D Gaussian window directly on target device."""

        def gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
            coords = torch.arange(window_size, device=device, dtype=dtype)
            coords -= window_size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g /= g.sum()
            return g

        sigma = 1.5
        window_1d = gaussian_1d(window_size, sigma).unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        return window_2d.expand(channel, 1, window_size, window_size).contiguous()

    class SSIMConfig:
        """Configuration for SSIM calculation to reduce function arguments."""

        def __init__(
            self, window: torch.Tensor, window_size: int, channel: int, max_val: float
        ):
            self.window = window
            self.window_size = window_size
            self.channel = channel
            self.max_val = max_val
            self.padding = window_size // 2
            self.c1 = (0.01 * max_val) ** 2
            self.c2 = (0.03 * max_val) ** 2

    def ssim_channel(
        img1: torch.Tensor,
        img2: torch.Tensor,
        config: SSIMConfig,
    ) -> torch.Tensor:
        """Calculate SSIM for a single channel."""
        # Ensure window has the same dtype as input
        window = config.window.to(dtype=img1.dtype)
        mu1 = functional.conv2d(
            img1, window, padding=config.padding, groups=config.channel
        )
        mu2 = functional.conv2d(
            img2, window, padding=config.padding, groups=config.channel
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            functional.conv2d(
                img1 * img1,
                window,
                padding=config.padding,
                groups=config.channel,
            )
            - mu1_sq
        )
        sigma2_sq = (
            functional.conv2d(
                img2 * img2,
                window,
                padding=config.padding,
                groups=config.channel,
            )
            - mu2_sq
        )
        sigma12 = (
            functional.conv2d(
                img1 * img2,
                window,
                padding=config.padding,
                groups=config.channel,
            )
            - mu1_mu2
        )

        ssim_map = ((2 * mu1_mu2 + config.c1) * (2 * sigma12 + config.c2)) / (
            (mu1_sq + mu2_sq + config.c1) * (sigma1_sq + sigma2_sq + config.c2)
        )
        return ssim_map.mean()

    # Tensors should already be on the same device - assert instead of moving
    assert pred.device == target.device, (
        f"Tensors must be on same device, got {pred.device} and {target.device}"
    )

    channel = pred.size(1)
    cache_key = (window_size, channel, pred.device, pred.dtype)

    # Check cache for window
    if cache_key not in _ssim_window_cache:
        window = create_window(window_size, channel, pred.device, pred.dtype)
        _ssim_window_cache[cache_key] = window
    else:
        window = _ssim_window_cache[cache_key]

    # Create SSIM configuration
    config = SSIMConfig(window, window_size, channel, max_val)

    return ssim_channel(pred, target, config)


def calculate_reconstruction_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> dict[str, torch.Tensor]:
    """Calculate MSE, PSNR, and SSIM for reconstruction evaluation.

    Args:
        pred: Predicted images [B, C, H, W] (normalized)
        target: Target images [B, C, H, W] (normalized)
        mean: Normalization mean used during preprocessing
        std: Normalization std used during preprocessing

    Returns:
        Dictionary with mse, psnr, ssim values as tensors
    """

    # Unnormalize images for proper PSNR/SSIM calculation
    def unnormalize(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for i, (m, s) in enumerate(zip(mean, std, strict=False)):
            x[:, i] = x[:, i] * s + m
        return x.clamp(0, 1)

    pred_unnorm = unnormalize(pred)
    target_unnorm = unnormalize(target)

    # Calculate metrics (all return tensors now)
    mse = functional.mse_loss(pred_unnorm, target_unnorm)
    psnr = calculate_psnr(pred_unnorm, target_unnorm, max_val=1.0)
    ssim = calculate_ssim(pred_unnorm, target_unnorm, max_val=1.0)

    return {"mse": mse, "psnr": psnr, "ssim": ssim}
