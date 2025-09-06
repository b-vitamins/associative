"""Audio reconstruction metrics.

Provides metrics for evaluating audio reconstruction quality,
with a clean API following PyTorch conventions.
"""

import logging
from typing import Any

import torch
from pesq import pesq as compute_pesq
from pystoi import stoi as compute_stoi
from torch import Tensor, nn
from torchmetrics.audio import SignalDistortionRatio

# Setup logger
logger = logging.getLogger(__name__)

# Constants
SIXTEEN_KHZ = 16_000
MULTICHANNEL_DIMS = 3  # (batch, channels, samples)
TEN_KHZ = 10_000
EIGHT_KHZ = 8_000


class PESQ(nn.Module):
    """Perceptual Evaluation of Speech Quality.

    Implements ITU-T P.862 PESQ for speech quality assessment.
    Scores range from -0.5 to 4.5 (theoretical), though implementations
    may return slightly higher values for identical signals.

    Args:
        mode: "wb" (wideband) or "nb" (narrowband)
        sample_rate: Must be 16000 for wb or 8000 for nb
        device: Device for computation
        dtype: Data type for computation
    """

    _sum: Tensor
    count: Tensor

    def __init__(
        self,
        mode: str = "wb",
        sample_rate: int = 16000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if mode == "wb" and sample_rate != SIXTEEN_KHZ:
            raise ValueError(
                f"Wideband PESQ requires 16kHz sample rate, got {sample_rate}"
            )
        if mode == "nb" and sample_rate != EIGHT_KHZ:
            raise ValueError(
                f"Narrowband PESQ requires 8kHz sample rate, got {sample_rate}"
            )

        self.mode = mode
        self.sample_rate = sample_rate
        self.device = device
        self.dtype = dtype or torch.float32

        # State for accumulation
        self.register_buffer("_sum", torch.tensor(0.0, dtype=self.dtype))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute PESQ for single batch.

        Args:
            predictions: Predicted audio
            targets: Target audio of same shape

        Returns:
            Average PESQ score
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        # Handle multi-channel by averaging
        if predictions.dim() == MULTICHANNEL_DIMS:
            predictions = predictions.mean(dim=1)
            targets = targets.mean(dim=1)
        elif predictions.dim() == 1:  # Single audio
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        batch_size = predictions.shape[0]
        total_score = 0.0
        valid_count = 0

        # Move to CPU for PESQ computation
        pred_cpu = predictions.detach().cpu().numpy()
        target_cpu = targets.detach().cpu().numpy()

        # Compute PESQ for each sample
        for i in range(batch_size):
            try:
                score = compute_pesq(
                    self.sample_rate,
                    target_cpu[i],  # reference (clean)
                    pred_cpu[i],  # degraded (predicted)
                    self.mode,
                )
                total_score += score
                valid_count += 1
            except Exception as e:
                logger.debug(f"PESQ computation failed for sample {i}: {e}")
                continue

        if valid_count == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)

        return torch.tensor(
            total_score / valid_count, dtype=self.dtype, device=self.device
        )

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        score = self.forward(predictions, targets)
        if score != 0:  # Only update if we got valid scores
            batch_size = predictions.shape[0] if predictions.dim() > 1 else 1
            self._sum += score * batch_size
            self.count += batch_size

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        if self.count == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        return self._sum / self.count.to(dtype=self.dtype)

    def reset(self) -> None:
        """Reset accumulated state."""
        self._sum.zero_()
        self.count.zero_()


class STOI(nn.Module):
    """Short-Time Objective Intelligibility.

    Measures speech intelligibility on a scale from 0 to 1,
    though extended STOI can return slightly negative values.

    Args:
        extended: Use extended STOI for better accuracy
        sample_rate: Sample rate (10kHz minimum recommended)
        device: Device for computation
        dtype: Data type for computation
    """

    _sum: Tensor
    count: Tensor

    def __init__(
        self,
        extended: bool = True,
        sample_rate: int = 16000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.extended = extended
        self.sample_rate = sample_rate
        self.device = device
        self.dtype = dtype or torch.float32

        if sample_rate < TEN_KHZ:
            logger.warning(
                f"STOI performs best with sample_rate >= 10kHz, got {sample_rate}Hz"
            )

        # State for accumulation
        self.register_buffer("_sum", torch.tensor(0.0, dtype=self.dtype))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute STOI for single batch.

        Args:
            predictions: Predicted audio
            targets: Target audio of same shape

        Returns:
            Average STOI score
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        # Handle multi-channel by averaging
        if predictions.dim() == MULTICHANNEL_DIMS:
            predictions = predictions.mean(dim=1)
            targets = targets.mean(dim=1)
        elif predictions.dim() == 1:  # Single audio
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        batch_size = predictions.shape[0]
        total_score = 0.0
        valid_count = 0

        # Move to CPU for STOI computation
        pred_cpu = predictions.detach().cpu().numpy()
        target_cpu = targets.detach().cpu().numpy()

        # Compute STOI for each sample
        for i in range(batch_size):
            try:
                score = compute_stoi(
                    target_cpu[i],  # clean (reference)
                    pred_cpu[i],  # denoised (predicted)
                    self.sample_rate,
                    extended=self.extended,
                )
                total_score += score
                valid_count += 1
            except Exception as e:
                logger.debug(f"STOI computation failed for sample {i}: {e}")
                continue

        if valid_count == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)

        return torch.tensor(
            total_score / valid_count, dtype=self.dtype, device=self.device
        )

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        score = self.forward(predictions, targets)
        if score != 0:  # Only update if we got valid scores
            batch_size = predictions.shape[0] if predictions.dim() > 1 else 1
            self._sum += score * batch_size
            self.count += batch_size

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        if self.count == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        return self._sum / self.count.to(dtype=self.dtype)

    def reset(self) -> None:
        """Reset accumulated state."""
        self._sum.zero_()
        self.count.zero_()


class SDR(nn.Module):
    """Signal-to-Distortion Ratio.

    Measures ratio of signal power to distortion power in dB.
    Higher values indicate better reconstruction quality.

    Args:
        use_cg_iter: Number of conjugate gradient iterations
        filter_length: Length of distortion filter
        zero_mean: Whether to zero-mean signals
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        use_cg_iter: int | None = None,
        filter_length: int = 512,
        zero_mean: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.use_cg_iter = use_cg_iter
        self.filter_length = filter_length
        self.zero_mean = zero_mean
        self.device = device
        self.dtype = dtype or torch.float32

        # Use torchmetrics implementation
        self._metric = SignalDistortionRatio(
            use_cg_iter=use_cg_iter,
            filter_length=filter_length,
            zero_mean=zero_mean,
        )
        if device:
            self._metric = self._metric.to(device)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute SDR for single batch."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        # Ensure proper shape
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        return self._metric(predictions, targets)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update running statistics."""
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

        # Ensure proper shape
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(0)
            targets = targets.unsqueeze(0)

        self._metric.update(predictions, targets)

    def compute(self) -> Tensor:
        """Compute accumulated average."""
        return self._metric.compute()

    def reset(self) -> None:
        """Reset accumulated state."""
        self._metric.reset()


class AudioReconstructionMetrics(nn.Module):
    """Comprehensive audio reconstruction metrics.

    Combines PESQ, STOI, and SDR for complete audio quality assessment.

    Args:
        include_pesq: Whether to compute PESQ
        include_stoi: Whether to compute STOI
        pesq_mode: "wb" or "nb" for PESQ
        sample_rate: Audio sample rate
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(  # noqa: PLR0913
        self,
        include_pesq: bool = True,
        include_stoi: bool = True,
        pesq_mode: str = "wb",
        sample_rate: int = 16000,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device
        self.dtype = dtype or torch.float32

        # SDR is always included
        self.sdr = SDR(device=device, dtype=dtype)

        # Optional PESQ
        if include_pesq:
            # Check compatibility
            if pesq_mode == "wb" and sample_rate != SIXTEEN_KHZ:
                logger.warning(
                    f"PESQ wideband requires 16kHz, got {sample_rate}Hz. Disabling PESQ."
                )
                self.pesq = None
            elif pesq_mode == "nb" and sample_rate != EIGHT_KHZ:
                logger.warning(
                    f"PESQ narrowband requires 8kHz, got {sample_rate}Hz. Disabling PESQ."
                )
                self.pesq = None
            else:
                self.pesq = PESQ(
                    mode=pesq_mode,
                    sample_rate=sample_rate,
                    device=device,
                    dtype=dtype,
                )
        else:
            self.pesq = None

        # Optional STOI
        if include_stoi:
            self.stoi = STOI(
                sample_rate=sample_rate,
                device=device,
                dtype=dtype,
            )
        else:
            self.stoi = None

    def forward(self, predictions: Tensor, targets: Tensor) -> dict[str, float]:
        """Compute all metrics for single batch."""
        results = {"sdr": self.sdr(predictions, targets).item()}

        if self.pesq is not None:
            results["pesq"] = self.pesq(predictions, targets).item()

        if self.stoi is not None:
            results["stoi"] = self.stoi(predictions, targets).item()

        return results

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update all metrics."""
        self.sdr.update(predictions, targets)

        if self.pesq is not None:
            self.pesq.update(predictions, targets)

        if self.stoi is not None:
            self.stoi.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        """Compute all accumulated metrics."""
        results = {"sdr": self.sdr.compute().item()}

        if self.pesq is not None:
            results["pesq"] = self.pesq.compute().item()

        if self.stoi is not None:
            results["stoi"] = self.stoi.compute().item()

        return results

    def reset(self) -> None:
        """Reset all metrics."""
        self.sdr.reset()

        if self.pesq is not None:
            self.pesq.reset()

        if self.stoi is not None:
            self.stoi.reset()


# Registry for dynamic metric creation
_AUDIO_METRIC_REGISTRY = {
    "pesq": PESQ,
    "stoi": STOI,
    "sdr": SDR,
    "audio_reconstruction": AudioReconstructionMetrics,
}


def register_audio_metric(name: str, metric_class: type[nn.Module]) -> None:
    """Register an audio metric.

    Args:
        name: Metric name
        metric_class: Metric class
    """
    _AUDIO_METRIC_REGISTRY[name] = metric_class


def get_audio_metric(name: str, **kwargs: Any) -> nn.Module:
    """Get audio metric by name.

    Args:
        name: Metric name
        **kwargs: Arguments for metric constructor

    Returns:
        Initialized metric

    Raises:
        KeyError: If metric not found
    """
    if name not in _AUDIO_METRIC_REGISTRY:
        available = list(_AUDIO_METRIC_REGISTRY.keys())
        raise KeyError(f"Metric '{name}' not found. Available: {available}")

    metric_class = _AUDIO_METRIC_REGISTRY[name]
    return metric_class(**kwargs)
