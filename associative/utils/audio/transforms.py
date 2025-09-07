"""Audio transforms for MET data preprocessing.

Provides composable audio transforms following PyTorch patterns.
Includes masking, noise addition, filtering, and feature extraction.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torchaudio
from torch import Tensor

# Constants
SPECTROGRAM_DIMS = 2  # (frequency, time) for spectrograms


class AudioTransform(ABC):
    """Abstract base class for audio transforms.

    All audio transforms should inherit from this class and implement
    the __call__ method. Follows PyTorch's transform pattern.
    """

    @abstractmethod
    def __call__(self, audio: Tensor) -> Any:
        """Apply transform to audio.

        Args:
            audio: Input audio tensor

        Returns:
            Transformed audio or tuple of (transformed, auxiliary)
        """
        pass

    def __repr__(self) -> str:
        """String representation of transform."""
        return f"{self.__class__.__name__}()"


class MelSpectrogram(AudioTransform):
    """Convert waveform to mel-spectrogram representation.

    Computes mel-scaled spectrogram for audio feature extraction,
    commonly used as input to audio encoders.
    """

    def __init__(  # noqa: PLR0913
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int | None = None,
        f_min: float = 0.0,
        f_max: float | None = None,
        power: float = 2.0,
        normalized: bool = False,
    ):
        """Initialize mel-spectrogram transform.

        Args:
            sample_rate: Sample rate of audio
            n_mels: Number of mel bins
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT (defaults to n_fft)
            f_min: Minimum frequency
            f_max: Maximum frequency (defaults to sample_rate/2)
            power: Exponent for magnitude spectrogram
            normalized: Normalize mel filter bank area
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2
        self.power = power
        self.normalized = normalized

        # Validate parameters
        if self.win_length > self.n_fft:
            raise ValueError(
                f"win_length ({self.win_length}) cannot exceed n_fft ({self.n_fft})"
            )

        # Create mel transform
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            f_min=self.f_min,
            f_max=self.f_max,
            power=self.power,
            normalized=self.normalized,
        )

    def __call__(self, audio: Tensor) -> Tensor:
        """Convert audio to mel-spectrogram.

        Args:
            audio: Waveform of shape (samples,) or (channels, samples)

        Returns:
            Mel-spectrogram of shape (n_mels, time_frames) or
            (channels, n_mels, time_frames)
        """
        return self._mel_transform(audio)

    def __repr__(self) -> str:
        return (
            f"MelSpectrogram(n_mels={self.n_mels}, n_fft={self.n_fft}, "
            f"hop_length={self.hop_length})"
        )


class ApplyAudioMask(AudioTransform):
    """Apply masking to audio for self-supervised learning.

    Masks portions of audio signal for reconstruction tasks.
    Supports various masking strategies.
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio: float = 0.5,
        mask_type: Literal["random", "block", "time", "frequency"] = "block",
        mask_value: float = 0.0,
        min_masks: int = 1,
        max_masks: int = 10,
        min_mask_length: int = 100,
        max_mask_length: int = 1000,
    ):
        """Initialize audio masking transform.

        Args:
            mask_ratio: Fraction of audio to mask [0, 1]
            mask_type: Masking strategy
            mask_value: Value for masked positions
            min_masks: Minimum number of masks
            max_masks: Maximum number of masks
            min_mask_length: Minimum mask length in samples
            max_mask_length: Maximum mask length in samples

        Raises:
            ValueError: If mask_ratio not in [0, 1]
        """
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_value = mask_value
        self.min_masks = min_masks
        self.max_masks = max_masks
        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length

    def __call__(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        """Apply masking to audio.

        Args:
            audio: Input audio tensor

        Returns:
            Tuple of (masked_audio, mask) where mask is binary (1=masked)
        """
        mask = self._generate_mask(audio.shape, audio.device)
        masked_audio = audio.clone()
        masked_audio[mask == 1] = self.mask_value
        return masked_audio, mask

    def _generate_mask(
        self, shape: tuple[int, ...], device: torch.device | None = None
    ) -> Tensor:
        """Generate mask pattern based on strategy."""
        # Handle edge cases
        if self.mask_ratio == 0.0:
            return torch.zeros(shape, device=device)
        if self.mask_ratio == 1.0:
            return torch.ones(shape, device=device)

        # Dispatch to specific mask generators
        mask_generators = {
            "random": self._generate_random_mask,
            "block": self._generate_block_mask,
            "time": self._generate_time_mask,
            "frequency": self._generate_frequency_mask,
        }

        generator = mask_generators.get(self.mask_type)
        if generator:
            return generator(shape, device)

        return torch.zeros(shape, device=device)

    def _generate_random_mask(
        self, shape: tuple[int, ...], device: torch.device | None
    ) -> Tensor:
        """Generate random sample mask."""
        mask = torch.zeros(shape, device=device)
        total_samples = mask.numel()
        n_mask = int(total_samples * self.mask_ratio)
        indices = torch.randperm(total_samples, device=device)[:n_mask]
        mask.view(-1)[indices] = 1
        return mask

    def _generate_block_mask(
        self, shape: tuple[int, ...], device: torch.device | None
    ) -> Tensor:
        """Generate contiguous block mask."""
        mask = torch.zeros(shape, device=device)
        if len(shape) == 1:
            return self._generate_block_mask_1d(shape[0], device)

        # Multi-channel: apply independently to each channel
        for ch in range(shape[0]):
            mask[ch] = self._generate_block_mask_1d(shape[1], device)
        return mask

    def _generate_time_mask(
        self, shape: tuple[int, ...], device: torch.device | None
    ) -> Tensor:
        """Generate time-frame mask for spectrograms."""
        mask = torch.zeros(shape, device=device)
        if len(shape) == SPECTROGRAM_DIMS:
            n_time_frames = shape[1]
            n_mask_frames = int(n_time_frames * self.mask_ratio)
            time_indices = torch.randperm(n_time_frames, device=device)[:n_mask_frames]
            mask[:, time_indices] = 1
        return mask

    def _generate_frequency_mask(
        self, shape: tuple[int, ...], device: torch.device | None
    ) -> Tensor:
        """Generate frequency-bin mask for spectrograms."""
        mask = torch.zeros(shape, device=device)
        if len(shape) == SPECTROGRAM_DIMS:
            n_freq_bins = shape[0]
            n_mask_bins = int(n_freq_bins * self.mask_ratio)
            freq_indices = torch.randperm(n_freq_bins, device=device)[:n_mask_bins]
            mask[freq_indices, :] = 1
        return mask

    def _generate_block_mask_1d(
        self, length: int, device: torch.device | None = None
    ) -> Tensor:
        """Generate 1D block mask with non-overlapping blocks."""
        mask = torch.zeros(length, device=device)
        target_masked = int(length * self.mask_ratio)

        # Randomly choose number of masks
        n_masks = int(
            torch.randint(
                self.min_masks, self.max_masks + 1, (1,), device=device
            ).item()
        )

        # Calculate average mask length
        avg_mask_length = target_masked / n_masks

        # Keep track of placed masks to avoid overlap
        mask_segments = self._place_mask_segments(
            length, n_masks, avg_mask_length, device
        )

        # Fallback: ensure at least one segment if none were placed
        if not mask_segments and target_masked > 0:
            # Place a single segment at a random position
            mask_len = min(target_masked, length)
            start = int(
                torch.randint(
                    0, max(1, length - mask_len + 1), (1,), device=device
                ).item()
            )
            end = start + mask_len
            mask_segments = [(start, end)]

        # Apply mask segments
        for start, end in mask_segments:
            mask[start:end] = 1

        # Adjust if needed
        return self._adjust_mask_coverage(mask, mask_segments, target_masked, length)

    def _place_mask_segments(
        self,
        length: int,
        n_masks: int,
        avg_mask_length: float,
        device: torch.device | None,
    ) -> list[tuple[int, int]]:
        """Place non-overlapping mask segments."""
        mask_segments = []

        for _ in range(int(n_masks)):
            if len(mask_segments) >= n_masks:
                break

            segment = self._try_place_single_segment(
                length, n_masks, avg_mask_length, mask_segments, device
            )
            if segment:
                mask_segments.append(segment)

        return mask_segments

    def _try_place_single_segment(
        self,
        length: int,
        n_masks: int,
        avg_mask_length: float,
        existing_segments: list[tuple[int, int]],
        device: torch.device | None,
    ) -> tuple[int, int] | None:
        """Try to place a single mask segment without overlap."""
        # Calculate mask length with some variation
        min_len = max(self.min_mask_length, int(avg_mask_length * 0.5))
        max_len = min(
            self.max_mask_length, int(avg_mask_length * 1.5), length // n_masks
        )

        if min_len > max_len:
            return None

        mask_len = int(
            torch.randint(int(min_len), int(max_len) + 1, (1,), device=device).item()
        )

        # Try to find non-overlapping position
        return self._find_non_overlapping_position(
            length, mask_len, existing_segments, device
        )

    def _find_non_overlapping_position(
        self,
        length: int,
        mask_len: int,
        existing_segments: list[tuple[int, int]],
        device: torch.device | None,
        max_attempts: int = 50,
    ) -> tuple[int, int] | None:
        """Find a non-overlapping position for a mask segment."""
        for _ in range(max_attempts):
            # Random start position
            max_start = int(length - mask_len)
            if max_start <= 0:
                return None

            start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
            end = start + mask_len

            # Check for overlap
            if not self._has_overlap(start, end, existing_segments):
                return (start, end)

        return None

    def _has_overlap(
        self, start: int, end: int, segments: list[tuple[int, int]]
    ) -> bool:
        """Check if a segment overlaps with existing segments."""
        for seg_start, seg_end in segments:
            if not (end <= seg_start or start >= seg_end):
                return True
        return False

    def _adjust_mask_coverage(
        self,
        mask: Tensor,
        mask_segments: list[tuple[int, int]],
        target_masked: int,
        length: int,
    ) -> Tensor:
        """Adjust mask coverage if below target."""
        current_masked = mask.sum().item()
        if current_masked < target_masked * 0.9 and mask_segments:
            remaining = int(target_masked - current_masked)
            per_segment = remaining // len(mask_segments)

            for _i, (start, end) in enumerate(mask_segments):
                extend = min(per_segment, length - end, start)
                if extend > 0:
                    # Extend to the right if possible
                    if end + extend <= length:
                        mask[end : end + extend] = 1
                    # Or extend to the left
                    elif start - extend >= 0:
                        mask[start - extend : start] = 1

        return mask


class AddAudioNoise(AudioTransform):
    """Add noise to audio for robustness testing.

    Supports various noise types with accurate spectral characteristics.
    """

    def __init__(
        self,
        noise_type: Literal["white", "pink", "brown", "invalid"] = "white",
        snr_db: float | None = 10.0,
        min_snr: float = 0.0,
        max_snr: float = 20.0,
    ):
        """Initialize noise addition transform.

        Args:
            noise_type: Type of noise to add
            snr_db: Target SNR in dB (None for random)
            min_snr: Minimum SNR for random sampling
            max_snr: Maximum SNR for random sampling
        """
        self.noise_type = noise_type
        self.snr_db = snr_db
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, audio: Tensor) -> Tensor:
        """Add noise to audio.

        Args:
            audio: Clean audio signal

        Returns:
            Noisy audio signal
        """
        # Determine SNR
        if self.snr_db is None:
            snr_db = torch.rand(1).item() * (self.max_snr - self.min_snr) + self.min_snr
        else:
            snr_db = self.snr_db

        # Generate noise
        noise = self._generate_noise(audio.shape, self.noise_type, audio.device)

        # Calculate signal and noise power
        signal_power = (audio**2).mean()
        target_noise_power = signal_power / (10 ** (snr_db / 10))

        # Scale noise to target power
        current_noise_power = (noise**2).mean()
        if current_noise_power > 0:
            noise_scale = torch.sqrt(target_noise_power / current_noise_power)
            noise = noise * noise_scale

        return audio + noise

    def _generate_noise(
        self,
        shape: tuple[int, ...],
        noise_type: str,
        device: torch.device | None = None,
    ) -> Tensor:
        """Generate noise with correct spectral characteristics."""
        if noise_type == "white":
            return torch.randn(shape, device=device)

        if noise_type == "pink":
            return self._generate_colored_noise(shape, -1.0, device)

        if noise_type == "brown":
            return self._generate_colored_noise(shape, -2.0, device)

        # Fall back to white noise for invalid types
        return torch.randn(shape, device=device)

    def _generate_colored_noise(
        self, shape: tuple[int, ...], slope: float, device: torch.device | None = None
    ) -> Tensor:
        """Generate colored noise using FFT method for accurate spectrum."""
        if len(shape) == 1:
            return self._generate_colored_noise_1d(shape[0], slope, device)

        # Multi-channel: generate independently for each channel
        noise = torch.zeros(shape, device=device)
        for ch in range(shape[0]):
            noise[ch] = self._generate_colored_noise_1d(shape[1], slope, device)
        return noise

    def _generate_colored_noise_1d(
        self, length: int, slope: float, device: torch.device | None = None
    ) -> Tensor:
        """Generate 1D colored noise with specified spectral slope."""
        # Generate white noise in frequency domain
        # Use length // 2 + 1 for rfft output size
        freq_size = length // 2 + 1

        # Generate random phase and amplitude
        phase = torch.rand(freq_size, device=device) * 2 * torch.pi
        amplitude = torch.randn(freq_size, device=device).abs()

        # Apply frequency shaping
        freqs = torch.arange(freq_size, dtype=torch.float32, device=device)
        # Avoid division by zero at DC
        freqs[0] = 1.0

        # Apply 1/f^alpha shaping (slope is negative, so -slope/2 is positive)
        freq_filter = 1.0 / (freqs ** (-slope / 2.0))
        freq_filter[0] = freq_filter[1]  # Set DC to same as first AC component

        # Create complex spectrum with proper shaping
        spectrum_real = amplitude * freq_filter * torch.cos(phase)
        spectrum_imag = amplitude * freq_filter * torch.sin(phase)

        # DC and Nyquist must be real for real output
        spectrum_real[0] = amplitude[0] * freq_filter[0]  # DC is real
        spectrum_imag[0] = 0.0
        if length % 2 == 0:
            spectrum_real[-1] = amplitude[-1] * freq_filter[-1]  # Nyquist is real
            spectrum_imag[-1] = 0.0

        # Combine into complex spectrum
        spectrum = torch.complex(spectrum_real, spectrum_imag)

        # Convert to time domain
        noise = torch.fft.irfft(spectrum, n=length)

        # Normalize to unit variance
        return noise / noise.std()


class BandpassFilter(AudioTransform):
    """Apply bandpass filtering to audio.

    Uses butterworth-style IIR filtering for accurate frequency response.
    """

    def __init__(
        self,
        low_freq: float = 300.0,
        high_freq: float = 3400.0,
        sample_rate: int = 16000,
        order: int = 5,
    ):
        """Initialize bandpass filter.

        Args:
            low_freq: Lower cutoff frequency (Hz)
            high_freq: Upper cutoff frequency (Hz)
            sample_rate: Sample rate of audio
            order: Filter order (higher = sharper cutoff)
        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sample_rate = sample_rate
        self.order = order

        # Validate frequencies
        nyquist = sample_rate / 2
        if not 0 < low_freq < high_freq < nyquist:
            raise ValueError(
                f"Invalid frequencies: need 0 < {low_freq} < {high_freq} < {nyquist}"
            )

    def __call__(self, audio: Tensor) -> Tensor:
        """Apply bandpass filter to audio.

        Args:
            audio: Input audio signal

        Returns:
            Filtered audio signal
        """
        # Use frequency domain filtering for better control
        # This provides more accurate passband preservation

        # Store original device
        orig_device = audio.device
        orig_shape = audio.shape

        # Handle different input shapes
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension

        # Process each channel
        filtered_channels = []
        for ch_idx in range(audio.shape[0]):
            ch_audio = audio[ch_idx]

            # Compute FFT
            spectrum = torch.fft.rfft(ch_audio)
            freqs = torch.fft.rfftfreq(ch_audio.shape[0], 1 / self.sample_rate)

            # Create butterworth-like frequency response
            # Using cascaded first-order sections for stability
            h_response = torch.ones_like(freqs, dtype=torch.complex64)

            # Highpass section (remove below low_freq)
            for _ in range(self.order):
                h_response *= (1j * freqs / self.low_freq) / (
                    1 + 1j * freqs / self.low_freq
                )

            # Lowpass section (remove above high_freq)
            for _ in range(self.order):
                h_response *= 1 / (1 + 1j * freqs / self.high_freq)

            # Apply filter
            filtered_spectrum = spectrum * h_response.to(spectrum.device)

            # Convert back to time domain
            filtered = torch.fft.irfft(filtered_spectrum, n=ch_audio.shape[0])
            filtered_channels.append(filtered)

        # Stack channels
        filtered = torch.stack(filtered_channels, dim=0)

        # Restore original shape
        if len(orig_shape) == 1:
            filtered = filtered.squeeze(0)

        return filtered.to(orig_device)

    def __repr__(self) -> str:
        return (
            f"BandpassFilter(low={self.low_freq}Hz, high={self.high_freq}Hz, "
            f"sr={self.sample_rate})"
        )
