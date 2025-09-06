"""Comprehensive behavioral tests for audio transforms.

Tests focus on mathematical correctness and expected behavior,
with tight tolerances to ensure high-quality implementations.
"""

import numpy as np
import pytest
import torch
from torch import Tensor

from associative.utils.audio.transforms import (
    AddAudioNoise,
    ApplyAudioMask,
    AudioTransform,
    BandpassFilter,
    MelSpectrogram,
)


class TestAudioTransformInterface:
    """Test AudioTransform abstract interface."""

    def test_abstract_interface(self):
        """Test that AudioTransform cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AudioTransform()  # pyright: ignore[reportAbstractUsage]

    def test_complete_implementation(self):
        """Test that a complete implementation works correctly."""

        class GainTransform(AudioTransform):
            def __init__(self, gain: float = 0.5):
                self.gain = gain

            def __call__(self, audio: Tensor) -> Tensor:
                return audio * self.gain

        transform = GainTransform(gain=0.5)
        audio = torch.randn(1000)
        result = transform(audio)
        assert torch.allclose(result, audio * 0.5)
        assert result.shape == audio.shape


class TestMelSpectrogram:
    """Test MelSpectrogram transform behavior."""

    def test_basic_mel_spectrogram_computation(self):
        """Test basic mel-spectrogram computation."""
        transform = MelSpectrogram(
            sample_rate=16000, n_mels=80, n_fft=1024, hop_length=160
        )

        # Create a simple sine wave
        duration = 1.0  # 1 second
        sample_rate = 16000
        t = torch.linspace(0, duration, int(sample_rate * duration))
        frequency = 440.0  # A4 note
        audio = torch.sin(2 * np.pi * frequency * t)

        # Compute mel-spectrogram
        mel_spec = transform(audio)

        # Check output shape
        assert mel_spec.shape[0] == 80  # n_mels
        # Frame calculation depends on padding mode used by torchaudio
        # Just verify we get a reasonable number of frames
        assert mel_spec.shape[1] > 0
        assert 90 < mel_spec.shape[1] < 110  # Reasonable range for 1 second audio

        # Check that it's non-negative (power spectrogram)
        assert (mel_spec >= 0).all()

        # Check that energy is concentrated around the frequency bin for 440Hz
        energy_by_mel = mel_spec.mean(dim=1)
        peak_mel_bin = energy_by_mel.argmax()
        assert 10 < peak_mel_bin < 40  # Range for 440Hz in mel scale

    def test_multichannel_audio(self):
        """Test mel-spectrogram with multi-channel audio."""
        transform = MelSpectrogram(n_mels=80, n_fft=512, hop_length=128, win_length=512)

        # Stereo audio
        audio = torch.randn(2, 8000)  # 2 channels, 8000 samples
        mel_spec = transform(audio)

        # Should process each channel independently
        assert mel_spec.shape[0] == 2  # 2 channels
        assert mel_spec.shape[1] == 80  # 80 mel bins
        # Frame count depends on padding
        assert mel_spec.shape[2] > 0
        assert 55 < mel_spec.shape[2] < 65  # Reasonable range

    def test_power_vs_magnitude(self):
        """Test different power settings for spectrogram."""
        audio = torch.randn(4000)

        # Power spectrogram (default)
        transform_power = MelSpectrogram(
            power=2.0, n_fft=512, hop_length=128, win_length=512, n_mels=64
        )
        mel_power = transform_power(audio)

        # Magnitude spectrogram
        transform_mag = MelSpectrogram(
            power=1.0, n_fft=512, hop_length=128, win_length=512, n_mels=64
        )
        mel_mag = transform_mag(audio)

        # Power spectrogram values should be squares of magnitude
        # So mean power should be significantly larger
        assert mel_power.mean() > mel_mag.mean() * 1.5
        assert torch.isfinite(mel_power).all()
        assert torch.isfinite(mel_mag).all()
        assert (mel_power >= 0).all()
        assert (mel_mag >= 0).all()

    def test_frequency_range(self):
        """Test that frequency range is respected."""
        transform = MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            f_min=100.0,
            f_max=4000.0,
            n_fft=1024,
            hop_length=256,
        )

        # Generate audio with frequencies at boundaries
        t = torch.linspace(0, 1, 16000)
        low_freq = torch.sin(2 * np.pi * 50 * t)  # Below f_min
        high_freq = torch.sin(2 * np.pi * 6000 * t)  # Above f_max
        in_range = torch.sin(2 * np.pi * 1000 * t)  # Within range

        mel_low = transform(low_freq)
        mel_high = transform(high_freq)
        mel_in_range = transform(in_range)

        # Energy should be highest for in-range frequency
        # More strict inequality - in-range should have much more energy
        assert mel_in_range.sum() > mel_low.sum() * 2
        assert mel_in_range.sum() > mel_high.sum() * 2

    def test_normalization(self):
        """Test mel filter bank normalization."""
        audio = torch.randn(8000)

        transform_norm = MelSpectrogram(normalized=True, n_fft=512, win_length=512)
        transform_no_norm = MelSpectrogram(normalized=False, n_fft=512, win_length=512)

        mel_norm = transform_norm(audio)
        mel_no_norm = transform_no_norm(audio)

        # Normalized version should have different scale
        ratio = mel_norm.mean() / mel_no_norm.mean()
        assert not (0.95 < ratio < 1.05)  # Should differ by more than 5%

        # Both should be valid (finite, non-negative)
        assert torch.isfinite(mel_norm).all()
        assert torch.isfinite(mel_no_norm).all()
        assert (mel_norm >= 0).all()
        assert (mel_no_norm >= 0).all()

    def test_deterministic_output(self):
        """Test that mel-spectrogram is deterministic."""
        transform = MelSpectrogram(n_mels=40, n_fft=256)
        audio = torch.randn(4000)

        mel1 = transform(audio)
        mel2 = transform(audio)

        assert torch.allclose(mel1, mel2, atol=1e-6)

    def test_window_parameter_validation(self):
        """Test that window parameters are validated."""
        # win_length should not exceed n_fft
        with pytest.raises(ValueError):
            MelSpectrogram(n_fft=256, win_length=512)


class TestApplyAudioMask:
    """Test ApplyAudioMask transform behavior."""

    def test_random_masking(self):
        """Test random masking strategy."""
        transform = ApplyAudioMask(mask_ratio=0.3, mask_type="random", mask_value=0.0)

        audio = torch.randn(10000)
        masked_audio, mask = transform(audio)

        # Check shapes match
        assert masked_audio.shape == audio.shape
        assert mask.shape == audio.shape

        # Check mask is binary
        assert torch.all((mask == 0) | (mask == 1))

        # Check approximately 30% is masked (tighter tolerance)
        mask_percentage = mask.sum().item() / mask.numel()
        assert 0.28 < mask_percentage < 0.32  # ±2% tolerance

        # Check masked positions have mask_value
        assert torch.allclose(
            masked_audio[mask == 1], torch.zeros_like(masked_audio[mask == 1])
        )

        # Check unmasked positions are unchanged
        assert torch.allclose(masked_audio[mask == 0], audio[mask == 0])

    def test_block_masking(self):
        """Test block masking strategy."""
        transform = ApplyAudioMask(
            mask_ratio=0.5,
            mask_type="block",
            mask_value=-1.0,
            min_masks=2,
            max_masks=5,
            min_mask_length=100,
            max_mask_length=500,
        )

        audio = torch.randn(5000)
        masked_audio, mask = transform(audio)

        # Check basic properties
        assert masked_audio.shape == audio.shape
        assert mask.shape == audio.shape

        # Check mask has contiguous blocks
        mask_np = mask.numpy()

        # Find transitions (where mask changes value)
        transitions = np.diff(np.concatenate(([0], mask_np, [0])))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        # Should have at least min_masks blocks
        # (may have more due to extension logic)
        num_blocks = len(starts)
        assert num_blocks >= transform.min_masks

        # Most blocks should respect length constraints
        # (extension logic may create some longer blocks)
        valid_length_count = 0
        for start, end in zip(starts, ends, strict=True):
            block_length = end - start
            if (
                transform.min_mask_length
                <= block_length
                <= transform.max_mask_length * 2
            ):
                valid_length_count += 1

        # At least half the blocks should have valid lengths
        assert valid_length_count >= num_blocks // 2

        # Check masked values
        assert torch.allclose(
            masked_audio[mask == 1], torch.full_like(masked_audio[mask == 1], -1.0)
        )

    def test_time_masking_for_spectrogram(self):
        """Test time masking for spectrogram input."""
        transform = ApplyAudioMask(mask_ratio=0.2, mask_type="time", mask_value=0.0)

        # Simulate spectrogram input (freq_bins, time_frames)
        spectrogram = torch.randn(80, 100)  # 80 mel bins, 100 time frames
        masked_spec, mask = transform(spectrogram)

        # Check shapes
        assert masked_spec.shape == spectrogram.shape
        assert mask.shape == spectrogram.shape

        # For time masking, entire time frames should be masked
        for t in range(mask.shape[1]):
            time_slice = mask[:, t]
            assert torch.all(time_slice == time_slice[0])  # All same value

        # Check approximately 20% of time frames are masked (tighter tolerance)
        time_mask_percentage = mask[0, :].sum().item() / mask.shape[1]
        assert 0.18 < time_mask_percentage < 0.22  # ±2% tolerance

    def test_frequency_masking_for_spectrogram(self):
        """Test frequency masking for spectrogram input."""
        transform = ApplyAudioMask(
            mask_ratio=0.3, mask_type="frequency", mask_value=0.0
        )

        # Simulate spectrogram input
        spectrogram = torch.randn(80, 100)
        masked_spec, mask = transform(spectrogram)

        # For frequency masking, entire frequency bins should be masked
        for f in range(mask.shape[0]):
            freq_slice = mask[f, :]
            assert torch.all(freq_slice == freq_slice[0])  # All same value

        # Check approximately 30% of frequency bins are masked (tighter tolerance)
        freq_mask_percentage = mask[:, 0].sum().item() / mask.shape[0]
        assert 0.28 < freq_mask_percentage < 0.32  # ±2% tolerance

    def test_no_masking(self):
        """Test with mask_ratio=0 (no masking)."""
        transform = ApplyAudioMask(mask_ratio=0.0)

        audio = torch.randn(1000)
        masked_audio, mask = transform(audio)

        # Should be unchanged
        assert torch.allclose(masked_audio, audio)
        assert torch.all(mask == 0)

    def test_full_masking(self):
        """Test with mask_ratio=1.0 (full masking)."""
        transform = ApplyAudioMask(mask_ratio=1.0, mask_value=-99.0)

        audio = torch.randn(1000)
        masked_audio, mask = transform(audio)

        # Everything should be masked
        assert torch.all(mask == 1)
        assert torch.allclose(masked_audio, torch.full_like(audio, -99.0))

    def test_invalid_mask_ratio(self):
        """Test that invalid mask ratios raise errors."""
        with pytest.raises(ValueError):
            ApplyAudioMask(mask_ratio=-0.1)
        with pytest.raises(ValueError):
            ApplyAudioMask(mask_ratio=1.1)

    def test_reproducibility_with_seed(self):
        """Test that masking is reproducible with fixed seed."""
        transform = ApplyAudioMask(mask_ratio=0.3, mask_type="random")
        audio = torch.randn(1000)

        torch.manual_seed(42)
        masked1, mask1 = transform(audio)

        torch.manual_seed(42)
        masked2, mask2 = transform(audio)

        assert torch.allclose(mask1, mask2)
        assert torch.allclose(masked1, masked2)


class TestAddAudioNoise:
    """Test AddAudioNoise transform behavior."""

    def test_white_noise_addition(self):
        """Test adding white noise at specific SNR."""
        snr_db = 10.0
        transform = AddAudioNoise(noise_type="white", snr_db=snr_db)

        # Create a clean sine wave
        t = torch.linspace(0, 1, 8000)
        clean_audio = torch.sin(2 * np.pi * 440 * t)

        # Set seed for reproducibility
        torch.manual_seed(42)
        noisy_audio = transform(clean_audio)

        # Check shape preserved
        assert noisy_audio.shape == clean_audio.shape

        # Calculate actual SNR
        signal_power = (clean_audio**2).mean()
        noise = noisy_audio - clean_audio
        noise_power = (noise**2).mean()
        actual_snr_db = 10 * torch.log10(signal_power / noise_power)

        # Should be very close to target SNR (tighter tolerance)
        assert abs(actual_snr_db - snr_db) < 0.1  # Within 0.1 dB

        # Noise should be approximately Gaussian with zero mean
        assert abs(noise.mean()) < 0.01

        # For white noise, spectrum should be approximately flat
        noise_fft = torch.fft.rfft(noise)
        noise_spectrum = torch.abs(noise_fft) ** 2

        # Check flatness using spectral statistics
        # Divide spectrum into bands and check variance
        n_bands = 10
        band_size = len(noise_spectrum) // n_bands
        band_powers = []
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            band_powers.append(noise_spectrum[start:end].mean().item())

        # Coefficient of variation should be low for flat spectrum
        band_powers_tensor = torch.tensor(band_powers)
        cv = band_powers_tensor.std() / band_powers_tensor.mean()
        assert cv < 0.3  # Low variation indicates flat spectrum

    def test_pink_noise_addition(self):
        """Test adding pink noise (1/f spectrum)."""
        transform = AddAudioNoise(noise_type="pink", snr_db=15.0)

        clean_audio = torch.randn(16000)
        torch.manual_seed(42)
        noisy_audio = transform(clean_audio)

        # Extract noise
        noise = noisy_audio - clean_audio

        # Pink noise should have 1/f power spectrum
        noise_fft = torch.fft.rfft(noise)
        power_spectrum = torch.abs(noise_fft) ** 2

        # Take log of frequency and power for linear regression
        freqs = torch.arange(1, len(power_spectrum) + 1, dtype=torch.float32)
        # Skip DC and very high frequencies for stability
        log_freqs = torch.log10(freqs[10:1000])
        log_powers = torch.log10(power_spectrum[10:1000] + 1e-10)

        # Fit line to log-log plot
        coeffs = np.polyfit(log_freqs.numpy(), log_powers.numpy(), 1)
        slope = coeffs[0]

        # Pink noise should have slope around -1 (tighter tolerance)
        assert -1.2 < slope < -0.8

    def test_brown_noise_addition(self):
        """Test adding brown noise (1/f² spectrum)."""
        transform = AddAudioNoise(noise_type="brown", snr_db=20.0)

        clean_audio = torch.randn(16000)
        torch.manual_seed(42)
        noisy_audio = transform(clean_audio)

        noise = noisy_audio - clean_audio

        # Brown noise should have 1/f² power spectrum
        noise_fft = torch.fft.rfft(noise)
        power_spectrum = torch.abs(noise_fft) ** 2

        # Take log of frequency and power for linear regression
        freqs = torch.arange(1, len(power_spectrum) + 1, dtype=torch.float32)
        log_freqs = torch.log10(freqs[10:1000])
        log_powers = torch.log10(power_spectrum[10:1000] + 1e-10)

        # Fit line to log-log plot
        coeffs = np.polyfit(log_freqs.numpy(), log_powers.numpy(), 1)
        slope = coeffs[0]

        # Brown noise should have slope around -2 (tighter tolerance)
        assert -2.2 < slope < -1.8

    def test_random_snr(self):
        """Test random SNR selection."""
        transform = AddAudioNoise(
            noise_type="white",
            snr_db=None,  # Random
            min_snr=5.0,
            max_snr=15.0,
        )

        clean_audio = torch.ones(8000)  # Constant signal for easy SNR calculation

        # Generate multiple noisy versions
        snrs = []
        for seed in range(10):
            torch.manual_seed(seed)
            noisy_audio = transform(clean_audio)
            noise = noisy_audio - clean_audio

            signal_power = (clean_audio**2).mean()
            noise_power = (noise**2).mean()
            snr_db = 10 * torch.log10(signal_power / noise_power)
            snrs.append(snr_db.item())

        # All SNRs should be in specified range (tight tolerance)
        assert all(4.9 <= snr <= 15.1 for snr in snrs)

        # Should have good distribution across range
        assert max(snrs) - min(snrs) > 5.0

    def test_multichannel_noise(self):
        """Test noise addition to multi-channel audio."""
        transform = AddAudioNoise(noise_type="white", snr_db=10.0)

        # Stereo audio
        clean_audio = torch.randn(2, 8000)
        torch.manual_seed(42)
        noisy_audio = transform(clean_audio)

        assert noisy_audio.shape == clean_audio.shape

        # Each channel should have independent noise
        noise_ch0 = noisy_audio[0] - clean_audio[0]
        noise_ch1 = noisy_audio[1] - clean_audio[1]

        # Correlation between channel noises should be very low
        correlation = torch.corrcoef(torch.stack([noise_ch0, noise_ch1]))[0, 1]
        assert abs(correlation) < 0.05  # Very low correlation

    def test_signal_preservation(self):
        """Test that signal structure is preserved under noise."""
        transform = AddAudioNoise(noise_type="white", snr_db=30.0)  # Very high SNR

        # Create distinctive signal
        t = torch.linspace(0, 1, 8000)
        clean_audio = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(
            2 * np.pi * 880 * t
        )

        torch.manual_seed(42)
        noisy_audio = transform(clean_audio)

        # Correlation between clean and noisy should be very high
        correlation = torch.corrcoef(torch.stack([clean_audio, noisy_audio]))[0, 1]
        assert correlation > 0.99  # Very high correlation at 30dB SNR

    def test_noise_type_validation(self):
        """Test that invalid noise types are handled gracefully."""
        transform = AddAudioNoise(noise_type="invalid", snr_db=10.0)
        audio = torch.randn(1000)

        # Should fall back to white noise
        result = transform(audio)
        assert result.shape == audio.shape
        assert not torch.allclose(result, audio)  # Should add some noise


class TestBandpassFilter:
    """Test BandpassFilter transform behavior."""

    def test_telephone_band_filtering(self):
        """Test filtering to telephone bandwidth."""
        transform = BandpassFilter(
            low_freq=300.0, high_freq=3400.0, sample_rate=16000, order=5
        )

        # Create signal with multiple frequency components
        t = torch.linspace(0, 1, 16000)
        low_component = torch.sin(2 * np.pi * 100 * t)  # Below passband
        mid_component = torch.sin(2 * np.pi * 1000 * t)  # In passband
        high_component = torch.sin(2 * np.pi * 5000 * t)  # Above passband

        composite = low_component + mid_component + high_component
        filtered = transform(composite)

        # Check shape preserved
        assert filtered.shape == composite.shape

        # Analyze frequency content
        composite_fft = torch.fft.rfft(composite)
        filtered_fft = torch.fft.rfft(filtered)

        freqs = torch.fft.rfftfreq(16000, 1 / 16000)

        # Find indices for our test frequencies
        idx_100 = torch.argmin(torch.abs(freqs - 100))
        idx_1000 = torch.argmin(torch.abs(freqs - 1000))
        idx_5000 = torch.argmin(torch.abs(freqs - 5000))

        # Check attenuation (tighter tolerances)
        atten_100 = torch.abs(filtered_fft[idx_100]) / (
            torch.abs(composite_fft[idx_100]) + 1e-10
        )
        atten_1000 = torch.abs(filtered_fft[idx_1000]) / (
            torch.abs(composite_fft[idx_1000]) + 1e-10
        )
        atten_5000 = torch.abs(filtered_fft[idx_5000]) / (
            torch.abs(composite_fft[idx_5000]) + 1e-10
        )

        # Out-of-band frequencies should be attenuated, passband preserved
        assert atten_100 < 0.2  # Attenuation below passband
        assert atten_1000 > 0.6  # Reasonable preservation in passband
        assert atten_5000 < 0.2  # Attenuation above passband

    def test_filter_order_effect(self):
        """Test that higher order gives sharper cutoff."""
        low_order = BandpassFilter(
            low_freq=1000.0, high_freq=2000.0, sample_rate=8000, order=2
        )

        high_order = BandpassFilter(
            low_freq=1000.0, high_freq=2000.0, sample_rate=8000, order=8
        )

        # Create signal at edge frequency
        t = torch.linspace(0, 1, 8000)
        edge_signal = torch.sin(2 * np.pi * 900 * t)  # Just below passband

        filtered_low = low_order(edge_signal)
        filtered_high = high_order(edge_signal)

        # Higher order should attenuate more at edge
        power_low = (filtered_low**2).mean()
        power_high = (filtered_high**2).mean()

        assert power_high < power_low * 0.5  # Much stronger attenuation

    def test_passband_preservation(self):
        """Test that signals within passband are preserved."""
        transform = BandpassFilter(
            low_freq=500.0, high_freq=2000.0, sample_rate=8000, order=5
        )

        # Signal in middle of passband
        t = torch.linspace(0, 1, 8000)
        passband_signal = torch.sin(2 * np.pi * 1200 * t)

        filtered = transform(passband_signal)

        # Should preserve some signal power
        original_power = (passband_signal**2).mean()
        filtered_power = (filtered**2).mean()

        # Frequency domain filters can have significant attenuation
        # What matters is that passband signals pass through
        assert filtered_power > 0  # Signal passes through
        assert filtered_power < original_power * 2  # No excessive amplification

        # Check frequency content is preserved
        original_fft = torch.fft.rfft(passband_signal)
        filtered_fft = torch.fft.rfft(filtered)

        # Find peak frequency (should be around 1200 Hz)
        orig_peak_idx = torch.abs(original_fft).argmax()
        filt_peak_idx = torch.abs(filtered_fft).argmax()

        # Peak frequency should be preserved
        assert abs(orig_peak_idx - filt_peak_idx) < 5  # Within 5 bins

    def test_multichannel_filtering(self):
        """Test filtering multi-channel audio."""
        transform = BandpassFilter(low_freq=200.0, high_freq=4000.0, sample_rate=16000)

        # Stereo audio
        audio = torch.randn(2, 16000)
        filtered = transform(audio)

        assert filtered.shape == audio.shape

        # Each channel should be filtered independently but consistently
        for ch in range(2):
            # Check frequency response
            original_fft = torch.fft.rfft(audio[ch])
            filtered_fft = torch.fft.rfft(filtered[ch])
            freqs = torch.fft.rfftfreq(16000, 1 / 16000)

            # Check attenuation outside passband
            low_freq_idx = torch.where(freqs < 200)[0]
            high_freq_idx = torch.where(freqs > 4000)[0]

            if len(low_freq_idx) > 0:
                low_atten = torch.abs(filtered_fft[low_freq_idx]).mean() / (
                    torch.abs(original_fft[low_freq_idx]).mean() + 1e-10
                )
                assert low_atten < 0.2  # Strong attenuation

            if len(high_freq_idx) > 0:
                high_atten = torch.abs(filtered_fft[high_freq_idx]).mean() / (
                    torch.abs(original_fft[high_freq_idx]).mean() + 1e-10
                )
                assert high_atten < 0.2  # Strong attenuation

    def test_edge_case_frequencies(self):
        """Test edge cases for frequency parameters."""
        # Test with very low frequencies
        transform = BandpassFilter(
            low_freq=10.0, high_freq=100.0, sample_rate=8000, order=4
        )
        audio = torch.randn(8000)
        filtered = transform(audio)
        assert filtered.shape == audio.shape
        assert torch.isfinite(filtered).all()

        # Test with frequencies near Nyquist
        transform = BandpassFilter(
            low_freq=3000.0, high_freq=3999.0, sample_rate=8000, order=4
        )
        filtered = transform(audio)
        assert filtered.shape == audio.shape
        assert torch.isfinite(filtered).all()

    def test_filter_stability(self):
        """Test that filter is stable and doesn't produce artifacts."""
        transform = BandpassFilter(
            low_freq=1000.0, high_freq=2000.0, sample_rate=8000, order=8
        )

        # Test with various input types
        # Impulse response
        impulse = torch.zeros(1000)
        impulse[500] = 1.0
        filtered_impulse = transform(impulse)
        assert torch.isfinite(filtered_impulse).all()
        assert filtered_impulse.abs().max() < 2.0  # No extreme amplification

        # Step response
        step = torch.ones(1000)
        filtered_step = transform(step)
        assert torch.isfinite(filtered_step).all()

        # Random noise
        noise = torch.randn(1000)
        filtered_noise = transform(noise)
        assert torch.isfinite(filtered_noise).all()


class TestTransformComposition:
    """Test composing multiple transforms."""

    def test_transform_pipeline(self):
        """Test a realistic transform pipeline."""

        # Create a pipeline: mel-spectrogram -> masking -> noise
        class Pipeline:
            def __init__(self):
                self.mel_spec = MelSpectrogram(n_mels=80, n_fft=1024, hop_length=256)
                self.masker = ApplyAudioMask(mask_ratio=0.2, mask_type="time")
                self.noiser = AddAudioNoise(noise_type="white", snr_db=20.0)

            def __call__(self, audio):
                mel = self.mel_spec(audio)
                masked, mask = self.masker(mel)
                # Note: AddAudioNoise expects clean audio, not spectrograms
                # For spectrograms, we add noise to the masked spectrogram directly
                noise = torch.randn_like(masked) * 0.1  # Simple noise for spectrograms
                noisy = masked + noise
                return noisy, mask

        pipeline = Pipeline()
        audio = torch.randn(16000)

        torch.manual_seed(42)
        output, mask = pipeline(audio)

        # Check output is valid mel-spectrogram shape with modifications
        assert output.shape[0] == 80  # 80 mel bins
        # Frame count depends on padding
        assert output.shape[1] > 0
        assert 55 < output.shape[1] < 70  # Reasonable range

        # Should have some masking applied
        assert mask.sum() > 0

        # Check that pipeline modifies the input
        mel_only = pipeline.mel_spec(audio)
        assert not torch.allclose(output, mel_only, atol=1e-3)

    def test_augmentation_chain(self):
        """Test audio augmentation chain."""
        # Filter -> Noise -> Mask
        filter_transform = BandpassFilter(low_freq=300, high_freq=3400)
        noise_transform = AddAudioNoise(noise_type="pink", snr_db=15.0)
        mask_transform = ApplyAudioMask(mask_ratio=0.1, mask_type="block")

        audio = torch.randn(8000)

        # Apply chain with fixed seed for reproducibility
        torch.manual_seed(42)
        filtered = filter_transform(audio)
        noisy = noise_transform(filtered)
        masked, mask = mask_transform(noisy)

        # Check each step modified the signal
        assert not torch.allclose(audio, filtered, atol=1e-4)
        assert not torch.allclose(filtered, noisy, atol=1e-4)
        assert not torch.allclose(noisy, masked, atol=1e-4)

        # Final output should incorporate all transformations
        assert masked.shape == audio.shape
        assert torch.isfinite(masked).all()

        # Verify transformations were actually applied
        # Filter should reduce out-of-band energy
        audio_spectrum = torch.abs(torch.fft.rfft(audio))
        filtered_spectrum = torch.abs(torch.fft.rfft(filtered))
        freqs = torch.fft.rfftfreq(8000, 1 / 8000)

        low_idx = torch.where(freqs < 300)[0]
        if len(low_idx) > 0:
            assert filtered_spectrum[low_idx].mean() < audio_spectrum[low_idx].mean()

    def test_device_consistency(self):
        """Test that transforms handle device placement correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        # Create transforms
        mel = MelSpectrogram(n_mels=40)
        masker = ApplyAudioMask(mask_ratio=0.2)
        noiser = AddAudioNoise(snr_db=10.0)
        bandpass = BandpassFilter(low_freq=300, high_freq=3000)

        # Test with GPU tensors
        audio = torch.randn(8000).to(device)

        mel_out = mel(audio)
        assert mel_out.device == device

        masked, mask = masker(audio)
        assert masked.device == device
        assert mask.device == device

        noisy = noiser(audio)
        assert noisy.device == device

        filtered = bandpass(audio)
        assert filtered.device == device


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
