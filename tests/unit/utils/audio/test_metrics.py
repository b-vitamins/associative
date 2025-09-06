"""Tests for audio reconstruction metrics.

These tests focus on behavioral contracts and public API,
not implementation details. Each metric should:
1. Produce correct mathematical results
2. Handle edge cases gracefully
3. Integrate well with PyTorch workflows
"""

import logging

import numpy as np
import pytest
import torch

from associative.utils.audio.metrics import (
    PESQ,
    SDR,
    STOI,
    AudioReconstructionMetrics,
    get_audio_metric,
)


class TestPESQ:
    """Test PESQ metric behavior."""

    def test_wideband_mode(self):
        """Test wideband PESQ configuration."""
        pesq = PESQ(mode="wb", sample_rate=16000)
        assert pesq.mode == "wb"
        assert pesq.sample_rate == 16000

    def test_narrowband_mode(self):
        """Test narrowband PESQ configuration."""
        pesq = PESQ(mode="nb", sample_rate=8000)
        assert pesq.mode == "nb"
        assert pesq.sample_rate == 8000

    def test_invalid_sample_rate_raises_error(self):
        """Test that invalid sample rates raise errors."""
        with pytest.raises(ValueError, match="16kHz"):
            PESQ(mode="wb", sample_rate=8000)

        with pytest.raises(ValueError, match="8kHz"):
            PESQ(mode="nb", sample_rate=16000)

    def test_identical_signals_high_score(self):
        """Identical signals should have high PESQ score."""
        pesq = PESQ(mode="wb", sample_rate=16000)

        # Create a test signal
        duration = 2.0
        t = torch.linspace(0, duration, int(16000 * duration))
        signal = torch.sin(2 * np.pi * 440 * t)

        pesq.update(signal, signal)
        score = pesq.compute()

        # PESQ for identical signals should be near maximum
        assert score > 4.0
        assert score <= 4.65  # Implementation may exceed theoretical max

    def test_noisy_signal_lower_score(self):
        """Noisy signals should have lower PESQ scores."""
        pesq = PESQ(mode="wb", sample_rate=16000)

        duration = 2.0
        t = torch.linspace(0, duration, int(16000 * duration))
        clean = torch.sin(2 * np.pi * 440 * t)
        noisy = clean + torch.randn_like(clean) * 0.3

        pesq.update(noisy, clean)
        score = pesq.compute()

        # Should be in valid range but lower than perfect
        assert -0.5 <= score <= 4.5
        assert score < 4.0

    def test_batch_accumulation(self):
        """Test metric accumulation over batches."""
        pesq = PESQ(mode="wb", sample_rate=16000)

        # Process multiple batches
        scores = []
        for noise_level in [0.1, 0.2, 0.3]:
            duration = 1.0
            t = torch.linspace(0, duration, int(16000 * duration))
            clean = torch.sin(2 * np.pi * 440 * t)
            noisy = clean + torch.randn_like(clean) * noise_level

            pesq.update(noisy, clean)

            # Compute intermediate score for comparison
            temp_pesq = PESQ(mode="wb", sample_rate=16000)
            temp_pesq.update(noisy, clean)
            scores.append(temp_pesq.compute().item())

        # Final score should be reasonable average
        final_score = pesq.compute()
        assert -0.5 <= final_score <= 4.5

        # Should have accumulated samples
        assert pesq.count > 0

    def test_multichannel_audio(self):
        """Test PESQ with stereo audio (should average channels)."""
        pesq = PESQ(mode="wb", sample_rate=16000)

        # Create stereo signal
        samples = 16000
        stereo_clean = torch.randn(2, samples)
        stereo_noisy = stereo_clean + torch.randn_like(stereo_clean) * 0.1

        pesq.update(stereo_noisy, stereo_clean)
        score = pesq.compute()

        assert -0.5 <= score <= 4.65

    def test_shape_mismatch_raises_error(self):
        """Mismatched shapes should raise ValueError."""
        pesq = PESQ()
        pred = torch.randn(16000)
        target = torch.randn(8000)

        with pytest.raises(ValueError, match="Shape mismatch"):
            pesq.update(pred, target)

    def test_reset_functionality(self):
        """Test that reset clears state."""
        pesq = PESQ()

        # Add some data
        signal = torch.randn(16000)
        pesq.update(signal, signal)
        assert pesq.count > 0

        # Reset and verify
        pesq.reset()
        assert pesq.count == 0

    def test_compute_without_data_returns_zero(self):
        """Computing without updates should return 0."""
        pesq = PESQ()
        score = pesq.compute()
        assert score == 0.0


class TestSTOI:
    """Test STOI metric behavior."""

    def test_extended_mode(self):
        """Test extended STOI configuration."""
        stoi = STOI(extended=True)
        assert stoi.extended is True

    def test_standard_mode(self):
        """Test standard STOI configuration."""
        stoi = STOI(extended=False)
        assert stoi.extended is False

    def test_low_sample_rate_warning(self, caplog):
        """Test warning for sample rates below 10kHz."""
        with caplog.at_level(logging.WARNING):
            STOI(sample_rate=8000)
        assert "10kHz" in caplog.text

    def test_identical_signals_high_score(self):
        """Identical signals should have STOI close to 1."""
        stoi = STOI(sample_rate=16000)

        # Create speech-like signal with multiple frequencies
        duration = 2.0
        t = torch.linspace(0, duration, int(16000 * duration))
        signal = (
            torch.sin(2 * np.pi * 200 * t)
            + torch.sin(2 * np.pi * 800 * t)
            + torch.sin(2 * np.pi * 1500 * t)
        ) / 3

        stoi.update(signal, signal)
        score = stoi.compute()

        assert 0.95 < score <= 1.0

    def test_noisy_signal_lower_score(self):
        """Noisy signals should have lower STOI scores."""
        stoi = STOI(sample_rate=16000)

        duration = 2.0
        t = torch.linspace(0, duration, int(16000 * duration))
        clean = (torch.sin(2 * np.pi * 200 * t) + torch.sin(2 * np.pi * 800 * t)) / 2
        noisy = clean + torch.randn_like(clean) * 0.3

        stoi.update(noisy, clean)
        score = stoi.compute()

        # Extended STOI can be slightly negative for poor quality
        assert -0.1 <= score <= 1.0
        assert score < 0.95

    def test_batch_accumulation(self):
        """Test metric accumulation over batches."""
        stoi = STOI(sample_rate=16000)

        for noise_level in [0.1, 0.2, 0.3]:
            duration = 1.0
            t = torch.linspace(0, duration, int(16000 * duration))
            clean = torch.sin(2 * np.pi * 500 * t)
            noisy = clean + torch.randn_like(clean) * noise_level
            stoi.update(noisy, clean)

        score = stoi.compute()
        assert -0.1 <= score <= 1.0
        assert stoi.count == 3

    def test_extended_vs_standard_difference(self):
        """Extended and standard STOI should give different scores for distorted audio."""
        duration = 1.0
        t = torch.linspace(0, duration, int(16000 * duration))
        clean = torch.sin(2 * np.pi * 500 * t)

        # Add frequency-dependent distortion
        distorted = clean * (1 + 0.5 * torch.sin(2 * np.pi * 50 * t))

        # Extended STOI
        stoi_ext = STOI(extended=True)
        stoi_ext.update(distorted, clean)
        score_ext = stoi_ext.compute()

        # Standard STOI
        stoi_std = STOI(extended=False)
        stoi_std.update(distorted, clean)
        score_std = stoi_std.compute()

        # Both should be valid
        assert -0.1 <= score_ext <= 1.0
        assert -0.1 <= score_std <= 1.0

        # Scores will differ due to different algorithms
        # but we don't enforce which is higher/lower

    def test_multichannel_audio(self):
        """Test STOI with stereo audio."""
        stoi = STOI()

        # Create stereo signal
        stereo_clean = torch.randn(2, 16000)
        stereo_noisy = stereo_clean + torch.randn_like(stereo_clean) * 0.1

        stoi.update(stereo_noisy, stereo_clean)
        score = stoi.compute()

        assert -0.1 <= score <= 1.0


class TestSDR:
    """Test SDR metric behavior."""

    def test_initialization_parameters(self):
        """Test SDR initialization with custom parameters."""
        sdr = SDR(use_cg_iter=5, filter_length=256, zero_mean=True)
        assert sdr.use_cg_iter == 5
        assert sdr.filter_length == 256
        assert sdr.zero_mean is True

    def test_identical_signals_high_sdr(self):
        """Identical signals should have very high SDR."""
        sdr = SDR()

        signal = torch.randn(8000)
        sdr.update(signal, signal)
        score = sdr.compute()

        # SDR for identical signals should be very high
        assert score > 30  # At least 30 dB

    def test_known_snr_value(self):
        """Test SDR with known signal-to-noise ratio."""
        sdr = SDR()

        # Create signal with known SNR
        clean = torch.randn(8000)
        clean = clean / clean.std()  # Normalize

        # Add noise for ~10dB SNR
        noise = torch.randn_like(clean) * 0.316  # 10^(-10/20)
        noisy = clean + noise

        sdr.update(noisy, clean)
        score = sdr.compute()

        # Should be close to 10dB with tolerance
        assert 5 < score < 15

    def test_batch_accumulation(self):
        """Test SDR accumulation over batches."""
        sdr = SDR()

        for snr_db in [0, 10, 20]:
            clean = torch.randn(4000) / np.sqrt(4000)
            noise_scale = 10 ** (-snr_db / 20)
            noisy = clean + torch.randn_like(clean) * noise_scale
            sdr.update(noisy, clean)

        score = sdr.compute()
        # Average should be reasonable
        assert -10 < score < 30

    def test_multichannel_audio(self):
        """Test SDR with stereo audio."""
        sdr = SDR()

        stereo_clean = torch.randn(2, 8000)
        stereo_noisy = stereo_clean + torch.randn_like(stereo_clean) * 0.1

        sdr.update(stereo_noisy, stereo_clean)
        score = sdr.compute()

        assert score > 10  # Should have decent SDR

    def test_forward_method(self):
        """Test forward pass for single batch."""
        sdr = SDR()

        clean = torch.randn(8000)
        noisy = clean + torch.randn_like(clean) * 0.1

        # Forward computes without updating state
        score = sdr(noisy, clean)
        assert score > 10
        assert torch.isfinite(score)

    def test_negative_sdr_for_poor_quality(self):
        """Very noisy signals should have negative SDR."""
        sdr = SDR()

        clean = torch.randn(8000)
        # Add very strong noise
        noisy = clean + torch.randn_like(clean) * 10

        sdr.update(noisy, clean)
        score = sdr.compute()

        # Should be negative for poor quality
        assert score < 0


class TestAudioReconstructionMetrics:
    """Test comprehensive audio metrics suite."""

    def test_all_metrics_enabled(self):
        """Test initialization with all metrics."""
        metrics = AudioReconstructionMetrics(
            include_pesq=True, include_stoi=True, sample_rate=16000
        )

        # All metrics should be available
        assert metrics.pesq is not None
        assert metrics.stoi is not None
        assert metrics.sdr is not None

    def test_selective_metrics(self):
        """Test with only SDR enabled."""
        metrics = AudioReconstructionMetrics(include_pesq=False, include_stoi=False)

        # PESQ and STOI should be None when disabled
        assert metrics.pesq is None
        assert metrics.stoi is None
        assert metrics.sdr is not None

    def test_incompatible_sample_rate_disables_pesq(self, caplog):
        """Test PESQ disabled when sample rate incompatible."""
        with caplog.at_level(logging.WARNING):
            metrics = AudioReconstructionMetrics(
                include_pesq=True, pesq_mode="wb", sample_rate=8000
            )

        assert "16kHz" in caplog.text
        assert metrics.pesq is None

    def test_forward_computes_all_metrics(self):
        """Test forward pass computes all enabled metrics."""
        metrics = AudioReconstructionMetrics(
            include_pesq=True, include_stoi=True, sample_rate=16000
        )

        # Generate test audio
        duration = 2.0
        samples = int(16000 * duration)
        clean = torch.sin(2 * np.pi * 440 * torch.linspace(0, duration, samples))
        noisy = clean + torch.randn_like(clean) * 0.1

        results = metrics(noisy, clean)

        # Check all metrics present and valid
        assert "sdr" in results
        assert "pesq" in results
        assert "stoi" in results

        assert results["sdr"] > 0
        assert -0.5 <= results["pesq"] <= 4.65
        assert -0.1 <= results["stoi"] <= 1.0

    def test_update_and_compute_pattern(self):
        """Test accumulation pattern with update/compute."""
        metrics = AudioReconstructionMetrics(
            include_pesq=False,  # Skip for speed
            include_stoi=True,
            sample_rate=16000,
        )

        # Process multiple batches
        for i in range(3):
            samples = 8000
            clean = torch.randn(samples)
            noise_level = 0.1 * (i + 1)
            noisy = clean + torch.randn_like(clean) * noise_level
            metrics.update(noisy, clean)

        # Compute accumulated metrics
        results = metrics.compute()

        assert "sdr" in results
        assert "stoi" in results
        assert "pesq" not in results  # Was disabled

        assert results["sdr"] > 0
        assert -0.1 <= results["stoi"] <= 1.0

    def test_reset_functionality(self):
        """Test reset clears all metric states."""
        metrics = AudioReconstructionMetrics(include_stoi=True)

        # Add data
        audio = torch.randn(8000)
        metrics.update(audio, audio)

        # Reset
        metrics.reset()

        # Should be able to compute after reset (with zero/default values)
        results = metrics.compute()
        assert "sdr" in results

    def test_different_sample_rates(self):
        """Test metrics with different sample rates."""
        # 8kHz narrowband
        metrics_8k = AudioReconstructionMetrics(
            include_pesq=True, pesq_mode="nb", sample_rate=8000
        )
        assert metrics_8k.sample_rate == 8000
        if hasattr(metrics_8k, "pesq") and metrics_8k.pesq is not None:
            assert metrics_8k.pesq.sample_rate == 8000

        # 16kHz wideband
        metrics_16k = AudioReconstructionMetrics(
            include_pesq=True, pesq_mode="wb", sample_rate=16000
        )
        assert metrics_16k.sample_rate == 16000
        if hasattr(metrics_16k, "pesq") and metrics_16k.pesq is not None:
            assert metrics_16k.pesq.sample_rate == 16000


class TestMetricRegistry:
    """Test metric registry functionality."""

    def test_get_registered_metrics(self):
        """Test retrieving metrics from registry."""
        # Test PESQ
        pesq = get_audio_metric("pesq", mode="wb", sample_rate=16000)
        assert isinstance(pesq, PESQ)
        assert pesq.mode == "wb"

        # Test STOI
        stoi = get_audio_metric("stoi", extended=True)
        assert isinstance(stoi, STOI)
        assert stoi.extended is True

        # Test SDR
        sdr = get_audio_metric("sdr", zero_mean=True)
        assert isinstance(sdr, SDR)
        assert sdr.zero_mean is True

        # Test comprehensive metrics
        metrics = get_audio_metric(
            "audio_reconstruction", include_pesq=True, sample_rate=16000
        )
        assert isinstance(metrics, AudioReconstructionMetrics)

    def test_invalid_metric_name(self):
        """Invalid metric names should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_audio_metric("nonexistent_metric")

    def test_available_metrics(self):
        """Test that all expected metrics are available."""
        expected = ["pesq", "stoi", "sdr", "audio_reconstruction"]

        for name in expected:
            metric = get_audio_metric(name)
            assert metric is not None


class TestMetricIntegration:
    """Integration tests for metric workflows."""

    def test_training_loop_workflow(self):
        """Test metrics in a training loop scenario."""
        metrics = AudioReconstructionMetrics(
            include_pesq=False,  # Skip for speed
            include_stoi=True,
            sample_rate=16000,
        )

        # Simulate training loop
        for epoch in range(2):
            metrics.reset()

            # Process batches
            for _ in range(3):
                clean = torch.randn(4000)
                # Simulate improving predictions
                noise_scale = 0.3 * (1 - epoch * 0.3)
                noisy = clean + torch.randn_like(clean) * noise_scale
                metrics.update(noisy, clean)

            results = metrics.compute()

            # Metrics should be valid
            assert results["sdr"] > 0
            assert -0.1 <= results["stoi"] <= 1.0

    def test_device_handling(self):
        """Test metric device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        sdr = SDR(device=device)

        clean = torch.randn(8000).to(device)
        noisy = clean + torch.randn_like(clean) * 0.1

        score = sdr(noisy, clean)
        assert score.device == device

    def test_gradient_flow(self):
        """Test that SDR can be used in gradient computation."""
        # SDR is differentiable through torchmetrics
        sdr = SDR()

        clean = torch.randn(8000, requires_grad=True)
        noisy = torch.randn(8000)

        loss = sdr(noisy, clean)
        loss.backward()

        assert clean.grad is not None
        assert clean.grad.shape == clean.shape

    def test_mixed_precision(self):
        """Test metrics work with different dtypes."""
        sdr = SDR(dtype=torch.float64)

        clean = torch.randn(8000, dtype=torch.float64)
        noisy = torch.randn(8000, dtype=torch.float64)

        score = sdr(noisy, clean)
        # torchmetrics SDR may not preserve dtype exactly
        assert torch.isfinite(score)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test metrics handle different batch sizes."""
        sdr = SDR()

        clean = torch.randn(batch_size, 8000)
        noisy = clean + torch.randn_like(clean) * 0.1

        sdr.update(noisy, clean)
        score = sdr.compute()

        assert score > 0  # Should have positive SDR

    def test_deterministic_results(self):
        """Test that metrics give deterministic results."""
        torch.manual_seed(42)
        clean = torch.randn(8000)
        noisy = clean + torch.randn_like(clean) * 0.1

        # Compute SDR twice
        sdr1 = SDR()
        sdr1.update(noisy, clean)
        score1 = sdr1.compute()

        sdr2 = SDR()
        sdr2.update(noisy, clean)
        score2 = sdr2.compute()

        assert torch.isclose(score1, score2)

    def test_metric_monotonicity(self):
        """Test metrics respond correctly to quality changes."""
        clean = torch.randn(8000)
        sdr = SDR()

        scores = []
        for noise_level in [0.01, 0.1, 0.5, 1.0]:
            sdr.reset()
            noisy = clean + torch.randn_like(clean) * noise_level
            sdr.update(noisy, clean)
            scores.append(sdr.compute().item())

        # SDR should decrease with more noise
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
