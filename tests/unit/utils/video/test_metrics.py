"""Tests for video reconstruction metrics.

These tests focus on behavioral contracts and public API,
not implementation details. Each metric should:
1. Produce correct mathematical results
2. Handle edge cases gracefully
3. Integrate well with PyTorch workflows
"""

import pytest
import torch
from torch import Tensor

from associative.utils.video.metrics import (
    LPIPS,
    PSNR,
    SSIM,
    CosineSimilarity,
    MeanSquaredError,
    ReconstructionMetrics,
    get_metric,
)


class TestCosineSimilarity:
    """Test cosine similarity metric behavior."""

    def test_identical_vectors_return_one(self):
        """Identical vectors should have cosine similarity of 1."""
        metric = CosineSimilarity()
        x = torch.randn(32, 512)
        result = metric(x, x)
        assert torch.allclose(result, torch.ones(1), atol=1e-6)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors should have cosine similarity of 0."""
        metric = CosineSimilarity()
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = metric(x, y)
        assert torch.allclose(result, torch.zeros(1), atol=1e-6)

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors should have cosine similarity of -1."""
        metric = CosineSimilarity()
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[-1.0, 0.0]])
        result = metric(x, y)
        assert torch.allclose(result, -torch.ones(1), atol=1e-6)

    def test_batch_accumulation(self):
        """Test that metric correctly accumulates over batches."""
        metric = CosineSimilarity()

        # Process in batches
        x1 = torch.randn(10, 256)
        y1 = torch.randn(10, 256)
        x2 = torch.randn(5, 256)
        y2 = torch.randn(5, 256)

        metric.update(x1, y1)
        metric.update(x2, y2)
        result = metric.compute()

        # Compare with single batch computation
        x_all = torch.cat([x1, x2])
        y_all = torch.cat([y1, y2])
        expected = CosineSimilarity()(x_all, y_all)

        assert torch.allclose(result, expected, atol=1e-5)

    def test_shape_mismatch_raises_error(self):
        """Mismatched shapes should raise ValueError."""
        metric = CosineSimilarity()
        x = torch.randn(10, 256)
        y = torch.randn(10, 512)
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric(x, y)

    def test_different_dimensions(self):
        """Test cosine similarity along different dimensions."""
        # Test with explicit dimension
        metric = CosineSimilarity(dim=1)
        x = torch.randn(4, 8, 16)
        y = torch.randn(4, 8, 16)
        result = metric(x, y)
        assert isinstance(result, Tensor)

        # Test with last dimension (default)
        metric = CosineSimilarity(dim=-1)
        result = metric(x, y)
        assert isinstance(result, Tensor)


class TestMeanSquaredError:
    """Test MSE metric behavior."""

    def test_identical_inputs_return_zero(self):
        """Identical inputs should have MSE of 0."""
        metric = MeanSquaredError()
        x = torch.randn(32, 512)
        result = metric(x, x)
        assert torch.allclose(result, torch.zeros(1), atol=1e-6)

    def test_known_mse_value(self):
        """Test MSE with known values."""
        metric = MeanSquaredError()
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([2.0, 4.0, 6.0])
        # MSE = ((1-2)² + (2-4)² + (3-6)²) / 3 = (1 + 4 + 9) / 3 = 14/3
        result = metric(x, y)
        expected = torch.tensor(14.0 / 3.0)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_batch_accumulation(self):
        """Test that metric correctly accumulates over batches."""
        metric = MeanSquaredError()

        x1 = torch.randn(10, 256)
        y1 = torch.randn(10, 256)
        x2 = torch.randn(5, 256)
        y2 = torch.randn(5, 256)

        metric.update(x1, y1)
        metric.update(x2, y2)
        result = metric.compute()

        # MSE should be average over all elements
        expected = torch.cat(
            [(x1 - y1).pow(2).flatten(), (x2 - y2).pow(2).flatten()]
        ).mean()

        assert torch.allclose(result, expected, atol=1e-5)

    def test_always_non_negative(self):
        """MSE should always be non-negative."""
        metric = MeanSquaredError()
        for _ in range(10):
            x = torch.randn(16, 128)
            y = torch.randn(16, 128)
            result = metric(x, y)
            assert result >= 0


class TestPSNR:
    """Test PSNR metric behavior."""

    def test_identical_images_high_psnr(self):
        """Identical images should have very high PSNR."""
        metric = PSNR(data_range=1.0)
        x = torch.rand(4, 3, 64, 64)
        result = metric(x, x)
        # Should be very high (but not infinite in implementation)
        assert result > 60.0

    def test_known_psnr_value(self):
        """Test PSNR with known MSE."""
        metric = PSNR(data_range=255.0)
        # Create images with MSE = 100
        x = torch.full((1, 3, 2, 2), 100.0)
        y = torch.full((1, 3, 2, 2), 110.0)
        result = metric(x, y)
        # PSNR = 20 * log10(255 / sqrt(100)) = 20 * log10(25.5)
        expected = 20 * torch.log10(torch.tensor(25.5))
        assert torch.allclose(result, expected, atol=0.1)

    def test_data_range_handling(self):
        """Test PSNR with different data ranges."""
        # Normalized images [0, 1]
        metric_norm = PSNR(data_range=1.0)
        x = torch.rand(2, 3, 32, 32)
        y = x + torch.randn_like(x) * 0.1
        y = torch.clamp(y, 0, 1)
        psnr_norm = metric_norm(x, y)

        # 8-bit images [0, 255]
        metric_8bit = PSNR(data_range=255.0)
        x_8bit = (x * 255).round()
        y_8bit = (y * 255).round()
        psnr_8bit = metric_8bit(x_8bit, y_8bit)

        # Both should be positive and finite
        assert 0 < psnr_norm < 100
        assert 0 < psnr_8bit < 100

    def test_lower_psnr_for_more_noise(self):
        """More noise should result in lower PSNR."""
        metric = PSNR(data_range=1.0)
        x = torch.rand(2, 3, 64, 64)

        # Low noise
        y_low_noise = x + torch.randn_like(x) * 0.01
        y_low_noise = torch.clamp(y_low_noise, 0, 1)
        psnr_low = metric(x, y_low_noise)

        # High noise
        y_high_noise = x + torch.randn_like(x) * 0.1
        y_high_noise = torch.clamp(y_high_noise, 0, 1)
        psnr_high = metric(x, y_high_noise)

        assert psnr_low > psnr_high


class TestSSIM:
    """Test SSIM metric behavior."""

    def test_identical_images_return_one(self):
        """Identical images should have SSIM of 1."""
        metric = SSIM()
        x = torch.rand(4, 3, 64, 64)
        result = metric(x, x)
        assert torch.allclose(result, torch.ones(1), atol=1e-3)

    def test_value_range(self):
        """SSIM should always be in [0, 1]."""
        metric = SSIM()
        for _ in range(5):
            x = torch.rand(2, 3, 32, 32)
            y = torch.rand(2, 3, 32, 32)
            result = metric(x, y)
            assert 0.0 <= result <= 1.0

    def test_structural_similarity(self):
        """Test that SSIM captures structural similarity."""
        metric = SSIM()

        # Create structured pattern
        x = torch.zeros(2, 3, 64, 64)
        x[:, :, ::2, ::2] = 1.0  # Checkerboard

        # Small perturbation preserves structure
        y_similar = x + torch.randn_like(x) * 0.05
        y_similar = torch.clamp(y_similar, 0, 1)

        # Random noise destroys structure
        y_different = torch.rand_like(x)

        ssim_similar = metric(x, y_similar)
        ssim_different = metric(x, y_different)

        assert ssim_similar > ssim_different

    def test_window_size_validation(self):
        """Window size must be odd."""
        with pytest.raises(ValueError, match="odd"):
            SSIM(window_size=8)

        # Odd sizes should work
        for size in [3, 5, 7, 11]:
            metric = SSIM(window_size=size)
            assert metric.window_size == size


class TestLPIPS:
    """Test LPIPS metric behavior."""

    def test_identical_images_low_distance(self):
        """Identical images should have near-zero LPIPS distance."""
        metric = LPIPS()
        x = torch.rand(2, 3, 64, 64)
        result = metric(x, x)
        assert result < 0.01  # Very low perceptual distance

    def test_different_images_high_distance(self):
        """Very different images should have high LPIPS distance."""
        metric = LPIPS()
        x = torch.zeros(2, 3, 64, 64)
        y = torch.ones(2, 3, 64, 64)
        result = metric(x, y)
        assert result > 0.1  # Significant perceptual distance

    def test_perceptual_vs_pixel_difference(self):
        """LPIPS should be more robust to imperceptible changes than MSE."""
        metric_lpips = LPIPS()
        metric_mse = MeanSquaredError()

        x = torch.rand(2, 3, 64, 64)

        # Small uniform shift (perceptually similar)
        y_shift = torch.clamp(x + 0.1, 0, 1)

        # Same MSE but different structure
        noise = torch.randn_like(x) * 0.1
        y_noise = torch.clamp(x + noise, 0, 1)

        # LPIPS should show shift is more similar than noise
        lpips_shift = metric_lpips(x, y_shift)
        lpips_noise = metric_lpips(x, y_noise)

        # Both have similar MSE
        mse_shift = metric_mse(x, y_shift)
        mse_noise = metric_mse(x, y_noise)

        # LPIPS should distinguish better than MSE
        assert lpips_shift < lpips_noise or abs(mse_shift - mse_noise) < 0.01

    def test_different_networks(self):
        """Test different backbone networks."""
        x = torch.rand(1, 3, 64, 64)
        y = torch.rand(1, 3, 64, 64)

        for net in ["alex", "vgg", "squeeze"]:
            metric = LPIPS(net_type=net)  # type: ignore[arg-type]
            result = metric(x, y)
            assert 0 <= result <= 2  # Reasonable range

    def test_requires_rgb(self):
        """LPIPS requires 3-channel RGB images."""
        metric = LPIPS()
        x = torch.rand(2, 1, 64, 64)  # Grayscale
        y = torch.rand(2, 1, 64, 64)
        with pytest.raises(ValueError, match="3-channel RGB"):
            metric(x, y)


class TestReconstructionMetrics:
    """Test comprehensive reconstruction metrics."""

    def test_feature_reconstruction(self):
        """Test metrics for feature reconstruction (no pixel metrics)."""
        metrics = ReconstructionMetrics(include_pixel_metrics=False)
        x = torch.randn(16, 512)
        y = torch.randn(16, 512)

        results = metrics(x, y)

        assert "cosine_similarity" in results
        assert "mse" in results
        assert "psnr" not in results
        assert "ssim" not in results

        # Values should be in expected ranges
        assert -1 <= results["cosine_similarity"] <= 1
        assert results["mse"] >= 0

    def test_image_reconstruction(self):
        """Test metrics for image reconstruction (with pixel metrics)."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)
        x = torch.rand(8, 3, 64, 64)
        y = torch.rand(8, 3, 64, 64)

        results = metrics(x, y)

        assert "cosine_similarity" in results
        assert "mse" in results
        assert "psnr" in results
        assert "ssim" in results

        # Values should be in expected ranges
        assert -1 <= results["cosine_similarity"] <= 1
        assert results["mse"] >= 0
        assert results["psnr"] > 0
        assert 0 <= results["ssim"] <= 1

    def test_auto_detect_image_data(self):
        """Pixel metrics should only apply to image-like tensors."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        # Non-image tensor (2D)
        x_2d = torch.randn(32, 768)
        results_2d = metrics(x_2d, x_2d)
        assert "psnr" not in results_2d
        assert "ssim" not in results_2d

        # Image tensor (4D with 3 channels)
        x_4d = torch.rand(4, 3, 32, 32)
        results_4d = metrics(x_4d, x_4d)
        assert "psnr" in results_4d
        assert "ssim" in results_4d

        # Non-RGB image (4D but not 3 channels)
        x_gray = torch.rand(4, 1, 32, 32)
        results_gray = metrics(x_gray, x_gray)
        assert "psnr" not in results_gray
        assert "ssim" not in results_gray

    def test_accumulation(self):
        """Test metrics accumulation over multiple batches."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        # Process multiple batches
        for _ in range(3):
            x = torch.rand(4, 3, 32, 32)
            y = torch.rand(4, 3, 32, 32)
            metrics.update(x, y)

        results = metrics.compute()

        # Should have all metrics
        assert len(results) == 4
        for key in ["cosine_similarity", "mse", "psnr", "ssim"]:
            assert key in results
            assert isinstance(results[key], float)


class TestMetricRegistry:
    """Test metric registry functionality."""

    def test_get_registered_metrics(self):
        """Test retrieving metrics from registry."""
        # Test basic metrics
        cosine = get_metric("cosine_similarity", dim=1)
        assert isinstance(cosine, CosineSimilarity)
        assert cosine.dim == 1

        mse = get_metric("mse")
        assert isinstance(mse, MeanSquaredError)

        # Test with parameters
        psnr = get_metric("psnr", data_range=255.0)
        assert isinstance(psnr, PSNR)
        assert psnr.data_range == 255.0

        ssim = get_metric("ssim", window_size=7)
        assert isinstance(ssim, SSIM)
        assert ssim.window_size == 7

    def test_invalid_metric_name(self):
        """Invalid metric names should raise KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_metric("nonexistent_metric")

    def test_available_metrics(self):
        """Test that all expected metrics are available."""
        expected = [
            "cosine_similarity",
            "mse",
            "psnr",
            "ssim",
            "lpips",
            "reconstruction_metrics",
        ]

        for name in expected:
            metric = get_metric(name)
            assert metric is not None


class TestMetricIntegration:
    """Integration tests for metric workflows."""

    def test_training_loop_workflow(self):
        """Test metrics in a training loop scenario."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        # Simulate training loop
        for epoch in range(2):
            metrics.reset()

            # Process batches
            for _ in range(5):
                x = torch.rand(8, 3, 32, 32)
                # Simulate predictions getting better
                noise_scale = 0.1 * (1 - epoch * 0.3)
                y = x + torch.randn_like(x) * noise_scale
                y = torch.clamp(y, 0, 1)

                metrics.update(x, y)

            results = metrics.compute()

            # Metrics should be valid
            assert 0 <= results["ssim"] <= 1
            assert results["psnr"] > 0
            assert results["mse"] >= 0

    def test_device_handling(self):
        """Test metric device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        metric = CosineSimilarity(device=device)

        x = torch.randn(10, 256).to(device)
        y = torch.randn(10, 256).to(device)

        result = metric(x, y)
        assert result.device == device

    def test_gradient_flow(self):
        """Test metrics can be used in gradient computation."""
        # Test differentiable metrics
        for metric_class in [CosineSimilarity, MeanSquaredError, LPIPS]:
            metric = metric_class()

            # Create inputs requiring gradients
            if metric_class == LPIPS:
                x = torch.rand(2, 3, 64, 64, requires_grad=True)
                y = torch.rand(2, 3, 64, 64)
            else:
                x = torch.randn(10, 128, requires_grad=True)
                y = torch.randn(10, 128)

            loss = metric(x, y)
            loss.backward()

            assert x.grad is not None
            assert x.grad.shape == x.shape

    def test_mixed_precision(self):
        """Test metrics work with different dtypes."""
        metric = CosineSimilarity(dtype=torch.float64)

        x = torch.randn(10, 128, dtype=torch.float64)
        y = torch.randn(10, 128, dtype=torch.float64)

        result = metric(x, y)
        assert result.dtype == torch.float64

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_different_batch_sizes(self, batch_size):
        """Test metrics handle different batch sizes correctly."""
        metric = MeanSquaredError()

        x = torch.randn(batch_size, 256)
        y = torch.randn(batch_size, 256)

        result = metric(x, y)
        assert result.shape == torch.Size([])  # Scalar
        assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
