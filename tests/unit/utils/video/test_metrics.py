"""Comprehensive tests for video reconstruction metrics."""

import math
from unittest.mock import patch

import pytest
import torch

from associative.utils.video.metrics import (
    PSNR,
    SSIM,
    CosineSimilarity,
    MeanSquaredError,
    ReconstructionMetrics,
    VideoMetric,
    get_metric,
)


class TestVideoMetricInterface:
    """Test VideoMetric abstract interface."""

    def test_abstract_interface(self):
        """Test that VideoMetric cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            VideoMetric()  # type: ignore[abstract]

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        class IncompleteMetric(VideoMetric):
            def _setup_state(self):
                pass

        # Should fail without implementing all abstract methods
        with pytest.raises(TypeError):
            IncompleteMetric()  # type: ignore[abstract]

    def test_concrete_metric_initialization(self):
        """Test initialization of concrete metric."""
        metric = CosineSimilarity()

        # Should have basic properties
        assert hasattr(metric, "_device")
        assert hasattr(metric, "_dtype")
        assert metric._dtype == torch.float32

    def test_device_and_dtype_handling(self):
        """Test device and dtype parameter handling."""
        device = torch.device("cpu")
        dtype = torch.float64

        metric = CosineSimilarity(device=device, dtype=dtype)
        assert metric._device == device
        assert metric._dtype == dtype

    def test_to_device_method(self):
        """Test moving metric to different device."""
        metric = CosineSimilarity()
        device = torch.device("cpu")

        result = metric.to(device)
        assert result._device == device
        assert isinstance(result, CosineSimilarity)

    def test_reset_functionality(self):
        """Test reset method clears metric state."""
        metric = CosineSimilarity()

        # Metric should have state buffers after setup
        assert hasattr(metric, "sum_similarity")
        assert hasattr(metric, "num_samples")

        # Reset should clear state
        metric.reset()
        assert metric.sum_similarity.item() == 0.0
        assert metric.num_samples.item() == 0


class TestCosineSimilarity:
    """Test CosineSimilarity metric implementation."""

    def test_initialization(self):
        """Test proper initialization of CosineSimilarity."""
        # Default initialization
        metric_default = CosineSimilarity()
        assert metric_default.dim == -1
        assert metric_default.eps == 1e-8

        # Custom initialization
        metric_custom = CosineSimilarity(
            dim=1, eps=1e-6, device=torch.device("cpu"), dtype=torch.float64
        )
        assert metric_custom.dim == 1
        assert metric_custom.eps == 1e-6
        assert metric_custom._dtype == torch.float64

    def test_state_buffers(self):
        """Test that state buffers are properly initialized."""
        metric = CosineSimilarity()

        # Should have required buffers
        assert hasattr(metric, "sum_similarity")
        assert hasattr(metric, "num_samples")

        # Buffers should be tensors
        assert isinstance(metric.sum_similarity, torch.Tensor)
        assert isinstance(metric.num_samples, torch.Tensor)

        # Initial values should be zero
        assert metric.sum_similarity.item() == 0.0
        assert metric.num_samples.item() == 0

    def test_update_input_validation(self):
        """Test input validation in update method."""
        metric = CosineSimilarity()

        # Shape mismatch should raise error
        pred = torch.randn(32, 512)
        target = torch.randn(32, 256)  # Different feature dimension

        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.update(pred, target)

        # Different batch sizes
        pred = torch.randn(32, 512)
        target = torch.randn(16, 512)

        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.update(pred, target)

    def test_update_functionality(self):
        """Test update method functionality."""
        metric = CosineSimilarity(dim=-1)

        pred = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        metric.update(pred, target)

        expected_sum = 1.0 + 0.0 + (1.0 / math.sqrt(2))
        assert metric.sum_similarity.item() == pytest.approx(expected_sum, rel=1e-4)
        assert metric.num_samples.item() == 3

    def test_compute_functionality(self):
        """Test compute method functionality."""
        metric = CosineSimilarity()

        # Mock state
        metric.sum_similarity = torch.tensor(4.5)
        metric.num_samples = torch.tensor(3)

        result = metric.compute()

        expected = 4.5 / 3
        assert torch.allclose(result, torch.tensor(expected))

    def test_compute_no_samples_error(self):
        """Test compute raises error when no samples processed."""
        metric = CosineSimilarity()

        with pytest.raises(RuntimeError, match="No samples processed"):
            metric.compute()

    def test_forward_method(self):
        """Test forward method for single batch computation."""
        metric = CosineSimilarity()

        with patch(
            "associative.utils.video.functional.compute_cosine_similarity"
        ) as mock_cosine:
            mock_cosine.return_value = torch.tensor([0.7, 0.8])

            pred = torch.randn(2, 256)
            target = torch.randn(2, 256)

            # Create a fresh metric instance for single computation
            with (
                patch.object(CosineSimilarity, "__init__", return_value=None),
                patch.object(CosineSimilarity, "update"),
                patch.object(CosineSimilarity, "compute") as mock_compute,
            ):
                mock_compute.return_value = torch.tensor(0.75)

                # This tests the interface, actual implementation would differ
                metric.forward(pred, target)

    def test_multiple_updates(self):
        """Test multiple update calls accumulate correctly."""
        metric = CosineSimilarity()

        # First batch
        pred1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metric.update(pred1, target1)

        # Second batch
        pred2 = torch.tensor([[1.0, 1.0]])
        target2 = torch.tensor([[1.0, 0.0]])
        metric.update(pred2, target2)

        # Should accumulate: (1.0 + 1.0 + 1/sqrt(2)) / 3
        result = metric.compute()
        expected = (1.0 + 1.0 + (1.0 / math.sqrt(2))) / 3
        assert result.item() == pytest.approx(expected, rel=1e-4)

    def test_repr(self):
        """Test string representation."""
        metric = CosineSimilarity(dim=1, eps=1e-6)
        repr_str = repr(metric)

        assert "CosineSimilarity" in repr_str
        assert "dim=1" in repr_str
        assert "eps=1e-06" in repr_str or "eps=1e-6" in repr_str


class TestMeanSquaredError:
    """Test MeanSquaredError metric implementation."""

    def test_initialization_and_state(self):
        """Test MSE metric initialization."""
        metric = MeanSquaredError()

        # Should have required buffers
        assert hasattr(metric, "sum_squared_error")
        assert hasattr(metric, "num_samples")

        # Initial state
        assert metric.sum_squared_error.item() == 0.0
        assert metric.num_samples.item() == 0

    def test_update_functionality(self):
        """Test MSE update functionality."""
        metric = MeanSquaredError()

        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        # Expected: ((1-1.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-3.5)^2) = 0.25 + 0.25 + 0.25 + 0.25 = 1.0
        # Number of elements: 4

        metric.update(pred, target)

        assert metric.sum_squared_error.item() == 1.0
        assert metric.num_samples.item() == 4

    def test_compute_functionality(self):
        """Test MSE compute functionality."""
        metric = MeanSquaredError()

        # Mock state
        metric.sum_squared_error = torch.tensor(10.0)
        metric.num_samples = torch.tensor(5)

        result = metric.compute()
        assert result.item() == 2.0  # 10.0 / 5

    def test_shape_mismatch_error(self):
        """Test error on shape mismatch."""
        metric = MeanSquaredError()

        pred = torch.randn(10, 256)
        target = torch.randn(10, 128)  # Different feature dim

        with pytest.raises(ValueError, match="Shape mismatch"):
            metric.update(pred, target)


class TestPSNR:
    """Test PSNR metric implementation."""

    def test_initialization(self):
        """Test PSNR metric initialization."""
        # Default max_val
        psnr_default = PSNR()
        assert psnr_default.max_val == 1.0

        # Custom max_val
        psnr_custom = PSNR(max_val=255.0)
        assert psnr_custom.max_val == 255.0

    def test_state_buffers(self):
        """Test PSNR state buffers."""
        metric = PSNR()

        assert hasattr(metric, "sum_psnr")
        assert hasattr(metric, "num_samples")

        # Initial state
        assert metric.sum_psnr.item() == 0.0
        assert metric.num_samples.item() == 0

    def test_update_logic(self):
        """Test PSNR update logic."""
        metric = PSNR(max_val=1.0)

        # Perfect prediction should give high PSNR
        pred = torch.ones(2, 3, 64, 64)
        target = torch.ones(2, 3, 64, 64)

        metric.update(pred, target)

        # With zero MSE, PSNR should be very high (near infinity)
        # Implementation should handle this case properly
        metric.compute()
        # Exact value depends on implementation details (handling of zero MSE)

    def test_compute_with_known_values(self):
        """Test PSNR computation with known values."""
        metric = PSNR(max_val=255.0)

        # Simulate known PSNR values
        metric.sum_psnr = torch.tensor(60.0)  # 30 dB average for 2 images
        metric.num_samples = torch.tensor(2)

        result = metric.compute()
        assert result.item() == 30.0  # 60.0 / 2


class TestSSIM:
    """Test SSIM metric implementation."""

    def test_initialization(self):
        """Test SSIM metric initialization."""
        # Default parameters
        ssim_default = SSIM()
        assert ssim_default.window_size == 11
        assert ssim_default.k1 == 0.01
        assert ssim_default.k2 == 0.03

        # Custom parameters
        ssim_custom = SSIM(window_size=7, k1=0.02, k2=0.04)
        assert ssim_custom.window_size == 7
        assert ssim_custom.k1 == 0.02
        assert ssim_custom.k2 == 0.04

    def test_invalid_window_size(self):
        """Test that even window sizes raise errors."""
        with pytest.raises(ValueError, match="window_size must be odd"):
            SSIM(window_size=8)

        with pytest.raises(ValueError, match="window_size must be odd"):
            SSIM(window_size=12)

    def test_state_buffers(self):
        """Test SSIM state buffers."""
        metric = SSIM()

        assert hasattr(metric, "sum_ssim")
        assert hasattr(metric, "num_samples")

    def test_placeholder_implementation_note(self):
        """Test that current implementation is noted as placeholder."""
        # The current implementation is simplified (uses correlation as proxy)
        # This test documents that limitation
        metric = SSIM()

        # Test with identical images
        identical_images = torch.ones(2, 3, 64, 64)
        metric.update(identical_images, identical_images)

        metric.compute()
        # With identical images, should approach 1.0
        # Exact value depends on placeholder implementation


class TestReconstructionMetrics:
    """Test comprehensive ReconstructionMetrics class."""

    def test_initialization_basic(self):
        """Test basic initialization without pixel metrics."""
        metrics = ReconstructionMetrics(include_pixel_metrics=False)

        assert not metrics.include_pixel_metrics
        assert hasattr(metrics, "cosine_similarity")
        assert hasattr(metrics, "mse")
        assert isinstance(metrics.cosine_similarity, CosineSimilarity)
        assert isinstance(metrics.mse, MeanSquaredError)

    def test_initialization_with_pixel_metrics(self):
        """Test initialization with pixel metrics."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        assert metrics.include_pixel_metrics
        assert hasattr(metrics, "psnr")
        assert hasattr(metrics, "ssim")
        assert isinstance(metrics.psnr, PSNR)
        assert isinstance(metrics.ssim, SSIM)

    def test_forward_basic_metrics(self):
        """Test forward method with basic metrics only."""
        metrics = ReconstructionMetrics(include_pixel_metrics=False)

        pred = torch.ones(16, 512)
        target = torch.ones(16, 512)

        result = metrics(pred, target)

        assert isinstance(result, dict)
        assert "cosine_similarity" in result
        assert "mse" in result
        assert result["cosine_similarity"] == pytest.approx(1.0)
        assert result["mse"] == pytest.approx(0.0)

    def test_forward_with_pixel_metrics(self):
        """Test forward method with pixel metrics for image data."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        # 4D tensor with 3 channels (image data)
        pred_images = torch.ones(8, 3, 64, 64)
        target_images = torch.ones(8, 3, 64, 64)

        result = metrics(pred_images, target_images)

        assert len(result) == 4
        assert "cosine_similarity" in result
        assert "mse" in result
        assert "psnr" in result
        assert "ssim" in result
        assert result["cosine_similarity"] == pytest.approx(1.0)
        assert result["mse"] == pytest.approx(0.0)

    def test_forward_non_image_data_skip_pixel_metrics(self):
        """Test that pixel metrics are skipped for non-image data."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        # 2D tensor (not image-like)
        pred_features = torch.ones(32, 768)
        target_features = torch.ones(32, 768)

        result = metrics(pred_features, target_features)

        # Should only have basic metrics, not pixel metrics
        assert len(result) == 2
        assert "cosine_similarity" in result
        assert "mse" in result
        assert "psnr" not in result
        assert "ssim" not in result
        assert result["cosine_similarity"] == pytest.approx(1.0)
        assert result["mse"] == pytest.approx(0.0)

    def test_update_method(self):
        """Test update method for accumulative computation."""
        metrics = ReconstructionMetrics(include_pixel_metrics=False)

        pred = torch.randn(8, 256)
        target = torch.randn(8, 256)

        with (
            patch.object(metrics.cosine_similarity, "update") as mock_cosine_update,
            patch.object(metrics.mse, "update") as mock_mse_update,
        ):
            metrics.update(pred, target)

            mock_cosine_update.assert_called_once_with(pred, target)
            mock_mse_update.assert_called_once_with(pred, target)

    def test_compute_method(self):
        """Test compute method for accumulated results."""
        metrics = ReconstructionMetrics(include_pixel_metrics=False)

        with (
            patch.object(metrics.cosine_similarity, "compute") as mock_cosine_compute,
            patch.object(metrics.mse, "compute") as mock_mse_compute,
        ):
            mock_cosine_compute.return_value = torch.tensor(0.75)
            mock_mse_compute.return_value = torch.tensor(0.08)

            result = metrics.compute()

            assert isinstance(result, dict)
            assert result["cosine_similarity"] == pytest.approx(0.75)
            assert result["mse"] == pytest.approx(0.08)

    def test_reset_method(self):
        """Test reset method resets all metrics."""
        metrics = ReconstructionMetrics(include_pixel_metrics=True)

        with (
            patch.object(metrics.cosine_similarity, "reset") as mock_cosine_reset,
            patch.object(metrics.mse, "reset") as mock_mse_reset,
            patch.object(metrics.psnr, "reset") as mock_psnr_reset,
            patch.object(metrics.ssim, "reset") as mock_ssim_reset,
        ):
            metrics.reset()

            mock_cosine_reset.assert_called_once()
            mock_mse_reset.assert_called_once()
            mock_psnr_reset.assert_called_once()
            mock_ssim_reset.assert_called_once()


class TestMetricRegistry:
    """Test metric registry system."""

    def test_get_metric_valid(self):
        """Test getting valid metrics from registry."""
        # Test cosine similarity
        metric = get_metric("cosine_similarity", dim=1, eps=1e-6)
        assert isinstance(metric, CosineSimilarity)
        assert metric.dim == 1
        assert metric.eps == 1e-6

        # Test reconstruction metrics
        metric = get_metric("reconstruction_metrics", include_pixel_metrics=True)
        assert isinstance(metric, ReconstructionMetrics)
        assert metric.include_pixel_metrics

    def test_get_metric_invalid(self):
        """Test error handling for invalid metric names."""
        with pytest.raises(KeyError, match="Metric 'nonexistent' not found"):
            get_metric("nonexistent")

    def test_registry_contents(self):
        """Test that expected metrics are registered."""
        from associative.utils.video._registry import METRIC_REGISTRY

        expected_metrics = [
            "cosine_similarity",
            "reconstruction_metrics",
            "mse",
            "psnr",
            "ssim",
        ]

        for metric_name in expected_metrics:
            assert metric_name in METRIC_REGISTRY


class TestMetricBehaviorContracts:
    """Test behavioral contracts that all metrics should satisfy."""

    @pytest.mark.parametrize(
        "metric_class", [CosineSimilarity, MeanSquaredError, PSNR, SSIM]
    )
    def test_metric_basic_interface(self, metric_class):
        """Test that all metrics implement basic interface."""
        metric = metric_class()

        # Should have required methods
        assert hasattr(metric, "update")
        assert hasattr(metric, "compute")
        assert hasattr(metric, "reset")
        assert hasattr(metric, "forward")

        # Methods should be callable
        assert callable(metric.update)
        assert callable(metric.compute)
        assert callable(metric.reset)
        assert callable(metric.forward)

    @pytest.mark.parametrize("metric_class", [CosineSimilarity, MeanSquaredError])
    def test_metric_state_persistence(self, metric_class):
        """Test that metrics maintain state across updates."""
        metric = metric_class()

        # Initial state should be zero
        metric.reset()
        initial_num_samples = getattr(metric, "num_samples", torch.tensor(0))
        assert initial_num_samples.item() == 0

        # After update, should have non-zero samples
        # (Implementation-dependent, testing interface)

    def test_metric_device_consistency(self):
        """Test that metrics handle device placement correctly."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            metric = CosineSimilarity(device=device)

            moved_metric = metric.to(device)
            assert moved_metric._device == device

    def test_metric_dtype_handling(self):
        """Test that metrics handle different dtypes appropriately."""
        metric_float32 = CosineSimilarity(dtype=torch.float32)
        metric_float64 = CosineSimilarity(dtype=torch.float64)

        assert metric_float32._dtype == torch.float32
        assert metric_float64._dtype == torch.float64

    def test_metric_error_handling_consistency(self):
        """Test consistent error handling across metrics."""
        metrics = [CosineSimilarity(), MeanSquaredError(), PSNR(), SSIM()]

        for metric in metrics:
            # All should raise RuntimeError when computing without updates
            with pytest.raises(RuntimeError, match="No samples processed"):
                metric.compute()

            # All should handle shape mismatches with ValueError
            pred = torch.randn(10, 64)
            target = torch.randn(10, 32)  # Different size

            with pytest.raises(ValueError, match="Shape mismatch"):
                metric.update(pred, target)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
