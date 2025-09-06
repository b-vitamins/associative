"""Tests for video functional operations.

Tests focus on behavior and contracts, not implementation details.
Each function should handle edge cases gracefully and produce correct outputs.
"""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from associative.utils.video import functional as F


class TestVideoLoading:
    """Test video loading operations."""

    def test_load_video_validates_arguments(self):
        """Test that load_video validates its arguments properly."""
        # Invalid num_frames
        with pytest.raises(ValueError, match="num_frames must be positive"):
            F.load_video("test.mp4", num_frames=0)

        with pytest.raises(ValueError, match="num_frames must be positive"):
            F.load_video("test.mp4", num_frames=-5)

        # Invalid resolution
        with pytest.raises(ValueError, match="resolution must be positive"):
            F.load_video("test.mp4", num_frames=10, resolution=0)

        # Invalid sampling strategy
        with pytest.raises(ValueError, match="Invalid sampling_strategy"):
            F.load_video("test.mp4", num_frames=10, sampling_strategy="invalid")  # type: ignore

        # Nonexistent file
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            F.load_video("nonexistent.mp4", num_frames=10)

    @patch("associative.utils.video.functional.VideoReader")
    def test_load_video_uniform_sampling(self, mock_reader):
        """Test video loading with uniform sampling."""
        # Create a temporary video file path
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            # Mock video reader
            mock_instance = MagicMock()
            mock_instance.__len__.return_value = 100
            # Create mock frames
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_instance.__getitem__.return_value.asnumpy.return_value = mock_frame
            mock_reader.return_value = mock_instance

            frames = F.load_video(tmp.name, num_frames=10, resolution=224)

            # Check output shape and properties
            assert frames.shape == (10, 3, 224, 224)
            assert frames.dtype == torch.float32
            assert -1.5 <= frames.min() <= 1.5  # Normalized range
            assert -1.5 <= frames.max() <= 1.5

    @patch("associative.utils.video.functional.VideoReader")
    def test_load_video_different_strategies(self, mock_reader):
        """Test different sampling strategies."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            mock_instance = MagicMock()
            mock_instance.__len__.return_value = 50
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_instance.__getitem__.return_value.asnumpy.return_value = mock_frame
            mock_reader.return_value = mock_instance

            # Test each strategy
            for strategy in ["uniform", "random", "sequential"]:
                frames = F.load_video(
                    tmp.name,
                    num_frames=5,
                    resolution=128,
                    sampling_strategy=strategy,  # type: ignore
                )
                assert frames.shape == (5, 3, 128, 128)


class TestMasking:
    """Test masking operations."""

    def test_apply_mask_validates_inputs(self):
        """Test input validation for apply_mask."""
        frames = torch.randn(10, 3, 224, 224)

        # Invalid mask_ratio
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            F.apply_mask(frames, mask_ratio=-0.1)

        with pytest.raises(ValueError, match="mask_ratio must be in"):
            F.apply_mask(frames, mask_ratio=1.5)

        # Invalid mask_type
        with pytest.raises(ValueError, match="Invalid mask_type"):
            F.apply_mask(frames, mask_type="invalid")  # type: ignore

        # Wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            F.apply_mask(torch.randn(10, 224, 224), mask_ratio=0.5)

    def test_apply_mask_bottom_half(self):
        """Test bottom half masking."""
        frames = torch.ones(4, 3, 8, 8)
        masked_frames, mask = F.apply_mask(
            frames, mask_ratio=0.5, mask_type="bottom_half"
        )

        assert masked_frames.shape == frames.shape
        assert mask.shape == (4, 8, 8)
        assert mask.dtype == torch.bool

        # Bottom half should be masked
        assert mask[:, 4:, :].all()  # Bottom 4 rows masked
        assert not mask[:, :4, :].any()  # Top 4 rows not masked

        # Masked regions should be zero
        assert (masked_frames[:, :, 4:, :] == 0).all()

    def test_apply_mask_random(self):
        """Test random masking."""
        torch.manual_seed(42)
        frames = torch.ones(10, 3, 32, 32)
        masked_frames, mask = F.apply_mask(frames, mask_ratio=0.3, mask_type="random")

        assert masked_frames.shape == frames.shape
        assert mask.shape == (10, 32, 32)

        # Approximately 30% should be masked
        mask_ratio_actual = mask.float().mean().item()
        assert 0.25 <= mask_ratio_actual <= 0.35

    def test_apply_mask_none(self):
        """Test no masking."""
        frames = torch.randn(5, 3, 16, 16)
        masked_frames, mask = F.apply_mask(frames, mask_ratio=0.5, mask_type="none")

        assert torch.allclose(masked_frames, frames)
        assert not mask.any()  # No masking applied


class TestNoise:
    """Test noise addition operations."""

    def test_add_noise_validates_inputs(self):
        """Test input validation for add_noise."""
        frames = torch.randn(10, 512)

        # Invalid noise_std
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            F.add_noise(frames, noise_std=-0.1)

        # Invalid noise_type
        with pytest.raises(ValueError, match="Invalid noise_type"):
            F.add_noise(frames, noise_type="invalid")  # type: ignore

    def test_add_gaussian_noise(self):
        """Test Gaussian noise addition."""
        torch.manual_seed(42)
        frames = torch.zeros(100, 512)
        noisy = F.add_noise(frames, noise_std=0.1, noise_type="gaussian")

        assert noisy.shape == frames.shape
        assert not torch.allclose(noisy, frames)

        # Check noise statistics
        noise = noisy - frames
        assert abs(noise.mean().item()) < 0.02  # Should be near 0
        assert 0.08 <= noise.std().item() <= 0.12  # Should be near 0.1

    def test_add_uniform_noise(self):
        """Test uniform noise addition."""
        torch.manual_seed(42)
        frames = torch.zeros(100, 512)
        noisy = F.add_noise(frames, noise_std=0.1, noise_type="uniform")

        assert noisy.shape == frames.shape
        assert not torch.allclose(noisy, frames)

        # Check noise is in expected range
        noise = noisy - frames
        assert noise.min() >= -0.15
        assert noise.max() <= 0.15

    def test_no_noise_with_zero_std(self):
        """Test that zero std adds no noise."""
        frames = torch.randn(10, 256)
        noisy = F.add_noise(frames, noise_std=0.0)
        assert torch.allclose(noisy, frames)


class TestSampling:
    """Test frame sampling operations."""

    def test_uniform_sample_indices_validates_inputs(self):
        """Test input validation for uniform_sample_indices."""
        # Invalid num_frames
        with pytest.raises(ValueError, match="num_frames must be positive"):
            F.uniform_sample_indices(100, num_frames=0)

        # Invalid start_frame
        with pytest.raises(ValueError, match="must be non-negative"):
            F.uniform_sample_indices(100, num_frames=10, start_frame=-1)

        # Too many frames requested
        with pytest.raises(ValueError, match="Cannot sample"):
            F.uniform_sample_indices(50, num_frames=100)

    def test_uniform_sample_indices_basic(self):
        """Test basic uniform sampling."""
        indices = F.uniform_sample_indices(100, num_frames=10)

        assert indices.shape == (10,)
        assert indices.dtype == torch.long
        assert indices[0] >= 0
        assert indices[-1] <= 99

        # Should be monotonically increasing
        assert (indices[1:] >= indices[:-1]).all()

    def test_uniform_sample_indices_with_offset(self):
        """Test uniform sampling with start frame."""
        indices = F.uniform_sample_indices(200, num_frames=20, start_frame=50)

        assert indices.shape == (20,)
        assert indices[0] >= 50
        assert indices[-1] <= 199

    def test_uniform_sample_edge_cases(self):
        """Test edge cases for uniform sampling."""
        # Sample all frames
        indices = F.uniform_sample_indices(10, num_frames=10)
        assert indices.shape == (10,)

        # Sample single frame
        indices = F.uniform_sample_indices(100, num_frames=1)
        assert indices.shape == (1,)


class TestResizing:
    """Test frame resizing operations."""

    def test_resize_frames_validates_inputs(self):
        """Test input validation for resize_frames."""
        frames = torch.randn(10, 3, 128, 128)

        # Invalid size
        with pytest.raises(ValueError, match="Size must be positive"):
            F.resize_frames(frames, size=0)

        with pytest.raises(ValueError, match="Size must be positive"):
            F.resize_frames(frames, size=(224, -1))

        # Invalid interpolation
        with pytest.raises(ValueError, match="Unsupported interpolation"):
            F.resize_frames(frames, size=224, interpolation="invalid")

        # Wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            F.resize_frames(torch.randn(10, 128, 128), size=224)

    def test_resize_frames_square(self):
        """Test square resizing."""
        frames = torch.randn(5, 3, 128, 128)
        resized = F.resize_frames(frames, size=224)

        assert resized.shape == (5, 3, 224, 224)
        assert resized.dtype == frames.dtype

    def test_resize_frames_rectangular(self):
        """Test rectangular resizing."""
        frames = torch.randn(3, 3, 128, 128)
        resized = F.resize_frames(frames, size=(256, 192))

        assert resized.shape == (3, 3, 256, 192)

    def test_resize_frames_interpolation_modes(self):
        """Test different interpolation modes."""
        frames = torch.randn(2, 3, 64, 64)

        for mode in ["bilinear", "nearest", "bicubic"]:
            resized = F.resize_frames(frames, size=128, interpolation=mode)
            assert resized.shape == (2, 3, 128, 128)

    def test_resize_preserves_content_structure(self):
        """Test that resizing preserves content structure."""
        # Create a simple pattern
        frames = torch.zeros(1, 3, 8, 8)
        frames[:, :, :4, :4] = 1.0  # Top-left quadrant

        resized = F.resize_frames(frames, size=16)

        # Top-left should still be brighter
        top_left = resized[:, :, :8, :8].mean()
        bottom_right = resized[:, :, 8:, 8:].mean()
        assert top_left > bottom_right


class TestNormalization:
    """Test normalization operations."""

    def test_normalize_frames_validates_inputs(self):
        """Test input validation for normalize_frames."""
        frames = torch.randn(10, 3, 224, 224)

        # Wrong number of mean/std values
        with pytest.raises(ValueError, match="Mean and std must have 3 values"):
            F.normalize_frames(frames, mean=(0.5, 0.5), std=(0.5, 0.5, 0.5))  # type: ignore

        # Zero std
        with pytest.raises(RuntimeError, match="Standard deviation cannot be zero"):
            F.normalize_frames(frames, mean=(0.5, 0.5, 0.5), std=(0.5, 0.0, 0.5))

    def test_normalize_frames_basic(self):
        """Test basic normalization."""
        frames = torch.ones(4, 3, 32, 32) * 0.5
        normalized = F.normalize_frames(
            frames, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        assert normalized.shape == frames.shape
        assert torch.allclose(normalized, torch.zeros_like(normalized))

    def test_normalize_frames_inverts_correctly(self):
        """Test that normalization can be inverted."""
        frames = torch.rand(2, 3, 16, 16)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        normalized = F.normalize_frames(frames, mean=mean, std=std)

        # Manually invert
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        denormalized = normalized * std_t + mean_t

        assert torch.allclose(denormalized, frames, atol=1e-6)


class TestSimilarity:
    """Test similarity computation."""

    def test_cosine_similarity_validates_inputs(self):
        """Test input validation for cosine similarity."""
        pred = torch.randn(32, 512)

        # Shape mismatch
        with pytest.raises(ValueError, match="must have same shape"):
            F.compute_cosine_similarity(pred, torch.randn(32, 256))

        # Empty tensors
        with pytest.raises(
            RuntimeError, match="Cannot compute similarity on empty tensors"
        ):
            F.compute_cosine_similarity(torch.empty(0, 512), torch.empty(0, 512))

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = torch.randn(10, 256)
        similarity = F.compute_cosine_similarity(vec, vec, dim=-1)

        assert similarity.shape == (10,)
        assert torch.allclose(similarity, torch.ones(10), atol=1e-6)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        vec2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        similarity = F.compute_cosine_similarity(vec1, vec2, dim=-1)

        assert torch.allclose(similarity, torch.zeros(2), atol=1e-6)

    def test_cosine_similarity_numerical_stability(self):
        """Test numerical stability with zero vectors."""
        zero_vec = torch.zeros(5, 100)
        nonzero_vec = torch.randn(5, 100)

        # Should handle zero vectors gracefully
        similarity = F.compute_cosine_similarity(zero_vec, nonzero_vec, eps=1e-8)
        assert torch.isfinite(similarity).all()


class TestUtilityFunctions:
    """Test utility tensor operations."""

    def test_stack_video_frames(self):
        """Test stacking video frames."""
        frames1 = torch.randn(10, 3, 32, 32)
        frames2 = torch.randn(10, 3, 32, 32)
        frames3 = torch.randn(10, 3, 32, 32)

        stacked = F.stack_video_frames([frames1, frames2, frames3])

        assert stacked.shape == (3, 10, 3, 32, 32)
        assert torch.allclose(stacked[0], frames1)
        assert torch.allclose(stacked[1], frames2)
        assert torch.allclose(stacked[2], frames3)

    def test_stack_video_frames_validates_inputs(self):
        """Test input validation for stack_video_frames."""
        # Empty list
        with pytest.raises(RuntimeError, match="Cannot stack empty list"):
            F.stack_video_frames([])

        # Inconsistent shapes
        frames1 = torch.randn(10, 3, 32, 32)
        frames2 = torch.randn(10, 3, 64, 64)
        with pytest.raises(ValueError, match="All frames must have same shape"):
            F.stack_video_frames([frames1, frames2])

    def test_flatten_video_frames(self):
        """Test flattening video frames."""
        frames = torch.randn(5, 3, 32, 32)
        flattened = F.flatten_video_frames(frames)

        assert flattened.shape == (5, 3 * 32 * 32)
        assert flattened.shape[0] == frames.shape[0]

    def test_reshape_to_frames(self):
        """Test reshaping flattened vectors to frames."""
        flattened = torch.randn(8, 3 * 64 * 64)
        frames = F.reshape_to_frames(flattened, channels=3, height=64, width=64)

        assert frames.shape == (8, 3, 64, 64)

    def test_reshape_to_frames_validates_dimensions(self):
        """Test dimension validation for reshape_to_frames."""
        flattened = torch.randn(10, 1000)  # Wrong dimension

        with pytest.raises(ValueError, match="Feature dimension.*doesn't match"):
            F.reshape_to_frames(flattened, channels=3, height=32, width=32)

    def test_flatten_reshape_roundtrip(self):
        """Test that flatten and reshape are inverses."""
        frames = torch.randn(7, 3, 28, 28)
        flattened = F.flatten_video_frames(frames)
        reconstructed = F.reshape_to_frames(flattened, 3, 28, 28)

        assert torch.allclose(reconstructed, frames)


class TestIntegration:
    """Test combinations of functional operations."""

    def test_resize_and_normalize_pipeline(self):
        """Test typical preprocessing pipeline."""
        frames = torch.rand(10, 3, 128, 128)

        # Resize then normalize
        resized = F.resize_frames(frames, size=224)
        normalized = F.normalize_frames(
            resized, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        assert normalized.shape == (10, 3, 224, 224)
        assert -2 <= normalized.min() <= 2  # Roughly normalized range
        assert -2 <= normalized.max() <= 2

    def test_mask_and_noise_pipeline(self):
        """Test masking followed by noise addition."""
        frames = torch.ones(5, 3, 16, 16)

        # Apply mask then add noise
        masked, mask = F.apply_mask(frames, mask_ratio=0.5, mask_type="bottom_half")
        noisy = F.add_noise(masked, noise_std=0.1)

        assert noisy.shape == frames.shape
        # Top half should be noisy but around 1
        top_half_mean = noisy[:, :, :8, :].mean().item()
        assert 0.9 <= top_half_mean <= 1.1

        # Bottom half should be noisy but around 0
        bottom_half_mean = noisy[:, :, 8:, :].mean().item()
        assert -0.1 <= bottom_half_mean <= 0.1

    def test_device_consistency(self):
        """Test that operations preserve device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        frames_cpu = torch.randn(3, 3, 32, 32)
        frames_gpu = frames_cpu.cuda()

        # Test various operations
        resized_cpu = F.resize_frames(frames_cpu, size=64)
        resized_gpu = F.resize_frames(frames_gpu, size=64)
        assert resized_cpu.device == frames_cpu.device
        assert resized_gpu.device == frames_gpu.device

        noisy_cpu = F.add_noise(frames_cpu, noise_std=0.1)
        noisy_gpu = F.add_noise(frames_gpu, noise_std=0.1)
        assert noisy_cpu.device == frames_cpu.device
        assert noisy_gpu.device == frames_gpu.device

    def test_dtype_preservation(self):
        """Test that operations preserve data types appropriately."""
        frames_f32 = torch.randn(2, 3, 16, 16, dtype=torch.float32)
        frames_f64 = torch.randn(2, 3, 16, 16, dtype=torch.float64)

        # Most operations should preserve dtype
        resized_f32 = F.resize_frames(frames_f32, size=32)
        resized_f64 = F.resize_frames(frames_f64, size=32)
        assert resized_f32.dtype == torch.float32
        assert resized_f64.dtype == torch.float64

        normalized_f32 = F.normalize_frames(frames_f32)
        normalized_f64 = F.normalize_frames(frames_f64)
        assert normalized_f32.dtype == torch.float32
        assert normalized_f64.dtype == torch.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
