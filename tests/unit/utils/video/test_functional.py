"""Comprehensive tests for video functional operations."""

from unittest.mock import MagicMock, patch

import pytest
import torch

import associative.utils.video.functional as video_functional


class TestVideoLoading:
    """Test video loading functional operations."""

    def test_load_video_parameter_validation(self):
        """Test parameter validation for load_video."""
        # Test invalid num_frames
        with pytest.raises(ValueError, match="num_frames must be positive"):
            video_functional.load_video("test.mp4", num_frames=0)

        with pytest.raises(ValueError, match="num_frames must be positive"):
            video_functional.load_video("test.mp4", num_frames=-5)

        # Test invalid resolution
        with pytest.raises(ValueError, match="resolution must be positive"):
            video_functional.load_video("test.mp4", num_frames=10, resolution=0)

        with pytest.raises(ValueError, match="resolution must be positive"):
            video_functional.load_video("test.mp4", num_frames=10, resolution=-224)

        # Test invalid sampling strategy
        with pytest.raises(ValueError, match="Invalid sampling_strategy"):
            video_functional.load_video(
                "test.mp4",
                num_frames=10,
                sampling_strategy="invalid",  # type: ignore[arg-type]
            )

    def test_load_video_file_errors(self):
        """Test file-related errors in load_video."""
        # Test nonexistent file
        with pytest.raises(FileNotFoundError, match="doesn't exist"):
            video_functional.load_video("nonexistent_video.mp4", num_frames=10)

    def test_load_video_output_contract(self):
        """Test expected output format of load_video."""
        # This test would run when implementation is available
        if hasattr(video_functional.load_video, "_implemented"):
            with patch("associative.utils.video.functional.VideoReader") as mock_reader:
                # Mock video reader
                mock_instance = MagicMock()
                mock_instance.__len__.return_value = 100
                mock_instance.__getitem__.return_value.asnumpy.return_value = (
                    torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
                )
                mock_reader.return_value = mock_instance

                frames = video_functional.load_video(
                    "test.mp4", num_frames=50, resolution=224
                )

                # Verify output contract
                assert frames.shape == (50, 3, 224, 224)
                assert frames.dtype == torch.float32
                assert frames.min() >= -1.0 and frames.max() <= 1.0  # Normalized


class TestMasking:
    """Test video masking operations."""

    def test_apply_mask_parameter_validation(self):
        """Test parameter validation for apply_mask."""
        frames = torch.randn(10, 3, 224, 224)

        # Test invalid mask_ratio
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            video_functional.apply_mask(frames, mask_ratio=-0.1)

        with pytest.raises(ValueError, match="mask_ratio must be in"):
            video_functional.apply_mask(frames, mask_ratio=1.5)

        # Test invalid mask_type
        with pytest.raises(ValueError, match="Unknown mask type"):
            video_functional.apply_mask(frames, mask_type="invalid")  # type: ignore[arg-type]

    def test_apply_mask_input_validation(self):
        """Test input tensor validation for apply_mask."""
        # Test wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            video_functional.apply_mask(torch.randn(10, 224, 224), mask_ratio=0.5)

        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            video_functional.apply_mask(torch.randn(10, 3, 224, 224, 5), mask_ratio=0.5)

    def test_apply_mask_output_contract(self):
        """Test expected output format of apply_mask."""
        frames = torch.randn(8, 3, 224, 224)

        if hasattr(video_functional.apply_mask, "_implemented"):
            masked_frames, mask = video_functional.apply_mask(
                frames, mask_ratio=0.5, mask_type="bottom_half"
            )

            # Verify output shapes
            assert masked_frames.shape == frames.shape
            assert mask.shape == (8, 224, 224)
            assert mask.dtype == torch.bool

            # Test different mask types preserve shape
            for mask_type in ["bottom_half", "random", "none"]:
                masked, m = video_functional.apply_mask(
                    frames,
                    mask_ratio=0.3,
                    mask_type=mask_type,  # type: ignore[arg-type]
                )
                assert masked.shape == frames.shape
                assert m.shape == (8, 224, 224)
                assert m.dtype == torch.bool


class TestNoise:
    """Test noise addition operations."""

    def test_add_noise_parameter_validation(self):
        """Test parameter validation for add_noise."""
        frames = torch.randn(10, 512)

        # Test invalid noise_std
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            video_functional.add_noise(frames, noise_std=-0.1)

        # Test invalid noise_type
        with pytest.raises(ValueError, match="Invalid noise_type"):
            video_functional.add_noise(frames, noise_type="invalid")  # type: ignore[arg-type]

    def test_add_noise_output_contract(self):
        """Test expected output format of add_noise."""
        input_tensor = torch.randn(16, 768)

        if hasattr(video_functional.add_noise, "_implemented"):
            noise_std = 0.1
            noisy = video_functional.add_noise(
                input_tensor, noise_std=noise_std, noise_type="gaussian"
            )

            # Verify output contract
            assert noisy.shape == input_tensor.shape
            assert noisy.dtype == input_tensor.dtype

            # With non-zero std, output should differ from input
            if noise_std > 0:
                assert not torch.allclose(noisy, input_tensor)

            # Test different noise types
            for noise_type in ["gaussian", "uniform"]:
                noisy = video_functional.add_noise(
                    input_tensor,
                    noise_std=0.05,
                    noise_type=noise_type,  # type: ignore[arg-type]
                )
                assert noisy.shape == input_tensor.shape


class TestSampling:
    """Test frame sampling operations."""

    def test_uniform_sample_indices_parameter_validation(self):
        """Test parameter validation for uniform_sample_indices."""
        # Test invalid parameters
        with pytest.raises(ValueError, match="num_frames must be positive"):
            video_functional.uniform_sample_indices(100, num_frames=0)

        with pytest.raises(ValueError, match="num_frames must be positive"):
            video_functional.uniform_sample_indices(100, num_frames=-10)

        with pytest.raises(ValueError, match="negative values"):
            video_functional.uniform_sample_indices(100, num_frames=10, start_frame=-1)

        with pytest.raises(ValueError, match="num_frames > total_frames"):
            video_functional.uniform_sample_indices(50, num_frames=100)

    def test_uniform_sample_indices_output_contract(self):
        """Test expected output format of uniform_sample_indices."""
        if hasattr(video_functional.uniform_sample_indices, "_implemented"):
            indices = video_functional.uniform_sample_indices(1000, 100, start_frame=0)

            # Verify output contract
            assert indices.shape == (100,)
            assert indices.dtype == torch.long
            assert torch.all(indices >= 0)
            assert torch.all(indices < 1000)

            # Should be roughly uniformly spaced
            diffs = indices[1:] - indices[:-1]
            assert torch.all(diffs > 0)  # Monotonically increasing

            # Test with start_frame
            indices_offset = video_functional.uniform_sample_indices(
                1000, 50, start_frame=100
            )
            assert torch.all(indices_offset >= 100)


class TestResizing:
    """Test frame resizing operations."""

    def test_resize_frames_parameter_validation(self):
        """Test parameter validation for resize_frames."""
        frames = torch.randn(10, 3, 128, 128)

        # Test invalid size
        with pytest.raises(ValueError, match="non-positive values"):
            video_functional.resize_frames(frames, size=0)

        with pytest.raises(ValueError, match="non-positive values"):
            video_functional.resize_frames(frames, size=(224, -1))

        # Test invalid interpolation
        with pytest.raises(ValueError, match="unsupported interpolation"):
            video_functional.resize_frames(frames, size=224, interpolation="invalid")

    def test_resize_frames_input_validation(self):
        """Test input validation for resize_frames."""
        # Test wrong dimensions
        with pytest.raises(RuntimeError, match="wrong number of dimensions"):
            video_functional.resize_frames(torch.randn(10, 128, 128), size=224)

    def test_resize_frames_output_contract(self):
        """Test expected output format of resize_frames."""
        frames = torch.randn(5, 3, 128, 128)

        if hasattr(video_functional.resize_frames, "_implemented"):
            # Test square resize
            resized = video_functional.resize_frames(frames, size=224)
            assert resized.shape == (5, 3, 224, 224)

            # Test rectangular resize
            resized_rect = video_functional.resize_frames(frames, size=(256, 224))
            assert resized_rect.shape == (5, 3, 256, 224)

            # Test different interpolation modes
            for interp in ["bilinear", "nearest", "bicubic"]:
                resized = video_functional.resize_frames(
                    frames, size=224, interpolation=interp
                )
                assert resized.shape == (5, 3, 224, 224)


class TestNormalization:
    """Test frame normalization operations."""

    def test_normalize_frames_parameter_validation(self):
        """Test parameter validation for normalize_frames."""
        frames = torch.randn(10, 3, 224, 224)

        # Test invalid mean/std length
        with pytest.raises(ValueError, match="don't match number of channels"):
            video_functional.normalize_frames(
                frames,
                mean=(0.5, 0.5),  # type: ignore[arg-type]
                std=(0.5, 0.5, 0.5),  # type: ignore[arg-type]
            )

        with pytest.raises(ValueError, match="don't match number of channels"):
            video_functional.normalize_frames(
                frames,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5),  # type: ignore[arg-type]
            )

        # Test zero std
        with pytest.raises(RuntimeError, match="std value is zero"):
            video_functional.normalize_frames(
                frames, mean=(0.5, 0.5, 0.5), std=(0.5, 0.0, 0.5)
            )

    def test_normalize_frames_output_contract(self):
        """Test expected output format of normalize_frames."""
        frames = torch.rand(8, 3, 224, 224)  # [0, 1] range

        if hasattr(video_functional.normalize_frames, "_implemented"):
            normalized = video_functional.normalize_frames(
                frames, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            )

            # Verify output contract
            assert normalized.shape == frames.shape
            assert normalized.dtype == frames.dtype

            # Should be roughly in [-1, 1] range for this normalization
            assert normalized.min() >= -1.1  # Small tolerance for floating point
            assert normalized.max() <= 1.1


class TestSimilarity:
    """Test similarity computation operations."""

    def test_compute_cosine_similarity_parameter_validation(self):
        """Test parameter validation for compute_cosine_similarity."""
        pred = torch.randn(32, 512)

        # Test shape mismatch
        with pytest.raises(ValueError, match="different shapes"):
            video_functional.compute_cosine_similarity(pred, torch.randn(32, 256))

        with pytest.raises(ValueError, match="different shapes"):
            video_functional.compute_cosine_similarity(pred, torch.randn(16, 512))

    def test_compute_cosine_similarity_edge_cases(self):
        """Test edge cases for compute_cosine_similarity."""
        # Test empty tensors
        with pytest.raises(RuntimeError, match="empty"):
            video_functional.compute_cosine_similarity(
                torch.empty(0, 512), torch.empty(0, 512)
            )

        # Test zero vectors (should handle gracefully with eps)
        if hasattr(video_functional.compute_cosine_similarity, "_implemented"):
            zero_pred = torch.zeros(2, 10)
            zero_target = torch.zeros(2, 10)
            similarity = video_functional.compute_cosine_similarity(
                zero_pred, zero_target, eps=1e-8
            )
            assert torch.isfinite(similarity).all()  # Should not be NaN/inf

    def test_compute_cosine_similarity_output_contract(self):
        """Test expected output format of compute_cosine_similarity."""
        pred = torch.randn(16, 768)
        target = torch.randn(16, 768)

        if hasattr(video_functional.compute_cosine_similarity, "_implemented"):
            similarity = video_functional.compute_cosine_similarity(
                pred, target, dim=-1
            )

            # Verify output contract
            assert similarity.shape == (16,)
            assert similarity.dtype == pred.dtype
            assert torch.all(similarity >= -1.0)
            assert torch.all(similarity <= 1.0)

            # Test different dimensions
            similarity_dim0 = video_functional.compute_cosine_similarity(
                pred, target, dim=0
            )
            assert similarity_dim0.shape == (768,)


class TestUtilityFunctions:
    """Test utility tensor operations."""

    def test_stack_video_frames_validation(self):
        """Test validation for stack_video_frames."""
        # Test empty list
        with pytest.raises(RuntimeError, match="empty"):
            video_functional.stack_video_frames([])

        # Test inconsistent shapes
        frames1 = torch.randn(10, 3, 224, 224)
        frames2 = torch.randn(10, 3, 256, 256)  # Different size

        with pytest.raises(ValueError, match="inconsistent shapes"):
            video_functional.stack_video_frames([frames1, frames2])

    def test_stack_video_frames_output_contract(self):
        """Test expected output format of stack_video_frames."""
        if hasattr(video_functional.stack_video_frames, "_implemented"):
            frames1 = torch.randn(50, 3, 224, 224)
            frames2 = torch.randn(50, 3, 224, 224)
            frames3 = torch.randn(50, 3, 224, 224)

            stacked = video_functional.stack_video_frames([frames1, frames2, frames3])

            # Verify output contract
            assert stacked.shape == (3, 50, 3, 224, 224)  # (B, N, C, H, W)
            assert stacked.dtype == frames1.dtype

    def test_flatten_video_frames_output_contract(self):
        """Test expected output format of flatten_video_frames."""
        frames = torch.randn(100, 3, 224, 224)

        if hasattr(video_functional.flatten_video_frames, "_implemented"):
            flattened = video_functional.flatten_video_frames(frames)

            # Verify output contract
            expected_dim = 3 * 224 * 224
            assert flattened.shape == (100, expected_dim)
            assert flattened.dtype == frames.dtype

    def test_reshape_to_frames_parameter_validation(self):
        """Test parameter validation for reshape_to_frames."""
        # Test incorrect flattened dimension
        flattened = torch.randn(100, 1000)  # Wrong dimension

        with pytest.raises(ValueError, match="dimension doesn't match"):
            video_functional.reshape_to_frames(
                flattened, channels=3, height=224, width=224
            )

    def test_reshape_to_frames_output_contract(self):
        """Test expected output format of reshape_to_frames."""
        expected_dim = 3 * 224 * 224
        flattened = torch.randn(50, expected_dim)

        if hasattr(video_functional.reshape_to_frames, "_implemented"):
            frames = video_functional.reshape_to_frames(
                flattened, channels=3, height=224, width=224
            )

            # Verify output contract
            assert frames.shape == (50, 3, 224, 224)
            assert frames.dtype == flattened.dtype


class TestBatchOperations:
    """Test operations that work with batches."""

    def test_batch_consistency(self):
        """Test that operations handle batches consistently."""
        # Single frame operations should work with batches
        if hasattr(video_functional.resize_frames, "_implemented"):
            single_frame = torch.randn(1, 3, 128, 128)
            batch_frames = torch.randn(10, 3, 128, 128)

            single_resized = video_functional.resize_frames(single_frame, size=224)
            batch_resized = video_functional.resize_frames(batch_frames, size=224)

            assert single_resized.shape == (1, 3, 224, 224)
            assert batch_resized.shape == (10, 3, 224, 224)

    def test_device_consistency(self):
        """Test that operations preserve device placement."""
        if torch.cuda.is_available():
            frames_cpu = torch.randn(5, 3, 224, 224)
            frames_cuda = frames_cpu.cuda()

            if hasattr(video_functional.add_noise, "_implemented"):
                noisy_cpu = video_functional.add_noise(frames_cpu, noise_std=0.1)
                noisy_cuda = video_functional.add_noise(frames_cuda, noise_std=0.1)

                assert noisy_cpu.device == frames_cpu.device
                assert noisy_cuda.device == frames_cuda.device

    def test_dtype_consistency(self):
        """Test that operations preserve data types appropriately."""
        frames_float32 = torch.randn(5, 3, 224, 224, dtype=torch.float32)
        frames_float64 = torch.randn(5, 3, 224, 224, dtype=torch.float64)

        if hasattr(video_functional.normalize_frames, "_implemented"):
            norm_float32 = video_functional.normalize_frames(frames_float32)
            norm_float64 = video_functional.normalize_frames(frames_float64)

            assert norm_float32.dtype == torch.float32
            assert norm_float64.dtype == torch.float64


class TestErrorHandling:
    """Test comprehensive error handling."""

    def test_graceful_degradation(self):
        """Test that functions fail gracefully with informative errors."""
        # All parameter validation should use ValueError with descriptive messages
        with pytest.raises(ValueError):
            video_functional.load_video(
                "test.mp4", num_frames=-1
            )  # Specific error caught above

        # Runtime errors should use RuntimeError
        with pytest.raises(RuntimeError):
            video_functional.apply_mask(
                torch.randn(10, 224, 224), mask_ratio=0.5
            )  # Wrong dims

    def test_edge_case_handling(self):
        """Test handling of edge cases."""
        # Very small tensors
        tiny_frames = torch.randn(1, 3, 1, 1)

        if hasattr(video_functional.resize_frames, "_implemented"):
            # Should handle tiny images
            resized = video_functional.resize_frames(tiny_frames, size=224)
            assert resized.shape == (1, 3, 224, 224)

        # Very large mask ratios
        frames = torch.randn(5, 3, 224, 224)
        if hasattr(video_functional.apply_mask, "_implemented"):
            # Should handle edge ratios
            masked_all, mask_all = video_functional.apply_mask(frames, mask_ratio=0.99)
            assert masked_all.shape == frames.shape
            assert mask_all.sum() > 0  # Most pixels masked


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_cosine_similarity_stability(self):
        """Test numerical stability of cosine similarity."""
        if hasattr(video_functional.compute_cosine_similarity, "_implemented"):
            # Very small values (near machine epsilon)
            tiny_pred = torch.randn(10, 100) * 1e-7
            tiny_target = torch.randn(10, 100) * 1e-7

            similarity = video_functional.compute_cosine_similarity(
                tiny_pred, tiny_target, eps=1e-8
            )
            assert torch.isfinite(similarity).all()  # Should not overflow/underflow

            # Very large values
            large_pred = torch.randn(10, 100) * 1e6
            large_target = torch.randn(10, 100) * 1e6

            similarity_large = video_functional.compute_cosine_similarity(
                large_pred, large_target
            )
            assert torch.isfinite(similarity_large).all()

    def test_normalization_stability(self):
        """Test numerical stability of normalization."""
        if hasattr(video_functional.normalize_frames, "_implemented"):
            # Extreme values
            extreme_frames = torch.cat(
                [
                    torch.ones(2, 3, 224, 224) * 1e6,  # Very large
                    torch.ones(2, 3, 224, 224) * 1e-6,  # Very small
                ]
            )

            normalized = video_functional.normalize_frames(
                extreme_frames, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
            )

            assert torch.isfinite(normalized).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
