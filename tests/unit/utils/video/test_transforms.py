"""Tests for video transforms.

Focus on behavioral testing - verifying that transforms actually work correctly,
not just that they call the right functions. Each test should validate the
actual transformation behavior and output correctness.
"""

import tempfile

import numpy as np
import pytest
import torch
from torch import Tensor

from associative.utils.video.transforms import (
    AddNoise,
    ApplyMask,
    Compose,
    Lambda,
    LoadVideo,
    Normalize,
    Resize,
    ToTensor,
    UniformSample,
)


class TestCompose:
    """Test transform composition behavior."""

    def test_single_transform(self):
        """Test compose with single transform."""
        transform = Compose([Resize(128)])
        frames = torch.randn(10, 3, 256, 256)

        result = transform(frames)

        assert result.shape == (10, 3, 128, 128)
        assert result.dtype == frames.dtype

    def test_multiple_transforms_in_sequence(self):
        """Test that transforms are applied in correct order."""
        # Create a specific sequence where order matters
        transform = Compose(
            [
                UniformSample(num_frames=5),  # First reduce frames
                Resize(64),  # Then resize
                Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                ),  # Finally normalize
            ]
        )

        # Create frames with known values
        frames = torch.ones(20, 3, 128, 128) * 0.5
        result = transform(frames)

        # Check each step worked
        assert result.shape == (5, 3, 64, 64)  # Sampled and resized
        assert torch.allclose(
            result, torch.zeros_like(result), atol=1e-6
        )  # Normalized to 0

    def test_transform_with_tuple_return(self):
        """Test compose handles transforms that return tuples."""
        transform = Compose(
            [
                ApplyMask(mask_ratio=0.5, return_mask=True),  # Returns (frames, mask)
                Resize(32),  # Should only operate on frames
            ]
        )

        frames = torch.ones(4, 3, 64, 64)
        result = transform(frames)

        # Compose should extract tensor from tuple and pass it along
        assert isinstance(result, Tensor)
        assert result.shape == (4, 3, 32, 32)

    def test_empty_compose(self):
        """Test that empty compose list raises error."""
        with pytest.raises(TypeError, match="All transforms must be callable"):
            Compose([])

    def test_invalid_transform_in_list(self):
        """Test that non-callable items raise error."""
        with pytest.raises(TypeError, match="All transforms must be callable"):
            Compose([Resize(224), "not_a_transform", Normalize()])  # type: ignore[list-item]

    def test_repr(self):
        """Test string representation."""
        transform = Compose([UniformSample(num_frames=100), Resize(224)])

        repr_str = repr(transform)
        assert "Compose" in repr_str
        assert "UniformSample" in repr_str
        assert "Resize" in repr_str


class TestLoadVideo:
    """Test video loading transform behavior."""

    def test_initialization(self):
        """Test proper initialization."""
        loader = LoadVideo(num_frames=100, resolution=256, sampling_strategy="random")

        assert loader.num_frames == 100
        assert loader.resolution == 256
        assert loader.sampling_strategy == "random"

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="num_frames must be positive"):
            LoadVideo(num_frames=0)

        with pytest.raises(ValueError, match="resolution must be positive"):
            LoadVideo(num_frames=10, resolution=-1)

    def test_tensor_passthrough(self):
        """Test that tensor input is passed through unchanged."""
        loader = LoadVideo(num_frames=50, resolution=224)
        input_tensor = torch.randn(50, 3, 224, 224)

        result = loader(input_tensor)

        assert torch.equal(result, input_tensor)

    def test_invalid_input_type(self):
        """Test error on invalid input type."""
        loader = LoadVideo(num_frames=10)

        with pytest.raises(TypeError, match="Expected str or Tensor"):
            loader(123)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected str or Tensor"):
            loader([1, 2, 3])  # type: ignore[arg-type]

    def test_file_loading_with_mock_video(self):
        """Test file path loading with a temporary file."""
        loader = LoadVideo(num_frames=10, resolution=128)

        # Create a temporary file to test file existence check
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4") as tmp,
            pytest.raises(RuntimeError, match="Failed to load video"),
        ):
            # This will fail at video decode but tests the path handling
            loader(tmp.name)


class TestUniformSample:
    """Test uniform sampling transform behavior."""

    def test_basic_sampling(self):
        """Test that uniform sampling works correctly."""
        sampler = UniformSample(num_frames=5)

        # Create frames with identifiable pattern
        frames = torch.arange(20).view(20, 1, 1, 1).expand(20, 3, 32, 32).float()
        result = sampler(frames)

        assert result.shape == (5, 3, 32, 32)
        # Check that frames are evenly spaced
        frame_indices = result[:, 0, 0, 0].long()
        diffs = frame_indices[1:] - frame_indices[:-1]
        assert (diffs >= 3).all() and (diffs <= 5).all()  # Roughly uniform

    def test_sampling_with_start_frame(self):
        """Test sampling with custom start frame."""
        sampler = UniformSample(num_frames=3, start_frame=5)
        frames = torch.randn(20, 3, 16, 16)

        result = sampler(frames)

        assert result.shape == (3, 3, 16, 16)

    def test_sample_all_frames(self):
        """Test sampling when requesting all frames."""
        sampler = UniformSample(num_frames=10)
        frames = torch.randn(10, 3, 32, 32)

        result = sampler(frames)

        assert result.shape == frames.shape

    def test_insufficient_frames(self):
        """Test error when not enough frames available."""
        sampler = UniformSample(num_frames=20)
        frames = torch.randn(10, 3, 32, 32)

        with pytest.raises(ValueError, match="Not enough frames"):
            sampler(frames)

    def test_invalid_dimensions(self):
        """Test error on wrong input dimensions."""
        sampler = UniformSample(num_frames=5)

        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            sampler(torch.randn(10, 32, 32))  # 3D

        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            sampler(torch.randn(10, 3, 32, 32, 2))  # 5D


class TestResize:
    """Test resize transform behavior."""

    def test_square_resize(self):
        """Test resizing to square dimensions."""
        resize = Resize(64)
        frames = torch.randn(5, 3, 128, 128)

        result = resize(frames)

        assert result.shape == (5, 3, 64, 64)
        assert result.dtype == frames.dtype

    def test_rectangular_resize(self):
        """Test resizing to rectangular dimensions."""
        resize = Resize((128, 96))
        frames = torch.randn(3, 3, 256, 256)

        result = resize(frames)

        assert result.shape == (3, 3, 128, 96)

    def test_different_interpolation_modes(self):
        """Test different interpolation modes produce valid output."""
        frames = torch.randn(2, 3, 32, 32)

        for mode in ["bilinear", "nearest", "bicubic"]:
            resize = Resize(64, interpolation=mode)
            result = resize(frames)

            assert result.shape == (2, 3, 64, 64)
            assert torch.isfinite(result).all()

    def test_preserves_content_structure(self):
        """Test that resizing preserves relative content structure."""
        # Create frames with clear pattern
        frames = torch.zeros(1, 3, 8, 8)
        frames[:, :, :4, :4] = 1.0  # Top-left quadrant white

        resize = Resize(16)
        result = resize(frames)

        # Top-left should still be brighter than bottom-right
        top_left_mean = result[:, :, :8, :8].mean()
        bottom_right_mean = result[:, :, 8:, 8:].mean()
        assert top_left_mean > bottom_right_mean

    def test_invalid_size(self):
        """Test error on invalid size values."""
        with pytest.raises(ValueError, match="size must be positive"):
            Resize(0)

        with pytest.raises(ValueError, match="size values must be positive"):
            Resize((224, -1))


class TestNormalize:
    """Test normalization transform behavior."""

    def test_standard_normalization(self):
        """Test standard normalization with known values."""
        normalize = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # Create frames with known values
        frames = torch.ones(4, 3, 32, 32) * 0.5
        result = normalize(frames)

        # (0.5 - 0.5) / 0.5 = 0
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_imagenet_normalization(self):
        """Test with ImageNet normalization values."""
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        frames = torch.rand(2, 3, 64, 64)
        result = normalize(frames)

        assert result.shape == frames.shape
        # Check that each channel is normalized correctly
        for c in range(3):
            channel_mean = result[:, c].mean()
            channel_std = result[:, c].std()
            # Should be roughly centered and scaled
            assert abs(channel_mean) < 2.0  # Reasonable range
            assert 0.5 < channel_std < 5.0  # Reasonable range

    def test_denormalization_inverse(self):
        """Test that normalization can be inverted."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = Normalize(mean=mean, std=std)

        frames = torch.rand(3, 3, 16, 16)
        normalized = normalize(frames)

        # Manually denormalize
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        denormalized = normalized * std_t + mean_t

        assert torch.allclose(denormalized, frames, atol=1e-6)

    def test_wrong_number_of_channels(self):
        """Test error on wrong number of channels."""
        normalize = Normalize()

        with pytest.raises(RuntimeError, match="Expected 3 channels"):
            normalize(torch.randn(5, 1, 32, 32))  # Grayscale

        with pytest.raises(RuntimeError, match="Expected 3 channels"):
            normalize(torch.randn(5, 4, 32, 32))  # RGBA

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="must have length 3"):
            Normalize(mean=(0.5, 0.5), std=(0.5, 0.5, 0.5))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="cannot be zero"):
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.0, 0.5))


class TestApplyMask:
    """Test masking transform behavior."""

    def test_bottom_half_masking(self):
        """Test bottom half masking actually masks bottom half."""
        mask_transform = ApplyMask(mask_ratio=0.5, mask_type="bottom_half")
        frames = torch.ones(2, 3, 8, 8)

        result = mask_transform(frames)

        assert isinstance(result, Tensor)  # Should be tensor, not tuple
        assert result.shape == frames.shape
        # Top half should be unchanged
        assert torch.allclose(result[:, :, :4, :], frames[:, :, :4, :])
        # Bottom half should be masked (zero)
        assert torch.allclose(result[:, :, 4:, :], torch.zeros(2, 3, 4, 8))

    def test_random_masking(self):
        """Test random masking masks approximately correct ratio."""
        mask_transform = ApplyMask(mask_ratio=0.3, mask_type="random")
        frames = torch.ones(4, 3, 32, 32)

        result = mask_transform(frames)

        assert isinstance(result, Tensor)  # Should be tensor, not tuple
        assert result.shape == frames.shape
        # Approximately 30% should be masked
        masked_ratio = (result == 0).float().mean()
        assert 0.2 < masked_ratio < 0.4  # Allow some variance

    def test_no_masking(self):
        """Test that 'none' mask type doesn't mask anything."""
        mask_transform = ApplyMask(mask_type="none")
        frames = torch.randn(3, 3, 16, 16)

        result = mask_transform(frames)

        assert isinstance(result, Tensor)  # Should be tensor, not tuple
        assert torch.equal(result, frames)

    def test_custom_mask_value(self):
        """Test masking with custom value."""
        mask_transform = ApplyMask(
            mask_ratio=0.5, mask_type="bottom_half", mask_value=-1.0
        )
        frames = torch.ones(2, 3, 4, 4)

        result = mask_transform(frames)

        # Bottom half should be -1
        assert isinstance(result, Tensor)  # Should be tensor, not tuple
        assert torch.allclose(result[:, :, 2:, :], -torch.ones(2, 3, 2, 4))

    def test_return_mask(self):
        """Test returning mask along with frames."""
        mask_transform = ApplyMask(
            mask_ratio=0.5, mask_type="bottom_half", return_mask=True
        )
        frames = torch.ones(2, 3, 8, 8)

        result = mask_transform(frames)

        assert isinstance(result, tuple)
        assert len(result) == 2
        masked_frames, mask = result

        assert masked_frames.shape == frames.shape
        assert mask.shape == (2, 8, 8)
        assert mask.dtype == torch.bool
        # Bottom half of mask should be True
        assert mask[:, 4:, :].all()
        assert not mask[:, :4, :].any()

    def test_invalid_mask_ratio(self):
        """Test error on invalid mask ratio."""
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            ApplyMask(mask_ratio=1.5)

        with pytest.raises(ValueError, match="mask_ratio must be in"):
            ApplyMask(mask_ratio=-0.1)


class TestAddNoise:
    """Test noise addition transform behavior."""

    def test_gaussian_noise(self):
        """Test adding Gaussian noise."""
        noise_transform = AddNoise(noise_std=0.1, noise_type="gaussian")
        frames = torch.zeros(100, 3, 16, 16)

        result = noise_transform(frames)

        assert result.shape == frames.shape
        # Should have added noise
        assert not torch.allclose(result, frames)
        # Noise should be roughly Gaussian with std ~0.1
        assert -0.5 < result.mean() < 0.5
        assert 0.05 < result.std() < 0.15

    def test_uniform_noise(self):
        """Test adding uniform noise."""
        noise_transform = AddNoise(noise_std=0.1, noise_type="uniform")
        frames = torch.zeros(100, 3, 16, 16)

        result = noise_transform(frames)

        assert result.shape == frames.shape
        # Should have added noise
        assert not torch.allclose(result, frames)
        # Uniform noise should be bounded
        assert result.min() >= -0.1
        assert result.max() <= 0.1

    def test_zero_noise(self):
        """Test that zero std adds no noise."""
        noise_transform = AddNoise(noise_std=0.0)
        frames = torch.randn(5, 3, 32, 32)

        result = noise_transform(frames)

        assert torch.equal(result, frames)

    def test_works_with_any_shape(self):
        """Test that noise works with any tensor shape."""
        noise_transform = AddNoise(noise_std=0.05)

        # 1D
        tensor_1d = torch.zeros(100)
        result_1d = noise_transform(tensor_1d)
        assert result_1d.shape == tensor_1d.shape
        assert not torch.allclose(result_1d, tensor_1d)

        # 2D
        tensor_2d = torch.zeros(50, 100)
        result_2d = noise_transform(tensor_2d)
        assert result_2d.shape == tensor_2d.shape
        assert not torch.allclose(result_2d, tensor_2d)

    def test_invalid_noise_std(self):
        """Test error on negative noise std."""
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            AddNoise(noise_std=-0.1)


class TestToTensor:
    """Test tensor conversion transform behavior."""

    def test_numpy_to_tensor(self):
        """Test converting numpy array to tensor."""
        to_tensor = ToTensor(dtype=torch.float32)
        np_array = np.random.randn(5, 3, 32, 32).astype(np.float64)

        result = to_tensor(np_array)

        assert isinstance(result, Tensor)
        assert result.shape == (5, 3, 32, 32)
        assert result.dtype == torch.float32

    def test_list_to_tensor(self):
        """Test converting list to tensor."""
        to_tensor = ToTensor()
        list_data = [[1, 2, 3], [4, 5, 6]]

        result = to_tensor(list_data)

        assert isinstance(result, Tensor)
        assert result.shape == (2, 3)
        assert torch.equal(
            result, torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        )

    def test_tensor_dtype_conversion(self):
        """Test converting tensor to different dtype."""
        to_tensor = ToTensor(dtype=torch.float64)
        input_tensor = torch.randn(3, 3, 16, 16, dtype=torch.float32)

        result = to_tensor(input_tensor)

        assert result.dtype == torch.float64
        assert torch.allclose(result.float(), input_tensor)

    def test_device_placement(self):
        """Test placing tensor on specific device."""
        device = torch.device("cpu")
        to_tensor = ToTensor(device=device)
        np_array = np.random.randn(2, 3, 8, 8)

        result = to_tensor(np_array)

        assert result.device.type == "cpu"

    def test_conversion_error(self):
        """Test error handling for unconvertible data."""
        to_tensor = ToTensor()

        # Object that can't be converted
        class UnconvertibleObject:
            pass

        with pytest.raises(RuntimeError, match="Failed to convert to tensor"):
            to_tensor(UnconvertibleObject())


class TestLambda:
    """Test lambda transform behavior."""

    def test_simple_lambda(self):
        """Test simple lambda function."""
        # Clamp values to [0, 1]
        clamp_transform = Lambda(lambda x: torch.clamp(x, 0, 1))
        frames = torch.randn(5, 3, 16, 16) * 2  # Some values outside [0, 1]

        result = clamp_transform(frames)

        assert result.shape == frames.shape
        assert result.min() >= 0
        assert result.max() <= 1

    def test_complex_lambda(self):
        """Test more complex lambda function."""
        # Add 1 and take absolute value
        complex_transform = Lambda(lambda x: (x + 1).abs())
        frames = torch.randn(3, 3, 8, 8) - 0.5

        result = complex_transform(frames)

        assert result.shape == frames.shape
        assert (result >= 0).all()
        assert torch.allclose(result, (frames + 1).abs())

    def test_lambda_changing_shape(self):
        """Test lambda that changes tensor shape."""
        # Flatten spatial dimensions
        flatten_transform = Lambda(lambda x: x.view(x.shape[0], -1))
        frames = torch.randn(4, 3, 16, 16)

        result = flatten_transform(frames)

        assert result.shape == (4, 3 * 16 * 16)

    def test_invalid_lambda(self):
        """Test error on non-callable lambda."""
        with pytest.raises(TypeError, match="lambd must be callable"):
            Lambda("not_callable")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="lambd must be callable"):
            Lambda(42)  # type: ignore[arg-type]


class TestIntegration:
    """Integration tests for complex transform pipelines."""

    def test_video_preprocessing_pipeline(self):
        """Test complete video preprocessing pipeline."""
        pipeline = Compose(
            [
                UniformSample(num_frames=8),
                Resize(64),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # Create input frames
        frames = torch.rand(32, 3, 128, 128)
        result = pipeline(frames)

        # Verify complete pipeline
        assert result.shape == (8, 3, 64, 64)
        # Should be normalized (roughly centered)
        for c in range(3):
            channel_data = result[:, c]
            assert -3 < channel_data.mean() < 3
            assert 0.5 < channel_data.std() < 5

    def test_augmentation_pipeline(self):
        """Test augmentation pipeline with masking and noise."""
        pipeline = Compose(
            [
                ApplyMask(mask_ratio=0.25, mask_type="random"),
                AddNoise(noise_std=0.05),
                Lambda(lambda x: torch.clamp(x, -1, 1)),  # Clamp final values
            ]
        )

        frames = torch.randn(10, 3, 32, 32) * 0.5
        result = pipeline(frames)

        assert result.shape == frames.shape
        assert result.min() >= -1
        assert result.max() <= 1
        # Should be different from input due to mask and noise
        assert not torch.allclose(result, frames)

    def test_mixed_type_pipeline(self):
        """Test pipeline with mixed input/output types."""
        pipeline = Compose(
            [
                ToTensor(dtype=torch.float32),
                Resize(32),
                Normalize(),
            ]
        )

        # Start with numpy array
        np_frames = np.random.randn(5, 3, 64, 64).astype(np.float64)
        result = pipeline(np_frames)

        assert isinstance(result, Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (5, 3, 32, 32)

    def test_error_propagation_in_pipeline(self):
        """Test that errors in pipeline are properly reported."""
        pipeline = Compose([UniformSample(num_frames=10), Resize(224), Normalize()])

        # Input with too few frames
        frames = torch.randn(5, 3, 128, 128)

        with pytest.raises(ValueError, match="Not enough frames"):
            pipeline(frames)

    def test_deterministic_pipeline(self):
        """Test that deterministic transforms produce same output."""
        pipeline = Compose(
            [
                UniformSample(num_frames=4),
                Resize(32),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
            ]
        )

        frames = torch.randn(16, 3, 64, 64)

        # Multiple runs should produce identical results
        result1 = pipeline(frames)
        result2 = pipeline(frames)

        assert torch.equal(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
