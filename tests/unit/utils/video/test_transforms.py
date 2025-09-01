"""Comprehensive tests for video transform classes."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

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
    VideoTransform,
)


class TestVideoTransformInterface:
    """Test VideoTransform abstract interface."""

    def test_abstract_interface(self):
        """Test that VideoTransform cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            VideoTransform()  # type: ignore[abstract]

    def test_repr_method(self):
        """Test that concrete transforms implement __repr__."""
        transform = UniformSample(num_frames=100)
        repr_str = repr(transform)
        assert "UniformSample" in repr_str
        assert "100" in repr_str


class TestCompose:
    """Test Compose transform composition."""

    def test_initialization(self):
        """Test proper initialization of Compose."""
        transforms = [UniformSample(num_frames=50), Resize(224), Normalize()]
        compose = Compose(transforms)

        assert len(compose.transforms) == 3
        assert compose.transforms[0] == transforms[0]

    def test_invalid_initialization(self):
        """Test that invalid transforms raise errors."""
        # Non-callable items should raise TypeError
        with pytest.raises(TypeError, match="All transforms must be callable"):
            Compose([UniformSample(100), "not_callable", Resize(224)])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="All transforms must be callable"):
            Compose([UniformSample(100), 42])  # type: ignore[list-item]

    def test_composition_order(self):
        """Test that transforms are applied in order."""
        # Mock transforms to track call order
        transform1 = Mock()
        transform1.return_value = torch.randn(5, 3, 224, 224)
        transform2 = Mock()
        transform2.return_value = torch.randn(5, 3, 224, 224)
        transform3 = Mock()
        transform3.return_value = torch.randn(5, 3, 224, 224)

        compose = Compose([transform1, transform2, transform3])

        input_tensor = torch.randn(10, 3, 256, 256)
        compose(input_tensor)

        # Verify call order
        transform1.assert_called_once_with(input_tensor)
        transform2.assert_called_once_with(transform1.return_value)
        transform3.assert_called_once_with(transform2.return_value)

    def test_repr(self):
        """Test string representation of Compose."""
        transforms = [UniformSample(50), Resize(224)]
        compose = Compose(transforms)

        repr_str = repr(compose)
        assert "Compose" in repr_str
        assert "UniformSample" in repr_str
        assert "Resize" in repr_str
        assert "\n" in repr_str  # Multi-line format


class TestLoadVideo:
    """Test LoadVideo transform."""

    def test_initialization(self):
        """Test proper initialization of LoadVideo."""
        loader = LoadVideo(
            num_frames=512,
            resolution=224,
            sampling_strategy="uniform",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert loader.num_frames == 512
        assert loader.resolution == 224
        assert loader.sampling_strategy == "uniform"
        assert loader.device == torch.device("cpu")
        assert loader.dtype == torch.float32

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Invalid num_frames
        with pytest.raises(ValueError, match="num_frames must be positive"):
            LoadVideo(num_frames=0)

        with pytest.raises(ValueError, match="num_frames must be positive"):
            LoadVideo(num_frames=-10)

        # Invalid resolution
        with pytest.raises(ValueError, match="resolution must be positive"):
            LoadVideo(num_frames=100, resolution=0)

        with pytest.raises(ValueError, match="resolution must be positive"):
            LoadVideo(num_frames=100, resolution=-224)

    def test_call_with_path(self):
        """Test calling LoadVideo with file path."""
        loader = LoadVideo(num_frames=100, resolution=224)

        # Mock the functional call
        with patch("associative.utils.video.functional.load_video") as mock_load:
            mock_load.return_value = torch.randn(100, 3, 224, 224)

            result = loader("test_video.mp4")

            mock_load.assert_called_once_with(
                video_path="test_video.mp4",
                num_frames=100,
                resolution=224,
                sampling_strategy="uniform",
                device=None,
                dtype=None,
            )
            assert result.shape == (100, 3, 224, 224)

    def test_call_with_tensor(self):
        """Test calling LoadVideo with pre-loaded tensor."""
        loader = LoadVideo(num_frames=100, resolution=224)
        input_tensor = torch.randn(100, 3, 224, 224)

        result = loader(input_tensor)
        assert torch.equal(result, input_tensor)

    def test_call_with_invalid_input(self):
        """Test error handling for invalid input types."""
        loader = LoadVideo(num_frames=100, resolution=224)

        with pytest.raises(TypeError, match="Expected str or Tensor"):
            loader(123)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Expected str or Tensor"):
            loader(["not", "valid"])  # type: ignore[arg-type]

    def test_repr(self):
        """Test string representation."""
        loader = LoadVideo(num_frames=512, resolution=256, sampling_strategy="random")
        repr_str = repr(loader)

        assert "LoadVideo" in repr_str
        assert "512" in repr_str
        assert "256" in repr_str
        assert "random" in repr_str


class TestUniformSample:
    """Test UniformSample transform."""

    def test_initialization(self):
        """Test proper initialization of UniformSample."""
        sampler = UniformSample(num_frames=100, start_frame=10)

        assert sampler.num_frames == 100
        assert sampler.start_frame == 10

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Invalid num_frames
        with pytest.raises(ValueError, match="num_frames must be positive"):
            UniformSample(num_frames=0)

        with pytest.raises(ValueError, match="num_frames must be positive"):
            UniformSample(num_frames=-50)

        # Invalid start_frame
        with pytest.raises(ValueError, match="start_frame must be non-negative"):
            UniformSample(num_frames=100, start_frame=-1)

    def test_input_validation(self):
        """Test input tensor validation."""
        sampler = UniformSample(num_frames=50)

        # Wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            sampler(torch.randn(100, 224, 224))  # Missing channel dimension

        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            sampler(torch.randn(100))  # 1D tensor

        # Not enough frames
        with pytest.raises(ValueError, match="Not enough frames"):
            sampler(torch.randn(10, 3, 224, 224))  # Only 10 frames, need 50

    def test_sampling_logic(self):
        """Test uniform sampling logic."""
        sampler = UniformSample(num_frames=25)
        frames = torch.randn(100, 3, 224, 224)

        # Mock the functional call
        with patch(
            "associative.utils.video.functional.uniform_sample_indices"
        ) as mock_indices:
            mock_indices.return_value = torch.arange(0, 100, 4)  # Every 4th frame

            sampler(frames)

            mock_indices.assert_called_once_with(100, 25, 0)
            # Result should use the mocked indices
            # Can't test exact values without implementing, but shape should match

    def test_repr(self):
        """Test string representation."""
        sampler = UniformSample(num_frames=128, start_frame=5)
        repr_str = repr(sampler)

        assert "UniformSample" in repr_str
        assert "128" in repr_str
        assert "5" in repr_str


class TestResize:
    """Test Resize transform."""

    def test_initialization(self):
        """Test proper initialization of Resize."""
        # Square resize
        resize_square = Resize(224)
        assert resize_square.size == (224, 224)
        assert resize_square.interpolation == "bilinear"
        assert not resize_square.align_corners

        # Rectangular resize
        resize_rect = Resize((256, 224))
        assert resize_rect.size == (256, 224)

        # Custom parameters
        resize_custom = Resize(224, interpolation="nearest", align_corners=True)
        assert resize_custom.interpolation == "nearest"
        assert resize_custom.align_corners

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Invalid size
        with pytest.raises(ValueError, match="size must be positive"):
            Resize(0)

        with pytest.raises(ValueError, match="size must be positive"):
            Resize(-224)

        with pytest.raises(ValueError, match="All size values must be positive"):
            Resize((224, 0))

        with pytest.raises(ValueError, match="All size values must be positive"):
            Resize((224, -256))

    def test_input_validation(self):
        """Test input tensor validation."""
        resize = Resize(224)

        # Wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            resize(torch.randn(100, 224, 224))  # Missing channel dimension

    def test_resize_logic(self):
        """Test resize logic with mocked functional call."""
        resize = Resize((256, 224), interpolation="nearest")
        frames = torch.randn(10, 3, 128, 128)

        with patch("associative.utils.video.functional.resize_frames") as mock_resize:
            mock_resize.return_value = torch.randn(10, 3, 256, 224)

            resize(frames)

            mock_resize.assert_called_once_with(
                frames=frames,
                size=(256, 224),
                interpolation="nearest",
                align_corners=False,
            )

    def test_repr(self):
        """Test string representation."""
        resize = Resize((320, 240), interpolation="bicubic")
        repr_str = repr(resize)

        assert "Resize" in repr_str
        assert "(320, 240)" in repr_str
        assert "bicubic" in repr_str


class TestNormalize:
    """Test Normalize transform."""

    def test_initialization(self):
        """Test proper initialization of Normalize."""
        # Default initialization
        normalize_default = Normalize()
        assert normalize_default.mean == (0.5, 0.5, 0.5)
        assert normalize_default.std == (0.5, 0.5, 0.5)

        # Custom initialization
        normalize_custom = Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        assert normalize_custom.mean == (0.485, 0.456, 0.406)
        assert normalize_custom.std == (0.229, 0.224, 0.225)

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Wrong length mean/std
        with pytest.raises(ValueError, match="length 3"):
            Normalize(mean=(0.5, 0.5), std=(0.5, 0.5, 0.5))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="length 3"):
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5))  # type: ignore[arg-type]

        # Zero std values
        with pytest.raises(ValueError, match="std values cannot be zero"):
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.0, 0.5))

    def test_input_validation(self):
        """Test input tensor validation."""
        normalize = Normalize()

        # Wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            normalize(torch.randn(100, 224, 224))

        # Wrong number of channels
        with pytest.raises(RuntimeError, match="Expected 3 channels"):
            normalize(torch.randn(10, 1, 224, 224))

        with pytest.raises(RuntimeError, match="Expected 3 channels"):
            normalize(torch.randn(10, 4, 224, 224))

    def test_normalization_logic(self):
        """Test normalization logic with mocked functional call."""
        normalize = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        frames = torch.randn(8, 3, 224, 224)

        with patch(
            "associative.utils.video.functional.normalize_frames"
        ) as mock_normalize:
            mock_normalize.return_value = torch.randn(8, 3, 224, 224)

            normalize(frames)

            mock_normalize.assert_called_once_with(
                frames=frames, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            )

    def test_repr(self):
        """Test string representation."""
        normalize = Normalize(mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
        repr_str = repr(normalize)

        assert "Normalize" in repr_str
        assert "(0.1, 0.2, 0.3)" in repr_str
        assert "(0.4, 0.5, 0.6)" in repr_str


class TestApplyMask:
    """Test ApplyMask transform."""

    def test_initialization(self):
        """Test proper initialization of ApplyMask."""
        # Default initialization
        mask_default = ApplyMask()
        assert mask_default.mask_ratio == 0.5
        assert mask_default.mask_type == "bottom_half"
        assert mask_default.mask_value == 0.0
        assert not mask_default.return_mask

        # Custom initialization
        mask_custom = ApplyMask(
            mask_ratio=0.7, mask_type="random", mask_value=-1.0, return_mask=True
        )
        assert mask_custom.mask_ratio == 0.7
        assert mask_custom.mask_type == "random"
        assert mask_custom.mask_value == -1.0
        assert mask_custom.return_mask

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Invalid mask_ratio
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            ApplyMask(mask_ratio=-0.1)

        with pytest.raises(ValueError, match="mask_ratio must be in"):
            ApplyMask(mask_ratio=1.5)

        # Invalid mask_type
        with pytest.raises(ValueError, match="Invalid mask_type"):
            ApplyMask(mask_type="invalid")  # type: ignore[arg-type]

    def test_input_validation(self):
        """Test input tensor validation."""
        mask_transform = ApplyMask()

        # Wrong dimensions
        with pytest.raises(RuntimeError, match="Expected 4D tensor"):
            mask_transform(torch.randn(100, 224, 224))

    def test_masking_logic(self):
        """Test masking logic with mocked functional call."""
        mask_transform = ApplyMask(mask_ratio=0.3, mask_type="random", return_mask=True)
        frames = torch.randn(5, 3, 224, 224)

        with patch("associative.utils.video.functional.apply_mask") as mock_mask:
            mock_masked = torch.randn(5, 3, 224, 224)
            mock_mask_tensor = torch.randint(0, 2, (5, 224, 224), dtype=torch.bool)
            mock_mask.return_value = (mock_masked, mock_mask_tensor)

            result = mask_transform(frames)

            mock_mask.assert_called_once_with(
                frames=frames, mask_ratio=0.3, mask_type="random", mask_value=0.0
            )

            # Should return tuple when return_mask=True
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_return_mask_behavior(self):
        """Test different return_mask behaviors."""
        frames = torch.randn(3, 3, 224, 224)

        with patch("associative.utils.video.functional.apply_mask") as mock_mask:
            mock_masked = torch.randn(3, 3, 224, 224)
            mock_mask_tensor = torch.randint(0, 2, (3, 224, 224), dtype=torch.bool)
            mock_mask.return_value = (mock_masked, mock_mask_tensor)

            # return_mask=False should return only masked frames
            mask_no_return = ApplyMask(return_mask=False)
            result_no_return = mask_no_return(frames)
            assert not isinstance(result_no_return, tuple)

            # return_mask=True should return tuple
            mask_with_return = ApplyMask(return_mask=True)
            result_with_return = mask_with_return(frames)
            assert isinstance(result_with_return, tuple)
            assert len(result_with_return) == 2

    def test_repr(self):
        """Test string representation."""
        mask_transform = ApplyMask(mask_ratio=0.8, mask_type="random", return_mask=True)
        repr_str = repr(mask_transform)

        assert "ApplyMask" in repr_str
        assert "0.8" in repr_str
        assert "random" in repr_str
        assert "True" in repr_str


class TestAddNoise:
    """Test AddNoise transform."""

    def test_initialization(self):
        """Test proper initialization of AddNoise."""
        # Default initialization
        noise_default = AddNoise()
        assert noise_default.noise_std == 0.1
        assert noise_default.noise_type == "gaussian"

        # Custom initialization
        noise_custom = AddNoise(noise_std=0.05, noise_type="uniform")
        assert noise_custom.noise_std == 0.05
        assert noise_custom.noise_type == "uniform"

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Invalid noise_std
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            AddNoise(noise_std=-0.1)

        # Invalid noise_type
        with pytest.raises(ValueError, match="Invalid noise_type"):
            AddNoise(noise_type="invalid")  # type: ignore[arg-type]

    def test_noise_logic(self):
        """Test noise addition logic with mocked functional call."""
        noise_transform = AddNoise(noise_std=0.2, noise_type="uniform")
        frames = torch.randn(10, 768)  # Can work with any tensor shape

        with patch("associative.utils.video.functional.add_noise") as mock_noise:
            mock_noise.return_value = torch.randn(10, 768)

            noise_transform(frames)

            mock_noise.assert_called_once_with(
                frames=frames, noise_std=0.2, noise_type="uniform"
            )

    def test_repr(self):
        """Test string representation."""
        noise_transform = AddNoise(noise_std=0.15, noise_type="gaussian")
        repr_str = repr(noise_transform)

        assert "AddNoise" in repr_str
        assert "0.15" in repr_str
        assert "gaussian" in repr_str


class TestToTensor:
    """Test ToTensor transform."""

    def test_initialization(self):
        """Test proper initialization of ToTensor."""
        # Default initialization
        to_tensor_default = ToTensor()
        assert to_tensor_default.dtype == torch.float32
        assert to_tensor_default.device is None

        # Custom initialization
        device = torch.device("cpu")
        to_tensor_custom = ToTensor(dtype=torch.float64, device=device)
        assert to_tensor_custom.dtype == torch.float64
        assert to_tensor_custom.device == device

    def test_tensor_conversion(self):
        """Test tensor conversion from different input types."""
        to_tensor = ToTensor(dtype=torch.float32)

        # Test with numpy array
        np_array = np.random.randn(10, 3, 32, 32).astype(np.float32)

        with patch("torch.tensor") as mock_tensor:
            mock_tensor.return_value = torch.randn(10, 3, 32, 32)

            to_tensor(np_array)

            mock_tensor.assert_called_once()
            # First argument should be the numpy array
            args, kwargs = mock_tensor.call_args
            np.testing.assert_array_equal(args[0], np_array)
            assert kwargs["dtype"] == torch.float32

    def test_tensor_input_handling(self):
        """Test handling when input is already a tensor."""
        to_tensor = ToTensor(dtype=torch.float64, device=torch.device("cpu"))
        input_tensor = torch.randn(5, 3, 224, 224, dtype=torch.float32)

        to_tensor(input_tensor)

        # Should convert to target dtype and device
        # (Actual implementation would do this via .to() calls)

    def test_conversion_error_handling(self):
        """Test error handling during conversion."""
        to_tensor = ToTensor()

        # Mock torch.tensor to raise an exception
        with patch("torch.tensor") as mock_tensor:
            mock_tensor.side_effect = RuntimeError("Conversion failed")

            with pytest.raises(RuntimeError, match="Failed to convert to tensor"):
                to_tensor([1, 2, 3])  # Should trigger the exception

    def test_repr(self):
        """Test string representation."""
        to_tensor = ToTensor(dtype=torch.float16, device=torch.device("cpu"))
        repr_str = repr(to_tensor)

        assert "ToTensor" in repr_str
        assert "float16" in repr_str
        assert "cpu" in repr_str


class TestLambda:
    """Test Lambda transform."""

    def test_initialization(self):
        """Test proper initialization of Lambda."""

        # Valid callable
        def lambda_fn(x):
            return x * 2

        lambda_transform = Lambda(lambda_fn)
        assert lambda_transform.lambd == lambda_fn

    def test_invalid_initialization(self):
        """Test that invalid lambda raises error."""
        # Non-callable
        with pytest.raises(TypeError, match="lambd must be callable"):
            Lambda("not_callable")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="lambd must be callable"):
            Lambda(42)  # type: ignore[arg-type]

    def test_lambda_execution(self):
        """Test that lambda function is executed correctly."""
        # Simple lambda
        double_transform = Lambda(lambda x: x * 2)
        input_tensor = torch.randn(5, 10)

        # Mock the lambda to verify it's called
        mock_lambda = Mock(return_value=torch.randn(5, 10))
        double_transform.lambd = mock_lambda

        double_transform(input_tensor)

        mock_lambda.assert_called_once_with(input_tensor)

    def test_complex_lambda(self):
        """Test with more complex lambda functions."""

        # Multi-operation lambda
        def complex_lambda(x):
            return torch.clamp(x.abs(), min=0.1, max=1.0)

        complex_transform = Lambda(complex_lambda)

        # Should not raise any errors during initialization
        assert callable(complex_transform.lambd)

    def test_repr(self):
        """Test string representation."""
        lambda_transform = Lambda(lambda x: x)
        repr_str = repr(lambda_transform)

        assert "Lambda" in repr_str


class TestTransformIntegration:
    """Integration tests for transform composition and edge cases."""

    def test_complex_pipeline(self):
        """Test complex transform pipeline."""
        pipeline = Compose(
            [
                UniformSample(num_frames=100),
                Resize((256, 224)),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ApplyMask(mask_ratio=0.3, mask_type="random"),
                AddNoise(noise_std=0.01),
            ]
        )

        # Should initialize without errors
        assert len(pipeline.transforms) == 5

        # Each transform should be the correct type
        assert isinstance(pipeline.transforms[0], UniformSample)
        assert isinstance(pipeline.transforms[1], Resize)
        assert isinstance(pipeline.transforms[2], Normalize)
        assert isinstance(pipeline.transforms[3], ApplyMask)
        assert isinstance(pipeline.transforms[4], AddNoise)

    def test_transform_error_propagation(self):
        """Test that errors propagate correctly through pipeline."""
        pipeline = Compose(
            [
                UniformSample(num_frames=50),
                Resize(224),
                Lambda(lambda x: 1 / 0),  # Will cause division by zero
            ]
        )

        frames = torch.randn(100, 3, 224, 224)

        # Error should propagate through pipeline
        # Lambda causes division by zero
        with pytest.raises(ZeroDivisionError):
            pipeline(frames)

    def test_device_consistency_through_pipeline(self):
        """Test device consistency through transform pipeline."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        Compose([ToTensor(device=device), Resize(224), Normalize()])

        # Should maintain device consistency
        # (Would need implementation to test actual device placement)

    def test_batch_processing_consistency(self):
        """Test that transforms handle batches consistently."""
        Compose([Resize(224), Normalize()])

        Compose([Resize(224), Normalize()])

        # Single frame and batch should work with same transforms
        torch.randn(1, 3, 256, 256)
        torch.randn(10, 3, 256, 256)

        # Both should be processable by the same pipeline
        # (Would need implementation to test actual processing)

    def test_transform_parameter_validation_consistency(self):
        """Test that parameter validation is consistent across transforms."""
        # All transforms should handle negative values consistently
        with pytest.raises(ValueError):
            UniformSample(num_frames=-1)

        with pytest.raises(ValueError):
            Resize(size=-1)

        with pytest.raises(ValueError):
            AddNoise(noise_std=-1)

        # All should handle zero values appropriately
        with pytest.raises(ValueError):
            UniformSample(num_frames=0)

        with pytest.raises(ValueError):
            Resize(size=0)

    def test_memory_efficiency(self):
        """Test that transforms don't unnecessarily copy tensors."""
        # Some transforms should modify in-place when possible
        # Others should create new tensors only when necessary

        # This is more of a performance test that would require
        # implementation details to verify properly

        input_tensor = torch.randn(10, 3, 224, 224)
        id(input_tensor)

        # Lambda transform might reuse tensors
        identity_transform = Lambda(lambda x: x)
        identity_transform(input_tensor)

        # Identity should potentially reuse the same tensor
        # (Implementation dependent)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
