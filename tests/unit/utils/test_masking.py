"""Tests for spatial masking utilities.

These tests define the expected behavior for spatial masking functions
before implementation, ensuring correct behavior in normal and edge cases.
"""

import pytest
import torch


class TestSpatialMasking:
    """Test spatial masking for video frames."""

    def test_lower_half_masking(self):
        """Test masking the lower half of frames."""
        # Create test video: [num_frames, height, width, channels]
        video = torch.ones(10, 100, 100, 3)

        # Import will fail initially - that's expected
        from associative.utils.masking import apply_spatial_mask

        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.5, mask_type="lower"
        )

        # Check shape preservation
        assert masked_video.shape == video.shape
        assert mask.shape == video.shape

        # Check lower half is masked
        assert torch.all(masked_video[:, 50:, :, :] == 0)
        assert torch.all(mask[:, 50:, :, :] == 0)

        # Check upper half is unchanged
        assert torch.all(masked_video[:, :50, :, :] == 1)
        assert torch.all(mask[:, :50, :, :] == 1)

    def test_upper_half_masking(self):
        """Test masking the upper half of frames."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.5, mask_type="upper"
        )

        # Check upper half is masked
        assert torch.all(masked_video[:, :50, :, :] == 0)
        assert torch.all(mask[:, :50, :, :] == 0)

        # Check lower half is unchanged
        assert torch.all(masked_video[:, 50:, :, :] == 1)
        assert torch.all(mask[:, 50:, :, :] == 1)

    def test_custom_mask_ratio(self):
        """Test with different mask ratios."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        # Test 30% masking
        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.3, mask_type="lower"
        )

        # Bottom 30% should be masked (pixels 70-100)
        assert torch.all(masked_video[:, 70:, :, :] == 0)
        assert torch.all(masked_video[:, :70, :, :] == 1)

        # Test 70% masking
        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.7, mask_type="lower"
        )

        # Bottom 70% should be masked (pixels 30-100)
        assert torch.all(masked_video[:, 30:, :, :] == 0)
        assert torch.all(masked_video[:, :30, :, :] == 1)

    def test_edge_cases(self):
        """Test edge cases for mask ratios."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        # Test 0% masking (no mask)
        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.0, mask_type="lower"
        )
        assert torch.all(masked_video == 1)
        assert torch.all(mask == 1)

        # Test 100% masking (full mask)
        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=1.0, mask_type="lower"
        )
        assert torch.all(masked_video == 0)
        assert torch.all(mask == 0)

    def test_random_spatial_masking(self):
        """Test random spatial masking."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        # Set seed for reproducibility
        torch.manual_seed(42)

        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.5, mask_type="random"
        )

        # Check that approximately 50% of pixels are masked
        mask_percentage = (mask == 0).float().mean()
        assert abs(mask_percentage - 0.5) < 0.01  # Within 1% of target

        # Check that mask is applied correctly
        assert torch.all((mask == 0) == (masked_video == 0))
        assert torch.all((mask == 1) == (masked_video == video))

    def test_block_spatial_masking(self):
        """Test block-based spatial masking."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        masked_video, mask = apply_spatial_mask(
            video,
            mask_ratio=0.5,
            mask_type="block",
            block_size=10,  # 10x10 blocks
        )

        # Check that mask consists of blocks
        # Each 10x10 block should be either all 0 or all 1
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                block = mask[0, i : i + 10, j : j + 10, 0]
                assert torch.all(block == block[0, 0])  # All values in block are same

        # Check approximate ratio
        mask_percentage = (mask == 0).float().mean()
        assert abs(mask_percentage - 0.5) < 0.1  # Within 10% due to block quantization

    def test_different_input_shapes(self):
        """Test with various input shapes."""
        from associative.utils.masking import apply_spatial_mask

        # Test with different frame counts
        for num_frames in [1, 10, 100]:
            video = torch.ones(num_frames, 224, 224, 3)
            masked_video, mask = apply_spatial_mask(video, 0.5, "lower")
            assert masked_video.shape == video.shape

        # Test with different resolutions
        for height, width in [(100, 100), (224, 224), (256, 512)]:
            video = torch.ones(10, height, width, 3)
            masked_video, mask = apply_spatial_mask(video, 0.5, "lower")
            assert masked_video.shape == video.shape

        # Test with different channel counts
        for channels in [1, 3, 4]:
            video = torch.ones(10, 100, 100, channels)
            masked_video, mask = apply_spatial_mask(video, 0.5, "lower")
            assert masked_video.shape == video.shape

    def test_mask_preserves_values(self):
        """Test that unmasked regions preserve exact values."""
        video = torch.randn(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        masked_video, mask = apply_spatial_mask(video, 0.5, "lower")

        # Unmasked region should have exact same values
        assert torch.allclose(masked_video[:, :50, :, :], video[:, :50, :, :])

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        from associative.utils.masking import apply_spatial_mask

        video = torch.ones(10, 100, 100, 3)

        # Invalid mask ratio
        with pytest.raises(ValueError):
            apply_spatial_mask(video, -0.1, "lower")

        with pytest.raises(ValueError):
            apply_spatial_mask(video, 1.1, "lower")

        # Invalid mask type
        with pytest.raises(ValueError):
            apply_spatial_mask(video, 0.5, "invalid_type")

        # Wrong input dimensions
        with pytest.raises(ValueError):
            apply_spatial_mask(
                torch.ones(100, 100, 3), 0.5, "lower"
            )  # 3D instead of 4D

        with pytest.raises(ValueError):
            apply_spatial_mask(torch.ones(10, 100, 100, 3, 1), 0.5, "lower")  # 5D

    def test_left_right_masking(self):
        """Test left and right side masking."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        # Test left masking
        masked_video, mask = apply_spatial_mask(video, mask_ratio=0.3, mask_type="left")
        assert torch.all(masked_video[:, :, :30, :] == 0)  # Left 30% masked
        assert torch.all(masked_video[:, :, 30:, :] == 1)  # Right 70% unchanged

        # Test right masking
        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.4, mask_type="right"
        )
        assert torch.all(masked_video[:, :, 60:, :] == 0)  # Right 40% masked
        assert torch.all(masked_video[:, :, :60, :] == 1)  # Left 60% unchanged

    def test_center_masking(self):
        """Test center masking (masks center region)."""
        video = torch.ones(10, 100, 100, 3)

        from associative.utils.masking import apply_spatial_mask

        masked_video, mask = apply_spatial_mask(
            video, mask_ratio=0.5, mask_type="center"
        )

        # Center 50% should be masked (pixels 25-75 in both dimensions)
        assert torch.all(masked_video[:, 25:75, 25:75, :] == 0)
        # Borders should be unchanged
        assert torch.all(masked_video[:, :25, :, :] == 1)
        assert torch.all(masked_video[:, 75:, :, :] == 1)
        assert torch.all(masked_video[:, :, :25, :] == 1)
        assert torch.all(masked_video[:, :, 75:, :] == 1)
