"""Tests for MovieChat1K dataset."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision import transforms

from associative.datasets import MovieChat1K

# Test constants
TEST_NUM_FRAMES = 16
TEST_RESOLUTION = 224
TEST_MASK_RATIO = 0.5
TEST_NUM_VIDEOS = 3
TEST_NUM_SEGMENTS = 10
TEST_TOTAL_FRAMES = 100
TEST_STD_THRESHOLD = 2
TEST_MASK_LOWER_BOUND = 0.25
TEST_MASK_UPPER_BOUND = 0.35
TEST_NORMALIZE_MIN = -1.5
TEST_NORMALIZE_MAX = 1.5


class TestMovieChat1K:
    """Test suite for MovieChat1K dataset."""

    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory with mock video files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy .mp4 files
            for i in range(3):
                Path(tmpdir, f"video_{i}.mp4").touch()
            yield tmpdir

    @pytest.fixture
    def mock_video_reader(self):
        """Mock VideoReader to avoid needing actual video files."""
        with patch("associative.datasets.moviechat.VideoReader") as mock:
            # Create mock video reader that returns random frames
            mock_reader = MagicMock()
            mock_reader.__len__.return_value = 100  # 100 frames
            mock_reader.__getitem__.return_value = MagicMock(
                asnumpy=lambda: np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            mock.return_value = mock_reader
            yield mock

    def test_initialization(self, temp_video_dir):
        """Test dataset initialization."""
        dataset = MovieChat1K(
            root=temp_video_dir,
            num_frames=16,
            resolution=224,
            mask_config={"mask_type": "bottom_half", "mask_ratio": 0.5},
        )

        assert dataset.root == Path(temp_video_dir)
        assert dataset.num_frames == TEST_NUM_FRAMES
        assert dataset.resolution == TEST_RESOLUTION
        assert dataset.mask_ratio == TEST_MASK_RATIO
        assert dataset.mask_type == "bottom_half"
        assert len(dataset) == TEST_NUM_VIDEOS  # mock videos

    def test_no_videos_error(self):
        """Test error when no videos found."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="No .mp4 files found"),
        ):
            MovieChat1K(root=tmpdir)

    def test_frame_indices(self):
        """Test frame index calculation."""
        from associative.datasets.moviechat import get_frame_indices

        indices = get_frame_indices(
            num_frames=TEST_TOTAL_FRAMES, num_segments=TEST_NUM_SEGMENTS
        )

        assert len(indices) == TEST_NUM_SEGMENTS
        assert indices[0] >= 0
        assert indices[-1] < TEST_TOTAL_FRAMES
        # Check roughly uniform spacing
        diffs = np.diff(indices)
        assert np.std(diffs) < TEST_STD_THRESHOLD  # Should be roughly uniform

    def test_load_video_frames(self, temp_video_dir, mock_video_reader):
        """Test video frame loading."""
        dataset = MovieChat1K(
            root=temp_video_dir,
            num_frames=8,
            resolution=224,
        )

        video_path = Path(temp_video_dir, "video_0.mp4")
        frames = dataset.load_video_frames(video_path)

        assert frames.shape == (8, 3, 224, 224)
        assert frames.dtype == torch.float32
        # Check normalization (should be in [-1, 1] with default transform)
        assert frames.min() >= TEST_NORMALIZE_MIN
        assert frames.max() <= TEST_NORMALIZE_MAX

    def test_apply_mask_bottom_half(self, temp_video_dir):
        """Test bottom half masking."""
        dataset = MovieChat1K(
            root=temp_video_dir,
            mask_config={"mask_type": "bottom_half", "mask_ratio": 0.5},
        )

        frames = torch.randn(4, 3, 224, 224)
        masked_frames, mask = dataset.apply_mask(frames)

        assert masked_frames.shape == frames.shape
        assert mask.shape == (4, 224, 224)

        # Check bottom half is masked
        assert torch.all(masked_frames[:, :, 112:, :] == 0)
        assert torch.all(mask[:, 112:, :])  # All True
        assert torch.all(~mask[:, :112, :])  # All False

    def test_apply_mask_random(self, temp_video_dir):
        """Test random masking."""
        dataset = MovieChat1K(
            root=temp_video_dir,
            mask_config={"mask_type": "random", "mask_ratio": 0.3},
        )

        frames = torch.randn(4, 3, 224, 224)
        masked_frames, mask = dataset.apply_mask(frames)

        assert masked_frames.shape == frames.shape
        assert mask.shape == (4, 224, 224)

        # Check approximately 30% masked
        mask_ratio = mask.float().mean().item()
        assert TEST_MASK_LOWER_BOUND < mask_ratio < TEST_MASK_UPPER_BOUND

    def test_apply_mask_none(self, temp_video_dir):
        """Test no masking."""
        dataset = MovieChat1K(
            root=temp_video_dir,
            mask_config={"mask_type": "none"},
        )

        frames = torch.randn(4, 3, 224, 224)
        original_frames = frames.clone()
        masked_frames, mask = dataset.apply_mask(frames)

        assert torch.allclose(masked_frames, original_frames)
        assert torch.all(~mask)  # All False

    def test_compute_embeddings_mock(self, temp_video_dir):
        """Test embedding computation with mock model."""
        # Create mock embedding model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.return_value = torch.randn(8, 768)  # Batch of embeddings

        dataset = MovieChat1K(
            root=temp_video_dir,
            mask_config={"embedding_model": mock_model},
        )

        frames = torch.randn(8, 3, 224, 224)
        embeddings = dataset.compute_embeddings(frames)

        assert embeddings.shape == (8, 768)
        mock_model.assert_called_once()

    def test_compute_embeddings_error(self, temp_video_dir):
        """Test error when embedding model missing."""
        dataset = MovieChat1K(
            root=temp_video_dir,
        )

        frames = torch.randn(8, 3, 224, 224)

        with pytest.raises(ValueError, match="Embedding model required"):
            dataset.compute_embeddings(frames)

    def test_getitem_pixels(self, temp_video_dir, mock_video_reader):
        """Test __getitem__ for pixel-level experiments."""
        dataset = MovieChat1K(
            root=temp_video_dir,
            num_frames=8,
            resolution=224,
            mask_config={"mask_type": "bottom_half", "mask_ratio": 0.5},
        )

        sample = dataset[0]

        assert "frames" in sample
        assert "masked_frames" in sample
        assert "mask" in sample
        assert "video_path" in sample
        assert "video_name" in sample

        assert sample["frames"].shape == (8, 3, 224, 224)
        assert sample["masked_frames"].shape == (8, 3, 224, 224)
        assert sample["mask"].shape == (8, 224, 224)

        # Check bottom half is masked
        assert torch.all(sample["masked_frames"][:, :, 112:, :] == 0)

    def test_getitem_embeddings(self, temp_video_dir, mock_video_reader):
        """Test __getitem__ for embedding-level experiments."""
        # Mock embedding model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.return_value = torch.randn(8, 768)

        dataset = MovieChat1K(
            root=temp_video_dir,
            num_frames=8,
            resolution=224,
            mask_config={
                "return_embeddings": True,
                "embedding_model": mock_model,
                "noise_std": 0.1,
            },
        )

        # Set random seed for reproducibility
        torch.manual_seed(42)
        sample = dataset[0]

        assert sample["frames"].shape == (8, 768)
        assert sample["masked_frames"].shape == (8, 768)

        # Check noise was added
        diff = (sample["masked_frames"] - sample["frames"]).abs()
        assert diff.mean() > 0  # Should have noise

    def test_custom_transform(self, temp_video_dir, mock_video_reader):
        """Test with custom transform pipeline."""
        custom_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        dataset = MovieChat1K(
            root=temp_video_dir,
            num_frames=4,
            resolution=128,
            transform=custom_transform,
        )

        sample = dataset[0]
        assert sample["frames"].shape == (4, 3, 128, 128)

    def test_different_resolutions(self, temp_video_dir):
        """Test with different resolutions."""
        for resolution in [112, 224, 336]:
            dataset = MovieChat1K(
                root=temp_video_dir,
                resolution=resolution,
            )
            assert dataset.resolution == resolution

    def test_different_num_frames(self, temp_video_dir):
        """Test with different numbers of frames."""
        for num_frames in [8, 16, 32, 64]:
            dataset = MovieChat1K(
                root=temp_video_dir,
                num_frames=num_frames,
            )
            assert dataset.num_frames == num_frames

    def test_embedding_pooling(self, temp_video_dir):
        """Test handling of different embedding output formats."""
        dataset = MovieChat1K(
            root=temp_video_dir,
        )

        frames = torch.randn(8, 3, 224, 224)

        # Test with spatial features (need pooling)
        mock_model = MagicMock()
        # Use side_effect to return a new iterator each time
        mock_model.parameters.side_effect = lambda: iter([torch.tensor([1.0])])
        mock_model.return_value = torch.randn(8, 768, 7, 7)  # Spatial features
        dataset.embedding_model = mock_model

        embeddings = dataset.compute_embeddings(frames)
        assert embeddings.shape == (8, 768)  # Should be pooled

        # Test with tuple output
        mock_model.return_value = (torch.randn(8, 768), torch.randn(8, 10))
        embeddings = dataset.compute_embeddings(frames)
        assert embeddings.shape == (8, 768)  # Should take first element
