"""Tests for MovieChat1K dataset."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from associative.datasets.moviechat import MovieChat1K


class TestMovieChat1K:
    """Test suite for MovieChat1K dataset."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture(params=["train", "test"])
    def mock_video_dir(self, temp_cache_dir, request):
        """Create a mock video directory with test files."""
        split = request.param
        video_dir = Path(temp_cache_dir) / "moviechat" / split / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Create some dummy video files
        for i in range(3):
            (video_dir / f"video_{i}.mp4").touch()

        return temp_cache_dir, split

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_initialization_default_cache(self, split, temp_cache_dir):
        """Test dataset initialization with default XDG cache."""
        mock_xdg = str(Path(temp_cache_dir) / "test_cache")
        with patch.dict(os.environ, {"XDG_CACHE_HOME": mock_xdg}):
            # Create required directories for download=False
            cache_dir = Path(f"{mock_xdg}/associative/moviechat/{split}")
            video_dir = cache_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Create a dummy video file
            (video_dir / "test.mp4").touch()

            try:
                dataset = MovieChat1K(split=split, download=False)
                assert dataset.root == Path(f"{mock_xdg}/associative/moviechat/{split}")
            finally:
                # Cleanup
                import shutil

                if cache_dir.parent.exists():
                    shutil.rmtree(cache_dir.parent)

    def test_initialization_custom_root(self, mock_video_dir):
        """Test dataset initialization with custom root."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(root=mock_dir, split=split, download=False)
        assert dataset.root == Path(mock_dir) / "moviechat" / split
        assert dataset.split == split
        assert len(dataset) == 3  # 3 mock videos

    def test_initialization_parameters(self, mock_video_dir):
        """Test all initialization parameters."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            num_frames=256,
            resolution=336,
            download=False,
            max_videos=2,
        )

        assert dataset.num_frames == 256
        assert dataset.resolution == 336
        assert dataset.max_videos == 2
        assert len(dataset) == 2  # Limited to 2 videos

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_no_download_empty_cache(self, temp_cache_dir, split):
        """Test error when cache is empty and download=False."""
        with pytest.raises(RuntimeError, match="Video directory .* does not exist"):
            MovieChat1K(
                root=temp_cache_dir,
                split=split,
                download=False,
            )

    def test_invalid_split(self, mock_video_dir):
        """Test error with invalid split."""
        mock_dir, _ = mock_video_dir
        with pytest.raises(ValueError, match="Invalid split"):
            MovieChat1K(
                root=mock_dir,
                split="validation",  # pyright: ignore[reportArgumentType] # Invalid
                download=False,
            )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_getitem_structure(self, mock_reader, mock_video_dir):
        """Test __getitem__ returns correct structure."""
        mock_dir, split = mock_video_dir
        # Mock VideoReader
        reader = MagicMock()
        reader.__len__.return_value = 1000
        # Mock individual frame access
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.random.randint(
            0, 255, (224, 224, 3), dtype=np.uint8
        )
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            num_frames=8,
            resolution=224,
            download=False,
        )

        sample = dataset[0]

        # Check required keys
        assert "frames" in sample
        assert "video_id" in sample

        # Check shapes
        assert sample["frames"].shape == (8, 3, 224, 224)
        assert isinstance(sample["video_id"], str)

    @patch("associative.datasets.moviechat.VideoReader")
    def test_frame_sampling(self, mock_reader, mock_video_dir):
        """Test uniform frame sampling."""
        mock_dir, split = mock_video_dir
        # Track which indices are accessed
        accessed_indices = []

        def getitem_side_effect(idx):
            accessed_indices.append(idx)
            frame_mock = MagicMock()
            frame_mock.asnumpy.return_value = np.random.randint(
                0, 255, (224, 224, 3), dtype=np.uint8
            )
            return frame_mock

        reader = MagicMock()
        reader.__len__.return_value = 1000
        reader.__getitem__.side_effect = getitem_side_effect
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            num_frames=64,
            download=False,
        )

        _ = dataset[0]

        # Check that 64 frames were sampled
        assert len(accessed_indices) == 64
        # Check roughly uniform distribution
        assert min(accessed_indices) >= 0
        assert max(accessed_indices) < 1000

    def test_repr(self, mock_video_dir):
        """Test string representation."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            download=False,
        )

        repr_str = repr(dataset)
        assert "MovieChat1K" in repr_str
        assert split in repr_str
        assert "num_videos=3" in repr_str

    @pytest.mark.parametrize("num_frames", [64, 128, 256, 512])
    def test_various_frame_counts(self, mock_video_dir, num_frames):
        """Test with different frame counts."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            num_frames=num_frames,
            download=False,
        )

        assert dataset.num_frames == num_frames

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_download_with_token(self, temp_cache_dir, split):
        """Test download functionality with HF token."""
        with (
            patch.dict(os.environ, {"HF_TOKEN": "test_token"}),
            patch("associative.datasets.moviechat.list_repo_files") as mock_list,
            patch("associative.datasets.moviechat.hf_hub_download") as mock_download,
        ):
            # Mock repo files based on split
            if split == "train":
                mock_list.return_value = [
                    "raw_videos/video_0.mp4",
                    "jsons/video_0.json",
                    "movies/video_0.pt",
                ]
            else:
                mock_list.return_value = [
                    "videos/1.mp4",
                    "gt/1.json",
                    "annotations/2.json",
                ]

            # Mock download to create actual file in appropriate dir
            def download_side_effect(repo_id, filename, **kwargs):
                root = Path(temp_cache_dir) / "moviechat" / split
                if filename.endswith(".mp4"):
                    dir_path = root / "videos"
                elif filename.endswith(".json"):
                    dir_path = root / "metadata"
                else:
                    dir_path = root / "features"
                dir_path.mkdir(parents=True, exist_ok=True)
                tmp_file = dir_path / Path(filename).name
                tmp_file.touch()
                return str(tmp_file)

            mock_download.side_effect = download_side_effect

            dataset = MovieChat1K(
                root=temp_cache_dir,
                split=split,
                download=True,
                download_features=True,
            )

            # Check that download was called
            mock_list.assert_called_once()
            assert mock_download.call_count == len(mock_list.return_value)

            # Check directories have files
            assert len(list(dataset.video_dir.glob("*.mp4"))) > 0
            assert len(list(dataset.metadata_dir.glob("*.json"))) > 0
            if split == "train":
                assert len(list(dataset.features_dir.glob("*"))) > 0
            else:
                assert len(list(dataset.features_dir.glob("*"))) == 0

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_no_token_error(self, temp_cache_dir, split):
        """Test error when HF_TOKEN is missing."""
        # Remove HF_TOKEN if it exists
        env_copy = os.environ.copy()
        if "HF_TOKEN" in env_copy:
            del env_copy["HF_TOKEN"]

        with (
            patch.dict(os.environ, env_copy, clear=True),
            pytest.raises(
                ValueError, match="HF_TOKEN environment variable is required"
            ),
        ):
            MovieChat1K(
                root=temp_cache_dir,
                split=split,
                download=True,
            )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_video_loading_error(self, mock_reader, mock_video_dir):
        """Test handling of video loading errors."""
        mock_dir, split = mock_video_dir
        # Mock VideoReader to raise an error
        mock_reader.side_effect = Exception("Cannot decode video")

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            download=False,
        )

        with pytest.raises(Exception, match="Cannot decode video"):
            _ = dataset[0]

    @patch("associative.datasets.moviechat.VideoReader")
    def test_frames_returned(self, mock_reader, mock_video_dir):
        """Test that frames are properly returned."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 100
        # Mock individual frame access with consistent values
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.ones((224, 224, 3), dtype=np.uint8) * 128
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            num_frames=8,
            download=False,
        )

        sample = dataset[0]

        # Check that frames are returned
        assert "frames" in sample
        assert sample["frames"].shape == (8, 3, 224, 224)

        # Check that frames have been normalized (not raw pixel values)
        # Values of 128 after standard ImageNet normalization should be around 0
        # Just check that they're in a reasonable range
        assert -3 < sample["frames"].min() < 3
        assert -3 < sample["frames"].max() < 3

    @patch("associative.datasets.moviechat.VideoReader")
    def test_transform_applied(self, mock_reader, mock_video_dir):
        """Test that transforms are properly applied."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 100
        # Mock individual frame access
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,
            num_frames=4,
            resolution=112,
            download=False,
        )

        sample = dataset[0]

        # Check resolution was applied
        assert sample["frames"].shape == (4, 3, 112, 112)

        # Check normalization was applied (should be roughly in [-3, 3] after standard normalization)
        assert -4 < sample["frames"].min() < 0
        assert 0 < sample["frames"].max() < 4
