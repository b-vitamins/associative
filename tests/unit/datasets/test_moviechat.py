"""Tests for MovieChat1K dataset."""

import logging
import os
import tempfile
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from _pytest.logging import LogCaptureFixture

from associative.datasets.moviechat import MovieChat1K


class TestMovieChat1K:
    """Test suite for MovieChat1K dataset."""

    @pytest.fixture
    def temp_cache_dir(self) -> Generator[str, None, None]:
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture(params=["train", "test"])
    def mock_video_dir(self, temp_cache_dir: str, request: Any) -> tuple[str, str]:
        """Create a mock video directory with test files."""
        split = request.param
        video_dir = Path(temp_cache_dir) / "moviechat" / split / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            (video_dir / f"video_{i}.mp4").touch()

        return temp_cache_dir, split

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_initialization_default_cache(
        self, split: str, temp_cache_dir: str
    ) -> None:
        """Test dataset initialization with default XDG cache."""
        mock_xdg = str(Path(temp_cache_dir) / "test_cache")
        with patch.dict(os.environ, {"XDG_CACHE_HOME": mock_xdg}):
            cache_dir = Path(f"{mock_xdg}/associative/moviechat/{split}")
            video_dir = cache_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)

            (video_dir / "test.mp4").touch()

            try:
                dataset = MovieChat1K(split=split, download=False)  # type: ignore[arg-type]
                assert dataset.root == Path(f"{mock_xdg}/associative/moviechat/{split}")
            finally:
                import shutil

                if cache_dir.parent.exists():
                    shutil.rmtree(cache_dir.parent)

    def test_initialization_custom_root(self, mock_video_dir: tuple[str, str]) -> None:
        """Test dataset initialization with custom root."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(root=mock_dir, split=split, download=False)  # type: ignore[arg-type]
        assert dataset.root == Path(mock_dir) / "moviechat" / split
        assert dataset.split == split
        assert len(dataset) == 3

    def test_initialization_parameters(self, mock_video_dir: tuple[str, str]) -> None:
        """Test all initialization parameters."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=256,
            resolution=336,
            download=False,
            max_videos=2,
        )

        assert dataset.num_frames == 256
        assert dataset.resolution == 336
        assert dataset.max_videos == 2
        assert len(dataset) == 2

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_no_download_empty_cache(self, temp_cache_dir: str, split: str) -> None:
        """Test error when cache is empty and download=False."""
        with pytest.raises(RuntimeError, match="Video directory .* does not exist"):
            MovieChat1K(
                root=temp_cache_dir,
                split=split,  # type: ignore[arg-type]
                download=False,
            )

    def test_invalid_split(self, mock_video_dir: tuple[str, str]) -> None:
        """Test error with invalid split."""
        mock_dir, _ = mock_video_dir
        with pytest.raises(ValueError, match="Invalid split"):
            MovieChat1K(
                root=mock_dir,
                split="validation",  # type: ignore[arg-type]
                download=False,
            )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_getitem_structure(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test __getitem__ returns correct structure."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 1000
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.random.randint(  # type: ignore[attr-defined]
            0, 255, (224, 224, 3), dtype=np.uint8
        )
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=8,
            resolution=224,
            download=False,
        )

        sample = dataset[0]

        assert "frames" in sample
        assert "video_id" in sample

        assert sample["frames"].shape == (8, 3, 224, 224)
        assert isinstance(sample["video_id"], str)

    @patch("associative.datasets.moviechat.VideoReader")
    def test_frame_sampling(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test uniform frame sampling."""
        mock_dir, split = mock_video_dir
        accessed_indices: list[int] = []

        def getitem_side_effect(idx: int) -> Any:
            accessed_indices.append(idx)
            frame_mock = MagicMock()
            frame_mock.asnumpy.return_value = np.random.randint(  # type: ignore[attr-defined]
                0, 255, (224, 224, 3), dtype=np.uint8
            )
            return frame_mock

        reader = MagicMock()
        reader.__len__.return_value = 1000
        reader.__getitem__.side_effect = getitem_side_effect
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=64,
            download=False,
        )

        _ = dataset[0]

        assert len(accessed_indices) == 64
        assert min(accessed_indices) >= 0
        assert max(accessed_indices) < 1000

    def test_repr(self, mock_video_dir: tuple[str, str]) -> None:
        """Test string representation."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            download=False,
        )

        repr_str = repr(dataset)
        assert "MovieChat1K" in repr_str
        assert split in repr_str
        assert "num_videos=3" in repr_str

    @pytest.mark.parametrize("num_frames", [64, 128, 256, 512])
    def test_various_frame_counts(
        self, mock_video_dir: tuple[str, str], num_frames: int
    ) -> None:
        """Test with different frame counts."""
        mock_dir, split = mock_video_dir
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=num_frames,
            download=False,
        )

        assert dataset.num_frames == num_frames

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_download_with_token(self, temp_cache_dir: str, split: str) -> None:
        """Test download functionality with HF token."""
        with (
            patch.dict(os.environ, {"HF_TOKEN": "test_token"}),
            patch("associative.datasets.moviechat.list_repo_files") as mock_list,
            patch("associative.datasets.moviechat.hf_hub_download") as mock_download,
        ):
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

            def download_side_effect(repo_id: str, filename: str, **kwargs: Any) -> str:
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
                split=split,  # type: ignore[arg-type]
                download=True,
                download_features=True,
            )

            mock_list.assert_called_once()
            assert mock_download.call_count == len(mock_list.return_value)

            assert len(list(dataset.video_dir.glob("*.mp4"))) > 0
            assert len(list(dataset.metadata_dir.glob("*.json"))) > 0
            if split == "train":
                assert len(list(dataset.features_dir.glob("*"))) > 0
            else:
                assert len(list(dataset.features_dir.glob("*"))) == 0

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_no_token_error(self, temp_cache_dir: str, split: str) -> None:
        """Test error when HF_TOKEN is missing."""
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
                split=split,  # type: ignore[arg-type]
                download=True,
            )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_video_loading_error_fallback(
        self,
        mock_reader: MagicMock,
        mock_video_dir: tuple[str, str],
        caplog: LogCaptureFixture,
    ) -> None:
        """Test handling of video loading errors with fallback frames."""
        mock_dir, split = mock_video_dir

        reader = MagicMock()
        reader.__len__.return_value = 100

        call_count = [0]

        def frame_side_effect(idx: int) -> Any:
            call_count[0] += 1
            if call_count[0] in [3, 5]:
                raise Exception("Frame decode error")
            frame_mock = MagicMock()
            frame_mock.asnumpy.return_value = (
                np.ones((336, 336, 3), dtype=np.uint8) * 100
            )
            return frame_mock

        reader.__getitem__.side_effect = frame_side_effect
        mock_reader.return_value = reader

        custom_resolution = 336
        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=8,
            resolution=custom_resolution,
            download=False,
        )

        with patch("builtins.print") as mock_print:
            with caplog.at_level(logging.WARNING):
                sample = dataset[0]

                warning_prints = [
                    call
                    for call in mock_print.call_args_list
                    if "Warning" in str(call) or "warning" in str(call).lower()
                ]
                assert len(warning_prints) == 0, (
                    "Should use logging instead of print for warnings"
                )

                assert len(caplog.records) > 0, "Should log warnings for failed frames"
                assert any(
                    "Failed to load frame" in record.message
                    for record in caplog.records
                )

            assert sample["frames"].shape == (
                8,
                3,
                custom_resolution,
                custom_resolution,
            ), (
                f"Fallback frames should use self.resolution={custom_resolution}, not hardcoded dimensions"
            )

    def test_no_global_warning_suppression(self) -> None:
        """Test that importing MovieChat1K doesn't suppress warnings globally."""
        warnings.resetwarnings()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("Test warning before import", UserWarning, stacklevel=2)
            assert len(w) == 1

        from associative.datasets import (
            moviechat,  # type: ignore[import,unused-import] # noqa: F401
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("Test warning after import", UserWarning, stacklevel=2)
            assert len(w) == 1, "MovieChat1K should not suppress warnings globally"

    @patch("associative.datasets.moviechat.VideoReader")
    def test_uses_logging_not_print(
        self,
        mock_reader: MagicMock,
        mock_video_dir: tuple[str, str],
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that the dataset uses proper logging instead of print statements."""
        mock_dir, split = mock_video_dir

        reader = MagicMock()
        reader.__len__.return_value = 10

        call_count = [0]

        def frame_side_effect(idx: int) -> Any:
            call_count[0] += 1
            if call_count[0] == 3:
                raise Exception("Frame decode error")
            frame_mock = MagicMock()
            frame_mock.asnumpy.return_value = (
                np.ones((224, 224, 3), dtype=np.uint8) * 100
            )
            return frame_mock

        reader.__getitem__.side_effect = frame_side_effect
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=5,
            download=False,
        )

        with caplog.at_level(logging.WARNING), patch("builtins.print") as mock_print:
            _ = dataset[0]

            warning_prints = [
                call
                for call in mock_print.call_args_list
                if any(
                    word in str(call).lower() for word in ["warning", "error", "failed"]
                )
            ]
            assert len(warning_prints) == 0, (
                "Should use logging instead of print for warnings"
            )

        assert len(caplog.records) > 0, "Should log frame loading failures"
        assert any("frame" in record.message.lower() for record in caplog.records), (
            "Log should mention frame loading issue"
        )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_frames_returned(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test that frames are properly returned."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 100
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.ones((224, 224, 3), dtype=np.uint8) * 128
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=8,
            download=False,
        )

        sample = dataset[0]

        assert "frames" in sample
        assert sample["frames"].shape == (8, 3, 224, 224)

        assert 0 <= sample["frames"].min() <= 1
        assert 0 <= sample["frames"].max() <= 1

    @patch("associative.datasets.moviechat.VideoReader")
    def test_transform_applied(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test that transforms are properly applied."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 100
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.random.randint(  # type: ignore[attr-defined]
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=4,
            resolution=112,
            download=False,
        )

        sample = dataset[0]

        assert sample["frames"].shape == (4, 3, 112, 112)

        assert 0 <= sample["frames"].min() <= 1
        assert 0 <= sample["frames"].max() <= 1

    @patch("associative.datasets.moviechat.VideoReader")
    def test_frame_type_consistency(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test that frame output type is consistent regardless of transform."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 100
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.ones((224, 224, 3), dtype=np.uint8) * 128
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        dataset1 = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=4,
            download=False,
        )
        sample1 = dataset1[0]

        def custom_transform(x: Any) -> Any:
            return np.array(x, dtype=np.float32) / 255.0

        dataset2 = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=4,
            transform=custom_transform,
            download=False,
        )
        sample2 = dataset2[0]

        assert isinstance(sample1["frames"], torch.Tensor), (
            "Should always return torch.Tensor"
        )
        assert isinstance(sample2["frames"], torch.Tensor), (
            "Should always return torch.Tensor even with numpy transform"
        )

    def test_empty_dataset_validation(self, temp_cache_dir: str) -> None:
        """Test that empty dataset (no videos) raises informative error."""
        split = "train"
        video_dir = Path(temp_cache_dir) / "moviechat" / split / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RuntimeError, match="No videos found"):
            MovieChat1K(
                root=temp_cache_dir,
                split=split,
                download=False,
            )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_deterministic_sampling(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test that frame sampling can be deterministic with a seed."""
        mock_dir, split = mock_video_dir

        reader = MagicMock()
        reader.__len__.return_value = 1000

        accessed_indices: list[int] = []

        def track_access(idx: int) -> Any:
            accessed_indices.append(idx)
            frame_mock = MagicMock()
            frame_mock.asnumpy.return_value = np.ones((224, 224, 3), dtype=np.uint8)
            return frame_mock

        reader.__getitem__.side_effect = track_access
        mock_reader.return_value = reader

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=16,
            download=False,
            seed=42,
        )

        accessed_indices.clear()
        _ = dataset[0]
        first_indices: list[int] = accessed_indices.copy()

        accessed_indices.clear()
        _ = dataset[0]
        second_indices: list[int] = accessed_indices.copy()

        assert first_indices == second_indices, (
            "Frame sampling should be deterministic with seed"
        )

    @patch("associative.datasets.moviechat.VideoReader")
    def test_transform_validation(
        self, mock_reader: MagicMock, mock_video_dir: tuple[str, str]
    ) -> None:
        """Test that transform output is validated."""
        mock_dir, split = mock_video_dir
        reader = MagicMock()
        reader.__len__.return_value = 100
        frame_mock = MagicMock()
        frame_mock.asnumpy.return_value = np.ones((224, 224, 3), dtype=np.uint8) * 128
        reader.__getitem__.return_value = frame_mock
        mock_reader.return_value = reader

        def bad_transform(x: Any) -> str:
            return "not_a_tensor"

        dataset = MovieChat1K(
            root=mock_dir,
            split=split,  # type: ignore[arg-type]
            num_frames=4,
            transform=bad_transform,
            download=False,
        )

        with pytest.raises((TypeError, ValueError), match="Transform must return"):
            _ = dataset[0]

    @patch("associative.datasets.moviechat.VideoReader")
    def test_video_integrity_check(
        self,
        mock_reader: MagicMock,
        mock_video_dir: tuple[str, str],
        caplog: LogCaptureFixture,
    ) -> None:
        """Test that corrupted videos are detected and handled."""
        mock_dir, split = mock_video_dir

        mock_reader.side_effect = Exception("Corrupted video file")

        with (
            pytest.raises(
                RuntimeError,
                match="No valid videos found.*All videos appear to be corrupted",
            ),
            caplog.at_level(logging.WARNING),
        ):
            _ = MovieChat1K(
                root=mock_dir,
                split=split,  # type: ignore[arg-type]
                download=False,
                validate_videos=True,
            )

        assert len(caplog.records) > 0
        assert any(
            "corrupted" in record.message.lower() or "invalid" in record.message.lower()
            for record in caplog.records
        )
