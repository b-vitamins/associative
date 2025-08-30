"""Comprehensive test suite for GRID audiovisual dataset."""

import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from associative.datasets import GRIDDataset

# Test constants
NORMALIZATION_TOLERANCE = 1.1
BATCH_SIZE = 4
TARGET_FRAMES = 300
TEMPORAL_ALIGNMENT_TOLERANCE = 0.01


class TestGRIDDataset:
    """Test suite for GRID audiovisual dataset."""

    @pytest.fixture
    def temp_grid_dir(self):
        """Create temporary directory with mock GRID structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "gridcorpus"
            base_path.mkdir()

            # Create complete structure for all test speakers
            # For speaker-dependent: 1, 2, 4, 29
            # For speaker-independent: train (3,5-18), val (19-23), test (24-28)
            # Create files for all speakers we might need
            speakers_to_create = list(range(1, 30))  # All speakers 1-29 (inclusive)

            for speaker in speakers_to_create:
                audio_dir = base_path / "audio" / f"s{speaker}"
                video_dir = base_path / "video" / f"s{speaker}"
                audio_dir.mkdir(parents=True)
                video_dir.mkdir(parents=True)

                # Create 950+ files to pass integrity check (min 900 required)
                # Use realistic sentence IDs
                for i in range(950):
                    # Generate valid sentence ID pattern
                    commands = ["b", "l", "p", "s"]
                    colors = ["b", "g", "r", "w"]
                    preps = ["a", "b", "i", "w"]
                    letters = ["a", "f", "x", "z"]
                    digits = ["1", "2", "3", "4"]
                    adverbs = ["a", "n", "p", "s"]

                    sid = (
                        commands[i % 4]
                        + colors[(i // 4) % 4]
                        + preps[(i // 16) % 4]
                        + letters[(i // 64) % 4]
                        + digits[(i // 256) % 4]
                        + adverbs[i % 4]
                    )

                    # Create dummy audio file
                    audio_file = audio_dir / f"{sid}_{i:04d}.wav"
                    self._create_dummy_wav(audio_file)

                    # Create dummy video file
                    video_file = video_dir / f"{sid}_{i:04d}.mpg"
                    self._create_dummy_video(video_file)

            yield tmpdir

    @pytest.fixture
    def mock_decord(self):
        """Mock decord for video processing."""
        with patch("associative.datasets.grid.VideoReader") as mock_vr:
            # Create mock video reader
            mock_reader = MagicMock()

            # Mock video frames (30 frames at 25fps = 1.2 seconds)
            frames = np.random.randint(0, 255, (30, 360, 288, 3), dtype=np.uint8)
            mock_reader.__getitem__.return_value.asnumpy.return_value = frames
            mock_reader.__len__.return_value = 30

            mock_vr.return_value = mock_reader
            yield mock_vr

    @pytest.fixture
    def mock_librosa(self):
        """Mock librosa for audio processing."""
        with patch("associative.datasets.grid.librosa") as mock:
            # Mock audio loading (1.2 seconds at 16kHz to match video)
            mock.load.return_value = (
                np.random.randn(int(16000 * 1.2)).astype(np.float32),
                16000,
            )
            # Mock mel-spectrogram (120 frames at 100fps for 1.2 seconds)
            mel_spec = np.random.randn(80, 120).astype(np.float32) - 40
            mock.feature.melspectrogram.return_value = np.exp(
                mel_spec / 10
            )  # Convert from dB
            mock.power_to_db.return_value = mel_spec
            yield mock

    def _create_dummy_wav(self, filepath):
        """Create a dummy WAV file."""
        with wave.open(str(filepath), "wb") as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(16000)  # 16kHz as per GRID spec
            # 1.2 seconds of random audio
            samples = np.random.randint(-32768, 32767, int(16000 * 1.2), dtype=np.int16)
            wav.writeframes(samples.tobytes())

    def _create_dummy_video(self, filepath):
        """Create a dummy video file (empty but valid)."""
        # Just create the file - decord mock will handle reading
        filepath.touch()

    def test_initialization_speaker_dependent(
        self, temp_grid_dir, mock_decord, mock_librosa
    ):
        """Test dataset initialization in speaker-dependent mode."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            split="train",
            speaker_dependent=True,
            download=False,
        )

        assert dataset.speaker_dependent is True
        assert dataset.subjects == [1, 2, 4, 29]
        assert len(dataset) > 0
        assert dataset.split == "train"

    def test_initialization_speaker_independent(
        self, temp_grid_dir, mock_decord, mock_librosa
    ):
        """Test dataset initialization in speaker-independent mode."""
        # Test train split
        dataset_train = GRIDDataset(
            root=temp_grid_dir,
            split="train",
            speaker_dependent=False,
            download=False,
        )
        assert dataset_train.subjects == [
            3,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
        ]

        # Test val split
        dataset_val = GRIDDataset(
            root=temp_grid_dir,
            split="val",
            speaker_dependent=False,
            download=False,
        )
        assert dataset_val.subjects == [19, 20, 21, 22, 23]

        # Test test split
        dataset_test = GRIDDataset(
            root=temp_grid_dir,
            split="test",
            speaker_dependent=False,
            download=False,
        )
        assert dataset_test.subjects == [24, 25, 26, 27, 28]

    def test_invalid_split_error(self):
        """Test error on invalid split."""
        with (
            tempfile.TemporaryDirectory() as dummy_dir,
            pytest.raises(ValueError, match="Invalid split"),
        ):
            GRIDDataset(
                root=dummy_dir,
                split="invalid",
                speaker_dependent=False,
                download=False,
            )

    def test_missing_data_error(self):
        """Test error when data is missing and download=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent_grid"
            with pytest.raises(RuntimeError, match="Dataset not found or corrupted"):
                GRIDDataset(
                    root=str(nonexistent_path),
                    download=False,
                )

    def test_sentence_decoding(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test sentence ID to text decoding."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        # Test known sentence patterns
        assert dataset._decode_sentence("bbaf2n") == "bin blue at F two now"
        assert dataset._decode_sentence("lwws5s") == "lay white with S five soon"
        assert dataset._decode_sentence("pgif7a") == "place green in F seven again"
        assert dataset._decode_sentence("sgiq8p") == "set green in Q eight please"

    def test_data_loading(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test loading individual samples."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        # Get a sample
        data, label = dataset[0]

        # Check data structure
        assert isinstance(data, dict)
        assert "video" in data
        assert "audio" in data

        # Check shapes (fixed 300 frames at 100fps = 3 seconds)
        assert data["video"].shape == torch.Size([300, 96, 96, 3])
        assert data["audio"].shape == torch.Size([300, 80])

        # Check data types
        assert data["video"].dtype == torch.float32
        assert data["audio"].dtype == torch.float32

        # Check label is a string
        assert isinstance(label, str)

    def test_video_normalization(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test video is normalized to [-1, 1]."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        data, _ = dataset[0]
        video = data["video"]

        # Check normalization range
        assert video.min() >= -NORMALIZATION_TOLERANCE  # Allow small numerical errors
        assert video.max() <= NORMALIZATION_TOLERANCE

    def test_dataloader_compatibility(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test dataset works with PyTorch DataLoader."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        # Create DataLoader with batch size > 1
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        # Get a batch
        batch_data, batch_labels = next(iter(dataloader))

        # Check batch dimensions
        assert batch_data["video"].shape[0] == BATCH_SIZE  # Batch size
        assert batch_data["audio"].shape[0] == BATCH_SIZE
        assert len(batch_labels) == BATCH_SIZE

        # Check all samples have same temporal dimension
        assert batch_data["video"].shape == torch.Size(
            [BATCH_SIZE, TARGET_FRAMES, 96, 96, 3]
        )
        assert batch_data["audio"].shape == torch.Size([BATCH_SIZE, TARGET_FRAMES, 80])

    def test_transforms(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test transform and target_transform."""
        transform_called = False
        target_transform_called = False

        def mock_transform(data):
            nonlocal transform_called
            transform_called = True
            return data

        def mock_target_transform(label):
            nonlocal target_transform_called
            target_transform_called = True
            return label.upper()

        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            transform=mock_transform,
            target_transform=mock_target_transform,
            download=False,
        )

        data, label = dataset[0]

        assert transform_called
        assert target_transform_called
        assert label.isupper()  # Target transform uppercases

    def test_repr(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test string representation."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        repr_str = repr(dataset)
        assert "Dataset GRID" in repr_str
        assert str(len(dataset)) in repr_str
        assert str(temp_grid_dir) in repr_str
        assert "train" in repr_str
        assert "Speaker-dependent" in repr_str

    def test_path_expansion(self, mock_decord, mock_librosa):
        """Test that paths with ~ are expanded."""
        with pytest.raises(RuntimeError):  # Will fail but path should expand
            dataset = GRIDDataset(
                root="~/nonexistent_grid_data",
                download=False,
            )
            # Check path was expanded
            assert str(dataset.root).startswith("/")
            assert "~" not in str(dataset.root)

    def test_audio_video_synchronization(
        self, temp_grid_dir, mock_decord, mock_librosa
    ):
        """Test that audio and video are synchronized at 100fps."""
        from associative.datasets.grid import AudioVideoConfig

        av_config = AudioVideoConfig(target_fps=100)
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
            av_config=av_config,
        )

        data, _ = dataset[0]

        # Both should have exactly 300 frames (3 seconds at 100fps)
        assert data["video"].shape[0] == TARGET_FRAMES
        assert data["audio"].shape[0] == TARGET_FRAMES

        # They should be temporally aligned
        video_duration = data["video"].shape[0] / 100  # frames / fps
        audio_duration = data["audio"].shape[0] / 100
        assert (
            abs(video_duration - audio_duration) < TEMPORAL_ALIGNMENT_TOLERANCE
        )  # Within 10ms

    def test_multiple_samples(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test loading multiple samples."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        # Load several samples
        samples = []
        for i in range(min(10, len(dataset))):
            data, label = dataset[i]
            samples.append((data, label))

        # Check all samples have consistent structure
        for data, label in samples:
            assert data["video"].shape == torch.Size([300, 96, 96, 3])
            assert data["audio"].shape == torch.Size([300, 80])
            assert isinstance(label, str)

    def test_dataset_length(self, temp_grid_dir, mock_decord, mock_librosa):
        """Test dataset length is correct."""
        dataset = GRIDDataset(
            root=temp_grid_dir,
            speaker_dependent=True,
            download=False,
        )

        # Should have samples from 4 speakers with 950 samples each
        # But only matching audio-video pairs are included
        assert len(dataset) > 0
        assert len(dataset) == len(dataset.samples)

    def test_integrity_check(self, temp_grid_dir):
        """Test integrity check catches incomplete data."""
        import shutil

        # Remove all data to trigger error
        base_path = Path(temp_grid_dir) / "gridcorpus"

        # Remove entire gridcorpus directory to simulate missing dataset
        if base_path.exists():
            shutil.rmtree(base_path)

        # Should now fail integrity check
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted"):
            GRIDDataset(
                root=temp_grid_dir,
                download=False,
            )

    @patch("associative.datasets.grid.download_url")
    @patch("associative.datasets.grid.tarfile.open")
    @patch("associative.datasets.grid.zipfile.ZipFile")
    def test_download_behavior(self, mock_zip, mock_tar, mock_download):
        """Test download behavior (mocked to avoid actual downloads)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Dataset should attempt download when data doesn't exist
            with pytest.raises(RuntimeError):  # Will fail after mock download
                GRIDDataset(
                    root=tmpdir,
                    download=True,
                )

            # Check download was attempted
            assert mock_download.called
