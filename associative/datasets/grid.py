"""GRID Audiovisual Speech Dataset.

The GRID corpus is a large multitalker audiovisual sentence corpus
designed for research on speech perception. It consists of 34 speakers
(18 male, 16 female) each speaking 1000 sentences.

Reference:
    Cooke, M., Barker, J., Cunningham, S., & Shao, X. (2006).
    An audio-visual corpus for speech perception and automatic speech recognition.
    The Journal of the Acoustical Society of America, 120(5), 2421-2424.
"""

import os
import shutil
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

# Import dependencies
import librosa
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


@dataclass
class AudioVideoConfig:
    """Configuration for audio and video processing parameters."""

    video_fps: int = 25
    audio_sr: int = 16000
    video_size: tuple[int, int] = (96, 96)
    n_mels: int = 80
    hop_length: int = 160
    win_length: int = 800
    target_fps: int = 100


def safe_extract(archive, path: Path, members=None) -> None:
    """Safely extract archive to prevent path traversal attacks."""

    def is_safe_path(filepath: Path, basepath: Path) -> bool:
        try:
            # Resolve any relative paths and ensure it's within basepath
            resolved = (basepath / filepath).resolve()
            return resolved.is_relative_to(basepath.resolve())
        except (OSError, ValueError):
            return False

    if hasattr(archive, "getnames"):  # tarfile
        for member_name in archive.getnames():
            if not is_safe_path(Path(member_name), path):
                raise ValueError(f"Unsafe path in archive: {member_name}")
    elif hasattr(archive, "namelist"):  # zipfile
        for member_name in archive.namelist():
            if not is_safe_path(Path(member_name), path):
                raise ValueError(f"Unsafe path in archive: {member_name}")

    # If we get here, all paths are safe
    archive.extractall(path)  # noqa: S202


def download_url(url: str, root: str, filename: str) -> None:
    """Download URL to file."""
    import urllib.request
    from urllib.parse import urlparse

    # Validate URL scheme for security
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    fpath = root_path / filename

    # Check if already downloaded
    if fpath.exists():
        return  # Silently use cached file

    try:
        with urllib.request.urlopen(url) as response, open(fpath, "wb") as out_file:  # noqa: S310
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
    except Exception as e:
        if fpath.exists():
            os.remove(fpath)
        raise RuntimeError(f"Error downloading {url}: {e}") from e


class GRIDDataset(Dataset):
    """GRID Audiovisual Speech Dataset.

    Args:
        root: Root directory where dataset exists or will be saved
        split: Dataset split ('train', 'val', 'test')
        speaker_dependent: If True, use overlapped speakers (1,2,4,29),
                          otherwise use speaker-independent splits
        transform: A function/transform for the audio-video data
        target_transform: A function/transform for the target label
        download: If true, downloads the dataset from the internet
        av_config: Audio/video configuration (AudioVideoConfig instance).
                  If None, uses default configuration.
    """

    base_folder = "gridcorpus"
    base_url = "https://spandh.dcs.shef.ac.uk/gridcorpus"

    # Speaker-dependent subjects
    speaker_dependent_subjects: ClassVar[list[int]] = [1, 2, 4, 29]

    # Speaker-independent splits
    speaker_independent_splits: ClassVar[dict[str, list[int]]] = {
        "train": [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "val": [19, 20, 21, 22, 23],
        "test": [24, 25, 26, 27, 28],
    }

    def __init__(  # noqa: PLR0913
        self,
        root: str | Path,
        split: str = "train",
        speaker_dependent: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
        av_config: AudioVideoConfig | None = None,
    ) -> None:
        self.root = Path(os.path.expanduser(root))  # Handle ~ and env vars
        self.split = split
        self.speaker_dependent = speaker_dependent
        self.transform = transform
        self.target_transform = target_transform

        # Audio/video parameters
        if av_config is None:
            av_config = AudioVideoConfig()
        self.video_fps = av_config.video_fps
        self.audio_sr = av_config.audio_sr
        self.video_size = av_config.video_size
        self.n_mels = av_config.n_mels
        self.hop_length = av_config.hop_length
        self.win_length = av_config.win_length
        self.target_fps = av_config.target_fps

        # Determine subjects
        if speaker_dependent:
            self.subjects = self.speaker_dependent_subjects
        else:
            if split not in self.speaker_independent_splits:
                raise ValueError(f"Invalid split: {split}")
            self.subjects = self.speaker_independent_splits[split]

        # Check data integrity
        if not self._check_complete():
            if download:
                self._download_all()
                if not self._check_complete():
                    raise RuntimeError("Download failed. Please try again.")
            else:
                raise RuntimeError(
                    "Dataset not found or corrupted. "
                    "You can use download=True to download it"
                )

        # Load the dataset
        self._load_data()

    def _check_complete(self) -> bool:
        """Check if dataset is complete with ALL required data."""
        data_path = self.root / self.base_folder

        if not data_path.exists():
            return False

        # Track available subjects
        available_subjects = []

        for subject in self.subjects:
            audio_dir = data_path / "audio" / f"s{subject}"
            video_dir = data_path / "video" / f"s{subject}"

            if not audio_dir.exists() or not video_dir.exists():
                continue  # Skip missing subjects

            # Check if directories have any files (don't require a minimum)
            # Some subjects like s3 have fewer files on the server
            has_audio = any(audio_dir.glob("*.wav"))
            has_video = any(video_dir.glob("*.mpg"))

            if has_audio and has_video:
                available_subjects.append(subject)

        # Update subjects to only include available ones
        if available_subjects:
            self.subjects = available_subjects
            return True  # We have at least some subjects

        return False  # No subjects available

    def _download_all(self) -> None:
        """Download ALL data for the required subjects."""
        data_path = self.root / self.base_folder

        # Create directories if they don't exist
        data_path.mkdir(parents=True, exist_ok=True)
        (data_path / "raw").mkdir(exist_ok=True)
        (data_path / "raw" / "audio").mkdir(exist_ok=True)
        (data_path / "raw" / "video").mkdir(exist_ok=True)
        (data_path / "audio").mkdir(exist_ok=True)
        (data_path / "video").mkdir(exist_ok=True)

        # Track failed subjects for cleanup
        failed_subjects = []

        # Download all subjects
        for subject in self.subjects:
            # Check if this subject already exists completely
            subject_audio_dir = data_path / "audio" / f"s{subject}"
            subject_video_dir = data_path / "video" / f"s{subject}"

            # Skip if already downloaded and extracted (any non-empty directory is considered complete)
            if (
                subject_audio_dir.exists()
                and subject_video_dir.exists()
                and any(subject_audio_dir.glob("*.wav"))
                and any(subject_video_dir.glob("*.mpg"))
            ):
                continue

            try:
                # Download and extract audio
                audio_url = f"{self.base_url}/s{subject}/audio/s{subject}.tar"
                audio_tar = data_path / "raw" / "audio" / f"s{subject}.tar"

                # Only download if not cached
                if not audio_tar.exists():
                    download_url(audio_url, str(audio_tar.parent), audio_tar.name)

                with tarfile.open(audio_tar, "r") as tar:
                    safe_extract(tar, data_path / "audio")

                # Download and extract video
                video_url = f"{self.base_url}/s{subject}/video/s{subject}.mpg_vcd.zip"
                video_zip = data_path / "raw" / "video" / f"s{subject}.zip"

                # Only download if not cached
                if not video_zip.exists():
                    download_url(video_url, str(video_zip.parent), video_zip.name)

                with zipfile.ZipFile(video_zip, "r") as zf:
                    safe_extract(zf, data_path / "video")

            except Exception as e:
                # Check if it's a 404 error (subject not available on server)
                if "404" in str(e) or "Not Found" in str(e):
                    import warnings

                    warnings.warn(
                        f"Subject {subject} not available on server (404). Skipping...",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Remove this subject from our list
                    self.subjects = [s for s in self.subjects if s != subject]
                else:
                    # Other errors - track as failed
                    failed_subjects.append(subject)

                # Clean up this subject's partial data
                if subject_audio_dir.exists():
                    shutil.rmtree(subject_audio_dir)
                if subject_video_dir.exists():
                    shutil.rmtree(subject_video_dir)
                # Don't delete the tar/zip files - they might be valid

        # If any subjects failed, raise error
        if failed_subjects:
            raise RuntimeError(
                f"Download failed for subjects: {failed_subjects}. Please try again."
            )

    def _load_data(self) -> None:
        """Load dataset samples."""
        self.samples = []
        self.labels = []

        data_path = self.root / self.base_folder

        for subject in self.subjects:
            audio_dir = data_path / "audio" / f"s{subject}"
            video_dir = data_path / "video" / f"s{subject}"

            # Get all video files
            video_files = sorted(video_dir.glob("*.mpg"))

            for video_file in video_files:
                audio_file = audio_dir / f"{video_file.stem}.wav"

                # Only include if both files exist
                if audio_file.exists():
                    sentence_id = video_file.stem
                    sentence = self._decode_sentence(sentence_id)

                    self.samples.append(
                        {
                            "audio_path": str(audio_file),
                            "video_path": str(video_file),
                            "sentence_id": sentence_id,
                            "subject": subject,
                        }
                    )
                    self.labels.append(sentence)

        # Dataset loaded successfully

    def _decode_sentence(self, sentence_id: str) -> str:
        """Decode sentence from ID."""
        # GRID sentence structure: command color preposition letter digit adverb
        # Example: "bin blue at F 2 now"

        commands = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
        colors = {"b": "blue", "g": "green", "r": "red", "w": "white"}
        prepositions = {"a": "at", "b": "by", "i": "in", "w": "with"}
        letters = {
            "a": "A",
            "b": "B",
            "c": "C",
            "d": "D",
            "e": "E",
            "f": "F",
            "g": "G",
            "h": "H",
            "i": "I",
            "j": "J",
            "k": "K",
            "l": "L",
            "m": "M",
            "n": "N",
            "o": "O",
            "p": "P",
            "q": "Q",
            "r": "R",
            "s": "S",
            "t": "T",
            "u": "U",
            "v": "V",
            "x": "X",
            "y": "Y",
            "z": "Z",
        }
        digits = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "z": "zero",
        }
        adverbs = {"a": "again", "n": "now", "p": "please", "s": "soon"}

        try:
            parts = [
                commands.get(sentence_id[0], ""),
                colors.get(sentence_id[1], ""),
                prepositions.get(sentence_id[2], ""),
                letters.get(sentence_id[3], ""),
                digits.get(sentence_id[4], ""),
                adverbs.get(sentence_id[5], ""),
            ]
            return " ".join(filter(None, parts))
        except (IndexError, KeyError):
            return sentence_id

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], str]:
        """Get a sample with synchronized audio/video at 100fps."""
        sample = self.samples[idx]
        label = self.labels[idx]

        # Load video and audio
        video = self._load_video(sample["video_path"])
        audio = self._load_audio(sample["audio_path"])

        # Fixed length for batching: 300 frames (3 seconds at 100fps)
        target_len = 300

        # Synchronize and pad/truncate to fixed length
        min_len = min(video.shape[0], audio.shape[0])
        video = video[:min_len]
        audio = audio[:min_len]

        # Pad if too short
        if video.shape[0] < target_len:
            pad_len = target_len - video.shape[0]
            video = torch.cat([video, video[-1:].repeat(pad_len, 1, 1, 1)], dim=0)
            audio = torch.cat([audio, audio[-1:].repeat(pad_len, 1)], dim=0)
        # Truncate if too long
        else:
            video = video[:target_len]
            audio = audio[:target_len]

        # Apply transforms
        if self.transform:
            video = self.transform(video)
            audio = self.transform(audio)

        if self.target_transform:
            label = self.target_transform(label)

        return {"video": video, "audio": audio}, label

    def _load_video(self, path: str) -> torch.Tensor:
        """Load and process video - resample from 25fps to 100fps."""
        vr = VideoReader(path, ctx=cpu(0))

        # Get all frames from the video (at original 25fps)
        frames = vr[:].asnumpy()  # Shape: (num_frames, H, W, 3)

        # Resample from 25fps to 100fps (4x upsampling)
        # Simple frame repetition - each frame shown 4 times
        resampled = []
        for frame in frames:
            # Each frame is repeated 4 times for 100fps
            for _ in range(4):
                # Resize to 96x96 (face should already be cropped in the dataset)
                resized = (
                    torch.nn.functional.interpolate(
                        torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float(),
                        size=self.video_size,
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    .permute(1, 2, 0)
                    .numpy()
                )
                resampled.append(resized)

        if resampled:
            video = np.stack(resampled)
            video = video.astype(np.float32) / 255.0
            video = (video - 0.5) / 0.5  # Normalize to [-1, 1]
            return torch.from_numpy(video)

        # Fallback
        return torch.zeros(300, *self.video_size, 3)  # ~3 seconds at 100fps

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and process audio - mel-spectrogram at 100fps."""
        # Load audio at 16kHz
        audio, _ = librosa.load(path, sr=self.audio_sr)

        # Compute mel spectrogram
        # With hop_length=160 at 16kHz, we get 100fps
        # (16000 samples/sec) / (160 samples/hop) = 100 hops/sec = 100fps
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.audio_sr,
            n_mels=self.n_mels,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to (time, freq) -> (num_frames, 80)
        mel_spec = mel_spec.T

        return torch.from_numpy(mel_spec.astype(np.float32))

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __repr__(self) -> str:
        """String representation."""
        head = "Dataset GRID"
        body = [f"    Number of datapoints: {self.__len__()}"]
        body.append(f"    Root location: {self.root}")
        body.append(f"    Split: {self.split}")
        body.append(
            f"    Mode: {'Speaker-dependent' if self.speaker_dependent else 'Speaker-independent'}"
        )
        body.append(f"    Subjects: {self.subjects}")
        lines = [head, *body]
        return "\n".join(lines)
