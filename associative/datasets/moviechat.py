"""MovieChat-1K dataset for video reconstruction experiments.

MovieChat-1K is a benchmark dataset of 1000 long videos for video understanding.
This implementation automatically downloads videos from Hugging Face and caches them locally.

Environment Variables:
    HF_TOKEN: Hugging Face token for accessing the gated dataset.
              Get your token from https://huggingface.co/settings/tokens
              Export it as: export HF_TOKEN="hf_..."

Usage:
    >>> from associative.datasets.moviechat import MovieChat1K
    >>> # Uses XDG cache by default (~/.cache/associative/moviechat/)
    >>> dataset = MovieChat1K(split="train", num_frames=512)
    >>> sample = dataset[0]
    >>> video_frames = sample["frames"]  # [num_frames, C, H, W]
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore[import]
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MovieChat1K(Dataset[dict[str, Any]]):
    """MovieChat-1K video dataset.

    This dataset automatically downloads videos from Hugging Face on first use
    and caches them locally for subsequent runs.

    Args:
        root: Root directory for cache. If None, uses XDG_CACHE_HOME/associative/moviechat
        split: Dataset split ('train' or 'test')
        num_frames: Number of frames to sample from each video (default 512)
        resolution: Frame resolution to resize to (default 224)
        transform: Optional transform to apply to frames
        download: If True, download dataset if not found
        max_videos: Maximum number of videos to use (None for all, for testing only)
        download_features: If True, also download pre-extracted features (train split only; default False)
        seed: Random seed for deterministic frame sampling (default None for random)
        validate_videos: If True, validate video files during initialization (default False)
    """

    HF_REPO_IDS: ClassVar[dict[str, str]] = {
        "train": "Enxin/MovieChat-1K_train",
        "test": "Enxin/MovieChat-1K-test",
    }

    VIDEO_PREFIXES: ClassVar[dict[str, str]] = {
        "train": "raw_videos/",
        "test": "videos/",
    }

    METADATA_PREFIXES: ClassVar[dict[str, list[str]]] = {
        "train": ["jsons/"],
        "test": ["annotations/", "gt/"],
    }

    FEATURES_PREFIXES: ClassVar[dict[str, list[str]]] = {
        "train": ["movies/"],
        "test": [],
    }

    def __init__(  # noqa: PLR0913
        self,
        root: Path | str | None = None,
        split: Literal["train", "test"] = "train",
        num_frames: int = 512,
        resolution: int = 224,
        transform: Any | None = None,
        download: bool = True,
        max_videos: int | None = None,
        download_features: bool = False,
        seed: int | None = None,
        validate_videos: bool = False,
    ):
        """Initialize MovieChat-1K dataset.

        Sets up directories, validates parameters, downloads data if needed,
        and prepares the dataset for use.

        Args:
            root: Root directory for cache. If None, uses XDG_CACHE_HOME/associative/moviechat
            split: Dataset split ('train' or 'test')
            num_frames: Number of frames to sample from each video
            resolution: Frame resolution to resize to
            transform: Optional transform to apply to frames
            download: If True, download dataset if not found
            max_videos: Maximum number of videos to use (None for all)
            download_features: If True, also download pre-extracted features (train split only)
            seed: Random seed for deterministic frame sampling
            validate_videos: If True, validate video files during initialization

        Raises:
            ValueError: If split is not 'train' or 'test', or if HF_TOKEN is missing when download is True
            RuntimeError: If video directory doesn't exist and download is False
        """
        if root is None:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = Path(xdg_cache) / "associative" / "moviechat" / split
        else:
            self.root = Path(root).expanduser() / "moviechat" / split
        self.split: str = split
        self.num_frames: int = num_frames
        self.resolution: int = resolution
        self.transform: Any = transform
        self.download: bool = download
        self.max_videos: int | None = max_videos
        self.download_features: bool = download_features
        self.seed: int | None = seed
        self.validate_videos: bool = validate_videos
        self.samples: list[dict[str, Any]] = []

        self.rng: torch.Generator = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")

        self.video_dir = self.root / "videos"
        self.metadata_dir = self.root / "metadata"
        self.features_dir = self.root / "features"

        self.token = os.getenv("HF_TOKEN")
        if not self.token and download and not self._check_exists():
            raise ValueError(
                "HF_TOKEN environment variable is required to download MovieChat-1K.\n"
                "Please set it with: export HF_TOKEN='your_token_here'\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )

        if download and not self._check_exists():
            self._download()

        self._load_video_list()

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                ]
            )

    def _check_exists(self) -> bool:
        """Check if dataset files exist.

        Returns:
            bool: True if video directory exists and contains at least one MP4 file.
        """
        return self.video_dir.exists() and len(list(self.video_dir.glob("*.mp4"))) > 0

    def _download(self) -> None:  # noqa: C901
        """Download dataset from Hugging Face.

        Downloads videos, metadata, and optionally features from the HuggingFace repository.
        Uses HF_TOKEN from environment for authentication.

        Raises:
            RuntimeError: If HF_TOKEN is invalid or repository access fails.
        """
        repo_id = self.HF_REPO_IDS[self.split]
        video_prefix = self.VIDEO_PREFIXES[self.split]
        metadata_prefixes = self.METADATA_PREFIXES[self.split]
        features_prefixes = (
            self.FEATURES_PREFIXES[self.split] if self.download_features else []
        )

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = list_repo_files(repo_id, repo_type="dataset", token=self.token)
        except Exception as e:
            raise RuntimeError(
                f"Failed to access {repo_id}. Make sure your HF_TOKEN is valid.\n"
                f"Error: {e}"
            ) from e

        all_files: list[tuple[str, Path]] = []

        video_files = sorted(
            f for f in files if f.startswith(video_prefix) and f.endswith(".mp4")
        )
        if self.max_videos is not None:
            video_files = video_files[: self.max_videos]
        all_files.extend((f, self.video_dir) for f in video_files)

        json_files = sorted(
            f
            for f in files
            if any(f.startswith(p) for p in metadata_prefixes) and f.endswith(".json")
        )
        if self.max_videos is not None:
            json_files = json_files[: self.max_videos]
        all_files.extend((f, self.metadata_dir) for f in json_files)

        if features_prefixes:
            features_files = sorted(
                f for f in files if any(f.startswith(p) for p in features_prefixes)
            )
            if self.max_videos is not None:
                features_files = features_files[: self.max_videos]
            all_files.extend((f, self.features_dir) for f in features_files)

        total_files = len(all_files)
        if total_files == 0:
            logger.info("No files found to download.")
            return

        with tqdm(
            total=total_files, desc=f"Downloading {self.split} split", unit="file"
        ) as pbar:
            for file_path, output_dir in all_files:
                file_name = Path(file_path).name
                output_path = output_dir / file_name

                if output_path.exists():
                    pbar.update(1)
                    continue

                try:
                    cache_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type="dataset",
                        token=self.token,
                        cache_dir=self.root / ".cache",
                    )

                    if not output_path.exists():
                        cache_path = Path(cache_path).resolve()
                        output_path.symlink_to(cache_path)

                except Exception as e:
                    logger.error(f"Error downloading {file_name}: {e}")
                    continue

                pbar.update(1)

    def _validate_video_files(self, video_files: list[Path]) -> list[Path]:
        """Validate video files can be opened and contain frames.

        Args:
            video_files: List of video file paths to validate.

        Returns:
            List of valid video file paths.

        Raises:
            RuntimeError: If all videos are invalid.
        """
        valid_videos = []
        for video_path in video_files:
            try:
                vr = VideoReader(str(video_path.resolve()), ctx=cpu(0))
                if len(vr) > 0:
                    valid_videos.append(video_path)  # type: ignore[arg-type]
                else:
                    logger.warning(f"Skipping empty video: {video_path.name}")
            except Exception as e:
                logger.warning(
                    f"Skipping corrupted/invalid video {video_path.name}: {e}"
                )

        if len(valid_videos) == 0:  # type: ignore[arg-type]
            raise RuntimeError(
                f"No valid videos found in {self.video_dir}. "
                "All videos appear to be corrupted or empty."
            )

        return valid_videos  # type: ignore[return-value]

    def _load_metadata(self, video_id: str) -> dict[str, Any]:
        """Load metadata for a video if it exists.

        Args:
            video_id: ID of the video to load metadata for.

        Returns:
            Metadata dictionary or empty dict if not found/invalid.
        """
        metadata_path = self.metadata_dir / f"{video_id}.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {}

    def _load_video_list(self) -> None:
        """Load list of available videos.

        Scans video directory for MP4 files, optionally validates them,
        and creates sample list with metadata.

        Raises:
            RuntimeError: If video directory doesn't exist or no videos found.
        """
        if not self.video_dir.exists():
            raise RuntimeError(
                f"Video directory {self.video_dir} does not exist. "
                "Set download=True to download the dataset."
            )

        video_files = sorted(self.video_dir.glob("*.mp4"))

        if len(video_files) == 0:
            raise RuntimeError(
                f"No videos found in {self.video_dir}. "
                "Set download=True to download the dataset."
            )

        if self.validate_videos:
            video_files = self._validate_video_files(video_files)

        if self.max_videos is not None:
            video_files = video_files[: self.max_videos]

        self.samples = []
        for video_path in video_files:
            video_id = video_path.stem
            self.samples.append(
                {
                    "video_path": video_path,
                    "video_id": video_id,
                    "metadata": self._load_metadata(video_id),
                }
            )

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        """Load and sample frames from a video file.

        Loads video, samples frames uniformly, applies transforms, and handles
        errors gracefully with fallback frames.

        Args:
            video_path: Path to the video file.

        Returns:
            torch.Tensor: Tensor of shape [num_frames, C, H, W] if transform includes ToTensor,
                otherwise [num_frames, H, W, C].
        """
        video_path = video_path.resolve()

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        if total_frames > self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        else:
            indices = torch.arange(total_frames)
            if len(indices) < self.num_frames:
                repeats = self.num_frames // len(indices) + 1
                indices = indices.repeat(repeats)[: self.num_frames]

        frames: list[Any] = []
        for idx in indices:
            try:
                safe_idx = min(idx.item(), total_frames - 1)
                frame = vr[safe_idx].asnumpy()  # type: ignore[attr-defined]
            except Exception as e:
                if frames:
                    frame = frames[-1]
                    logger.warning(
                        f"Failed to load frame {idx.item()} from {video_path.name}: {e}, reusing last frame"
                    )
                    frames.append(frame)
                    continue
                frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
                logger.warning(
                    f"Failed to load frame {idx.item()} from {video_path.name}: {e}, using black frame"
                )

            if self.transform is not None:
                frame = self.transform(frame)
                if not isinstance(frame, torch.Tensor | np.ndarray):
                    raise TypeError(
                        f"Transform must return torch.Tensor or numpy.ndarray, "
                        f"got {type(frame).__name__}"
                    )

            frames.append(frame)

        result: torch.Tensor
        if isinstance(frames[0], torch.Tensor):
            result = torch.stack(frames)
        else:
            np_frames = np.stack(frames)
            result = torch.from_numpy(np_frames)  # type: ignore[arg-type]

        return result

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns:
            int: Number of video samples available.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - frames: Video frames tensor of shape [num_frames, C, H, W] or [num_frames, H, W, C]
                - video_id: String identifier for the video
                - metadata: Optional metadata dictionary with captions, QA pairs, etc.
                - caption: Optional caption string (if present in metadata)
                - qa_pairs: Optional QA pairs (if present in metadata)
        """
        sample = self.samples[index]

        frames = self._load_video_frames(sample["video_path"])

        output = {
            "frames": frames,
            "video_id": sample["video_id"],
        }

        if sample["metadata"]:
            output["metadata"] = sample["metadata"]
            if "caption" in sample["metadata"]:
                output["caption"] = sample["metadata"]["caption"]
            if "global" in sample["metadata"]:
                output["qa_pairs"] = sample["metadata"]["global"]

        return output

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns:
            str: Formatted string with dataset parameters.
        """
        return (
            f"MovieChat1K(split='{self.split}', "
            f"num_videos={len(self)}, "
            f"num_frames={self.num_frames}, "
            f"resolution={self.resolution})"
        )


def create_moviechat_dataloader(  # noqa: PLR0913
    *,
    root: Path | str | None = None,
    split: Literal["train", "test"] = "train",
    num_frames: int = 512,
    resolution: int = 224,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool | None = None,
    pin_memory: bool = True,
    download: bool = True,
    max_videos: int | None = None,
    download_features: bool = False,
    seed: int | None = None,
    validate_videos: bool = False,
    **kwargs: Any,
) -> tuple[MovieChat1K, torch.utils.data.DataLoader[dict[str, Any]]]:
    """Create MovieChat-1K dataset and dataloader.

    Args:
        root: Root directory for cache. If None, uses XDG_CACHE_HOME/associative/moviechat
        split: Dataset split ('train' or 'test')
        num_frames: Number of frames to sample per video
        resolution: Frame resolution
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        shuffle: Whether to shuffle (defaults to True for train)
        pin_memory: Whether to pin memory for CUDA
        download: Whether to download dataset if not found
        max_videos: Maximum number of videos to use
        download_features: If True, also download pre-extracted features (train split only)
        seed: Random seed for deterministic frame sampling
        validate_videos: If True, validate video files during initialization
        **kwargs: Additional arguments for MovieChat1K

    Returns:
        Tuple of (dataset, dataloader)

    Example:
        >>> os.environ["HF_TOKEN"] = "your_token_here"
        >>> dataset, dataloader = create_moviechat_dataloader(
        ...     split="train",
        ...     batch_size=8,
        ... )
        >>> for batch in dataloader:
        ...     frames = batch["frames"]
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = MovieChat1K(
        root=root,
        split=split,
        num_frames=num_frames,
        resolution=resolution,
        download=download,
        max_videos=max_videos,
        download_features=download_features,
        seed=seed,
        validate_videos=validate_videos,
        **kwargs,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(split == "train"),
    )

    return dataset, dataloader
