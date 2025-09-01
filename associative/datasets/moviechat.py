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
import os
import warnings
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download, list_repo_files
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


class MovieChat1K(Dataset):
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
    """

    # Dataset configuration
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

    ADDITIONAL_PREFIXES: ClassVar[dict[str, list[str]]] = {
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
    ):
        """Initialize MovieChat-1K dataset."""
        # Use XDG compliant cache directory
        if root is None:
            xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = Path(xdg_cache) / "associative" / "moviechat" / split
        else:
            self.root = Path(root).expanduser() / "moviechat" / split
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.transform = transform
        self.download = download
        self.max_videos = max_videos

        # Validate split
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'")

        # Setup directories
        self.video_dir = self.root / "videos"
        self.metadata_dir = self.root / "metadata"
        self.features_dir = self.root / "features"

        # Get HF token from environment
        self.token = os.getenv("HF_TOKEN")
        if not self.token and download and not self._check_exists():
            raise ValueError(
                "HF_TOKEN environment variable is required to download MovieChat-1K.\n"
                "Please set it with: export HF_TOKEN='your_token_here'\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )

        # Download if needed
        if download and not self._check_exists():
            self._download()

        # Load video list
        self._load_video_list()

        # Setup default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((resolution, resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return self.video_dir.exists() and len(list(self.video_dir.glob("*.mp4"))) > 0

    def _download(self) -> None:
        """Download dataset from Hugging Face."""
        repo_id = self.HF_REPO_IDS[self.split]
        video_prefix = self.VIDEO_PREFIXES[self.split]
        metadata_prefixes = self.METADATA_PREFIXES[self.split]
        additional_prefixes = self.ADDITIONAL_PREFIXES[self.split]

        # Create directories
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # List files in repository
        try:
            files = list_repo_files(repo_id, repo_type="dataset", token=self.token)
        except Exception as e:
            raise RuntimeError(
                f"Failed to access {repo_id}. Make sure your HF_TOKEN is valid.\n"
                f"Error: {e}"
            ) from e

        # Collect all files to download
        all_files = self._collect_files_to_download(
            files, video_prefix, metadata_prefixes, additional_prefixes
        )

        total_files = len(all_files)
        if total_files == 0:
            return

        # Single progress bar for all downloads
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
                    print(f"Error downloading {file_name}: {e}")
                    continue

                pbar.update(1)

    def _collect_files_to_download(
        self,
        files: list[str],
        video_prefix: str,
        metadata_prefixes: list[str],
        additional_prefixes: list[str],
    ) -> list[tuple[str, Path]]:
        all_files = []

        # Videos
        video_files = sorted(
            f for f in files if f.startswith(video_prefix) and f.endswith(".mp4")
        )
        if self.max_videos is not None:
            video_files = video_files[: self.max_videos]
        all_files.extend((f, self.video_dir) for f in video_files)

        # Metadata
        json_files = sorted(
            f
            for f in files
            if any(f.startswith(p) for p in metadata_prefixes) and f.endswith(".json")
        )
        if self.max_videos is not None:
            json_files = json_files[: self.max_videos]
        all_files.extend((f, self.metadata_dir) for f in json_files)

        # Additional
        if additional_prefixes:
            additional_files = sorted(
                f for f in files if any(f.startswith(p) for p in additional_prefixes)
            )
            if self.max_videos is not None:
                additional_files = additional_files[: self.max_videos]
            all_files.extend((f, self.features_dir) for f in additional_files)

        return all_files

    def _load_video_list(self) -> None:
        """Load list of available videos."""
        if not self.video_dir.exists():
            raise RuntimeError(
                f"Video directory {self.video_dir} does not exist. "
                "Set download=True to download the dataset."
            )

        # Get all video files
        video_files = sorted(self.video_dir.glob("*.mp4"))

        if len(video_files) == 0:
            raise RuntimeError(
                f"No videos found in {self.video_dir}. "
                "Set download=True to download the dataset."
            )

        # Limit if specified
        if self.max_videos is not None:
            video_files = video_files[: self.max_videos]

        # Create sample list
        self.samples = []
        for video_path in video_files:
            video_id = video_path.stem

            # Try to load metadata if available
            metadata_path = self.metadata_dir / f"{video_id}.json"
            metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                except json.JSONDecodeError:
                    pass

            self.samples.append(
                {
                    "video_path": video_path,
                    "video_id": video_id,
                    "metadata": metadata,
                }
            )

    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        """Load and sample frames from a video file.

        Returns:
            Tensor of shape [num_frames, C, H, W] if transform includes ToTensor,
            otherwise [num_frames, H, W, C]
        """
        # Resolve symlinks to get actual file path
        video_path = video_path.resolve()

        # Load video
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        # Sample frame indices uniformly
        if total_frames > self.num_frames:
            indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        else:
            # If video has fewer frames than requested, sample all and repeat
            indices = torch.arange(total_frames)
            if len(indices) < self.num_frames:
                # Repeat frames to reach num_frames
                repeats = self.num_frames // len(indices) + 1
                indices = indices.repeat(repeats)[: self.num_frames]

        # Extract frames
        frames = []
        for idx in indices:
            frame = vr[idx.item()].asnumpy()  # [H, W, C]

            # Apply transform if provided
            if self.transform is not None:
                frame = self.transform(frame)

            frames.append(frame)

        # Stack frames
        if isinstance(frames[0], torch.Tensor):
            frames = torch.stack(frames)  # [num_frames, C, H, W]
        else:
            frames = np.stack(frames)  # [num_frames, H, W, C]
            frames = torch.from_numpy(frames)

        return frames

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - frames: Video frames [num_frames, C, H, W] or [num_frames, H, W, C]
                - video_id: Video identifier
                - metadata: Video metadata (captions, QA pairs, etc.)
        """
        sample = self.samples[index]

        # Load video frames
        frames = self._load_video_frames(sample["video_path"])

        # Build output
        output = {
            "frames": frames,
            "video_id": sample["video_id"],
        }

        # Add metadata if available
        if sample["metadata"]:
            output["metadata"] = sample["metadata"]
            # Extract specific fields if present
            if "caption" in sample["metadata"]:
                output["caption"] = sample["metadata"]["caption"]
            if "global" in sample["metadata"]:
                output["qa_pairs"] = sample["metadata"]["global"]

        return output

    def __repr__(self) -> str:
        """String representation."""
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
    **kwargs: Any,
) -> tuple[MovieChat1K, torch.utils.data.DataLoader]:
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
        **kwargs: Additional arguments for MovieChat1K

    Returns:
        Tuple of (dataset, dataloader)

    Example:
        >>> # First time - will download entire dataset
        >>> os.environ["HF_TOKEN"] = "your_token_here"
        >>> dataset, dataloader = create_moviechat_dataloader(
        ...     split="train",
        ...     batch_size=8,
        ...     # max_videos=10,  # Uncomment to limit for testing
        ... )
        >>> for batch in dataloader:
        ...     frames = batch["frames"]  # [B, num_frames, C, H, W]
        ...     # Train your model...
    """
    # Default shuffle based on split
    if shuffle is None:
        shuffle = split == "train"

    # Create dataset
    dataset = MovieChat1K(
        root=root,
        split=split,
        num_frames=num_frames,
        resolution=resolution,
        download=download,
        max_videos=max_videos,
        **kwargs,
    )

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=(split == "train"),
    )

    return dataset, dataloader
