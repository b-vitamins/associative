"""MovieChat-1K dataset for video reconstruction experiments."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_frame_indices(num_frames: int, num_segments: int) -> np.ndarray:
    """Get uniformly sampled frame indices from video.

    Args:
        num_frames: Total number of frames in video
        num_segments: Number of frames to sample

    Returns:
        Array of frame indices
    """
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    return np.array(
        [start + int(np.round(seg_size * idx)) for idx in range(num_segments)]
    )


class MovieChat1K(Dataset):
    """MovieChat-1K video dataset for reconstruction experiments.

    This dataset loads videos and samples frames for memory reconstruction tasks.
    Supports both frame-level (pixel) and embedding-level experiments.
    """

    def __init__(
        self,
        root: str,
        num_frames: int = 512,
        resolution: int = 224,
        mask_config: dict[str, Any] | None = None,
        transform: transforms.Compose | None = None,
    ):
        """
        Args:
            root: Path to video directory
            num_frames: Number of frames to sample from each video
            resolution: Frame resolution (assumes square frames)
            mask_config: Dict with masking configuration containing:
                - mask_ratio: Fraction of frame to mask (default 0.5)
                - mask_type: Type of masking ("bottom_half", "random", "none")
                - noise_std: Standard deviation of Gaussian noise (for embeddings)
                - return_embeddings: Whether to return embeddings instead of pixels
                - embedding_model: Model to compute embeddings (if return_embeddings=True)
            transform: Optional transform pipeline
        """
        self.root = Path(root)
        self.num_frames = num_frames
        self.resolution = resolution

        # Set mask configuration defaults
        if mask_config is None:
            mask_config = {}
        self.mask_ratio = mask_config.get("mask_ratio", 0.5)
        self.mask_type = mask_config.get("mask_type", "bottom_half")
        self.noise_std = mask_config.get("noise_std", 0.0)
        self.return_embeddings = mask_config.get("return_embeddings", False)
        self.embedding_model = mask_config.get("embedding_model")

        # Find all video files
        self.video_files = sorted(self.root.glob("*.mp4"))

        if len(self.video_files) == 0:
            raise ValueError(f"No .mp4 files found in {root}")

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((resolution, resolution)),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.video_files)

    def load_video_frames(self, video_path: Path) -> torch.Tensor:
        """Load and sample frames from video.

        Args:
            video_path: Path to video file

        Returns:
            Tensor of frames [num_frames, C, H, W]
        """
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        frame_indices = get_frame_indices(total_frames, self.num_frames)

        frames = []
        for idx in frame_indices:
            frame = Image.fromarray(vr[idx].asnumpy())
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        return torch.stack(frames)

    def apply_mask(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply masking to frames.

        Args:
            frames: Input frames [N, C, H, W]

        Returns:
            (masked_frames, mask) where mask indicates masked regions
        """
        n_frames, channels, height, width = frames.shape
        masked_frames = frames.clone()

        if self.mask_type == "bottom_half":
            # Mask bottom half of each frame
            mask_height = int(height * self.mask_ratio)
            mask = torch.zeros(n_frames, height, width, dtype=torch.bool)
            mask[:, -mask_height:, :] = True
            masked_frames[:, :, -mask_height:, :] = 0.0

        elif self.mask_type == "random":
            # Random patch masking
            num_patches = int(n_frames * height * width * self.mask_ratio)
            mask = torch.zeros(n_frames * height * width, dtype=torch.bool)
            mask_idx = torch.randperm(n_frames * height * width)[:num_patches]
            mask[mask_idx] = True
            mask = mask.reshape(n_frames, height, width)

            for n in range(n_frames):
                masked_frames[n, :, mask[n]] = 0.0

        elif self.mask_type == "none":
            mask = torch.zeros(n_frames, height, width, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        return masked_frames, mask

    def compute_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for frames using the embedding model.

        Args:
            frames: Input frames [N, C, H, W]

        Returns:
            Frame embeddings [N, embed_dim]
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model required when return_embeddings=True")

        with torch.no_grad():
            # Move to device if needed
            device = next(self.embedding_model.parameters()).device
            frames = frames.to(device)

            # Compute embeddings
            embeddings = self.embedding_model(frames)

            # Handle different output formats
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Pool if needed (e.g., for models that return spatial features)
            min_embedding_dimensions = 2
            if embeddings.dim() > min_embedding_dimensions:
                embeddings = embeddings.mean(
                    dim=list(range(min_embedding_dimensions, embeddings.dim()))
                )

            return embeddings.cpu()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a video sample.

        Returns dict with:
            - frames: Original frames [N, C, H, W] or embeddings [N, embed_dim]
            - masked_frames: Masked frames or noisy embeddings
            - mask: Mask tensor
            - video_path: Path to source video
        """
        video_path = self.video_files[idx]

        # Load frames
        frames = self.load_video_frames(video_path)

        if self.return_embeddings:
            # Compute embeddings
            frames = self.compute_embeddings(frames)

            # Add noise for embedding experiments
            if self.noise_std > 0:
                noise = torch.randn_like(frames) * self.noise_std
                masked_frames = frames + noise
            else:
                masked_frames = frames.clone()

            mask = torch.zeros(frames.shape[0], dtype=torch.bool)
        else:
            # Apply masking for pixel experiments
            masked_frames, mask = self.apply_mask(frames)

        return {
            "frames": frames,
            "masked_frames": masked_frames,
            "mask": mask,
            "video_path": str(video_path),
            "video_name": video_path.stem,
        }
