"""Video data loading utilities following PyTorch DataLoader patterns.

This module provides video-specific data loading utilities that extend
PyTorch's DataLoader functionality for efficient video processing.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter

from ._registry import register_loader


class _VideoDataLoaderIter(_BaseDataLoaderIter):
    """Iterator for VideoDataLoader that handles device placement."""

    def __init__(self, loader: "VideoDataLoader") -> None:
        # Initialize the base iterator
        super().__init__(loader)
        self.device = loader.device

    def __next__(self) -> dict[str, Tensor]:
        batch = super().__next__()
        if self.device is not None:
            # Move batch to target device
            batch = move_batch_to_device(batch, self.device)
        return batch


class VideoDataLoader(DataLoader):
    """Enhanced DataLoader for video datasets.

    Extends PyTorch's DataLoader with video-specific functionality:
    - Memory-efficient video batch processing
    - Custom collate functions for variable-length videos
    - Device placement optimization for video data

    Args:
        dataset: Dataset to load from
        batch_size: Batch size for loading
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop incomplete final batch
        collate_fn: Custom collate function (defaults to video_collate_fn)
        device: Target device for batches
        prefetch_factor: Number of batches to prefetch per worker
        **kwargs: Additional arguments passed to base DataLoader

    Example:
        >>> from associative.datasets import MovieChat1K
        >>> dataset = MovieChat1K("path/to/videos", num_frames=512)
        >>> loader = VideoDataLoader(
        ...     dataset, batch_size=4, shuffle=True,
        ...     num_workers=4, device=torch.device("cuda")
        ... )
        >>> for batch in loader:
        ...     frames, masked_frames, masks = batch.values()
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Callable | None = None,
        device: torch.device | None = None,
        prefetch_factor: int = 2,
        **kwargs: Any,
    ):
        """Initialize video data loader.

        Args:
            dataset: Video dataset to load from
            batch_size: Number of videos per batch
            shuffle: Whether to shuffle dataset order
            num_workers: Number of parallel workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            drop_last: Drop incomplete batches
            collate_fn: Function to collate samples into batches
            device: Device to place batches on
            prefetch_factor: Batches to prefetch per worker
            **kwargs: Additional DataLoader arguments

        Raises:
            ValueError: If batch_size <= 0 or invalid parameters
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        # Default to video collate function if none provided
        if collate_fn is None:
            collate_fn = video_collate_fn

        self.device = device

        # Initialize base DataLoader
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            **kwargs,
        )

    def __iter__(self) -> "_VideoDataLoaderIter":
        """Iterate over batches, optionally moving to target device.

        Returns:
            Iterator over batched video data dictionaries
        """
        return _VideoDataLoaderIter(self)


class BatchProcessor:
    """Utility for processing large video datasets in batches.

    Handles memory-constrained processing of video data that may not fit
    in memory all at once. Provides utilities for batch processing with
    progress tracking and memory management.

    Args:
        batch_size: Size of processing batches
        device: Device for computation
        show_progress: Whether to show progress bar

    Example:
        >>> processor = BatchProcessor(batch_size=32, device="cuda")
        >>> embeddings = processor.process_videos(
        ...     video_list, embedding_model, desc="Extracting embeddings"
        ... )
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: torch.device | None = None,
        show_progress: bool = True,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Number of items to process per batch
            device: Device for computation
            show_progress: Whether to display progress information

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.batch_size = batch_size
        self.device = device
        self.show_progress = show_progress

    def process_tensors(
        self,
        tensors: list[Tensor],
        process_fn: Callable[[Tensor], Tensor],
        desc: str = "Processing",
    ) -> list[Tensor]:
        """Process list of tensors in batches.

        Args:
            tensors: List of tensors to process
            process_fn: Function to apply to each batch
            desc: Description for progress bar

        Returns:
            List of processed tensors

        Example:
            >>> tensors = [torch.randn(100, 512) for _ in range(10)]
            >>> processor = BatchProcessor(batch_size=3)
            >>> results = processor.process_tensors(
            ...     tensors, lambda x: torch.nn.functional.normalize(x, dim=-1)
            ... )
        """
        results = []

        if self.show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    range(0, len(tensors), self.batch_size), desc=desc, unit="batch"
                )
            except ImportError:
                # Fallback if tqdm not available
                iterator = range(0, len(tensors), self.batch_size)
        else:
            iterator = range(0, len(tensors), self.batch_size)

        for i in iterator:
            batch = tensors[i : i + self.batch_size]

            # Stack batch if multiple tensors
            if len(batch) > 1:
                try:
                    batched_tensor = torch.stack(batch)
                except RuntimeError as e:
                    raise ValueError(f"Cannot stack tensors in batch: {e}") from e
            else:
                batched_tensor = batch[0].unsqueeze(0)

            # Move to device if specified
            if self.device is not None:
                batched_tensor = batched_tensor.to(self.device)

            # Process batch
            with torch.no_grad():
                processed = process_fn(batched_tensor)

            # Split batch back into individual tensors
            if processed.shape[0] > 1:
                results.extend(processed)
            else:
                results.append(processed.squeeze(0))

        return results

    def process_dataset(
        self,
        dataset: Dataset,
        process_fn: Callable[[dict[str, Tensor]], Tensor],
        desc: str = "Processing dataset",
    ) -> list[Tensor]:
        """Process entire dataset in batches.

        Args:
            dataset: Dataset to process
            process_fn: Function to apply to each batch
            desc: Description for progress

        Returns:
            List of processed results
        """
        # Create loader for efficient batching
        loader = VideoDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            device=self.device,
        )

        results = []

        if self.show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(loader, desc=desc, unit="batch")
            except ImportError:
                iterator = loader
        else:
            iterator = loader

        for batch in iterator:
            with torch.no_grad():
                processed = process_fn(batch)

            # Handle different output formats
            if isinstance(processed, Tensor):
                if processed.dim() > 1:
                    results.extend(processed)
                else:
                    results.append(processed)
            elif isinstance(processed, list | tuple):
                results.extend(processed)
            else:
                results.append(processed)

        return results


def video_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """Default collate function for video data.

    Handles batching of video data dictionaries, properly stacking
    tensors and handling variable-length sequences.

    Args:
        batch: List of sample dictionaries from dataset

    Returns:
        Dictionary with batched tensors

    Raises:
        RuntimeError: If samples have incompatible shapes

    Example:
        >>> sample1 = {"frames": torch.randn(100, 3, 224, 224), "mask": torch.randn(100)}
        >>> sample2 = {"frames": torch.randn(100, 3, 224, 224), "mask": torch.randn(100)}
        >>> batch = video_collate_fn([sample1, sample2])
        >>> print(batch["frames"].shape)  # torch.Size([2, 100, 3, 224, 224])
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")

    # Get keys from first sample
    sample_keys = batch[0].keys()

    # Check all samples have same keys
    for i, sample in enumerate(batch[1:], 1):
        if set(sample.keys()) != set(sample_keys):
            raise RuntimeError(f"Sample {i} has different keys than sample 0")

    collated = {}

    for key in sample_keys:
        values = [sample[key] for sample in batch]

        # Handle different data types
        if isinstance(values[0], Tensor):
            try:
                # Stack tensors into batch dimension
                collated[key] = torch.stack(values, dim=0)
            except RuntimeError as e:
                raise RuntimeError(f"Cannot stack tensors for key '{key}': {e}") from e

        elif isinstance(values[0], str):
            # Keep strings as list
            collated[key] = values

        elif isinstance(values[0], int | float):
            # Convert numbers to tensor
            collated[key] = torch.tensor(values)

        else:
            # Keep other types as list
            collated[key] = values

    return collated


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move batch data to specified device.

    Args:
        batch: Dictionary containing batch data
        device: Target device

    Returns:
        Batch with tensors moved to device

    Example:
        >>> batch = {"frames": torch.randn(2, 100, 3, 224, 224)}
        >>> cuda_batch = move_batch_to_device(batch, torch.device("cuda"))
    """
    moved_batch = {}

    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved_batch[key] = value.to(device, non_blocking=True)
        else:
            # Keep non-tensors as-is
            moved_batch[key] = value

    return moved_batch


def create_video_loader(
    dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    device: torch.device | None = None,
    **kwargs: Any,
) -> VideoDataLoader:
    """Convenience function to create video loader with sensible defaults.

    Args:
        dataset: Video dataset
        batch_size: Batch size
        num_workers: Number of workers
        device: Target device
        **kwargs: Additional loader arguments

    Returns:
        Configured VideoDataLoader

    Example:
        >>> loader = create_video_loader(dataset, batch_size=8, device="cuda")
    """
    return VideoDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        shuffle=kwargs.pop("shuffle", True),
        pin_memory=kwargs.pop("pin_memory", device is not None),
        drop_last=kwargs.pop("drop_last", True),
        **kwargs,
    )


# Register loaders
register_loader("video_dataloader", VideoDataLoader)
register_loader("batch_processor", BatchProcessor)
