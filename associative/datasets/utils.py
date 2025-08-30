"""Utility functions for datasets."""

import torch
from torch.utils.data import DataLoader, Dataset

# Dataset normalization constants
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "imagenet32": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def generate_mask_indices(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random mask indices for image reconstruction.

    Args:
        batch_size: Batch size
        num_patches: Total number of patches
        mask_ratio: Ratio of patches to mask
        device: Device to create tensors on (default: CPU)

    Returns:
        batch_idx: Batch indices [batch_size, num_mask]
        mask_idx: Mask indices [batch_size, num_mask]
    """
    num_mask = int(num_patches * mask_ratio)

    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    mask_idx = torch.rand(batch_size, num_patches, device=device).argsort(dim=-1)[
        :, :num_mask
    ]

    return batch_idx, mask_idx


def unnormalize(
    x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    """Unnormalize tensor.

    Args:
        x: Normalized tensor
        mean: Mean values per channel
        std: Std values per channel

    Returns:
        Unnormalized tensor
    """
    device = x.device
    dtype = x.dtype

    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)

    return x * std_tensor + mean_tensor


def setup_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Setup data loaders for training and validation.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
