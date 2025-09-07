"""Masking utilities for temporal and spatial data.

This module provides utilities for creating masks for various reconstruction
and autoencoding tasks, including temporal masking for video/sequence data.
"""

import torch
from torch import Tensor

# Expected tensor dimensions
EMBEDDING_DIM = 3
MASK_2D_DIM = 2


def generate_random_mask(
    batch_size: int,
    sequence_length: int,
    mask_ratio: float,
    device: torch.device | None = None,
) -> Tensor:
    """Generate random temporal mask for sequence data.

    Args:
        batch_size: Batch size
        sequence_length: Length of the sequence (e.g., number of frames)
        mask_ratio: Ratio of positions to mask (0 to 1)
        device: Device to create tensor on

    Returns:
        Boolean mask tensor of shape (batch_size, sequence_length)
        where True indicates masked positions

    Example:
        >>> mask = generate_random_mask(2, 100, 0.5)
        >>> print(mask.shape)  # (2, 100)
        >>> print(mask.float().mean())  # ~0.5
    """
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    num_masked = int(sequence_length * mask_ratio)

    # Generate random noise for each sequence in batch
    noise = torch.rand(batch_size, sequence_length, device=device)

    # Sort to get indices of positions to mask
    ids_shuffle = torch.argsort(noise, dim=1)
    masked_indices = ids_shuffle[:, :num_masked]

    # Create boolean mask
    mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
    mask.scatter_(1, masked_indices, True)

    return mask


def generate_block_mask(
    batch_size: int,
    sequence_length: int,
    mask_ratio: float,
    block_size: int = 1,
    device: torch.device | None = None,
) -> Tensor:
    """Generate block-wise mask for sequence data.

    Masks contiguous blocks of the sequence, useful for testing
    temporal coherence in reconstruction.

    Args:
        batch_size: Batch size
        sequence_length: Length of the sequence
        mask_ratio: Ratio of positions to mask
        block_size: Size of contiguous blocks to mask
        device: Device to create tensor on

    Returns:
        Boolean mask tensor of shape (batch_size, sequence_length)

    Example:
        >>> mask = generate_block_mask(2, 100, 0.5, block_size=10)
        >>> # Will mask ~5 blocks of size 10 each
    """
    if not 0 <= mask_ratio <= 1:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if block_size > sequence_length:
        raise ValueError(f"block_size {block_size} > sequence_length {sequence_length}")

    num_blocks = sequence_length // block_size
    num_masked_blocks = int(num_blocks * mask_ratio)

    mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Randomly select blocks to mask
        block_indices = torch.randperm(num_blocks, device=device)[:num_masked_blocks]

        # Convert block indices to position indices
        for block_idx in block_indices:
            start = int(block_idx * block_size)
            end = min(start + block_size, sequence_length)
            mask[b, start:end] = True

    return mask


def apply_mask_to_embeddings(
    embeddings: Tensor,
    mask: Tensor,
    mask_value: float = 0.0,
) -> Tensor:
    """Apply mask to embedding sequences.

    Args:
        embeddings: Embeddings of shape (batch_size, seq_len, embed_dim)
        mask: Boolean mask of shape (batch_size, seq_len)
        mask_value: Value to use for masked positions

    Returns:
        Masked embeddings with same shape as input

    Example:
        >>> embeddings = torch.randn(2, 100, 768)
        >>> mask = generate_random_mask(2, 100, 0.5)
        >>> masked = apply_mask_to_embeddings(embeddings, mask)
    """
    if embeddings.dim() != EMBEDDING_DIM:
        raise ValueError(
            f"embeddings must be {EMBEDDING_DIM}D, got {embeddings.dim()}D"
        )
    if mask.dim() != MASK_2D_DIM:
        raise ValueError(f"mask must be {MASK_2D_DIM}D, got {mask.dim()}D")
    if embeddings.shape[:2] != mask.shape:
        raise ValueError(
            f"Shape mismatch: embeddings {embeddings.shape[:2]} vs mask {mask.shape}"
        )

    masked_embeddings = embeddings.clone()
    mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
    masked_embeddings[mask_expanded] = mask_value

    return masked_embeddings


def add_noise_to_embeddings(
    embeddings: Tensor,
    noise_std: float = 0.1,
    mask: Tensor | None = None,
) -> Tensor:
    """Add Gaussian noise to embeddings.

    Args:
        embeddings: Embeddings of shape (batch_size, seq_len, embed_dim)
        noise_std: Standard deviation of Gaussian noise
        mask: Optional mask to apply noise only to certain positions

    Returns:
        Noisy embeddings with same shape as input

    Example:
        >>> embeddings = torch.randn(2, 100, 768)
        >>> noisy = add_noise_to_embeddings(embeddings, noise_std=0.05)
    """
    noise = torch.randn_like(embeddings) * noise_std

    if mask is not None:
        if mask.dim() == MASK_2D_DIM:
            mask = mask.unsqueeze(-1)
        noise = noise * mask.float()

    return embeddings + noise
