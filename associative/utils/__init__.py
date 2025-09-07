"""Utility modules for associative memory models.

This package provides utility functions for graph processing, masking operations,
and other helper functions used across the associative memory framework.
"""

from .graph import (
    create_graph_mask_indices,
    get_graph_positional_encoding,
    prepare_graph_batch,
)
from .masking import (
    add_noise_to_embeddings,
    apply_mask_to_embeddings,
    generate_block_mask,
    generate_random_mask,
)

__all__ = [
    # Masking utilities
    "add_noise_to_embeddings",
    "apply_mask_to_embeddings",
    # Graph utilities
    "create_graph_mask_indices",
    "generate_block_mask",
    "generate_random_mask",
    "get_graph_positional_encoding",
    "prepare_graph_batch",
]
