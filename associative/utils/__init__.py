"""Utility modules for associative memory models.

This package provides utility functions for graph processing, masking operations,
and other helper functions used across the associative memory framework.
"""

from .graph import (
    create_graph_mask_indices,
    get_graph_positional_encoding,
    prepare_graph_batch,
)
from .masking import apply_spatial_mask

__all__ = [
    # Masking utilities
    "apply_spatial_mask",
    # Graph utilities
    "create_graph_mask_indices",
    "get_graph_positional_encoding",
    "prepare_graph_batch",
]
