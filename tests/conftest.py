"""Pytest configuration and shared fixtures."""

# Add project root to path for imports
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Test constants
TOLERANCE_ENERGY_DIFF = 1e-7
TOLERANCE_WEIGHT_STD_MIN = 0.015
TOLERANCE_WEIGHT_STD_MAX = 0.025
TOLERANCE_ZERO_ENERGY = 1e-6
TOLERANCE_INIT_STD = 0.01
MIN_GRAD_RATIO = 0.3
ADJACENCY_THRESHOLD = 0.7
GRAPH_ADJACENCY_DIM_THRESHOLD = 3


@pytest.fixture
def device():
    """Get test device (CPU for reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed_value = 42
    torch.manual_seed(seed_value)
    return seed_value


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def seq_length():
    """Standard sequence length for tests."""
    return 10


@pytest.fixture
def embed_dim():
    """Standard embedding dimension for tests."""
    return 64


@pytest.fixture
def num_heads():
    """Standard number of attention heads."""
    return 4


@pytest.fixture
def qk_dim():
    """Standard query/key dimension."""
    return 16


@pytest.fixture
def img_size():
    """Standard image size for vision tests."""
    return 32


@pytest.fixture
def patch_size():
    """Standard patch size for vision tests."""
    return 4


@pytest.fixture
def num_patches(img_size, patch_size):
    """Calculate number of patches."""
    return (img_size // patch_size) ** 2
