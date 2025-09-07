"""Comprehensive tests for continuous Hopfield networks.

These tests enforce proper PyTorch idioms while preserving mathematical correctness.
Key conventions tested:
1. Always expect batch dimensions (B, L, D) for patterns
2. Consistent batch handling across all inputs
3. No special-casing for single samples
4. Proper device and dtype propagation
5. Standard nn.Module patterns (forward method, reset_parameters, etc.)
"""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from associative.nn.modules.basis import GaussianBasis, RectangularBasis
from associative.nn.modules.config import (
    BasisConfig,
    ContinuousHopfieldConfig,
)
from associative.nn.modules.continuous import (
    ContinuousAttention,
    ContinuousHopfield,
    ContinuousHopfieldEnergy,
    ContinuousMemory,
)


class TestContinuousMemory:
    """Test ContinuousMemory implementation with PyTorch idioms."""

    @pytest.fixture
    def rectangular_memory(self):
        """Create continuous memory with rectangular basis."""
        basis = RectangularBasis(num_basis=4)
        return ContinuousMemory(basis, regularization=0.1)

    @pytest.fixture
    def gaussian_memory(self):
        """Create continuous memory with Gaussian basis."""
        basis = GaussianBasis(num_basis=5)
        return ContinuousMemory(basis, regularization=0.5)

    def test_initialization(self, rectangular_memory):
        """Test proper initialization following nn.Module patterns."""
        expected_regularization = 0.1
        assert rectangular_memory.regularization == expected_regularization
        assert not rectangular_memory.is_fitted

        # Check buffer is properly registered (not None)
        assert hasattr(rectangular_memory, "coefficients")

        # Test invalid regularization
        basis = RectangularBasis(num_basis=4)
        with pytest.raises(ValueError, match="regularization must be positive"):
            ContinuousMemory(basis, regularization=0)
        with pytest.raises(ValueError, match="regularization must be positive"):
            ContinuousMemory(basis, regularization=-0.5)

    def test_reset_parameters(self, rectangular_memory):
        """Test that reset_parameters method exists and works."""
        # Should have reset_parameters method
        assert hasattr(rectangular_memory, "reset_parameters")
        rectangular_memory.reset_parameters()

        # After reset, should be unfitted
        assert not rectangular_memory.is_fitted

    def test_fit_always_expects_batch(self, rectangular_memory):
        """Test that fit always expects batch dimension (B, L, D)."""
        num_patterns, dim = 10, 3

        # Without batch dimension should raise error
        patterns_no_batch = torch.randn(num_patterns, dim)
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            rectangular_memory.fit(patterns_no_batch)

        # With batch dimension should work
        batch_size = 2
        patterns = torch.randn(batch_size, num_patterns, dim)
        rectangular_memory.fit(patterns)

        assert rectangular_memory.is_fitted
        # Coefficients should be (B, N, D) for batched fitting
        assert rectangular_memory.coefficients.shape == (batch_size, 4, dim)

    def test_fit_single_batch(self, gaussian_memory):
        """Test fitting with single item in batch."""
        batch_size, num_patterns, dim = 1, 8, 4
        patterns = torch.randn(batch_size, num_patterns, dim)
        positions = torch.linspace(0.1, 0.9, num_patterns)

        gaussian_memory.fit(patterns, positions)

        assert gaussian_memory.is_fitted
        assert gaussian_memory.coefficients.shape == (1, 5, dim)

    def test_forward_method_exists(self, gaussian_memory):
        """Test that forward method exists (renamed from reconstruct)."""
        # Fit first
        patterns = torch.randn(1, 10, 3)
        gaussian_memory.fit(patterns)

        # Should have forward method
        assert hasattr(gaussian_memory, "forward")

        # Forward at single point (with batch)
        t = torch.tensor([[0.5]])  # Shape (1, 1)
        output = gaussian_memory.forward(t)

        assert output.shape == (1, 1, 3)  # (B, num_points, D)

    def test_forward_batch_consistency(self, gaussian_memory):
        """Test forward handles batches consistently."""
        # Fit with batch
        batch_size = 2
        patterns = torch.randn(batch_size, 10, 3)
        gaussian_memory.fit(patterns)

        # Forward at multiple points
        num_points = 20
        t = torch.linspace(0, 1, num_points).unsqueeze(0).expand(batch_size, -1)
        output = gaussian_memory.forward(t)

        assert output.shape == (batch_size, num_points, 3)

    def test_no_special_case_single_sample(self, rectangular_memory):
        """Test that single samples are not special-cased."""
        # Even single pattern should have batch dimension
        patterns = torch.randn(1, 20, 3)  # Batch size 1
        rectangular_memory.fit(patterns)

        # Query single point with batch
        t = torch.tensor([[0.5]])
        output = rectangular_memory.forward(t)

        # Should maintain batch dimension
        assert output.shape == (1, 1, 3)
        assert output.dim() == 3  # Always 3D

    def test_device_propagation(self, gaussian_memory):
        """Test proper device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Move memory to GPU
        gaussian_memory = gaussian_memory.cuda()

        # Fit with GPU tensors
        patterns = torch.randn(2, 10, 3).cuda()
        gaussian_memory.fit(patterns)

        # Forward should work on same device
        t = torch.linspace(0, 1, 5).unsqueeze(0).expand(2, -1).cuda()
        output = gaussian_memory.forward(t)

        assert output.device == patterns.device
        assert output.shape == (2, 5, 3)

    def test_dtype_propagation(self, rectangular_memory):
        """Test proper dtype handling."""
        # Test with float64
        patterns = torch.randn(1, 10, 3, dtype=torch.float64)
        rectangular_memory.fit(patterns)

        t = torch.linspace(0, 1, 5, dtype=torch.float64).unsqueeze(0)
        output = rectangular_memory.forward(t)

        assert output.dtype == torch.float64
        assert output.shape == (1, 5, 3)

    def test_reconstruction_quality_batched(self, gaussian_memory):
        """Test reconstruction quality with batched inputs."""
        batch_size, num_patterns, _dim = 2, 20, 2
        positions = torch.linspace(0, 1, num_patterns)

        # Create batch of smooth patterns
        patterns = torch.stack(
            [
                torch.stack(
                    [
                        torch.sin(2 * torch.pi * positions),
                        torch.cos(2 * torch.pi * positions),
                    ],
                    dim=-1,
                )
                for _ in range(batch_size)
            ]
        )

        gaussian_memory.fit(patterns, positions)

        # Reconstruct at original positions
        t_batch = positions.unsqueeze(0).expand(batch_size, -1)
        reconstructed = gaussian_memory.forward(t_batch)

        # Should approximate original patterns
        mse_threshold = 0.1
        mse = ((reconstructed - patterns) ** 2).mean()
        assert mse < mse_threshold

    def test_dataloader_compatibility(self, rectangular_memory):
        """Test compatibility with PyTorch DataLoader."""
        # Create dataset
        num_samples = 8
        datasets = []
        for _ in range(num_samples):
            patterns = torch.randn(1, 15, 4)  # Each sample has batch size 1
            datasets.append(patterns)

        dataset = TensorDataset(torch.cat(datasets))
        dataloader = DataLoader(dataset, batch_size=4)

        for batch in dataloader:
            patterns_batch = batch[0]  # Shape (4, 1, 15, 4)
            patterns_batch = patterns_batch.squeeze(1)  # Shape (4, 15, 4)

            rectangular_memory.fit(patterns_batch)
            assert rectangular_memory.is_fitted

            # Forward should work with batch
            t = torch.linspace(0, 1, 10).unsqueeze(0).expand(4, -1)
            output = rectangular_memory.forward(t)
            assert output.shape == (4, 10, 4)
            break  # Test first batch only


class TestContinuousHopfield:
    """Test ContinuousHopfield network with PyTorch idioms."""

    @pytest.fixture
    def config(self):
        """Create standard configuration."""
        return ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=8, basis_type="rectangular"),
            beta=2.0,
            regularization=0.1,
            integration_points=100,
            num_iterations=3,
        )

    @pytest.fixture
    def hopfield(self, config):
        """Create continuous Hopfield network."""
        return ContinuousHopfield(config)

    def test_initialization(self, hopfield, config):
        """Test proper initialization."""
        assert hopfield.config == config
        assert isinstance(hopfield.basis, nn.Module)
        assert isinstance(hopfield.memory, nn.Module)
        assert isinstance(hopfield.energy_fn, nn.Module)

        # Should have reset_parameters
        assert hasattr(hopfield, "reset_parameters")

    def test_forward_always_batch(self, hopfield):
        """Test forward always expects batch dimensions."""
        # Both memories and queries should have batch dimension
        memories = torch.randn(2, 16, 5)  # (B, L, D)
        queries = torch.randn(2, 3, 5)  # (B, Q, D)

        outputs, info = hopfield(memories, queries)

        assert outputs.shape == (2, 3, 5)  # (B, Q, D)
        assert isinstance(info, dict)

    def test_no_single_query_special_case(self, hopfield):
        """Test no special handling for single queries."""
        memories = torch.randn(1, 16, 5)  # Batch size 1
        query = torch.randn(1, 1, 5)  # Single query with batch

        output, info = hopfield(memories, query)

        # Should maintain all dimensions
        assert output.shape == (1, 1, 5)
        assert output.dim() == 3

    def test_consistent_batch_handling(self, hopfield):
        """Test memories and queries must have same batch size."""
        memories = torch.randn(2, 10, 4)
        queries = torch.randn(3, 5, 4)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            hopfield(memories, queries)

    def test_forward_returns_tuple(self, hopfield):
        """Test forward always returns tuple for consistency."""
        memories = torch.randn(2, 12, 3)
        queries = torch.randn(2, 4, 3)

        # Should always return tuple (output, info)
        result = hopfield(memories, queries)
        assert isinstance(result, tuple)
        output, info = result

        assert output.shape == (2, 4, 3)
        assert isinstance(info, dict)
        assert "num_iterations" in info

    def test_private_methods(self, hopfield):
        """Test that internal methods are private (start with _)."""
        # These should be private
        assert hasattr(hopfield, "_analytical_update")
        assert hasattr(hopfield, "_iterate")

        # Public API should be minimal
        public_methods = [
            m
            for m in dir(hopfield)
            if not m.startswith("_") and callable(getattr(hopfield, m))
        ]

        # Should only have essential public methods
        essential_public = {"forward", "energy", "reset_parameters"}
        for method in essential_public:
            assert method in public_methods

    def test_device_handling(self, hopfield):
        """Test proper device propagation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        hopfield = hopfield.cuda()
        memories = torch.randn(2, 10, 4).cuda()
        queries = torch.randn(2, 3, 4).cuda()

        output, info = hopfield(memories, queries)

        assert output.device == memories.device
        assert output.shape == (2, 3, 4)

    def test_dtype_consistency(self, hopfield):
        """Test dtype propagation through network."""
        memories = torch.randn(1, 8, 3, dtype=torch.float64)
        queries = torch.randn(1, 2, 3, dtype=torch.float64)

        output, info = hopfield(memories, queries)

        assert output.dtype == torch.float64
        assert output.shape == (1, 2, 3)

    def test_dataloader_integration(self, hopfield):
        """Test integration with DataLoader."""
        # Create dataset of memory-query pairs
        num_samples = 16
        memory_data = torch.randn(num_samples, 10, 4)
        query_data = torch.randn(num_samples, 3, 4)

        dataset = TensorDataset(memory_data, query_data)
        dataloader = DataLoader(dataset, batch_size=4)

        for memories, queries in dataloader:
            # Both have shape (4, ...)
            output, info = hopfield(memories, queries)

            assert output.shape == (4, 3, 4)
            assert info["num_iterations"] == hopfield.config.num_iterations
            break  # Test first batch

    def test_energy_with_batch(self, hopfield):
        """Test energy computation with batched inputs."""
        memories = torch.randn(2, 10, 4)
        queries = torch.randn(2, 5, 4)

        # Fit memory first
        hopfield.memory.fit(memories)

        energies = hopfield.energy(queries)

        assert energies.shape == (2, 5)  # (B, Q)
        assert torch.isfinite(energies).all()

    def test_gradient_flow_batched(self, hopfield):
        """Test gradient flow with batched operations."""
        memories = torch.randn(2, 8, 5, requires_grad=True)
        queries = torch.randn(2, 3, 5, requires_grad=True)

        outputs, _ = hopfield(memories, queries)
        loss = outputs.sum()
        loss.backward()

        assert memories.grad is not None
        assert queries.grad is not None
        assert memories.grad.shape == memories.shape
        assert queries.grad.shape == queries.shape


class TestContinuousHopfieldEnergy:
    """Test ContinuousHopfieldEnergy with proper conventions."""

    @pytest.fixture
    def energy_fn(self):
        """Create energy function."""
        return ContinuousHopfieldEnergy(beta=1.0, integration_points=100)

    def test_initialization(self, energy_fn):
        """Test proper initialization."""
        assert energy_fn.beta == 1.0
        assert energy_fn.integration_points == 100

        # Memory should be properly initialized (not None)
        assert hasattr(energy_fn, "memory_fn")

    def test_batch_handling(self, energy_fn):
        """Test energy handles batches properly."""
        # Set memory function
        basis = RectangularBasis(num_basis=4)
        memory = ContinuousMemory(basis, regularization=0.1)
        patterns = torch.randn(2, 10, 3)
        memory.fit(patterns)
        energy_fn.set_memory(memory)

        # Compute energy for batch of states
        states = torch.randn(2, 4, 3)  # (B, Q, D)
        energies = energy_fn(states)

        assert energies.shape == (2, 4)  # (B, Q)

    def test_gradient_methods_private(self, energy_fn):
        """Test gradient methods are private."""
        assert hasattr(energy_fn, "_grad_concave")
        # _grad_convex not needed since convex gradient is trivial (identity)


class TestContinuousAttention:
    """Test ContinuousAttention with PyTorch conventions."""

    @pytest.fixture
    def attention(self):
        """Create continuous attention module."""
        basis = RectangularBasis(num_basis=8)
        embed_dim = 64
        num_heads = 8
        return ContinuousAttention(
            embed_dim=embed_dim, num_heads=num_heads, basis=basis, beta=1.0
        )

    def test_initialization(self, attention):
        """Test proper initialization."""
        assert attention.embed_dim == 64
        assert attention.num_heads == 8
        assert attention.head_dim == 8

        # Should be proper nn.Module
        assert isinstance(attention, nn.Module)
        assert hasattr(attention, "reset_parameters")

    def test_forward_batch_only(self, attention):
        """Test forward requires batch dimension."""
        batch_size, query_len, key_len = 2, 3, 10
        embed_dim = 64

        query = torch.randn(batch_size, query_len, embed_dim)
        key = torch.randn(batch_size, key_len, embed_dim)
        value = torch.randn(batch_size, key_len, embed_dim)

        output = attention(query, key, value)

        assert output.shape == (batch_size, query_len, embed_dim)

    def test_no_explicit_loops(self):
        """Test that forward uses vectorized operations."""
        # This test would need to check implementation
        # For now, we ensure it handles multiple batches efficiently
        basis = RectangularBasis(num_basis=4)
        attention = ContinuousAttention(32, 4, basis, beta=1.0)

        # Larger batch to test efficiency
        query = torch.randn(8, 5, 32)
        key = torch.randn(8, 10, 32)
        value = torch.randn(8, 10, 32)

        import time

        start = time.time()
        output = attention(query, key, value)
        elapsed = time.time() - start

        assert output.shape == (8, 5, 32)
        # Should be reasonably fast (vectorized)
        assert elapsed < 1.0  # Less than 1 second for small inputs

    def test_device_dtype_handling(self, attention):
        """Test proper device and dtype propagation."""
        if torch.cuda.is_available():
            attention = attention.cuda()
            # Module has float32 by default, inputs can be different dtype
            query = torch.randn(2, 3, 64).cuda()
            key = torch.randn(2, 10, 64).cuda()
            value = torch.randn(2, 10, 64).cuda()

            output = attention(query, key, value)

            assert output.device == query.device
            # Output dtype matches module dtype (standard PyTorch behavior)
            assert output.dtype == next(attention.parameters()).dtype


class TestBatchedVideoReconstruction:
    """Test video reconstruction use case with proper batching."""

    def test_video_batch_processing(self):
        """Test processing multiple videos in batch."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=32, basis_type="rectangular"),
            beta=1.0,
            regularization=0.5,
            num_iterations=3,
        )
        hopfield = ContinuousHopfield(config)

        # Batch of videos
        batch_size = 4
        num_frames = 128
        feature_dim = 224 * 224 * 3  # Flattened frames

        # Simulate batch of video memories and masked queries
        memories = torch.randn(batch_size, num_frames, feature_dim)
        queries = torch.randn(batch_size, num_frames, feature_dim)

        outputs, info = hopfield(memories, queries)

        assert outputs.shape == (batch_size, num_frames, feature_dim)
        assert info["num_iterations"] == 3

    def test_mixed_batch_sizes(self):
        """Test handling different batch sizes in same session."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=16, basis_type="gaussian"),
            beta=1.0,
        )
        hopfield = ContinuousHopfield(config)

        # First batch
        memories1 = torch.randn(2, 64, 100)
        queries1 = torch.randn(2, 64, 100)
        output1, _ = hopfield(memories1, queries1)
        assert output1.shape == (2, 64, 100)

        # Different batch size
        memories2 = torch.randn(5, 64, 100)
        queries2 = torch.randn(5, 64, 100)
        output2, _ = hopfield(memories2, queries2)
        assert output2.shape == (5, 64, 100)


class TestModuleComposition:
    """Test that modules compose well with other PyTorch layers."""

    def test_sequential_composition(self):
        """Test using ContinuousHopfield in nn.Sequential-like setup."""

        class VideoReconstructionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(784, 256)
                self.hopfield = ContinuousHopfield(
                    ContinuousHopfieldConfig(
                        basis_config=BasisConfig(num_basis=16),
                        beta=1.0,
                    )
                )
                self.decoder = nn.Linear(256, 784)

            def forward(self, x):
                # x shape: (B, L, 784)
                encoded = self.encoder(x)  # (B, L, 256)

                # Use hopfield for denoising/reconstruction
                reconstructed, _ = self.hopfield(encoded, encoded)

                return self.decoder(reconstructed)  # (B, L, 784)

        model = VideoReconstructionModel()
        input_data = torch.randn(2, 32, 784)
        output = model(input_data)

        assert output.shape == (2, 32, 784)

    def test_parallel_data_handling(self):
        """Test with DataParallel (multi-GPU if available)."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=8),
            beta=1.0,
        )
        hopfield = ContinuousHopfield(config)

        if torch.cuda.device_count() > 1:
            hopfield = nn.DataParallel(hopfield)
            memories = torch.randn(8, 64, 128).cuda()
            queries = torch.randn(8, 64, 128).cuda()

            output, _ = hopfield(memories, queries)
            assert output.shape == (8, 64, 128)
        else:
            # Test passes if not enough GPUs
            pass


def test_mathematical_correctness_preserved():
    """Ensure mathematical behavior is preserved with new conventions."""
    # Test that the core mathematical operations still work correctly
    config = ContinuousHopfieldConfig(
        basis_config=BasisConfig(num_basis=5, basis_type="rectangular"),
        beta=1.0,
        regularization=0.5,
        num_iterations=3,
    )
    hopfield = ContinuousHopfield(config)

    # Create patterns that should be recalled
    memories = torch.eye(5).unsqueeze(0) * 2  # (1, 5, 5) - orthogonal patterns

    # Query with noisy version
    query_idx = 2
    queries = memories[:, [query_idx]] + 0.1 * torch.randn(1, 1, 5)

    output, _ = hopfield(memories, queries)

    # Should recall the correct pattern
    similarities = torch.cosine_similarity(output[0, 0], memories[0], dim=1)
    assert similarities.argmax() == query_idx  # Correct recall


def test_ridge_regression_correctness():
    """Test that ridge regression math is still correct."""
    basis = RectangularBasis(num_basis=4)
    memory = ContinuousMemory(basis, regularization=0.5)

    # Known patterns
    patterns = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]], [[0.0, -1.0]]]
    ).transpose(0, 1)  # Shape (1, 4, 2)

    memory.fit(patterns)

    # Reconstruct at original positions
    positions = torch.linspace(0, 1, 4).unsqueeze(0)  # (1, 4)
    reconstructed = memory.forward(positions)

    # Should approximate original patterns
    assert torch.allclose(reconstructed, patterns, atol=0.5)
