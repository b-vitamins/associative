"""Comprehensive tests for continuous Hopfield networks."""

import pytest
import torch

from associative.nn.modules.basis import GaussianBasis, RectangularBasis
from associative.nn.modules.config import (
    BasisConfig,
    CCCPConfig,
    ContinuousHopfieldConfig,
)
from associative.nn.modules.continuous import (
    ContinuousAttention,
    ContinuousHopfield,
    ContinuousMemory,
)


class TestContinuousMemory:
    """Test ContinuousMemory implementation."""

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
        """Test proper initialization."""
        expected_regularization = 0.1
        assert rectangular_memory.regularization == expected_regularization
        assert not rectangular_memory.is_fitted
        assert rectangular_memory.coefficients is None

        # Test invalid regularization
        basis = RectangularBasis(num_basis=4)
        with pytest.raises(ValueError, match="regularization must be positive"):
            ContinuousMemory(basis, regularization=0)
        with pytest.raises(ValueError, match="regularization must be positive"):
            ContinuousMemory(basis, regularization=-0.5)

    def test_fit_uniform_positions(self, rectangular_memory):
        """Test fitting with uniform positions."""
        # Create simple patterns
        num_patterns, dim = 10, 3
        patterns = torch.randn(num_patterns, dim)

        rectangular_memory.fit(patterns)

        assert rectangular_memory.is_fitted
        assert rectangular_memory.coefficients is not None
        assert rectangular_memory.coefficients.shape == (4, dim)  # (num_basis, D)

    def test_fit_custom_positions(self, gaussian_memory):
        """Test fitting with custom positions."""
        num_patterns, dim = 8, 4
        patterns = torch.randn(num_patterns, dim)
        positions = torch.linspace(0.1, 0.9, num_patterns)

        gaussian_memory.fit(patterns, positions)

        assert gaussian_memory.is_fitted
        assert gaussian_memory.coefficients.shape == (5, dim)

    def test_fit_batch(self, rectangular_memory):
        """Test fitting with batched patterns."""
        # Batch of patterns
        batch_size, num_patterns, dim = 2, 10, 3
        patterns = torch.randn(batch_size, num_patterns, dim)

        rectangular_memory.fit(patterns)

        # Should fit to flattened version or handle batches appropriately
        assert rectangular_memory.is_fitted

    def test_reconstruct_scalar(self, gaussian_memory):
        """Test reconstruction at scalar time point."""
        # Fit first
        patterns = torch.randn(10, 3)
        gaussian_memory.fit(patterns)

        # Reconstruct at single point
        t = torch.tensor(0.5)
        reconstructed = gaussian_memory.reconstruct(t)

        assert reconstructed.shape == (3,)  # (D,)

    def test_reconstruct_batch(self, gaussian_memory):
        """Test reconstruction at multiple time points."""
        # Fit first
        patterns = torch.randn(10, 3)
        gaussian_memory.fit(patterns)

        # Reconstruct at multiple points
        t = torch.linspace(0, 1, 20)
        reconstructed = gaussian_memory.reconstruct(t)

        assert reconstructed.shape == (20, 3)  # (num_points, D)

    def test_reconstruct_not_fitted(self, rectangular_memory):
        """Test that reconstruct raises error if not fitted."""
        t = torch.tensor(0.5)

        with pytest.raises(RuntimeError, match="not been fitted"):
            rectangular_memory.reconstruct(t)

    def test_reconstruction_quality(self, gaussian_memory):
        """Test that reconstruction approximates original patterns."""
        # Create smooth patterns
        num_patterns, _dim = 20, 2
        positions = torch.linspace(0, 1, num_patterns)
        patterns = torch.stack(
            [torch.sin(2 * torch.pi * positions), torch.cos(2 * torch.pi * positions)],
            dim=-1,
        )

        gaussian_memory.fit(patterns, positions)

        # Reconstruct at original positions
        reconstructed = gaussian_memory.reconstruct(positions)

        # Should approximate original patterns
        mse_threshold = 0.1
        mse = ((reconstructed - patterns) ** 2).mean()
        assert mse < mse_threshold  # Reasonable approximation

    def test_compress_ratio(self, rectangular_memory):
        """Test compression ratio computation."""
        patterns = torch.randn(20, 3)
        rectangular_memory.fit(patterns)

        ratio = rectangular_memory.compress_ratio()
        expected_ratio = 0.2
        assert ratio == 4 / 20  # num_basis / num_patterns
        assert ratio == expected_ratio

    def test_differentiability(self, gaussian_memory):
        """Test that reconstruction is differentiable."""
        patterns = torch.randn(10, 3)
        gaussian_memory.fit(patterns)

        t = torch.linspace(0, 1, 5, requires_grad=True)
        reconstructed = gaussian_memory.reconstruct(t)

        loss = reconstructed.sum()
        loss.backward()

        assert t.grad is not None


class TestContinuousHopfield:
    """Test ContinuousHopfield network implementation."""

    @pytest.fixture
    def config(self):
        """Create standard configuration."""
        return ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=8, basis_type="rectangular"),
            beta=2.0,
            regularization=0.1,
            integration_points=100,
            use_analytical_update=True,
        )

    @pytest.fixture
    def hopfield(self, config):
        """Create continuous Hopfield network."""
        return ContinuousHopfield(config)

    def test_initialization(self, hopfield, config):
        """Test proper initialization."""
        assert hopfield.config == config
        assert hopfield.basis is not None
        assert hopfield.memory is not None
        assert hopfield.optimizer is not None
        assert hopfield.energy_fn is not None

    def test_forward_single_query(self, hopfield):
        """Test forward pass with single query."""
        memories = torch.randn(16, 5)  # L=16, D=5
        query = torch.randn(5)

        output = hopfield(memories, query)

        assert output.shape == (5,)  # Same as query

    def test_forward_batch_queries(self, hopfield):
        """Test forward pass with batch of queries."""
        memories = torch.randn(16, 5)
        queries = torch.randn(3, 5)  # Batch of 3

        outputs = hopfield(memories, queries)

        assert outputs.shape == (3, 5)

    def test_forward_with_positions(self, hopfield):
        """Test forward pass with custom positions."""
        memories = torch.randn(10, 4)
        queries = torch.randn(2, 4)
        positions = torch.linspace(0.2, 0.8, 10)

        outputs = hopfield(memories, queries, positions=positions)

        assert outputs.shape == (2, 4)

    def test_forward_with_info(self, hopfield):
        """Test forward pass returning optimization info."""
        memories = torch.randn(12, 3)
        queries = torch.randn(2, 3)

        outputs, info = hopfield(memories, queries, return_info=True)

        assert outputs.shape == (2, 3)
        assert isinstance(info, dict)
        assert "trajectories" in info or "optimization_results" in info

    def test_analytical_update(self, hopfield):
        """Test analytical CCCP update."""
        memories = torch.randn(10, 4)
        queries = torch.randn(3, 4)

        # Fit memory first
        hopfield.memory.fit(memories)

        # Analytical update
        updated = hopfield.analytical_update(queries)

        assert updated.shape == queries.shape
        assert not torch.allclose(updated, queries)  # Should change

    def test_analytical_vs_iterative(self):
        """Test that analytical and iterative updates converge to same point."""
        # Analytical config
        config_analytical = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=6), beta=1.0, use_analytical_update=True
        )
        hopfield_analytical = ContinuousHopfield(config_analytical)

        # Iterative config
        config_iterative = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=6),
            beta=1.0,
            use_analytical_update=False,
            cccp_config=CCCPConfig(max_iterations=100, tolerance=1e-6),
        )
        hopfield_iterative = ContinuousHopfield(config_iterative)

        memories = torch.randn(8, 3)
        query = torch.randn(3)

        output_analytical = hopfield_analytical(memories, query)
        output_iterative = hopfield_iterative(memories, query)

        # Should converge to similar points
        assert torch.allclose(output_analytical, output_iterative, atol=1e-3)

    def test_energy_computation(self, hopfield):
        """Test energy computation."""
        memories = torch.randn(10, 4)
        queries = torch.randn(2, 4)

        # Fit memory
        hopfield.memory.fit(memories)

        energies = hopfield.energy(queries)

        assert energies.shape == (2,)
        assert torch.isfinite(energies).all()  # Energies should be finite (no NaN/inf)

    def test_iterate(self, hopfield):
        """Test fixed number of iterations."""
        memories = torch.randn(12, 5)
        queries = torch.randn(3, 5)

        # Fit memory
        hopfield.memory.fit(memories)

        # Single iteration
        output1 = hopfield.iterate(queries, num_iterations=1)
        assert output1.shape == queries.shape

        # Multiple iterations
        output5 = hopfield.iterate(queries, num_iterations=5)
        assert output5.shape == queries.shape

        # More iterations should converge further
        energy1 = hopfield.energy(output1)
        energy5 = hopfield.energy(output5)
        assert (energy5 <= energy1).all()  # Energy should decrease

    def test_memory_compression(self):
        """Test memory compression ratio."""
        compression_target = 0.25
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=5),
            memory_compression=compression_target,  # Use 25% of original size
        )
        hopfield = ContinuousHopfield(config)

        memories = torch.randn(20, 4)  # L=20
        queries = torch.randn(2, 4)

        hopfield(memories, queries)

        # Should use compression ratio to determine num_basis
        assert hopfield.memory.compress_ratio() <= compression_target

    def test_associative_recall(self, hopfield):
        """Test associative memory recall property."""
        # Store distinct patterns
        memories = torch.eye(5) * 2  # 5 orthogonal patterns

        # Query with noisy version of second pattern
        query = memories[1] + 0.1 * torch.randn(5)

        output = hopfield(memories, query)

        # Should recall pattern closest to query
        similarities = (
            memories @ output / (torch.norm(memories, dim=1) * torch.norm(output))
        )
        assert similarities.argmax() == 1  # Should recall second pattern


class TestContinuousAttention:
    """Test ContinuousAttention mechanism."""

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
        embed_dim = 64
        num_heads = 8
        head_dim = 8
        assert attention.embed_dim == embed_dim
        assert attention.num_heads == num_heads
        assert attention.head_dim == head_dim  # 64 / 8
        assert attention.beta == 1.0
        assert attention.basis is not None

    def test_forward_single_query(self, attention):
        """Test attention with single query."""
        batch_size, query_len, key_len, embed_dim = 2, 1, 10, 64
        query = torch.randn(batch_size, query_len, embed_dim)
        key = torch.randn(batch_size, key_len, embed_dim)
        value = torch.randn(batch_size, key_len, embed_dim)

        output = attention(query, key, value)

        assert output.shape == (batch_size, query_len, embed_dim)

    def test_forward_multiple_queries(self, attention):
        """Test attention with multiple queries."""
        batch_size, query_len, key_len, embed_dim = 2, 5, 10, 64
        query = torch.randn(batch_size, query_len, embed_dim)
        key = torch.randn(batch_size, key_len, embed_dim)
        value = torch.randn(batch_size, key_len, embed_dim)

        output = attention(query, key, value)

        assert output.shape == (batch_size, query_len, embed_dim)

    def test_forward_with_positions(self, attention):
        """Test attention with custom key positions."""
        batch_size, query_len, key_len, embed_dim = 1, 3, 8, 64
        query = torch.randn(batch_size, query_len, embed_dim)
        key = torch.randn(batch_size, key_len, embed_dim)
        value = torch.randn(batch_size, key_len, embed_dim)
        key_positions = torch.linspace(0.1, 0.9, key_len)

        output = attention(query, key, value, key_positions=key_positions)

        assert output.shape == (batch_size, query_len, embed_dim)

    def test_attention_density(self, attention):
        """Test attention probability density computation."""
        embed_dim = 64
        query = torch.randn(embed_dim)  # Single query
        t = torch.linspace(0, 1, 100)

        # First need to fit keys
        keys = torch.randn(10, embed_dim)
        attention.memory = ContinuousMemory(attention.basis, regularization=0.1)
        attention.memory.fit(keys)

        density = attention.compute_attention_density(query, t)

        assert density.shape == (100,)
        assert (density >= 0).all()  # Probability density

        # Should integrate to approximately 1
        integral = torch.trapz(density, t)
        assert torch.allclose(integral, torch.tensor(1.0), atol=0.1)

    def test_temperature_effect(self):
        """Test that temperature affects attention sharpness."""
        basis = GaussianBasis(num_basis=6)
        embed_dim = 64
        num_heads = 8

        # Low temperature (sharp attention)
        attention_sharp = ContinuousAttention(embed_dim, num_heads, basis, beta=10.0)

        # High temperature (soft attention)
        attention_soft = ContinuousAttention(embed_dim, num_heads, basis, beta=0.1)

        # Same inputs
        query = torch.randn(1, 1, embed_dim)
        key = torch.randn(1, 8, embed_dim)
        value = torch.randn(1, 8, embed_dim)

        output_sharp = attention_sharp(query, key, value)
        output_soft = attention_soft(query, key, value)

        # Both should have correct shape
        assert output_sharp.shape == (1, 1, embed_dim)
        assert output_soft.shape == (1, 1, embed_dim)

        # Outputs should differ due to temperature
        assert not torch.allclose(output_sharp, output_soft)


class TestContinuousHopfieldConfig:
    """Test ContinuousHopfieldConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        beta = 2.0
        regularization = 0.5
        integration_points = 200
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=10),
            beta=beta,
            regularization=regularization,
            integration_points=integration_points,
        )

        assert config.beta == beta
        assert config.regularization == regularization
        assert config.integration_points == integration_points
        assert config.use_analytical_update

    def test_invalid_config(self):
        """Test invalid configuration parameters."""
        basis_config = BasisConfig(num_basis=10)

        # Invalid beta
        with pytest.raises(ValueError, match="beta must be positive"):
            ContinuousHopfieldConfig(basis_config=basis_config, beta=0)

        # Invalid regularization
        with pytest.raises(ValueError, match="regularization must be positive"):
            ContinuousHopfieldConfig(basis_config=basis_config, regularization=-1)

        # Invalid integration_points
        with pytest.raises(ValueError, match="integration_points must be positive"):
            ContinuousHopfieldConfig(basis_config=basis_config, integration_points=0)

        # Invalid memory_compression
        with pytest.raises(ValueError, match="memory_compression must be in"):
            ContinuousHopfieldConfig(basis_config=basis_config, memory_compression=1.5)


class TestIntegration:
    """Integration tests for continuous Hopfield components."""

    def test_end_to_end_rectangular(self):
        """Test end-to-end with rectangular basis."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(
                num_basis=6, basis_type="rectangular", overlap=0.1
            ),
            beta=1.5,
            regularization=0.2,
            use_analytical_update=True,
        )

        hopfield = ContinuousHopfield(config)

        # Create and retrieve memories
        memories = torch.randn(12, 8)
        queries = torch.randn(4, 8)

        outputs = hopfield(memories, queries)

        assert outputs.shape == (4, 8)

        # Energy should be finite
        energies = hopfield.energy(outputs)
        assert torch.isfinite(energies).all()

    def test_end_to_end_gaussian(self):
        """Test end-to-end with Gaussian basis."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(
                num_basis=7, basis_type="gaussian", learnable=True, init_width=0.15
            ),
            beta=2.0,
            use_analytical_update=False,
            cccp_config=CCCPConfig(max_iterations=20),
        )

        hopfield = ContinuousHopfield(config)

        memories = torch.randn(14, 6)
        queries = torch.randn(3, 6)

        outputs, info = hopfield(memories, queries, return_info=True)

        assert outputs.shape == (3, 6)
        assert "optimization_results" in info

    def test_end_to_end_fourier(self):
        """Test end-to-end with Fourier basis."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(
                num_basis=8, basis_type="fourier", max_frequency=4
            ),
            beta=10.0,
        )

        hopfield = ContinuousHopfield(config)

        # Use periodic patterns
        num_patterns = 16
        positions = torch.linspace(0, 1, num_patterns)
        memories = torch.stack(
            [
                torch.sin(2 * torch.pi * positions),
                torch.cos(2 * torch.pi * positions),
                torch.sin(4 * torch.pi * positions),
            ],
            dim=-1,
        )

        queries = memories[[0, 8]] + 0.1 * torch.randn(2, 3)

        outputs = hopfield(memories, queries, positions=positions)

        assert outputs.shape == (2, 3)

        # Should recall patterns similar to queries
        similarity_threshold = 0.5
        for i in range(2):
            similarity = torch.cosine_similarity(outputs[i], memories[[0, 8]][i], dim=0)
            assert similarity > similarity_threshold  # Reasonable recall

    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        config = ContinuousHopfieldConfig(
            basis_config=BasisConfig(num_basis=4), beta=1.0
        )

        hopfield = ContinuousHopfield(config)

        memories = torch.randn(8, 5, requires_grad=True)
        queries = torch.randn(2, 5, requires_grad=True)

        outputs = hopfield(memories, queries)
        loss = outputs.sum()
        loss.backward()

        # Gradients should flow to inputs
        assert memories.grad is not None
        assert queries.grad is not None
        assert not torch.allclose(memories.grad, torch.zeros_like(memories.grad))
        assert not torch.allclose(queries.grad, torch.zeros_like(queries.grad))
