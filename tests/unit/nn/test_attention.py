"""Unit tests for attention modules."""

import pytest
import torch

from associative.nn.modules import EnergyAttention, GraphEnergyAttention
from associative.nn.modules.config import EnergyAttentionConfig
from tests.conftest import TOLERANCE_ENERGY_DIFF, TOLERANCE_INIT_STD


class TestEnergyAttention:
    """Test EnergyAttention module."""

    @pytest.fixture
    def config(self, embed_dim, num_heads, qk_dim):
        """Create attention config."""
        return EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=False
        )

    @pytest.fixture
    def attention(self, config, device):
        """Create attention module."""
        return EnergyAttention(config).to(device)

    def test_initialization(self, attention, config):
        """Test proper initialization."""
        assert attention.embed_dim == config.embed_dim
        assert attention.num_heads == config.num_heads
        assert attention.qk_dim == config.qk_dim

        # Check weight shapes
        assert attention.query_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )
        assert attention.key_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )

        # Check initialization scale
        assert (
            attention.query_proj.std().item() < TOLERANCE_INIT_STD
        )  # Should be ~0.002
        assert attention.key_proj.std().item() < TOLERANCE_INIT_STD

    def test_forward_shape(self, attention, batch_size, seq_length, embed_dim, device):
        """Test forward pass output shape."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        energy = attention(x)

        assert energy.shape == ()  # Scalar
        assert energy.dtype == x.dtype
        assert energy.requires_grad

    def test_energy_computation(self, attention, device):
        """Test energy computation correctness."""
        # Create simple input where we can verify energy
        x = torch.eye(3, 64, device=device).unsqueeze(0)  # [1, 3, 64]

        with torch.no_grad():
            # Set weights to identity-like for predictable behavior
            attention.query_proj.zero_()
            attention.key_proj.zero_()
            attention.query_proj[:, :, :16] = torch.eye(16).unsqueeze(0)
            attention.key_proj[:, :, :16] = torch.eye(16).unsqueeze(0)

        energy = attention(x)

        # Energy should be negative (logsumexp formulation)
        assert energy.item() < 0

    def test_attention_mask(
        self, attention, batch_size, seq_length, embed_dim, num_heads, device
    ):
        """Test attention masking."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        energy_unmasked = attention(x)
        energy_masked = attention(x, mask)

        # Masked energy should be different
        # Due to the large scale of energies, we check relative difference
        rel_diff = abs((energy_unmasked - energy_masked) / energy_unmasked)
        assert rel_diff > TOLERANCE_ENERGY_DIFF  # At least 0.0001% different

    def test_gradient_flow(self, attention, batch_size, seq_length, embed_dim, device):
        """Test gradient flow through attention."""
        x = torch.randn(
            batch_size, seq_length, embed_dim, device=device, requires_grad=True
        )

        energy = attention(x)
        energy.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() > 0  # Non-zero gradients

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_option(self, embed_dim, num_heads, qk_dim, bias, device):
        """Test attention with and without bias."""
        config = EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=bias
        )
        attention = EnergyAttention(config).to(device)

        if bias:
            assert attention.query_bias is not None
            assert attention.key_bias is not None
            assert attention.query_bias.shape == (qk_dim,)
            assert attention.key_bias.shape == (qk_dim,)
        else:
            assert attention.query_bias is None
            assert attention.key_bias is None

    def test_numerical_stability(self, attention, device):
        """Test numerical stability with extreme inputs."""
        # Very large inputs
        x_large = torch.randn(1, 5, 64, device=device) * 100
        energy_large = attention(x_large)
        assert torch.isfinite(energy_large)

        # Very small inputs
        x_small = torch.randn(1, 5, 64, device=device) * 0.001
        energy_small = attention(x_small)
        assert torch.isfinite(energy_small)


class TestGraphEnergyAttention:
    """Test GraphEnergyAttention module."""

    @pytest.fixture
    def config(self, embed_dim, num_heads, qk_dim):
        """Create attention config."""
        return EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=False
        )

    @pytest.fixture
    def attention(self, config, device):
        """Create graph attention module."""
        return GraphEnergyAttention(config).to(device)

    def test_initialization(self, attention, config):
        """Test proper initialization."""
        # Check parameter names
        assert hasattr(attention, "key_proj")
        assert hasattr(attention, "query_proj")
        assert hasattr(attention, "head_mix")  # Head mixing weights

        # Check shapes
        assert attention.key_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )
        assert attention.query_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )
        assert attention.head_mix.shape == (config.num_heads, config.num_heads)

    def test_batched_forward(
        self, attention, batch_size, seq_length, embed_dim, device
    ):
        """Test batched forward pass."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        energy = attention(x)

        assert energy.shape == ()
        assert energy.dtype == x.dtype

    def test_unbatched_forward(self, attention, seq_length, embed_dim, device):
        """Test unbatched forward pass."""
        x = torch.randn(seq_length, embed_dim, device=device)
        energy = attention(x)

        assert energy.shape == ()
        assert energy.dtype == x.dtype

    def test_adjacency_masking(
        self, attention, seq_length, embed_dim, num_heads, device
    ):
        """Test forward with adjacency matrix."""
        x = torch.randn(seq_length, embed_dim, device=device)

        # Create sparse adjacency matrix
        adj = torch.zeros(seq_length, seq_length, num_heads, device=device)
        # Add some connections
        adj[0, 1, :] = 1.0
        adj[1, 0, :] = 1.0
        adj[1, 2, :] = 1.0
        adj[2, 1, :] = 1.0

        energy_no_adj = attention(x)
        energy_with_adj = attention(x, adj)

        # Energies should be different
        assert not torch.allclose(energy_no_adj, energy_with_adj)

    def test_head_mixing(self, attention, seq_length, embed_dim, device):
        """Test head mixing weights effect."""
        x = torch.randn(seq_length, embed_dim, device=device)

        # Store original energy
        energy_orig = attention(x).item()

        # Modify head mixing weights
        with torch.no_grad():
            attention.head_mix.data = torch.eye(attention.num_heads, device=device)

        energy_identity = attention(x).item()

        # Should be different with different head mixing
        assert abs(energy_orig - energy_identity) > TOLERANCE_ENERGY_DIFF

    def test_gradient_flow_with_adjacency(self, attention, device):
        """Test gradient flow with adjacency matrix."""
        x = torch.randn(5, 64, device=device, requires_grad=True)
        adj = torch.ones(5, 5, 4, device=device) * 0.5

        energy = attention(x, adj)
        energy.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_empty_adjacency_handling(self, attention, device):
        """Test handling of empty adjacency (all zeros)."""
        x = torch.randn(5, 64, device=device)
        adj = torch.zeros(5, 5, 4, device=device)

        # Should not crash and return finite energy
        energy = attention(x, adj)
        assert torch.isfinite(energy)
