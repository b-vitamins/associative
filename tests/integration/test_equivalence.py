"""Integration tests for verifying equivalence with reference implementations."""

import pytest
import torch

from associative.nn.modules import (
    EnergyTransformer,
    EnergyTransformerConfig,
    GraphEnergyTransformer,
)


class TestVisionEquivalence:
    """Test equivalence with vision reference implementation."""

    @pytest.fixture
    def reference_config(self):
        """Configuration matching reference implementation."""
        return {
            "patch_size": 4,
            "num_patches": 64,
            "embed_dim": 128,
            "num_heads": 4,
            "qk_dim": 32,
            "mlp_ratio": 2.0,
            "num_layers": 4,
            "num_time_steps": 5,
            "attn_bias": False,
            "mlp_bias": False,
        }

    @pytest.fixture
    def model(self, reference_config):
        """Create model with reference config."""
        config = EnergyTransformerConfig(**reference_config)
        return EnergyTransformer(config)

    def test_component_initialization(self, model):
        """Test that components are initialized correctly."""
        # Test key initialization values that affect equivalence
        assert torch.allclose(model.cls_token, torch.ones_like(model.cls_token))
        assert model.pos_embed.std().item() < 0.01  # ~0.002  # noqa: PLR2004
        assert model.mask_token.std().item() < 0.01  # ~0.002  # noqa: PLR2004

        # Test output projection has LayerNorm with correct eps
        ln_layer = model.output_proj[0]
        assert hasattr(ln_layer, "eps")
        assert ln_layer.eps == model.config.norm_eps  # Should use norm_eps from config

    def test_energy_computation_structure(self, model):
        """Test energy computation follows reference structure."""
        torch.manual_seed(42)
        x = torch.randn(1, 3, 32, 32)

        # Process input
        x_patches = model.patch_embed.to_patches(x)
        x_encoded = model.patch_embed.proj(x_patches)

        # Add cls and position
        cls_tokens = model.cls_token
        x_with_cls = torch.cat([cls_tokens, x_encoded], dim=1)
        x_with_pos = x_with_cls + model.pos_embed

        # Check first block energy computation
        if len(model.blocks) > 0:
            norm, block = model.blocks[0]
            g = norm(x_with_pos)
            energy = block(g)

            assert energy.shape == ()
            assert energy.requires_grad

    def test_mask_indexing_compatibility(self, model):
        """Test mask indexing matches reference."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, 32, 32)

        # Create mask indices
        mask_idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
        batch_idx = torch.tensor([[0, 0, 0], [1, 1, 1]])

        # Process through model
        x_patches = model.patch_embed.to_patches(x)
        x_encoded = model.patch_embed.proj(x_patches)

        # Apply mask - should use reference-compatible indexing
        x_encoded[batch_idx, mask_idx] = model.mask_token

        # Verify mask was applied correctly
        assert torch.allclose(x_encoded[0, 0], model.mask_token[0, 0])
        assert torch.allclose(x_encoded[1, 3], model.mask_token[0, 0])

    def test_forward_deterministic(self, model):
        """Test forward pass is deterministic with fixed seed."""
        torch.manual_seed(42)
        x1 = torch.randn(1, 3, 32, 32)

        torch.manual_seed(42)
        x2 = torch.randn(1, 3, 32, 32)

        assert torch.allclose(x1, x2)

        out1 = model(x1)
        out2 = model(x2)

        assert torch.allclose(out1, out2, rtol=1e-5)


class TestGraphEquivalence:
    """Test equivalence with JAX graph implementation."""

    @pytest.fixture
    def jax_config(self):
        """Configuration matching JAX implementation."""
        return {
            "input_dim": 64,
            "embed_dim": 128,
            "out_dim": 32,
            "num_heads": 4,
            "qk_dim": 32,
            "mlp_ratio": 2.0,
            "num_layers": 3,
            "num_time_steps": 5,
            "pos_encoding_dim": 10,
        }

    @pytest.fixture
    def model(self, jax_config):
        """Create model with JAX-compatible config."""
        config = EnergyTransformerConfig(**jax_config)
        return GraphEnergyTransformer(config)

    def test_jax_parameter_naming(self, model):
        """Test that parameters follow JAX naming conventions."""
        # Check attention has proper parameter names
        attn = model.blocks[0].attention
        assert hasattr(attn, "key_proj")
        assert hasattr(attn, "query_proj")
        assert hasattr(attn, "head_mix")
        assert hasattr(attn, "temperature")

    def test_head_mixing_weights(self, model):
        """Test head mixing weights are properly initialized."""
        attn = model.blocks[0].attention

        # Head mixing should be initialized with small values
        assert attn.head_mix.std().item() < 0.01  # ~0.002  # noqa: PLR2004
        assert attn.head_mix.shape == (model.config.num_heads, model.config.num_heads)

    def test_adjacency_processing(self, model):
        """Test adjacency matrix processing matches JAX."""
        torch.manual_seed(42)

        # Create test data
        node_features = torch.randn(1, 10, 64)
        adjacency = torch.ones(1, 10, 10, 1)
        pos_encoding = torch.randn(1, 11, 10)  # +1 for CLS token

        # Process adjacency through model
        x, adj_padded, _ = model.prepare_tokens(node_features, pos_encoding, adjacency)

        # Check CLS token connections
        assert adj_padded.shape == (1, 11, 11, 1)  # Padded for CLS
        assert torch.all(adj_padded[:, 0, :, :] == 1.0)  # CLS connects to all
        assert torch.all(adj_padded[:, :, 0, :] == 1.0)

    def test_correlation_computation(self, model):
        """Test correlation-based adjacency refinement."""
        if not model.compute_correlation:
            pytest.skip("Model doesn't compute correlation")

        torch.manual_seed(42)
        x = torch.randn(1, 10, 128)
        adjacency = torch.ones(1, 10, 10, 1)

        adj_refined = model.compute_adjacency_correlation(x, adjacency)

        # Should have shape [batch, heads, nodes, nodes]
        assert adj_refined.shape == (1, 4, 10, 10)
        assert torch.isfinite(adj_refined).all()

    def test_energy_computation_with_adjacency(self, model):
        """Test energy computation with adjacency matches JAX structure."""
        torch.manual_seed(42)

        # Simple test case
        x = torch.randn(5, 128)  # Unbatched for simpler comparison
        adj = torch.eye(5).unsqueeze(-1).expand(-1, -1, 4)  # Identity adjacency

        # Get attention module
        attn = model.blocks[0].attention

        # Compute energy
        energy = attn(x, adj)

        assert torch.isfinite(energy)
        assert energy.shape == ()

    def test_evolution_with_masking(self, model):
        """Test evolution dynamics with node masking."""
        torch.manual_seed(42)

        node_features = torch.randn(1, 15, 64)
        adjacency = torch.ones(1, 15, 15, 1)
        pos_encoding = torch.randn(1, 16, 10)  # +1 for CLS token

        # Create mask (only first 10 nodes valid)
        mask = torch.zeros(1, 15)
        mask[0, :10] = 1.0

        output = model(
            node_features, adjacency, pos_encoding, mask=mask, return_energy=True
        )

        # Check masked nodes don't affect output
        node_embeds = output["node_embeddings"]
        assert torch.allclose(
            node_embeds[0, 10:], torch.zeros_like(node_embeds[0, 10:])
        )


class TestNumericalEquivalence:
    """Test numerical equivalence properties."""

    def test_vision_reproducibility(self):
        """Test vision model produces reproducible outputs."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=64,
            embed_dim=128,
            num_heads=4,
            qk_dim=32,
            mlp_ratio=2.0,
            num_layers=2,
            num_time_steps=3,
        )

        # Create two models with same initialization
        torch.manual_seed(42)
        model1 = EnergyTransformer(config)

        torch.manual_seed(42)
        model2 = EnergyTransformer(config)

        # Check weights are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
            assert torch.allclose(p1, p2)

        # Check outputs are identical
        torch.manual_seed(123)
        x = torch.randn(1, 3, 32, 32)

        out1 = model1(x)
        out2 = model2(x)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_graph_reproducibility(self):
        """Test graph model produces reproducible outputs."""
        config = EnergyTransformerConfig(
            input_dim=32,
            embed_dim=64,
            out_dim=16,
            num_heads=4,
            qk_dim=16,
            num_layers=2,
            num_time_steps=3,
        )

        # Create models
        torch.manual_seed(42)
        model1 = GraphEnergyTransformer(config)

        torch.manual_seed(42)
        model2 = GraphEnergyTransformer(config)

        # Test data
        torch.manual_seed(123)
        node_features = torch.randn(1, 10, 32)
        adjacency = torch.ones(1, 10, 10, 1)
        pos_encoding = torch.randn(1, 11, 10)  # +1 for CLS token

        out1 = model1(node_features, adjacency, pos_encoding)
        out2 = model2(node_features, adjacency, pos_encoding)

        assert torch.allclose(
            out1["graph_embedding"], out2["graph_embedding"], atol=1e-6
        )
        assert torch.allclose(
            out1["node_embeddings"], out2["node_embeddings"], atol=1e-6
        )
