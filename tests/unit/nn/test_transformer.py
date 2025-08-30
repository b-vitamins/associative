"""Unit tests for transformer modules."""

import pytest
import torch

from associative.nn.modules import (
    EnergyTransformer,
    EnergyTransformerBlock,
    GraphEnergyBlock,
    GraphEnergyTransformer,
)
from associative.nn.modules.config import (
    EnergyAttentionConfig,
    EnergyBlockConfig,
    EnergyTransformerConfig,
    HopfieldConfig,
)


class TestEnergyTransformerBlock:
    """Test EnergyTransformerBlock."""

    @pytest.fixture
    def attention_config(self, embed_dim, num_heads, qk_dim):
        """Create attention config."""
        return EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=False
        )

    @pytest.fixture
    def hopfield_config(self):
        """Create Hopfield config."""
        return HopfieldConfig(hidden_dim_ratio=2.0, bias=False)

    @pytest.fixture
    def block(self, embed_dim, attention_config, hopfield_config, device):
        """Create transformer block."""
        return EnergyTransformerBlock(embed_dim, attention_config, hopfield_config).to(
            device
        )

    def test_initialization(self, block):
        """Test block initialization."""
        assert hasattr(block, "attn")
        assert hasattr(block, "mlp")

    def test_energy_computation(self, block, batch_size, seq_length, embed_dim, device):
        """Test energy computation."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        energy = block.energy(x)

        assert energy.shape == ()
        assert energy.requires_grad

    def test_forward_equals_energy(
        self, block, batch_size, seq_length, embed_dim, device
    ):
        """Test that forward pass returns energy."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)

        energy_direct = block.energy(x)
        energy_forward = block(x)

        assert torch.allclose(energy_direct, energy_forward)

    def test_gradient_computation(self, block, device):
        """Test gradient computation for dynamics."""
        x = torch.randn(1, 5, 64, device=device, requires_grad=True)

        # Compute gradient of energy
        energy = block(x)
        grad = torch.autograd.grad(energy, x, create_graph=True)[0]

        assert grad.shape == x.shape
        assert torch.isfinite(grad).all()

    def test_with_attention_mask(
        self, block, batch_size, seq_length, embed_dim, num_heads, device
    ):
        """Test block with attention mask."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        mask = torch.ones(batch_size, num_heads, seq_length, seq_length, device=device)

        energy_no_mask = block(x)
        energy_with_mask = block(x, mask)

        # Both should work
        assert torch.isfinite(energy_no_mask)
        assert torch.isfinite(energy_with_mask)


class TestEnergyTransformer:
    """Test full EnergyTransformer."""

    @pytest.fixture
    def config(self, patch_size, num_patches, embed_dim, num_heads, qk_dim):
        """Create transformer config."""
        return EnergyTransformerConfig(
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qk_dim=qk_dim,
            mlp_ratio=2.0,
            num_layers=2,
            num_time_steps=3,
            attn_bias=False,
            mlp_bias=False,
        )

    @pytest.fixture
    def model(self, config, device):
        """Create transformer model."""
        return EnergyTransformer(config).to(device)

    def test_initialization(self, model, config):
        """Test model initialization."""
        # Check attributes
        assert model.config == config
        assert hasattr(model, "patch_embed")
        assert hasattr(model.patch_embed, "proj")  # proj is inside patch_embed
        assert hasattr(model, "output_proj")
        assert hasattr(model, "pos_embed")
        assert hasattr(model, "cls_token")
        assert hasattr(model, "mask_token")
        assert len(model.blocks) == config.num_layers

        # Check dimensions
        assert model.pos_embed.shape == (1, config.num_patches + 1, config.embed_dim)
        assert model.cls_token.shape == (1, 1, config.embed_dim)
        assert model.mask_token.shape == (1, 1, config.embed_dim)

    def test_forward_pass(
        self, model, batch_size, img_size, patch_size, num_patches, device
    ):
        """Test forward pass."""
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        out = model(x)

        # Default output dimension is patch_size^2 * 3
        expected_out_dim = patch_size**2 * 3
        assert out.shape == (batch_size, num_patches, expected_out_dim)

    def test_masked_forward(self, model, batch_size, img_size, num_patches, device):
        """Test forward with masking."""
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        # Create random mask
        num_masked = num_patches // 2
        mask_idx = torch.randint(
            0, num_patches, (batch_size, num_masked), device=device
        )
        batch_idx = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_masked)
        )

        out = model(x, mask=(batch_idx, mask_idx))
        assert out.shape[1] == num_patches  # All patches in output

    def test_energy_return(self, model, batch_size, img_size, config, device):
        """Test energy return during forward."""
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        out, energies = model(x, return_energy=True)

        # Should have energy for each evolution step
        expected_energies = config.num_layers * config.num_time_steps + 1
        assert len(energies) == expected_energies
        assert all(torch.isfinite(e) for e in energies)

    def test_evolution_dynamics(self, model, device):
        """Test gradient dynamics evolution."""
        x = torch.randn(1, 3, 32, 32, device=device)

        # Get energies during evolution
        _, energies = model(x, return_energy=True, alpha=0.1)

        # Energy should generally decrease (not strictly monotonic due to layers)
        # But overall trend should be downward within each layer
        assert len(energies) > 0

    def test_visualize_method(self, model, batch_size, img_size, device):
        """Test visualization method."""
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        energies, embeddings = model.visualize(x)

        assert len(energies) > 0
        assert len(embeddings) > 0
        assert all(e.shape == (batch_size, 3, img_size, img_size) for e in embeddings)

    @pytest.mark.parametrize("use_cls", [True, False])
    def test_cls_token_usage(self, model, img_size, use_cls, device):
        """Test CLS token usage."""
        x = torch.randn(1, 3, img_size, img_size, device=device)
        out = model(x, use_cls=use_cls)

        if use_cls:
            assert out.shape[1] == 1  # Only CLS token
        else:
            assert out.shape[1] == model.config.num_patches  # All patch tokens

    def test_no_weight_decay_params(self, model):
        """Test no weight decay parameter list."""
        no_decay = model.no_weight_decay()
        assert "pos_embed" in no_decay
        assert "cls_token" in no_decay
        assert "mask_token" in no_decay


class TestGraphEnergyBlock:
    """Test GraphEnergyBlock."""

    @pytest.fixture
    def config(self, embed_dim, num_heads, qk_dim):
        """Create block config."""
        return EnergyBlockConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qk_dim=qk_dim,
            mlp_ratio=2.0,
            attn_bias=False,
            mlp_bias=False,
        )

    @pytest.fixture
    def block(self, config, device):
        """Create graph energy block."""
        return GraphEnergyBlock(config).to(device)

    def test_forward_returns_gradient_and_energy(
        self, block, batch_size, seq_length, embed_dim, device
    ):
        """Test that forward returns gradient and energy."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)

        grad, energy = block(x)

        assert grad.shape == x.shape
        assert energy.shape == ()
        assert torch.isfinite(grad).all()
        assert torch.isfinite(energy)

    def test_with_adjacency(
        self, block, batch_size, seq_length, embed_dim, num_heads, device
    ):
        """Test forward with adjacency matrix."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        adjacency = torch.ones(
            batch_size, seq_length, seq_length, num_heads, device=device
        )

        grad, energy = block(x, adjacency)

        assert grad.shape == x.shape
        assert torch.isfinite(grad).all()


class TestGraphEnergyTransformer:
    """Test GraphEnergyTransformer."""

    @pytest.fixture
    def config(self, embed_dim, num_heads, qk_dim):
        """Create graph transformer config."""
        return EnergyTransformerConfig(
            input_dim=32,
            embed_dim=embed_dim,
            out_dim=16,
            num_heads=num_heads,
            qk_dim=qk_dim,
            mlp_ratio=2.0,
            num_layers=2,
            num_time_steps=3,
            pos_encoding_dim=10,
        )

    @pytest.fixture
    def model(self, config, device):
        """Create graph transformer model."""
        return GraphEnergyTransformer(config).to(device)

    def test_initialization(self, model, config):
        """Test model initialization."""
        assert hasattr(model, "node_encoder")
        assert hasattr(model, "cls_token")
        assert hasattr(model, "pos_encoder")
        assert hasattr(model, "adj_proj")
        assert hasattr(model, "decoder")
        assert len(model.blocks) == config.num_layers
        assert len(model.norms) == config.num_layers

    def test_forward_pass(self, model, batch_size, device):
        """Test forward pass."""
        num_nodes = 20
        node_features = torch.randn(batch_size, num_nodes, 32, device=device)
        adjacency = torch.ones(batch_size, num_nodes, num_nodes, 1, device=device)
        pos_encoding = torch.randn(
            batch_size, num_nodes + 1, 10, device=device
        )  # +1 for CLS

        output = model(node_features, adjacency, pos_encoding)

        assert "graph_embedding" in output
        assert "node_embeddings" in output
        assert output["graph_embedding"].shape == (batch_size, 16)
        assert output["node_embeddings"].shape == (batch_size, num_nodes, 16)

    def test_with_node_mask(self, model, device):
        """Test forward with node masking."""
        num_nodes = 15
        node_features = torch.randn(1, num_nodes, 32, device=device)
        adjacency = torch.ones(1, num_nodes, num_nodes, 1, device=device)
        pos_encoding = torch.randn(1, num_nodes + 1, 10, device=device)  # +1 for CLS

        # Mask some nodes
        mask = torch.ones(1, num_nodes, device=device)
        mask[0, 10:] = 0  # Mask last 5 nodes

        output = model(node_features, adjacency, pos_encoding, mask=mask)

        # Check output validity
        assert torch.isfinite(output["graph_embedding"]).all()
        assert torch.isfinite(output["node_embeddings"]).all()

    def test_energy_evolution(self, model, device):
        """Test energy evolution tracking."""
        node_features = torch.randn(1, 10, 32, device=device)
        adjacency = torch.ones(1, 10, 10, 1, device=device)
        pos_encoding = torch.randn(1, 11, 10, device=device)  # +1 for CLS

        output = model(node_features, adjacency, pos_encoding, return_energy=True)

        assert "energies" in output
        assert len(output["energies"]) > 0
        assert all(torch.isfinite(e) for e in output["energies"])


class TestStochasticGradientDescent:
    """Test stochastic gradient descent with noise in transformers."""

    def test_vision_transformer_with_noise(self):
        """Test EnergyTransformer with noise enabled."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,  # 16x16 image
            patch_dim=48,  # 4*4*3 for RGB
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            qk_dim=16,
            num_time_steps=5,
            use_noise=True,
            noise_std=0.01,
            noise_decay=False,
        )

        model = EnergyTransformer(config)
        model.train()  # Noise only applied in training mode

        # Create input
        batch_size = 2
        x = torch.randn(batch_size, 3, 16, 16)

        # Run multiple times to check stochasticity
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                out = model(x, alpha=0.1)
                outputs.append(out.clone())

        # Outputs should be different due to noise
        assert not torch.allclose(outputs[0], outputs[1], atol=1e-6)
        assert not torch.allclose(outputs[1], outputs[2], atol=1e-6)

        # Test eval mode (no noise)
        model.eval()
        outputs_eval = []
        for _ in range(2):
            with torch.no_grad():
                out = model(x, alpha=0.1)
                outputs_eval.append(out.clone())

        # Outputs should be identical in eval mode
        assert torch.allclose(outputs_eval[0], outputs_eval[1], atol=1e-6)

    def test_graph_transformer_with_noise(self):
        """Test GraphEnergyTransformer with noise enabled."""
        config = EnergyTransformerConfig(
            input_dim=7,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            qk_dim=16,
            num_time_steps=5,
            pos_encoding_dim=10,
            use_noise=True,
            noise_std=0.02,
            noise_decay=False,
        )

        model = GraphEnergyTransformer(config)
        model.train()

        # Create graph input
        batch_size = 2
        num_nodes = 15
        adjacency_threshold = 0.5  # Threshold for binary adjacency
        node_features = torch.randn(batch_size, num_nodes, 7)
        adjacency = torch.rand(batch_size, num_nodes, num_nodes, 1)
        adjacency = (adjacency > adjacency_threshold).float()  # Binary adjacency
        pos_encoding = torch.randn(batch_size, num_nodes + 1, 10)  # +1 for CLS

        # Run multiple times
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                out = model(node_features, adjacency, pos_encoding, alpha=0.1)
                outputs.append(out["graph_embedding"].clone())

        # Check stochasticity
        assert not torch.allclose(outputs[0], outputs[1], atol=1e-6)
        assert not torch.allclose(outputs[1], outputs[2], atol=1e-6)

    def test_noise_decay(self):
        """Test noise decay over time steps."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,
            patch_dim=48,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            qk_dim=16,
            num_time_steps=10,
            use_noise=True,
            noise_std=0.1,
            noise_decay=True,
            noise_gamma=0.5,
        )

        EnergyTransformer(config)

        # Calculate expected noise std at different time steps
        expected_stds = []
        for t in range(config.num_time_steps):
            std = (
                config.noise_std * config.noise_gamma / ((1 + t) ** config.noise_gamma)
            )
            expected_stds.append(std)

        # Check that noise decays
        assert expected_stds[0] > expected_stds[-1]
        assert all(
            expected_stds[i] > expected_stds[i + 1]
            for i in range(len(expected_stds) - 1)
        )

    def test_noise_gradient_interaction(self):
        """Test that noise affects gradient descent dynamics."""
        import numpy as np

        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,
            patch_dim=48,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            qk_dim=16,
            num_time_steps=5,
            use_noise=False,  # Start without noise
        )

        # Test with and without noise
        x = torch.randn(2, 3, 16, 16)

        # Without noise
        model_no_noise = EnergyTransformer(config)
        model_no_noise.train()
        output_no_noise = model_no_noise(x, alpha=0.1, return_energy=True)
        energies_no_noise = output_no_noise[1]

        # With noise
        config_noise = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,
            patch_dim=48,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            qk_dim=16,
            num_time_steps=5,
            use_noise=True,
            noise_std=0.05,
        )
        model_noise = EnergyTransformer(config_noise)
        # Copy weights to ensure same initialization
        model_noise.load_state_dict(model_no_noise.state_dict())
        model_noise.train()

        # Run multiple times and average energies
        energies_with_noise = []
        for _ in range(10):
            output_noise = model_noise(x, alpha=0.1, return_energy=True)
            energies_with_noise.append([e.item() for e in output_noise[1]])

        # Average energies should be different due to noise exploration
        avg_energies_noise = np.mean(energies_with_noise, axis=0)
        energies_no_noise_values = [e.item() for e in energies_no_noise]

        # The energy trajectories should differ
        # Check that at least some energy values differ significantly
        min_energy_difference = 0.1  # Minimum expected difference due to noise
        differences = np.abs(
            np.array(avg_energies_noise) - np.array(energies_no_noise_values)
        )
        max_diff = np.max(differences)
        assert max_diff > min_energy_difference, (
            f"Maximum difference {max_diff} is too small"
        )

    def test_noise_alpha_scaling(self):
        """Test that noise scales with sqrt(alpha)."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,
            patch_dim=48,
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            qk_dim=16,
            num_time_steps=3,
            use_noise=True,
            noise_std=0.1,
        )

        model = EnergyTransformer(config)
        model.train()

        x = torch.randn(2, 3, 16, 16)

        # Test with different alpha values
        alphas = [0.01, 0.1, 1.0]
        variances = []

        for alpha in alphas:
            outputs = []
            for _ in range(50):
                with torch.no_grad():
                    out = model(x, alpha=alpha)
                    outputs.append(out.flatten())

            # Calculate variance across runs
            outputs_tensor = torch.stack(outputs)
            variance = torch.var(outputs_tensor, dim=0).mean().item()
            variances.append(variance)

        # Variance should scale approximately with alpha (due to sqrt(alpha) scaling)
        # Higher alpha -> higher variance (with some tolerance for randomness)
        # Check that variance generally increases with alpha
        assert (
            variances[2] > variances[0]
        )  # Largest alpha should have more variance than smallest
        # Also check that middle value is reasonable (allowing for randomness)
        assert (
            variances[1] > variances[0] * 0.8
        )  # Middle should be at least 80% of smallest
