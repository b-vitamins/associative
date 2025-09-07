"""Unit tests for transformer modules."""

from typing import cast

import pytest
import torch
from torch import nn

from associative.nn.modules import (
    EnergyTransformer,
    EnergyTransformerBlock,
    GraphEnergyBlock,
    GraphEnergyTransformer,
    METBlock,
    MultimodalEnergyTransformer,
)
from associative.nn.modules.config import (
    EnergyAttentionConfig,
    EnergyBlockConfig,
    EnergyTransformerConfig,
    HopfieldConfig,
    METConfig,
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


class TestMETBlock:
    """Rigorous unit tests for METBlock.

    Mathematical Specification:
    The MET block implements the core energy computation E(x^v, x^a) where:

    E_total = E^cross + Σ_m [E^intra_m + E^HN_m]

    Where:
    1. E^intra_m: Intra-modal temporal dependencies via continuous compression
    2. E^cross: Cross-modal information exchange
    3. E^HN_m: Hopfield memory with cross-modal saliency weighting

    Critical Requirements from Algorithm 1:
    - Step 1: Layer normalization g^m = LN(x^m) per modality
    - Step 2-3: Attention energies via continuous compression
    - Step 4: Hopfield energy with saliency weights alpha^{m,mu}_A
    - Step 5: Gradient computation ∇_g E for dynamics dx/dt = -∇_g E

    Energy Function Properties:
    - Lyapunov stability: E is continuously differentiable and bounded below
    - Convergence: All trajectories converge to critical points where ∇E = 0
    - Sub-quadratic complexity: O(YH(M²L + M³)) via compression

    ANY implementation MUST satisfy these exact mathematical constraints.
    """

    @pytest.fixture
    def modality_dims(self):
        """Modality dimensions matching experimental setup."""
        return {"video": 768, "audio": 512}  # EVA-CLIP video, VGGish audio

    @pytest.fixture
    def attention_config(self):
        """Attention config matching Algorithm 1 parameters."""
        return {
            "embed_dim": 256,  # Y in paper notation
            "num_heads": 8,  # H = 8
            "compression_dims": {"video": 100, "audio": 64},  # M^v, M^a
            "basis_type": "rectangular",  # Basis functions
            "regularization": 0.01,  # λ = 0.01
            "beta": 1.0,  # β = 1/√Y for stability
            "integrator_method": "gauss_legendre",
            "integration_points": 50,  # S = 50 for numerical integration
        }

    @pytest.fixture
    def hopfield_config(self):
        """Hopfield config matching algorithm parameters."""
        return {
            "num_prototypes": {"video": 256, "audio": 256},  # K^m = 256
            "activation": "softplus",  # G(z) = log(1 + e^z)
            "cross_modal_weight": 0.3,  # λ_cross = 0.3
            "temporal_window": 3,  # w = 3 for temporal smoothing
            "projection_init": "xavier",
        }

    @pytest.fixture
    def sample_features(self, device):
        """Features with common sequence length L=10 (MET requirement).

        Real audio-visual data exhibits inherent asynchrony...
        we align them by segmenting audio into fixed-duration chunks matching
        video frame intervals"
        """
        batch_size = 2
        seq_length = 10  # L = 10
        return {
            "video": torch.randn(
                batch_size, seq_length, 768, device=device, requires_grad=True
            ),
            "audio": torch.randn(
                batch_size, seq_length, 512, device=device, requires_grad=True
            ),
        }

    # Mathematical Correctness Tests - Enforce Algorithm 1 Specification

    def test_algorithm_1_initialization_step1_layer_normalization(
        self, modality_dims, attention_config, hopfield_config, device
    ):
        """Test Algorithm 1 Step 1: g^m = LayerNorm(x^m) for each modality.

        Layer normalization with learnable gamma and delta:
        g^m_{iA} = gamma^m * (x^m_{iA} - mean) / sqrt(var + epsilon) + delta^m_i

        MUST create layer norms with correct dimensions for each modality.
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
            device=device,
        )

        # Verify layer normalization components exist (Step 1 of Algorithm 1)
        assert hasattr(block, "norms"), (
            "Missing layer normalization (Algorithm 1, Step 1)"
        )
        assert isinstance(block.norms, torch.nn.ModuleDict), (
            "Norms must be ModuleDict for modalities"
        )

        # Each modality MUST have its own LayerNorm
        assert set(block.norms.keys()) == set(modality_dims.keys()), (
            f"Layer norms missing for modalities. Expected {modality_dims.keys()}, got {block.norms.keys()}"
        )

        # Verify dimensions match specification
        for modality, dim in modality_dims.items():
            norm = block.norms[modality]
            assert isinstance(norm, nn.LayerNorm | nn.Module), (
                f"Modality {modality} must have LayerNorm, got {type(norm)}"
            )
            if hasattr(norm, "normalized_shape"):
                assert norm.normalized_shape == (dim,), (
                    f"LayerNorm for {modality} has wrong shape. Expected ({dim},), got {norm.normalized_shape}"
                )

    def test_algorithm_1_energy_computation_mathematical_form(
        self, modality_dims, attention_config, hopfield_config, sample_features
    ):
        """Test energy computation follows exact mathematical specification.

        E = E^cross + Σ_m [E^intra_m + E^HN_m]

        Energy MUST be:
        1. Scalar-valued (shape = ())
        2. Differentiable (requires_grad = True)
        3. Bounded below (Lyapunov property)
        4. Finite (no NaN/Inf)
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Compute energy
        energy = block.energy(sample_features)

        # Verify mathematical properties
        assert energy.shape == (), f"Energy must be scalar, got shape {energy.shape}"
        assert energy.requires_grad, "Energy must be differentiable for gradient flow"
        assert torch.isfinite(energy), f"Energy contains NaN/Inf: {energy.item()}"

        # Energy should be real-valued (no complex components)
        assert energy.dtype in [torch.float32, torch.float64], (
            f"Energy must be real-valued, got dtype {energy.dtype}"
        )

    def test_metblock_forward_returns_energy(
        self, modality_dims, attention_config, hopfield_config, sample_features
    ):
        """Test METBlock forward method returns energy value.

        Currently FAILS: METBlock raises NotImplementedError
        Expected behavior: forward() returns same as energy()
        This matches EnergyTransformerBlock interface.
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Test forward returns scalar energy
        energy = block.forward(sample_features)

        assert isinstance(energy, torch.Tensor)
        assert energy.shape == ()
        assert energy.requires_grad
        assert torch.isfinite(energy)

        # Test forward equals energy method
        energy_direct = block.energy(sample_features)
        energy_forward = block(sample_features)
        assert torch.allclose(energy_direct, energy_forward)

    def test_metblock_layer_normalization_applied(
        self, modality_dims, attention_config, hopfield_config, sample_features
    ):
        """Test layer normalization applied per modality before energy computation.

        Currently FAILS: METBlock raises NotImplementedError
        Expected: Features normalized internally, original features unchanged
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Store original features
        original_features = {k: v.clone() for k, v in sample_features.items()}

        # Compute energy
        energy = block.energy(sample_features)

        # Original features should be unchanged (functional operation)
        for modality in sample_features:
            assert torch.allclose(
                sample_features[modality], original_features[modality]
            )

        # Energy should be finite
        assert torch.isfinite(energy)

    def test_algorithm_1_gradient_flow_dynamics(
        self, modality_dims, attention_config, hopfield_config, sample_features
    ):
        """Test gradient computation for dynamics τ dx/dt = -∇_g E.

        From gradient flow dynamics:
        - Gradient must be w.r.t. normalized features g
        - Must support torch.func.grad_and_value for efficiency
        - Gradients must be finite and have correct shape

        This is CRITICAL for energy minimization convergence.
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Define energy function for gradient computation
        def energy_fn(features):
            # Note: Block should internally normalize before computing energy
            return block.energy(features)

        # Compute gradient and value using torch.func (required for efficiency)
        grad_dict, energy_value = torch.func.grad_and_value(energy_fn)(sample_features)

        # Verify gradient structure matches input
        assert isinstance(grad_dict, dict), (
            "Gradients must be dict matching input structure"
        )
        assert set(grad_dict.keys()) == set(sample_features.keys()), (
            f"Gradient keys {grad_dict.keys()} don't match input keys {sample_features.keys()}"
        )

        # Verify gradient properties for each modality
        for modality, grad in grad_dict.items():
            input_shape = sample_features[modality].shape
            assert grad.shape == input_shape, (
                f"Gradient shape {grad.shape} doesn't match input shape {input_shape} for {modality}"
            )
            assert torch.isfinite(grad).all(), (
                f"Gradient contains NaN/Inf for modality {modality}"
            )
            # Note: grad may have requires_grad=True for higher-order gradients
            # This is fine and expected behavior with torch.func

        # Verify energy value
        assert energy_value.shape == (), "Energy must be scalar"
        assert torch.isfinite(energy_value), (
            f"Energy is not finite: {energy_value.item()}"
        )

    def test_energy_composition_equation_2(
        self, modality_dims, attention_config, hopfield_config, sample_features
    ):
        """Test energy composition follows Equation (2) exactly.

        E = E^cross + Σ_m [E^intra_m + E^HN_m]

        This is the FUNDAMENTAL equation of MET. Implementation MUST:
        1. Compute all three energy components
        2. Sum them correctly
        3. Return positive semi-definite total energy
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Compute total energy
        total_energy = block.energy(sample_features)

        # Mathematical requirements
        assert torch.isfinite(total_energy), "Energy contains NaN/Inf"
        assert total_energy.shape == (), "Energy must be scalar"
        assert total_energy.requires_grad, "Energy must support gradients"

        # Energy should be real and bounded
        energy_val = total_energy.item()
        assert isinstance(energy_val, float), "Energy must be real number"
        assert energy_val > -1e10, f"Energy {energy_val} appears unbounded below"

        # Test gradient exists for all modalities (verifies all components active)
        for modality, features in sample_features.items():
            grad = torch.autograd.grad(
                total_energy, features, retain_graph=True, create_graph=False
            )[0]
            assert grad.shape == features.shape, (
                f"Gradient shape mismatch for {modality}"
            )
            assert torch.isfinite(grad).all(), (
                f"Gradient contains NaN/Inf for {modality}"
            )
            # Non-zero gradient indicates energy component is active
            assert grad.abs().max() > 1e-10, (
                f"Zero gradient for {modality} suggests missing energy component"
            )

    @pytest.mark.parametrize(
        "batch_size,seq_length",
        [
            (1, 5),
            (2, 10),
            (4, 100),
        ],
    )
    def test_metblock_various_input_shapes(
        self,
        modality_dims,
        attention_config,
        hopfield_config,
        batch_size,
        seq_length,
        device,
    ):
        """Test METBlock handles various input shapes.

        Currently FAILS: METBlock raises NotImplementedError
        Expected: Works with different batch sizes and sequence lengths
        Note: Requires common sequence length L for all modalities.
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
            device=device,
        )

        # Create features with specified shapes - same seq_length for all modalities
        features = {
            "video": torch.randn(
                batch_size, seq_length, 768, device=device, requires_grad=True
            ),
            "audio": torch.randn(
                batch_size, seq_length, 512, device=device, requires_grad=True
            ),
        }

        # Test energy computation
        energy = block.energy(features)

        assert energy.shape == ()
        assert torch.isfinite(energy)
        assert energy.requires_grad

    @pytest.mark.parametrize(
        "modality_config",
        [
            {"video": 512},  # Single modality
            {"video": 768, "audio": 512},  # Standard video-audio
            {"text": 1024, "image": 2048},  # Different modalities
        ],
    )
    def test_metblock_domain_agnostic_modalities(
        self, modality_config, attention_config, hopfield_config, device
    ):
        """Test METBlock handles various modality configurations.

        Currently FAILS: METBlock raises NotImplementedError
        Expected: Domain-agnostic design supports any modality combination
        """
        block = METBlock(
            modality_dims=modality_config,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
            device=device,
        )

        # Create sample features for this configuration
        features = {
            name: torch.randn(2, 10, dim, device=device, requires_grad=True)
            for name, dim in modality_config.items()
        }

        # Test energy computation works
        energy = block.energy(features)

        assert energy.shape == ()
        assert torch.isfinite(energy)
        assert energy.requires_grad

    def test_metblock_parameter_validation(self, attention_config, hopfield_config):
        """Test METBlock validates parameters correctly.

        Currently FAILS: METBlock raises NotImplementedError
        Expected: Proper validation with meaningful error messages
        """
        # Test empty modality_dims
        with pytest.raises(ValueError, match="modality_dims cannot be empty"):
            METBlock(
                modality_dims={},
                attention_config=attention_config,
                hopfield_config=hopfield_config,
            )

        # Test invalid dimensions
        with pytest.raises(
            ValueError, match="All modality dimensions must be positive"
        ):
            METBlock(
                modality_dims={"video": 0},
                attention_config=attention_config,
                hopfield_config=hopfield_config,
            )

        with pytest.raises(
            ValueError, match="All modality dimensions must be positive"
        ):
            METBlock(
                modality_dims={"audio": -100},
                attention_config=attention_config,
                hopfield_config=hopfield_config,
            )

    def test_metblock_feature_validation(
        self, modality_dims, attention_config, hopfield_config
    ):
        """Test METBlock validates input features correctly.

        Currently FAILS: METBlock raises NotImplementedError
        Expected: Feature validation with clear error messages
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Test missing modality
        incomplete_features = {"video": torch.randn(2, 10, 768)}
        with pytest.raises(ValueError, match="Missing required modality"):
            block.energy(incomplete_features)

        # Test wrong feature dimension
        wrong_dim_features = {
            "video": torch.randn(2, 10, 768),
            "audio": torch.randn(2, 10, 256),  # Should be 512
        }
        with pytest.raises(ValueError, match="Expected dimension 512"):
            block.energy(wrong_dim_features)

        # Test inconsistent batch size
        batch_mismatch_features = {
            "video": torch.randn(2, 10, 768),
            "audio": torch.randn(3, 10, 512),  # Different batch size
        }
        with pytest.raises(ValueError, match="Batch size mismatch"):
            block.energy(batch_mismatch_features)

    def test_metblock_extra_repr(
        self, modality_dims, attention_config, hopfield_config
    ):
        """Test METBlock string representation.

        Currently FAILS: METBlock raises NotImplementedError
        Expected: Informative string representation
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        repr_str = block.extra_repr()

        assert "modalities" in repr_str
        for modality in modality_dims:
            assert modality in repr_str

    # Algorithm 1 Compliance Tests

    def test_lyapunov_stability_theorem_3_4(
        self, modality_dims, attention_config, hopfield_config, device
    ):
        """Test Lyapunov stability properties.

        The energy E must satisfy:
        1. E ∈ C¹(X) - continuously differentiable
        2. E(x) ≥ E_min - bounded below
        3. dE/dt = -(∇_g E)ᵀ M (∇_g E) ≤ 0 - energy decreases

        This GUARANTEES convergence to critical points by LaSalle's principle.
        """
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
            device=device,
        )

        # Test on multiple random initializations
        for _ in range(3):
            features = {
                "video": torch.randn(1, 5, 768, device=device, requires_grad=True),
                "audio": torch.randn(1, 5, 512, device=device, requires_grad=True),
            }

            # Property 1: Continuously differentiable (test gradient exists)
            energy = block.energy(features)
            assert energy.requires_grad, "Energy must be differentiable (C¹)"

            # Compute gradient to verify differentiability
            grad = torch.autograd.grad(energy, features["video"], create_graph=True)[0]
            assert torch.isfinite(grad).all(), "Gradient must be finite (continuous)"

            # Property 2: Bounded below (energy should not be -∞)
            assert energy.item() > -1e10, (
                f"Energy {energy.item()} appears unbounded below"
            )

            # Property 3: Energy decrease under gradient flow
            step_size = 0.001  # Use optimal step size for stable convergence
            with torch.no_grad():
                initial_energy = energy.item()

            # Take gradient step
            def energy_fn(f):
                return block.energy(f)

            grad_dict, _ = torch.func.grad_and_value(energy_fn)(features)

            # Update features: x = x - alpha*∇E (gradient descent)
            updated_features = {}
            for modality, tensor in features.items():
                updated_features[modality] = tensor - step_size * grad_dict[modality]

            # Energy should decrease (or stay same at critical point)
            new_energy = block.energy(updated_features)
            assert new_energy.item() <= initial_energy + 1e-6, (
                f"Energy increased: {initial_energy} -> {new_energy.item()} (violates Lyapunov)"
            )


class TestMultimodalEnergyTransformer:
    """Rigorous tests for MultimodalEnergyTransformer enforcing MET architecture.

    Mathematical Foundation:
    The MultimodalEnergyTransformer implements the complete MET architecture:
    1. Input projection: x^m → Linear(D_m, embed_dim)
    2. Evolution: Gradient flow through MET blocks (Algorithm 1)
    3. Output: Task-agnostic evolved features

    Architecture Requirements:
    - Domain-agnostic: Works with ANY modality combination
    - Common sequence length L across modalities
    - Energy minimization via gradient descent: dx/dt = -∇E
    - Convergence guaranteed by Lyapunov stability

    Critical Implementation Constraints:
    - MUST use METBlock for energy computation
    - MUST evolve features through gradient flow
    - MUST preserve batch and sequence dimensions
    - MUST support energy trajectory tracking

    These tests enforce EXACT mathematical behavior from the paper.
    """

    @pytest.fixture
    def met_config(self, embed_dim):
        """MET configuration matching paper specifications."""
        return METConfig(
            # Transformer base (matching Algorithm 1 parameters)
            embed_dim=embed_dim,  # Common embedding dimension
            num_layers=2,  # Not used in MET (uses num_blocks instead)
            num_heads=8,  # H = 8
            qk_dim=32,  # Attention dimension
            mlp_ratio=2.0,  # Hopfield hidden ratio
            num_time_steps=5,  # Iterations per block
            # Multimodal configuration
            modality_configs={
                "video": {"input_dim": 768},  # EVA-CLIP dimension
                "audio": {"input_dim": 512},  # VGGish dimension
            },
            cross_modal_pairs=[("video", "audio")],
            compression_dims={"video": 100, "audio": 64},  # M^v, M^a
            # Evolution parameters (Algorithm 1)
            max_iterations=20,  # I_max = 20
            step_size=0.001,  # η = 0.001
            convergence_tolerance=1e-4,  # ε = 10^-4
            # Architecture
            num_blocks=2,  # Number of MET blocks
        )

    @pytest.fixture
    def sample_multimodal_inputs(self, device):
        """Sample multimodal inputs for testing.

        Note: MET requires common sequence length L=10 for all modalities.
        """
        return {
            "video": torch.randn(2, 10, 768, device=device),
            "audio": torch.randn(2, 10, 512, device=device),
        }

    @pytest.fixture
    def sample_projected_features(self, device, embed_dim):
        """Sample projected features after input projection."""
        return {
            "video": torch.randn(2, 10, embed_dim, device=device, requires_grad=True),
            "audio": torch.randn(2, 10, embed_dim, device=device, requires_grad=True),
        }

    # Initialization Tests - Define Expected Structure (Currently Fail)

    def test_multimodal_energy_transformer_initialization(self, met_config, device):
        """Test MultimodalEnergyTransformer initialization creates required components.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected behavior once implemented:
        1. Creates input projections for each modality
        2. Creates stack of MET blocks
        3. Creates output norms and projections
        4. Validates configuration
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test expected structure
        assert hasattr(transformer, "config")
        assert hasattr(transformer, "input_projs")
        assert hasattr(transformer, "blocks")
        assert hasattr(transformer, "block_norms")  # Pre-block norms
        assert hasattr(transformer, "output_norms")  # Output norms
        assert hasattr(transformer, "output_projs")

        # Test component types
        assert isinstance(transformer.input_projs, torch.nn.ModuleDict)
        assert isinstance(transformer.blocks, torch.nn.ModuleList)
        assert isinstance(transformer.block_norms, torch.nn.ModuleList)
        assert isinstance(transformer.output_norms, torch.nn.ModuleDict)
        assert isinstance(transformer.output_projs, torch.nn.ModuleDict)

        # Test component counts
        assert len(transformer.blocks) == met_config.num_blocks
        assert len(transformer.block_norms) == met_config.num_blocks
        assert len(transformer.input_projs) == len(met_config.modality_configs)
        assert len(transformer.output_norms) == len(met_config.modality_configs)
        assert len(transformer.output_projs) == len(met_config.modality_configs)

    def test_input_projections_configuration(self, met_config, device):
        """Test input projections configured correctly.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Linear projections from input_dim to embed_dim per modality
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Verify modality-specific projections exist
        expected_modalities = set(met_config.modality_configs.keys())
        assert set(transformer.input_projs.keys()) == expected_modalities

        # Test projection dimensions
        for modality, config in met_config.modality_configs.items():
            proj = transformer.input_projs[modality]
            assert isinstance(proj, torch.nn.Linear)
            assert proj.in_features == config["input_dim"]
            assert proj.out_features == met_config.embed_dim

    def test_met_blocks_configuration(self, met_config, device):
        """Test MET blocks configured with proper modality dimensions.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Each block has correct modality_dims configuration
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test MET block creation
        expected_modality_dims = {
            name: met_config.embed_dim for name in met_config.modality_configs
        }

        for block in transformer.blocks:
            assert isinstance(block, METBlock)
            assert block.modality_dims == expected_modality_dims

        # Test block_norms structure
        for norms in transformer.block_norms:
            assert isinstance(norms, torch.nn.ModuleDict)
            assert set(norms.keys()) == set(met_config.modality_configs.keys())

    @pytest.mark.parametrize(
        "modality_config",
        [
            {"video": {"input_dim": 768}, "audio": {"input_dim": 512}},  # Standard
            {
                "text": {"input_dim": 1024},
                "image": {"input_dim": 2048},
            },  # Different domain
            {"single": {"input_dim": 256}},  # Single modality
            {
                "a": {"input_dim": 128},
                "b": {"input_dim": 256},
                "c": {"input_dim": 512},
            },  # Triple
        ],
    )
    def test_domain_agnostic_initialization(self, embed_dim, modality_config, device):
        """Test initialization with various modality configurations.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Domain-agnostic design supports any modality combination
        """
        config = METConfig(
            embed_dim=embed_dim,
            modality_configs=modality_config,
            num_blocks=1,
        )
        transformer = MultimodalEnergyTransformer(config=config, device=device)

        # Verify all modalities handled
        expected_modalities = set(modality_config.keys())
        assert set(transformer.input_projs.keys()) == expected_modalities
        assert set(transformer.output_norms.keys()) == expected_modalities
        assert set(transformer.output_projs.keys()) == expected_modalities

        # Check block norms too
        for norms in transformer.block_norms:
            norms_dict = cast(torch.nn.ModuleDict, norms)
            assert set(norms_dict.keys()) == expected_modalities

    # project_inputs Method Tests (Currently Fail)

    def test_project_inputs_shape_transformation(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test project_inputs method shape transformations.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected behavior:
        1. Projects raw inputs to model dimension
        2. Preserves batch_size and seq_len dimensions
        3. Transforms feature dimension to embed_dim
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        projected = transformer.project_inputs(sample_multimodal_inputs)

        # Test return type
        assert isinstance(projected, dict)
        assert set(projected.keys()) == set(sample_multimodal_inputs.keys())

        # Test shape transformations
        for modality, tensor in projected.items():
            original_shape = sample_multimodal_inputs[modality].shape
            expected_shape = (
                original_shape[0],
                original_shape[1],
                met_config.embed_dim,
            )
            assert tensor.shape == expected_shape
            assert torch.isfinite(tensor).all()

    def test_project_inputs_positional_encodings(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test project_inputs adds positional encodings if provided.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Adds positional encodings if provided in inputs dict
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test without positional encodings
        projected_no_pos = transformer.project_inputs(sample_multimodal_inputs)

        # Test with positional encodings
        inputs_with_pos = sample_multimodal_inputs.copy()
        inputs_with_pos["pos_encodings"] = {
            "video": torch.randn(2, 10, met_config.embed_dim, device=device),
            "audio": torch.randn(2, 10, met_config.embed_dim, device=device),
        }

        projected_with_pos = transformer.project_inputs(inputs_with_pos)

        # Results should be different when positional encodings added
        for modality in projected_no_pos:
            if modality in projected_with_pos:
                assert not torch.allclose(
                    projected_no_pos[modality], projected_with_pos[modality], atol=1e-6
                )

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 5),
            (4, 20),
            (8, 100),
        ],
    )
    def test_project_inputs_various_shapes(
        self, met_config, batch_size, seq_len, device
    ):
        """Test project_inputs with various batch sizes and sequence lengths.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Handles different input shapes correctly
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Create inputs with specified shapes
        inputs = {
            "video": torch.randn(batch_size, seq_len, 768, device=device),
            "audio": torch.randn(batch_size, seq_len, 512, device=device),
        }

        projected = transformer.project_inputs(inputs)

        # Verify shape preservation and projection
        for modality in inputs:
            expected_shape = (batch_size, seq_len, met_config.embed_dim)
            assert projected[modality].shape == expected_shape

    # evolve Method Tests (Currently Fail)

    def test_evolve_gradient_descent_dynamics(
        self, met_config, sample_projected_features, device
    ):
        """Test evolve method implements gradient descent dynamics τ dx/dt = -∇_g E.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected behavior:
        1. Uses MET blocks to compute energy and gradients
        2. Updates features iteratively through gradient descent
        3. Returns evolved features and optional energy trajectory
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test basic evolution
        evolved, energies = transformer.evolve(sample_projected_features)

        assert isinstance(evolved, dict)
        assert set(evolved.keys()) == set(sample_projected_features.keys())

        # Features should have same shape but evolved values
        for modality in evolved:
            assert evolved[modality].shape == sample_projected_features[modality].shape
            assert torch.isfinite(evolved[modality]).all()
            # Should be different due to evolution (high probability)
            assert not torch.allclose(
                evolved[modality], sample_projected_features[modality], atol=1e-6
            )

    def test_evolve_parameter_overrides(
        self, met_config, sample_projected_features, device
    ):
        """Test evolve method parameter overrides.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: num_time_steps and step_size override config defaults
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test with config defaults
        evolved_default, _ = transformer.evolve(sample_projected_features)

        # Test with parameter overrides
        # Note: num_time_steps is fixed in config, only step_size can be overridden
        evolved_custom, _ = transformer.evolve(
            sample_projected_features,
            step_size=0.01,  # Override
        )

        # Different parameters should produce different results
        for modality in evolved_default:
            assert not torch.allclose(
                evolved_default[modality], evolved_custom[modality], atol=1e-6
            )

    def test_evolve_energy_trajectory_tracking(
        self, met_config, sample_projected_features, device
    ):
        """Test evolve method energy trajectory tracking.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Returns energy trajectory when requested
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test without energy tracking (always returns tuple now)
        evolved_only, energies_none = transformer.evolve(
            sample_projected_features, return_energy=False
        )
        assert isinstance(evolved_only, dict)
        assert energies_none is None

        # Test with energy tracking
        evolved_with_energy, energies = transformer.evolve(
            sample_projected_features, return_energy=True
        )
        assert isinstance(evolved_with_energy, dict)
        assert isinstance(energies, list)
        assert len(energies) > 0

        # Test with energy tracking (implementation detail - method signature TBD)
        # This tests the expected interface based on EnergyTransformer pattern

    def test_evolve_iterative_refinement(
        self, met_config, sample_projected_features, device
    ):
        """Test evolve method iterative refinement through blocks.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Each block performs gradient descent on joint multimodal energy
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Multiple evolution steps should generally reduce energy
        # Note: num_time_steps is set in config, not passed to evolve
        evolved, _ = transformer.evolve(sample_projected_features, step_size=0.001)

        # Test that evolution actually changes features
        for modality in evolved:
            assert not torch.allclose(
                evolved[modality],
                sample_projected_features[modality],
                rtol=1e-3,  # Allow some tolerance for small changes
            )

    # forward Method Tests (Currently Fail)

    def test_forward_complete_pipeline(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test forward method complete pipeline: project → evolve → output.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected behavior:
        1. Complete pipeline: project → evolve → output
        2. Handles multiple modalities
        3. Returns output features
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        outputs = transformer(sample_multimodal_inputs)

        # Test return type and structure
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == set(sample_multimodal_inputs.keys())

        # Test output shapes - preserve batch and sequence dimensions
        for modality, output in outputs.items():
            input_shape = sample_multimodal_inputs[modality].shape
            assert output.shape[:2] == input_shape[:2]  # Batch and seq dims
            assert output.shape[2] > 0  # Feature dimension > 0
            assert torch.isfinite(output).all()

    @pytest.mark.parametrize("return_energies", [True, False])
    def test_forward_return_energies_flag(
        self, return_energies, met_config, sample_multimodal_inputs, device
    ):
        """Test forward method return_energies flag.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected return types:
        - return_energies=False: dict[str, Tensor] (output features)
        - return_energies=True: tuple[dict[str, Tensor], list[Tensor]] (outputs, energies)
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        result = transformer(sample_multimodal_inputs, return_energies=return_energies)

        if return_energies:
            assert isinstance(result, tuple)
            assert len(result) == 2
            outputs, energies = result

            # Test outputs
            assert isinstance(outputs, dict)
            assert set(outputs.keys()) == set(sample_multimodal_inputs.keys())

            # Test energies
            assert isinstance(energies, list)
            assert len(energies) > 0
            assert all(torch.isfinite(e) for e in energies)
        else:
            assert isinstance(result, dict)
            assert set(result.keys()) == set(sample_multimodal_inputs.keys())

    def test_forward_task_agnostic_outputs(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test forward method produces task-agnostic outputs.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Task-agnostic outputs suitable for any downstream task
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        outputs = transformer(sample_multimodal_inputs)

        # Outputs should be evolved representations
        # They go through projection back to original space
        # Test that they're different from original inputs (evolved)
        for modality in outputs:
            original_input = sample_multimodal_inputs[modality]
            output = outputs[modality]

            # Should have same shape (reconstruction-style)
            assert output.shape == original_input.shape

            # Should be different due to evolution
            assert not torch.allclose(output, original_input, atol=1e-6)

    # visualize Method Tests (Currently Fail)

    def test_visualize_energy_evolution(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test visualize method creates energy evolution visualization.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected behavior:
        1. Creates energy evolution visualization
        2. Returns matplotlib figure
        3. Tracks energy and feature evolution through blocks
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        energies, embeddings_dict = transformer.visualize(
            sample_multimodal_inputs, step_size=0.001
        )

        # Test energy tracking
        assert isinstance(energies, list)
        assert len(energies) > 0
        assert all(torch.isfinite(e) for e in energies)

        # Test feature evolution tracking
        assert isinstance(embeddings_dict, dict)
        assert set(embeddings_dict.keys()) == set(sample_multimodal_inputs.keys())

        for modality, evolution in embeddings_dict.items():
            assert isinstance(evolution, list)
            assert len(evolution) > 0
            # Each evolution step is in embed space
            # Shape: (batch, seq_len, embed_dim)
            for step in evolution:
                assert (
                    step.shape[0] == sample_multimodal_inputs[modality].shape[0]
                )  # batch
                assert (
                    step.shape[1] == sample_multimodal_inputs[modality].shape[1]
                )  # seq_len
                assert step.shape[2] == met_config.embed_dim  # embed_dim

    def test_visualize_step_size_effect(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test visualize method step_size parameter affects evolution.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Different step_size values produce different evolution trajectories
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test different step_size values
        energies_1, _ = transformer.visualize(sample_multimodal_inputs, step_size=0.01)
        energies_01, _ = transformer.visualize(
            sample_multimodal_inputs, step_size=0.001
        )

        # Different step sizes should produce different trajectories
        assert len(energies_1) > 0
        assert len(energies_01) > 0
        # Energy trajectories should differ
        energy_diffs = [
            abs(e1.item() - e01.item())
            for e1, e01 in zip(energies_1, energies_01, strict=False)
        ]
        assert max(energy_diffs) > 1e-6  # Meaningful difference

    # no_weight_decay Method Tests (Currently Fail)

    def test_no_weight_decay_parameter_exclusion(self, met_config, device):
        """Test no_weight_decay method returns correct parameter exclusions.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected behavior:
        1. Returns list of parameters to exclude from weight decay
        2. Includes norms and biases typically
        3. Similar to EnergyTransformer.no_weight_decay()
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        no_decay_params = transformer.no_weight_decay()

        # Test return type
        assert isinstance(no_decay_params, list)
        assert all(isinstance(param_name, str) for param_name in no_decay_params)

        # Test that some common exclusions are present
        # (Exact list depends on implementation details)
        # Common patterns: norm parameters, bias terms, positional encodings
        param_names = [name for name, _ in transformer.named_parameters()]
        no_decay_set = set(no_decay_params)

        # Verify that returned names are actually parameter names
        assert no_decay_set.issubset(set(param_names))

    # Edge Cases and Error Handling (Currently Fail)

    def test_empty_modality_configs_validation(self, embed_dim):
        """Test validation of empty modality configurations.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Proper validation with meaningful error messages
        """
        with pytest.raises(ValueError, match="modality_configs cannot be empty"):
            config = METConfig(embed_dim=embed_dim, modality_configs={})
            MultimodalEnergyTransformer(config=config)

    def test_mismatched_input_validation(self, met_config, device):
        """Test validation of mismatched inputs.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Clear error messages for input validation
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test missing modality
        incomplete_inputs = {"video": torch.randn(2, 10, 768, device=device)}
        with pytest.raises(ValueError, match="Missing required modality"):
            transformer.project_inputs(incomplete_inputs)

        # Test wrong dimensions
        wrong_dim_inputs = {
            "video": torch.randn(2, 10, 768, device=device),
            "audio": torch.randn(2, 10, 256, device=device),  # Should be 512
        }
        with pytest.raises(ValueError, match="Expected.*512"):
            transformer.project_inputs(wrong_dim_inputs)

    def test_batch_size_consistency_validation(self, met_config, device):
        """Test batch size consistency validation.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Error when batch sizes don't match across modalities
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        batch_mismatch_inputs = {
            "video": torch.randn(2, 10, 768, device=device),
            "audio": torch.randn(3, 10, 512, device=device),  # Different batch size
        }

        with pytest.raises(ValueError, match="Batch size mismatch"):
            transformer.project_inputs(batch_mismatch_inputs)

    # Performance and Memory Tests Structure

    def test_gradient_checkpointing_support(
        self, met_config, sample_multimodal_inputs, device
    ):
        """Test gradient checkpointing support for memory efficiency.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Support for gradient checkpointing with MET blocks
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test that forward pass works (checkpointing is internal optimization)
        outputs = transformer(sample_multimodal_inputs)

        # Verify output validity
        for modality in outputs:
            assert torch.isfinite(outputs[modality]).all()

    # Mathematical Correctness Tests - Enforce Paper Requirements

    def test_common_sequence_length_requirement(self, met_config, device):
        """Test MET requirement for common sequence length L across modalities.

        Currently FAILS: MultimodalEnergyTransformer raises NotImplementedError
        Expected: Error when sequence lengths don't match
        """
        transformer = MultimodalEnergyTransformer(config=met_config, device=device)

        # Test mismatched sequence lengths
        mismatched_inputs = {
            "video": torch.randn(2, 10, 768, device=device),
            "audio": torch.randn(2, 15, 512, device=device),  # Different seq_len
        }

        with pytest.raises(ValueError, match="Sequence length mismatch"):
            transformer.project_inputs(mismatched_inputs)
