"""Comprehensive unit tests for Hopfield module and variants."""

import math

import pytest
import torch
from torch.nn import functional

from associative.nn.modules import Hopfield
from associative.nn.modules.config import HopfieldConfig
from associative.nn.modules.hopfield import CrossModalHopfield
from tests.conftest import TOLERANCE_ZERO_ENERGY


class TestHopfield:
    """Test Hopfield energy module."""

    @pytest.fixture
    def config(self):
        """Create Hopfield config."""
        return HopfieldConfig(hidden_dim_ratio=4.0, bias=False)

    @pytest.fixture
    def hopfield(self, embed_dim, config, device):
        """Create Hopfield module."""
        return Hopfield(embed_dim, config=config).to(device)

    def test_initialization(self, hopfield, embed_dim, config):
        """Test proper initialization."""
        expected_hidden_dim = int(embed_dim * config.hidden_dim_ratio)

        # Check layer exists
        assert hasattr(hopfield, "proj")
        assert isinstance(hopfield.proj, torch.nn.Linear)

        # Check dimensions
        assert hopfield.proj.in_features == embed_dim
        assert hopfield.proj.out_features == expected_hidden_dim

        # Check weight initialization exists (PyTorch default)
        weight_std = hopfield.proj.weight.std().item()
        assert weight_std > 0  # Just check it's initialized

    def test_energy_computation(
        self, hopfield, batch_size, seq_length, embed_dim, device
    ):
        """Test energy computation."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        energy = hopfield(x)

        # Check output
        assert energy.shape == ()  # Scalar
        assert energy.dtype == x.dtype
        assert energy.item() < 0  # Hopfield energy is negative

    def test_energy_formula(self, hopfield, device):
        """Test the Hopfield energy formula correctness."""
        # Create simple input
        x = torch.ones(1, 2, 64, device=device)

        # Compute energy manually (including ReLU activation)
        with torch.no_grad():
            proj_out = hopfield.proj(x)
            relu_out = torch.relu(proj_out)
            expected_energy = -0.5 * (relu_out**2).sum()

        # Compute using module
        energy = hopfield(x)

        assert torch.allclose(energy, expected_energy, rtol=1e-5)

    def test_gradient_flow(self, hopfield, batch_size, seq_length, embed_dim, device):
        """Test gradient flow through Hopfield."""
        x = torch.randn(
            batch_size, seq_length, embed_dim, device=device, requires_grad=True
        )

        energy = hopfield(x)
        energy.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() > 0

    @pytest.mark.parametrize("hidden_ratio", [1.0, 2.0, 4.0, 8.0])
    def test_different_ratios(self, embed_dim, hidden_ratio, device):
        """Test different hidden dimension ratios."""
        config = HopfieldConfig(hidden_dim_ratio=hidden_ratio, bias=False)
        hopfield = Hopfield(embed_dim, config=config).to(device)

        expected_hidden = int(embed_dim * hidden_ratio)
        assert hopfield.proj.out_features == expected_hidden

        # Test forward pass
        x = torch.randn(1, 5, embed_dim, device=device)
        energy = hopfield(x)
        assert torch.isfinite(energy)

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_option(self, embed_dim, bias, device):
        """Test Hopfield with and without bias."""
        config = HopfieldConfig(hidden_dim_ratio=2.0, bias=bias)
        hopfield = Hopfield(embed_dim, config=config).to(device)

        if bias:
            assert hopfield.proj.bias is not None
            assert hopfield.proj.bias.shape == (hopfield.proj.out_features,)
            # Bias should be initialized (PyTorch default is uniform)
            assert torch.isfinite(hopfield.proj.bias).all()
        else:
            assert hopfield.proj.bias is None

    def test_energy_minimization_property(self, hopfield, device):
        """Test that energy decreases with gradient descent."""
        x = torch.randn(1, 5, 64, device=device, requires_grad=True)

        # Initial energy
        energy_init = hopfield(x).item()

        # Take a gradient step
        energy = hopfield(x)
        x_grad = torch.autograd.grad(energy, x)[0]
        x_updated = x - 0.1 * x_grad

        # New energy should be lower
        energy_new = hopfield(x_updated).item()
        assert energy_new < energy_init

    def test_numerical_stability(self, hopfield, device):
        """Test numerical stability with extreme inputs."""
        # Large inputs
        x_large = torch.randn(1, 5, 64, device=device) * 100
        energy_large = hopfield(x_large)
        assert torch.isfinite(energy_large)

        # Small inputs
        x_small = torch.randn(1, 5, 64, device=device) * 0.001
        energy_small = hopfield(x_small)
        assert torch.isfinite(energy_small)

        # Zero input should give zero energy
        x_zero = torch.zeros(1, 5, 64, device=device)
        energy_zero = hopfield(x_zero)
        assert abs(energy_zero.item()) < TOLERANCE_ZERO_ENERGY


class TestHopfieldVariants:
    """Test different Hopfield activation variants."""

    @pytest.fixture
    def input_tensor(self):
        """Create test input tensor."""
        batch_size, seq_len, embed_dim = 2, 10, 64
        return torch.randn(batch_size, seq_len, embed_dim)

    def test_relu_variant(self, input_tensor):
        """Test ReLU Hopfield variant."""
        config = HopfieldConfig(activation_type="relu", hidden_dim_ratio=2.0)
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Forward pass
        energy = layer(input_tensor)

        # Check output is scalar
        assert energy.dim() == 0
        assert energy.dtype == torch.float32

        # Check energy is negative (ReLU squared is negative)
        assert energy <= 0

    def test_gelu_variant(self, input_tensor):
        """Test GELU Hopfield variant."""
        config = HopfieldConfig(activation_type="gelu", hidden_dim_ratio=2.0)
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Forward pass
        energy = layer(input_tensor)

        # Check output
        assert energy.dim() == 0
        assert energy <= 0

    def test_tanh_variant(self, input_tensor):
        """Test Tanh Hopfield variant."""
        config = HopfieldConfig(activation_type="tanh", hidden_dim_ratio=2.0)
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Forward pass
        energy = layer(input_tensor)

        # Check output
        assert energy.dim() == 0
        assert energy <= 0

    def test_softmax_variant(self, input_tensor):
        """Test Softmax/LogSumExp Hopfield variant."""
        config = HopfieldConfig(activation_type="softmax", hidden_dim_ratio=2.0)
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Forward pass
        energy = layer(input_tensor)

        # Check output
        assert energy.dim() == 0
        assert energy <= 0

        # Check beta is set correctly
        assert hasattr(layer, "beta")
        assert layer.beta == math.sqrt(layer.hidden_features)

    def test_manhattan_variant(self, input_tensor):
        """Test Manhattan distance Hopfield variant."""
        config = HopfieldConfig(
            activation_type="manhattan", hidden_dim_ratio=2.0, bias=True
        )
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Check correct attributes
        assert hasattr(layer, "weight")
        assert hasattr(layer, "bias")
        assert not hasattr(layer, "proj")

        # Forward pass
        energy = layer(input_tensor)

        # Check output
        assert energy.dim() == 0
        assert energy <= 0

        # Check weight shape
        assert layer.weight.shape == (layer.hidden_features, layer.in_features)
        assert layer.bias.shape == (layer.hidden_features,)

    def test_manhattan_no_bias(self, input_tensor):
        """Test Manhattan variant without bias."""
        config = HopfieldConfig(
            activation_type="manhattan", hidden_dim_ratio=2.0, bias=False
        )
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Check no bias
        assert layer.bias is None

        # Forward pass should still work
        energy = layer(input_tensor)
        assert energy.dim() == 0

    def test_custom_activation(self, input_tensor):
        """Test custom activation function."""

        def custom_activation(x):
            return -torch.sum(x**3)

        config = HopfieldConfig(
            activation_type=None,  # Use custom
            activation=custom_activation,
            hidden_dim_ratio=2.0,
        )
        layer = Hopfield(input_tensor.shape[-1], config=config)

        # Forward pass
        energy = layer(input_tensor)

        # Check output
        assert energy.dim() == 0

    def test_gradient_flow_all_variants(self, input_tensor):
        """Test gradient flow through all variants."""
        variants = ["relu", "gelu", "tanh", "softmax", "manhattan"]

        for variant in variants:
            config = HopfieldConfig(activation_type=variant, hidden_dim_ratio=2.0)
            layer = Hopfield(input_tensor.shape[-1], config=config)

            # Enable gradients
            input_tensor.requires_grad_(True)

            # Forward pass
            energy = layer(input_tensor)

            # Backward pass
            energy.backward()

            # Check gradients exist
            assert input_tensor.grad is not None
            assert not torch.isnan(input_tensor.grad).any()
            assert not torch.isinf(input_tensor.grad).any()

            # Reset gradients
            input_tensor.grad = None

    def test_invalid_activation_type(self):
        """Test error on invalid activation type."""
        with pytest.raises(ValueError, match="activation_type must be one of"):
            HopfieldConfig(activation_type="invalid")

    def test_energy_values_comparison(self, input_tensor):
        """Compare energy values across different variants."""
        variants = {
            "relu": HopfieldConfig(activation_type="relu"),
            "gelu": HopfieldConfig(activation_type="gelu"),
            "tanh": HopfieldConfig(activation_type="tanh"),
            "softmax": HopfieldConfig(activation_type="softmax"),
            "manhattan": HopfieldConfig(activation_type="manhattan"),
        }

        energies = {}
        for name, config in variants.items():
            layer = Hopfield(input_tensor.shape[-1], config=config)
            layer.eval()  # Ensure same behavior
            with torch.no_grad():
                energies[name] = layer(input_tensor).item()

        # All energies should be negative
        for name, energy in energies.items():
            assert energy <= 0, f"{name} energy should be negative"

        # Energies should be different (variants produce different values)
        energy_values = list(energies.values())
        assert len(set(energy_values)) > 1, (
            "Different variants should produce different energies"
        )


class TestCrossModalHopfield:
    """Comprehensive unit tests for CrossModalHopfield class.

    Tests cover exact mathematical specifications:
    - Memory energy: E^HN = -Sum_m Sum_A Sum_mu G(alpha^{m,mu}_A * xi^{m,mu}^T g^m_A)
    - Saliency weights: alpha^{m,mu}_A = lambda_cross [xi^{m,mu}^T W^{m'->m} g_bar^{m'}_A]^2 + (1-lambda_cross) [xi^{m,mu}^T g^m_A]^2
    - Temporal smoothing: g_bar^{m'}_A = 1/|N(A)| Sum_{B in N(A)} g^{m'}_B
    - Cross-modal projections: W^{a->v} in R^{D_v x D_a}, W^{v->a} in R^{D_a x D_v}
    - Memory patterns: xi^m in R^{K^m x D_m} initialized via Xavier
    - Activation functions: G(z) = log(1 + e^z) (softplus) or 1/2[z]_+² (ReLU)
    """

    @pytest.fixture
    def modality_dims(self):
        """Standard modality dimensions."""
        return {"video": 768, "audio": 512}

    @pytest.fixture
    def device(self):
        """Test device."""
        return torch.device("cpu")

    @pytest.fixture
    def cross_modal_hopfield(self, modality_dims, device):
        """Create CrossModalHopfield instance for testing."""
        return CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=256,
            cross_weight=0.3,
            temporal_window=3,
            activation_type="softplus",
            device=device,
        )

    # Tests for __init__ method
    def test_init_creates_prototypes_with_correct_shapes(self, device):
        """Test prototype matrices xi^m in R^(K^m x D_m) are created correctly."""
        modality_dims = {"video": 768, "audio": 512, "text": 256}
        num_prototypes = {"video": 128, "audio": 64, "text": 32}

        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=num_prototypes,
            activation_type="softplus",
            device=device,
        )

        # Check prototype matrices exist and have correct shapes
        assert hasattr(hopfield, "prototypes")
        assert "video" in hopfield.prototypes
        assert "audio" in hopfield.prototypes
        assert "text" in hopfield.prototypes

        # Verify shapes: xi^m in R^(K^m x D_m)
        assert hopfield.prototypes["video"].shape == (128, 768)
        assert hopfield.prototypes["audio"].shape == (64, 512)
        assert hopfield.prototypes["text"].shape == (32, 256)

    def test_init_prototypes_xavier_initialization(self, modality_dims, device):
        """Test prototypes are initialized with Xavier method for stable gradients."""
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=256,
            activation_type="softplus",
            device=device,
        )

        for modality, dim in modality_dims.items():
            prototypes = hopfield.prototypes[modality]

            # Check Xavier initialization bounds: ±√(6/(fan_in + fan_out))
            # For prototypes: fan_in = dim, fan_out = 256 (num_prototypes)
            bound = math.sqrt(6.0 / (dim + 256))

            assert prototypes.abs().max() <= bound * 1.1  # Allow small tolerance
            assert prototypes.std() > 0  # Not zero-initialized
            assert torch.isfinite(prototypes).all()

    def test_init_cross_projections_correct_shapes(self, device):
        """Test cross-modal projections W^{m->m'} in R^{D_m' x D_m}."""
        modality_dims = {"video": 768, "audio": 512}

        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=256,
            activation_type="softplus",
            device=device,
        )

        # Check projection matrices exist
        assert hasattr(hopfield, "cross_projs")

        # According to paper: W^{a->v} in R^{D_v x D_a}, W^{v->a} in R^{D_a x D_v}
        assert "audio_to_video" in hopfield.cross_projs
        assert "video_to_audio" in hopfield.cross_projs

        # Check shapes
        assert hopfield.cross_projs["audio_to_video"].shape == (768, 512)  # D_v x D_a
        assert hopfield.cross_projs["video_to_audio"].shape == (512, 768)  # D_a x D_v

    # Tests for compute_saliency_weights method
    def test_saliency_weights_mathematical_formula(self, device):
        """Test exact formula: alpha^{m,mu}_A = lambda_cross [xi^{m,mu}^T W^{m'->m} g_bar^{m'}_A]^2 + (1-lambda_cross) [xi^{m,mu}^T g^m_A]^2."""
        modality_dims = {"video": 4, "audio": 3}  # Small dims for manual verification
        lambda_cross = 0.3

        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=2,
            cross_weight=lambda_cross,
            temporal_window=3,
            device=device,
        )

        # Create test data
        _batch, _seq_len = 1, 1
        g_video = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device)  # (1, 1, 4)
        g_audio_smoothed = torch.tensor([[[0.5, 1.5, 2.5]]], device=device)  # (1, 1, 3)

        # Set known prototypes for video
        hopfield.prototypes["video"].data = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # ξ^{v,0}
                [0.0, 1.0, 0.0, 0.0],  # ξ^{v,1}
            ],
            device=device,
        )

        # Set known cross-projection W^{a→v}
        hopfield.cross_projs["audio_to_video"].data = torch.eye(
            4, 3, device=device
        )  # Simple projection

        saliency = hopfield.compute_saliency_weights(g_video, g_audio_smoothed, "video")

        # Verify shape: (batch, seq_len, num_prototypes) = (1, 1, 2)
        assert saliency.shape == (1, 1, 2)

        # Manually compute expected values for EXACT mathematical verification
        # This test MUST fail if implementation deviates from paper formula

        # For prototype 0: xi^{v,0} = [1, 0, 0, 0]
        # Cross term: xi^{v,0}^T W^{a->v} g_bar^a = [1,0,0,0] @ [[1,0,0],[0,1,0],[0,0,1],[0,0,0]] @ [0.5,1.5,2.5]^T = 0.5
        # Intra term: xi^{v,0}^T g^v = [1,0,0,0] @ [1,2,3,4]^T = 1.0
        # alpha^{v,0} = 0.3 * (0.5)^2 + 0.7 * (1.0)^2 = 0.3 * 0.25 + 0.7 * 1.0 = 0.075 + 0.7 = 0.775

        expected_alpha0 = lambda_cross * (0.5**2) + (1 - lambda_cross) * (1.0**2)
        assert torch.allclose(
            saliency[0, 0, 0], torch.tensor(expected_alpha0), atol=1e-6
        ), (
            f"Saliency formula MUST match paper exactly: got {saliency[0, 0, 0]}, expected {expected_alpha0}"
        )

        # All saliency weights should be non-negative (squared terms)
        assert (saliency >= 0).all()

    def test_saliency_weights_cross_modal_influence(self, device):
        """Test cross-modal influence changes with lambda_cross parameter."""
        modality_dims = {"video": 64, "audio": 32}
        batch, seq_len = 2, 5

        g_video = torch.randn(batch, seq_len, 64, device=device)
        g_audio = torch.randn(batch, seq_len, 32, device=device)

        # Test different cross_weight values
        for lambda_cross in [0.0, 0.5, 1.0]:
            hopfield = CrossModalHopfield(
                modality_dims=modality_dims,
                num_prototypes=32,
                cross_weight=lambda_cross,
                device=device,
            )

            saliency = hopfield.compute_saliency_weights(g_video, g_audio, "video")

            # Shape: (batch, seq_len, num_prototypes)
            assert saliency.shape == (2, 5, 32)
            assert (saliency >= 0).all()  # Non-negative due to squaring
            assert torch.isfinite(saliency).all()

    def test_saliency_weights_with_temporal_smoothing_integration(self, device):
        """Test saliency weights use temporally smoothed complementary features."""
        modality_dims = {"video": 16, "audio": 8}

        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=4,
            cross_weight=0.4,
            temporal_window=3,
            device=device,
        )

        batch, seq_len = 1, 7
        g_video = torch.randn(batch, seq_len, 16, device=device)
        g_audio = torch.randn(batch, seq_len, 8, device=device)

        # Test that saliency computation handles temporal smoothing internally
        saliency = hopfield.compute_saliency_weights(g_video, g_audio, "video")

        assert saliency.shape == (1, 7, 4)
        assert (saliency >= 0).all()
        assert torch.isfinite(saliency).all()

    # Tests for temporal_smooth method
    def test_temporal_smooth_mathematical_formula(self, device):
        """Test exact formula: g_bar^{m'}_A = 1/|N(A)| Sum_{B in N(A)} g^{m'}_B with N(A) = {B : |B - A| <= floor(w/2)}."""
        modality_dims = {"video": 64, "audio": 32}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=16,
            temporal_window=3,  # w=3, so floor(w/2) = 1
            device=device,
        )

        # Create specific test sequence for manual verification
        _batch, _seq_len, _dim = 1, 5, 2
        features = torch.tensor(
            [[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]],
            device=device,
        )

        smoothed = hopfield.temporal_smooth(features)

        # Verify shape is preserved
        assert smoothed.shape == (1, 5, 2)

        # Manual verification for window=3:
        # Position 0: N(0) = {0, 1} -> average of positions 0,1
        # Position 1: N(1) = {0, 1, 2} -> average of positions 0,1,2
        # Position 2: N(2) = {1, 2, 3} -> average of positions 1,2,3
        # Position 3: N(3) = {2, 3, 4} -> average of positions 2,3,4
        # Position 4: N(4) = {3, 4} -> average of positions 3,4

        expected = torch.tensor(
            [[[1.5, 15.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [4.5, 45.0]]],
            device=device,
        )
        assert torch.allclose(smoothed, expected, atol=1e-6)

    def test_temporal_smooth_conv1d_performance_requirement(self, device):
        """Test requirement to use F.conv1d with uniform kernel for efficiency."""
        modality_dims = {"audio": 128}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            temporal_window=5,
            device=device,
        )

        # Test with realistic sizes
        batch, seq_len, dim = 4, 1024, 128
        features = torch.randn(batch, seq_len, dim, device=device)

        # Should handle large tensors efficiently
        smoothed = hopfield.temporal_smooth(features)

        assert smoothed.shape == features.shape
        assert torch.isfinite(smoothed).all()

        # Test that it actually smooths (reduced variance)
        original_std = features.std(dim=1, keepdim=True)
        smoothed_std = smoothed.std(dim=1, keepdim=True)
        assert (smoothed_std <= original_std + 1e-6).all()  # Smoothing reduces variance

    def test_temporal_smooth_edge_padding_reflect_mode(self, device):
        """Test edge case handling with 'reflect' padding."""
        modality_dims = {"video": 4}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            temporal_window=7,  # Large window relative to sequence
            device=device,
        )

        # Very short sequence to test edge handling
        _batch, _seq_len, _dim = 1, 3, 4
        features = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]],
            device=device,
        )

        smoothed = hopfield.temporal_smooth(features)

        # Should handle gracefully without errors
        assert smoothed.shape == features.shape
        assert torch.isfinite(smoothed).all()

        # With reflect padding, edges should be smoothed versions of nearby values
        assert not torch.allclose(
            smoothed, features
        )  # Should be different due to smoothing

    def test_temporal_smooth_different_window_sizes(self, device):
        """Test effect of different window sizes w (typically 3-5 for speech)."""
        modality_dims = {"audio": 32}

        batch, seq_len, dim = 2, 20, 32
        features = torch.randn(batch, seq_len, dim, device=device)

        smoothed_results = {}
        for window in [3, 5, 7]:
            hopfield = CrossModalHopfield(
                modality_dims=modality_dims,
                temporal_window=window,
                device=device,
            )

            smoothed = hopfield.temporal_smooth(features)
            smoothed_results[window] = smoothed

            assert smoothed.shape == features.shape

            # Larger windows should produce smoother results (lower std deviation)
            if window > 3:
                prev_std = smoothed_results[window - 2].std()
                curr_std = smoothed.std()
                # Allow some tolerance since this is stochastic
                assert curr_std <= prev_std * 1.1

    # Tests for compute_memory_energy method
    def test_memory_energy_mathematical_formula(self, device):
        """Test exact formula: E^HN_m = -Sum_A Sum_mu G(alpha^{m,mu}_A * xi^{m,mu}^T g^m_A).

        This test is the GROUND TRUTH for the memory energy computation.
        ANY deviation from the paper formula MUST cause this test to fail.
        """
        modality_dims = {"video": 3, "audio": 2}  # Small for manual verification
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=2,
            cross_weight=0.5,
            activation_type="softplus",  # G(z) = log(1 + e^z)
            device=device,
        )

        # Set known values for verification
        _batch, _seq_len = 1, 2
        g_video = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device
        )  # (1, 2, 3)
        saliency = torch.tensor([[[2.0, 3.0], [1.0, 4.0]]], device=device)  # (1, 2, 2)

        # Set known prototypes
        hopfield.prototypes["video"].data = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # ξ^{v,0}
                [0.0, 1.0, 0.0],  # ξ^{v,1}
            ],
            device=device,
        )

        energy = hopfield.compute_memory_energy(g_video, saliency, "video")

        # EXACT manual calculation per paper formula:
        # E^HN_m = -Sum_A Sum_mu G(alpha^{m,mu}_A * xi^{m,mu}^T g^m_A)
        #
        # Position A=0: g^v = [1,0,0]
        #   mu=0: alpha=2.0, xi^T g = [1,0,0]*[1,0,0] = 1.0, G(2.0*1.0) = G(2.0) = log(1+e^2)
        #   mu=1: alpha=3.0, xi^T g = [0,1,0]*[1,0,0] = 0.0, G(3.0*0.0) = G(0.0) = log(1+e^0) = log(2)
        # Position A=1: g^v = [0,1,0]
        #   mu=0: alpha=1.0, xi^T g = [1,0,0]*[0,1,0] = 0.0, G(1.0*0.0) = G(0.0) = log(2)
        #   mu=1: alpha=4.0, xi^T g = [0,1,0]*[0,1,0] = 1.0, G(4.0*1.0) = G(4.0) = log(1+e^4)
        # E^HN = -(G(2.0) + log(2) + log(2) + G(4.0))

        expected_energy = -(
            functional.softplus(torch.tensor(2.0))
            + math.log(2)
            + math.log(2)
            + functional.softplus(torch.tensor(4.0))
        )
        assert torch.allclose(energy, expected_energy, atol=1e-6), (
            f"Memory energy MUST match paper formula exactly: got {energy}, expected {expected_energy}"
        )

        # Energy should be scalar and negative
        assert energy.dim() == 0
        assert energy.item() < 0

    def test_memory_energy_different_activation_functions(self, device):
        """Test with different activation functions G(z)."""
        modality_dims = {"audio": 64}
        batch, seq_len = 2, 5

        g_audio = torch.randn(batch, seq_len, 64, device=device)
        saliency = torch.rand(batch, seq_len, 32, device=device) + 0.1  # Avoid zeros

        for activation in ["softplus", "relu"]:
            hopfield = CrossModalHopfield(
                modality_dims=modality_dims,
                num_prototypes=32,
                activation_type=activation,
                device=device,
            )

            energy = hopfield.compute_memory_energy(g_audio, saliency, "audio")

            # Should be scalar and negative (sum of negative activations)
            assert energy.dim() == 0
            assert energy.item() <= 0  # Allow zero for ReLU if all terms are negative
            assert torch.isfinite(energy)

    def test_memory_energy_saliency_weighting_effect(self, device):
        """Test how saliency weights modulate energy magnitude."""
        modality_dims = {"video": 32}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=16,
            activation_type="softplus",
            device=device,
        )

        batch, seq_len = 1, 3
        g_video = torch.randn(batch, seq_len, 32, device=device)

        # Test with different saliency patterns
        uniform_saliency = torch.ones(batch, seq_len, 16, device=device)
        high_saliency = torch.ones(batch, seq_len, 16, device=device) * 5.0
        low_saliency = torch.ones(batch, seq_len, 16, device=device) * 0.1

        energy_uniform = hopfield.compute_memory_energy(
            g_video, uniform_saliency, "video"
        )
        energy_high = hopfield.compute_memory_energy(g_video, high_saliency, "video")
        energy_low = hopfield.compute_memory_energy(g_video, low_saliency, "video")

        # Higher saliency should lead to more negative energy (stronger attraction)
        assert energy_high.item() <= energy_uniform.item()
        assert energy_uniform.item() <= energy_low.item()

    def test_memory_energy_with_batch_and_sequence_dimensions(self, device):
        """Test energy computation handles batching correctly."""
        modality_dims = {"text": 128}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=64,
            activation_type="relu",  # G(z) = 1/2[z]_+²
            device=device,
        )

        # Test various batch and sequence sizes
        test_shapes = [(1, 1), (4, 8), (2, 16), (8, 32)]

        for batch, seq_len in test_shapes:
            g_text = torch.randn(batch, seq_len, 128, device=device)
            saliency = torch.rand(batch, seq_len, 64, device=device)

            energy = hopfield.compute_memory_energy(g_text, saliency, "text")

            # Should always return scalar regardless of input dimensions
            assert energy.dim() == 0
            assert torch.isfinite(energy)

    # Tests for forward method
    def test_forward_total_energy_computation(self, device):
        """Test forward computes total energy E^HN = Sum_m E^HN_m across all modalities."""
        modality_dims = {"video": 32, "audio": 16}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=8,
            cross_weight=0.4,
            activation_type="softplus",
            device=device,
        )

        batch, seq_len = 2, 5
        features = {
            "video": torch.randn(batch, seq_len, 32, device=device),
            "audio": torch.randn(batch, seq_len, 16, device=device),
        }

        # Test default behavior (return_components=False)
        total_energy = hopfield(features)

        # Should return scalar
        assert total_energy.dim() == 0
        assert torch.isfinite(total_energy)
        assert total_energy.item() <= 0  # Negative energy

        # Test return_components=True
        component_energies = hopfield(features, return_components=True)

        # Should return dict with per-modality energies
        assert isinstance(component_energies, dict)
        assert "video" in component_energies
        assert "audio" in component_energies

        # Each component should be scalar and negative
        for energy in component_energies.values():
            assert energy.dim() == 0
            assert torch.isfinite(energy)
            assert energy.item() <= 0

        # Total should equal sum of components
        manual_total = torch.tensor(0.0, device=device)
        for energy in component_energies.values():
            manual_total = manual_total + energy
        assert torch.allclose(total_energy, manual_total, atol=1e-6)

    def test_forward_with_multiple_modalities(self, device):
        """Test forward handles arbitrary number of modalities correctly."""
        modality_dims = {"video": 128, "audio": 64, "text": 32, "sensor": 16}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes={"video": 32, "audio": 16, "text": 8, "sensor": 4},
            cross_weight=0.25,
            device=device,
        )

        batch, seq_len = 3, 6
        features = {
            "video": torch.randn(batch, seq_len, 128, device=device),
            "audio": torch.randn(batch, seq_len, 64, device=device),
            "text": torch.randn(batch, seq_len, 32, device=device),
            "sensor": torch.randn(batch, seq_len, 16, device=device),
        }

        # Test total energy
        total_energy = hopfield(features)
        assert total_energy.dim() == 0
        assert torch.isfinite(total_energy)

        # Test component energies
        components = hopfield(features, return_components=True)
        assert len(components) == 4
        assert all(modality in components for modality in modality_dims)

        # All components should be finite and non-positive
        for energy in components.values():
            assert energy.dim() == 0
            assert torch.isfinite(energy)
            assert energy.item() <= 0

    def test_forward_cross_modal_influence_with_lambda_cross(self, device):
        """Test forward pass demonstrates cross-modal influence via lambda_cross parameter."""
        modality_dims = {"video": 64, "audio": 32}
        batch, seq_len = 1, 8

        # Fix inputs for reproducible comparison
        torch.manual_seed(42)
        features = {
            "video": torch.randn(batch, seq_len, 64, device=device),
            "audio": torch.randn(batch, seq_len, 32, device=device),
        }

        energies = {}
        for lambda_cross in [0.0, 0.5, 1.0]:
            hopfield = CrossModalHopfield(
                modality_dims=modality_dims,
                num_prototypes=16,
                cross_weight=lambda_cross,
                device=device,
            )

            # Reset random seed for fair comparison
            torch.manual_seed(42)
            energy = hopfield(features)
            energies[lambda_cross] = energy.item()

            assert torch.isfinite(energy)

        # Different lambda_cross values should produce different energies
        assert len(set(energies.values())) > 1

    def test_forward_gradient_flow_compatibility(self, device):
        """Test forward pass supports gradient flow for energy-based dynamics."""
        modality_dims = {"video": 16, "audio": 12}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=4,
            device=device,
        )

        batch, seq_len = 1, 3
        features = {
            "video": torch.randn(batch, seq_len, 16, device=device, requires_grad=True),
            "audio": torch.randn(batch, seq_len, 12, device=device, requires_grad=True),
        }

        # Forward pass
        energy = hopfield(features)

        # Backward pass should work for gradient-based dynamics
        energy.backward()

        # Check gradients exist and are finite
        for feature_tensor in features.values():
            assert feature_tensor.grad is not None
            assert torch.isfinite(feature_tensor.grad).all()
            assert feature_tensor.grad.abs().max() > 0  # Non-trivial gradients

    # Tests for retrieve_patterns method
    def test_retrieve_patterns_top_k_prototype_selection(self, device):
        """Test retrieve_patterns returns top-k most similar prototypes."""
        modality_dims = {"video": 8}  # Small for manual verification
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=4,
            device=device,
        )

        # Set known prototypes
        hopfield.prototypes["video"].data = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ξ₀
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ξ₁
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # ξ₂
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # ξ₃
            ],
            device=device,
        )

        # Query most similar to ξ₁
        features = {
            "video": torch.tensor(
                [[[0.1, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]], device=device
            )
        }

        indices, similarities = hopfield.retrieve_patterns(features, "video", top_k=3)

        # Check shapes
        assert indices.shape == (1, 1, 3)  # (batch, seq_len, top_k)
        assert similarities.shape == (1, 1, 3)

        # Most similar should be prototype 1
        assert indices[0, 0, 0].item() == 1
        assert similarities[0, 0, 0].item() > similarities[0, 0, 1].item()

    def test_retrieve_patterns_similarity_ordering(self, device):
        """Test similarities are returned in descending order."""
        modality_dims = {"audio": 32}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=16,
            device=device,
        )

        batch, seq_len = 2, 4
        features = {"audio": torch.randn(batch, seq_len, 32, device=device)}

        indices, similarities = hopfield.retrieve_patterns(features, "audio", top_k=8)

        # Check shapes
        assert indices.shape == (2, 4, 8)
        assert similarities.shape == (2, 4, 8)

        # Similarities should be in descending order for each position
        for b in range(batch):
            for t in range(seq_len):
                sims = similarities[b, t, :]
                # Check monotonic non-increasing
                assert (sims[:-1] >= sims[1:]).all()

    # Tests for validation and edge cases
    def test_empty_modality_dims_raises_error(self, device):
        """Test ValueError for empty modality_dims."""
        with pytest.raises(ValueError, match="modality_dims cannot be empty"):
            CrossModalHopfield(
                modality_dims={},
                device=device,
            )

    def test_mismatched_num_prototypes_dict_raises_error(self, device):
        """Test ValueError for mismatched num_prototypes dict keys."""
        modality_dims = {"video": 768, "audio": 512}
        num_prototypes = {"video": 256}  # Missing "audio"

        with pytest.raises(
            ValueError, match="num_prototypes keys must match modality_dims"
        ):
            CrossModalHopfield(
                modality_dims=modality_dims,
                num_prototypes=num_prototypes,
                device=device,
            )

    def test_invalid_modality_dimensions_raise_error(self, device):
        """Test ValueError for non-positive modality dimensions."""
        invalid_dims = {"video": 0, "audio": -512}

        with pytest.raises(
            ValueError, match="All modality dimensions must be positive"
        ):
            CrossModalHopfield(
                modality_dims=invalid_dims,
                device=device,
            )

    def test_invalid_num_prototypes_raise_error(self, device):
        """Test ValueError for non-positive prototype counts."""
        modality_dims = {"video": 768}

        with pytest.raises(ValueError, match="All prototype counts must be positive"):
            CrossModalHopfield(
                modality_dims=modality_dims,
                num_prototypes={"video": 0},
                device=device,
            )

    def test_cross_weight_validation_comprehensive(self, device):
        """Test cross_weight validation covers edge cases."""
        modality_dims = {"video": 64}

        # Valid edge cases
        for valid_weight in [0.0, 1.0]:
            hopfield = CrossModalHopfield(
                modality_dims=modality_dims,
                cross_weight=valid_weight,
                device=device,
            )
            assert hopfield.cross_weight == valid_weight

        # Invalid cases
        for invalid_weight in [-0.001, 1.001, float("inf"), float("-inf")]:
            with pytest.raises(ValueError, match=r"cross_weight must be in \[0, 1\]"):
                CrossModalHopfield(
                    modality_dims=modality_dims,
                    cross_weight=invalid_weight,
                    device=device,
                )

    def test_temporal_window_validation_comprehensive(self, device):
        """Test temporal_window validation covers all edge cases."""
        modality_dims = {"audio": 128}

        # Valid odd positive integers
        for valid_window in [1, 3, 5, 7, 9, 11]:
            hopfield = CrossModalHopfield(
                modality_dims=modality_dims,
                temporal_window=valid_window,
                device=device,
            )
            assert hopfield.temporal_window == valid_window

        # Invalid cases
        invalid_windows = [
            (0, "positive"),
            (-1, "positive"),
            (-5, "positive"),
            (2, "odd"),
            (4, "odd"),
            (10, "odd"),
        ]

        for invalid_window, _error_type in invalid_windows:
            with pytest.raises(ValueError, match="temporal_window"):
                CrossModalHopfield(
                    modality_dims=modality_dims,
                    temporal_window=invalid_window,
                    device=device,
                )
