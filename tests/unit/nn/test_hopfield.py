"""Comprehensive unit tests for Hopfield module and variants."""

import math

import pytest
import torch

from associative.nn.modules import Hopfield
from associative.nn.modules.config import HopfieldConfig
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
