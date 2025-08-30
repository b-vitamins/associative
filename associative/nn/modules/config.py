"""Configuration classes for associative memory models."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.nn import functional


@dataclass(frozen=True)
class EnergyBlockConfig:
    """Configuration for energy transformer blocks.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        qk_dim: Query/Key dimension per head
        mlp_ratio: MLP hidden dimension ratio
        attn_bias: Whether to use bias in attention
        mlp_bias: Whether to use bias in MLP
        attn_beta: Attention temperature scaling
    """

    embed_dim: int
    num_heads: int = 12
    qk_dim: int = 64
    mlp_ratio: float = 4.0
    attn_bias: bool = False
    mlp_bias: bool = False
    attn_beta: float | None = None


@dataclass(frozen=True)
class EnergyAttentionConfig:
    """Configuration for energy-based attention layers.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        qk_dim: Query/Key dimension per head
        beta: Temperature scaling factor (default: 1/sqrt(qk_dim))
        bias: Whether to use bias in projections
    """

    embed_dim: int
    num_heads: int = 12
    qk_dim: int = 64
    beta: float | None = None
    bias: bool = False

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.qk_dim <= 0:
            raise ValueError(f"qk_dim must be positive, got {self.qk_dim}")


@dataclass(frozen=True)
class HopfieldConfig:
    """Configuration for Hopfield layers.

    Args:
        hidden_dim_ratio: Ratio of hidden dimension to input dimension
        activation_type: Type of activation function ("relu", "gelu", "tanh", "softmax", "manhattan")
        activation: Custom energy activation function (used if activation_type is None)
        bias: Whether to use bias in linear projection
    """

    hidden_dim_ratio: float = 4.0
    activation_type: str | None = "relu"
    activation: Callable[[torch.Tensor], torch.Tensor] = field(
        default=lambda x: -0.5 * (functional.relu(x) ** 2.0).sum()
    )
    bias: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim_ratio <= 0:
            raise ValueError(
                f"hidden_dim_ratio must be positive, got {self.hidden_dim_ratio}"
            )
        valid_types = {"relu", "gelu", "tanh", "softmax", "manhattan", None}
        if self.activation_type not in valid_types:
            raise ValueError(
                f"activation_type must be one of {valid_types}, got {self.activation_type}"
            )


@dataclass(frozen=True)
class EnergyTransformerConfig:
    """Configuration for associative memory models.

    Args:
        patch_size: Size of image patches (for vision models)
        num_patches: Number of patches (computed from image size)
        patch_dim: Dimension of patch embeddings
        input_dim: Input feature dimension (for graph models)
        embed_dim: Embedding dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        qk_dim: Query/Key dimension per head
        mlp_ratio: MLP hidden dimension ratio
        num_time_steps: Number of gradient descent steps per block
        step_size: Gradient descent step size
        attn_bias: Whether to use bias in attention
        mlp_bias: Whether to use bias in MLP
        attn_beta: Attention temperature scaling
        pos_encoding_dim: Positional encoding dimension (for graphs)
        attention_config: Optional attention configuration
        hopfield_config: Optional Hopfield configuration
        norm_eps: Layer normalization epsilon
        out_dim: Output dimension
    """

    # Common parameters
    embed_dim: int = 256
    num_layers: int = 1
    num_heads: int = 12
    qk_dim: int = 64
    mlp_ratio: float = 4.0
    num_time_steps: int = 12
    step_size: float = 1.0
    attn_bias: bool = False
    mlp_bias: bool = False
    attn_beta: float | None = None
    norm_eps: float = 1e-5
    out_dim: int | None = None
    hopfield_activation_type: str = "relu"  # Activation type for Hopfield layers

    # Stochastic gradient descent parameters
    use_noise: bool = False  # Whether to add noise during gradient descent
    noise_std: float = 0.02  # Standard deviation of noise
    noise_decay: bool = False  # Whether to decay noise over time steps
    noise_gamma: float = 0.55  # Decay factor for noise

    # Vision-specific parameters
    patch_size: int = 4
    num_patches: int = 64  # For 32x32 images with patch_size=4
    patch_dim: int | None = None

    # Graph-specific parameters
    input_dim: int | None = None
    pos_encoding_dim: int | None = None

    # Optional configurations
    attention_config: EnergyAttentionConfig | None = None
    hopfield_config: HopfieldConfig | None = None

    def _validate_base_params(self) -> None:
        """Validate base parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_time_steps <= 0:
            raise ValueError(
                f"num_time_steps must be positive, got {self.num_time_steps}"
            )
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")

    def _validate_vision_params(self) -> None:
        """Validate vision-specific parameters."""
        if self.input_dim is None and self.patch_size is not None:
            if self.patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {self.patch_size}")
            if self.num_patches <= 0:
                raise ValueError(
                    f"num_patches must be positive, got {self.num_patches}"
                )

    def _set_default_configs(self) -> None:
        """Set default configurations if not provided."""
        if self.attention_config is None:
            object.__setattr__(
                self,
                "attention_config",
                EnergyAttentionConfig(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    qk_dim=self.qk_dim,
                ),
            )

        if self.hopfield_config is None:
            object.__setattr__(
                self,
                "hopfield_config",
                HopfieldConfig(
                    hidden_dim_ratio=self.mlp_ratio,
                    activation_type=self.hopfield_activation_type,
                    bias=self.mlp_bias,
                ),
            )

    def _set_default_dims(self) -> None:
        """Set default dimensions."""
        if self.out_dim is None:
            if self.patch_size is not None and self.input_dim is None:
                # Vision model: use patch_dim if provided, else RGB patches
                if self.patch_dim is not None:
                    object.__setattr__(self, "out_dim", self.patch_dim)
                else:
                    object.__setattr__(self, "out_dim", self.patch_size**2 * 3)
            else:
                # Graph model: same as embed_dim
                object.__setattr__(self, "out_dim", self.embed_dim)

        if self.patch_dim is None and self.patch_size is not None:
            object.__setattr__(self, "patch_dim", self.patch_size**2 * 3)

    def __post_init__(self) -> None:
        self._validate_base_params()
        self._validate_vision_params()
        self._set_default_configs()
        self._set_default_dims()
