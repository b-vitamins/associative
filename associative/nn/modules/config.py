"""Configuration classes for associative memory models.

This module provides dataclass configurations for various components of associative
memory models, including transformers, attention mechanisms, Hopfield networks, and
specialized modules for multimodal and continuous memory systems.

Classes:
    EnergyBlockConfig: Configuration for energy transformer blocks
    EnergyAttentionConfig: Configuration for energy-based attention layers
    HopfieldConfig: Configuration for Hopfield memory layers
    EnergyTransformerConfig: Main configuration for energy transformers
    BasisConfig: Configuration for basis functions in continuous memory
    ContinuousHopfieldConfig: Configuration for continuous Hopfield networks
    METConfig: Configuration for Multimodal Energy Transformers
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.nn import functional


@dataclass(frozen=True)
class EnergyBlockConfig:
    """Configuration for energy transformer blocks.

    Defines parameters for individual transformer blocks in energy-based models,
    including attention and MLP layer configurations.

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        qk_dim: Query/Key dimension per head
        mlp_ratio: MLP hidden dimension ratio
        attn_bias: Whether to use bias in attention projections
        mlp_bias: Whether to use bias in MLP layers
        attn_beta: Attention temperature scaling factor
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

    Specifies parameters for attention mechanisms that compute energy functions
    instead of traditional softmax attention weights.

    Attributes:
        embed_dim: Input embedding dimension
        num_heads: Number of attention heads
        qk_dim: Query/Key dimension per head
        beta: Temperature scaling factor (defaults to 1/sqrt(qk_dim) if None)
        bias: Whether to use bias terms in query/key projections
    """

    embed_dim: int
    num_heads: int = 12
    qk_dim: int = 64
    beta: float | None = None
    bias: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.qk_dim <= 0:
            raise ValueError(f"qk_dim must be positive, got {self.qk_dim}")


@dataclass(frozen=True)
class HopfieldConfig:
    """Configuration for Hopfield memory layers.

    Defines parameters for modern Hopfield networks that store and retrieve
    patterns through energy-based dynamics.

    Attributes:
        hidden_dim_ratio: Ratio of hidden dimension to input dimension
        activation_type: Type of activation function ("relu", "gelu", "tanh", "softmax", "manhattan")
        activation: Custom energy activation function (used if activation_type is None)
        bias: Whether to use bias in linear projection layers
    """

    hidden_dim_ratio: float = 4.0
    activation_type: str | None = "relu"
    activation: Callable[[torch.Tensor], torch.Tensor] = field(
        default=lambda x: -0.5 * (functional.relu(x) ** 2.0).sum()
    )
    bias: bool = False

    def __post_init__(self) -> None:
        """Validate Hopfield configuration parameters."""
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

    # Mixed precision training
    enable_amp: bool = False  # Enable automatic mixed precision

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


@dataclass(frozen=True)
class BasisConfig:
    """Configuration for basis functions.

    Args:
        num_basis: Number of basis functions
        basis_type: Type of basis ("rectangular", "gaussian", "fourier", "polynomial")
        domain: Domain of basis functions as (start, end)
        learnable: Whether basis parameters are learnable
        overlap: Overlap factor for rectangular basis (0-1)
        init_width: Initial width for Gaussian basis
        max_frequency: Maximum frequency for Fourier basis
    """

    num_basis: int
    basis_type: str = "rectangular"
    domain: tuple[float, float] = (0.0, 1.0)
    learnable: bool = False
    overlap: float = 0.0
    init_width: float | None = None
    max_frequency: int | None = None

    def __post_init__(self) -> None:
        if self.num_basis <= 0:
            raise ValueError(f"num_basis must be positive, got {self.num_basis}")

        valid_types = {"rectangular", "gaussian", "fourier", "polynomial"}
        if self.basis_type not in valid_types:
            raise ValueError(
                f"basis_type must be one of {valid_types}, got {self.basis_type}"
            )

        if self.domain[0] >= self.domain[1]:
            raise ValueError(f"Invalid domain {self.domain}, start must be < end")

        if not 0 <= self.overlap <= 1:
            raise ValueError(f"overlap must be in [0, 1], got {self.overlap}")


@dataclass(frozen=True)
class ContinuousHopfieldConfig:
    """Configuration for continuous Hopfield networks.

    Args:
        basis_config: Configuration for basis functions
        beta: Inverse temperature parameter
        regularization: Ridge regression regularization
        integration_points: Number of points for numerical integration
        num_iterations: Number of iterations for updates (default: 3)
        use_analytical_update: Whether to use analytical CCCP solution
        memory_compression: Target compression ratio N/L (None = use basis_config.num_basis)
    """

    basis_config: BasisConfig
    beta: float = 1.0
    regularization: float = 0.5
    integration_points: int = 500
    num_iterations: int = 3
    use_analytical_update: bool = True
    memory_compression: float | None = None

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.regularization <= 0:
            raise ValueError(
                f"regularization must be positive, got {self.regularization}"
            )
        if self.integration_points <= 0:
            raise ValueError(
                f"integration_points must be positive, got {self.integration_points}"
            )
        if self.memory_compression is not None and not 0 < self.memory_compression <= 1:
            raise ValueError(
                f"memory_compression must be in (0, 1], got {self.memory_compression}"
            )


@dataclass(frozen=True)
class METConfig(EnergyTransformerConfig):
    """Configuration for Multimodal Energy Transformer.

    Extends EnergyTransformerConfig with multimodal-specific parameters
    for continuous compression, cross-modal attention, and gradient flow.

    Args:
        # Modality specifications
        modality_configs: Dict mapping modality names to their configs
        cross_modal_pairs: List of enabled cross-modal attention pairs

        # Compression parameters
        compression_dims: Dict of compression dimensions M per modality
        basis_types: Dict of basis function types per modality
        regularizations: Dict of ridge regression λ per modality

        # Cross-modal parameters
        cross_modal_weight: λ_cross for cross-modal influence [0,1]
        temporal_window: Window size for temporal smoothing

        # Hopfield memory
        num_prototypes: Number of prototypes K per modality
        hopfield_activation: Activation function for memory energy

        # Integration parameters
        integrator_method: Numerical integration method
        integration_points: Number of quadrature points

        # Gradient flow dynamics
        max_iterations: Maximum iterations per block
        step_size: Gradient descent step size η
        time_constant: Dynamics time constant τ
        convergence_tolerance: Convergence criterion ε

        # Architecture
        num_blocks: Number of MET blocks
        share_cross_projections: Whether to share projections across blocks

        # Training
        track_trajectories: Whether to save evolution trajectories
        use_stochastic_depth: Whether to use stochastic depth
        stochastic_depth_rate: Drop rate for stochastic depth
    """

    # Modality specifications
    modality_configs: dict[str, dict] | None = None
    cross_modal_pairs: list[tuple[str, str]] | None = None

    # Compression parameters (defaults for video/audio)
    compression_dims: dict[str, int] | None = None
    basis_types: dict[str, str] | None = None
    regularizations: dict[str, float] | None = None

    # Cross-modal parameters
    cross_modal_weight: float = 0.3
    temporal_window: int = 3

    # Hopfield memory
    num_prototypes: dict[str, int] | int = 256
    hopfield_activation: str = "softplus"

    # Integration parameters
    integrator_method: str = "gauss"
    integration_points: int = 50

    # Gradient flow dynamics
    max_iterations: int = 20
    step_size: float = 0.001  # η for gradient descent (optimal for stable convergence)
    time_constant: float = 1.0
    convergence_tolerance: float = 1e-4

    # Architecture
    num_blocks: int = 4
    share_cross_projections: bool = False

    # Training
    track_trajectories: bool = False
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1

    # Mixed precision training
    enable_amp: bool = False  # Enable automatic mixed precision

    def __post_init__(self) -> None:
        """Validate and set default MET configurations."""
        # Call parent post_init
        super().__post_init__()

        # Set default modality configs if not provided
        if self.modality_configs is None:
            object.__setattr__(
                self,
                "modality_configs",
                {
                    "video": {
                        "embed_dim": 768,
                        "compression_dim": 100,
                        "num_heads": 8,
                        "qk_dim": 64,
                        "basis_type": "rectangular",
                        "regularization": 0.01,
                    },
                    "audio": {
                        "embed_dim": 512,
                        "compression_dim": 100,
                        "num_heads": 8,
                        "qk_dim": 64,
                        "basis_type": "rectangular",
                        "regularization": 0.01,
                    },
                },
            )

        # Extract compression dims if not explicitly set
        if self.compression_dims is None and self.modality_configs:
            object.__setattr__(
                self,
                "compression_dims",
                {
                    name: cfg.get("compression_dim", 100)
                    for name, cfg in self.modality_configs.items()
                },
            )

        # Extract basis types if not explicitly set
        if self.basis_types is None and self.modality_configs:
            object.__setattr__(
                self,
                "basis_types",
                {
                    name: cfg.get("basis_type", "rectangular")
                    for name, cfg in self.modality_configs.items()
                },
            )

        # Extract regularizations if not explicitly set
        if self.regularizations is None and self.modality_configs:
            object.__setattr__(
                self,
                "regularizations",
                {
                    name: cfg.get("regularization", 0.01)
                    for name, cfg in self.modality_configs.items()
                },
            )

        # Set default cross-modal pairs (all pairs) if not specified
        if self.cross_modal_pairs is None and self.modality_configs:
            modalities = list(self.modality_configs.keys())
            pairs = []
            for i, m1 in enumerate(modalities):
                for m2 in modalities[i + 1 :]:
                    pairs.extend([(m1, m2), (m2, m1)])
            object.__setattr__(self, "cross_modal_pairs", pairs)

        # Validate parameters
        self._validate_met_params()

    def _validate_met_params(self) -> None:
        """Validate MET-specific parameters."""
        self._validate_cross_modal_params()
        self._validate_gradient_flow_params()
        self._validate_method_params()

    def _validate_cross_modal_params(self) -> None:
        """Validate cross-modal related parameters."""
        # Validate cross-modal weight
        if not 0 <= self.cross_modal_weight <= 1:
            raise ValueError(
                f"cross_modal_weight must be in [0, 1], got {self.cross_modal_weight}"
            )

        # Validate temporal window
        if self.temporal_window <= 0 or self.temporal_window % 2 == 0:
            raise ValueError(
                f"temporal_window must be positive odd number, got {self.temporal_window}"
            )

        # Validate integration points
        if self.integration_points <= 0:
            raise ValueError(
                f"integration_points must be positive, got {self.integration_points}"
            )

    def _validate_gradient_flow_params(self) -> None:
        """Validate gradient flow parameters."""
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.time_constant <= 0:
            raise ValueError(
                f"time_constant must be positive, got {self.time_constant}"
            )
        if self.convergence_tolerance <= 0:
            raise ValueError(
                f"convergence_tolerance must be positive, got {self.convergence_tolerance}"
            )

        # Validate stochastic depth rate
        if not 0 <= self.stochastic_depth_rate <= 1:
            raise ValueError(
                f"stochastic_depth_rate must be in [0, 1], got {self.stochastic_depth_rate}"
            )

    def _validate_method_params(self) -> None:
        """Validate method and activation parameters."""
        # Validate integrator method
        valid_methods = {"trapezoidal", "simpson", "gauss", "monte_carlo", "adaptive"}
        if self.integrator_method not in valid_methods:
            raise ValueError(
                f"integrator_method must be one of {valid_methods}, got {self.integrator_method}"
            )

        # Validate Hopfield activation
        valid_activations = {"softplus", "relu"}
        if self.hopfield_activation not in valid_activations:
            raise ValueError(
                f"hopfield_activation must be one of {valid_activations}, "
                f"got {self.hopfield_activation}"
            )
