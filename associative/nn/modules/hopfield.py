"""Hopfield network layers for associative memory models.

This module implements modern Hopfield networks with various activation functions
and cross-modal memory capabilities. These networks store and retrieve patterns
through energy-based dynamics and support multimodal associative memory.

Classes:
    Hopfield: Standard Hopfield memory layer with configurable activations
    CrossModalHopfield: Multi-modality Hopfield with cross-modal interactions
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .config import HopfieldConfig
from .utils import Lambda


class Hopfield(nn.Module):
    """Hopfield network layer with configurable energy activation functions.

    Implements modern Hopfield networks that compute energy functions with various
    activation types including ReLU, GELU, tanh, softmax, and Manhattan distance
    variants. The energy is computed as -0.5 * activation(Wx)^2.sum() for most
    variants, with special handling for Manhattan distance.

    Attributes:
        in_features: Input feature dimension
        hidden_features: Hidden layer dimension
        config: Configuration object specifying activation type and other parameters
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        config: HopfieldConfig | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize Hopfield network layer.

        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension. If None, computed from config.hidden_dim_ratio
            config: Hopfield configuration object. Defaults to HopfieldConfig() if None
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        config = config or HopfieldConfig()
        hidden_features = hidden_features or int(in_features * config.hidden_dim_ratio)

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.config = config

        # Create the appropriate Hopfield variant based on activation type
        if config.activation_type == "relu":
            self.activation = Lambda(lambda x: -0.5 * (functional.relu(x) ** 2.0).sum())
        elif config.activation_type == "gelu":
            self.activation = Lambda(lambda x: -0.5 * (functional.gelu(x) ** 2.0).sum())
        elif config.activation_type == "tanh":
            self.activation = Lambda(lambda x: -0.5 * (torch.tanh(x) ** 2.0).sum())
        elif config.activation_type == "softmax":
            # LogSumExp (Softmax) variant
            self.beta = math.sqrt(hidden_features)
            self.activation = Lambda(
                lambda x: -0.5 * (torch.logsumexp(self.beta * x, dim=-1) ** 2.0).sum()
            )
        elif config.activation_type == "manhattan":
            # Manhattan distance variant - will be handled differently
            self.activation = None
        elif config.activation_type is None:
            # Use custom activation if provided
            self.activation = Lambda(config.activation)
        else:
            raise ValueError(f"Unknown activation type: {config.activation_type}")

        if config.activation_type == "manhattan":
            # For Manhattan variant, we need weight matrix instead of projection
            self.weight = nn.Parameter(
                torch.empty(hidden_features, in_features, **factory_kwargs)
            )
            if config.bias:
                self.bias = nn.Parameter(torch.empty(hidden_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)

            # Initialize parameters
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
        else:
            # Standard linear projection for other variants
            self.proj = nn.Linear(
                in_features, hidden_features, bias=config.bias, **factory_kwargs
            )

    def forward(self, x: Tensor) -> Tensor:
        """Compute Hopfield energy function.

        Args:
            x: Input tensor of shape (batch, seq_len, in_features) or (batch, in_features)

        Returns:
            Scalar energy value for the input patterns
        """
        if self.config.activation_type == "manhattan":
            # Manhattan distance energy - optimized computation
            # x: [batch, seq, in_features]
            # weight: [hidden_features, in_features]

            # More memory-efficient computation using einsum
            # Compute L1 distances without explicit broadcasting
            batch_size, seq_len = x.shape[:2]
            x_flat = x.view(-1, self.in_features)  # [batch*seq, in_features]

            # Compute distances for all pairs efficiently
            distances = torch.cdist(
                x_flat, self.weight, p=1
            )  # [batch*seq, hidden_features]
            distances = distances.view(batch_size, seq_len, self.hidden_features)

            # Normalize distances by dimensionality to prevent overflow
            distances = distances / self.in_features

            if self.bias is not None:
                distances = distances + self.bias

            # Energy is negative sum of similarities (inverse distances)
            # Use exp(-distances) to convert distances to similarities
            similarities = torch.exp(-distances)
            return -similarities.sum()
        # Standard variants
        hidden = self.proj(x)
        if self.activation is not None:
            return self.activation(hidden)
        raise ValueError(
            f"Invalid state: activation is None for type {self.config.activation_type}"
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, hidden_features={self.hidden_features}, activation={self.config.activation_type}"


class CrossModalHopfield(nn.Module):
    """Cross-modal Hopfield memory with saliency-weighted pattern retrieval.

    Implements modality-specific Hopfield memories influenced by complementary
    modalities through cross-modal saliency weights. The energy function combines
    intra-modal and cross-modal similarities with learnable prototypes.

    Energy function: E^HN = -Sum_m Sum_A Sum_mu G(alpha^{m,mu}_A * xi^{m,mu}^T g^m_A)
    where alpha blends intra-modal and cross-modal similarities.

    Attributes:
        modality_dims: Dictionary mapping modality names to their feature dimensions
        num_prototypes: Dictionary of prototype counts per modality
        cross_weight: Lambda_cross weight for cross-modal influence [0,1]
        temporal_window: Window size for temporal smoothing (must be odd)
        activation_type: Activation function type ("softplus" or "relu")
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        num_prototypes: dict[str, int] | int = 256,
        cross_weight: float = 0.3,
        temporal_window: int = 3,
        activation_type: str = "softplus",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize cross-modal Hopfield memory.

        Args:
            modality_dims: Dictionary mapping modality names to feature dimensions
            num_prototypes: Number of prototypes K per modality (int for all, or dict per modality)
            cross_weight: Lambda_cross weight for cross-modal influence, in range [0,1]
            temporal_window: Window size for temporal smoothing (must be positive odd number)
            activation_type: Activation function type ("softplus" or "relu")
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Validate inputs
        self._validate_inputs(
            modality_dims, cross_weight, temporal_window, activation_type
        )
        num_prototypes = self._process_num_prototypes(num_prototypes, modality_dims)

        # Store configuration
        self.modality_dims = modality_dims
        self.num_prototypes = num_prototypes
        self.cross_weight = cross_weight
        self.temporal_window = temporal_window
        self.activation_type = activation_type

        # Initialize activation function
        self.activation = self._create_activation_function(activation_type)

        # Initialize prototypes and cross-modal projections
        self._initialize_parameters(modality_dims, num_prototypes, factory_kwargs)

    def _validate_inputs(
        self,
        modality_dims: dict[str, int],
        cross_weight: float,
        temporal_window: int,
        activation_type: str,
    ) -> None:
        """Validate input parameters."""
        if not modality_dims:
            raise ValueError("modality_dims cannot be empty")
        if not all(dim > 0 for dim in modality_dims.values()):
            raise ValueError("All modality dimensions must be positive")
        if not 0.0 <= cross_weight <= 1.0:
            raise ValueError(f"cross_weight must be in [0, 1], got {cross_weight}")
        if temporal_window <= 0 or temporal_window % 2 == 0:
            raise ValueError(
                f"temporal_window must be positive odd, got {temporal_window}"
            )
        if activation_type not in ["softplus", "relu"]:
            raise ValueError(
                f"activation must be 'softplus' or 'relu', got {activation_type}"
            )

    def _process_num_prototypes(
        self,
        num_prototypes: dict[str, int] | int,
        modality_dims: dict[str, int],
    ) -> dict[str, int]:
        """Process and validate num_prototypes parameter."""
        if isinstance(num_prototypes, int):
            if num_prototypes <= 0:
                raise ValueError("num_prototypes must be positive")
            return {m: num_prototypes for m in modality_dims}

        if set(num_prototypes.keys()) != set(modality_dims.keys()):
            raise ValueError("num_prototypes keys must match modality_dims")
        if not all(k > 0 for k in num_prototypes.values()):
            raise ValueError("All prototype counts must be positive")
        return num_prototypes

    def _create_activation_function(self, activation_type: str):
        """Create the appropriate activation function."""
        if activation_type == "softplus":
            return functional.softplus
        return lambda x: 0.5 * (functional.relu(x) ** 2)

    def _initialize_parameters(
        self,
        modality_dims: dict[str, int],
        num_prototypes: dict[str, int],
        factory_kwargs: dict,
    ) -> None:
        """Initialize prototypes and cross-modal projection parameters."""
        self.prototypes = nn.ParameterDict()
        self.cross_projs = nn.ParameterDict()

        # Initialize prototypes
        for modality, dim in modality_dims.items():
            prototype = nn.Parameter(
                torch.empty(num_prototypes[modality], dim, **factory_kwargs)
            )
            # Xavier initialization
            bound = math.sqrt(6.0 / (dim + num_prototypes[modality]))
            nn.init.uniform_(prototype, -bound, bound)
            self.prototypes[modality] = prototype

        # Initialize cross-modal projections
        modalities = list(modality_dims.keys())
        for src_mod in modalities:
            for dst_mod in modalities:
                if src_mod != dst_mod:
                    src_dim = modality_dims[src_mod]
                    dst_dim = modality_dims[dst_mod]
                    proj_name = f"{src_mod}_to_{dst_mod}"
                    proj = nn.Parameter(torch.empty(dst_dim, src_dim, **factory_kwargs))
                    nn.init.xavier_uniform_(proj)
                    self.cross_projs[proj_name] = proj

    def compute_saliency_weights(
        self,
        features: Tensor,
        complementary: Tensor,
        modality: str,
    ) -> Tensor:
        """Compute cross-modal saliency weights alpha.

        Formula: alpha^{m,mu}_A = lambda_cross[xi^mu*W*g_bar]^2 + (1-lambda_cross)[xi^mu*g]^2

        Args:
            features: Current modality features g^m (batch, seq_len, dim)
            complementary: Complementary modality features g^m' (batch, seq_len, dim')
            modality: Current modality name

        Returns:
            Saliency weights (batch, seq_len, num_prototypes)
        """
        # Get prototypes for this modality
        prototypes = self.prototypes[modality]  # (K_m, D_m)

        # Apply temporal smoothing to complementary features
        smoothed = self.temporal_smooth(complementary)

        # Project complementary features to current modality dimension
        # Find the right projection matrix
        comp_modality = None
        for mod in self.modality_dims:
            if mod != modality and complementary.shape[-1] == self.modality_dims[mod]:
                comp_modality = mod
                break

        if comp_modality and f"{comp_modality}_to_{modality}" in self.cross_projs:
            proj = self.cross_projs[f"{comp_modality}_to_{modality}"]
            projected = torch.matmul(smoothed, proj.T)  # (batch, seq_len, D_m)
        else:
            # Fallback: truncate or pad to match dimension
            target_dim = self.modality_dims[modality]
            if smoothed.shape[-1] >= target_dim:
                projected = smoothed[..., :target_dim]
            else:
                pad_size = target_dim - smoothed.shape[-1]
                projected = functional.pad(smoothed, (0, pad_size))

        # Compute similarities: (batch, seq_len, D_m) @ (D_m, K_m) -> (batch, seq_len, K_m)
        intra_sim = torch.matmul(features, prototypes.T)
        cross_sim = torch.matmul(projected, prototypes.T)

        # Compute saliency as weighted sum of squared similarities
        return (
            self.cross_weight * cross_sim.square()
            + (1 - self.cross_weight) * intra_sim.square()
        )

    def temporal_smooth(self, features: Tensor) -> Tensor:
        """Apply temporal smoothing with window averaging.

        Formula: ḡ^m_A = 1/|N(A)| Σ_{B∈N(A)} g^m_B
        where N(A) = {B : |B - A| ≤ ⌊w/2⌋}

        Args:
            features: Input features (batch, seq_len, dim)

        Returns:
            Smoothed features (batch, seq_len, dim)
        """
        if self.temporal_window == 1:
            return features

        batch_size, seq_len, dim = features.shape
        half_window = self.temporal_window // 2

        # For very short sequences, average all positions
        if seq_len <= half_window:
            return features.mean(dim=1, keepdim=True).expand_as(features)

        # Use cumsum for efficient window averaging
        smoothed = torch.zeros_like(features)
        cumsum = torch.cumsum(features, dim=1)

        for t in range(seq_len):
            # Window boundaries: N(A) = {B : |B - A| ≤ ⌊w/2⌋}
            start = max(0, t - half_window)
            end = min(seq_len, t + half_window + 1)

            # Compute window average using cumsum
            if start == 0:
                window_sum = cumsum[:, end - 1, :]
            else:
                window_sum = cumsum[:, end - 1, :] - cumsum[:, start - 1, :]

            smoothed[:, t, :] = window_sum / (end - start)

        return smoothed

    def compute_memory_energy(
        self,
        features: Tensor,
        saliency: Tensor,
        modality: str,
    ) -> Tensor:
        """Compute Hopfield memory energy.

        Formula: E^HN_m = -Sum_A Sum_mu G(alpha^{m,mu}_A * xi^{m,mu}^T g^m_A)

        Args:
            features: Features g^m (batch, seq_len, dim)
            saliency: Saliency weights alpha (batch, seq_len, num_prototypes)
            modality: Modality name

        Returns:
            Scalar energy value
        """
        prototypes = self.prototypes[modality]  # (K_m, D_m)

        # Compute prototype similarities: ξ^μ·g
        similarities = torch.matmul(features, prototypes.T)  # (batch, seq_len, K_m)

        # Apply saliency weighting: alpha*(xi^mu*g)
        weighted = saliency * similarities

        # Apply activation function G and sum
        activated = self.activation(weighted)

        return -activated.sum()

    def forward(
        self,
        features: dict[str, Tensor],
        return_components: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute total cross-modal Hopfield energy.

        Args:
            features: Dict mapping modality names to features (batch, seq_len, dim)
            return_components: Whether to return per-modality energies

        Returns:
            Total energy (scalar) or dict of per-modality energies
        """
        modalities = list(features.keys())
        modality_energies = {}

        for i, modality in enumerate(modalities):
            # Get current and complementary modality features
            g_m = features[modality]

            # For cross-modal influence, use next modality in cycle
            if len(modalities) > 1:
                comp_idx = (i + 1) % len(modalities)
                g_comp = features[modalities[comp_idx]]
            else:
                g_comp = g_m  # Single modality case

            # Compute saliency weights and memory energy
            saliency = self.compute_saliency_weights(g_m, g_comp, modality)
            energy = self.compute_memory_energy(g_m, saliency, modality)
            modality_energies[modality] = energy

        if return_components:
            return modality_energies

        # Return total energy as tensor
        first_feature = next(iter(features.values()))
        total = torch.tensor(
            0.0, device=first_feature.device, dtype=first_feature.dtype
        )
        for energy in modality_energies.values():
            total = total + energy
        return total

    def retrieve_patterns(
        self,
        features: dict[str, Tensor],
        modality: str,
        top_k: int = 5,
    ) -> tuple[Tensor, Tensor]:
        """Retrieve closest prototype patterns.

        Args:
            features: Input features dict
            modality: Which modality to retrieve patterns for
            top_k: Number of closest prototypes to return

        Returns:
            (indices, similarities) both of shape (batch, seq_len, top_k)
        """
        g_m = features[modality]
        prototypes = self.prototypes[modality]

        # Compute similarities: g·ξ^T
        similarities = torch.matmul(g_m, prototypes.T)

        # Get top-k (returns values and indices)
        top_values, top_indices = torch.topk(
            similarities, k=top_k, dim=-1, largest=True
        )
        return top_indices, top_values

    def extra_repr(self) -> str:
        """String representation."""
        modality_info = ", ".join(
            f"{name}(D={dim}, K={self.num_prototypes.get(name, 0)})"
            for name, dim in self.modality_dims.items()
        )
        return f"modalities=[{modality_info}], cross_weight={self.cross_weight}"
