"""Energy-based attention mechanisms for associative memory models.

This module implements attention mechanisms that compute energy functions rather than
traditional softmax-based attention. The energy-based approach enables associative
memory dynamics through gradient-based optimization and supports various modalities
including vision, graphs, and multimodal scenarios.

Classes:
    EnergyAttention: Basic energy-based multi-head attention
    GraphEnergyAttention: Graph-aware attention with adjacency masking
    MultimodalEnergyAttention: Continuous compression attention for multiple modalities
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .basis import ContinuousCompression
else:
    ContinuousCompression = None
from .config import EnergyAttentionConfig
from .integrator import GaussLegendreIntegrator


class EnergyAttention(nn.Module):
    """Energy-based multi-head attention mechanism.

    Computes attention weights as an energy function using logsumexp rather than
    traditional softmax attention. The energy is computed as the negative logsumexp
    of scaled attention scores, enabling gradient-based optimization for associative
    memory dynamics.

    Attributes:
        embed_dim: Input embedding dimension
        num_heads: Number of attention heads
        qk_dim: Dimension of query/key projections
        scale: Temperature scaling factor (beta)
    """

    def __init__(
        self,
        config: EnergyAttentionConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize energy attention layer.

        Args:
            config: Configuration object specifying layer parameters
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.qk_dim = config.qk_dim
        self.scale = config.beta or (1.0 / math.sqrt(config.qk_dim))

        # Initialize projection weights
        self.query_proj = nn.Parameter(
            torch.empty(
                (config.num_heads, config.qk_dim, config.embed_dim), **factory_kwargs
            )
        )
        self.key_proj = nn.Parameter(
            torch.empty(
                (config.num_heads, config.qk_dim, config.embed_dim), **factory_kwargs
            )
        )

        if config.bias:
            self.query_bias = nn.Parameter(torch.empty(config.qk_dim, **factory_kwargs))
            self.key_bias = nn.Parameter(torch.empty(config.qk_dim, **factory_kwargs))
        else:
            self.register_parameter("query_bias", None)
            self.register_parameter("key_bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights using normal distribution."""
        nn.init.normal_(self.query_proj, mean=0.0, std=0.002)
        nn.init.normal_(self.key_proj, mean=0.0, std=0.002)
        if self.query_bias is not None:
            nn.init.zeros_(self.query_bias)
            nn.init.zeros_(self.key_bias)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Compute energy-based attention.

        Computes queries and keys from hidden states, then calculates attention scores
        and returns the negative logsumexp as the energy value for gradient descent.

        Args:
            hidden_states: Input features of shape [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask of shape [batch, num_heads, seq_len, seq_len]

        Returns:
            Scalar energy value for optimization
        """
        # Compute queries and keys
        queries = torch.einsum("bnd,hqd->bnhq", hidden_states, self.query_proj)
        keys = torch.einsum("bnd,hqd->bnhq", hidden_states, self.key_proj)

        if self.query_bias is not None:
            queries = queries + self.query_bias
            keys = keys + self.key_bias

        # Compute attention scores
        scores = torch.einsum("bnhq,bmhq->bhnm", queries, keys)

        if attention_mask is not None:
            scores = scores * attention_mask

        # Compute energy using logsumexp
        return -torch.logsumexp(self.scale * scores, dim=-1).sum() / self.scale

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"qk_dim={self.qk_dim}"
        )


class GraphEnergyAttention(nn.Module):
    """Graph-aware energy-based multi-head attention mechanism.

    Implements energy-based attention for graph data with adjacency matrix masking
    and head mixing weights. This mechanism combines multi-head attention with
    graph structure information through adjacency matrices and cross-head mixing.

    Attributes:
        embed_dim: Input embedding dimension
        num_heads: Number of attention heads
        qk_dim: Dimension of query/key projections
        head_mix: Parameter matrix for mixing attention heads
        temperature: Per-head temperature scaling factors
    """

    temperature: Tensor

    def __init__(
        self,
        config: EnergyAttentionConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize graph energy attention layer.

        Args:
            config: Configuration object specifying layer parameters
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.qk_dim = config.qk_dim

        # Initialize parameters following PyTorch naming conventions
        self.key_proj = nn.Parameter(
            torch.empty(
                (config.num_heads, config.qk_dim, config.embed_dim), **factory_kwargs
            )
        )
        self.query_proj = nn.Parameter(
            torch.empty(
                (config.num_heads, config.qk_dim, config.embed_dim), **factory_kwargs
            )
        )

        # Head mixing weights - unique to graph attention
        self.head_mix = nn.Parameter(
            torch.empty((config.num_heads, config.num_heads), **factory_kwargs)
        )

        if config.bias:
            self.key_bias = nn.Parameter(
                torch.empty((config.num_heads, config.qk_dim), **factory_kwargs)
            )
            self.query_bias = nn.Parameter(
                torch.empty((config.num_heads, config.qk_dim), **factory_kwargs)
            )
        else:
            self.register_parameter("key_bias", None)
            self.register_parameter("query_bias", None)

        # Temperature scaling (per head)
        beta_val = (
            config.beta if config.beta is not None else 1.0 / math.sqrt(config.qk_dim)
        )
        self.register_buffer(
            "temperature", torch.ones(config.num_heads, **factory_kwargs) * beta_val
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights using normal distribution."""
        nn.init.normal_(self.key_proj, mean=0.0, std=0.002)
        nn.init.normal_(self.query_proj, mean=0.0, std=0.002)
        nn.init.normal_(self.head_mix, mean=0.0, std=0.002)

        if self.key_bias is not None:
            nn.init.zeros_(self.key_bias)
            nn.init.zeros_(self.query_bias)

    def forward(
        self,
        hidden_states: Tensor,
        adjacency: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute graph-aware energy-based attention.

        Handles both batched and unbatched inputs with optional adjacency matrix
        masking. Applies head mixing and computes energy via logsumexp with
        proper handling of masked edges.

        Args:
            hidden_states: Input features of shape [batch, seq_len, embed_dim] or [seq_len, embed_dim]
            adjacency: Optional adjacency matrix of shape [seq_len, seq_len, num_heads]
            attention_mask: Unused, kept for compatibility

        Returns:
            Scalar energy value for optimization
        """
        # Handle both batched and unbatched inputs
        if hidden_states.dim() == 2:  # noqa: PLR2004
            # Unbatched [seq_len, embed_dim]
            return self._forward_single(hidden_states, adjacency)

        # Batched implementation for better efficiency
        if adjacency is None or adjacency.dim() == 3:  # noqa: PLR2004
            # Can process entire batch at once when adjacency is None or shared
            return self._forward_batched(hidden_states, adjacency)

        # Fall back to sequential processing only when necessary
        batch_size = hidden_states.shape[0]
        energies = []
        for i in range(batch_size):
            # Extract adjacency for this batch if provided
            adj_i = adjacency[i] if adjacency.dim() > 3 else adjacency  # noqa: PLR2004
            energies.append(self._forward_single(hidden_states[i], adj_i))
        return torch.stack(energies).sum()

    def _forward_batched(
        self, hidden_states: Tensor, adj: Tensor | None = None
    ) -> Tensor:
        """Efficient batched forward pass for multiple samples.

        Args:
            hidden_states: Batched input of shape [batch, seq_len, embed_dim]
            adj: Optional adjacency matrix shared across batch

        Returns:
            Scalar energy value summed over batch
        """
        hidden_states.shape[0]

        # Compute keys and queries for entire batch
        # [batch, seq, heads, head_dim]
        keys = torch.einsum("bkd,hzd->bkhz", hidden_states, self.key_proj)
        queries = torch.einsum("bqd,hzd->bqhz", hidden_states, self.query_proj)

        if self.key_bias is not None:
            keys = keys + self.key_bias
            queries = queries + self.query_bias

        # Attention scores: [batch, heads, seq, seq]
        scores = torch.einsum("h,bqhz,bkhz->bhqk", self.temperature, queries, keys)

        if adj is not None:
            # With adjacency matrix (shared across batch)
            # Mix heads and apply adjacency mask
            # [batch, seq, seq, heads]
            mixed_scores = torch.einsum("bhqk,hj->bqkj", scores, self.head_mix)
            mixed_scores = mixed_scores * adj

            # Mask empty edges
            mixed_scores = torch.where(mixed_scores == 0, float("-inf"), mixed_scores)

            # Compute logsumexp over key dimension
            lse_scores = torch.logsumexp(mixed_scores, dim=2)  # [batch, seq, heads]

            # Handle -inf values
            lse_scores = torch.where(lse_scores == float("-inf"), 0.0, lse_scores)

            # Sum over sequence dimension
            head_sums = lse_scores.sum(1)  # [batch, heads]

            # Final energy
            energies = ((-1.0 / self.temperature) * head_sums).sum(1)  # [batch]
            return energies.sum()

        # Without adjacency
        mixed_scores = torch.einsum("bhqk,hj->bqkj", scores, self.head_mix)
        lse_scores = torch.logsumexp(mixed_scores, dim=2)  # [batch, seq, heads]
        head_sums = lse_scores.sum(1)  # [batch, heads]
        energies = ((-1.0 / self.temperature) * head_sums).sum(1)  # [batch]
        return energies.sum()

    def _forward_single(self, g: Tensor, adj: Tensor | None = None) -> Tensor:
        """Forward pass for single unbatched example.

        Args:
            g: Input features of shape [seq_len, embed_dim]
            adj: Optional adjacency matrix of shape [seq_len, seq_len, num_heads]

        Returns:
            Scalar energy value for single example
        """
        # Compute keys and queries
        keys = torch.einsum("kd,hzd->khz", g, self.key_proj)  # [seq, heads, head_dim]
        queries = torch.einsum(
            "qd,hzd->qhz", g, self.query_proj
        )  # [seq, heads, head_dim]

        if self.key_bias is not None:
            keys = keys + self.key_bias
            queries = queries + self.query_bias

        # Attention scores: [heads, seq, seq]
        scores = torch.einsum("h,qhz,khz->hqk", self.temperature, queries, keys)

        if adj is not None:
            # With adjacency matrix
            # Mix heads and apply adjacency mask
            mixed_scores = (scores.permute(1, 2, 0) @ self.head_mix) * adj

            # Mask empty edges
            mixed_scores = torch.where(mixed_scores == 0, float("-inf"), mixed_scores)

            # Compute logsumexp over key dimension
            lse_scores = torch.logsumexp(mixed_scores, dim=1)  # [seq, heads]

            # Handle -inf values
            lse_scores = torch.where(lse_scores == float("-inf"), 0.0, lse_scores)

            # Sum over sequence dimension
            head_sums = lse_scores.sum(0)  # [heads]

            # Final energy
            return ((-1.0 / self.temperature) * head_sums).sum()
        # Without adjacency
        mixed_scores = scores.permute(1, 2, 0) @ self.head_mix
        lse_scores = torch.logsumexp(mixed_scores, dim=1)  # [seq, heads]
        head_sums = lse_scores.sum(0)  # [heads]
        return ((-1.0 / self.temperature) * head_sums).sum()

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"qk_dim={self.qk_dim}"
        )


class MultimodalEnergyAttention(nn.Module):
    """Multimodal energy attention with continuous compression.

    Implements continuous memory compression for multimodal attention across
    heterogeneous input dimensions. Uses basis functions to compress memory
    representations and supports cross-modal attention between different
    modality types through learned projections.

    Attributes:
        modality_configs: Configuration for each modality
        modalities: List of modality names
        cross_modal_pairs: Pairs of modalities for cross-attention
        integrator: Numerical integrator for partition function computation
        beta: Temperature scaling factor for numerical stability
    """

    def __init__(
        self,
        modality_configs: dict[str, dict],
        cross_modal_pairs: list[tuple[str, str]] | None = None,
        num_integration_points: int = 50,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize multimodal energy attention layer.

        Args:
            modality_configs: Dictionary mapping modality names to config dictionaries
            cross_modal_pairs: List of modality pairs for cross-attention. Defaults to all pairs.
            num_integration_points: Number of quadrature points for numerical integration. Defaults to 50.
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.modality_configs = modality_configs
        self.modalities = list(modality_configs.keys())

        # Set up cross-modal pairs (all pairs if None)
        if cross_modal_pairs is None:
            cross_modal_pairs = [
                (m1, m2) for m1 in self.modalities for m2 in self.modalities if m1 != m2
            ]
        self.cross_modal_pairs = cross_modal_pairs

        # Validate configs and set defaults
        for config in modality_configs.values():
            config.setdefault("basis_type", "rectangular")
            config.setdefault("regularization", 0.01)

        # Initialize parameters and modules
        self.query_proj = nn.ParameterDict()
        self.key_proj = nn.ParameterDict()
        self.cross_proj = nn.ParameterDict()
        self.compressions: nn.ModuleDict = nn.ModuleDict()

        # Create projections and compressions for each modality
        for name, config in modality_configs.items():
            # Query/key projections: (num_heads, qk_dim, embed_dim)
            shape = (config["num_heads"], config["qk_dim"], config["embed_dim"])
            self.query_proj[name] = nn.Parameter(torch.empty(shape, **factory_kwargs))
            self.key_proj[name] = nn.Parameter(torch.empty(shape, **factory_kwargs))

            # Initialize projections
            nn.init.normal_(self.query_proj[name], std=0.002)
            nn.init.normal_(self.key_proj[name], std=0.002)

            # Compression module
            from .basis import ContinuousCompression, create_basis

            basis = create_basis(config["basis_type"], config["compression_dim"])
            self.compressions[name] = ContinuousCompression(
                basis=basis,
                regularization=config["regularization"],
                cache_operators=True,
            )

        # Create cross-modal projections
        for source, target in cross_modal_pairs:
            source_dim = modality_configs[source]["embed_dim"]
            target_dim = modality_configs[target]["embed_dim"]

            # W^{source→target} ∈ R^{target_dim x source_dim}
            key = f"{source}_to_{target}"
            self.cross_proj[key] = nn.Parameter(
                torch.empty(target_dim, source_dim, **factory_kwargs)
            )
            nn.init.normal_(self.cross_proj[key], std=0.002)

        # Numerical integrator for partition functions
        self.integrator = GaussLegendreIntegrator(
            domain=(0.0, 1.0), num_points=num_integration_points
        )

        # Store beta = 1/√Y (assumed same qk_dim for all modalities)
        self.beta = 1.0 / math.sqrt(next(iter(modality_configs.values()))["qk_dim"])

    def _get_beta(self) -> float:
        """Get beta scaling factor for numerical stability.

        Returns:
            Beta value: β = 1/√Y where Y is the query/key dimension
        """
        return self.beta

    def _compute_intra_energy(self, x: Tensor, modality: str) -> Tensor:
        """Compute intra-modal energy with regularization.

        Computes E^intra = -1/β Σ log ∫exp(βs(t))dt + ||x||²/2 where s(t) are
        attention scores with continuous compression.

        Args:
            x: Input features for modality
            modality: Name of the modality

        Returns:
            Scalar energy value including regularization term
        """
        # Project to queries and keys
        queries = torch.einsum("bld,hqd->blhq", x, self.query_proj[modality])
        keys = torch.einsum("bld,hqd->blhq", x, self.key_proj[modality])

        # Reshape for compression: (batch, heads, qk_dim, seq_len)
        queries = queries.permute(0, 2, 3, 1)
        keys = keys.permute(0, 2, 3, 1)

        # Compress keys: L → M dimensions
        compression_module = cast("ContinuousCompression", self.compressions[modality])
        compressed = compression_module.compress(keys)

        # Compute partition function via numerical integration
        log_z = self._compute_log_partition(queries, compressed, modality)

        # Energy with regularization
        return -log_z / self.beta + 0.5 * torch.sum(x**2)

    def _compute_cross_energy(
        self, x_source: Tensor, x_target: Tensor, source: str, target: str
    ) -> Tensor:
        """Compute cross-modal energy between two modalities.

        Computes E^cross = -1/β Σ log ∫exp(βs(t))dt where s(t) are attention
        scores between projected source features and compressed target keys.

        Args:
            x_source: Source modality features
            x_target: Target modality features
            source: Name of source modality
            target: Name of target modality

        Returns:
            Scalar cross-modal energy value
        """
        # Project source features to target space
        proj_key = f"{source}_to_{target}"
        x_proj = torch.einsum("bld,Dd->blD", x_source, self.cross_proj[proj_key])

        # Get queries from projected source (using target's projections)
        queries = torch.einsum("bld,hqd->blhq", x_proj, self.query_proj[target])
        queries = queries.permute(0, 2, 3, 1)

        # Get compressed keys from target
        keys = torch.einsum("bld,hqd->blhq", x_target, self.key_proj[target])
        keys = keys.permute(0, 2, 3, 1)
        compression_module = cast("ContinuousCompression", self.compressions[target])
        compressed = compression_module.compress(keys)

        # Compute partition function
        log_z = self._compute_log_partition(queries, compressed, target)

        # Cross-modal energy without regularization
        return -log_z / self.beta

    def compress_modality(
        self, features: Tensor, modality: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Helper method for tests: compute queries, compressed keys, and original keys."""
        # Project to queries and keys
        queries = torch.einsum("bld,hqd->blhq", features, self.query_proj[modality])
        keys = torch.einsum("bld,hqd->blhq", features, self.key_proj[modality])

        # Reshape: (batch, heads, qk_dim, seq_len)
        queries = queries.permute(0, 2, 3, 1)
        keys = keys.permute(0, 2, 3, 1)

        # Compress keys
        compression_module = cast("ContinuousCompression", self.compressions[modality])
        compressed = compression_module.compress(keys)

        return queries, compressed, keys

    def _compute_log_partition(
        self, queries: Tensor, compressed_keys: Tensor, modality: str
    ) -> Tensor:
        """Compute log Z = log ∫exp(βs(t))dt using Gauss-Legendre quadrature."""
        # Get quadrature points and weights
        t_points, weights = self.integrator.get_quadrature_points()

        # Reconstruct keys at quadrature points
        # compressed_keys: (batch, heads, qk_dim, M)
        # reconstructed: (batch, heads, qk_dim, num_points)
        compression_module = cast("ContinuousCompression", self.compressions[modality])
        reconstructed = compression_module.reconstruct(compressed_keys, t_points)

        # Compute scores s(t) = Q^T K̄(t)
        # queries: (batch, heads, qk_dim, seq_len)
        # scores: (batch, heads, seq_len, num_points)
        scores = torch.einsum("bhql,bhqt->bhlt", queries, reconstructed)

        # LogSumExp trick for numerical stability
        max_scores = scores.max(dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(self.beta * (scores - max_scores))

        # Numerical integration: ∫f(t)dt ≈ Σ w_i f(t_i)
        integral = torch.sum(exp_scores * weights.view(1, 1, 1, -1), dim=-1)

        # Complete LogSumExp and sum over batch, heads, sequence
        log_integral = torch.log(integral + 1e-15) + self.beta * max_scores.squeeze(-1)
        return log_integral.sum()

    def forward(
        self,
        features: dict[str, Tensor],
        return_breakdown: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute total multimodal energy E = E^intra + E^cross.

        Args:
            features: Dict of modality_name -> (batch, seq_len, embed_dim)
            return_breakdown: Whether to return component breakdown

        Returns:
            Total energy (scalar) or dict with breakdown if requested
        """
        energies = {}

        # Compute intra-modal energies
        for modality, x in features.items():
            energies[f"intra_{modality}"] = self._compute_intra_energy(x, modality)

        # Compute cross-modal energies
        for source, target in self.cross_modal_pairs:
            if source in features and target in features:
                key = f"cross_{source}_{target}"
                energies[key] = self._compute_cross_energy(
                    features[source], features[target], source, target
                )

        if energies:
            total_val = sum(energies.values())
            total = (
                total_val
                if isinstance(total_val, torch.Tensor)
                else torch.tensor(float(total_val))
            )
        else:
            total = torch.tensor(0.0)

        if return_breakdown:
            energies["total"] = total
            return energies
        return total

    # Keep these for compatibility with tests
    def compute_intra_modal_energy(
        self, features: dict[str, Tensor], return_components: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Compute intra-modal energies (kept for test compatibility)."""
        components = {}
        for modality, x in features.items():
            components[modality] = self._compute_intra_energy(x, modality)
        if return_components:
            return components
        if not components:
            return torch.tensor(0.0)
        total_val = sum(components.values())
        return (
            total_val
            if isinstance(total_val, torch.Tensor)
            else torch.tensor(float(total_val))
        )

    def compute_cross_modal_energy(
        self, features: dict[str, Tensor], return_components: bool = False
    ) -> Tensor | dict[tuple[str, str], Tensor]:
        """Compute cross-modal energies (kept for test compatibility)."""
        components = {}
        for source, target in self.cross_modal_pairs:
            if source in features and target in features:
                components[(source, target)] = self._compute_cross_energy(
                    features[source], features[target], source, target
                )
        if return_components:
            return components
        if not components:
            return torch.tensor(0.0)
        total_val = sum(components.values())
        return (
            total_val
            if isinstance(total_val, torch.Tensor)
            else torch.tensor(float(total_val))
        )

    def compute_partition_function(
        self, scores_func: Callable[[Tensor], Tensor], beta: float
    ) -> Tensor:
        """Compute partition function (kept for test compatibility)."""
        # This is a bit hacky but needed for tests that call this directly
        t_points, weights = self.integrator.get_quadrature_points()
        scores = scores_func(t_points)
        max_scores = scores.max(dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(beta * (scores - max_scores))
        integral = torch.sum(exp_scores * weights.view(1, 1, 1, -1), dim=-1)
        log_integral = torch.log(integral + 1e-15) + beta * max_scores.squeeze(-1)
        return log_integral.sum()

    def extra_repr(self) -> str:
        return (
            f"modalities={self.modalities}, "
            f"cross_pairs={len(self.cross_modal_pairs)}, "
            f"integration_points={self.integrator.num_points}"
        )
