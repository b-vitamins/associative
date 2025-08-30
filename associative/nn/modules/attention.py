"""Energy-based attention mechanisms."""

import math

import torch
from torch import Tensor, nn

from .config import EnergyAttentionConfig


class EnergyAttention(nn.Module):
    """Energy-based multi-head attention.

    Computes attention as an energy function rather than
    traditional softmax attention.
    """

    def __init__(
        self,
        config: EnergyAttentionConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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
        nn.init.normal_(self.query_proj, mean=0.0, std=0.002)
        nn.init.normal_(self.key_proj, mean=0.0, std=0.002)
        if self.query_bias is not None:
            nn.init.zeros_(self.query_bias)
            nn.init.zeros_(self.key_bias)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, embed_dim]
            attention_mask: Optional [batch, num_heads, seq_len, seq_len]

        Returns:
            energy: Scalar energy value
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
    """Graph-aware energy-based multi-head attention.

    Implements the exact attention mechanism from JAX graph ET,
    including head mixing weights and proper adjacency masking.
    """

    temperature: Tensor

    def __init__(
        self,
        config: EnergyAttentionConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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
        """
        Forward pass matching JAX implementation exactly.

        Args:
            hidden_states: [batch, seq_len, embed_dim] or [seq_len, embed_dim]
            adjacency: Optional [seq_len, seq_len, num_heads] adjacency matrix
            attention_mask: Not used (for compatibility)

        Returns:
            energy: Scalar energy value
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
        """Efficient batched forward pass."""
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
        """Forward for single example matching JAX exactly."""
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
