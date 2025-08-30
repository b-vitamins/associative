"""Functional implementations for associative memory models."""

import math

import torch
from torch import Tensor
from torch.nn import functional


def energy_attention(
    query: Tensor,
    key: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    scale: float | None = None,
    mask: Tensor | None = None,
    q_bias: Tensor | None = None,
    k_bias: Tensor | None = None,
) -> Tensor:
    """Compute energy-based attention.

    Args:
        query: Query tensor [batch, seq_len, embed_dim]
        key: Key tensor [batch, seq_len, embed_dim]
        q_weight: Query projection weights [num_heads, qk_dim, embed_dim]
        k_weight: Key projection weights [num_heads, qk_dim, embed_dim]
        scale: Temperature scaling (default: 1/sqrt(qk_dim))
        mask: Optional attention mask
        q_bias: Optional query bias [qk_dim]
        k_bias: Optional key bias [qk_dim]

    Returns:
        Energy scalar
    """
    # Project queries and keys
    q = torch.einsum("bnd,hqd->bnhq", query, q_weight)
    k = torch.einsum("bnd,hqd->bnhq", key, k_weight)

    if q_bias is not None:
        q = q + q_bias
    if k_bias is not None:
        k = k + k_bias

    # Compute attention scores
    scores = torch.einsum("bnhq,bmhq->bhnm", q, k)

    if mask is not None:
        scores = scores * mask

    # Default scale
    if scale is None:
        qk_dim = q_weight.shape[1]
        scale = 1.0 / math.sqrt(qk_dim)

    # Compute energy
    return -torch.logsumexp(scale * scores, dim=-1).sum() / scale


def hopfield_energy(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    activation: str | None = "relu_squared",
) -> Tensor:
    """Compute Hopfield network energy.

    Args:
        x: Input tensor [batch, seq_len, in_features]
        weight: Projection weight [out_features, in_features]
        bias: Optional bias [out_features]
        activation: Energy activation function

    Returns:
        Energy scalar
    """
    # Linear projection
    hidden = functional.linear(x, weight, bias)

    # Apply activation
    if activation == "relu_squared":
        energy = -0.5 * (functional.relu(hidden) ** 2.0).sum()
    elif activation == "gelu":
        energy = -functional.gelu(hidden).sum()
    elif activation == "tanh":
        energy = -torch.tanh(hidden).abs().sum()
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return energy
