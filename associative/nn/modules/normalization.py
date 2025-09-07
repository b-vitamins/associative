"""Normalization layers for associative memory models.

This module provides specialized normalization layers optimized for energy-based
associative memory models, including energy-specific variants of layer normalization.

Classes:
    EnergyLayerNorm: Layer normalization with single gamma parameter for energy models
"""

import torch
from torch import Tensor, nn


class EnergyLayerNorm(nn.Module):
    """Energy-based layer normalization with simplified parameterization.

    Similar to standard LayerNorm but uses a single scalar gamma parameter instead
    of per-element weights. This reduces parameters while maintaining normalization
    benefits for energy-based models.

    The normalization is computed as:
    output = gamma * (input - mean) / sqrt(var + eps) + bias

    Attributes:
        normalized_shape: Shape of the normalization (single int for feature dimension)
        eps: Small epsilon for numerical stability
        elementwise_affine: Whether to apply learnable affine transformation
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize energy layer normalization.

        Args:
            normalized_shape: Feature dimension to normalize over
            eps: Small epsilon for numerical stability. Defaults to 1e-5.
            elementwise_affine: Whether to learn affine parameters. Defaults to True.
            bias: Whether to use bias parameter. Defaults to True.
            device: Device to place parameters on. Defaults to None.
            dtype: Data type for parameters. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(1, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        """Apply energy layer normalization.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute mean and variance using more efficient single-pass algorithm
        mean = x.mean(dim=-1, keepdim=True)
        # Use E[X²] - E[X]² formula which requires only one pass through the data
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation
        if self.weight is not None:
            normalized = normalized * self.weight
        if self.bias is not None:
            normalized = normalized + self.bias

        return normalized

    def extra_repr(self) -> str:
        return (
            f"{self.normalized_shape[0]}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
