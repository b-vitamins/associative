"""Normalization layers for associative memory models."""

import torch
from torch import Tensor, nn


class EnergyLayerNorm(nn.Module):
    """Energy-based layer normalization.

    Similar to LayerNorm but with single gamma parameter.
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
