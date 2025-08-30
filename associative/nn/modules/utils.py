"""Utility modules for associative memory models."""

from collections.abc import Callable

from torch import Tensor, nn


class Lambda(nn.Module):
    """Wrapper for lambda functions to make them nn.Module compatible."""

    def __init__(self, fn: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)
