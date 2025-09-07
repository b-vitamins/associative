"""Utility modules and helper classes for associative memory models.

This module provides small utility classes that support the main associative
memory components, including functional wrappers and helper modules.

Classes:
    Lambda: Wrapper to convert functions into nn.Module instances
"""

from collections.abc import Callable

from torch import Tensor, nn


class Lambda(nn.Module):
    """Wrapper for lambda functions to make them nn.Module compatible.

    Allows arbitrary functions to be used as PyTorch modules, enabling them
    to be part of the model graph and participate in gradient computation.
    This is particularly useful for activation functions in energy-based models.

    Attributes:
        fn: The wrapped function that takes and returns tensors
    """

    def __init__(self, fn: Callable[[Tensor], Tensor]) -> None:
        """Initialize Lambda wrapper.

        Args:
            fn: Function to wrap, must take a Tensor and return a Tensor
        """
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """Apply the wrapped function.

        Args:
            x: Input tensor

        Returns:
            Output of the wrapped function applied to input
        """
        return self.fn(x)
