"""Hopfield network layers for associative memory models."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from .config import HopfieldConfig
from .utils import Lambda


class Hopfield(nn.Module):
    """Hopfield network layer with energy function.

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (computed from ratio if None)
        config: Hopfield configuration
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        config: HopfieldConfig | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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
        """Compute Hopfield energy."""
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
