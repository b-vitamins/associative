"""Continuous Hopfield networks and memory representations.

This module implements continuous-time Hopfield networks that compress
discrete memories into continuous representations using basis functions,
as described in "Modern Hopfield Networks with Continuous-Time Memories"
(Santos et al., 2025).

Classes:
    ContinuousMemory: Memory representation using basis function compression
    ContinuousHopfield: Full Hopfield network with CCCP optimization
    ContinuousAttention: Continuous attention mechanism for transformers
"""

import torch
from torch import Tensor, nn

from .basis import BasisFunction
from .config import ContinuousHopfieldConfig

# Dimension constants for tensor validation
_DIM_1D = 1
_DIM_2D = 2
_DIM_3D = 3


class ContinuousHopfieldEnergy(nn.Module):
    """Energy function for continuous Hopfield networks.

    Implements the energy from Santos et al. (2025):
    E(q) = -1/β log ∫ exp(β·x̄(t)ᵀq) dt + 0.5||q||²

    where x̄(t) is the continuous memory representation.

    Args:
        beta: Inverse temperature parameter
        integration_points: Number of points for numerical integration
        device: Device for computations
        dtype: Data type for computations
    """

    def __init__(
        self,
        beta: float = 1.0,
        integration_points: int = 500,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize continuous Hopfield energy.

        Args:
            beta: Inverse temperature (higher = sharper attention)
            integration_points: Points for numerical integration
            device: Device for computations
            dtype: Data type for computations
        """
        super().__init__()
        self.beta = beta
        self.integration_points = integration_points
        self.device = device
        self.dtype = dtype

        # Initialize memory function properly (not None)
        self.memory_fn: ContinuousMemory | None = None

    def set_memory(self, memory: "ContinuousMemory"):
        """Set the continuous memory module.

        Args:
            memory: ContinuousMemory module with fitted coefficients
        """
        self.memory_fn = memory

    def forward(self, states: Tensor) -> Tensor:
        """Compute total energy E(states).

        Args:
            states: Input tensor of shape (B, Q, D)

        Returns:
            Energy values of shape (B, Q)

        Raises:
            RuntimeError: If memory has not been set
        """
        return self._concave_part(states) + self._convex_part(states)

    def _concave_part(self, states: Tensor) -> Tensor:
        """Compute concave energy: -1/β log ∫ exp(β·x̄(t)ᵀq) dt.

        Args:
            states: Query states of shape (B, Q, D)

        Returns:
            Concave energy of shape (B, Q)
        """
        if self.memory_fn is None:
            raise RuntimeError("Memory function not set")

        batch_size, num_queries, dim = states.shape
        device = states.device
        dtype = states.dtype

        # Create integration points
        time_points = torch.linspace(
            0, 1, self.integration_points, device=device, dtype=dtype
        )
        time_points_batch = time_points.unsqueeze(0).expand(batch_size, -1)  # (B, N)

        # Get memory values at integration points
        memory_values = self.memory_fn(time_points_batch)  # (B, N, D)

        # Compute inner products: (B, Q, D) @ (B, D, N) -> (B, Q, N)
        inner_products = torch.bmm(states, memory_values.transpose(1, 2))
        scaled_inner = self.beta * inner_products  # (B, Q, N)

        # Numerical integration with stability
        max_inner = scaled_inner.max(dim=2, keepdim=True)[0]  # (B, Q, 1)
        exp_relative = torch.exp(scaled_inner - max_inner)  # (B, Q, N)

        # Trapezoidal integration
        integral = torch.trapz(exp_relative, time_points, dim=2)  # (B, Q)
        log_integral = max_inner.squeeze(2) + torch.log(integral)  # (B, Q)

        return -log_integral / self.beta

    def _convex_part(self, states: Tensor) -> Tensor:
        """Compute convex energy: 0.5 * ||q||².

        Args:
            states: Query states of shape (B, Q, D)

        Returns:
            Convex energy of shape (B, Q)
        """
        return 0.5 * (states**2).sum(dim=2)  # (B, Q)

    def _grad_concave(self, states: Tensor) -> Tensor:
        """Gradient of concave part: -E_p[x̄(t)] where p(t) ∝ exp(β·x̄(t)ᵀq).

        This is the analytical CCCP update from Proposition 1.

        Args:
            states: Query vectors of shape (B, Q, D)

        Returns:
            Gradient of shape (B, Q, D)
        """
        if self.memory_fn is None:
            raise RuntimeError("Memory function not set")

        batch_size, num_queries, dim = states.shape
        device = states.device
        dtype = states.dtype

        # Create integration points
        time_points = torch.linspace(
            0, 1, self.integration_points, device=device, dtype=dtype
        )
        time_points_batch = time_points.unsqueeze(0).expand(batch_size, -1)  # (B, N)

        # Get memory values
        memory_values = self.memory_fn(time_points_batch)  # (B, N, D)

        # Compute attention weights
        inner_products = torch.bmm(states, memory_values.transpose(1, 2))  # (B, Q, N)
        scaled_inner = self.beta * inner_products

        # Stable softmax over integration points
        max_inner = scaled_inner.max(dim=2, keepdim=True)[0]
        exp_relative = torch.exp(scaled_inner - max_inner)

        # Trapezoidal weights for integration
        dt = 1.0 / (self.integration_points - 1) if self.integration_points > 1 else 1.0
        trap_weights = (
            torch.ones(self.integration_points, device=device, dtype=dtype) * dt
        )
        trap_weights[0] *= 0.5
        trap_weights[-1] *= 0.5

        # Apply trapezoidal weights
        exp_weighted = exp_relative * trap_weights.unsqueeze(0).unsqueeze(
            0
        )  # (B, Q, N)
        normalization = exp_weighted.sum(dim=2, keepdim=True)  # (B, Q, 1)
        attention_weights = exp_weighted / normalization  # (B, Q, N)

        # Compute expectation: (B, Q, N) @ (B, N, D) -> (B, Q, D)
        expectation = torch.bmm(attention_weights, memory_values)

        return -expectation


class ContinuousMemory(nn.Module):
    """Continuous memory representation using basis functions.

    Represents discrete memory patterns X in R^(B x L x D) as a continuous
    function x_bar(t) = B^T psi(t), where B in R^(B x N x D) are coefficients
    and psi(t) in R^N are basis functions.

    The coefficients B are computed via ridge regression:
    B^T = X^T F^T (FF^T + lambda*I)^(-1)
    where F[i,j] = psi_i(t_j) is the design matrix.

    Args:
        basis: Basis function module
        regularization: Ridge regression regularization parameter lambda
        device: Device for computations
        dtype: Data type for computations
    """

    def __init__(
        self,
        basis: BasisFunction,
        regularization: float = 0.5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize continuous memory.

        Args:
            basis: Basis function module to use
            regularization: Ridge regression regularization (lambda > 0)
            device: Device for computations
            dtype: Data type for computations

        Raises:
            ValueError: If regularization <= 0
        """
        super().__init__()
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")

        self.basis = basis
        self.regularization = regularization
        self.device = device
        self.dtype = dtype

        # Properly initialize coefficients buffer (not None)
        self.register_buffer(
            "coefficients",
            torch.zeros(1, basis.num_basis, 1, device=device, dtype=dtype),
        )
        self.is_fitted = False
        self.num_patterns = 0
        self.batch_size = 1

    def reset_parameters(self):
        """Reset module parameters to initial state."""
        self.coefficients = torch.zeros_like(self.coefficients)
        self.is_fitted = False
        self.num_patterns = 0
        self.batch_size = 1

    def fit(self, patterns: Tensor, positions: Tensor | None = None) -> None:
        """Fit continuous representation to discrete patterns.

        Computes coefficients B via ridge regression to minimize:
        ||X - B^T Psi||^2 + lambda*||B||^2

        Args:
            patterns: Memory patterns of shape (B, L, D)
            positions: Time positions for each pattern, shape (L,)
                      If None, uses uniform spacing in [0, 1]

        Raises:
            ValueError: If patterns shape is invalid (must be 3D)
        """
        if patterns.dim() != _DIM_3D:
            raise ValueError(f"Expected 3D tensor (B, L, D), got {patterns.dim()}D")

        batch_size, num_patterns, dim = patterns.shape
        self.batch_size = batch_size
        self.num_patterns = num_patterns

        device = patterns.device if self.device is None else self.device
        dtype = patterns.dtype if self.dtype is None else self.dtype

        # Move patterns to correct device/dtype
        patterns = patterns.to(device=device, dtype=dtype)

        if positions is None:
            # Use uniform positions t_i = i/L
            positions = (
                torch.arange(num_patterns, device=device, dtype=dtype) / num_patterns
            )
        elif positions.dim() != 1 or positions.size(0) != num_patterns:
            raise ValueError(f"positions must be shape (L,), got {positions.shape}")
        else:
            positions = positions.to(device=device, dtype=dtype)

        # Compute design matrix F: (N, L) where F[i,j] = psi_i(t_j)
        design_matrix = self.basis.design_matrix(positions)  # (N, L)

        # Compute coefficients for each batch item
        coefficients_list = []
        for b in range(batch_size):
            # Extract patterns for this batch
            patterns_b = patterns[b]  # (L, D)

            # Compute FF^T + λI for inversion
            gram_matrix = design_matrix @ design_matrix.T  # (N, N)
            identity = torch.eye(self.basis.num_basis, device=device, dtype=dtype)
            regularized_gram = gram_matrix + self.regularization * identity  # (N, N)

            # Compute: B = (FF^T + λI)^(-1) @ F @ X
            gram_inverse = torch.linalg.inv(regularized_gram)  # (N, N)
            coefficients_b = gram_inverse @ design_matrix @ patterns_b  # (N, D)
            coefficients_list.append(coefficients_b)

        # Stack coefficients for all batch items
        self.coefficients = torch.stack(coefficients_list, dim=0)  # (B, N, D)
        self.is_fitted = True

    def forward(self, time_points: Tensor) -> Tensor:
        """Reconstruct memory values at given time points.

        Computes x_bar(t) = B^T psi(t) for continuous memory representation.

        Args:
            time_points: Time points of shape (B, T) or (T,)
                        If (T,), will be expanded to match batch size

        Returns:
            Reconstructed values of shape (B, T, D)

        Raises:
            RuntimeError: If memory has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Memory has not been fitted to data")

        # Handle different input shapes
        if time_points.dim() == _DIM_1D:
            # Expand to batch dimension
            time_points = time_points.unsqueeze(0).expand(self.batch_size, -1)
        elif time_points.dim() != _DIM_2D:
            raise ValueError(f"time_points must be 1D or 2D, got {time_points.dim()}D")

        batch_size, num_points = time_points.shape

        # Check batch size consistency
        if batch_size != self.batch_size:
            if batch_size == 1:
                # Broadcast single time points to all batches
                time_points = time_points.expand(self.batch_size, -1)
                batch_size = self.batch_size
            else:
                raise ValueError(
                    f"Batch size mismatch: got {batch_size}, expected {self.batch_size}"
                )

        # Evaluate basis functions at time points for each batch
        # We need to handle this batch-wise since basis.evaluate expects 1D input
        basis_values_list = []
        for b in range(batch_size):
            psi = self.basis.evaluate(time_points[b])  # (N, T)
            basis_values_list.append(psi)

        basis_values = torch.stack(basis_values_list, dim=0)  # (B, N, T)

        # Compute reconstruction: (B, N, D).transpose @ (B, N, T) -> (B, D, T)
        reconstructed = torch.bmm(
            self.coefficients.transpose(1, 2),  # (B, D, N)
            basis_values,  # (B, N, T)
        )  # (B, D, T)

        # Return in standard shape (B, T, D)
        return reconstructed.transpose(1, 2)

    def compress_ratio(self) -> float:
        """Compute memory compression ratio N/L.

        Returns:
            Ratio of basis functions to original patterns

        Raises:
            RuntimeError: If memory has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Memory has not been fitted to data")
        return self.basis.num_basis / self.num_patterns


class ContinuousHopfield(nn.Module):
    """Continuous Hopfield network with iterative CCCP updates.

    Implements associative memory retrieval where memories are stored
    as a continuous function and queries are updated iteratively using
    the analytical CCCP update rule from Proposition 1.

    The energy function is:
    E(q) = -1/beta log integral(exp(beta*x_bar(t)^T q) dt) + 0.5*||q||^2

    Args:
        config: Configuration object
        device: Device for computations
        dtype: Data type for computations
    """

    def __init__(
        self,
        config: ContinuousHopfieldConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize continuous Hopfield network.

        Args:
            config: Configuration for the network
            device: Device for computations
            dtype: Data type for computations
        """
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype

        # Create basis functions
        from typing import Any

        from .basis import create_basis

        basis_kwargs: dict[str, Any] = {"domain": self.config.basis_config.domain}
        basis_type = self.config.basis_config.basis_type

        if basis_type == "rectangular":
            basis_kwargs["overlap"] = self.config.basis_config.overlap
        elif basis_type == "gaussian":
            if self.config.basis_config.learnable:
                basis_kwargs["learnable_widths"] = self.config.basis_config.learnable
            if self.config.basis_config.init_width is not None:
                basis_kwargs["init_width"] = self.config.basis_config.init_width
        elif (
            basis_type == "fourier"
            and self.config.basis_config.max_frequency is not None
        ):
            basis_kwargs["max_frequency"] = self.config.basis_config.max_frequency

        self.basis = create_basis(
            basis_type, self.config.basis_config.num_basis, **basis_kwargs
        )

        # Create memory and energy modules
        self.memory = ContinuousMemory(
            self.basis, self.config.regularization, device=device, dtype=dtype
        )

        self.energy_fn = ContinuousHopfieldEnergy(
            beta=self.config.beta,
            integration_points=self.config.integration_points,
            device=device,
            dtype=dtype,
        )

    def reset_parameters(self):
        """Reset all parameters to initial state."""
        self.memory.reset_parameters()
        # Basis functions may have learnable parameters
        if hasattr(self.basis, "reset_parameters"):
            reset_fn = self.basis.reset_parameters
            if callable(reset_fn):
                reset_fn()

    def forward(
        self,
        memories: Tensor,
        queries: Tensor,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Retrieve memories for given queries using iterative updates.

        Args:
            memories: Discrete memory patterns of shape (B, L, D)
            queries: Query vectors of shape (B, Q, D)
            positions: Optional time positions for memories, shape (L,)

        Returns:
            Tuple of (retrieved_patterns, info_dict) where:
                retrieved_patterns: Shape (B, Q, D)
                info_dict: Contains 'num_iterations' and other metadata

        Raises:
            ValueError: If batch dimensions don't match
        """
        # Validate inputs
        if memories.dim() != _DIM_3D:
            raise ValueError(f"memories must be 3D (B, L, D), got {memories.dim()}D")
        if queries.dim() != _DIM_3D:
            raise ValueError(f"queries must be 3D (B, Q, D), got {queries.dim()}D")

        batch_size_mem, num_patterns, dim_mem = memories.shape
        batch_size_q, num_queries, dim_q = queries.shape

        if batch_size_mem != batch_size_q:
            raise ValueError(
                f"Batch size mismatch: memories {batch_size_mem}, queries {batch_size_q}"
            )
        if dim_mem != dim_q:
            raise ValueError(f"Dimension mismatch: memories {dim_mem}, queries {dim_q}")

        # Fit memory to patterns
        self.memory.fit(memories, positions)
        self.energy_fn.set_memory(self.memory)

        # Perform iterative updates
        outputs = self._iterate(queries, num_iterations=self.config.num_iterations)

        # Return tuple for consistency
        info = {
            "num_iterations": self.config.num_iterations,
            "compression_ratio": self.memory.compress_ratio(),
        }

        return outputs, info

    def _analytical_update(self, queries: Tensor) -> Tensor:
        """Compute analytical CCCP update (Proposition 1 from paper).

        The update rule is: q_{t+1} = E_p[x_bar(t)]
        where p(t) = exp(beta*x_bar(t)^T q_t) / Z is the Gibbs distribution.

        Args:
            queries: Query vectors of shape (B, Q, D)

        Returns:
            Updated queries of shape (B, Q, D)
        """
        if self.energy_fn.memory_fn is None:
            raise RuntimeError("Memory not set in energy function")
        return -self.energy_fn._grad_concave(queries)

    def energy(self, queries: Tensor) -> Tensor:
        """Compute energy for given queries.

        Args:
            queries: Query vectors of shape (B, Q, D)

        Returns:
            Energy values of shape (B, Q)

        Raises:
            RuntimeError: If memory has not been fitted
        """
        if self.energy_fn.memory_fn is None:
            if not self.memory.is_fitted:
                raise RuntimeError("Memory not fitted")
            self.energy_fn.set_memory(self.memory)
        return self.energy_fn(queries)

    def _iterate(self, queries: Tensor, num_iterations: int = 1) -> Tensor:
        """Perform specified number of Hopfield iterations.

        Args:
            queries: Initial query vectors of shape (B, Q, D)
            num_iterations: Number of update steps

        Returns:
            Updated query vectors of shape (B, Q, D)
        """
        if self.energy_fn.memory_fn is None:
            if not self.memory.is_fitted:
                raise RuntimeError("Memory not fitted")
            self.energy_fn.set_memory(self.memory)

        current = queries
        for _ in range(num_iterations):
            current = self._analytical_update(current)
        return current


class ContinuousAttention(nn.Module):
    """Continuous attention mechanism compatible with transformers.

    Replaces discrete key-value attention with continuous memory:
    Instead of attending to discrete keys K, we attend to a continuous
    function k_bar(t) and compute expectations under the attention distribution.

    This implements the inf-memory transformer concept from
    Martins et al. (2022).

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        basis: Basis functions for continuous representation
        beta: Temperature parameter
        bias: Whether to use bias in projections
        device: Device for computations
        dtype: Data type for computations
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        basis: BasisFunction,
        beta: float = 1.0,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize continuous attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            basis: Basis functions for continuous keys/values
            beta: Temperature for attention
            bias: Whether to use bias in projections
            device: Device for computations
            dtype: Data type for computations
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.basis = basis
        self.beta = beta
        self.device = device
        self.dtype = dtype

        # Projection layers
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Move to device/dtype if specified
        if device is not None or dtype is not None:
            self.query_proj = self.query_proj.to(device=device, dtype=dtype)
            self.key_proj = self.key_proj.to(device=device, dtype=dtype)
            self.value_proj = self.value_proj.to(device=device, dtype=dtype)
            self.out_proj = self.out_proj.to(device=device, dtype=dtype)

        self.integration_points = 500

    def reset_parameters(self):
        """Reset module parameters."""
        self.query_proj.reset_parameters()
        self.key_proj.reset_parameters()
        self.value_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_positions: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute continuous attention.

        Args:
            query: Query tensor of shape (B, L_q, D)
            key: Key tensor of shape (B, L_k, D)
            value: Value tensor of shape (B, L_v, D)
            key_positions: Time positions for keys, shape (L_k,)
            attention_mask: Optional mask (currently unused)

        Returns:
            Attention output of shape (B, L_q, D)
        """
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)
        value_len = value.size(1)

        if key_len != value_len:
            raise ValueError("Key and value must have same sequence length")

        # Ensure inputs match module dtype/device
        module_dtype = next(self.parameters()).dtype
        module_device = next(self.parameters()).device

        query = query.to(dtype=module_dtype, device=module_device)
        key = key.to(dtype=module_dtype, device=module_device)
        value = value.to(dtype=module_dtype, device=module_device)

        # Project inputs
        query = self.query_proj(query)  # (B, L_q, D)
        key = self.key_proj(key)  # (B, L_k, D)
        value = self.value_proj(value)  # (B, L_v, D)

        # Reshape for multi-head attention
        query = query.view(batch_size, query_len, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)  # (B, H, L_q, head_dim)

        key = key.view(batch_size, key_len, self.num_heads, self.head_dim)
        key = key.transpose(1, 2)  # (B, H, L_k, head_dim)

        value = value.view(batch_size, value_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)  # (B, H, L_v, head_dim)

        # Default positions if not provided
        if key_positions is None:
            key_positions = torch.linspace(
                0, 1, key_len, device=query.device, dtype=query.dtype
            )

        # Fit continuous memories for each head (vectorized)
        # Reshape for batch processing: (B*H, L_k, head_dim)
        key_flat = key.reshape(batch_size * self.num_heads, key_len, self.head_dim)
        value_flat = value.reshape(
            batch_size * self.num_heads, value_len, self.head_dim
        )

        # Create continuous memories
        memory_key = ContinuousMemory(
            self.basis, regularization=0.5, device=query.device, dtype=query.dtype
        )
        memory_value = ContinuousMemory(
            self.basis, regularization=0.5, device=query.device, dtype=query.dtype
        )

        # Fit memories
        memory_key.fit(key_flat, key_positions)
        memory_value.fit(value_flat, key_positions)

        # Create integration points
        time_points = torch.linspace(
            0, 1, self.integration_points, device=query.device, dtype=query.dtype
        )
        time_points_batch = time_points.unsqueeze(0).expand(
            batch_size * self.num_heads, -1
        )

        # Reconstruct continuous keys and values
        key_continuous = memory_key(time_points_batch)  # (B*H, T, head_dim)
        value_continuous = memory_value(time_points_batch)  # (B*H, T, head_dim)

        # Reshape query for batch processing
        query_flat = query.reshape(
            batch_size * self.num_heads, query_len, self.head_dim
        )

        # Compute attention scores
        scores = torch.bmm(query_flat, key_continuous.transpose(1, 2))  # (B*H, L_q, T)
        scores = scores * self.beta

        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)  # (B*H, L_q, T)

        # Apply attention to values
        output = torch.bmm(attention_weights, value_continuous)  # (B*H, L_q, head_dim)

        # Reshape back to separate batch and heads
        output = output.view(batch_size, self.num_heads, query_len, self.head_dim)
        output = output.transpose(1, 2)  # (B, L_q, H, head_dim)
        output = output.reshape(batch_size, query_len, self.embed_dim)

        # Final projection
        return self.out_proj(output)
