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

from typing import Any

import torch
from torch import Tensor, nn

from .basis import BasisFunction
from .config import ContinuousHopfieldConfig
from .optimization import CCCPOptimizer, ContinuousHopfieldEnergy


class ContinuousMemory(nn.Module):
    """Continuous memory representation using basis functions.

    Represents discrete memory patterns X in R^(L x D) as a continuous
    function x_bar(t) = B^T psi(t), where B in R^(N x D) are coefficients
    and psi(t) in R^N are basis functions.

    The coefficients B are computed via ridge regression:
    B^T = X^T F^T (FF^T + lambda*I)^(-1)
    where F[i,j] = psi_i(t_j) is the design matrix.

    Attributes:
        basis: Basis function module
        regularization: Ridge regression regularization parameter lambda
        coefficients: Learned coefficients B (set by fit())
        is_fitted: Whether the memory has been fitted to data
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

        Note:
            The memory must be fitted with discrete patterns via fit()
            before it can be used for reconstruction.
        """
        super().__init__()
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")

        self.basis = basis
        self.regularization = regularization
        self.register_buffer("coefficients", None)
        self.is_fitted = False
        self.num_patterns = 0

    def fit(self, patterns: Tensor, positions: Tensor | None = None) -> None:
        """Fit continuous representation to discrete patterns.

        Computes coefficients B via ridge regression to minimize:
        ||X - B^T Psi||^2 + lambda*||B||^2

        Args:
            patterns: Memory patterns of shape (L, D) or (batch, L, D)
            positions: Time positions for each pattern, shape (L,)
                      If None, uses uniform spacing in [0, 1]

        Note:
            After fitting, the continuous memory approximates:
            x_bar(t_i) approx patterns[i] for each position t_i

        Raises:
            ValueError: If patterns shape is invalid
        """
        if patterns.dim() not in (2, 3):
            raise ValueError(f"patterns must be 2D or 3D, got {patterns.dim()}D")

        # Handle 3D tensors (batch of patterns)
        batch_dim = 3
        if patterns.dim() == batch_dim:
            batch_size, num_patterns, dim = patterns.shape
            patterns = patterns.reshape(batch_size * num_patterns, dim)
            if positions is None:
                positions = torch.linspace(
                    0, 1, num_patterns, device=patterns.device
                ).repeat(batch_size)
            else:
                if positions.dim() != 1 or positions.size(0) != num_patterns:
                    raise ValueError(
                        f"positions must be shape (L,), got {positions.shape}"
                    )
                positions = positions.repeat(batch_size)
        else:
            num_patterns, dim = patterns.shape
            if positions is None:
                positions = torch.linspace(0, 1, num_patterns, device=patterns.device)
            elif positions.dim() != 1 or positions.size(0) != num_patterns:
                raise ValueError(f"positions must be shape (L,), got {positions.shape}")

        self.num_patterns = patterns.size(0)

        design_matrix = self.basis.design_matrix(positions)  # (N, total_L)
        f_ft = design_matrix @ design_matrix.T  # (N, N)
        eye = torch.eye(
            self.basis.num_basis, device=design_matrix.device, dtype=design_matrix.dtype
        )
        inv = torch.linalg.inv(f_ft + self.regularization * eye)  # (N, N)
        self.coefficients = inv @ design_matrix @ patterns  # (N, D)
        self.is_fitted = True

    def reconstruct(self, t: Tensor) -> Tensor:
        """Reconstruct memory values at given time points.

        Computes x_bar(t) = B^T psi(t) for continuous memory representation.

        Args:
            t: Time points of shape (..., num_points) or scalar

        Returns:
            Reconstructed values of shape (..., num_points, dim)
            For scalar input, returns shape (dim,)

        Raises:
            RuntimeError: If memory has not been fitted

        Note:
            This function is differentiable and can be used in
            gradient-based optimization.
        """
        if not self.is_fitted:
            raise RuntimeError("Memory has not been fitted to data")

        psi = self.basis.evaluate(t)  # (N, num_points) or (N,)

        if t.dim() == 0:
            return self.coefficients.T @ psi  # (D, N) @ (N,) -> (D,)
        return (self.coefficients.T @ psi).T  # (D, num_points).T -> (num_points, D)

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
    """Continuous Hopfield network with CCCP optimization.

    Implements associative memory retrieval where memories are stored
    as a continuous function and queries are optimized to minimize
    the continuous energy function.

    The energy function is:
    E(q) = -1/beta log integral(exp(beta*x_bar(t)^T q) dt) + 0.5*||q||^2

    Attributes:
        config: Configuration object
        basis: Basis function module
        memory: Continuous memory representation
        optimizer: CCCP optimizer for energy minimization
        energy_fn: Continuous Hopfield energy function
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

        basis_type = self.config.basis_config.basis_type
        basis_kwargs: dict[str, Any] = {"domain": self.config.basis_config.domain}
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

        from .basis import create_basis

        self.basis = create_basis(
            basis_type, self.config.basis_config.num_basis, **basis_kwargs
        )

        self.memory = ContinuousMemory(self.basis, self.config.regularization)

        self.optimizer = CCCPOptimizer(
            max_iterations=self.config.cccp_config.max_iterations,
            tolerance=self.config.cccp_config.tolerance,
            step_size=self.config.cccp_config.step_size,
            momentum=self.config.cccp_config.momentum,
            track_trajectory=self.config.cccp_config.track_trajectory,
            use_line_search=self.config.cccp_config.use_line_search,
        )

        self.energy_fn = ContinuousHopfieldEnergy(
            beta=self.config.beta, integration_points=self.config.integration_points
        )

    def forward(
        self,
        memories: Tensor,
        queries: Tensor,
        positions: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict]:
        """Retrieve memories for given queries.

        Args:
            memories: Discrete memory patterns of shape (L, D)
            queries: Query vectors of shape (B, D) or (D,)
            positions: Optional time positions for memories
            return_info: Whether to return optimization info

        Returns:
            If return_info=False: Retrieved patterns of shape (B, D)
            If return_info=True: Tuple of (patterns, info_dict) where
                info_dict contains optimization trajectories and stats

        Note:
            The retrieval process:
            1. Fits continuous memory to discrete patterns
            2. Optimizes queries to minimize energy
            3. Returns converged query states
        """
        self.memory.fit(memories, positions)
        self.energy_fn.set_memory(self.memory.reconstruct)

        is_single = queries.dim() == 1
        if is_single:
            queries = queries.unsqueeze(0)

        results = []
        infos = []
        for query_vec in queries:
            result = self.optimizer.minimize(self.energy_fn, query_vec)
            results.append(result.optimal_point)
            infos.append(result)

        outputs = torch.stack(results)
        if is_single:
            outputs = outputs.squeeze(0)

        if return_info:
            return outputs, {"optimization_results": infos}
        return outputs

    def analytical_update(self, queries: Tensor) -> Tensor:
        """Compute analytical CCCP update (Proposition 1 from paper).

        The update rule is: q_{t+1} = E_p[x_bar(t)]
        where p(t) = exp(beta*x_bar(t)^T q_t) / Z is the Gibbs distribution.

        Args:
            queries: Query vectors of shape (B, D) or (D,)

        Returns:
            Updated queries of same shape

        Note:
            This gives the exact CCCP update in one step for the
            specific energy function used in continuous Hopfield networks.
        """
        if self.energy_fn.memory is None:
            if not self.memory.is_fitted:
                raise RuntimeError("Memory not fitted")
            self.energy_fn.set_memory(self.memory.reconstruct)
        return -self.energy_fn.grad_concave(queries)

    def energy(self, queries: Tensor) -> Tensor:
        """Compute energy for given queries.

        Args:
            queries: Query vectors of shape (B, D) or (D,)

        Returns:
            Energy values of shape (B,) or scalar

        Raises:
            RuntimeError: If memory has not been fitted
        """
        if self.energy_fn.memory is None:
            if not self.memory.is_fitted:
                raise RuntimeError("Memory not fitted")
            self.energy_fn.set_memory(self.memory.reconstruct)
        return self.energy_fn(queries)

    def iterate(
        self, queries: Tensor, num_iterations: int = 1, use_analytical: bool = True
    ) -> Tensor:
        """Perform specified number of Hopfield iterations.

        Args:
            queries: Initial query vectors
            num_iterations: Number of update steps
            use_analytical: Use analytical or CCCP update

        Returns:
            Updated query vectors

        Note:
            This is useful for matching the iterative dynamics
            of discrete Hopfield networks.
        """
        if self.energy_fn.memory is None:
            if not self.memory.is_fitted:
                raise RuntimeError("Memory not fitted")
            self.energy_fn.set_memory(self.memory.reconstruct)

        x_prev = None
        for _ in range(num_iterations):
            if use_analytical:
                queries = self.analytical_update(queries)
            else:
                queries = self.optimizer.step(self.energy_fn, queries, x_prev)
            x_prev = queries
        return queries


class ContinuousAttention(nn.Module):
    """Continuous attention mechanism compatible with transformers.

    Replaces discrete key-value attention with continuous memory:
    Instead of attending to discrete keys K, we attend to a continuous
    function k_bar(t) and compute expectations under the attention distribution.

    This implements the inf-memory transformer concept from
    Martins et al. (2022).

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        basis: Basis functions for continuous representation
        beta: Temperature parameter
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
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.integration_points = 500
        self.memory: ContinuousMemory | None = None

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
            key_positions: Time positions for keys
            attention_mask: Optional mask

        Returns:
            Attention output of shape (B, L_q, D)

        Note:
            The continuous attention:
            1. Fits continuous key/value functions
            2. Computes attention distribution over continuous domain
            3. Returns expected value under this distribution
        """
        batch_size, query_len, embed_dim = query.shape
        key_len = key.size(1)
        value_len = value.size(1)
        assert key_len == value_len, "Key and value must have same sequence length"

        query = (
            self.query_proj(query)
            .view(batch_size, query_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, H, Lq, hd)
        key = (
            self.key_proj(key)
            .view(batch_size, key_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, H, Lk, hd)
        value = (
            self.value_proj(value)
            .view(batch_size, value_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, H, Lv, hd)

        if key_positions is None:
            key_positions = torch.linspace(0, 1, key_len, device=query.device)

        outputs = []
        for b_idx in range(batch_size):
            head_outputs = []
            for h_idx in range(self.num_heads):
                memory_k = ContinuousMemory(self.basis, regularization=0.1)
                memory_k.fit(key[b_idx, h_idx], key_positions)
                memory_v = ContinuousMemory(self.basis, regularization=0.1)
                memory_v.fit(value[b_idx, h_idx], key_positions)

                t = torch.linspace(0, 1, self.integration_points, device=query.device)
                k_bar = memory_k.reconstruct(t)  # (num_points, hd)
                v_bar = memory_v.reconstruct(t)  # (num_points, hd)

                q_head = query[b_idx, h_idx]  # (Lq, hd)
                inner = self.beta * (q_head @ k_bar.T)  # (Lq, num_points)
                soft = torch.softmax(inner, dim=-1)  # (Lq, num_points)
                out_head = soft @ v_bar  # (Lq, hd)
                head_outputs.append(out_head)
            out_b = torch.cat(head_outputs, dim=-1)  # (Lq, D)
            outputs.append(out_b)

        outputs = torch.stack(outputs, dim=0)  # (B, Lq, D)
        return self.out_proj(outputs)

    def compute_attention_density(self, query: Tensor, t: Tensor) -> Tensor:
        """Compute attention probability density p(t|q).

        Args:
            query: Query vector of shape (D,)
            t: Time points of shape (num_points,)

        Returns:
            Probability density values of shape (num_points,)

        Note:
            p(t|q) is proportional to exp(beta * q^T k_bar(t))
        """
        if self.memory is None:
            raise RuntimeError("Memory not set for density computation")
        k_bar = self.memory.reconstruct(t)  # (num_points, D)
        inner = self.beta * (query @ k_bar.T)  # (num_points,)
        exp_inner = torch.exp(inner - inner.max())
        z = exp_inner.sum() * (t[1] - t[0])
        return exp_inner / z
