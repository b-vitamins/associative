"""Energy optimization framework using Concave-Convex Procedure (CCCP).

This module implements the CCCP algorithm for minimizing energy functions
that can be decomposed into concave and convex parts, as described in
Yuille & Rangarajan (2003) and applied in Santos et al. (2025).

The CCCP algorithm iteratively minimizes E(x) = E_concave(x) + E_convex(x)
by linearizing the concave part at each step, resulting in guaranteed convergence
to local minima for continuous Hopfield networks and other energy-based models.

Classes:
    OptimizationResult: Dataclass containing optimization results and trajectory
    EnergyFunction: Abstract base class for CCCP-compatible energy functions
    CCCPOptimizer: Main optimizer implementing the CCCP algorithm
    ContinuousHopfieldEnergy: Specific energy function for continuous Hopfield networks
    QuadraticConvexSolver: Solver for quadratic programming subproblems
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class OptimizationResult:
    """Result of an optimization procedure.

    Attributes:
        optimal_point: The optimized tensor
        final_energy: Final energy value
        num_iterations: Number of iterations performed
        converged: Whether the optimization converged
        trajectory: Optional list of intermediate points
        energy_history: Optional list of energy values at each iteration
        gradient_norms: Optional list of gradient norms at each iteration
    """

    optimal_point: Tensor
    final_energy: float
    num_iterations: int
    converged: bool
    trajectory: list[Tensor] | None = None
    energy_history: list[float] | None = None
    gradient_norms: list[float] | None = None


class EnergyFunction(nn.Module, ABC):
    """Abstract base class for energy functions E(state) = E_concave(state) + E_convex(state).

    The CCCP algorithm requires the energy to be decomposed into a concave
    and a convex part. This decomposition is not unique, but different
    decompositions lead to different convergence properties.

    Note:
        In the Hopfield context, we have:
        - E_concave(q) = -1/β log ∫ exp(β·x̄(t)ᵀq) dt
        - E_convex(q) = 0.5 * ||q||²
    """

    @abstractmethod
    def concave_part(self, state: Tensor) -> Tensor:
        """Compute the concave part of the energy.

        Args:
            state: Input tensor of shape (..., dim)

        Returns:
            Scalar or tensor of shape (...,) containing concave energy values

        Note:
            This function should be concave (negative semidefinite Hessian).
            In practice, we often have -log(sum(exp(...))) terms here.
        """
        pass

    @abstractmethod
    def convex_part(self, state: Tensor) -> Tensor:
        """Compute the convex part of the energy.

        Args:
            state: Input tensor of shape (..., dim)

        Returns:
            Scalar or tensor of shape (...,) containing convex energy values

        Note:
            This function should be convex (positive semidefinite Hessian).
            Common examples: quadratic terms, norms, linear functions.
        """
        pass

    def forward(self, state: Tensor) -> Tensor:
        """Compute total energy E(state) = E_concave(state) + E_convex(state).

        Args:
            state: Input tensor

        Returns:
            Total energy value(s)
        """
        return self.concave_part(state) + self.convex_part(state)

    @abstractmethod
    def grad_concave(self, state: Tensor) -> Tensor:
        """Compute gradient of the concave part.

        Args:
            state: Input tensor of shape (..., dim)

        Returns:
            Gradient tensor of same shape as input

        Note:
            This can be computed via autograd, but explicit implementation
            can be more efficient and numerically stable.
        """
        pass

    @abstractmethod
    def grad_convex(self, state: Tensor) -> Tensor:
        """Compute gradient of the convex part.

        Args:
            state: Input tensor of shape (..., dim)

        Returns:
            Gradient tensor of same shape as input

        Note:
            For quadratic convex parts, this is often just the state itself.
        """
        pass


class CCCPOptimizer(nn.Module):
    """Concave-Convex Procedure optimizer for energy minimization.

    The CCCP algorithm iteratively minimizes E(state) = E_concave(state) + E_convex(state) by:
    1. Linearizing the concave part at the current point
    2. Minimizing the resulting convex problem

    This leads to the update rule:
    ∇E_convex(state_{t+1}) = -∇E_concave(state_t)

    Attributes:
        max_iterations: Maximum number of CCCP iterations
        tolerance: Convergence tolerance for ||state_{t+1} - state_t||
        step_size: Step size for damped updates (0 < a ≤ 1)
        momentum: Momentum coefficient for accelerated CCCP
        track_trajectory: Whether to store intermediate points
        use_line_search: Whether to use backtracking line search
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        step_size: float = 1.0,
        momentum: float = 0.0,
        track_trajectory: bool = False,
        use_line_search: bool = False,
    ):
        """Initialize CCCP optimizer.

        Args:
            max_iterations: Maximum iterations before stopping
            tolerance: Convergence tolerance for point difference
            step_size: Damping factor for updates (1.0 = no damping)
            momentum: Momentum coefficient (0 = no momentum)
            track_trajectory: Whether to record optimization path
            use_line_search: Whether to adaptively choose step size

        Raises:
            ValueError: If parameters are out of valid ranges
        """
        super().__init__()
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        if not 0 < step_size <= 1:
            raise ValueError(f"step_size must be in (0, 1], got {step_size}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_size = step_size
        self.momentum = momentum
        self.track_trajectory = track_trajectory
        self.use_line_search = use_line_search

    def step(
        self,
        energy_fn: EnergyFunction,
        current_state: Tensor,
        prev_state: Tensor | None = None,
    ) -> Tensor:
        """Perform one CCCP iteration.

        The update solves: ∇E_convex(new_state) = -∇E_concave(current_state)

        Args:
            energy_fn: Energy function to minimize
            current_state: Current point
            prev_state: Previous point (for momentum)

        Returns:
            Updated point state_{t+1}

        Note:
            The exact solution depends on the form of E_convex.
            For E_convex(state) = 0.5||state||², we get new_state = -∇E_concave(current_state).
        """
        grad_concave = energy_fn.grad_concave(current_state)
        proposed_state = -grad_concave
        if self.momentum > 0 and prev_state is not None:
            velocity = current_state - prev_state
            proposed_state += self.momentum * velocity
        direction = proposed_state - current_state
        if self.use_line_search:
            alpha = self._line_search(energy_fn, current_state, direction)
        else:
            alpha = self.step_size
        return current_state + alpha * direction

    def minimize(
        self, energy_fn: EnergyFunction, initial_state: Tensor
    ) -> OptimizationResult:
        """Minimize energy function using CCCP.

        Args:
            energy_fn: Energy function to minimize
            initial_state: Initial point

        Returns:
            OptimizationResult containing solution and convergence info

        Note:
            The algorithm stops when either:
            - ||state_{t+1} - state_t|| < tolerance (converged)
            - num_iterations >= max_iterations (not converged)
        """
        current_state = initial_state.clone()
        trajectory: list[Tensor] | None = (
            [current_state.clone()] if self.track_trajectory else None
        )
        energy_history = [energy_fn(current_state).mean().item()]
        prev_state = None
        converged = False
        num_iterations = 0
        for i in range(self.max_iterations):
            next_state = self.step(energy_fn, current_state, prev_state)
            # Handle batched tensors - compute norm along last dimension and take mean
            diff = torch.norm(next_state - current_state, dim=-1).mean()
            num_iterations = i + 1
            if self.track_trajectory and trajectory is not None:
                trajectory.append(next_state.clone())
            energy_history.append(energy_fn(next_state).mean().item())
            if diff < self.tolerance:
                converged = True
                break
            prev_state = current_state
            current_state = next_state
        return OptimizationResult(
            optimal_point=current_state,
            final_energy=energy_history[-1],
            num_iterations=num_iterations,
            converged=converged,
            trajectory=trajectory,
            energy_history=energy_history,
            gradient_norms=None,
        )

    def _line_search(
        self,
        energy_fn: EnergyFunction,
        current_state: Tensor,
        direction: Tensor,
        max_backtracks: int = 10,
    ) -> float:
        """Backtracking line search for step size selection.

        Finds alpha such that E(state + alpha*direction) < E(state) - c*alpha*||∇E(state)||²

        Args:
            energy_fn: Energy function
            current_state: Current point
            direction: Search direction
            max_backtracks: Maximum backtracking iterations

        Returns:
            Optimal step size alpha
        """
        c = 1e-4
        grad = energy_fn.grad_concave(current_state) + energy_fn.grad_convex(
            current_state
        )
        # Handle batched tensors - compute dot product along last dimension
        dot = (grad * direction).sum(dim=-1).mean()
        alpha = 1.0
        current_energy = energy_fn(current_state).mean()
        for _ in range(max_backtracks):
            trial_state = current_state + alpha * direction
            trial_energy = energy_fn(trial_state).mean()
            if trial_energy <= current_energy + c * alpha * dot:
                return alpha
            alpha *= 0.5
        return alpha


class QuadraticConvexSolver:
    """Solver for CCCP subproblems with quadratic convex parts.

    When E_convex(state) = 0.5 * state^T Q state + b^T state, the CCCP update becomes:
    Q state_{t+1} + b = -∇E_concave(state_t)

    This class efficiently solves such systems.
    """

    @staticmethod
    def solve(
        q_matrix: Tensor | None, b_vector: Tensor | None, grad_concave: Tensor
    ) -> Tensor:
        """Solve the CCCP subproblem for quadratic convex part.

        Args:
            q_matrix: Positive definite matrix (None for identity)
            b_vector: Linear term (None for zero)
            grad_concave: Gradient of concave part at current point

        Returns:
            Solution state_{t+1}

        Note:
            For q_matrix = I and b_vector = 0 (common case), this reduces to
            state_{t+1} = -grad_concave
        """
        rhs = -grad_concave
        if b_vector is not None:
            rhs -= b_vector
        if q_matrix is None:
            return rhs
        if grad_concave.dim() == 1:
            return torch.linalg.solve(q_matrix, rhs)
        return torch.linalg.solve(q_matrix, rhs.T).T


class ContinuousHopfieldEnergy(EnergyFunction):
    """Energy function for continuous Hopfield networks.

    Implements the energy from Santos et al. (2025):
    E(q) = -1/β log ∫ exp(β·x̄(t)ᵀq) dt + 0.5||q||²

    where x̄(t) is the continuous memory representation.

    Attributes:
        beta: Inverse temperature parameter
        integration_points: Number of points for numerical integration
    """

    def __init__(self, beta: float = 1.0, integration_points: int = 500):
        """Initialize continuous Hopfield energy.

        Args:
            beta: Inverse temperature (higher = sharper attention)
            integration_points: Points for numerical integration

        Note:
            The continuous memory must be set via set_memory()
            before computing energies.
        """
        super().__init__()
        self.beta = beta
        self.integration_points = integration_points
        self.memory = None

    def set_memory(self, memory_fn: Callable[[Tensor], Tensor]) -> None:
        """Set the continuous memory function x̄(t).

        Args:
            memory_fn: Function that takes time points and returns memory values
                      Should accept Tensor of shape (num_points,) and return
                      Tensor of shape (num_points, dim)
        """
        self.memory = memory_fn

    def concave_part(self, state: Tensor) -> Tensor:
        """Compute concave energy: -1/β log ∫ exp(β·x̄(t)ᵀq) dt.

        Uses numerical integration with log-sum-exp for stability.
        """
        if self.memory is None:
            raise ValueError("Memory function not set")
        device = state.device
        t = torch.linspace(0, 1, self.integration_points, device=device)
        dt = t[1] - t[0]
        x_bar = self.memory(t)  # (N, d)
        batch_shape = state.shape[:-1]
        dim = state.shape[-1]
        q_flat = state.reshape(-1, dim)  # (B, d)
        q_flat.shape[0]
        inner = self.beta * torch.matmul(x_bar, q_flat.T)  # (N, B)
        max_inner = inner.max(dim=0)[0]  # (B,)
        exp_rel = torch.exp(inner - max_inner.unsqueeze(0))  # (N, B)
        sum_exp = exp_rel.sum(dim=0)  # (B,)
        log_integral = max_inner + torch.log(sum_exp * dt)  # (B,)
        concave = -log_integral / self.beta
        return concave.reshape(batch_shape)

    def convex_part(self, state: Tensor) -> Tensor:
        """Compute convex energy: 0.5 * ||q||²."""
        return 0.5 * (state**2).sum(dim=-1)

    def grad_concave(self, state: Tensor) -> Tensor:
        """Gradient of concave part: -E_p[x̄(t)] where p(t) ∝ exp(β·x̄(t)ᵀq)."""
        if self.memory is None:
            raise ValueError("Memory function not set")
        device = state.device
        t = torch.linspace(0, 1, self.integration_points, device=device)
        x_bar = self.memory(t)  # (N, d)
        batch_shape = state.shape[:-1]
        dim = state.shape[-1]
        q_flat = state.reshape(-1, dim)  # (B, d)
        q_flat.shape[0]
        inner = self.beta * torch.matmul(x_bar, q_flat.T)  # (N, B)
        max_inner = inner.max(dim=0)[0]  # (B,)
        exp_rel = torch.exp(inner - max_inner.unsqueeze(0))  # (N, B)
        sum_exp = exp_rel.sum(dim=0)  # (B,)
        p = exp_rel / sum_exp.unsqueeze(0)  # (N, B)
        expect = torch.matmul(p.T, x_bar)  # (B, d)
        return -expect.reshape((*batch_shape, dim))

    def grad_convex(self, state: Tensor) -> Tensor:
        """Gradient of convex part: q."""
        return state
