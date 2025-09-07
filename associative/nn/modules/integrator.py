"""Numerical integration utilities for continuous representations.

This module provides reusable numerical integration methods for computing
integrals over continuous functions, particularly for partition functions
and expectations in energy-based models. All integrators support batched
operations and are optimized for PyTorch tensors with GPU acceleration.

Classes:
    Integrator: Abstract base class for all integration methods
    TrapezoidalIntegrator: Trapezoidal rule integration (O(1/n²) accuracy)
    SimpsonIntegrator: Simpson's rule integration (O(1/n⁴) accuracy)
    GaussLegendreIntegrator: Gauss-Legendre quadrature (exponential accuracy for smooth functions)
    MonteCarloIntegrator: Monte Carlo integration with importance sampling
    AdaptiveIntegrator: Adaptive refinement with automatic error control

Functions:
    create_integrator: Factory function to create integrators by name
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import Tensor


class Integrator(ABC):
    """Abstract base class for numerical integration methods.

    Integrators compute definite integrals of functions over specified domains
    using various numerical approximation techniques.

    Attributes:
        domain: Tuple of (start, end) defining integration bounds
        num_points: Number of evaluation points for numerical approximation

    Performance Requirements:
        - MUST support batched operations without loops (vectorized)
        - MUST cache quadrature points/weights to avoid recomputation
        - MUST use torch.jit.script compatible operations for JIT compilation
        - SHOULD use fp16/bfloat16 when accuracy permits (>1e-4 tolerance)

    Memory Requirements:
        - Quadrature points: O(num_points) per integrator instance
        - Function evaluations: O(batch_size * num_points) temporary
        - MUST NOT materialize full batch x points matrix if avoidable

    Hardware Optimization:
        - CUDA: Use torch.cuda.amp for mixed precision
        - CPU: Leverage MKL/BLAS for matrix operations
        - TPU: Ensure XLA compatibility (no dynamic shapes)
    """

    def __init__(self, domain: tuple[float, float] = (0.0, 1.0), num_points: int = 50):
        """Initialize integrator.

        Args:
            domain: Integration domain as (start, end)
            num_points: Number of points for numerical approximation

        Raises:
            ValueError: If domain is invalid or num_points <= 0
        """
        if domain[0] >= domain[1]:
            raise ValueError(f"Invalid domain {domain}, start must be < end")
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")

        self.domain = domain
        self.num_points = num_points

    @abstractmethod
    def integrate(self, func: Callable[[Tensor], Tensor]) -> Tensor:
        """Compute the definite integral of a function.

        Args:
            func: Function to integrate, mapping points to values
                  Should handle batched inputs of shape (..., num_points)

        Returns:
            Integral value(s) as tensor, preserving batch dimensions

        Note:
            The function should be vectorized for efficiency.
            Integration is performed over the last dimension.
        """
        pass

    @abstractmethod
    def get_quadrature_points(self) -> tuple[Tensor, Tensor]:
        """Get quadrature points and weights for integration.

        Returns:
            Tuple of (points, weights) where:
            - points: Tensor of shape (num_points,) with evaluation points
            - weights: Tensor of shape (num_points,) with integration weights

        Note:
            This is useful for caching quadrature points when integrating
            multiple functions over the same domain.
        """
        pass


class TrapezoidalIntegrator(Integrator):
    """Trapezoidal rule integration.

    Approximates the integral using piecewise linear interpolation:
    ∫f(x)dx ≈ Σ (f(x_i) + f(x_{i+1})) * Δx / 2

    Attributes:
        points: Cached quadrature points
        weights: Cached integration weights
    """

    def __init__(self, domain: tuple[float, float] = (0.0, 1.0), num_points: int = 50):
        """Initialize trapezoidal integrator.

        Args:
            domain: Integration domain
            num_points: Number of equally-spaced evaluation points

        Note:
            Error scales as O(1/n²) for smooth functions.
            Exact for linear functions.
        """
        super().__init__(domain, num_points)
        self._cache_quadrature()

    def _cache_quadrature(self) -> None:
        """Precompute and cache quadrature points and weights."""
        # Create equally spaced points
        self.points = torch.linspace(
            self.domain[0], self.domain[1], self.num_points, dtype=torch.float32
        )

        # Trapezoidal rule weights: 1, 1, 1, ..., 1 except endpoints get 0.5
        self.weights = torch.ones(self.num_points, dtype=torch.float32)
        self.weights[0] = 0.5
        self.weights[-1] = 0.5

        # Scale by step size
        h = (self.domain[1] - self.domain[0]) / (self.num_points - 1)
        self.weights *= h

    def integrate(self, func: Callable[[Tensor], Tensor]) -> Tensor:
        """Compute integral using trapezoidal rule.

        Args:
            func: Function to integrate

        Returns:
            Integral approximation

        Example:
            >>> integrator = TrapezoidalIntegrator((0, 1), 100)
            >>> result = integrator.integrate(lambda x: x**2)  # ∫x²dx from 0 to 1
        """
        # Evaluate function at quadrature points
        func_values = func(self.points)

        # Compute weighted sum
        return torch.sum(func_values * self.weights)

    def get_quadrature_points(self) -> tuple[Tensor, Tensor]:
        """Return cached quadrature points and weights."""
        return self.points, self.weights


class SimpsonIntegrator(Integrator):
    """Simpson's rule integration.

    Approximates the integral using piecewise quadratic interpolation:
    ∫f(x)dx ≈ Δx/3 * Σ (f(x_{2i}) + 4*f(x_{2i+1}) + f(x_{2i+2}))

    Attributes:
        points: Cached quadrature points
        weights: Cached integration weights
    """

    def __init__(self, domain: tuple[float, float] = (0.0, 1.0), num_points: int = 51):
        """Initialize Simpson integrator.

        Args:
            domain: Integration domain
            num_points: Number of evaluation points (must be odd)

        Raises:
            ValueError: If num_points is even

        Note:
            Error scales as O(1/n⁴) for smooth functions.
            Exact for polynomials up to degree 3.
        """
        if num_points % 2 == 0:
            raise ValueError(
                f"Simpson's rule requires odd num_points, got {num_points}"
            )
        super().__init__(domain, num_points)
        self._cache_quadrature()

    def _cache_quadrature(self) -> None:
        """Precompute and cache quadrature points and weights."""
        # Create equally spaced points
        self.points = torch.linspace(
            self.domain[0], self.domain[1], self.num_points, dtype=torch.float32
        )

        # Simpson's rule weights: 1, 4, 2, 4, 2, ..., 4, 1
        self.weights = torch.ones(self.num_points, dtype=torch.float32)

        # Set the pattern: endpoints = 1, odd indices = 4, even (middle) = 2
        for i in range(1, self.num_points - 1):
            if i % 2 == 1:  # Odd indices (1, 3, 5, ...)
                self.weights[i] = 4.0
            else:  # Even indices (2, 4, 6, ...)
                self.weights[i] = 2.0

        # Scale by h/3 where h is the step size
        h = (self.domain[1] - self.domain[0]) / (self.num_points - 1)
        self.weights *= h / 3.0

    def integrate(self, func: Callable[[Tensor], Tensor]) -> Tensor:
        """Compute integral using Simpson's rule.

        Args:
            func: Function to integrate

        Returns:
            Integral approximation

        Example:
            >>> integrator = SimpsonIntegrator((0, 1), 101)
            >>> result = integrator.integrate(lambda x: torch.exp(x))
        """
        # Evaluate function at quadrature points
        func_values = func(self.points)

        # Compute weighted sum
        return torch.sum(func_values * self.weights)

    def get_quadrature_points(self) -> tuple[Tensor, Tensor]:
        """Return cached quadrature points and weights."""
        return self.points, self.weights


class GaussLegendreIntegrator(Integrator):
    """Gauss-Legendre quadrature integration.

    Uses optimally-placed quadrature points based on Legendre polynomials
    to achieve high accuracy with fewer function evaluations.

    Attributes:
        points: Cached Gauss-Legendre nodes mapped to domain
        weights: Cached Gauss-Legendre weights scaled for domain

    Performance Requirements:
        - MUST precompute nodes/weights using torch.linalg.eigh (O(n³))
        - MUST cache in GPU memory if available (register_buffer)
        - MUST use matrix-vector products for weighted sums
        - Optimal for smooth functions: 50 points sufficient for 1e-8 accuracy

    Memory Requirements:
        - Static: 2*num_points floats for nodes+weights
        - Runtime: O(batch_size * num_points) for function evaluations
        - MUST reuse buffers across multiple integrations

    Hardware Optimization:
        - GPU: Keep nodes/weights in constant memory if possible
        - Use torch.einsum for efficient weighted summation
        - Fuse operations: (func(points) * weights).sum() in single kernel
    """

    def __init__(self, domain: tuple[float, float] = (0.0, 1.0), num_points: int = 50):
        """Initialize Gauss-Legendre integrator.

        Args:
            domain: Integration domain
            num_points: Number of Gauss-Legendre points

        Note:
            Error scales exponentially for analytic functions.
            Exact for polynomials up to degree 2n-1 with n points.
            Most efficient for smooth functions.
        """
        super().__init__(domain, num_points)
        self._cache_quadrature()

    def _cache_quadrature(self) -> None:
        """Compute and cache Gauss-Legendre nodes and weights.

        Maps standard [-1, 1] Gauss-Legendre points to the target domain.
        """
        # Compute Gauss-Legendre nodes and weights on [-1, 1]
        # Using the eigenvalue method for the Jacobi matrix
        n = self.num_points

        # Create the Jacobi matrix for Legendre polynomials
        # Diagonal is zero, off-diagonal elements are β_j = j/sqrt(4j²-1)
        beta = torch.zeros(n - 1, dtype=torch.float64)
        for j in range(1, n):
            beta[j - 1] = j / torch.sqrt(
                torch.tensor(4 * j * j - 1, dtype=torch.float64)
            )

        # Create the tridiagonal matrix
        diag = torch.zeros(n, dtype=torch.float64)
        tridiagonal_matrix = (
            torch.diag(diag) + torch.diag(beta, 1) + torch.diag(beta, -1)
        )

        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = torch.linalg.eigh(tridiagonal_matrix)

        # Nodes are the eigenvalues
        nodes_std = eigenvals.float()

        # Weights are 2 * (first component of eigenvector)²
        weights_std = 2 * (eigenvecs[0, :] ** 2).float()

        # Map from [-1, 1] to target domain [a, b]
        a, b = self.domain
        # Linear transformation: x = (b-a)/2 * t + (a+b)/2
        self.points = (b - a) / 2 * nodes_std + (a + b) / 2
        self.weights = weights_std * (b - a) / 2

    def integrate(self, func: Callable[[Tensor], Tensor]) -> Tensor:
        """Compute integral using Gauss-Legendre quadrature.

        Args:
            func: Function to integrate

        Returns:
            Integral approximation

        Example:
            >>> integrator = GaussLegendreIntegrator((0, 1), 20)
            >>> result = integrator.integrate(lambda x: 1 / (1 + x**2))
        """
        # Evaluate function at quadrature points
        func_values = func(self.points)

        # Compute weighted sum over the last dimension (integration points)
        # Preserve any leading batch dimensions
        if func_values.dim() > 1:
            # Batched case: sum only over the last dimension
            return torch.sum(func_values * self.weights, dim=-1)
        # Scalar case: sum over all dimensions (backward compatibility)
        return torch.sum(func_values * self.weights)

    def get_quadrature_points(self) -> tuple[Tensor, Tensor]:
        """Return cached quadrature points and weights."""
        return self.points, self.weights


class MonteCarloIntegrator(Integrator):
    """Monte Carlo integration with importance sampling.

    Estimates integrals using random sampling:
    ∫f(x)p(x)dx ≈ (1/N) * Σ f(x_i) where x_i ~ p(x)

    Attributes:
        importance_dist: Optional importance sampling distribution
        use_quasi_random: Whether to use quasi-random (low-discrepancy) sequences
    """

    def __init__(
        self,
        domain: tuple[float, float] = (0.0, 1.0),
        num_points: int = 1000,
        importance_dist: Callable[[int], Tensor] | None = None,
        use_quasi_random: bool = False,
    ):
        """Initialize Monte Carlo integrator.

        Args:
            domain: Integration domain
            num_points: Number of random samples
            importance_dist: Optional importance sampling distribution
            use_quasi_random: Use Sobol sequences instead of pseudorandom

        Note:
            Error scales as O(1/√n) regardless of dimension.
            Useful for high-dimensional integrals or rough functions.
        """
        super().__init__(domain, num_points)
        self.importance_dist = importance_dist
        self.use_quasi_random = use_quasi_random

    def integrate(self, func: Callable[[Tensor], Tensor]) -> Tensor:
        """Compute integral using Monte Carlo sampling.

        Args:
            func: Function to integrate

        Returns:
            Integral estimate with statistical uncertainty

        Example:
            >>> integrator = MonteCarloIntegrator((0, 1), 10000)
            >>> result = integrator.integrate(lambda x: torch.sin(x))
        """
        # Generate points and weights
        points, weights = self.get_quadrature_points()

        # Evaluate function at sample points
        func_values = func(points)

        # Compute Monte Carlo estimate
        return torch.sum(func_values * weights)

    def get_quadrature_points(self) -> tuple[Tensor, Tensor]:
        """Generate random quadrature points and uniform weights.

        Note:
            Points are regenerated on each call for Monte Carlo.
            Use caching explicitly if deterministic behavior is needed.
        """
        a, b = self.domain

        if self.use_quasi_random:
            # Generate quasi-random (Sobol) sequence
            # For 1D, we can use a simple low-discrepancy sequence
            # Sobol sequences for 1D are essentially van der Corput sequences in base 2
            indices = torch.arange(self.num_points, dtype=torch.float32)
            # Van der Corput sequence in base 2
            points_uniform = torch.zeros_like(indices)
            for i in range(self.num_points):
                n = i
                base = 2.0
                fraction = 0.0
                power = 0.5
                while n > 0:
                    remainder = n % 2
                    fraction += remainder * power
                    power /= base
                    n //= 2
                points_uniform[i] = fraction
        # Generate pseudorandom points
        elif self.importance_dist is not None:
            # Use importance sampling distribution
            points_uniform = self.importance_dist(self.num_points)
        else:
            # Uniform distribution
            points_uniform = torch.rand(self.num_points)

        # Map from [0,1] to domain [a,b]
        points = a + (b - a) * points_uniform

        # Monte Carlo weights are uniform: (domain width) / num_points
        domain_width = b - a
        weights = torch.full((self.num_points,), domain_width / self.num_points)

        return points, weights


class AdaptiveIntegrator(Integrator):
    """Adaptive integration with automatic error control.

    Recursively subdivides the domain to achieve target accuracy,
    concentrating points where the function varies rapidly.

    Attributes:
        base_integrator: Underlying integration method
        tolerance: Target relative error tolerance
        max_subdivisions: Maximum recursion depth
    """

    def __init__(
        self,
        domain: tuple[float, float] = (0.0, 1.0),
        tolerance: float = 1e-6,
        max_subdivisions: int = 10,
        base_method: str = "gauss",
    ):
        """Initialize adaptive integrator.

        Args:
            domain: Integration domain
            tolerance: Target relative error
            max_subdivisions: Maximum subdivision levels
            base_method: Base integration method ("gauss", "simpson", etc.)

        Note:
            Automatically refines integration where needed.
            More efficient than uniform refinement for non-smooth functions.
        """
        super().__init__(domain, num_points=15)  # Initial points per subdivision
        self.tolerance = tolerance
        self.max_subdivisions = max_subdivisions
        self.base_method = base_method

    def integrate(self, func: Callable[[Tensor], Tensor]) -> Tensor:
        """Compute integral with adaptive refinement.

        Args:
            func: Function to integrate

        Returns:
            Integral value within specified tolerance

        Raises:
            RuntimeError: If tolerance cannot be achieved

        Example:
            >>> integrator = AdaptiveIntegrator((0, 1), tolerance=1e-8)
            >>> result = integrator.integrate(lambda x: 1/torch.sqrt(x + 0.01))
        """
        return self._adaptive_integrate(func, self.domain, 0)

    def _adaptive_integrate(
        self, func: Callable[[Tensor], Tensor], domain: tuple[float, float], depth: int
    ) -> Tensor:
        """Recursively integrate with adaptive subdivision."""
        if depth > self.max_subdivisions:
            raise RuntimeError(
                f"Could not achieve tolerance within {self.max_subdivisions} subdivisions"
            )

        # Create base integrator for this subdomain
        if self.base_method == "gauss":
            base_integrator = GaussLegendreIntegrator(domain, self.num_points)
        elif self.base_method == "simpson":
            # Ensure odd points for Simpson
            points = (
                self.num_points if self.num_points % 2 == 1 else self.num_points + 1
            )
            base_integrator = SimpsonIntegrator(domain, points)
        else:  # trapezoidal
            base_integrator = TrapezoidalIntegrator(domain, self.num_points)

        # Compute integral with base method
        result_fine = base_integrator.integrate(func)

        # Compute integral with coarser method for error estimate
        coarse_points = max(3, self.num_points // 2)
        if self.base_method == "gauss":
            coarse_integrator = GaussLegendreIntegrator(domain, coarse_points)
        elif self.base_method == "simpson":
            coarse_points = (
                coarse_points if coarse_points % 2 == 1 else coarse_points + 1
            )
            coarse_integrator = SimpsonIntegrator(domain, coarse_points)
        else:  # trapezoidal
            coarse_integrator = TrapezoidalIntegrator(domain, coarse_points)

        result_coarse = coarse_integrator.integrate(func)

        # Estimate error
        error = torch.abs(result_fine - result_coarse)
        relative_error = error / (torch.abs(result_fine) + 1e-15)

        # Check if tolerance is met
        if relative_error <= self.tolerance:
            return result_fine

        # Subdivide domain and recurse
        mid = (domain[0] + domain[1]) / 2
        left_domain = (domain[0], mid)
        right_domain = (mid, domain[1])

        left_result = self._adaptive_integrate(func, left_domain, depth + 1)
        right_result = self._adaptive_integrate(func, right_domain, depth + 1)

        return left_result + right_result

    def get_quadrature_points(self) -> tuple[Tensor, Tensor]:
        """Not applicable for adaptive integration.

        Raises:
            NotImplementedError: Adaptive integration doesn't use fixed points
        """
        raise NotImplementedError("Adaptive integration uses dynamic point placement")


def create_integrator(
    method: str = "gauss",
    domain: tuple[float, float] = (0.0, 1.0),
    num_points: int = 50,
    **kwargs,
) -> Integrator:
    """Factory function to create integrators.

    Args:
        method: Integration method ("trapezoidal", "simpson", "gauss", "monte_carlo", "adaptive")
        domain: Integration bounds
        num_points: Number of quadrature points
        **kwargs: Additional method-specific arguments

    Returns:
        Initialized integrator instance

    Raises:
        ValueError: If method is not recognized

    Example:
        >>> integrator = create_integrator("gauss", (0, 1), 30)
        >>> integral = integrator.integrate(lambda x: x**2)
    """
    method = method.lower()

    if method == "trapezoidal":
        return TrapezoidalIntegrator(domain=domain, num_points=num_points)

    if method == "simpson":
        # Ensure odd num_points for Simpson's rule
        if num_points % 2 == 0:
            num_points += 1
        return SimpsonIntegrator(domain=domain, num_points=num_points)

    if method == "gauss":
        return GaussLegendreIntegrator(domain=domain, num_points=num_points)

    if method == "monte_carlo":
        # Extract Monte Carlo specific arguments
        importance_dist = kwargs.get("importance_dist")
        use_quasi_random = kwargs.get("use_quasi_random", False)
        return MonteCarloIntegrator(
            domain=domain,
            num_points=num_points,
            importance_dist=importance_dist,
            use_quasi_random=use_quasi_random,
        )

    if method == "adaptive":
        # Extract adaptive specific arguments
        tolerance = kwargs.get("tolerance", 1e-6)
        max_subdivisions = kwargs.get("max_subdivisions", 10)
        base_method = kwargs.get("base_method", "gauss")
        return AdaptiveIntegrator(
            domain=domain,
            tolerance=tolerance,
            max_subdivisions=max_subdivisions,
            base_method=base_method,
        )

    supported_methods = ["trapezoidal", "simpson", "gauss", "monte_carlo", "adaptive"]
    raise ValueError(
        f"Unknown integration method: {method}. Supported methods: {supported_methods}"
    )
