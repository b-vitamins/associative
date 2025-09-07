"""Basis function implementations for continuous memory representations.

This module provides a framework for basis functions used to represent
continuous signals from discrete samples, following the formulation in
"Modern Hopfield Networks with Continuous-Time Memories" (Santos et al., 2025).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn


class BasisFunction(nn.Module, ABC):
    """Abstract base class for basis functions used in continuous memory.

    A basis function psi(t) maps continuous time t in [0, 1] to a vector space,
    enabling the representation of discrete memories as continuous functions
    through linear combinations: x_bar(t) = sum(beta_i * psi_i(t)).

    Attributes:
        num_basis: Number of basis functions
        domain: Tuple of (start, end) defining the domain, default (0, 1)
    """

    def __init__(self, num_basis: int, domain: tuple[float, float] = (0.0, 1.0)):
        """Initialize basis function.

        Args:
            num_basis: Number of basis functions to use
            domain: Domain of the basis functions as (start, end)

        Raises:
            ValueError: If num_basis <= 0 or domain is invalid
        """
        super().__init__()
        if num_basis <= 0:
            raise ValueError(f"num_basis must be positive, got {num_basis}")
        if domain[0] >= domain[1]:
            raise ValueError(f"Invalid domain {domain}, start must be < end")

        self.num_basis = num_basis
        self.domain = domain

    @abstractmethod
    def evaluate(self, time_points: Tensor) -> Tensor:
        """Evaluate basis functions at given time points.

        Args:
            time_points: Time points tensor of shape (..., num_points) or scalar
               Values should be in the domain [domain[0], domain[1]]

        Returns:
            Tensor of shape (..., num_basis, num_points) containing
            evaluations of each basis function at each time point.
            For scalar input, returns shape (num_basis,)

        Note:
            - Must handle both scalar and batched inputs efficiently
            - Should return 0 for t outside the domain
            - Must be differentiable for gradient-based optimization
        """
        pass

    def integrate(
        self,
        integrand: Callable[[Tensor], Tensor],
        num_points: int = 500,
        method: str = "trapezoidal",
    ) -> Tensor:
        """Numerically integrate a function over the basis domain.

        Args:
            integrand: Function to integrate, takes time points and returns values
            num_points: Number of points for numerical integration
            method: Integration method ("trapezoidal", "simpson", "gauss")

        Returns:
            Integral value(s) as a tensor

        Raises:
            ValueError: If method is not supported

        Note:
            This is used for computing expectations E_p[psi(t)] in the update rule
        """
        raise NotImplementedError

    def design_matrix(self, time_points: Tensor) -> Tensor:
        """Compute design matrix F for ridge regression.

        The design matrix F[i,j] = psi_i(t_j) is used to fit coefficients
        via ridge regression: B = X^T F^T (FF^T + lambda*I)^(-1)

        Args:
            time_points: Tensor of shape (num_samples,) with time coordinates

        Returns:
            Design matrix of shape (num_basis, num_samples)

        Note:
            This matrix is crucial for fitting continuous representations
            to discrete data points efficiently.
        """
        # Default implementation: use evaluate method
        return self.evaluate(time_points)


class RectangularBasis(BasisFunction):
    """Rectangular (box) basis functions with uniform spacing.

    Each basis function is a rectangular pulse:
    psi_i(t) = 1 if t is in [mu_i - w/2, mu_i + w/2], 0 otherwise

    where mu_i are uniformly spaced centers and w is the width.

    Attributes:
        centers: Tensor of shape (num_basis,) with function centers
        width: Width of each rectangular function
    """

    def __init__(
        self,
        num_basis: int,
        domain: tuple[float, float] = (0.0, 1.0),
        overlap: float = 0.0,
    ):
        """Initialize rectangular basis functions.

        Args:
            num_basis: Number of basis functions
            domain: Domain of the functions
            overlap: Overlap factor between adjacent functions (0 to 1)
                    0 = no overlap, 1 = complete overlap

        Raises:
            ValueError: If overlap not in [0, 1]

        Note:
            Centers are computed as: mu_i = (i + 0.5) * (domain_width / num_basis)
            Width is: w = (1 + overlap) * (domain_width / num_basis)
        """
        super().__init__(num_basis, domain)
        if not 0 <= overlap <= 1:
            raise ValueError(f"overlap must be in [0, 1], got {overlap}")
        self.overlap = overlap

        # Compute domain width
        domain_start, domain_end = domain
        domain_width = domain_end - domain_start

        # Compute centers: mu_i = domain_start + (i + 0.5) * (domain_width / num_basis)
        centers = domain_start + (
            torch.arange(self.num_basis, dtype=torch.float32) + 0.5
        ) * (domain_width / self.num_basis)
        self.register_buffer("centers", centers)

        # Compute width: w = (1 + overlap) * (domain_width / num_basis)
        self.width = (1 + overlap) * (domain_width / self.num_basis)

    def evaluate(self, time_points: Tensor) -> Tensor:
        """Evaluate rectangular basis functions.

        Returns 1 where t falls within each rectangle, 0 elsewhere.
        Handles batched inputs efficiently using broadcasting.
        """
        # Ensure t is a tensor
        if not isinstance(time_points, torch.Tensor):
            t = torch.tensor(time_points, dtype=torch.float32)
        else:
            t = time_points

        # Check if t is scalar
        is_scalar = t.dim() == 0
        if is_scalar:
            t = t.unsqueeze(0)  # Add a dimension for batch processing

        # Get domain bounds
        domain_start, domain_end = self.domain

        # Expand dimensions for broadcasting
        t_expanded = t.unsqueeze(0) if t.dim() == 1 else t.unsqueeze(-2)

        centers_expanded: Tensor = cast(Tensor, self.centers).unsqueeze(
            -1
        )  # (num_basis, 1)

        # Compute half-width
        half_width = self.width / 2

        # Compute bounds
        lower_bound = centers_expanded - half_width
        upper_bound = centers_expanded + half_width

        # Hack to include the domain_end in the last basis
        eps = 1e-6 * (domain_end - domain_start)
        upper_bound = upper_bound.clone()
        upper_bound[-1] += eps

        # Check if t is within [lower, upper)
        within_bounds = (t_expanded >= lower_bound) & (t_expanded < upper_bound)

        # Scale by 1/sqrt(width) for orthonormality
        # Integral of psi_i^2 = width * (1/sqrt(width))^2 = 1
        scale = 1.0 / torch.sqrt(torch.tensor(self.width))
        result = within_bounds.float() * scale

        # Attach grad_fn without changing values
        result = result + 0.0 * t_expanded

        # Set values outside domain to 0
        outside_mask = (t < domain_start) | (t > domain_end)
        mask = (~outside_mask).float().unsqueeze(-2)  # (..., 1, num_points)
        result = result * mask

        # Handle scalar case
        if is_scalar:
            result = result.squeeze()

        return result


class GaussianBasis(BasisFunction):
    """Gaussian radial basis functions.

    Each basis function is a Gaussian:
    psi_i(t) = exp(-||t - mu_i||^2 / (2*sigma_i^2))

    Attributes:
        centers: Tensor of shape (num_basis,) with Gaussian centers
        log_widths: Log of widths for numerical stability (learnable if specified)
    """

    def __init__(
        self,
        num_basis: int,
        domain: tuple[float, float] = (0.0, 1.0),
        learnable_widths: bool = False,
        init_width: float | None = None,
    ):
        """Initialize Gaussian basis functions.

        Args:
            num_basis: Number of basis functions
            domain: Domain of the functions
            learnable_widths: Whether widths should be learnable parameters
            init_width: Initial width (defaults to 1/num_basis if None)

        Note:
            Centers are initialized uniformly across the domain.
            Widths can be learned during training if learnable_widths=True.
        """
        super().__init__(num_basis, domain)
        self.learnable_widths = learnable_widths
        domain_start, domain_end = domain
        domain_width = domain_end - domain_start
        if init_width is None:
            init_width = domain_width / num_basis
        log_width_init = torch.log(torch.full((num_basis,), init_width))
        if learnable_widths:
            self.log_widths = nn.Parameter(log_width_init)
        else:
            self.register_buffer("log_widths", log_width_init)
        centers = torch.linspace(domain_start, domain_end, num_basis)
        self.register_buffer("centers", centers)

    def evaluate(self, time_points: Tensor) -> Tensor:
        """Evaluate Gaussian basis functions.

        Uses log-space computation for numerical stability:
        psi_i(t) = exp(-0.5 * ((t - mu_i) / sigma_i)^2)
        """
        # Ensure t is a tensor
        if not isinstance(time_points, torch.Tensor):
            t = torch.tensor(time_points, dtype=torch.float32)
        else:
            t = time_points

        # Check if t is scalar
        is_scalar = t.dim() == 0
        if is_scalar:
            t = t.unsqueeze(0)

        # Expand dimensions
        t_expanded = t.unsqueeze(-2)  # (..., 1, num_points)
        centers_expanded: Tensor = cast(Tensor, self.centers).unsqueeze(
            -1
        )  # (num_basis, 1)

        # Compute difference
        diff = t_expanded - centers_expanded

        # Compute sigma
        sigma: Tensor = (
            cast(Tensor, self.log_widths).exp().unsqueeze(-1)
        )  # (num_basis, 1)

        # Compute exponent
        exponent = -0.5 * (diff / sigma) ** 2

        # Compute result
        result = exponent.exp()

        # Set outside domain to 0
        domain_start, domain_end = self.domain
        outside_mask = (t < domain_start) | (t > domain_end)
        mask = (~outside_mask).float().unsqueeze(-2)
        result = result * mask

        if is_scalar:
            result = result.squeeze()

        return result


class FourierBasis(BasisFunction):
    """Fourier basis functions for periodic signals.

    Basis functions are sines and cosines of different frequencies:
    psi_2k(t) = cos(2*pi*k*t / T)
    psi_2k+1(t) = sin(2*pi*k*t / T)

    where T is the period and k = 0, 1, ..., num_basis//2

    Attributes:
        frequencies: Tensor of frequencies for each basis function
        use_complex: Whether to use complex exponentials instead
    """

    def __init__(
        self,
        num_basis: int,
        domain: tuple[float, float] = (0.0, 1.0),
        use_complex: bool = False,
        max_frequency: int | None = None,
    ):
        """Initialize Fourier basis functions.

        Args:
            num_basis: Number of basis functions (must be even if not complex)
            domain: Domain of the functions
            use_complex: Use complex exponentials instead of sin/cos
            max_frequency: Maximum frequency (defaults to num_basis//2)

        Raises:
            ValueError: If num_basis is odd and use_complex=False

        Note:
            For real-valued basis, we alternate between cos and sin.
            For complex, we use exp(2*pi*i*k*t/T).
        """
        super().__init__(num_basis, domain)
        if not use_complex and num_basis % 2 != 0:
            raise ValueError("num_basis must be even for real Fourier basis")
        self.use_complex = use_complex
        self.max_frequency = max_frequency or num_basis // 2
        self.period = domain[1] - domain[0]
        self.domain_start = domain[0]

        # Initialize frequencies and types
        if use_complex:
            self.register_buffer(
                "freqs", torch.arange(0, num_basis, dtype=torch.float32)
            )
        else:
            freqs = []
            is_cos = []
            for i in range(num_basis):
                if i % 2 == 0:
                    freq = i // 2
                    cos = True
                else:
                    freq = (i // 2) + 1
                    cos = False
                freqs.append(freq)
                is_cos.append(cos)
            self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))
            self.register_buffer("is_cos", torch.tensor(is_cos))
            # Scales for normalization
            freqs_tensor = cast(Tensor, self.freqs)
            condition = freqs_tensor == 0
            true_val = torch.tensor(1.0, dtype=torch.float32)
            false_val = torch.sqrt(torch.tensor(2.0, dtype=torch.float32))
            scale_values = torch.where(condition, true_val, false_val)
            self.register_buffer("scale", scale_values)

    def evaluate(self, time_points: Tensor) -> Tensor:
        """Evaluate Fourier basis functions.

        Computes sin and cos (or complex exp) at appropriate frequencies.
        """
        # Ensure t is a tensor
        if not isinstance(time_points, torch.Tensor):
            t = torch.tensor(time_points, dtype=torch.float32)
        else:
            t = time_points

        # Check if t is scalar
        is_scalar = t.dim() == 0
        if is_scalar:
            t = t.unsqueeze(0)

        # Normalize t to [0, 1]
        tau = (t - self.domain_start) / self.period

        # Expand
        tau_expanded = tau.unsqueeze(-2)  # (..., 1, num_points)

        if self.use_complex:
            freqs_expanded: Tensor = cast(Tensor, self.freqs).unsqueeze(-1)
            arg = 2 * torch.pi * freqs_expanded * tau_expanded
            result = torch.exp(1j * arg)
        else:
            freqs_expanded: Tensor = cast(Tensor, self.freqs).unsqueeze(-1)
            arg = 2 * torch.pi * freqs_expanded * tau_expanded
            cos_mask: Tensor = cast(Tensor, self.is_cos).unsqueeze(-1)
            values = torch.where(cos_mask, torch.cos(arg), torch.sin(arg))
            scale_expanded: Tensor = cast(Tensor, self.scale).unsqueeze(-1)
            result = values * scale_expanded

        # Set outside domain to 0
        domain_start, domain_end = self.domain
        outside_mask = (t < domain_start) | (t > domain_end)
        mask = (~outside_mask).to(result.dtype).unsqueeze(-2)
        result = result * mask

        if is_scalar:
            result = result.squeeze()

        return result


class PolynomialBasis(BasisFunction):
    """Polynomial basis functions.

    Each basis function is a monomial: psi_i(t) = t^i for i = 0, 1, ..., num_basis-1

    Attributes:
        powers: Tensor of shape (num_basis,) with powers [0, 1, ..., num_basis-1]
        normalize: Whether to normalize polynomials for stability
    """

    def __init__(
        self,
        num_basis: int,
        domain: tuple[float, float] = (0.0, 1.0),
        normalize: bool = True,
    ):
        """Initialize polynomial basis functions.

        Args:
            num_basis: Number of basis functions (polynomial degree + 1)
            domain: Domain of the functions
            normalize: Whether to normalize to [-1, 1] for numerical stability
        """
        super().__init__(num_basis, domain)
        self.normalize = normalize
        self.register_buffer("powers", torch.arange(num_basis, dtype=torch.float32))

    def evaluate(self, time_points: Tensor) -> Tensor:
        """Evaluate polynomial basis functions.

        Computes t^i for each power i.
        """
        if not isinstance(time_points, torch.Tensor):
            t = torch.tensor(time_points, dtype=torch.float32)
        else:
            t = time_points

        is_scalar = t.dim() == 0
        if is_scalar:
            t = t.unsqueeze(0)

        # Normalize to [-1, 1] if requested
        if self.normalize:
            domain_start, domain_end = self.domain
            t_norm = 2 * (t - domain_start) / (domain_end - domain_start) - 1
        else:
            t_norm = t

        # Expand dimensions
        t_expanded = t_norm.unsqueeze(-2)  # (..., 1, num_points)
        powers_expanded = cast(Tensor, self.powers).unsqueeze(-1)  # (num_basis, 1)

        # Compute t^i for each power
        result = torch.pow(t_expanded, powers_expanded)

        # Set outside domain to 0
        domain_start, domain_end = self.domain
        outside_mask = (t < domain_start) | (t > domain_end)
        mask = (~outside_mask).float().unsqueeze(-2)
        result = result * mask

        if is_scalar:
            result = result.squeeze()

        return result


def create_basis(basis_type: str, num_basis: int, **kwargs) -> BasisFunction:
    """Factory function to create basis functions.

    Args:
        basis_type: Type of basis ("rectangular", "gaussian", "fourier", "polynomial")
        num_basis: Number of basis functions
        **kwargs: Additional arguments passed to the basis constructor

    Returns:
        Initialized basis function module

    Raises:
        ValueError: If basis_type is not recognized

    Example:
        >>> basis = create_basis("gaussian", 32, learnable_widths=True)
        >>> t = torch.linspace(0, 1, 100)
        >>> psi = basis.evaluate(t)  # Shape: (32, 100)
    """
    basis_type = basis_type.lower()
    if basis_type == "rectangular":
        return RectangularBasis(num_basis, **kwargs)
    if basis_type == "gaussian":
        return GaussianBasis(num_basis, **kwargs)
    if basis_type == "fourier":
        return FourierBasis(num_basis, **kwargs)
    if basis_type == "polynomial":
        return PolynomialBasis(num_basis, **kwargs)
    raise ValueError(f"{basis_type} not recognized")


class ContinuousCompression(nn.Module):
    """Continuous compression via ridge regression on basis functions.

    Implements the compression mechanism:
    - Projects discrete sequences onto continuous basis functions
    - Uses ridge regression to find optimal coefficients
    - Enables reconstruction at arbitrary time points

    The compression operator R = (FF^T + λI)^(-1)F maps sequences
    from L points to M basis coefficients where M << L.

    Attributes:
        basis: Basis function module (rectangular, Gaussian, etc.)
        regularization: Ridge regression regularization parameter λ
        compression_dim: Number of basis functions M
        cached_design_matrix: Optional cached F for efficiency
        cached_regression_operator: Optional cached R for efficiency

    Performance Requirements:
        - MUST use Cholesky decomposition for (FF^T + λI)^(-1): O(M³)
        - MUST cache R matrix when seq_len is fixed to avoid O(M³) per forward
        - MUST batch matrix operations: compress all heads/dims simultaneously
        - MUST use torch.compile for basis evaluation loops
        - Target: <5ms for M=100, L=1024, batch=32 on V100

    Memory Requirements:
        - Design matrix F: MxL floats (cache if L fixed)
        - Regression operator R: MxL floats (cache if affordable)
        - Gram matrix FF^T: MxM floats (temporary, small)
        - Peak memory: O(MxL + batchxHxYxL) where H=heads, Y=qk_dim
        - MUST use in-place operations for large tensors

    Hardware Optimization:
        - GPU: Use cuBLAS for matrix multiplications (automatic via PyTorch)
        - Rectangular basis: Optimize via sparse operations (most entries zero)
        - CPU: Use MKL for Cholesky (torch.linalg.cholesky)
        - Mixed precision: Keep basis evaluation in fp32, compression in fp16
        - Flash Attention v2 compatible: Ensure M x M fits in SRAM (M≤128 ideal)
    """

    def __init__(
        self,
        basis: BasisFunction,
        regularization: float = 0.5,
        cache_operators: bool = False,
    ):
        """Initialize continuous compression module.

        Args:
            basis: Basis function instance for compression
            regularization: Ridge regression regularization λ > 0
            cache_operators: Whether to cache F and R matrices

        Raises:
            ValueError: If regularization <= 0

        Note:
            Caching operators speeds up repeated compressions on
            sequences of the same length but uses more memory.
        """
        super().__init__()
        if regularization <= 0:
            raise ValueError(f"regularization must be positive, got {regularization}")

        self.basis = basis
        self.regularization = regularization
        self.compression_dim = basis.num_basis
        self.cache_operators = cache_operators
        # Legacy cache fields for backward compatibility
        self.cached_design_matrix = None
        self.cached_regression_operator = None
        # New intelligent cache dictionaries
        self._design_matrix_cache = {}
        self._regression_operator_cache = {}

    def compute_design_matrix(self, seq_len: int) -> Tensor:
        """Compute design matrix F for given sequence length.

        The design matrix F[j, A] = ψ_j(A/L) evaluates each basis
        function at normalized time points.

        Args:
            seq_len: Length L of the sequence to compress

        Returns:
            Design matrix of shape (M, L) where M is compression_dim

        Note:
            Time points are normalized to [0, 1] via t_A = A/L.
            Results may be cached if cache_operators=True.
        """
        # Check new intelligent cache first
        if seq_len in self._design_matrix_cache:
            return self._design_matrix_cache[seq_len]

        # Check legacy cache if enabled (for backward compatibility)
        if (
            self.cache_operators
            and self.cached_design_matrix is not None
            and self.cached_design_matrix.shape[1] == seq_len
        ):
            return self.cached_design_matrix

        # Compute normalized time points: t_A = A/L for A in [0, L-1]
        time_points = torch.arange(seq_len, dtype=torch.float32) / seq_len

        # Evaluate basis functions: F[j, A] = ψ_j(A/L)
        design_matrix = self.basis.evaluate(time_points)

        # Cache in new intelligent cache
        self._design_matrix_cache[seq_len] = design_matrix

        # Cache in legacy cache if enabled (for backward compatibility)
        if self.cache_operators:
            self.cached_design_matrix = design_matrix

        return design_matrix

    def compute_regression_operator(self, design_matrix: Tensor) -> Tensor:
        """Compute ridge regression operator R.

        R = (FF^T + λI)^(-1)F enables efficient compression via
        matrix multiplication: B = RK^T.

        Args:
            design_matrix: F matrix of shape (M, L)

        Returns:
            Regression operator of shape (M, L)

        Note:
            Uses Cholesky decomposition for numerical stability.
            Complexity: O(M^3) for inversion + O(M^2 L) for multiplication.

        Performance Contract:
            - MUST use torch.linalg.cholesky_ex for GPU efficiency
            - MUST check info tensor for numerical issues
            - SHOULD use double precision for ill-conditioned matrices
            - MUST cache result if called with same L repeatedly
        """
        m, seq_length = design_matrix.shape

        # Create cache key from design matrix shape
        cache_key = (m, seq_length)

        # Check new intelligent cache first
        if cache_key in self._regression_operator_cache:
            return self._regression_operator_cache[cache_key]

        # Check legacy cache if enabled (for backward compatibility)
        if (
            self.cache_operators
            and self.cached_regression_operator is not None
            and self.cached_regression_operator.shape == (m, seq_length)
        ):
            return self.cached_regression_operator

        # Compute Gram matrix: FF^T
        gram_matrix = design_matrix @ design_matrix.T

        # Add regularization: FF^T + λI
        # Use small epsilon if regularization is exactly 0 to avoid singularity
        effective_regularization = (
            self.regularization if self.regularization > 0 else 1e-6
        )
        regularized_gram = gram_matrix + effective_regularization * torch.eye(
            m, dtype=gram_matrix.dtype, device=gram_matrix.device
        )

        # Use Cholesky decomposition for numerical stability
        # Solve (FF^T + λI) @ R = F  =>  R = (FF^T + λI)^(-1) @ F
        chol, info = torch.linalg.cholesky_ex(regularized_gram)

        # Check for numerical issues
        if info.max() > 0:
            # Fall back to direct solve if Cholesky fails
            regression_operator = torch.linalg.solve(regularized_gram, design_matrix)
        else:
            # Use Cholesky solve for efficiency
            regression_operator = torch.cholesky_solve(design_matrix, chol)

        # Cache in new intelligent cache
        self._regression_operator_cache[cache_key] = regression_operator

        # Cache in legacy cache if enabled (for backward compatibility)
        if self.cache_operators:
            self.cached_regression_operator = regression_operator

        return regression_operator

    def compress(
        self,
        keys: Tensor,
        seq_len: int | None = None,
    ) -> Tensor:
        """Compress sequences to basis coefficients.

        Computes B = RK^T where K are the keys to compress.

        Args:
            keys: Tensor of shape (..., H, L) or (..., Y, H, L) to compress
            seq_len: Optional sequence length (inferred from keys if None)

        Returns:
            Coefficients of shape (..., H, M) or (..., Y, H, M)

        Example:
            >>> compression = ContinuousCompression(basis, regularization=0.01)
            >>> keys = torch.randn(8, 64, 1024)  # (heads, dim, seq_len)
            >>> coeffs = compression.compress(keys)  # (8, 64, M)

        Note:
            Handles both 3D (H, Y, L) and 4D (Y, H, L) key tensors.
            The compression preserves all batch dimensions.
            Uses cached operators when seq_len is consistent for performance.
        """
        # Infer sequence length from keys if not provided
        actual_seq_len: int
        actual_seq_len = int(keys.shape[-1]) if seq_len is None else seq_len

        # Get design matrix and regression operator (uses caching internally)
        design_matrix = self.compute_design_matrix(actual_seq_len)
        regression_operator = self.compute_regression_operator(design_matrix)

        # Compute B = R @ K^T
        # keys: (..., Y, H, L) -> transpose to (..., Y, L, H) for einsum
        # regression_operator: (M, L)
        # Result: (..., Y, M, H) -> transpose to (..., Y, H, M)

        # Transpose last two dimensions: (..., Y, H, L) -> (..., Y, L, H)
        keys_transposed = keys.transpose(-2, -1)

        # Apply regression operator: (M, L) @ (..., Y, L, H) -> (..., Y, M, H)
        coefficients = torch.einsum(
            "ml,...lh->...mh", regression_operator, keys_transposed
        )

        # Transpose back to get (..., Y, H, M)
        return coefficients.transpose(-2, -1)

    def reconstruct(
        self,
        coefficients: Tensor,
        time_points: Tensor,
    ) -> Tensor:
        """Reconstruct continuous functions from coefficients.

        Computes K̄(t) = Σ_j B_j ψ_j(t) for arbitrary time points.

        Args:
            coefficients: Basis coefficients of shape (..., M)
            time_points: Time points of shape (T,) or scalar

        Returns:
            Reconstructed values of shape (..., T) or (...,) for scalar t

        Example:
            >>> t = torch.linspace(0, 1, 100)
            >>> reconstructed = compression.reconstruct(coeffs, t)

        Note:
            Time points should be in the basis function domain.
            Supports arbitrary resolution reconstruction.
        """
        # Evaluate basis functions at time points
        # basis_values: (M, T) or (M,) for scalar
        basis_values = self.basis.evaluate(time_points)

        # Compute K̄(t) = Σ_j B_j ψ_j(t) = B^T @ ψ(t)
        # coefficients: (..., M), basis_values: (M, T) -> (..., T)
        # or coefficients: (..., M), basis_values: (M,) -> (...,)

        if time_points.dim() == 0:  # scalar time point
            # basis_values: (M,), coefficients: (..., M) -> (...)
            reconstructed = torch.einsum("...m,m->...", coefficients, basis_values)
        else:
            # basis_values: (M, T), coefficients: (..., M) -> (..., T)
            reconstructed = torch.einsum("...m,mt->...t", coefficients, basis_values)

        return reconstructed

    def compute_continuous_scores(
        self,
        queries: Tensor,
        compressed_keys: Tensor,
        time_points: Tensor | None = None,
    ) -> Tensor:
        """Compute continuous attention scores s(t) = K̄(t)^T Q.

        Evaluates query-key similarities at specified time points
        using the compressed representation.

        Args:
            queries: Query tensor of shape (..., Y, H, L_q)
            compressed_keys: Compressed keys of shape (..., Y, H, M)
            time_points: Evaluation points (default: linspace(0, 1, 50))

        Returns:
            Scores of shape (..., H, L_q, T) where T is num time points

        Note:
            This is used for computing partition functions in MET.
            Default time points suitable for numerical integration.
        """
        # Use default time points if not provided
        if time_points is None:
            time_points = torch.linspace(0, 1, 50, device=queries.device)

        # Reconstruct keys at time points: K̄(t)
        # compressed_keys: (..., Y, H, M), time_points: (T,)
        # reconstructed_keys: (..., Y, H, T)
        reconstructed_keys = self.reconstruct(compressed_keys, time_points)

        # Compute scores s(t) = Q^T @ K̄(t)
        # queries: (..., Y, H, L_q), reconstructed_keys: (..., Y, H, T)
        # scores: (..., H, L_q, T)
        return torch.einsum("...yhq,...yht->...hqt", queries, reconstructed_keys)

    def forward(
        self,
        keys: Tensor,
        queries: Tensor | None = None,
        return_coefficients: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compress keys and optionally compute continuous scores.

        Args:
            keys: Keys to compress of shape (..., H, L)
            queries: Optional queries for score computation
            return_coefficients: Whether to return compression coefficients

        Returns:
            If queries provided and return_coefficients=False:
                Continuous scores of shape (..., H, L_q, T)
            If queries provided and return_coefficients=True:
                Tuple of (scores, coefficients)
            If no queries:
                Compressed coefficients of shape (..., H, M)

        Note:
            Main entry point for continuous compression in attention.
        """
        # Compress keys to basis coefficients
        coefficients = self.compress(keys)

        # If no queries, just return coefficients
        if queries is None:
            return coefficients

        # Compute continuous scores
        scores = self.compute_continuous_scores(queries, coefficients)

        # Return based on return_coefficients flag
        if return_coefficients:
            return scores, coefficients
        return scores

    def clear_cache(self) -> None:
        """Clear all cached matrices to free memory.

        This method clears both the new intelligent caches and legacy caches.
        Useful when switching to different sequence lengths or when memory
        is constrained.

        Note:
            After clearing cache, next compress() call will recompute matrices.
        """
        self._design_matrix_cache.clear()
        self._regression_operator_cache.clear()
        self.cached_design_matrix = None
        self.cached_regression_operator = None

    def extra_repr(self) -> str:
        """String representation of module configuration."""
        cache_info = f"cached_seqs={len(self._design_matrix_cache)}"
        return (
            f"compression_dim={self.compression_dim}, "
            f"regularization={self.regularization}, "
            f"basis_type={self.basis.__class__.__name__}, "
            f"{cache_info}"
        )
