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
        result = within_bounds.float()

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


def create_basis(basis_type: str, num_basis: int, **kwargs) -> BasisFunction:
    """Factory function to create basis functions.

    Args:
        basis_type: Type of basis ("rectangular", "gaussian", "fourier")
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
    raise ValueError(f"{basis_type} not recognized")
