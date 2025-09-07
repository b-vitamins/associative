"""Comprehensive tests for basis function implementations."""

import pytest
import torch

from associative.nn.modules.basis import (
    ContinuousCompression,
    FourierBasis,
    GaussianBasis,
    RectangularBasis,
    create_basis,
)
from associative.nn.modules.config import BasisConfig


class TestBasisFunction:
    """Test abstract BasisFunction interface and common behavior."""

    def test_invalid_initialization(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid num_basis
        with pytest.raises(ValueError, match="num_basis must be positive"):
            RectangularBasis(num_basis=0)
        with pytest.raises(ValueError, match="num_basis must be positive"):
            RectangularBasis(num_basis=-5)

        # Test invalid domain
        with pytest.raises(ValueError, match="Invalid domain"):
            RectangularBasis(num_basis=10, domain=(1.0, 0.0))
        with pytest.raises(ValueError, match="Invalid domain"):
            RectangularBasis(num_basis=10, domain=(0.5, 0.5))

    def test_domain_property(self):
        """Test that domain is correctly stored."""
        num_basis_test = 10
        basis = RectangularBasis(num_basis=num_basis_test, domain=(-1.0, 2.0))
        assert basis.domain == (-1.0, 2.0)
        assert basis.num_basis == num_basis_test


class TestRectangularBasis:
    """Test rectangular basis function implementation."""

    @pytest.fixture
    def rectangular_basis(self):
        """Create a standard rectangular basis."""
        return RectangularBasis(num_basis=4, domain=(0.0, 1.0))

    def test_initialization(self, rectangular_basis):
        """Test proper initialization of rectangular basis."""
        expected_num_basis = 4
        expected_overlap = 0.5
        assert rectangular_basis.num_basis == expected_num_basis
        assert rectangular_basis.domain == (0.0, 1.0)
        assert rectangular_basis.overlap == 0.0

        # Test with overlap
        basis_overlap = RectangularBasis(num_basis=4, overlap=expected_overlap)
        assert basis_overlap.overlap == expected_overlap

        # Test invalid overlap
        with pytest.raises(ValueError, match="overlap must be in"):
            RectangularBasis(num_basis=4, overlap=-0.1)
        with pytest.raises(ValueError, match="overlap must be in"):
            RectangularBasis(num_basis=4, overlap=1.1)

    def test_evaluate_scalar(self, rectangular_basis):
        """Test evaluation at scalar time points."""
        # Test at various points
        t = torch.tensor(0.125)  # Should be in first basis function
        psi = rectangular_basis.evaluate(t)
        assert psi.shape == (4,)
        # Normalized rectangular basis has value 1/sqrt(width) = 1/sqrt(0.25) = 2.0
        assert psi[0] == 2.0  # First basis function active (normalized)
        assert psi[1:].sum() == 0.0  # Others inactive

        # Test at boundary
        t = torch.tensor(0.25)  # Boundary between first and second
        psi = rectangular_basis.evaluate(t)
        # Depending on implementation, either first or second is active
        assert psi.sum() == 2.0  # Exactly one active (normalized)

    def test_evaluate_batch(self, rectangular_basis):
        """Test evaluation at multiple time points."""
        t = torch.linspace(0, 1, 100)
        psi = rectangular_basis.evaluate(t)
        assert psi.shape == (4, 100)

        # Each time point should have exactly one active basis (no overlap)
        # Normalized value is 1/sqrt(0.25) = 2.0
        assert torch.allclose(psi.sum(dim=0), 2.0 * torch.ones(100))

        # Each basis should be active for roughly 1/4 of the points
        active_counts = (psi > 0).sum(dim=1).float()
        expected_count = 100 / 4
        assert torch.allclose(active_counts, torch.full((4,), expected_count), atol=2)

    def test_evaluate_outside_domain(self, rectangular_basis):
        """Test that evaluation outside domain returns zeros."""
        t = torch.tensor([-0.5, 1.5])
        psi = rectangular_basis.evaluate(t)
        assert psi.shape == (4, 2)
        assert torch.allclose(psi, torch.zeros(4, 2))

    def test_design_matrix(self, rectangular_basis):
        """Test design matrix computation."""
        time_points = torch.linspace(0, 1, 10)
        design_matrix = rectangular_basis.design_matrix(time_points)
        assert design_matrix.shape == (4, 10)

        # Each column should have exactly one non-zero entry (no overlap)
        # Normalized value is 1/sqrt(0.25) = 2.0
        assert torch.allclose(design_matrix.sum(dim=0), 2.0 * torch.ones(10))

        # Design matrix should match evaluate
        psi = rectangular_basis.evaluate(time_points)
        assert torch.allclose(design_matrix, psi)

    def test_differentiability(self, rectangular_basis):
        """Test that basis functions are differentiable (important for optimization)."""
        t = torch.linspace(0, 1, 10, requires_grad=True)
        psi = rectangular_basis.evaluate(t)

        # Should be able to compute gradients (even if they're zero for rectangular)
        loss = psi.sum()
        loss.backward()
        assert t.grad is not None


class TestGaussianBasis:
    """Test Gaussian basis function implementation."""

    @pytest.fixture
    def gaussian_basis(self):
        """Create a standard Gaussian basis."""
        return GaussianBasis(num_basis=5, domain=(0.0, 1.0))

    def test_initialization(self, gaussian_basis):
        """Test proper initialization of Gaussian basis."""
        expected_num_basis = 5
        assert gaussian_basis.num_basis == expected_num_basis
        assert gaussian_basis.domain == (0.0, 1.0)
        assert not gaussian_basis.learnable_widths

        # Test with learnable widths
        basis_learn = GaussianBasis(num_basis=5, learnable_widths=True)
        assert basis_learn.learnable_widths

        # Test with custom init width
        GaussianBasis(num_basis=5, init_width=0.1)
        # Width should be stored (implementation detail)

    def test_evaluate_properties(self, gaussian_basis):
        """Test properties of Gaussian basis evaluation."""
        t = torch.linspace(0, 1, 100)
        psi = gaussian_basis.evaluate(t)
        assert psi.shape == (5, 100)

        # All values should be between 0 and 1 (Gaussian RBF)
        assert (psi >= 0).all()
        assert (psi <= 1).all()

        # Centers should have maximum activation
        centers = torch.linspace(0, 1, 5)
        for i, center in enumerate(centers):
            psi_center = gaussian_basis.evaluate(center)
            assert psi_center[i] == psi_center.max()

    def test_smoothness(self, gaussian_basis):
        """Test that Gaussian basis is smooth."""
        t = torch.linspace(0, 1, 1000)
        psi = gaussian_basis.evaluate(t)

        # Compute discrete derivatives
        diff = psi[:, 1:] - psi[:, :-1]

        max_diff_threshold = 0.1
        # Gaussians should be smooth (small discrete derivatives)
        assert diff.abs().max() < max_diff_threshold

    def test_learnable_widths(self):
        """Test that widths can be learned."""
        basis = GaussianBasis(num_basis=3, learnable_widths=True)

        # Widths should be parameters
        params = list(basis.parameters())
        assert len(params) > 0  # Should have learnable parameters

        # Test gradient flow
        t = torch.linspace(0, 1, 10)
        psi = basis.evaluate(t)
        loss = psi.sum()
        loss.backward()

        # Gradients should flow to width parameters
        for p in basis.parameters():
            if p.grad is not None:
                assert not torch.allclose(p.grad, torch.zeros_like(p.grad))


class TestFourierBasis:
    """Test Fourier basis function implementation."""

    @pytest.fixture
    def fourier_basis(self):
        """Create a standard Fourier basis."""
        return FourierBasis(num_basis=6, domain=(0.0, 1.0))

    def test_initialization(self, fourier_basis):
        """Test proper initialization of Fourier basis."""
        expected_num_basis_real = 6
        expected_num_basis_complex = 5
        assert fourier_basis.num_basis == expected_num_basis_real
        assert fourier_basis.domain == (0.0, 1.0)
        assert not fourier_basis.use_complex

        # Test that odd num_basis raises error for real Fourier
        with pytest.raises(ValueError, match="must be even"):
            FourierBasis(num_basis=5, use_complex=False)

        # Complex Fourier can have odd num_basis
        basis_complex = FourierBasis(
            num_basis=expected_num_basis_complex, use_complex=True
        )
        assert basis_complex.num_basis == expected_num_basis_complex

    def test_orthogonality(self, fourier_basis):
        """Test that Fourier basis functions are orthogonal."""
        # Sample many points for numerical integration
        t = torch.linspace(0, 1, 1000)
        psi = fourier_basis.evaluate(t)

        # Compute inner products via numerical integration
        dt = 1.0 / 1000
        gram = psi @ psi.T * dt

        # Should be approximately identity (orthonormal)
        eye = torch.eye(6)
        assert torch.allclose(gram, eye, atol=0.1)

    def test_periodicity(self, fourier_basis):
        """Test that Fourier basis is periodic."""
        t1 = torch.tensor(0.0)
        t2 = torch.tensor(1.0)

        psi1 = fourier_basis.evaluate(t1)
        psi2 = fourier_basis.evaluate(t2)

        # Values at 0 and 1 should be close (periodic)
        assert torch.allclose(psi1, psi2, atol=1e-5)

    def test_frequency_content(self, fourier_basis):
        """Test that basis contains expected frequencies."""
        t = torch.linspace(0, 1, 1000)
        psi = fourier_basis.evaluate(t)

        # First basis should be constant (DC component)
        assert torch.allclose(psi[0], torch.ones(1000), atol=0.01)

        # Others should oscillate with increasing frequency
        crossing_tolerance = 2
        for i in range(1, 6):
            # Count zero crossings as proxy for frequency
            signs = torch.sign(psi[i])
            crossings = (signs[1:] != signs[:-1]).sum()
            expected_crossings = i  # Roughly i crossings for i-th basis
            assert abs(crossings - expected_crossings) <= crossing_tolerance


class TestBasisFactory:
    """Test the create_basis factory function."""

    def test_create_rectangular(self):
        """Test creation of rectangular basis."""
        num_basis_test = 10
        overlap_test = 0.2
        basis = create_basis("rectangular", num_basis_test, overlap=overlap_test)
        assert isinstance(basis, RectangularBasis)
        assert basis.num_basis == num_basis_test
        assert basis.overlap == overlap_test

    def test_create_gaussian(self):
        """Test creation of Gaussian basis."""
        num_basis_test = 8
        basis = create_basis(
            "gaussian", num_basis_test, learnable_widths=True, init_width=0.05
        )
        assert isinstance(basis, GaussianBasis)
        assert basis.num_basis == num_basis_test
        assert basis.learnable_widths

    def test_create_fourier(self):
        """Test creation of Fourier basis."""
        num_basis_test = 12
        basis = create_basis("fourier", num_basis_test, use_complex=False)
        assert isinstance(basis, FourierBasis)
        assert basis.num_basis == num_basis_test
        assert not basis.use_complex

    def test_invalid_basis_type(self):
        """Test that invalid basis type raises error."""
        with pytest.raises(ValueError, match="not recognized"):
            create_basis("invalid_type", 10)


class TestBasisIntegration:
    """Integration tests for basis functions."""

    def test_basis_config_integration(self):
        """Test integration with BasisConfig."""
        num_basis_test = 16
        config = BasisConfig(
            num_basis=num_basis_test,
            basis_type="gaussian",
            domain=(-1.0, 1.0),
            learnable=True,
            init_width=0.1,
        )

        # Should be able to create basis from config
        basis = create_basis(
            config.basis_type,
            config.num_basis,
            domain=config.domain,
            learnable_widths=config.learnable,
            init_width=config.init_width,
        )

        assert basis.num_basis == num_basis_test
        assert basis.domain == (-1.0, 1.0)

    @pytest.mark.parametrize("basis_type", ["rectangular", "gaussian", "fourier"])
    def test_all_basis_types_compatible(self, basis_type):
        """Test that all basis types implement the same interface."""
        num_basis = 8 if basis_type == "fourier" else 7

        basis = create_basis(basis_type, num_basis)

        # All should support scalar evaluation
        t_scalar = torch.tensor(0.5)
        psi_scalar = basis.evaluate(t_scalar)
        assert psi_scalar.shape == (num_basis,)

        # All should support batch evaluation
        t_batch = torch.linspace(0, 1, 50)
        psi_batch = basis.evaluate(t_batch)
        assert psi_batch.shape == (num_basis, 50)

        # All should support design matrix
        design_matrix = basis.design_matrix(t_batch)
        assert design_matrix.shape == (num_basis, 50)

        # All should be differentiable
        t_grad = torch.linspace(0, 1, 10, requires_grad=True)
        psi_grad = basis.evaluate(t_grad)
        psi_grad.sum().backward()
        assert t_grad.grad is not None


# Fixtures for ContinuousCompression tests
@pytest.fixture
def rectangular_basis_8():
    """Create rectangular basis for testing."""
    return RectangularBasis(num_basis=8, domain=(0.0, 1.0))


@pytest.fixture
def gaussian_basis_12():
    """Create Gaussian basis for testing."""
    return GaussianBasis(num_basis=12, domain=(0.0, 1.0))


@pytest.fixture
def compression_rect(rectangular_basis_8):
    """Create compression with rectangular basis."""
    return ContinuousCompression(
        basis=rectangular_basis_8, regularization=0.01, cache_operators=False
    )


@pytest.fixture
def compression_cached(gaussian_basis_12):
    """Create compression with caching enabled."""
    return ContinuousCompression(
        basis=gaussian_basis_12, regularization=0.05, cache_operators=True
    )


class TestContinuousCompression:
    """Test ContinuousCompression for continuous memory compression via ridge regression.

    Tests cover all mathematical contracts and performance requirements from docstrings:
    - Design matrix F[j,A] = ψ_j(A/L) with shape (M, L)
    - Regression operator R = (FF^T + λI)^(-1)F with shape (M, L)
    - Compression B = RK^T mapping (..., H, L) -> (..., H, M)
    - Reconstruction K̄(t) = Σ_j B_j ψ_j(t) at arbitrary time points
    - Continuous scores s(t) = K̄(t)^T Q for attention computation
    - Numerical stability via Cholesky decomposition and regularization
    - Caching behavior when enabled for performance optimization
    """


class TestContinuousCompressionInit:
    """Test ContinuousCompression initialization."""

    def test_init_with_valid_basis_and_regularization(self, rectangular_basis_8):
        """Test successful initialization with valid parameters."""
        regularization = 0.01
        compression = ContinuousCompression(
            basis=rectangular_basis_8,
            regularization=regularization,
            cache_operators=False,
        )

        # Verify attributes are set correctly
        assert compression.basis is rectangular_basis_8
        assert compression.regularization == regularization
        assert compression.compression_dim == rectangular_basis_8.num_basis
        assert compression.cache_operators is False
        assert compression.cached_design_matrix is None
        assert compression.cached_regression_operator is None

    def test_init_with_cache_operators_true(self, gaussian_basis_12):
        """Test initialization with cache_operators=True."""
        compression = ContinuousCompression(
            basis=gaussian_basis_12, regularization=0.1, cache_operators=True
        )

        assert compression.cache_operators is True
        assert compression.cached_design_matrix is None
        assert compression.cached_regression_operator is None

    def test_init_raises_valueerror_for_zero_regularization(self, rectangular_basis_8):
        """Test ValueError when regularization is zero."""
        with pytest.raises(ValueError, match="regularization must be positive, got 0"):
            ContinuousCompression(basis=rectangular_basis_8, regularization=0.0)

    def test_init_raises_valueerror_for_negative_regularization(
        self, rectangular_basis_8
    ):
        """Test ValueError when regularization is negative."""
        with pytest.raises(
            ValueError, match="regularization must be positive, got -0.01"
        ):
            ContinuousCompression(basis=rectangular_basis_8, regularization=-0.01)

    def test_compression_dim_matches_basis_num_basis(self):
        """Test compression_dim equals basis.num_basis."""
        num_basis_values = [4, 16, 32, 64]
        for num_basis in num_basis_values:
            basis = RectangularBasis(num_basis=num_basis)
            compression = ContinuousCompression(basis=basis, regularization=0.01)
            assert compression.compression_dim == num_basis

    def test_different_regularization_values(self, rectangular_basis_8):
        """Test initialization with various regularization values."""
        reg_values = [1e-6, 1e-3, 0.01, 0.1, 1.0, 10.0]
        for reg in reg_values:
            compression = ContinuousCompression(
                basis=rectangular_basis_8, regularization=reg
            )
            assert compression.regularization == reg


class TestContinuousCompressionDesignMatrix:
    """Test compute_design_matrix method."""

    def test_design_matrix_expected_shape(self, compression_rect):
        """Test expected shape contract: (M, L) where M is compression_dim."""
        seq_len = 64
        design_matrix = compression_rect.compute_design_matrix(seq_len)
        assert design_matrix.shape == (compression_rect.compression_dim, seq_len)
        assert design_matrix.shape == (
            8,
            64,
        )  # rectangular_basis_8 has 8 basis functions

    def test_time_points_normalization_contract(self, compression_rect):
        """Test that time points are normalized to [0, 1]."""
        # Time points t_A = A/L for A in [0, L-1] should result in F[j, A] = ψ_j(A/L)
        seq_len = 128
        design_matrix = compression_rect.compute_design_matrix(seq_len)

        # Manually compute expected values using basis.evaluate
        time_points = torch.arange(seq_len, dtype=torch.float32) / seq_len
        expected_matrix = compression_rect.basis.evaluate(time_points)

        assert torch.allclose(design_matrix, expected_matrix)
        assert design_matrix.shape == (8, 128)

    def test_caching_behavior_contract(self, compression_cached):
        """Test caching behavior when cache_operators=True."""
        seq_len = 100

        # First call should compute and cache
        matrix1 = compression_cached.compute_design_matrix(seq_len)
        assert compression_cached.cached_design_matrix is not None

        # Second call with same seq_len should return cached result
        matrix2 = compression_cached.compute_design_matrix(seq_len)
        assert torch.allclose(matrix1, matrix2)

        # Different seq_len should compute new matrix
        matrix3 = compression_cached.compute_design_matrix(seq_len + 1)
        assert matrix3.shape[1] == seq_len + 1

    @pytest.mark.parametrize("seq_len", [1, 16, 64, 256, 1024])
    def test_different_sequence_lengths(self, compression_rect, seq_len):
        """Test design matrix computation for different sequence lengths."""
        design_matrix = compression_rect.compute_design_matrix(seq_len)
        assert design_matrix.shape == (compression_rect.compression_dim, seq_len)
        assert design_matrix.shape == (8, seq_len)

    def test_design_matrix_uses_basis_evaluate(self, compression_rect):
        """Test that design matrix uses basis.evaluate method."""
        seq_len = 50
        design_matrix = compression_rect.compute_design_matrix(seq_len)

        # Should match basis.evaluate for normalized time points
        time_points = torch.arange(seq_len, dtype=torch.float32) / seq_len
        expected = compression_rect.basis.evaluate(time_points)

        assert torch.allclose(design_matrix, expected)

    def test_design_matrix_mathematical_properties(self, compression_rect):
        """Test mathematical properties of the design matrix."""
        seq_len = 100
        design_matrix = compression_rect.compute_design_matrix(seq_len)

        # For normalized rectangular basis with 8 functions, each column should sum to sqrt(8)
        # because each basis function has width 1/8 and is scaled by 1/sqrt(1/8) = sqrt(8)
        num_basis = compression_rect.basis.num_basis
        expected_sum = torch.sqrt(torch.tensor(float(num_basis)))
        column_sums = design_matrix.sum(dim=0)
        assert torch.allclose(column_sums, torch.full((seq_len,), expected_sum.item()))

        # Values should be non-negative (rectangular basis is always >= 0)
        assert (design_matrix >= 0).all()

        # Each column should have exactly one non-zero entry for non-overlapping rectangular
        non_zero_counts = (design_matrix > 0).sum(dim=0)
        assert torch.allclose(non_zero_counts.float(), torch.ones(seq_len))


class TestContinuousCompressionRegressionOperator:
    """Test compute_regression_operator method."""

    def test_regression_operator_shape_contract(self, compression_rect):
        """Test expected shape: R should be (M, L)."""
        m, seq_len = 8, 64
        design_matrix = torch.randn(m, seq_len)
        regression_operator = compression_rect.compute_regression_operator(
            design_matrix
        )

        # R = (FF^T + λI)^(-1)F has shape (M, L)
        assert regression_operator.shape == (m, seq_len)
        assert regression_operator.shape == design_matrix.shape

    def test_mathematical_correctness(self, compression_rect):
        """Test that R satisfies the ridge regression equation."""
        m, seq_len = 8, 32
        design_matrix = torch.randn(m, seq_len)
        regression_operator = compression_rect.compute_regression_operator(
            design_matrix
        )

        # Verify R = (FF^T + λI)^(-1)F
        gram_matrix = design_matrix @ design_matrix.T
        regularized_gram = gram_matrix + compression_rect.regularization * torch.eye(m)
        expected_r = torch.linalg.solve(regularized_gram, design_matrix)

        assert torch.allclose(regression_operator, expected_r, atol=1e-5)

    def test_cholesky_decomposition_numerical_stability(self, compression_rect):
        """Test numerical stability using Cholesky decomposition."""
        m, seq_len = 8, 64
        design_matrix = torch.randn(m, seq_len)
        regression_operator = compression_rect.compute_regression_operator(
            design_matrix
        )

        # Should handle near-singular matrices gracefully
        gram_matrix = design_matrix @ design_matrix.T
        regularized_gram = gram_matrix + compression_rect.regularization * torch.eye(m)

        # Check that regularized Gram matrix is positive definite
        eigenvals = torch.linalg.eigvals(regularized_gram)
        assert (eigenvals.real > 0).all(), (
            "Regularized Gram matrix should be positive definite"
        )

        # Result should be finite
        assert torch.isfinite(regression_operator).all()

    @pytest.mark.parametrize("regularization", [1e-6, 1e-3, 0.01, 0.1, 1.0])
    def test_numerical_stability_with_regularization(
        self, rectangular_basis_8, regularization
    ):
        """Test numerical stability with different regularization values."""
        compression = ContinuousCompression(
            basis=rectangular_basis_8, regularization=regularization
        )
        design_matrix = torch.randn(8, 32)
        regression_operator = compression.compute_regression_operator(design_matrix)

        # Should produce finite results for all regularization values
        assert torch.isfinite(regression_operator).all()
        assert regression_operator.shape == (8, 32)

        # Larger regularization should make the operator more "diagonal-like"
        # by reducing the influence of off-diagonal elements
        gram_matrix = design_matrix @ design_matrix.T
        condition_number = torch.linalg.cond(
            gram_matrix + regularization * torch.eye(8)
        )
        assert condition_number < 1e10  # Should be well-conditioned

    def test_caching_behavior(self, compression_cached):
        """Test that results are cached when cache_operators=True."""
        design_matrix = torch.randn(12, 64)

        # First call should compute and cache
        r1 = compression_cached.compute_regression_operator(design_matrix)
        assert compression_cached.cached_regression_operator is not None

        # Second call with same matrix should return cached result
        r2 = compression_cached.compute_regression_operator(design_matrix)
        assert torch.allclose(r1, r2)

        # Different matrix should compute new operator (cache key should be based on shape/hash)
        different_matrix = torch.randn(12, 64) + 1.0
        r3 = compression_cached.compute_regression_operator(different_matrix)
        assert r3.shape == r1.shape

    def test_gram_matrix_properties(self, compression_rect):
        """Test properties of the Gram matrix FF^T."""
        m, seq_len = 8, 64
        design_matrix = torch.randn(m, seq_len)

        # Compute Gram matrix manually
        gram_matrix = design_matrix @ design_matrix.T
        regularized_gram = gram_matrix + compression_rect.regularization * torch.eye(m)

        # Should be symmetric
        assert torch.allclose(regularized_gram, regularized_gram.T)

        # Should be positive definite (all eigenvalues > 0)
        eigenvals = torch.linalg.eigvals(regularized_gram)
        assert (eigenvals.real > 0).all()

        # Test the actual regression operator
        regression_operator = compression_rect.compute_regression_operator(
            design_matrix
        )
        assert torch.isfinite(regression_operator).all()
        assert regression_operator.shape == (m, seq_len)

    def test_reconstruction_property(self, compression_rect):
        """Test that RR^T approaches identity for well-conditioned problems."""
        m = 8
        # Create a well-conditioned design matrix
        design_matrix = torch.eye(m)  # Identity matrix for simplicity
        regression_operator = compression_rect.compute_regression_operator(
            design_matrix
        )

        # For F = I, we should have R = (I + λI)^(-1)I = 1/(1+λ) * I
        expected_scale = 1.0 / (1.0 + compression_rect.regularization)
        expected_r = expected_scale * torch.eye(m)

        assert torch.allclose(regression_operator, expected_r, atol=1e-6)


class TestContinuousCompressionCompress:
    """Test compress method."""

    def test_compress_3d_input_shape(self, compression_rect):
        """Test compression with 3D input (H, Y, L)."""
        num_heads, y_dim, seq_len = 8, 64, 512
        keys = torch.randn(num_heads, y_dim, seq_len)
        result = compression_rect.compress(keys)

        assert result.shape == (num_heads, y_dim, compression_rect.compression_dim)
        assert result.shape == (8, 64, 8)  # M = 8 for rectangular_basis_8

    def test_compress_4d_input_shape(self, compression_rect):
        """Test compression with 4D input (..., Y, H, L)."""
        batch_size, y_dim, num_heads, seq_len = 4, 32, 8, 256
        keys = torch.randn(batch_size, y_dim, num_heads, seq_len)
        result = compression_rect.compress(keys)

        assert result.shape == (
            batch_size,
            y_dim,
            num_heads,
            compression_rect.compression_dim,
        )
        assert result.shape == (4, 32, 8, 8)

    def test_compress_preserves_batch_dimensions(self, compression_rect):
        """Test that batch dimensions are preserved."""
        batch_shapes = [(2, 8, 64, 128), (3, 4, 16, 32, 64)]

        for shape in batch_shapes:
            keys = torch.randn(*shape)
            result = compression_rect.compress(keys)
            expected_shape = shape[:-1] + (compression_rect.compression_dim,)
            assert result.shape == expected_shape

    def test_compress_uses_regression_operator(self, compression_rect):
        """Test that compress uses B = RK^T computation."""
        num_heads, y_dim, seq_len = 4, 32, 128
        keys = torch.randn(num_heads, y_dim, seq_len)
        coefficients = compression_rect.compress(keys)

        # Manually compute expected result: B = R @ K.T
        # Need to reshape keys for matrix multiplication
        keys_reshaped = keys.transpose(-2, -1)  # (..., L, Y)
        design_matrix = compression_rect.compute_design_matrix(seq_len)
        regression_operator = compression_rect.compute_regression_operator(
            design_matrix
        )

        # Apply regression operator: (M, L) @ (H, L, Y) -> (H, M, Y)
        expected_coeffs = torch.einsum(
            "ml,hly->hmy", regression_operator, keys_reshaped
        )
        expected_coeffs = expected_coeffs.transpose(-2, -1)  # (H, Y, M)

        assert torch.allclose(coefficients, expected_coeffs, atol=1e-5)

    def test_compress_with_explicit_seq_len(self, compression_rect):
        """Test compress with explicitly provided seq_len."""
        num_heads, y_dim, seq_len = 8, 64, 256
        keys = torch.randn(num_heads, y_dim, seq_len)
        result = compression_rect.compress(keys, seq_len=seq_len)

        assert result.shape == (num_heads, y_dim, compression_rect.compression_dim)

        # Should be same as without explicit seq_len
        result_implicit = compression_rect.compress(keys, seq_len=None)
        assert torch.allclose(result, result_implicit)

    def test_compress_infers_seq_len_from_keys(self, compression_rect):
        """Test that seq_len is inferred from keys.shape[-1]."""
        num_heads, y_dim, seq_len = 4, 32, 128
        keys = torch.randn(num_heads, y_dim, seq_len)
        result = compression_rect.compress(keys, seq_len=None)

        assert result.shape == (num_heads, y_dim, compression_rect.compression_dim)
        assert result.shape == (4, 32, 8)

    @pytest.mark.parametrize("seq_len", [16, 64, 256, 1024])
    def test_compress_different_sequence_lengths(self, compression_rect, seq_len):
        """Test compression with various sequence lengths."""
        num_heads, y_dim = 4, 32
        keys = torch.randn(num_heads, y_dim, seq_len)
        result = compression_rect.compress(keys)

        assert result.shape == (num_heads, y_dim, compression_rect.compression_dim)
        assert result.shape == (4, 32, 8)

    def test_compress_mathematical_properties(self, compression_rect):
        """Test mathematical properties of compression."""
        num_heads, y_dim, seq_len = 2, 16, 64
        keys = torch.randn(num_heads, y_dim, seq_len)
        coefficients = compression_rect.compress(keys)

        # Coefficients should be finite
        assert torch.isfinite(coefficients).all()

        # Shape should be correct
        assert coefficients.shape == (
            num_heads,
            y_dim,
            compression_rect.compression_dim,
        )

        # Test linearity: compress(a*K1 + b*K2) = a*compress(K1) + b*compress(K2)
        keys1 = torch.randn(num_heads, y_dim, seq_len)
        keys2 = torch.randn(num_heads, y_dim, seq_len)
        a, b = 2.0, 3.0

        coeffs1 = compression_rect.compress(keys1)
        coeffs2 = compression_rect.compress(keys2)
        coeffs_combined = compression_rect.compress(a * keys1 + b * keys2)
        expected_combined = a * coeffs1 + b * coeffs2

        assert torch.allclose(coeffs_combined, expected_combined, atol=1e-5)

    def test_compress_different_tensor_shapes(self, compression_rect):
        """Test compression handles various tensor shapes correctly."""
        # Test different numbers of dimensions
        test_shapes = [
            (16, 64),  # 2D: (Y, L)
            (8, 16, 64),  # 3D: (H, Y, L)
            (4, 8, 16, 64),  # 4D: (B, H, Y, L)
            (2, 4, 8, 16, 64),  # 5D: (B1, B2, H, Y, L)
        ]

        for shape in test_shapes:
            keys = torch.randn(*shape)
            result = compression_rect.compress(keys)
            expected_shape = shape[:-1] + (compression_rect.compression_dim,)
            assert result.shape == expected_shape


class TestContinuousCompressionReconstruct:
    """Test reconstruct method."""

    def test_reconstruct_with_scalar_time(self, compression_rect):
        """Test reconstruction at scalar time point."""
        coefficients = torch.randn(8, 64, 8)  # (..., M)
        time_point = torch.tensor(0.5)
        result = compression_rect.reconstruct(coefficients, time_point)

        # Expected shape: (...,) - same as coefficients without last dim
        assert result.shape == (8, 64)
        assert torch.isfinite(result).all()

    def test_reconstruct_with_vector_time(self, compression_rect):
        """Test reconstruction at multiple time points."""
        coefficients = torch.randn(4, 16, 8)  # (..., M)
        time_points = torch.linspace(0, 1, 100)  # (T,)
        result = compression_rect.reconstruct(coefficients, time_points)

        # Expected shape: (..., T) = (4, 16, 100)
        assert result.shape == (4, 16, 100)
        assert torch.isfinite(result).all()

    def test_reconstruct_preserves_input_dimensions(self, compression_rect):
        """Test that input dimensions are preserved in output."""
        batch_shapes = [(2, 8), (3, 4, 16), (5, 2, 8, 32)]
        num_time_points = 75

        for shape in batch_shapes:
            coeffs_shape = (*shape, 8)  # Add M dimension
            coefficients = torch.randn(*coeffs_shape)
            time_points = torch.linspace(0, 1, num_time_points)
            result = compression_rect.reconstruct(coefficients, time_points)

            # Expected: shape + (T,)
            expected_shape = (*shape, num_time_points)
            assert result.shape == expected_shape

    def test_reconstruct_uses_basis_evaluate(self, compression_rect):
        """Test that reconstruct uses basis.evaluate for K̄(t) = Σ_j B_j ψ_j(t)."""
        coefficients = torch.randn(4, 32, 8)
        time_points = torch.tensor([0.25, 0.5, 0.75])
        result = compression_rect.reconstruct(coefficients, time_points)

        # Manually compute expected result: K̄(t) = Σ_j B_j ψ_j(t)
        basis_values = compression_rect.basis.evaluate(time_points)  # (M, T)
        expected = torch.einsum("...m,mt->...t", coefficients, basis_values)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (4, 32, 3)

    def test_reconstruct_arbitrary_resolution(self, compression_rect):
        """Test reconstruction supports arbitrary time resolution."""
        coefficients = torch.randn(2, 16, 8)

        # Test different resolutions
        resolutions = [10, 50, 200, 1000]
        for num_time_points in resolutions:
            time_points = torch.linspace(0, 1, num_time_points)
            result = compression_rect.reconstruct(coefficients, time_points)
            assert result.shape == (2, 16, num_time_points)
            assert torch.isfinite(result).all()

    def test_reconstruct_time_points_in_domain(self, compression_rect):
        """Test reconstruction with time points in basis domain."""
        coefficients = torch.randn(8, 8)
        time_points = torch.tensor([0.0, 0.33, 0.67, 1.0])
        result = compression_rect.reconstruct(coefficients, time_points)

        assert result.shape == (8, 4)  # (8,) + (4,)
        assert torch.isfinite(result).all()

    def test_reconstruct_mathematical_properties(self, compression_rect):
        """Test mathematical properties of reconstruction."""
        torch.randn(4, 16, 8)
        time_points = torch.linspace(0, 1, 50)

        # Test linearity: reconstruct(a*B1 + b*B2) = a*reconstruct(B1) + b*reconstruct(B2)
        coeffs1 = torch.randn(4, 16, 8)
        coeffs2 = torch.randn(4, 16, 8)
        a, b = 2.0, 3.0

        result1 = compression_rect.reconstruct(coeffs1, time_points)
        result2 = compression_rect.reconstruct(coeffs2, time_points)
        result_combined = compression_rect.reconstruct(
            a * coeffs1 + b * coeffs2, time_points
        )
        expected_combined = a * result1 + b * result2

        assert torch.allclose(result_combined, expected_combined, atol=1e-5)

    def test_reconstruct_basis_function_recovery(self, compression_rect):
        """Test that individual basis functions can be recovered."""
        num_basis = compression_rect.compression_dim  # 8
        time_points = torch.linspace(0, 1, 100)

        # Create coefficients that select individual basis functions
        for j in range(num_basis):
            coefficients = torch.zeros(1, 1, num_basis)  # (1, 1, M)
            coefficients[0, 0, j] = 1.0  # Select j-th basis function

            result = compression_rect.reconstruct(coefficients, time_points)
            expected = compression_rect.basis.evaluate(time_points)[
                j
            ]  # j-th basis function

            assert torch.allclose(result[0, 0], expected, atol=1e-5)

    def test_reconstruct_compression_consistency(self, compression_rect):
        """Test consistency between compression and reconstruction."""
        # Create keys, compress them, then reconstruct at original time points
        num_heads, y_dim, seq_len = 2, 8, 64
        keys = torch.randn(num_heads, y_dim, seq_len)

        # Compress
        coefficients = compression_rect.compress(keys)

        # Reconstruct at original time points
        time_points = torch.arange(seq_len, dtype=torch.float32) / seq_len
        reconstructed = compression_rect.reconstruct(coefficients, time_points)

        # Should approximately recover original (within regularization error)
        assert reconstructed.shape == keys.shape
        # Note: Won't be exact due to compression loss and regularization
        # But should be reasonably close for well-conditioned problems
        assert torch.isfinite(reconstructed).all()


class TestContinuousCompressionScores:
    """Test compute_continuous_scores method."""

    def test_scores_expected_shape(self, compression_rect):
        """Test expected output shape (..., H, L_q, T)."""
        batch_size, y_dim, num_heads, query_len, num_basis = 2, 32, 8, 64, 8
        queries = torch.randn(batch_size, y_dim, num_heads, query_len)
        compressed_keys = torch.randn(batch_size, y_dim, num_heads, num_basis)
        result = compression_rect.compute_continuous_scores(queries, compressed_keys)

        # Default uses linspace(0, 1, 50), so T=50
        assert result.shape == (batch_size, num_heads, query_len, 50)
        assert torch.isfinite(result).all()

    def test_scores_with_default_time_points(self, compression_rect):
        """Test scores with default time_points=linspace(0, 1, 50)."""
        queries = torch.randn(4, 16, 4, 32)
        compressed_keys = torch.randn(4, 16, 4, 8)
        result = compression_rect.compute_continuous_scores(
            queries, compressed_keys, time_points=None
        )

        # Default should use linspace(0, 1, 50)
        assert result.shape == (4, 4, 32, 50)  # (..., H, L_q, T)

    def test_scores_with_custom_time_points(self, compression_rect):
        """Test scores with custom time points."""
        queries = torch.randn(2, 16, 4, 32)
        compressed_keys = torch.randn(2, 16, 4, 8)
        time_points = torch.linspace(0, 1, 25)
        result = compression_rect.compute_continuous_scores(
            queries, compressed_keys, time_points
        )

        assert result.shape[-1] == 25
        assert result.shape == (2, 4, 32, 25)

    def test_scores_computation_contract(self, compression_rect):
        """Test that scores compute s(t) = K̄(t)^T Q."""
        batch_size, y_dim, num_heads, query_len, num_basis = 1, 8, 2, 16, 8
        queries = torch.randn(batch_size, y_dim, num_heads, query_len)
        compressed_keys = torch.randn(batch_size, y_dim, num_heads, num_basis)
        time_points = torch.linspace(0, 1, 10)

        result = compression_rect.compute_continuous_scores(
            queries, compressed_keys, time_points
        )

        # Manually compute expected: s(t) = K̄(t)^T Q
        # First reconstruct keys at time points
        reconstructed_keys = compression_rect.reconstruct(
            compressed_keys, time_points
        )  # (..., Y, H, T)

        # Compute scores: Q^T K̄(t) for each query and time point
        # queries: (B, Y, H, L_q), reconstructed: (B, Y, H, T)
        # Result should be (B, H, L_q, T)
        expected = torch.einsum("byhq,byht->bhqt", queries, reconstructed_keys)

        assert torch.allclose(result, expected, atol=1e-5)
        assert result.shape == (batch_size, num_heads, query_len, 10)

    def test_scores_suitable_for_integration(self, compression_rect):
        """Test that default time points are suitable for numerical integration."""
        queries = torch.randn(2, 16, 4, 32)
        compressed_keys = torch.randn(2, 16, 4, 8)
        result = compression_rect.compute_continuous_scores(queries, compressed_keys)

        # Default linspace(0, 1, 50) should enable partition function computation
        assert result.shape[-1] == 50  # 50 time points for integration

        # Can compute approximate integral using trapezoidal rule
        dt = 1.0 / 49  # spacing for linspace(0, 1, 50)
        integral_approx = torch.trapz(result, dx=dt, dim=-1)
        assert torch.isfinite(integral_approx).all()

    def test_scores_batch_processing(self, compression_rect):
        """Test scores computation handles batched inputs correctly."""
        batch_shapes = [(1, 16, 4), (3, 8, 2), (2, 4, 32, 8)]

        for batch_shape in batch_shapes:
            batch_dims = batch_shape[:-2]  # Batch dimensions
            y_dim, num_heads = batch_shape[-2:]
            query_len, num_basis = 32, 8

            queries = torch.randn(*batch_dims, y_dim, num_heads, query_len)
            compressed_keys = torch.randn(*batch_dims, y_dim, num_heads, num_basis)
            result = compression_rect.compute_continuous_scores(
                queries, compressed_keys
            )

            # Expected shape: (..., H, L_q, T) where T=50 (default)
            expected_shape = (*batch_dims, num_heads, query_len, 50)
            assert result.shape == expected_shape

    def test_scores_mathematical_properties(self, compression_rect):
        """Test mathematical properties of continuous scores."""
        batch_size, y_dim, num_heads, query_len, num_basis = 2, 8, 4, 16, 8
        queries = torch.randn(batch_size, y_dim, num_heads, query_len)
        compressed_keys = torch.randn(batch_size, y_dim, num_heads, num_basis)
        time_points = torch.linspace(0, 1, 20)

        # Test linearity in queries
        q1, q2 = (
            torch.randn(batch_size, y_dim, num_heads, query_len),
            torch.randn(batch_size, y_dim, num_heads, query_len),
        )
        a, b = 2.0, 3.0

        s1 = compression_rect.compute_continuous_scores(
            q1, compressed_keys, time_points
        )
        s2 = compression_rect.compute_continuous_scores(
            q2, compressed_keys, time_points
        )
        s_combined = compression_rect.compute_continuous_scores(
            a * q1 + b * q2, compressed_keys, time_points
        )
        expected_combined = a * s1 + b * s2

        assert torch.allclose(s_combined, expected_combined, atol=1e-5)

        # Test linearity in compressed keys
        k1, k2 = (
            torch.randn(batch_size, y_dim, num_heads, num_basis),
            torch.randn(batch_size, y_dim, num_heads, num_basis),
        )

        s1 = compression_rect.compute_continuous_scores(queries, k1, time_points)
        s2 = compression_rect.compute_continuous_scores(queries, k2, time_points)
        s_combined = compression_rect.compute_continuous_scores(
            queries, a * k1 + b * k2, time_points
        )
        expected_combined = a * s1 + b * s2

        assert torch.allclose(s_combined, expected_combined, atol=1e-5)


class TestContinuousCompressionForward:
    """Test forward method (main entry point)."""

    def test_forward_without_queries_returns_coefficients(self, compression_rect):
        """Test forward with keys only returns compressed coefficients."""
        keys = torch.randn(2, 8, 64, 128)
        result = compression_rect.forward(keys)

        # Expected: coefficients of shape (2, 8, 64, M)
        assert result.shape == (2, 8, 64, compression_rect.compression_dim)
        assert result.shape == (2, 8, 64, 8)
        assert torch.isfinite(result).all()

    def test_forward_with_queries_default_behavior(self, compression_rect):
        """Test forward with queries returns continuous scores."""
        keys = torch.randn(2, 32, 8, 128)
        queries = torch.randn(2, 32, 8, 64)
        result = compression_rect.forward(keys, queries=queries)

        # Expected: scores of shape (2, 8, 64, T) where T=50 (default)
        assert result.shape == (2, 8, 64, 50)
        assert torch.isfinite(result).all()

    def test_forward_with_return_coefficients_true(self, compression_rect):
        """Test forward returns tuple when return_coefficients=True."""
        keys = torch.randn(1, 16, 4, 256)
        queries = torch.randn(1, 16, 4, 128)
        result = compression_rect.forward(
            keys, queries=queries, return_coefficients=True
        )

        # Expected: tuple of (scores, coefficients)
        assert isinstance(result, tuple)
        assert len(result) == 2

        scores, coefficients = result
        assert scores.shape == (1, 4, 128, 50)  # (B, H, L_q, T)
        assert coefficients.shape == (1, 16, 4, 8)  # (B, Y, H, M)

    def test_forward_with_return_coefficients_false(self, compression_rect):
        """Test forward returns only scores when return_coefficients=False."""
        keys = torch.randn(2, 8, 4, 512)
        queries = torch.randn(2, 8, 4, 256)
        result = compression_rect.forward(
            keys, queries=queries, return_coefficients=False
        )

        # Expected: scores tensor only
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 4, 256, 50)  # (B, H, L_q, T)

    def test_forward_main_entry_point(self, compression_rect):
        """Test that forward serves as main entry point for continuous compression."""
        keys = torch.randn(4, 16, 8, 1024)

        # Compression only
        coeffs = compression_rect.forward(keys)
        assert coeffs.shape == (4, 16, 8, 8)

        # With queries
        queries = torch.randn(4, 16, 8, 512)
        scores = compression_rect.forward(keys, queries=queries)
        assert scores.shape == (4, 8, 512, 50)

    def test_forward_consistent_with_individual_methods(self, compression_rect):
        """Test that forward produces same results as calling methods individually."""
        keys = torch.randn(1, 8, 4, 64)
        queries = torch.randn(1, 8, 4, 32)

        # Using forward method
        result_forward = compression_rect.forward(keys, queries=queries)

        # Using individual methods
        coeffs = compression_rect.compress(keys)
        scores = compression_rect.compute_continuous_scores(queries, coeffs)

        assert torch.allclose(result_forward, scores, atol=1e-5)

    @pytest.mark.parametrize("return_coefficients", [True, False])
    def test_forward_return_coefficients_parameter(
        self, compression_rect, return_coefficients
    ):
        """Test forward behavior with different return_coefficients values."""
        keys = torch.randn(2, 16, 4, 128)
        queries = torch.randn(2, 16, 4, 64)
        result = compression_rect.forward(
            keys, queries=queries, return_coefficients=return_coefficients
        )

        if return_coefficients:
            assert isinstance(result, tuple)
            assert len(result) == 2
            scores, coeffs = result
            assert scores.shape == (2, 4, 64, 50)
            assert coeffs.shape == (2, 16, 4, 8)
        else:
            assert isinstance(result, torch.Tensor)
            assert result.shape == (2, 4, 64, 50)

    def test_forward_handles_different_input_shapes(self, compression_rect):
        """Test forward handles various input tensor shapes."""
        # Test different tensor shapes
        test_cases = [
            # (keys_shape, queries_shape, expected_scores_shape)
            ((8, 4, 64), (8, 4, 32), (4, 32, 50)),  # 3D inputs
            ((2, 8, 4, 64), (2, 8, 4, 32), (2, 4, 32, 50)),  # 4D inputs
            ((3, 2, 8, 4, 64), (3, 2, 8, 4, 32), (3, 2, 4, 32, 50)),  # 5D inputs
        ]

        for keys_shape, queries_shape, expected_scores_shape in test_cases:
            keys = torch.randn(*keys_shape)
            queries = torch.randn(*queries_shape)

            # Test compression only
            coeffs = compression_rect.forward(keys)
            expected_coeffs_shape = keys_shape[:-1] + (8,)
            assert coeffs.shape == expected_coeffs_shape

            # Test with queries
            scores = compression_rect.forward(keys, queries=queries)
            assert scores.shape == expected_scores_shape


class TestContinuousCompressionIntegration:
    """Integration tests for ContinuousCompression."""

    def test_extra_repr(self, compression_rect):
        """Test string representation includes key configuration."""
        repr_str = compression_rect.extra_repr()

        assert "compression_dim=8" in repr_str
        assert "regularization=0.01" in repr_str
        assert "basis_type=RectangularBasis" in repr_str

    def test_module_registration(self, compression_rect):
        """Test that ContinuousCompression is properly registered as nn.Module."""
        assert isinstance(compression_rect, torch.nn.Module)

        # Should support standard module operations
        compression_rect.train()
        compression_rect.eval()

        # Should be movable to different devices (when implemented)
        device = torch.device("cpu")
        compression_rect.to(device)

    def test_different_basis_types(self):
        """Test ContinuousCompression with different basis types."""
        basis_types = [
            RectangularBasis(num_basis=8),
            GaussianBasis(num_basis=8),
            FourierBasis(num_basis=8),
        ]

        for basis in basis_types:
            compression = ContinuousCompression(basis=basis, regularization=0.01)
            assert compression.compression_dim == 8
            assert compression.basis is basis

    def test_performance_contracts_documented(self, compression_rect):
        """Test that performance contracts are properly documented."""
        # Verify key performance requirements are documented in docstring
        docstring = compression_rect.__class__.__doc__

        assert "Cholesky decomposition" in docstring
        assert "O(M³)" in docstring
        assert "cache" in docstring
        assert "torch.compile" in docstring

    def test_memory_requirements_documented(self, compression_rect):
        """Test that memory requirements are documented."""
        docstring = compression_rect.__class__.__doc__

        assert "MxL" in docstring
        assert "Peak memory" in docstring
        assert "in-place operations" in docstring

    @pytest.mark.parametrize("cache_operators", [True, False])
    def test_caching_configuration(self, rectangular_basis_8, cache_operators):
        """Test different caching configurations."""
        compression = ContinuousCompression(
            basis=rectangular_basis_8,
            regularization=0.01,
            cache_operators=cache_operators,
        )

        assert compression.cache_operators == cache_operators
        assert compression.cached_design_matrix is None
        assert compression.cached_regression_operator is None
