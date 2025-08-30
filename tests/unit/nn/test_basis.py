"""Comprehensive tests for basis function implementations."""

import pytest
import torch

from associative.nn.modules.basis import (
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
        assert psi[0] == 1.0  # First basis function active
        assert psi[1:].sum() == 0.0  # Others inactive

        # Test at boundary
        t = torch.tensor(0.25)  # Boundary between first and second
        psi = rectangular_basis.evaluate(t)
        # Depending on implementation, either first or second is active
        assert psi.sum() == 1.0  # Exactly one active

    def test_evaluate_batch(self, rectangular_basis):
        """Test evaluation at multiple time points."""
        t = torch.linspace(0, 1, 100)
        psi = rectangular_basis.evaluate(t)
        assert psi.shape == (4, 100)

        # Each time point should have exactly one active basis (no overlap)
        assert torch.allclose(psi.sum(dim=0), torch.ones(100))

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
        assert torch.allclose(design_matrix.sum(dim=0), torch.ones(10))

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
