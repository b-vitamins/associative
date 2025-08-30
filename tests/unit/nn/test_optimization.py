"""Comprehensive tests for CCCP optimization framework."""

import pytest
import torch
from torch import Tensor

from associative.nn.modules.config import CCCPConfig
from associative.nn.modules.optimization import (
    CCCPOptimizer,
    ContinuousHopfieldEnergy,
    EnergyFunction,
    OptimizationResult,
    QuadraticConvexSolver,
)


class SimpleQuadraticEnergy(EnergyFunction):
    """Simple test energy: E(x) = -log(sum(exp(x))) + 0.5||x||²."""

    def concave_part(self, state: Tensor) -> Tensor:
        return -torch.logsumexp(state, dim=-1)

    def convex_part(self, state: Tensor) -> Tensor:
        return 0.5 * (state**2).sum(dim=-1)

    def grad_concave(self, state: Tensor) -> Tensor:
        return -torch.softmax(state, dim=-1)

    def grad_convex(self, state: Tensor) -> Tensor:
        return state


class TestEnergyFunction:
    """Test EnergyFunction abstract class and interface."""

    def test_total_energy(self):
        """Test that total energy is sum of parts."""
        energy = SimpleQuadraticEnergy()
        x = torch.randn(10)

        total = energy(x)
        concave = energy.concave_part(x)
        convex = energy.convex_part(x)

        assert torch.allclose(total, concave + convex)

    def test_batch_handling(self):
        """Test that energy functions handle batched inputs."""
        energy = SimpleQuadraticEnergy()

        # Single input
        x_single = torch.randn(5)
        e_single = energy(x_single)
        assert e_single.shape == ()

        # Batch input
        x_batch = torch.randn(3, 5)
        e_batch = energy(x_batch)
        assert e_batch.shape == (3,)

        # Results should match for same inputs
        assert torch.allclose(e_batch[0], energy(x_batch[0]))


class TestCCCPOptimizer:
    """Test CCCP optimizer implementation."""

    @pytest.fixture
    def optimizer(self):
        """Create standard CCCP optimizer."""
        return CCCPOptimizer(max_iterations=100, tolerance=1e-6, step_size=1.0)

    def test_initialization(self, optimizer):
        """Test proper initialization of CCCP optimizer."""
        expected_max_iter = 100
        expected_tolerance = 1e-6
        assert optimizer.max_iterations == expected_max_iter
        assert optimizer.tolerance == expected_tolerance
        assert optimizer.step_size == 1.0
        assert optimizer.momentum == 0.0
        assert not optimizer.track_trajectory
        assert not optimizer.use_line_search

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            CCCPOptimizer(max_iterations=0)

        with pytest.raises(ValueError, match="tolerance must be positive"):
            CCCPOptimizer(tolerance=-1e-6)

        with pytest.raises(ValueError, match="step_size must be in"):
            CCCPOptimizer(step_size=0)

        with pytest.raises(ValueError, match="step_size must be in"):
            CCCPOptimizer(step_size=1.5)

        with pytest.raises(ValueError, match="momentum must be in"):
            CCCPOptimizer(momentum=-0.1)

        with pytest.raises(ValueError, match="momentum must be in"):
            CCCPOptimizer(momentum=1.0)

    def test_single_step(self, optimizer):
        """Test single CCCP step."""
        energy = SimpleQuadraticEnergy()
        x_current = torch.randn(5)

        x_next = optimizer.step(energy, x_current)

        # Should return updated point
        assert x_next.shape == x_current.shape
        assert not torch.allclose(x_next, x_current)

        # For this energy, analytical solution is x_next = -grad_concave(x_current)
        expected = -energy.grad_concave(x_current)
        assert torch.allclose(x_next, expected, atol=1e-5)

    def test_step_with_momentum(self):
        """Test CCCP step with momentum."""
        optimizer = CCCPOptimizer(momentum=0.5)
        energy = SimpleQuadraticEnergy()

        x_prev = torch.randn(5)
        x_current = torch.randn(5)

        x_next = optimizer.step(energy, x_current, x_prev)

        # With momentum, should differ from standard update
        x_next_no_momentum = optimizer.step(energy, x_current, None)
        assert not torch.allclose(x_next, x_next_no_momentum)

    def test_minimize_convergence(self, optimizer):
        """Test that minimize converges to a fixed point."""
        energy = SimpleQuadraticEnergy()
        x0 = torch.randn(5)

        result = optimizer.minimize(energy, x0)

        assert isinstance(result, OptimizationResult)
        assert result.optimal_point.shape == x0.shape
        assert result.converged
        assert result.num_iterations < optimizer.max_iterations

        # At convergence, gradient condition should be satisfied
        # ∇convex(x*) = -∇concave(x*)
        x_opt = result.optimal_point
        grad_convex = energy.grad_convex(x_opt)
        grad_concave = energy.grad_concave(x_opt)
        assert torch.allclose(grad_convex, -grad_concave, atol=1e-4)

    def test_minimize_with_trajectory(self):
        """Test trajectory tracking during optimization."""
        optimizer = CCCPOptimizer(track_trajectory=True)
        energy = SimpleQuadraticEnergy()
        x0 = torch.randn(5)

        result = optimizer.minimize(energy, x0)

        assert result.trajectory is not None
        assert len(result.trajectory) == result.num_iterations + 1  # Including x0
        assert torch.allclose(result.trajectory[0], x0)
        assert torch.allclose(result.trajectory[-1], result.optimal_point)

        # Energy should decrease
        energy_decrease_tolerance = 1e-6
        assert result.energy_history is not None
        energies = torch.tensor(result.energy_history)
        assert (
            torch.diff(energies) <= energy_decrease_tolerance
        ).all()  # Non-increasing with tolerance for floating-point precision

    def test_minimize_max_iterations(self):
        """Test that optimization stops at max_iterations."""
        max_iter = 5
        optimizer = CCCPOptimizer(max_iterations=max_iter, tolerance=1e-12)
        energy = SimpleQuadraticEnergy()
        x0 = torch.randn(100)  # Large dimension, slow convergence

        result = optimizer.minimize(energy, x0)

        assert result.num_iterations == max_iter
        assert not result.converged

    def test_line_search(self):
        """Test line search functionality."""
        optimizer = CCCPOptimizer(use_line_search=True)
        energy = SimpleQuadraticEnergy()

        x_current = torch.randn(5)
        direction = -energy.grad_convex(x_current)

        alpha = optimizer._line_search(energy, x_current, direction)

        assert 0 < alpha <= 1

        # Energy should decrease in the direction
        e_current = energy(x_current)
        e_next = energy(x_current + alpha * direction)
        assert e_next < e_current


class TestQuadraticConvexSolver:
    """Test solver for quadratic convex subproblems."""

    def test_identity_quadratic(self):
        """Test solving with Q = I, b = 0."""
        grad_concave = torch.randn(5)

        x_next = QuadraticConvexSolver.solve(None, None, grad_concave)

        # For Q = I, b = 0: x_next = -grad_concave
        assert torch.allclose(x_next, -grad_concave)

    def test_general_quadratic(self):
        """Test solving with general positive definite Q."""
        # Create positive definite Q
        a = torch.randn(5, 5)
        q = a.T @ a + torch.eye(5)
        b = torch.randn(5)
        grad_concave = torch.randn(5)

        x_next = QuadraticConvexSolver.solve(q, b, grad_concave)

        # Should satisfy: Q x_next + b = -grad_concave
        residual = q @ x_next + b + grad_concave
        assert torch.allclose(residual, torch.zeros(5), atol=1e-5)

    def test_batch_solving(self):
        """Test batch solving of quadratic problems."""
        q = torch.eye(3)
        b = torch.zeros(3)
        grad_concave_batch = torch.randn(10, 3)

        x_next_batch = QuadraticConvexSolver.solve(q, b, grad_concave_batch)

        assert x_next_batch.shape == (10, 3)

        # Each should solve independently
        for i in range(10):
            x_single = QuadraticConvexSolver.solve(q, b, grad_concave_batch[i])
            assert torch.allclose(x_next_batch[i], x_single)


class TestContinuousHopfieldEnergy:
    """Test ContinuousHopfieldEnergy implementation."""

    @pytest.fixture
    def energy(self):
        """Create continuous Hopfield energy."""
        beta = 2.0
        integration_points = 100
        return ContinuousHopfieldEnergy(
            beta=beta, integration_points=integration_points
        )

    def test_initialization(self, energy):
        """Test proper initialization."""
        expected_beta = 2.0
        expected_points = 100
        assert energy.beta == expected_beta
        assert energy.integration_points == expected_points
        assert energy.memory is None

    def test_set_memory(self, energy):
        """Test setting memory function."""

        def memory_fn(t):
            # Simple linear memory
            return t.unsqueeze(-1) * torch.ones(t.shape[0], 5)

        energy.set_memory(memory_fn)
        assert energy.memory is not None

        # Test memory evaluation
        t = torch.linspace(0, 1, 10)
        mem = energy.memory(t)
        assert mem.shape == (10, 5)

    def test_energy_computation(self, energy):
        """Test energy computation with memory."""

        # Set up simple memory
        def memory_fn(t):
            return torch.ones(t.shape[0], 3)

        energy.set_memory(memory_fn)

        q = torch.randn(3)

        # Compute energy parts
        concave = energy.concave_part(q)
        convex = energy.convex_part(q)

        assert concave.shape == ()
        assert convex.shape == ()

        # Convex part should be 0.5||q||²
        expected_convex = 0.5 * (q**2).sum()
        assert torch.allclose(convex, expected_convex)

        # Total energy
        total = energy(q)
        assert torch.allclose(total, concave + convex)

    def test_gradients(self, energy):
        """Test gradient computation."""

        def memory_fn(t):
            return torch.sin(2 * torch.pi * t).unsqueeze(-1) * torch.ones(t.shape[0], 3)

        energy.set_memory(memory_fn)

        q = torch.randn(3)

        # Analytical gradients
        grad_concave = energy.grad_concave(q)
        grad_convex = energy.grad_convex(q)

        assert grad_concave.shape == q.shape
        assert grad_convex.shape == q.shape

        # grad_convex should be q
        assert torch.allclose(grad_convex, q)

        # Check via autograd
        q_auto = q.clone().requires_grad_(True)
        concave_auto = energy.concave_part(q_auto)
        grad_concave_auto = torch.autograd.grad(concave_auto, q_auto)[0]

        assert torch.allclose(grad_concave, grad_concave_auto, atol=1e-4)

    def test_gibbs_distribution(self, energy):
        """Test that gradient gives correct Gibbs expectation."""

        # Use simple discrete-like memory
        def memory_fn(t):
            # Two memory points
            midpoint = 0.5
            mask1 = t < midpoint
            mask2 = ~mask1
            mem = torch.zeros(t.shape[0], 2)
            mem[mask1, 0] = 1.0
            mem[mask2, 1] = 1.0
            return mem

        energy.set_memory(memory_fn)

        q = torch.tensor([1.0, 0.0])

        # grad_concave should be -E_p[x̄(t)]
        grad = energy.grad_concave(q)

        # For this setup, should favor first memory
        assert grad[0] < grad[1]  # Negative of expectation


class TestCCCPConfig:
    """Test CCCPConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CCCPConfig()
        default_max_iter = 100
        default_tolerance = 1e-6

        assert config.max_iterations == default_max_iter
        assert config.tolerance == default_tolerance
        assert config.step_size == 1.0
        assert config.momentum == 0.0
        assert not config.track_trajectory
        assert not config.use_line_search

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid max_iterations
        with pytest.raises(ValueError):
            CCCPConfig(max_iterations=-1)

        # Invalid tolerance
        with pytest.raises(ValueError):
            CCCPConfig(tolerance=0)

        # Invalid step_size
        with pytest.raises(ValueError):
            CCCPConfig(step_size=2.0)

        # Invalid momentum
        with pytest.raises(ValueError):
            CCCPConfig(momentum=1.5)


class TestOptimizationIntegration:
    """Integration tests for optimization components."""

    def test_cccp_with_continuous_hopfield_energy(self):
        """Test CCCP optimization of continuous Hopfield energy."""
        # Set up energy
        energy = ContinuousHopfieldEnergy(beta=1.0, integration_points=50)

        # Define oscillating memory
        def memory_fn(t):
            return torch.stack(
                [torch.sin(2 * torch.pi * t), torch.cos(2 * torch.pi * t)], dim=-1
            )

        energy.set_memory(memory_fn)

        # Set up optimizer
        optimizer = CCCPOptimizer(max_iterations=50, tolerance=1e-5)

        # Initial query
        q0 = torch.randn(2)

        # Optimize
        result = optimizer.minimize(energy, q0)

        assert result.converged

        # Result should be a fixed point
        q_opt = result.optimal_point
        update = optimizer.step(energy, q_opt)
        assert torch.allclose(q_opt, update, atol=1e-4)

    def test_different_integration_points(self):
        """Test that more integration points give better accuracy."""
        energy_coarse = ContinuousHopfieldEnergy(beta=1.0, integration_points=10)
        energy_fine = ContinuousHopfieldEnergy(beta=1.0, integration_points=1000)

        def memory_fn(t):
            return torch.exp(-((t - 0.5) ** 2) / 0.1).unsqueeze(-1) * torch.ones(
                t.shape[0], 3
            )

        energy_coarse.set_memory(memory_fn)
        energy_fine.set_memory(memory_fn)

        q = torch.ones(3)

        # Fine integration should be more accurate
        # (Can't directly test accuracy without ground truth,
        # but energies should be similar)
        e_coarse = energy_coarse(q)
        e_fine = energy_fine(q)

        # Should be close but not identical
        energy_diff_threshold = 0.1
        assert abs(e_coarse - e_fine) < energy_diff_threshold
        assert e_coarse != e_fine
