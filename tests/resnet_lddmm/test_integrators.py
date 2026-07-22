"""Tests for ForwardEuler integrator."""

import torch
import pytest
from src.resnet_lddmm.integrators import Integrator, ForwardEuler
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.trajectory import Trajectory


class TestIntegratorABC:
    """Tests for Integrator abstract base class."""

    def test_integrator_is_abstract(self):
        """Verify Integrator cannot be instantiated."""
        with pytest.raises(TypeError):
            Integrator()


class TestForwardEulerBasics:
    """Basic tests for ForwardEuler integrator."""

    def test_returns_trajectory(self):
        """Verify ForwardEuler.integrate returns a Trajectory."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=5, width=64)
        q0 = torch.randn(2, 10, 3)
        code = None

        result = integrator.integrate(q0, field, code, num_steps=5)
        assert isinstance(result, Trajectory)

    def test_points_shape_k_plus_1(self):
        """Verify points has K+1 snapshots."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=7, width=64)
        q0 = torch.randn(3, 20, 3)
        num_steps = 7

        traj = integrator.integrate(q0, field, None, num_steps)
        assert traj.points.shape == (num_steps + 1, 3, 20, 3)

    def test_velocities_shape_k(self):
        """Verify velocities has K snapshots."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=5, width=64)
        q0 = torch.randn(2, 15, 3)
        num_steps = 5

        traj = integrator.integrate(q0, field, None, num_steps)
        assert traj.velocities.shape == (num_steps, 2, 15, 3)

    def test_first_point_is_q0(self):
        """Verify points[0] equals the initial position."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=3, width=64)
        q0 = torch.randn(2, 8, 3)

        traj = integrator.integrate(q0, field, None, num_steps=3)
        assert torch.equal(traj.points[0], q0)


class TestForwardEulerZeroInitField:
    """Tests verifying identity property with zero-initialized field."""

    def test_zero_init_field_returns_identity(self):
        """Verify flow(q0).end == q0 exactly when field is zero-initialized.

        Zero-initialized fields output [0,0,0] velocity, so q should never change.
        """
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=10, width=256)
        # TimeVaryingField blocks are zero-initialized (proj.weight = 0)
        q0 = torch.randn(2, 20, 3)

        traj = integrator.integrate(q0, field, None, num_steps=10)
        # All velocities should be zero
        assert torch.allclose(traj.velocities, torch.zeros_like(traj.velocities), atol=1e-6)
        # End should equal start
        assert torch.allclose(traj.end, q0, atol=1e-6)

    def test_identity_property_various_steps(self):
        """Verify identity holds for different num_steps."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=20, width=256)
        q0 = torch.randn(1, 10, 3)

        for num_steps in [1, 5, 10, 20]:
            traj = integrator.integrate(q0, field, None, num_steps=num_steps)
            assert torch.allclose(traj.end, q0, atol=1e-6)


class TestForwardEulerSingleStepExactness:
    """Test single-step exactness: q1 == q0 + (1/K)*v."""

    def test_single_step_update(self):
        """Verify one step of Euler: q1 = q0 + dt*v with dt = 1/num_steps."""
        torch.manual_seed(42)
        num_steps = 10
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        q0 = torch.zeros(1, 5, 3)

        # Get velocity at step 0
        v0 = field(q0, step=0, code=None)

        # Integrate
        traj = integrator.integrate(q0, field, None, num_steps)

        # Expected: q1 = q0 + (1/10) * v0
        dt = 1.0 / num_steps
        expected_q1 = q0 + dt * v0

        assert torch.allclose(traj.points[1], expected_q1, atol=1e-6)


class TestForwardEulerGradientFlow:
    """Tests verifying gradients flow through all steps."""

    def test_gradients_flow_through_steps(self):
        """Verify gradients reach all blocks during backprop."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=5, width=64)
        q0 = torch.randn(1, 3, 3, requires_grad=True)

        traj = integrator.integrate(q0, field, None, num_steps=5)
        loss = traj.end.sum()
        loss.backward()

        assert q0.grad is not None
        # Check that blocks have gradients
        for block in field.blocks:
            for param in block.parameters():
                assert param.grad is not None
                assert torch.all(torch.isfinite(param.grad))

    def test_first_block_has_gradient(self):
        """Verify gradients reach the first block."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=5, width=64)
        q0 = torch.randn(1, 5, 3, requires_grad=True)

        traj = integrator.integrate(q0, field, None, num_steps=5)
        # Use end position loss (which depends on all steps and velocities)
        loss = traj.end.pow(2).sum()
        loss.backward()

        # First block should have gradient (even if zero-initialized, gradients flow)
        first_block_grads = [p.grad for p in field.blocks[0].parameters()]
        assert all(g is not None for g in first_block_grads)


class TestForwardEulerDifferentNumSteps:
    """Tests with various numbers of integration steps."""

    def test_single_step(self):
        """Verify integration works with num_steps=1."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=1, width=64)
        q0 = torch.randn(2, 5, 3)

        traj = integrator.integrate(q0, field, None, num_steps=1)
        assert traj.points.shape == (2, 2, 5, 3)
        assert traj.velocities.shape == (1, 2, 5, 3)

    def test_many_steps(self):
        """Verify integration works with large num_steps."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=100, width=64)
        q0 = torch.randn(1, 5, 3)

        traj = integrator.integrate(q0, field, None, num_steps=100)
        assert traj.points.shape == (101, 1, 5, 3)
        assert traj.velocities.shape == (100, 1, 5, 3)

    def test_dt_scales_with_num_steps(self):
        """Verify dt = 1/num_steps."""
        integrator = ForwardEuler()
        q0 = torch.randn(1, 3, 3)

        for num_steps in [1, 5, 10, 20]:
            field = TimeVaryingField(num_blocks=num_steps, width=64)
            traj = integrator.integrate(q0, field, None, num_steps)
            expected_dt = 1.0 / num_steps
            assert abs(traj.dt - expected_dt) < 1e-6


class TestForwardEulerWithCode:
    """Tests with conditioning code."""

    def test_integrate_accepts_code(self):
        """Verify integrate accepts code argument without error."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=3, width=64)
        q0 = torch.randn(2, 5, 3)
        code = torch.randn(2, 128)  # Ignored by NoConditioning

        traj = integrator.integrate(q0, field, code, num_steps=3)
        assert isinstance(traj, Trajectory)

    def test_integrate_code_none(self):
        """Verify integrate works with code=None."""
        integrator = ForwardEuler()
        field = TimeVaryingField(num_blocks=3, width=64)
        q0 = torch.randn(2, 5, 3)

        traj = integrator.integrate(q0, field, None, num_steps=3)
        assert isinstance(traj, Trajectory)
