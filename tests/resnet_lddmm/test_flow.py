"""Tests for NeuralODEFlow."""

import torch
import pytest
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.integrators import ForwardEuler
from src.resnet_lddmm.trajectory import Trajectory


class TestNeuralODEFlowInit:
    """Tests for NeuralODEFlow initialization."""

    def test_creation(self):
        """Verify NeuralODEFlow can be instantiated."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()

        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)
        assert isinstance(flow, torch.nn.Module)

    def test_default_num_steps(self):
        """Verify default num_steps is 10."""
        field = TimeVaryingField(num_blocks=10, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()

        flow = NeuralODEFlow(field, direct, inverse)
        assert flow.num_steps == 10

    def test_custom_num_steps(self):
        """Verify custom num_steps is stored."""
        field = TimeVaryingField(num_blocks=7, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()

        flow = NeuralODEFlow(field, direct, inverse, num_steps=7)
        assert flow.num_steps == 7


class TestNeuralODEFlowForward:
    """Tests for forward() method."""

    def test_forward_returns_trajectory(self):
        """Verify forward() returns a Trajectory."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        q0 = torch.randn(2, 10, 3)
        result = flow(q0)
        assert isinstance(result, Trajectory)

    def test_forward_shape(self):
        """Verify forward() produces correct shapes."""
        num_steps = 7
        B, N = 3, 15
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)

        q0 = torch.randn(B, N, 3)
        traj = flow(q0)

        assert traj.points.shape == (num_steps + 1, B, N, 3)
        assert traj.velocities.shape == (num_steps, B, N, 3)
        assert traj.dt == 1.0 / num_steps

    def test_forward_with_code(self):
        """Verify forward() accepts code argument."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        q0 = torch.randn(2, 10, 3)
        code = torch.randn(2, 128)

        traj = flow(q0, code=code)
        assert isinstance(traj, Trajectory)

    def test_forward_without_code(self):
        """Verify forward() works with code=None (default)."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        q0 = torch.randn(2, 10, 3)
        traj = flow(q0)
        assert isinstance(traj, Trajectory)

    def test_forward_zero_init_gives_identity(self):
        """Verify zero-init field gives identity transformation."""
        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)

        q0 = torch.randn(2, 10, 3)
        traj = flow(q0)

        # Zero-init field should produce zero velocities
        assert torch.allclose(traj.velocities, torch.zeros_like(traj.velocities), atol=1e-6)
        # End should equal start
        assert torch.allclose(traj.end, q0, atol=1e-6)


class TestNeuralODEFlowInverse:
    """Tests for inverse() method."""

    def test_inverse_returns_trajectory(self):
        """Verify inverse() returns a Trajectory."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        qT = torch.randn(2, 10, 3)
        result = flow.inverse(qT)
        assert isinstance(result, Trajectory)

    def test_inverse_shape(self):
        """Verify inverse() produces correct shapes."""
        num_steps = 7
        B, N = 3, 15
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)

        qT = torch.randn(B, N, 3)
        traj = flow.inverse(qT)

        assert traj.points.shape == (num_steps + 1, B, N, 3)
        assert traj.velocities.shape == (num_steps, B, N, 3)

    def test_inverse_with_code(self):
        """Verify inverse() accepts code argument."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        qT = torch.randn(2, 10, 3)
        code = torch.randn(2, 128)

        traj = flow.inverse(qT, code=code)
        assert isinstance(traj, Trajectory)

    def test_inverse_zero_init_gives_identity(self):
        """Verify zero-init field gives identity for inverse."""
        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)

        qT = torch.randn(2, 10, 3)
        traj = flow.inverse(qT)

        # Zero-init gives identity
        assert torch.allclose(traj.end, qT, atol=1e-6)


class TestNeuralODEFlowBidirectional:
    """Tests combining forward and inverse."""

    def test_forward_and_inverse_have_same_field(self):
        """Verify forward and inverse use the same field."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        # Both should use the same field instance
        assert flow.field is field

    def test_forward_inverse_with_zero_init_is_circular(self):
        """Verify forward → inverse on zero-init field returns to start."""
        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)

        q0 = torch.randn(2, 10, 3)

        # Forward with zero-init: q0 → q0 (identity)
        fwd_traj = flow(q0)
        qT = fwd_traj.end

        # Inverse: qT → qT (identity)
        inv_traj = flow.inverse(qT)
        q_recovered = inv_traj.end

        # Should recover q0 exactly
        assert torch.allclose(q_recovered, q0, atol=1e-6)


class TestNeuralODEFlowDifferentIntegrators:
    """Tests with different integrator configurations."""

    def test_same_integrator_for_both(self):
        """Verify flow works when direct == inverse."""
        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        integrator = ForwardEuler()
        flow = NeuralODEFlow(field, integrator, integrator, num_steps=num_steps)

        q0 = torch.randn(1, 5, 3)
        traj_fwd = flow(q0)
        traj_inv = flow.inverse(q0)

        # With same integrator and zero-init field, both should be identity
        assert torch.allclose(traj_fwd.end, q0, atol=1e-6)
        assert torch.allclose(traj_inv.end, q0, atol=1e-6)


class TestNeuralODEFlowGradients:
    """Tests verifying gradient flow through flow."""

    def test_gradients_flow_through_forward(self):
        """Verify gradients flow through forward pass."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        q0 = torch.randn(1, 5, 3, requires_grad=True)
        traj = flow(q0)
        loss = traj.end.sum()
        loss.backward()

        assert q0.grad is not None

    def test_field_parameters_have_gradients(self):
        """Verify field parameters receive gradients."""
        field = TimeVaryingField(num_blocks=5, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        flow = NeuralODEFlow(field, direct, inverse, num_steps=5)

        q0 = torch.randn(1, 5, 3)
        traj = flow(q0)
        loss = traj.kinetic_energy()
        loss.backward()

        # Check that field parameters have gradients
        for block in field.blocks:
            for param in block.parameters():
                assert param.grad is not None


class TestNeuralODEFlowNumStepsEffect:
    """Tests showing effect of num_steps parameter."""

    def test_num_steps_affects_trajectory_length(self):
        """Verify num_steps controls trajectory length."""
        field = TimeVaryingField(num_blocks=20, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        q0 = torch.randn(1, 5, 3)

        for num_steps in [1, 5, 10, 20]:
            flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)
            traj = flow(q0)

            assert traj.points.shape[0] == num_steps + 1
            assert traj.velocities.shape[0] == num_steps

    def test_dt_scales_with_num_steps(self):
        """Verify dt = 1/num_steps in trajectory."""
        field = TimeVaryingField(num_blocks=20, width=64)
        direct = ForwardEuler()
        inverse = ForwardEuler()
        q0 = torch.randn(1, 5, 3)

        for num_steps in [1, 5, 10, 20]:
            flow = NeuralODEFlow(field, direct, inverse, num_steps=num_steps)
            traj = flow(q0)

            expected_dt = 1.0 / num_steps
            assert abs(traj.dt - expected_dt) < 1e-6
