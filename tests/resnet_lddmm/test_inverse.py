"""Invertibility tests for ModifiedEuler integrator (STEPS T21)."""

import pytest
import torch
from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.fields.time_varying import TimeVaryingField, StationaryField
from src.resnet_lddmm.flow import NeuralODEFlow


@pytest.fixture
def time_varying_field():
    """Create a time-varying velocity field for testing."""
    return TimeVaryingField(num_blocks=10, width=64, activation="relu")


@pytest.fixture
def stationary_field():
    """Create a stationary velocity field for testing."""
    return StationaryField(num_blocks=10, width=64, activation="relu")


@pytest.fixture
def sample_points():
    """Create sample point cloud [B=2, N=10, 3]."""
    torch.manual_seed(42)
    return torch.randn(2, 10, 3)


class TestModifiedEulerBasics:
    """Basic functionality tests for ModifiedEuler."""

    def test_modified_euler_output_shape(self, stationary_field, sample_points):
        """Test that ModifiedEuler produces correct output shape."""
        integrator = ModifiedEuler()
        trajectory = integrator.integrate(sample_points, stationary_field, None, num_steps=10)

        assert trajectory.points.shape == (11, 2, 10, 3)  # K+1 steps
        assert trajectory.velocities.shape == (10, 2, 10, 3)  # K velocities
        assert trajectory.dt == 0.1

    def test_modified_euler_deterministic(self, stationary_field, sample_points):
        """Test that ModifiedEuler is deterministic (no randomness)."""
        integrator = ModifiedEuler()

        traj1 = integrator.integrate(sample_points, stationary_field, None, num_steps=10)
        traj2 = integrator.integrate(sample_points, stationary_field, None, num_steps=10)

        assert torch.allclose(traj1.points, traj2.points, atol=1e-6)
        assert torch.allclose(traj1.velocities, traj2.velocities, atol=1e-6)

    def test_dt_ownership(self, stationary_field, sample_points):
        """Test that dt is correctly computed as 1/num_steps."""
        integrator = ModifiedEuler()

        for num_steps in [5, 10, 20, 50]:
            trajectory = integrator.integrate(sample_points, stationary_field, None, num_steps)
            expected_dt = 1.0 / num_steps
            assert abs(trajectory.dt - expected_dt) < 1e-6


class TestInvertibilityPropertyTimeVarying:
    """Property tests: φ⁻¹(φ(q)) ≈ q for time-varying fields."""

    @pytest.mark.slow
    def test_invertibility_time_varying_small(self, time_varying_field, sample_points):
        """Test that forward→inverse recovers original points (time-varying field)."""
        num_steps = 10
        flow = NeuralODEFlow(
            field=time_varying_field,
            direct=ForwardEuler(),
            inverse=ModifiedEuler(),
            num_steps=num_steps,
        )

        # Forward: q0 → q_T
        traj_fwd = flow.forward(sample_points)
        q_T = traj_fwd.end

        # Inverse: q_T → q0_recovered
        traj_bwd = flow.inverse(q_T)
        q0_recovered = traj_bwd.end

        # Check invertibility: q0_recovered ≈ q0
        error = torch.norm(q0_recovered - sample_points, p=2)
        max_error = torch.max(torch.abs(q0_recovered - sample_points))

        # Typical tolerance for 10 steps and 64-width network
        assert error < 0.5, f"Forward→inverse error {error:.4f} exceeds tolerance"
        assert max_error < 0.1, f"Max pointwise error {max_error:.4f} exceeds tolerance"

    @pytest.mark.slow
    def test_invertibility_time_varying_random_clouds(self, time_varying_field):
        """Test invertibility over multiple random clouds (time-varying field)."""
        torch.manual_seed(123)
        num_steps = 10
        flow = NeuralODEFlow(
            field=time_varying_field,
            direct=ForwardEuler(),
            inverse=ModifiedEuler(),
            num_steps=num_steps,
        )

        for trial in range(3):
            q0 = torch.randn(1, 20, 3)  # Different sizes too
            q_T = flow.forward(q0).end
            q0_recovered = flow.inverse(q_T).end

            error = torch.norm(q0_recovered - q0, p=2)
            assert error < 0.5, f"Trial {trial}: error {error:.4f}"


class TestInvertibilityPropertyStationary:
    """Property tests: φ⁻¹(φ(q)) ≈ q for stationary fields."""

    @pytest.mark.slow
    def test_invertibility_stationary_small(self, stationary_field, sample_points):
        """Test that forward→inverse recovers original points (stationary field)."""
        num_steps = 10
        flow = NeuralODEFlow(
            field=stationary_field,
            direct=ForwardEuler(),
            inverse=ModifiedEuler(),
            num_steps=num_steps,
        )

        # Forward: q0 → q_T
        traj_fwd = flow.forward(sample_points)
        q_T = traj_fwd.end

        # Inverse: q_T → q0_recovered
        traj_bwd = flow.inverse(q_T)
        q0_recovered = traj_bwd.end

        # Check invertibility: q0_recovered ≈ q0
        error = torch.norm(q0_recovered - sample_points, p=2)
        max_error = torch.max(torch.abs(q0_recovered - sample_points))

        # Typical tolerance for 10 steps and 64-width network
        assert error < 0.5, f"Forward→inverse error {error:.4f} exceeds tolerance"
        assert max_error < 0.1, f"Max pointwise error {max_error:.4f} exceeds tolerance"

    @pytest.mark.slow
    def test_invertibility_stationary_random_clouds(self, stationary_field):
        """Test invertibility over multiple random clouds (stationary field)."""
        torch.manual_seed(123)
        num_steps = 10
        flow = NeuralODEFlow(
            field=stationary_field,
            direct=ForwardEuler(),
            inverse=ModifiedEuler(),
            num_steps=num_steps,
        )

        for trial in range(3):
            q0 = torch.randn(1, 20, 3)
            q_T = flow.forward(q0).end
            q0_recovered = flow.inverse(q_T).end

            error = torch.norm(q0_recovered - q0, p=2)
            assert error < 0.5, f"Trial {trial}: error {error:.4f}"


class TestReversedBlockOrder:
    """Test that ModifiedEuler correctly reverses block order for time-varying fields."""

    @pytest.mark.slow
    def test_time_varying_block_reversal(self, time_varying_field):
        """Verify that time-varying inverse uses reversed step order."""
        torch.manual_seed(456)
        num_steps = 5  # Small number for clarity

        q0 = torch.randn(1, 8, 3)

        flow = NeuralODEFlow(
            field=time_varying_field,
            direct=ForwardEuler(),
            inverse=ModifiedEuler(),
            num_steps=num_steps,
        )

        # Forward uses steps 0, 1, 2, 3, 4
        traj_fwd = flow.forward(q0)

        # Inverse should use steps 4, 3, 2, 1, 0 (reversed)
        # This is implicitly tested by invertibility, but we verify structure:
        traj_bwd = flow.inverse(traj_fwd.end)

        # If block reversal is wrong, invertibility fails significantly
        error = torch.norm(traj_bwd.end - q0, p=2)
        assert error < 0.3, (
            f"Block reversal failure: error {error:.4f} suggests "
            "ModifiedEuler not using reversed step order for time-varying field"
        )


class TestInverseDiagnostic:
    """Test the inverse_residual diagnostic."""

    def test_inverse_residual_computation(self):
        """Test that inverse_residual correctly computes ‖φ⁻¹(φ(q)) - q‖."""
        from src.resnet_lddmm.diagnostics import inverse_residual

        # Create a simple flow
        field = StationaryField(num_blocks=5, width=32, activation="relu")
        flow = NeuralODEFlow(
            field=field,
            direct=ForwardEuler(),
            inverse=ModifiedEuler(),
            num_steps=5,
        )

        q0 = torch.randn(1, 5, 3)
        traj_fwd = flow.forward(q0)
        traj_bwd = flow.inverse(traj_fwd.end)

        # Compute residual
        residual = inverse_residual(traj_fwd, traj_bwd)

        # Should be a small positive number (invertibility error)
        assert isinstance(residual, torch.Tensor)
        assert residual.item() >= 0
        assert residual.item() < 1.0, f"Residual {residual:.4f} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
