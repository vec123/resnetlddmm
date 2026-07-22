"""Tests for Trajectory."""

import torch
import pytest
from src.resnet_lddmm.trajectory import Trajectory


class TestTrajectoryImmutability:
    """Tests for Trajectory frozen dataclass properties."""

    def test_trajectory_is_frozen(self):
        """Verify Trajectory is immutable (frozen dataclass)."""
        points = torch.randn(5, 2, 10, 3)  # [K+1, B, N, 3]
        velocities = torch.randn(4, 2, 10, 3)  # [K, B, N, 3]
        traj = Trajectory(points, velocities, 0.1)

        with pytest.raises(AttributeError):
            traj.dt = 0.2


class TestTrajectoryEnd:
    """Tests for the .end property."""

    def test_end_shape(self):
        """Verify .end returns last snapshot with correct shape."""
        K = 10
        B = 3
        N = 20
        points = torch.randn(K + 1, B, N, 3)
        velocities = torch.randn(K, B, N, 3)
        traj = Trajectory(points, velocities, 0.1)

        end = traj.end
        assert end.shape == (B, N, 3)

    def test_end_is_last_point(self):
        """Verify .end returns points[-1] exactly."""
        points = torch.randn(6, 2, 8, 3)
        velocities = torch.randn(5, 2, 8, 3)
        traj = Trajectory(points, velocities, 0.1)

        assert torch.equal(traj.end, points[-1])


class TestTrajectoryKineticEnergy:
    """Tests for kinetic_energy() computation."""

    def test_kinetic_energy_zero_velocity(self):
        """Verify zero velocities give zero kinetic energy."""
        K = 5
        B = 2
        N = 10
        points = torch.randn(K + 1, B, N, 3)
        velocities = torch.zeros(K, B, N, 3)
        traj = Trajectory(points, velocities, 0.1)

        ke = traj.kinetic_energy()
        assert torch.allclose(ke, torch.tensor(0.0), atol=1e-6)
        assert ke.shape == ()  # scalar

    def test_kinetic_energy_hand_computed_fixture(self):
        """Verify kinetic energy against hand-computed values.

        Setup: 2 steps, 1 batch, 2 points, all velocities = [1, 0, 0].
        Expected: 0.5 × (mean over B,N of ||v||²) × sum over steps × dt
                = 0.5 × 1.0 × 2 × 0.2 = 0.2
        """
        K = 2  # 2 integration steps
        B = 1  # 1 batch
        N = 2  # 2 points
        dt = 0.2

        # All velocities are [1, 0, 0] (unit norm along x-axis)
        velocities = torch.ones(K, B, N, 3)
        velocities[..., 1:] = 0  # zero out y, z

        # Points can be anything (not used in kinetic_energy)
        points = torch.zeros(K + 1, B, N, 3)

        traj = Trajectory(points, velocities, dt)
        ke = traj.kinetic_energy()

        # Formula: 0.5 × Σ_k(mean over B,N of ||v_k||²) × dt
        # Each ||v||² = 1, mean([1,1]) = 1, Σ(1 + 1) = 2
        # Result: 0.5 × 2 × 0.2 = 0.2
        expected = torch.tensor(0.2)
        assert torch.allclose(ke, expected, atol=1e-6)

    def test_kinetic_energy_single_step_exactness(self):
        """Verify kinetic energy formula for a single integration step.

        One step: v = [2, 0, 0], B=1, N=1, dt=1.
        Energy = 0.5 × (2²) × 1 = 2.0
        """
        K = 1  # single step
        B = 1
        N = 1
        dt = 1.0

        velocities = torch.tensor([[[[2.0, 0.0, 0.0]]]])  # [K, B, N, 3]
        points = torch.zeros(K + 1, B, N, 3)

        traj = Trajectory(points, velocities, dt)
        ke = traj.kinetic_energy()

        expected = 0.5 * (2.0**2) * 1.0  # = 2.0
        assert torch.allclose(ke, torch.tensor(expected), atol=1e-6)

    def test_kinetic_energy_scaling_with_dt(self):
        """Verify kinetic energy scales linearly with dt."""
        K = 3
        B = 1
        N = 2

        # Constant velocity magnitude
        velocities = torch.ones(K, B, N, 3)
        points = torch.zeros(K + 1, B, N, 3)

        dt1 = 0.1
        ke1 = Trajectory(points, velocities, dt1).kinetic_energy()

        dt2 = 0.2
        ke2 = Trajectory(points, velocities, dt2).kinetic_energy()

        # ke2 should be exactly 2 × ke1
        assert torch.allclose(ke2, 2 * ke1, atol=1e-6)

    def test_kinetic_energy_scaling_with_velocities(self):
        """Verify kinetic energy scales with velocity magnitude."""
        K = 2
        B = 1
        N = 1
        dt = 0.5

        points = torch.zeros(K + 1, B, N, 3)

        # Velocity [1, 1, 1] has norm² = 3
        velocities1 = torch.ones(K, B, N, 3)
        ke1 = Trajectory(points, velocities1, dt).kinetic_energy()

        # Velocity [2, 2, 2] has norm² = 12 (4x larger)
        velocities2 = 2 * torch.ones(K, B, N, 3)
        ke2 = Trajectory(points, velocities2, dt).kinetic_energy()

        # ke2 should be exactly 4 × ke1
        assert torch.allclose(ke2, 4 * ke1, atol=1e-6)

    def test_kinetic_energy_batch_averaging(self):
        """Verify kinetic energy averages over batch and points correctly."""
        K = 1
        B = 2
        N = 2

        # Create distinct velocities for each batch/point
        # Batch 0: [1, 0, 0], [2, 0, 0]
        # Batch 1: [3, 0, 0], [4, 0, 0]
        velocities = torch.zeros(K, B, N, 3)
        velocities[0, 0, 0, 0] = 1.0
        velocities[0, 0, 1, 0] = 2.0
        velocities[0, 1, 0, 0] = 3.0
        velocities[0, 1, 1, 0] = 4.0

        points = torch.zeros(K + 1, B, N, 3)
        dt = 1.0

        traj = Trajectory(points, velocities, dt)
        ke = traj.kinetic_energy()

        # Expected: 0.5 * mean(1² + 2² + 3² + 4²) * dt
        #         = 0.5 * (1 + 4 + 9 + 16) / 4 * 1
        #         = 0.5 * 7.5 = 3.75
        expected = 0.5 * 7.5
        assert torch.allclose(ke, torch.tensor(expected), atol=1e-6)

    def test_kinetic_energy_is_scalar(self):
        """Verify kinetic energy output is always a scalar."""
        for K in [1, 5, 10]:
            for B in [1, 3]:
                for N in [4, 20]:
                    velocities = torch.randn(K, B, N, 3)
                    points = torch.randn(K + 1, B, N, 3)
                    traj = Trajectory(points, velocities, 0.1)

                    ke = traj.kinetic_energy()
                    assert ke.shape == ()


class TestTrajectoryShapes:
    """Tests for trajectory shape contracts."""

    def test_points_shape_k_plus_1(self):
        """Verify points has K+1 time snapshots."""
        for K in [1, 5, 20]:
            points = torch.randn(K + 1, 2, 10, 3)
            velocities = torch.randn(K, 2, 10, 3)
            traj = Trajectory(points, velocities, 0.1)

            assert traj.points.shape[0] == K + 1

    def test_velocities_shape_k(self):
        """Verify velocities has K time steps."""
        for K in [1, 5, 20]:
            points = torch.randn(K + 1, 2, 10, 3)
            velocities = torch.randn(K, 2, 10, 3)
            traj = Trajectory(points, velocities, 0.1)

            assert traj.velocities.shape[0] == K

    def test_batch_and_points_dims_match(self):
        """Verify batch (B) and points (N) dims match between points and velocities."""
        K = 7
        B = 3
        N = 15

        points = torch.randn(K + 1, B, N, 3)
        velocities = torch.randn(K, B, N, 3)
        traj = Trajectory(points, velocities, 0.1)

        assert traj.points.shape[1] == traj.velocities.shape[1] == B
        assert traj.points.shape[2] == traj.velocities.shape[2] == N
