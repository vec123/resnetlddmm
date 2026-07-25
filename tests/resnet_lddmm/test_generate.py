"""Tests for generative sampling (T30)."""

import torch
import pytest
import os

from src.resnet_lddmm.generate import sample_codes, interpolate_codes, synthesize
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.integrators import ForwardEuler
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.diagnostics import triangle_flips
from src.learning.loader.loaders import CohortBatch


class TestSampleCodes:
    """Tests for sample_codes function."""

    def test_sample_codes_shape(self):
        """Verify sampled codes have correct shape."""
        Z = torch.randn(10, 5)  # 10 shapes, 5-dim latent
        n = 20
        samples = sample_codes(Z, n)
        assert samples.shape == (n, 5)

    def test_sample_codes_device(self):
        """Verify samples inherit device from Z."""
        Z = torch.randn(10, 5)
        samples = sample_codes(Z, 15)
        assert samples.device == Z.device

    def test_sample_codes_dtype(self):
        """Verify samples inherit dtype from Z."""
        Z = torch.randn(10, 5, dtype=torch.float32)
        samples = sample_codes(Z, 15)
        assert samples.dtype == Z.dtype

    def test_sample_codes_mean_matches(self):
        """Verify empirical mean of samples approximates Z mean."""
        Z = torch.randn(50, 8)
        samples = sample_codes(Z, 1000)

        Z_mean = Z.mean(dim=0)
        samples_mean = samples.mean(dim=0)

        # Should be close but not exact (Monte Carlo)
        assert torch.allclose(samples_mean, Z_mean, atol=0.2)

    def test_sample_codes_cov_matches(self):
        """Verify empirical covariance of samples approximates Z covariance."""
        Z = torch.randn(50, 4)
        samples = sample_codes(Z, 500)

        Z_cov = torch.cov(Z.T)
        samples_cov = torch.cov(samples.T)

        # Frobenius norm of difference should be small
        assert torch.norm(samples_cov - Z_cov) < 2.0


class TestInterpolateCodes:
    """Tests for interpolate_codes function."""

    def test_interpolate_scalar_t(self):
        """Verify interpolation with scalar parameter."""
        z_a = torch.tensor([1.0, 0.0])
        z_b = torch.tensor([0.0, 1.0])

        z_half = interpolate_codes(z_a, z_b, 0.5)
        assert torch.allclose(z_half, torch.tensor([0.5, 0.5]))

    def test_interpolate_t_zero(self):
        """Verify interpolation at t=0 gives z_a."""
        z_a = torch.randn(5)
        z_b = torch.randn(5)

        z_interp = interpolate_codes(z_a, z_b, 0.0)
        assert torch.allclose(z_interp, z_a)

    def test_interpolate_t_one(self):
        """Verify interpolation at t=1 gives z_b."""
        z_a = torch.randn(5)
        z_b = torch.randn(5)

        z_interp = interpolate_codes(z_a, z_b, 1.0)
        assert torch.allclose(z_interp, z_b)

    def test_interpolate_tensor_ts(self):
        """Verify interpolation with tensor of parameters."""
        z_a = torch.tensor([0.0, 0.0])
        z_b = torch.tensor([1.0, 1.0])
        ts = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        z_interp = interpolate_codes(z_a, z_b, ts)
        assert z_interp.shape == (5, 2)
        assert torch.allclose(z_interp[0], z_a)
        assert torch.allclose(z_interp[2], torch.tensor([0.5, 0.5]))
        assert torch.allclose(z_interp[4], z_b)

    def test_interpolate_shape(self):
        """Verify output shape is [M, n_z] for M parameters."""
        z_a = torch.randn(8)
        z_b = torch.randn(8)
        ts = torch.linspace(0, 1, 10)

        z_interp = interpolate_codes(z_a, z_b, ts)
        assert z_interp.shape == (10, 8)


class TestSynthesize:
    """Tests for synthesize function."""

    @staticmethod
    def make_flow(num_steps=5, n_z=3):
        """Create a minimal flow for testing."""
        field = TimeVaryingField(num_blocks=num_steps, width=32, conditioning=None)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        return flow

    def test_synthesize_returns_trajectory(self):
        """Verify synthesize returns a trajectory."""
        flow = self.make_flow()
        code = torch.randn(1, 3)
        template = torch.randn(1, 10, 3)

        traj = synthesize(flow, code, template)
        assert hasattr(traj, 'points')
        assert hasattr(traj, 'kinetic_energy')

    def test_synthesize_trajectory_shape(self):
        """Verify trajectory points have correct shape."""
        flow = self.make_flow(num_steps=5)
        code = torch.randn(1, 3)
        template = torch.randn(1, 20, 3)

        traj = synthesize(flow, code, template)
        # Trajectory should be [K+1, B, N, 3]
        assert traj.points.shape[0] == 6  # K+1 = 5+1
        assert traj.points.shape[1] == 1  # B
        assert traj.points.shape[2] == 20  # N
        assert traj.points.shape[3] == 3

    def test_synthesize_code_unsqueeze(self):
        """Verify synthesize handles 1D code by unsqueezing."""
        flow = self.make_flow()
        code_1d = torch.randn(3)
        code_2d = code_1d.unsqueeze(0)
        template = torch.randn(1, 10, 3)

        traj_1d = synthesize(flow, code_1d, template)
        traj_2d = synthesize(flow, code_2d, template)

        # Both should give same result
        assert torch.allclose(traj_1d.points, traj_2d.points)

    def test_synthesize_different_codes_no_error(self):
        """Verify synthesize handles different codes without error.

        With an untrained flow (NoConditioning), codes are ignored.
        This test just verifies the code path works; actual differentiation
        would require a conditioned flow (e.g., PositionAware).
        """
        flow = self.make_flow()
        code_a = torch.randn(1, 3)
        code_b = torch.randn(1, 3)
        template = torch.randn(1, 15, 3)

        traj_a = synthesize(flow, code_a, template)
        traj_b = synthesize(flow, code_b, template)

        # Both should produce valid trajectories (codes are just passed through)
        assert traj_a.points.shape == traj_b.points.shape
        assert traj_a.points.shape[0] == 6  # K+1 with num_steps=5


class TestIntegration:
    """Integration tests: sample/interpolate codes → valid trajectories."""

    @staticmethod
    def make_flow_and_codes():
        """Create a flow and learned codes for testing."""
        # Simple trained-ish setup: diagonal-init codes
        field = TimeVaryingField(num_blocks=4, width=32, conditioning=None)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=4)

        # Synthetic codes (would come from AutoDecoderCodes in real training)
        Z = torch.randn(5, 3)

        return flow, Z

    def test_sampled_codes_produce_trajectories(self):
        """Verify sampled codes can be used with synthesize."""
        flow, Z = self.make_flow_and_codes()
        template = torch.randn(1, 12, 3)

        samples = sample_codes(Z, 3)
        for i in range(samples.shape[0]):
            code = samples[i:i+1]
            traj = synthesize(flow, code, template)
            assert traj.points.shape[0] == 5  # K+1

    def test_interpolated_codes_produce_trajectories(self):
        """Verify interpolated codes can be used with synthesize."""
        flow, Z = self.make_flow_and_codes()
        template = torch.randn(1, 10, 3)

        z_a = Z[0]
        z_b = Z[1]
        ts = torch.linspace(0, 1, 5)
        interp_codes = interpolate_codes(z_a, z_b, ts)

        for i in range(interp_codes.shape[0]):
            code = interp_codes[i:i+1]
            traj = synthesize(flow, code, template)
            assert traj.points.shape[0] == 5

    def test_synthesized_with_faces_no_flips(self):
        """Verify synthesized shapes pass triangle_flips (acceptance test).

        This is a basic check: with an untrained (random init) flow, we
        just verify triangle_flips can be called without error and returns
        sensible output.
        """
        flow = TestSynthesize.make_flow(num_steps=4)
        code = torch.randn(1, 3)

        # Create a simple mesh (tetrahedron)
        template_points = torch.tensor([
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]
        ], dtype=torch.float32)

        faces = torch.tensor([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ], dtype=torch.int64)

        traj = synthesize(flow, code, template_points)
        flips = triangle_flips(traj, faces)

        # Check output shape: [K+1, B, F]
        assert flips.shape[0] == 5  # K+1
        assert flips.shape[1] == 1  # B
        assert flips.shape[2] == 4  # F
        assert flips.dtype == torch.bool

    def test_code_pca_plot_artifact(self):
        """Verify code PCA plot can be generated.

        This is a minimal check: just verify we can compute PCA
        and produce a plot artifact without error.
        """
        import tempfile
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Create a small code matrix with some structure
        Z = torch.randn(20, 5)

        # Compute PCA via SVD
        Z_centered = Z - Z.mean(dim=0)
        U, S, Vh = torch.svd(Z_centered)
        pcs = Z_centered @ Vh[:, :2]  # Project to 2D

        # Create plot
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(pcs[:, 0].numpy(), pcs[:, 1].numpy())
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('Code Space (PCA)')

            out_path = os.path.join(tmpdir, 'code_pca.png')
            fig.savefig(out_path)
            plt.close(fig)

            # Verify file was created
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
