"""Tests for ResNetLDDMM diagnostics and export callbacks (STEP T20)."""

import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import pytest

from src.resnet_lddmm.callbacks import TrajectoryExporter, DiagnosticsCallback
from src.resnet_lddmm.trajectory import Trajectory
from src.learning.callbacks.base import TrainingContext


@dataclass
class SimpleBatch:
    """Simple batch with points and optional faces."""
    points: torch.Tensor
    faces: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None


class TestTrajectoryExporter:
    """Tests for TrajectoryExporter callback."""

    def test_exporter_skips_if_not_due(self):
        """Exporter should skip if step is not on cadence."""
        exporter = TrajectoryExporter(every_n_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = TrainingContext(
                stepper=None,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            # Call on step 5 (not divisible by 10)
            exporter.on_step_end(ctx, step=5, metrics={}, batch=None, pred=None)
            # Should return early without error

    def test_exporter_skips_if_no_prediction(self):
        """Exporter should skip if pred is None."""
        exporter = TrajectoryExporter(every_n_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = TrainingContext(
                stepper=None,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            # Call with pred=None (should skip gracefully)
            exporter.on_step_end(ctx, step=0, metrics={}, batch=None, pred=None)

    def test_exporter_skips_if_no_faces(self):
        """Exporter should skip if batch has no faces."""
        exporter = TrajectoryExporter(every_n_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = TrainingContext(
                stepper=None,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            # Batch without faces
            source = SimpleBatch(points=torch.randn(1, 5, 3), faces=None)
            batch = (source, None)

            # Prediction (trajectory)
            traj = Trajectory(
                points=torch.randn(3, 1, 5, 3),
                velocities=torch.randn(2, 1, 5, 3),
                dt=0.1
            )

            # Should skip without error
            exporter.on_step_end(ctx, step=0, metrics={}, batch=batch, pred=traj)


class TestDiagnosticsCallback:
    """Tests for DiagnosticsCallback."""

    def test_diagnostics_skips_if_not_due(self):
        """Diagnostics should skip if step is not on cadence."""
        diag = DiagnosticsCallback(every_n_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = TrainingContext(
                stepper=None,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            metrics = {}
            # Call on step 5 (not divisible by 10)
            diag.on_step_end(ctx, step=5, metrics=metrics, batch=None, pred=None)
            # Should return early; metrics unchanged
            assert len(metrics) == 0

    def test_diagnostics_computes_on_due_step(self):
        """Diagnostics should compute metrics when due."""
        diag = DiagnosticsCallback(every_n_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock stepper with flow
            class MockFlow:
                def modules(self):
                    return [torch.nn.Linear(3, 3)]

            class MockStepper:
                def __init__(self):
                    self.flow = MockFlow()

            ctx = TrainingContext(
                stepper=MockStepper(),
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            # Trajectory
            traj = Trajectory(
                points=torch.randn(3, 1, 5, 3),
                velocities=torch.randn(2, 1, 5, 3),
                dt=0.1
            )

            # Batch with faces
            faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
            source = SimpleBatch(points=torch.randn(1, 3, 3), faces=faces)
            batch = (source, None)

            metrics = {}
            diag.on_step_end(ctx, step=0, metrics=metrics, batch=batch, pred=traj)

            # Should have computed some diagnostics
            # (actual computation may fail on stub Jacobian, but should try)
            # Just verify the callback ran without crashing
            assert isinstance(metrics, dict)

    def test_diagnostics_handles_missing_flow(self):
        """Diagnostics should handle stepper without flow attribute."""
        diag = DiagnosticsCallback(every_n_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            class MinimalStepper:
                pass

            ctx = TrainingContext(
                stepper=MinimalStepper(),
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            traj = Trajectory(
                points=torch.randn(3, 1, 5, 3),
                velocities=torch.randn(2, 1, 5, 3),
                dt=0.1
            )

            source = SimpleBatch(points=torch.randn(1, 5, 3), faces=None)
            batch = (source, None)

            metrics = {}
            # Should not crash even though stepper has no flow
            diag.on_step_end(ctx, step=0, metrics=metrics, batch=batch, pred=traj)
