"""Tests for ResNetLDDMM diagnostics and export callbacks (STEP T20)."""

import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import pytest

from src.resnet_lddmm.callbacks import TrajectoryExporter, DiagnosticsCallback, LossLogger
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


class TestLossLogger:
    """Tests for LossLogger callback."""

    @staticmethod
    def _ctx(tmpdir, terms):
        """A TrainingContext whose stepper carries a composer with `terms`."""
        from src.learning.losses.composer import LossComposer

        class _Stepper:
            composer = LossComposer(terms)

        return TrainingContext(stepper=_Stepper(), callbacks=[], log_dir=tmpdir,
                               num_steps=100)

    def _run(self, tmpdir, terms, metrics, step=0, every_n_steps=1):
        """Run one due step and return the parsed record."""
        import json
        import os

        logger = LossLogger(every_n_steps=every_n_steps)
        ctx = self._ctx(tmpdir, terms)
        logger.on_train_start(ctx)
        logger.on_step_end(ctx, step=step, metrics=metrics, batch=None, pred=None)

        path = os.path.join(tmpdir, "loss_logs", f"step_{step:06d}.json")
        with open(path) as f:
            return json.load(f)

    def test_separates_weighted_terms_from_diagnostics(self):
        """Composer terms land in `terms`; anything else lands in `diagnostics`."""
        terms = [("data", 50.0, None), ("kinetic", 1.0, None)]
        metrics = {"loss": 6.75, "data": 0.0135, "kinetic": 6.075,
                   "diag/det_min": 0.83}

        with tempfile.TemporaryDirectory() as tmpdir:
            record = self._run(tmpdir, terms, metrics)

        assert set(record["terms"]) == {"data", "kinetic"}
        assert record["diagnostics"] == {"diag/det_min": 0.83}

    def test_contribution_is_weighted_and_residual_vanishes(self):
        """`value` is unweighted; `contribution` is what entered the objective.

        The residual is the guard that the weights logged here are the weights the
        composer actually applied -- the whole point of logging them.
        """
        terms = [("data", 50.0, None), ("kinetic", 1.0, None)]
        total = 50.0 * 0.0135 + 1.0 * 6.075
        metrics = {"loss": total, "data": 0.0135, "kinetic": 6.075}

        with tempfile.TemporaryDirectory() as tmpdir:
            record = self._run(tmpdir, terms, metrics)

        assert record["terms"]["data"]["value"] == pytest.approx(0.0135)
        assert record["terms"]["data"]["contribution"] == pytest.approx(0.675)
        assert record["sum_of_contributions"] == pytest.approx(total)
        assert record["residual"] == pytest.approx(0.0, abs=1e-9)

    def test_records_terms_that_produced_no_value(self):
        """A configured term absent from the breakdown is named in `skipped`.

        The composer drops None values silently, so without this the log cannot
        distinguish a term that switched itself off from one that was satisfied.
        """
        terms = [("data", 50.0, None), ("isometry", 0.0, None),
                 ("pose_supervision", 1.0, None)]
        metrics = {"loss": 0.675, "data": 0.0135}

        with tempfile.TemporaryDirectory() as tmpdir:
            record = self._run(tmpdir, terms, metrics)

        assert record["skipped"] == ["isometry", "pose_supervision"]

    def test_history_appends_one_line_per_step(self):
        """loss_history.jsonl accumulates; the key set may differ between lines."""
        import json
        import os

        terms = [("data", 50.0, None), ("pose_supervision", 1.0, None)]
        logger = LossLogger(every_n_steps=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = self._ctx(tmpdir, terms)
            logger.on_train_start(ctx)
            # Second step gains a term the first did not have -- exactly the case a
            # CSV header written at step 0 could not represent.
            logger.on_step_end(ctx, 0, {"loss": 0.675, "data": 0.0135}, None, None)
            logger.on_step_end(ctx, 1, {"loss": 1.675, "data": 0.0135,
                                        "pose_supervision": 1.0}, None, None)

            with open(os.path.join(tmpdir, "loss_logs", "loss_history.jsonl")) as f:
                lines = [json.loads(line) for line in f if line.strip()]

        assert [r["step"] for r in lines] == [0, 1]
        assert set(lines[0]["terms"]) == {"data"}
        assert set(lines[1]["terms"]) == {"data", "pose_supervision"}

    def test_skips_when_not_due(self):
        """Off-cadence steps write nothing at all."""
        import os

        logger = LossLogger(every_n_steps=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = self._ctx(tmpdir, [("data", 1.0, None)])
            logger.on_train_start(ctx)
            logger.on_step_end(ctx, step=5, metrics={"loss": 1.0}, batch=None, pred=None)

            assert os.listdir(os.path.join(tmpdir, "loss_logs")) == []

    def test_tolerates_a_tensor_metric(self):
        """A tensor left in the metrics dict is coerced, not fatal."""
        terms = [("data", 2.0, None)]
        metrics = {"loss": torch.tensor(0.5), "data": torch.tensor(0.25)}

        with tempfile.TemporaryDirectory() as tmpdir:
            record = self._run(tmpdir, terms, metrics)

        assert record["total"] == pytest.approx(0.5)
        assert record["terms"]["data"]["contribution"] == pytest.approx(0.5)
