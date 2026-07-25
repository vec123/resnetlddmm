"""Tests for callback compatibility with both legacy and new steppers (STEP T18)."""

import tempfile
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.learning.callbacks.checkpointing import CheckpointWriter
from src.learning.callbacks.validation import ValidationRunner
from src.learning.callbacks.base import TrainingContext


@dataclass
class SimpleBatch:
    """Simple batch for testing."""
    points: torch.Tensor
    weights: torch.Tensor | None = None


class LegacyStepper:
    """Legacy stepper with .encoder and .decoder attributes (e.g., from older codebase)."""

    def __init__(self):
        self.encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 16))
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=0.001
        )

    def eval_step(self, source, target):
        """Simple eval step that returns (trajectory, loss, breakdown)."""
        # Dummy implementation for testing
        dummy_pred = torch.randn(1, 10, 3)
        loss = torch.tensor(1.0)
        breakdown = {"data": 0.5, "kinetic": 0.5}
        return dummy_pred, loss, breakdown


class NewProtocolStepper:
    """New stepper using the four-method protocol: state_dict/load_state_dict/train/eval."""

    def __init__(self):
        self.flow = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 16))
        self.codes = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))
        self.optimizer = torch.optim.Adam(
            list(self.flow.parameters()) + list(self.codes.parameters()),
            lr=0.001
        )
        self._train_mode = True

    def state_dict(self):
        """Return state dict with new protocol keys."""
        return {
            "flow": self.flow.state_dict(),
            "codes": self.codes.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state dict from new protocol format."""
        self.flow.load_state_dict(state_dict["flow"])
        self.codes.load_state_dict(state_dict["codes"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def train(self):
        """Set to training mode."""
        self.flow.train()
        self.codes.train()
        self._train_mode = True

    def eval(self):
        """Set to eval mode."""
        self.flow.eval()
        self.codes.eval()
        self._train_mode = False

    def eval_step(self, source, target):
        """Simple eval step that returns (trajectory, loss, breakdown)."""
        # Dummy implementation for testing
        dummy_pred = torch.randn(1, 10, 3)
        loss = torch.tensor(1.0)
        breakdown = {"data": 0.5, "kinetic": 0.5}
        return dummy_pred, loss, breakdown


class TestCheckpointWriterCompat:
    """Tests for CheckpointWriter compatibility with both stepper protocols."""

    def test_checkpoint_writer_legacy_stepper(self):
        """Verify CheckpointWriter saves legacy stepper with encoder/decoder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stepper = LegacyStepper()
            ctx = TrainingContext(
                stepper=stepper,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            writer = CheckpointWriter(every_n_steps=1, directory="checkpoints", verbose=False)
            writer.save(ctx, step=0)

            # Verify checkpoint was created with legacy keys
            checkpoint_path = os.path.join(tmpdir, "checkpoints", "step_0.pt")
            assert os.path.exists(checkpoint_path)

            checkpoint = torch.load(checkpoint_path)
            assert "step" in checkpoint
            assert "encoder" in checkpoint
            assert "decoder" in checkpoint
            assert "optimizer" in checkpoint

    def test_checkpoint_writer_new_stepper(self):
        """Verify CheckpointWriter saves new stepper with state_dict protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stepper = NewProtocolStepper()
            ctx = TrainingContext(
                stepper=stepper,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            writer = CheckpointWriter(every_n_steps=1, directory="checkpoints", verbose=False)
            writer.save(ctx, step=0)

            # Verify checkpoint was created with new protocol keys
            checkpoint_path = os.path.join(tmpdir, "checkpoints", "step_0.pt")
            assert os.path.exists(checkpoint_path)

            checkpoint = torch.load(checkpoint_path)
            assert "step" in checkpoint
            assert "flow" in checkpoint
            assert "codes" in checkpoint
            assert "optimizer" in checkpoint

    def test_checkpoint_writer_final_checkpoint(self):
        """Verify final checkpoint is saved on train_end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stepper = NewProtocolStepper()
            ctx = TrainingContext(
                stepper=stepper,
                callbacks=[],
                log_dir=tmpdir,
                num_steps=100,
            )

            writer = CheckpointWriter(every_n_steps=100, directory="checkpoints", verbose=False, save_final=True)
            writer.on_step_end(ctx, step=5, metrics={}, batch=None, pred=None)
            writer.on_train_end(ctx)

            # Verify final checkpoint was created
            final_path = os.path.join(tmpdir, "checkpoints", "final.pt")
            assert os.path.exists(final_path)

            checkpoint = torch.load(final_path)
            assert checkpoint["step"] == 5


class TestValidationRunnerCompat:
    """Tests for ValidationRunner compatibility with both stepper protocols."""

    def test_validation_runner_legacy_stepper(self):
        """Verify ValidationRunner toggles legacy stepper modes correctly."""
        stepper = LegacyStepper()
        stepper.encoder.train()
        stepper.decoder.train()

        # Verify they start in train mode
        assert stepper.encoder.training
        assert stepper.decoder.training

        # Create a simple loader
        source = SimpleBatch(torch.randn(1, 5, 3))
        target = SimpleBatch(torch.randn(1, 5, 3))
        loader = iter([(source, target)])

        ctx = TrainingContext(
            stepper=stepper,
            callbacks=[],
            log_dir="/tmp",
            num_steps=100,
        )

        runner = ValidationRunner(val_loader=loader, num_val_batches=1)
        runner._evaluate(ctx)

        # After _evaluate, should be back in train mode
        assert stepper.encoder.training
        assert stepper.decoder.training

    def test_validation_runner_new_stepper(self):
        """Verify ValidationRunner toggles new stepper modes correctly."""
        stepper = NewProtocolStepper()
        stepper.train()

        # Verify it starts in train mode
        assert stepper._train_mode

        # Create a simple loader
        source = SimpleBatch(torch.randn(1, 5, 3))
        target = SimpleBatch(torch.randn(1, 5, 3))
        loader = iter([(source, target)])

        ctx = TrainingContext(
            stepper=stepper,
            callbacks=[],
            log_dir="/tmp",
            num_steps=100,
        )

        runner = ValidationRunner(val_loader=loader, num_val_batches=1)
        runner._evaluate(ctx)

        # After _evaluate, should be back in train mode
        assert stepper._train_mode

    def test_validation_runner_new_stepper_eval_mode(self):
        """Verify new stepper is set to eval during _evaluate."""
        stepper = NewProtocolStepper()

        # Mock eval_step to check mode
        modes_seen = []

        original_eval_step = stepper.eval_step
        def tracked_eval_step(*args, **kwargs):
            modes_seen.append(stepper._train_mode)
            return original_eval_step(*args, **kwargs)

        stepper.eval_step = tracked_eval_step

        # Create a simple one-batch loader
        source = SimpleBatch(torch.randn(1, 5, 3))
        target = SimpleBatch(torch.randn(1, 5, 3))
        loader = iter([(source, target)])

        ctx = TrainingContext(
            stepper=stepper,
            callbacks=[],
            log_dir="/tmp",
            num_steps=100,
        )

        runner = ValidationRunner(val_loader=loader, num_val_batches=1)
        runner._evaluate(ctx)

        # Verify eval_step was called in eval mode (False)
        assert False in modes_seen, f"Expected eval mode (False) in {modes_seen}"

    def test_validation_runner_with_callbacks(self):
        """Verify ValidationRunner broadcasts on_validation_end correctly."""
        stepper = NewProtocolStepper()
        validation_calls = []

        class TrackingCallback:
            def __init__(self):
                self.name = "tracker"

            def on_validation_end(self, ctx, step, metrics, batch, pred):
                validation_calls.append((step, metrics))

            def on_train_start(self, ctx):
                pass

            def on_step_end(self, ctx, step, metrics, batch, pred):
                pass

            def on_train_end(self, ctx):
                pass

        tracker = TrackingCallback()

        # Create a simple one-batch loader
        source = SimpleBatch(torch.randn(1, 5, 3))
        target = SimpleBatch(torch.randn(1, 5, 3))
        loader = iter([(source, target)])

        ctx = TrainingContext(
            stepper=stepper,
            callbacks=[tracker],
            log_dir="/tmp",
            num_steps=100,
        )

        runner = ValidationRunner(val_loader=loader, num_val_batches=1)
        runner.on_step_end(ctx, step=0, metrics={}, batch=None, pred=None)

        # Verify callback was called
        assert len(validation_calls) == 1
        assert validation_calls[0][0] == 0  # step
