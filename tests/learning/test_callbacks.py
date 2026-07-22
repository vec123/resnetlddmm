"""Tests for the callback split (INSTRUCTIONS.md T12).

Covers the Observer contract itself (hooks fire, cadences are per-callback, a
callback can end the run), the append-only JSONL fix for E1, and the acceptance
test T12 names: an EarlyStopping callback written WITHOUT editing the orchestrator.
"""

import json
import os
import tempfile

import pytest

from src.learning.callbacks.base import Callback, TrainingContext
from src.learning.callbacks.metrics import MetricsRecorder, MetricsPlotter
from src.learning.callbacks.checkpointing import CheckpointWriter
from src.learning.callbacks.early_stopping import EarlyStopping
from src.learning.callbacks.validation import ValidationRunner
from src.learning.trainers.E3_end2end import TrainingOrchestrator


class _FakeStepper:
    """Minimal stepper: returns a loss that falls, then plateaus."""
    def __init__(self, losses=None):
        self.losses = losses
        self.calls = 0
        self.eval_calls = 0
        self.encoder = _FakeModule()
        self.decoder = _FakeModule()
        self.optimizer = _FakeModule()

    def _loss_at(self, i):
        if self.losses is None:
            return 1.0
        return self.losses[min(i, len(self.losses) - 1)]

    def train_step(self, *batch):
        loss = self._loss_at(self.calls)
        self.calls += 1
        return None, loss, {"recon": loss}

    def eval_step(self, *batch):
        loss = self._loss_at(self.eval_calls)
        self.eval_calls += 1
        return None, loss, {"recon": loss}


class _FakeModule:
    def eval(self):
        pass

    def train(self):
        pass

    def state_dict(self):
        return {}


class _Loader:
    def __iter__(self):
        while True:
            yield (None, None, None, None)


# --------------------------------------------------------------------------- #
# The Observer contract
# --------------------------------------------------------------------------- #
def test_base_callback_hooks_are_all_no_ops():
    """Default no-op bodies are what let a subclass override two hooks, not four."""
    cb = Callback()
    ctx = TrainingContext(stepper=None, log_dir="", num_steps=0)
    cb.on_train_start(ctx)
    cb.on_step_end(ctx, 0, {}, None, None)
    cb.on_validation_end(ctx, 0, {}, None, None)
    cb.on_train_end(ctx)


@pytest.mark.parametrize("every,expected", [
    (1, [0, 1, 2, 3]),
    (2, [0, 2]),
    (3, [0, 3]),
    (0, []),        # 0 disables the callback entirely
])
def test_cadence_is_per_callback(every, expected):
    cb = Callback(every_n_steps=every)
    assert [s for s in range(4) if cb._due(s)] == expected


def test_context_stop_ends_the_run_early():
    """ctx.stop() is the entire mechanism a callback needs to halt training --
    the orchestrator checks one flag and knows nothing about why."""
    class _StopAtTwo(Callback):
        def on_step_end(self, ctx, step, metrics, batch, pred):
            if step == 2:
                ctx.stop("because")

    stepper = _FakeStepper()
    ctx = TrainingOrchestrator(stepper=stepper, dataloader=_Loader(),
                               callbacks=[_StopAtTwo()]).run(num_steps=10)

    assert stepper.calls == 3, f"expected to stop after step 2, ran {stepper.calls}"
    assert ctx.should_stop and ctx.stop_reason == "because"


# --------------------------------------------------------------------------- #
# MetricsRecorder: the E1 fix
# --------------------------------------------------------------------------- #
def test_metrics_jsonl_is_append_only_one_record_per_term():
    """E1 fix: the old logger re-serialized the whole history on every call.
    Each event must now be one appended line, in the documented schema."""
    with tempfile.TemporaryDirectory() as tmp:
        recorder = MetricsRecorder(every_n_steps=1, verbose=False)
        TrainingOrchestrator(stepper=_FakeStepper(), dataloader=_Loader(),
                             callbacks=[recorder], log_dir=tmp).run(num_steps=3)

        path = os.path.join(tmp, "metrics.jsonl")
        records = [json.loads(line) for line in open(path, encoding="utf-8")]

    # 3 steps x 2 terms (loss, recon), one line each.
    assert len(records) == 6, f"expected 6 append-only records, got {len(records)}"
    assert set(records[0]) == {"step", "split", "term", "value"}
    assert {r["term"] for r in records} == {"loss", "recon"}
    assert {r["split"] for r in records} == {"train"}
    assert [r["step"] for r in records] == [0, 0, 1, 1, 2, 2]


def test_metrics_jsonl_survives_a_run_that_never_ends_cleanly():
    """Lines are flushed per event, so a killed run keeps everything it logged --
    which the old write-at-the-end-only JSON could not offer."""
    with tempfile.TemporaryDirectory() as tmp:
        recorder = MetricsRecorder(every_n_steps=1, verbose=False)
        ctx = TrainingContext(stepper=None, log_dir=tmp, num_steps=2)
        recorder.on_train_start(ctx)
        recorder.on_step_end(ctx, 0, {"loss": 1.0}, None, None)
        # deliberately NO on_train_end -- simulating a crash mid-run
        with open(os.path.join(tmp, "metrics.jsonl"), encoding="utf-8") as f:
            lines = f.readlines()
        # Windows won't delete the tempdir while the handle is open; a real crashed
        # process has it closed by the OS, so releasing it here is faithful.
        recorder._fh.close()

    assert len(lines) == 1, "flushed record missing after an unclean exit"
    assert json.loads(lines[0])["value"] == 1.0


def test_recorder_splits_train_and_validation_series():
    with tempfile.TemporaryDirectory() as tmp:
        recorder = MetricsRecorder(every_n_steps=1, verbose=False)
        TrainingOrchestrator(
            stepper=_FakeStepper(), dataloader=_Loader(),
            callbacks=[recorder, ValidationRunner(_Loader(), every_n_steps=2)],
            log_dir=tmp).run(num_steps=4)

        records = [json.loads(l) for l in open(os.path.join(tmp, "metrics.jsonl"), encoding="utf-8")]
        history = json.load(open(os.path.join(tmp, "metrics.json")))

    assert {r["split"] for r in records} == {"train", "val"}
    assert {"train/loss", "train/recon", "val/loss", "val/recon"} <= set(history)


# --------------------------------------------------------------------------- #
# CheckpointWriter
# --------------------------------------------------------------------------- #
def test_checkpoint_writer_honors_its_own_cadence():
    with tempfile.TemporaryDirectory() as tmp:
        TrainingOrchestrator(
            stepper=_FakeStepper(), dataloader=_Loader(),
            callbacks=[CheckpointWriter(every_n_steps=2, verbose=False)],
            log_dir=tmp).run(num_steps=5)

        written = sorted(os.listdir(os.path.join(tmp, "checkpoints")))

    # final.pt is written unconditionally on train end, alongside the cadence saves.
    assert written == ["final.pt", "step_0.pt", "step_2.pt", "step_4.pt"], written


def test_checkpoint_writer_always_saves_final_weights():
    """Cadence alone loses the trained weights whenever a run ends off-cadence --
    here the last step is 4, which never hits every_n_steps=10."""
    with tempfile.TemporaryDirectory() as tmp:
        TrainingOrchestrator(
            stepper=_FakeStepper(), dataloader=_Loader(),
            callbacks=[CheckpointWriter(every_n_steps=10, verbose=False)],
            log_dir=tmp).run(num_steps=5)

        path = os.path.join(tmp, "checkpoints", "final.pt")
        assert os.path.exists(path), "final weights were not saved"
        import torch
        assert torch.load(path)["step"] == 4, "final.pt should record the last step"


# --------------------------------------------------------------------------- #
# T12's acceptance test: a NEW behavior, with zero orchestrator edits
# --------------------------------------------------------------------------- #
def test_early_stopping_halts_a_plateaued_run():
    """The proof the abstraction is finished: EarlyStopping adds metric watching,
    patience and run termination without one line changing in TrainingOrchestrator,
    TrainingStepper, or any other callback."""
    # Improves for two validations, then flatlines.
    stepper = _FakeStepper(losses=[1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    stopper = EarlyStopping(monitor="loss", patience=2, verbose=False)

    ctx = TrainingOrchestrator(
        stepper=stepper, dataloader=_Loader(),
        callbacks=[ValidationRunner(_Loader(), every_n_steps=1), stopper],
    ).run(num_steps=20)

    assert ctx.should_stop, "early stopping never fired"
    assert stepper.calls < 20, f"run was not cut short: {stepper.calls} steps"
    assert "did not improve" in ctx.stop_reason


def test_early_stopping_does_not_fire_while_improving():
    stepper = _FakeStepper(losses=[1.0, 0.9, 0.8, 0.7, 0.6])
    stopper = EarlyStopping(monitor="loss", patience=2, verbose=False)

    ctx = TrainingOrchestrator(
        stepper=stepper, dataloader=_Loader(),
        callbacks=[ValidationRunner(_Loader(), every_n_steps=1), stopper],
    ).run(num_steps=5)

    assert not ctx.should_stop
    assert stepper.calls == 5


def test_early_stopping_ignores_a_metric_it_cannot_see():
    """A monitor naming a term this run doesn't produce must be inert, not fatal."""
    stepper = _FakeStepper()
    stopper = EarlyStopping(monitor="nonexistent", patience=1, verbose=False)

    ctx = TrainingOrchestrator(
        stepper=stepper, dataloader=_Loader(),
        callbacks=[ValidationRunner(_Loader(), every_n_steps=1), stopper],
    ).run(num_steps=3)

    assert not ctx.should_stop
    assert stepper.calls == 3