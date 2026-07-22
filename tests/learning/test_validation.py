"""Tests for the validation path added to the training loop.

Covers the contract that makes validation trustworthy and useful:

    * ``TrainingStepper.eval_step`` computes a finite loss WITHOUT moving any weights
      (no optimizer step, no gradient leak from being called next to training);
    * ``TrainingOrchestrator`` runs validation at the ``val_every`` cadence, logs a
      per-term ``val/<term>`` split alongside ``val/loss``, and asks the logger to
      save the validation VTPs;
    * end-to-end, a run leaves validation prediction VTPs under ``vtk/validation`` and a
      ``metrics.png`` + ``metrics.json`` behind.

Batches are the 4-tuple ``(graph, super_graph, true_verts, padding_mask)`` the real
loaders yield; ``super_graph`` is None (the full-graph path). Run with pytest, or:
    python tests/learning/test_validation.py
"""

import os
import sys
import math
import tempfile

import torch


from torch_geometric.data import Data

from src.learning.trainers.E3_end2end import TrainingStepper, TrainingOrchestrator
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.callbacks.base import Callback
from src.learning.callbacks.metrics import MetricsRecorder, MetricsPlotter
from src.learning.callbacks.visualization import GeometryVisualizer
from src.learning.callbacks.validation import ValidationRunner
from src.learning.logger.train_logs import TrainingLogger
from src.learning.models.group_encoder import GroupEncoder
from src.learning.models.folding_decoder import FoldingDecoder


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_batch():
    """A small 2-graph batch as ``(graph, super_graph, true_verts, padding_mask)``."""
    torch.manual_seed(0)

    def ring(offset, n):
        s = torch.arange(n) + offset
        d = (torch.arange(n) + 1) % n + offset
        return torch.stack([torch.cat([s, d]), torch.cat([d, s])])

    edge_index = torch.cat([ring(0, 6), ring(6, 6)], dim=1)
    pos = torch.randn(12, 3)
    batch = torch.tensor([0] * 6 + [1] * 6)
    x = torch.ones(12, 1)  # 1x0e input scalar per node

    graph = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
    true_verts = torch.randn(2, 10, 3)
    padding_mask = torch.ones(2, 10, dtype=torch.bool)
    return graph, None, true_verts, padding_mask


def make_models():
    layers_cfg = [{
        'in_irreps': '1x0e',
        'target_irreps': '4x0e + 2x1o',
        'spatial_sh_lmax': 1,
        'interaction_sh_lmax': 4,
    }]
    encoder = GroupEncoder(layers_cfg=layers_cfg, latent_dim=4,
                            output_irreps='4x0e + 2x1o', verbose=False)
    decoder = FoldingDecoder(num_samples=16, latent_dim=4, n_freqs=2, verbose=False)
    return encoder, decoder


# --------------------------------------------------------------------------- #
# TrainingStepper.eval_step
# --------------------------------------------------------------------------- #
def test_eval_step_does_not_update_parameters():
    """Validation must compute a finite loss WITHOUT moving any weights."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')
    graph, super_graph, true_verts, mask = make_batch()

    before = decoder.fold2.weight.detach().clone()
    pred, loss, breakdown = stepper.eval_step(graph, super_graph, true_verts, mask)
    after = decoder.fold2.weight.detach()

    assert isinstance(loss, float) and math.isfinite(loss), f"bad val loss: {loss}"
    assert pred.shape == (2, 16, 3), f"unexpected decoded shape {tuple(pred.shape)}"
    assert torch.allclose(before, after), "eval_step must NOT update parameters"


# --------------------------------------------------------------------------- #
# TrainingOrchestrator validation cadence (isolated from the real stepper)
# --------------------------------------------------------------------------- #
class _DummyModule:
    """Stands in for encoder/decoder so run_validation can toggle eval()/train()."""
    def eval(self):
        pass

    def train(self):
        pass


class _RecordingStepper:
    def __init__(self):
        self.calls = 0
        self.eval_calls = 0
        self.encoder = _DummyModule()
        self.decoder = _DummyModule()

    def train_step(self, *batch):
        self.calls += 1
        # (pred, loss, breakdown)
        return None, 0.5, {"recon": 0.4}

    def eval_step(self, *batch):
        self.eval_calls += 1
        # (pred, loss, breakdown) -- same shape as train_step post-T10
        return None, 0.25, {"recon": 0.2}


class _RecordingCallback(Callback):
    """Observes validation broadcasts without doing any work of its own."""
    def __init__(self):
        super().__init__(every_n_steps=1)
        self.validated = []
        self.val_metrics = []

    def on_validation_end(self, ctx, step, metrics, batch, pred):
        self.validated.append(step)
        self.val_metrics.append(metrics)


class _InfiniteLoader:
    """Iterable that yields a fixed (empty) 4-tuple batch forever."""
    def __iter__(self):
        while True:
            yield (None, None, None, None)


def test_validation_runner_broadcasts_at_its_own_cadence():
    stepper = _RecordingStepper()
    observer = _RecordingCallback()
    orch = TrainingOrchestrator(
        stepper=stepper, dataloader=_InfiniteLoader(),
        callbacks=[ValidationRunner(_InfiniteLoader(), every_n_steps=2), observer])

    orch.run(num_steps=4)

    assert stepper.eval_calls == 2, f"expected val at steps 0,2 -> 2 eval calls, got {stepper.eval_calls}"
    # ValidationRunner broadcasts on_validation_end to its PEERS -- that fan-out is
    # what keeps the orchestrator ignorant that a validation set exists at all.
    assert observer.validated == [0, 2], f"val broadcast cadence wrong: {observer.validated}"


def test_no_validation_runner_means_no_validation():
    stepper = _RecordingStepper()
    observer = _RecordingCallback()
    orch = TrainingOrchestrator(stepper=stepper, dataloader=_InfiniteLoader(),
                                callbacks=[observer])

    orch.run(num_steps=4)

    assert stepper.eval_calls == 0, "no ValidationRunner -> validation must not run"
    assert observer.validated == [], "no ValidationRunner -> no validation events"


def test_validation_metrics_carry_total_and_terms():
    stepper = _RecordingStepper()
    observer = _RecordingCallback()
    TrainingOrchestrator(
        stepper=stepper, dataloader=_InfiniteLoader(),
        callbacks=[ValidationRunner(_InfiniteLoader(), every_n_steps=1), observer],
    ).run(num_steps=1)

    assert observer.val_metrics == [{"loss": 0.25, "recon": 0.2}], \
        f"unexpected validation metrics: {observer.val_metrics}"


# --------------------------------------------------------------------------- #
# End-to-end: validation artifacts on disk
# --------------------------------------------------------------------------- #
class _OneBatchLoader:
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        while True:
            yield self.batch


def test_eval_step_is_deterministic():
    """T10 step 2's intentional behavior change: validation used to reparameterize
    with fresh random noise under no_grad, so the same weights on the same data
    scored differently every call. It must now be repeatable."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-2, device='cpu')
    encoder.eval()
    decoder.eval()
    batch = make_batch()

    _, loss_a, _ = stepper.eval_step(*batch)
    _, loss_b, _ = stepper.eval_step(*batch)

    assert loss_a == loss_b, (
        f"eval_step is not deterministic ({loss_a} != {loss_b}); it is injecting noise")


def test_eval_step_breakdown_matches_train_step_keys():
    """T11 depends on this: val/<term> can only line up against train/<term> if
    both paths key their breakdowns identically. `contrastive` is the documented
    exception -- it is training-only and two-view-only."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=1e-2, device='cpu',
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("kl", 0.1)]))
    batch = make_batch()

    _, _, train_breakdown = stepper.train_step(*batch)
    _, _, eval_breakdown = stepper.eval_step(*batch)

    assert set(train_breakdown) == set(eval_breakdown) == {"recon", "kl"}


def test_metrics_json_carries_per_term_train_and_val_series():
    """T11's deliverable: a real run's metrics.json must contain val/<term> keys,
    not just one opaque val_loss -- and each must have a train/<term> counterpart
    so the two are directly comparable on the plot."""
    import json

    encoder, decoder = make_models()
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=1e-3, device='cpu',
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("kl", 0.1)]))

    with tempfile.TemporaryDirectory() as tmp:
        recorder = MetricsRecorder(every_n_steps=1, verbose=False)
        orch = TrainingOrchestrator(
            stepper=stepper, dataloader=_OneBatchLoader(make_batch()),
            callbacks=[recorder,
                       ValidationRunner(_OneBatchLoader(make_batch()), every_n_steps=1)],
            log_dir=tmp)
        orch.run(num_steps=2)

        history = json.load(open(os.path.join(tmp, "metrics.json")))

        assert {"val/loss", "val/recon", "val/kl"} <= set(history), \
            f"validation per-term split missing from metrics.json: {sorted(history)}"
        assert {"train/loss", "train/recon", "train/kl"} <= set(history), \
            f"train per-term series missing from metrics.json: {sorted(history)}"

        # Symmetry: every val term has a train counterpart, which is what makes a
        # train-vs-val comparison per term possible at all.
        val_terms = {k.split("/", 1)[1] for k in history if k.startswith("val/")}
        train_terms = {k.split("/", 1)[1] for k in history if k.startswith("train/")}
        assert val_terms <= train_terms, f"val-only terms: {val_terms - train_terms}"

        assert "val_loss" not in history, "old un-prefixed key should be gone"


def test_validation_logs_every_configured_term():
    """The split must follow the loss config, not a hardcoded recon/kl pair."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=1e-3, device='cpu',
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("frobenius", 0.01)]))

    observer = _RecordingCallback()
    TrainingOrchestrator(
        stepper=stepper, dataloader=_OneBatchLoader(make_batch()),
        callbacks=[ValidationRunner(_OneBatchLoader(make_batch()), every_n_steps=1),
                   observer],
    ).run(num_steps=1)

    terms = set().union(*observer.val_metrics)
    assert "frobenius" in terms, f"configured term not reported: {sorted(terms)}"
    assert "kl" not in terms, "unconfigured term must not appear"


def test_validation_writes_vtps_and_metrics_plot():
    """A real run must write validation prediction VTPs under vtk/validation and leave a
    metrics.png + metrics.json behind — the artifacts requested."""
    encoder, decoder = make_models()
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-3, device='cpu')

    with tempfile.TemporaryDirectory() as tmp:
        recorder = MetricsRecorder(every_n_steps=1, verbose=False)
        orch = TrainingOrchestrator(
            stepper=stepper, dataloader=_OneBatchLoader(make_batch()),
            callbacks=[recorder,
                       MetricsPlotter(recorder, every_n_steps=1, verbose=False),
                       GeometryVisualizer(every_n_steps=1),
                       ValidationRunner(_OneBatchLoader(make_batch()), every_n_steps=1)],
            log_dir=tmp)
        orch.run(num_steps=2)

        val_dir = os.path.join(tmp, "vtk", "validation")
        assert os.path.isdir(val_dir), "validation VTP subdir was not created"
        val_vtps = [f for f in os.listdir(val_dir) if f.endswith(".vtp")]
        assert any(f.startswith("pred_shape") for f in val_vtps), \
            f"no validation prediction VTPs written: {val_vtps}"

        assert os.path.exists(os.path.join(tmp, "metrics.png")), "metrics plot not written"
        assert os.path.exists(os.path.join(tmp, "metrics.json")), "metrics history not written"


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {str(exc)[:140]}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")


if __name__ == '__main__':
    _run_all()
