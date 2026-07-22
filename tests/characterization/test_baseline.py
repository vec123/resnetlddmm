"""Characterization tests for the current (pre-refactor) training pipeline.

These do NOT assert correctness. They pin what the pipeline computes TODAY --
load_dataset -> build_training_graph -> GroupEncoder -> FoldingDecoder ->
TrainingStepper -- so later refactoring tasks (T3-T11 in INSTRUCTIONS.md) fail
loudly the moment they change a number, not just an interface.

Expected values live in baseline_expected.json, not inline: an INTENDED change
regenerates the file and the git diff shows exactly what moved.

Two known-wrong behaviors are deliberately NOT pinned here (see INSTRUCTIONS.md T2):
  * validation reparameterizes with random noise under no_grad (fixed in T10)
  * run_validation discards avg_recon_loss / avg_kl_loss (fixed in T11)
This file only exercises the TRAIN path, never eval_step / run_validation.
"""

import json
import os
import random

import numpy as np
import pytest
import torch

from src.paths import get_project_root
from src.learning.helpers import load_dataset, build_training_graph
from src.learning.models.group_encoder import GroupEncoder
from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.trainers.E3_end2end import TrainingStepper
from src.learning.losses.composer import LossComposer, LossTerm

SEED = 0
LATENT_DIM = 5
NUM_SAMPLES = 16          # perfect square, required by FoldingDecoder's grid
NUM_STEPS = 3
R_MAX = 0.15
DROPOUT_RATE = 0.95

EXPECTED_PATH = os.path.join(os.path.dirname(__file__), "baseline_expected.json")
DATA_DIR = os.path.join(get_project_root(), "tests", "data")


def _seed_everything():
    """Mirrors scripts/equivariant_gnn_train.py:115-121 exactly."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)              # seeds CPU + the default CUDA generator
    torch.cuda.manual_seed_all(SEED)     # all GPUs (redundant with above but explicit)
    key = torch.Generator(device="cpu")
    key.manual_seed(SEED)
    return key


def _build_encoder():
    layers_cfg = [{
        "in_irreps": "1x0e",
        "target_irreps": "8x0e + 4x1o",
        "spatial_sh_lmax": 1,
        "interaction_sh_lmax": 4,
    }]
    return GroupEncoder(
        layers_cfg=layers_cfg,
        latent_dim=LATENT_DIM,
        output_irreps=f"{LATENT_DIM}x0e + 2x1o",
        readout="mean",
        transformer_type="se3",
        transformer_cfg={"num_layers": 1, "num_heads": 2, "num_channels": 4, "lmax": 1},
        area_pool=False,
        verbose=False,
    )


def _build_decoder():
    return FoldingDecoder(num_samples=NUM_SAMPLES, latent_dim=LATENT_DIM, n_freqs=2, verbose=False)


@pytest.fixture(scope="module")
def baseline_run():
    """Runs the pipeline ONCE (seeded, tiny, CPU) and returns every pinned quantity.

    device='cpu' (not the production CUDA-only gate in equivariant_gnn_train.py) so the
    test is deterministic and runs anywhere -- see test_trainer_e2e.py for the same
    convention. Everything upstream of the device (seeding, data, graph, model config)
    mirrors production.
    """
    key = _seed_everything()
    print(f"\n[baseline] seed={SEED}")

    shape_vertices, shape_mask = load_dataset(
        data_path=DATA_DIR, parts=None, load_fields=False,
        shuffle=True, verbose=False, seed=SEED,
    )
    print(f"[baseline] loaded {shape_vertices.shape[0]} shapes, "
          f"vertices padded to {shape_vertices.shape[1]}")

    graph, supergraph = build_training_graph(
        shape_vertices, shape_mask, key,
        r_max=R_MAX, dropout_rate=DROPOUT_RATE, use_supernodes=False,
    )
    print(f"[baseline] graph: num_nodes={graph.num_nodes} "
          f"num_edges={graph.edge_index.shape[1]} "
          f"shapes_in_batch={int(graph.batch.max()) + 1}")

    encoder = _build_encoder()
    decoder = _build_decoder()
    # Post-T10 the loss weights live on a composer, not on the stepper. These terms
    # reproduce the pre-T10 default exactly (recon + 0.1*kl, no contrastive), which
    # is why the pinned numbers below are unchanged by that refactor.
    composer = LossComposer([LossTerm("recon", 1.0), LossTerm("kl", 0.1)])
    stepper = TrainingStepper(encoder, decoder, learning_rate=1e-3,
                              composer=composer, device="cpu")
    print(f"[baseline] stepper device={stepper.device}")

    with torch.no_grad():
        enc_out = stepper.encode(graph, supergraph)
    mu_shape = tuple(enc_out.mu.shape)
    logvar_shape = tuple(enc_out.logvar.shape)
    print(f"[baseline] mu.shape={mu_shape} logvar.shape={logvar_shape}")

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_params += sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"[baseline] trainable params: encoder+decoder={n_params}")

    batch = (graph, supergraph, shape_vertices, shape_mask)
    steps = []
    pred_shape = None
    for step in range(NUM_STEPS):
        pred, loss, breakdown = stepper.train_step(*batch)
        if pred_shape is None:
            pred_shape = tuple(pred.shape)
            print(f"[baseline] pred.shape={pred_shape}")
        # Terms the composer skipped are absent from `breakdown` post-T10; they are
        # recorded as 0.0 to keep the pinned JSON's shape stable across the refactor
        # (`contrastive` genuinely contributes nothing on this single-view path).
        recon = breakdown.get("recon", 0.0)
        kl = breakdown.get("kl", 0.0)
        contrastive = breakdown.get("contrastive", 0.0)
        print(f"[baseline] step {step:2d} | loss={loss:.6f} recon={recon:.6f} "
              f"kl={kl:.6f} contrastive={contrastive:.6f}")
        steps.append({"loss": loss, "recon": recon, "kl": kl, "contrastive": contrastive})

    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.edge_index.shape[1],
        "mu_shape": list(mu_shape),
        "logvar_shape": list(logvar_shape),
        "pred_shape": list(pred_shape),
        "num_trainable_params": n_params,
        "steps": steps,
    }


@pytest.fixture(scope="module")
def expected():
    with open(EXPECTED_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Pinned characteristics -- each its own test, per INSTRUCTIONS.md T2 step 3.
# --------------------------------------------------------------------------- #
def test_graph_shape_is_pinned(baseline_run, expected):
    assert baseline_run["num_nodes"] == expected["num_nodes"]
    assert baseline_run["num_edges"] == expected["num_edges"]


def test_encoder_output_shapes_are_pinned(baseline_run, expected):
    assert baseline_run["mu_shape"] == expected["mu_shape"]
    assert baseline_run["logvar_shape"] == expected["logvar_shape"]


def test_decoder_output_shape_is_pinned(baseline_run, expected):
    assert baseline_run["pred_shape"] == expected["pred_shape"]


def test_trainable_param_count_is_pinned(baseline_run, expected):
    assert baseline_run["num_trainable_params"] == expected["num_trainable_params"]


def test_loss_sequence_is_pinned(baseline_run, expected):
    got, want = baseline_run["steps"], expected["steps"]
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        for key in ("loss", "recon", "kl", "contrastive"):
            assert g[key] == pytest.approx(w[key], rel=1e-5), (
                f"step {i} term {key!r}: got {g[key]!r}, expected {w[key]!r}"
            )
