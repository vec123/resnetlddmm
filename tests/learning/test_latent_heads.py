"""Tests for the LatentHead strategies (INSTRUCTIONS.md T9).

The gaussian path's NUMBERS are pinned by tests/characterization (T9 step 3 is a
behavior-preserving move); these tests cover the CONTRACT both heads satisfy and
the new deterministic/auto-encoder mode that has no baseline yet.
"""

import math

import pytest
import torch
from torch_geometric.data import Data

from src.learning.models.group_encoder import GroupEncoder
from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.models.latent_heads import (
    GaussianLatentHead,
    DeterministicLatentHead,
    LatentHead,
)
from src.learning.registry import Registry
from src.learning.trainers.E3_end2end import TrainingStepper
from src.learning.losses.composer import LossComposer, LossTerm

LATENT_DIM = 4


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_batch():
    """A small 2-graph batch (mirrors tests/learning/test_trainer_e2e.py)."""
    torch.manual_seed(0)

    def ring(offset, n):
        s = torch.arange(n) + offset
        d = (torch.arange(n) + 1) % n + offset
        return torch.stack([torch.cat([s, d]), torch.cat([d, s])])

    edge_index = torch.cat([ring(0, 6), ring(6, 6)], dim=1)
    pos = torch.randn(12, 3)
    batch = torch.tensor([0] * 6 + [1] * 6)
    x = torch.ones(12, 1)

    graph = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
    true_verts = torch.randn(2, 10, 3)
    padding_mask = torch.ones(2, 10, dtype=torch.bool)
    return graph, None, true_verts, padding_mask


def make_encoder(latent_mode, readout="mean"):
    layers_cfg = [{
        "in_irreps": "1x0e",
        "target_irreps": "4x0e + 2x1o",
        "spatial_sh_lmax": 1,
        "interaction_sh_lmax": 4,
    }]
    return GroupEncoder(
        layers_cfg=layers_cfg, latent_dim=LATENT_DIM,
        output_irreps=f"{LATENT_DIM}x0e + 2x1o",
        readout=readout, latent_mode=latent_mode, verbose=False,
    )


def head_inputs(n_pool=6, num_graphs=2):
    """The (scalars, weights, batch, num_graphs) contract a head consumes."""
    torch.manual_seed(0)
    scalars = torch.randn(n_pool, LATENT_DIM)
    batch = torch.tensor([0] * (n_pool // 2) + [1] * (n_pool - n_pool // 2))
    # Per-shape weights summing to 1 within each shape, as the encoder produces.
    weights = torch.full((n_pool, 1), 2.0 / n_pool)
    return scalars, weights, batch, num_graphs


# --------------------------------------------------------------------------- #
# The shared contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("head_cls", [GaussianLatentHead, DeterministicLatentHead])
@pytest.mark.parametrize("readout", ["mean", "attention"])
def test_heads_emit_same_latent_shape(head_cls, readout):
    """Both heads emit [B, latent_dim] under both readouts -- that shared shape is
    what makes latent_mode a pure ablation switch for the decoder."""
    head = head_cls(LATENT_DIM, readout=readout)
    out = head(*head_inputs())

    latent = out.sample()
    assert latent.shape == (2, LATENT_DIM)


def test_gaussian_head_sets_mu_logvar_and_kl_is_finite():
    head = GaussianLatentHead(LATENT_DIM)
    out = head(*head_inputs())

    assert out.mu is not None and out.logvar is not None
    assert out.mu.shape == (2, LATENT_DIM)
    assert out.logvar.shape == (2, LATENT_DIM)
    kl = out.kl()
    assert kl is not None and torch.isfinite(kl)


def test_deterministic_head_sets_latent_and_has_no_kl():
    """The auto-encoder mode's defining property: no posterior, so kl() is None
    and the T8 composer skips a `kl` term without the trainer branching."""
    head = DeterministicLatentHead(LATENT_DIM)
    out = head(*head_inputs())

    assert out.latent is not None
    assert out.mu is None and out.logvar is None
    assert out.kl() is None


def test_deterministic_sample_is_deterministic():
    """No sampling: repeated sample() calls must be identical, in train mode too
    (the Gaussian head would inject fresh noise here)."""
    head = DeterministicLatentHead(LATENT_DIM)
    out = head(*head_inputs())
    assert torch.equal(out.sample(), out.sample())


def test_gaussian_sample_is_stochastic_but_deterministic_flag_pins_it():
    head = GaussianLatentHead(LATENT_DIM)
    out = head(*head_inputs())

    torch.manual_seed(1)
    a = out.sample()
    torch.manual_seed(2)
    b = out.sample()
    assert not torch.equal(a, b), "reparameterized sample should vary with RNG"
    assert torch.equal(out.sample(deterministic=True), out.mu)


def test_unknown_readout_rejected():
    with pytest.raises(ValueError, match="readout"):
        GaussianLatentHead(LATENT_DIM, readout="banana")


def test_latent_head_base_is_abstract():
    with pytest.raises(TypeError):
        LatentHead(LATENT_DIM)


# --------------------------------------------------------------------------- #
# Registry wiring (T9 step 6)
# --------------------------------------------------------------------------- #
def test_both_heads_are_registered():
    assert Registry.available("latent_head") == ["deterministic", "gaussian"]


@pytest.mark.parametrize("name,expected", [
    ("gaussian", GaussianLatentHead),
    ("deterministic", DeterministicLatentHead),
])
def test_registry_creates_each_head(name, expected):
    head = Registry.create("latent_head", name, latent_dim=LATENT_DIM)
    assert isinstance(head, expected)


# --------------------------------------------------------------------------- #
# GroupEncoder selects its head from latent_mode (T9 step 5)
# --------------------------------------------------------------------------- #
def test_encoder_gaussian_mode_produces_mu():
    encoder = make_encoder("gaussian")
    assert isinstance(encoder.latent_head, GaussianLatentHead)

    graph, supergraph, _, _ = make_batch()
    out = encoder(graph, supergraph)
    assert out.mu is not None and out.mu.shape == (2, LATENT_DIM)
    assert out.kl() is not None


def test_encoder_deterministic_mode_produces_latent_and_pose():
    """The pose fields must survive the head swap -- forward attaches them to
    whatever the head returned, without knowing which head it holds."""
    encoder = make_encoder("deterministic")
    assert isinstance(encoder.latent_head, DeterministicLatentHead)

    graph, supergraph, _, _ = make_batch()
    out = encoder(graph, supergraph)
    assert out.latent is not None and out.latent.shape == (2, LATENT_DIM)
    assert out.mu is None
    assert out.kl() is None
    assert out.rotation.shape == (2, 3, 3)
    assert out.translation.shape == (2, 3)


def test_encoder_rejects_unknown_latent_mode():
    with pytest.raises(ValueError, match="latent_head"):
        make_encoder("banana")


# --------------------------------------------------------------------------- #
# End-to-end: the auto-encoder mode trains (T9's headline deliverable)
# --------------------------------------------------------------------------- #
def test_deterministic_mode_trains_end_to_end():
    """The T9+T10 headline: an auto-encoder run trains through the SAME trainer as
    the VAE, with no branch anywhere -- EncoderOutput.sample() hands back the
    deterministic latent and kl() returns None, so the composer just skips KL."""
    encoder = make_encoder("deterministic")
    decoder = FoldingDecoder(num_samples=16, latent_dim=LATENT_DIM, n_freqs=2, verbose=False)
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=1e-2, device="cpu",
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("frobenius", 0.01)]))

    before = decoder.fold2.weight.detach().clone()
    pred, loss, breakdown = stepper.train_step(*make_batch())

    assert pred.shape == (2, 16, 3)
    assert math.isfinite(loss)
    assert set(breakdown) == {"recon", "frobenius"}, "AE mode must regularize by Frobenius, not KL"
    assert not torch.allclose(before, decoder.fold2.weight.detach()), "no gradient reached the decoder"


def test_gaussian_and_deterministic_share_one_trainer_path():
    """latent_mode is a pure ablation switch: the only difference visible to the
    trainer is which regularizer term shows up in the breakdown."""
    terms_by_mode = {}
    for mode, reg in [("gaussian", LossTerm("kl", 0.1)),
                      ("deterministic", LossTerm("frobenius", 0.01))]:
        encoder = make_encoder(mode)
        decoder = FoldingDecoder(num_samples=16, latent_dim=LATENT_DIM, n_freqs=2, verbose=False)
        stepper = TrainingStepper(
            encoder, decoder, learning_rate=1e-2, device="cpu",
            composer=LossComposer([LossTerm("recon", 1.0), reg]))
        _, loss, breakdown = stepper.train_step(*make_batch())
        assert math.isfinite(loss)
        terms_by_mode[mode] = set(breakdown)

    assert terms_by_mode == {"gaussian": {"recon", "kl"},
                             "deterministic": {"recon", "frobenius"}}


def test_kl_term_is_skipped_in_deterministic_mode():
    """Even if a config slips a `kl` term past validation, an AE run can't produce
    one -- kl() is None, so the composer drops it instead of crashing."""
    encoder = make_encoder("deterministic")
    decoder = FoldingDecoder(num_samples=16, latent_dim=LATENT_DIM, n_freqs=2, verbose=False)
    stepper = TrainingStepper(
        encoder, decoder, learning_rate=1e-2, device="cpu",
        composer=LossComposer([LossTerm("recon", 1.0), LossTerm("kl", 0.1)]))

    _, loss, breakdown = stepper.train_step(*make_batch())
    assert math.isfinite(loss)
    assert "kl" not in breakdown