"""Tests for LossComposer and frobenius_latent_loss (INSTRUCTIONS.md T8)."""

import math

import pytest
import torch

from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.losses.losses import frobenius_latent_loss


# --------------------------------------------------------------------------- #
# LossComposer
# --------------------------------------------------------------------------- #
def test_weighted_sum_and_breakdown():
    composer = LossComposer([
        LossTerm("recon", weight=1.0),
        LossTerm("kl", weight=0.5),
    ])
    values = {"recon": torch.tensor(2.0), "kl": torch.tensor(4.0)}

    total, breakdown = composer.compute(values)

    assert total.item() == pytest.approx(1.0 * 2.0 + 0.5 * 4.0)
    assert breakdown == {"recon": pytest.approx(2.0), "kl": pytest.approx(4.0)}


def test_none_valued_term_is_skipped_not_zero_filled():
    """The exact case INSTRUCTIONS.md T8 asks for: a composer with
    {"recon": t, "kl": None} must return recon alone and a breakdown with one key."""
    composer = LossComposer([
        LossTerm("recon", weight=1.0),
        LossTerm("kl", weight=0.1),
    ])
    t = torch.tensor(3.0)

    total, breakdown = composer.compute({"recon": t, "kl": None})

    assert total.item() == pytest.approx(3.0)
    assert breakdown == {"recon": pytest.approx(3.0)}
    assert len(breakdown) == 1
    assert "kl" not in breakdown


def test_term_missing_from_values_is_also_skipped():
    """A term absent from `values` entirely behaves the same as an explicit None."""
    composer = LossComposer([LossTerm("recon", weight=1.0), LossTerm("kl", weight=0.1)])

    total, breakdown = composer.compute({"recon": torch.tensor(1.5)})

    assert total.item() == pytest.approx(1.5)
    assert breakdown == {"recon": pytest.approx(1.5)}


def test_total_is_a_scalar_tensor():
    composer = LossComposer([LossTerm("recon", weight=1.0)])
    total, _ = composer.compute({"recon": torch.tensor(1.0)})
    assert total.dim() == 0


def test_all_terms_skipped_returns_scalar_zero():
    composer = LossComposer([LossTerm("recon", weight=1.0), LossTerm("kl", weight=0.1)])
    total, breakdown = composer.compute({"recon": None, "kl": None})
    assert total.dim() == 0
    assert total.item() == 0.0
    assert breakdown == {}


def test_non_finite_total_raises():
    composer = LossComposer([LossTerm("recon", weight=1.0)])
    with pytest.raises(FloatingPointError):
        composer.compute({"recon": torch.tensor(float("nan"))})


def test_plain_tuples_work_same_as_lossterm():
    """The sketch's contract is "(name, weight, kwargs)" -- plain tuples, not
    just LossTerm instances, must work."""
    composer = LossComposer([("recon", 2.0, {})])
    total, breakdown = composer.compute({"recon": torch.tensor(3.0)})
    assert total.item() == pytest.approx(6.0)
    assert breakdown == {"recon": pytest.approx(3.0)}


# --------------------------------------------------------------------------- #
# frobenius_latent_loss
# --------------------------------------------------------------------------- #
def test_frobenius_latent_loss_zero_for_zero_latent():
    z = torch.zeros(4, 5)
    assert frobenius_latent_loss(z).item() == 0.0


def test_frobenius_latent_loss_matches_manual_norm_over_batch():
    z = torch.tensor([[3.0, 4.0], [0.0, 0.0]])  # norms: 5.0, 0.0 -> squared: 25, 0
    expected = (25.0 + 0.0) / 2  # mean over batch, matching ||Z||_F^2 / B
    assert frobenius_latent_loss(z).item() == pytest.approx(expected)


def test_frobenius_latent_loss_is_batch_size_independent_in_expectation():
    """Doubling the batch by duplicating rows should leave the mean unchanged --
    this is the whole point of dividing by B instead of summing."""
    torch.manual_seed(0)
    z = torch.randn(6, 4)
    doubled = torch.cat([z, z], dim=0)
    assert frobenius_latent_loss(z).item() == pytest.approx(frobenius_latent_loss(doubled).item())
