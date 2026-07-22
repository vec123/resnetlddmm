"""
Tests for config/models.py:
ExperimentConfig.validate()
and GraphSpec, the parameter object it embeds.
"""

import copy
import warnings

import pytest
import torch

from config.config_fields import (
    ExperimentConfig,
    LossTermConfig,
    GraphSpec,
)


# --------------------------------------------------------------------------- #
# ExperimentConfig.validate()
# --------------------------------------------------------------------------- #
def test_default_config_is_valid():
    cfg = ExperimentConfig()
    cfg.validate()  # must not raise


def test_deterministic_latent_mode_with_kl_term_raises():
    cfg = ExperimentConfig()
    cfg.encoder.latent_mode = "deterministic"
    cfg.training.losses.terms = [
        LossTermConfig(name="recon"),
        LossTermConfig(name="kl", weight=0.1),
    ]
    with pytest.raises(ValueError, match="deterministic"):
        cfg.validate()


def test_deterministic_latent_mode_without_kl_term_is_valid():
    cfg = ExperimentConfig()
    cfg.encoder.latent_mode = "deterministic"
    cfg.training.losses.terms = [
        LossTermConfig(name="recon"),
        LossTermConfig(name="frobenius", weight=0.01),
    ]
    cfg.validate()  # must not raise


def test_unknown_loss_term_raises():
    cfg = ExperimentConfig()
    cfg.training.losses.terms = [LossTermConfig(name="banana")]
    with pytest.raises(ValueError, match="unknown loss term"):
        cfg.validate()


def test_unknown_decoder_type_raises():
    cfg = ExperimentConfig()
    cfg.decoder.decoder_type = "banana"
    with pytest.raises(ValueError, match="decoder_type"):
        cfg.validate()


def test_non_square_num_samples_with_folding_decoder_raises():
    cfg = ExperimentConfig()
    cfg.decoder.decoder_type = "folding"
    cfg.decoder.num_samples = 250  # not a perfect square
    with pytest.raises(ValueError, match="perfect square"):
        cfg.validate()


def test_non_square_num_samples_with_sphere_folding_decoder_is_valid():
    """SphereFoldingDecoder's Fibonacci-sphere base has no grid requirement --
    the perfect-square check must NOT fire for decoder_type='sphere_folding'."""
    cfg = ExperimentConfig()
    cfg.decoder.decoder_type = "sphere_folding"
    cfg.decoder.num_samples = 250
    cfg.validate()  # must not raise


def test_large_frobenius_and_contrastive_weights_warn():
    cfg = ExperimentConfig()
    cfg.training.losses.terms = [
        LossTermConfig(name="recon"),
        LossTermConfig(name="frobenius", weight=2.0),
        LossTermConfig(name="contrastive", weight=2.0),
    ]
    with pytest.warns(UserWarning, match="frobenius"):
        cfg.validate()


def test_small_frobenius_and_contrastive_weights_do_not_warn():
    cfg = ExperimentConfig()
    cfg.training.losses.terms = [
        LossTermConfig(name="recon"),
        LossTermConfig(name="frobenius", weight=0.01),
        LossTermConfig(name="contrastive", weight=0.01),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        cfg.validate()


# --------------------------------------------------------------------------- #
# GraphSpec
# --------------------------------------------------------------------------- #
def test_graph_spec_defaults_match_build_training_graph():
    spec = GraphSpec()
    assert spec.r_max == 0.1
    assert spec.dropout_rate == 0.8
    assert spec.n_supernodes == 10
    assert spec.use_supernodes is False
    assert spec.area_k == 8


def test_graph_spec_rejects_inverted_range():
    with pytest.raises(ValueError, match="low <= high"):
        GraphSpec(r_max=(0.5, 0.1))


def test_graph_spec_resolve_is_reproducible_from_seed():
    spec = GraphSpec(r_max=(0.1, 0.3), dropout_rate=(0.7, 0.9))

    rng_a = torch.Generator().manual_seed(0)
    rng_b = torch.Generator().manual_seed(0)
    resolved_a = spec.resolve(rng_a)
    resolved_b = spec.resolve(rng_b)

    assert resolved_a.r_max == resolved_b.r_max
    assert resolved_a.dropout_rate == resolved_b.dropout_rate


def test_graph_spec_resolve_leaves_fixed_fields_untouched():
    spec = GraphSpec(r_max=0.2, n_supernodes=15, sampling_mode_graph="fps")
    resolved = spec.resolve(rng=None)
    assert resolved == spec  # no range fields -> resolves to an equal spec
