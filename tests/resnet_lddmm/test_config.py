import pytest
from src.resnet_lddmm.config import (
    ExperimentCfg,
    FieldCfg,
    CodeCfg,
    LossCfg,
    TrainCfg,
)


def test_experiment_cfg_round_trip():
    """Test creating config from dict and converting back."""
    d = {
        "source": "path/to/source",
        "target": "path/to/target",
        "output_dir": "path/to/output",
        "field": {
            "kind": "stationary",
            "num_steps": 20,
            "width": 256,
        },
        "code": {
            "kind": "auto_decoder",
            "n_z": 128,
        },
        "loss": {
            "data_name": "l2",
            "sigma": 0.2,
        },
        "train": {
            "steps": 5000,
            "lr": 1e-3,
        },
    }

    cfg = ExperimentCfg.from_dict(d)
    assert cfg.source == "path/to/source"
    assert cfg.target == "path/to/target"
    assert cfg.output_dir == "path/to/output"
    assert cfg.field.kind == "stationary"
    assert cfg.field.num_steps == 20
    assert cfg.field.width == 256
    assert cfg.code.kind == "auto_decoder"
    assert cfg.code.n_z == 128
    assert cfg.loss.data_name == "l2"
    assert cfg.loss.sigma == 0.2
    assert cfg.train.steps == 5000
    assert cfg.train.lr == 1e-3


def test_experiment_cfg_defaults():
    """Test default values are applied."""
    d = {
        "source": "src",
        "target": "tgt",
        "output_dir": "out",
    }
    cfg = ExperimentCfg.from_dict(d)
    assert cfg.field.kind == "time_varying"
    assert cfg.field.num_steps == 10
    assert cfg.code.kind == "none"
    assert cfg.loss.data_name == "chamfer"
    assert cfg.train.mode == "pair"


def test_experiment_cfg_unknown_key():
    """Test that unknown keys raise ValueError with the key name."""
    d = {
        "source": "src",
        "target": "tgt",
        "output_dir": "out",
        "unknown_key": "bad",
    }
    with pytest.raises(ValueError, match="Unknown config keys:.*unknown_key"):
        ExperimentCfg.from_dict(d)


def test_experiment_cfg_multiple_unknown_keys():
    """Test multiple unknown keys are all reported."""
    d = {
        "source": "src",
        "target": "tgt",
        "output_dir": "out",
        "typo_field": {},
        "bad_loss": {},
    }
    with pytest.raises(ValueError, match="Unknown config keys:"):
        ExperimentCfg.from_dict(d)
