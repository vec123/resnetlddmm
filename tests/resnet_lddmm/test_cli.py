"""Tests for CLI (STEP T17)."""

import tempfile
import os
from pathlib import Path

import pytest
import yaml

from src.resnet_lddmm.cli import load_config
from src.resnet_lddmm.config import ExperimentCfg


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self):
        """Verify load_config parses valid YAML and returns ExperimentCfg."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "valid.yaml")
            config_data = {
                "source": "data/hand/template.vtp",
                "target": "data/hand/template.vtp",
                "output_dir": tmpdir,
                "field": {
                    "kind": "time_varying",
                    "num_steps": 5,
                    "width": 128,
                },
                "code": {"kind": "none"},
                "loss": {"data_name": "l2", "sigma": 0.1},
                "train": {"steps": 10, "lr": 0.001, "seed": 42},
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            cfg = load_config(config_path)

            assert isinstance(cfg, ExperimentCfg)
            assert cfg.source == "data/hand/template.vtp"
            assert cfg.target == "data/hand/template.vtp"
            assert cfg.output_dir == tmpdir
            assert cfg.field.num_steps == 5
            assert cfg.field.width == 128
            assert cfg.code.kind == "none"
            assert cfg.loss.data_name == "l2"
            assert cfg.train.steps == 10
            assert cfg.train.seed == 42

    def test_load_config_file_not_found(self):
        """Verify load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/to/config.yaml")

    def test_load_config_invalid_yaml(self):
        """Verify load_config raises error for malformed YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "invalid.yaml")
            with open(config_path, "w") as f:
                f.write("{ invalid yaml content: [")

            with pytest.raises(Exception):  # yaml.YAMLError or ValueError
                load_config(config_path)

    def test_load_config_not_dict(self):
        """Verify load_config raises error if YAML is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "list.yaml")
            with open(config_path, "w") as f:
                yaml.dump(["item1", "item2"], f)

            with pytest.raises(ValueError, match="must be a YAML dict"):
                load_config(config_path)

    def test_load_config_missing_required_key(self):
        """Verify load_config raises error if required keys are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "incomplete.yaml")
            config_data = {
                "source": "data/hand/template.vtp",
                # Missing target and output_dir
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises((KeyError, TypeError)):
                load_config(config_path)

    def test_load_config_unknown_key(self):
        """Verify load_config raises error for unknown config keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "unknown_key.yaml")
            config_data = {
                "source": "data/hand/template.vtp",
                "target": "data/hand/template.vtp",
                "output_dir": tmpdir,
                "unknown_key": "should_fail",
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            with pytest.raises(ValueError, match="Unknown config keys"):
                load_config(config_path)

    def test_load_config_with_minimal_defaults(self):
        """Verify load_config uses defaults for optional sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "minimal.yaml")
            config_data = {
                "source": "data/hand/template.vtp",
                "target": "data/hand/template.vtp",
                "output_dir": tmpdir,
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            cfg = load_config(config_path)

            # Verify defaults are applied
            assert cfg.field.kind == "time_varying"
            assert cfg.field.num_steps == 10
            assert cfg.code.kind == "none"
            assert cfg.loss.data_name == "chamfer"
            assert cfg.train.steps == 2000

    def test_load_config_partial_subsections(self):
        """Verify load_config merges partial subsections with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "partial.yaml")
            config_data = {
                "source": "data/hand/template.vtp",
                "target": "data/hand/template.vtp",
                "output_dir": tmpdir,
                "field": {
                    "num_steps": 7,
                    # width, activation will use defaults
                },
                "train": {
                    "steps": 100,
                    # lr, seed, etc. will use defaults
                },
            }

            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            cfg = load_config(config_path)

            assert cfg.field.num_steps == 7
            assert cfg.field.width == 512  # default
            assert cfg.train.steps == 100
            assert cfg.train.lr == 1e-4  # default


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
SHIPPED_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


class TestShippedConfigs:
    """Every config in configs/ parses and carries sane hyperparameters.

    Discovered by glob rather than named one by one. The version of this that
    named configs/hand_pair_poc.yaml outlived the file -- it was deleted along
    with the other PoC configs, and the test then failed on a missing path
    rather than on anything about the code. A glob cannot rot that way: a
    deleted config simply stops being a case, and a new one is covered for free.
    """

    def test_configs_directory_is_not_empty(self):
        """Guard the glob itself -- zero cases would make the suite below vacuous."""
        assert SHIPPED_CONFIGS, f"no *.yaml found in {CONFIGS_DIR}"

    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    def test_shipped_config_loads(self, config_path):
        """Verify the config parses into an ExperimentCfg."""
        cfg = load_config(str(config_path))

        assert isinstance(cfg, ExperimentCfg)
        assert cfg.source, "config declares no source"
        assert cfg.target, "config declares no target"

    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda p: p.name)
    def test_shipped_config_has_reasonable_hyperparameters(self, config_path):
        """Verify the config's field / training / loss settings are in range."""
        cfg = load_config(str(config_path))

        # Field setup
        assert cfg.field.num_steps > 0
        assert cfg.field.width > 0
        assert cfg.field.activation in ("relu", "elu", "tanh")

        # Training setup
        assert cfg.train.steps > 0
        assert cfg.train.lr > 0
        assert cfg.loss.sigma > 0

        # Loss weights
        assert cfg.loss.kinetic_weight >= 0
        assert cfg.loss.code_reg_weight >= 0
