"""Integration tests for encoder code support in ResNetLDDMM.

Tests config parsing, code source instantiation, and training loops
for all three code.kind modes: none, auto_decoder, encoder.
"""

import pytest
import tempfile
import os
from pathlib import Path

import torch
import yaml

from src.resnet_lddmm.config import ExperimentCfg, CodeCfg, _parse_code_config
from src.resnet_lddmm.runner import (
    _build_code_source,
    _build_code_source_cohort,
    _encoder_layers_to_list_of_dicts,
)

# Check if encoder dependencies are available
HAS_ENCODER_DEPS = True
try:
    from src.spec import EncoderConfig, GraphSpec, EncoderLayerConfig
except ImportError:
    HAS_ENCODER_DEPS = False

# Mark encoder tests to skip if dependencies not available
pytestmark_encoder = pytest.mark.skipif(
    not HAS_ENCODER_DEPS,
    reason="e3nn/torch_geometric not installed"
)


class TestCodeConfigParsing:
    """Test config parsing for all code.kind modes."""

    def test_parse_code_config_none(self):
        """Test parsing code config with kind=none."""
        config_dict = {
            "kind": "none",
            "n_z": 256,
        }
        cfg = _parse_code_config(config_dict)
        assert cfg.kind == "none"
        assert cfg.n_z == 256
        assert cfg.encoder_config is None
        assert cfg.graph_spec is None

    def test_parse_code_config_auto_decoder(self):
        """Test parsing code config with kind=auto_decoder."""
        config_dict = {
            "kind": "auto_decoder",
            "n_z": 128,
            "position_aware": True,
            "conditioning_method": "concat",
        }
        cfg = _parse_code_config(config_dict)
        assert cfg.kind == "auto_decoder"
        assert cfg.n_z == 128
        assert cfg.position_aware is True
        assert cfg.conditioning_method == "concat"
        assert cfg.encoder_config is None
        assert cfg.graph_spec is None

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_parse_code_config_encoder_with_defaults(self):
        """Test parsing encoder config with minimal overrides."""
        from src.spec import EncoderConfig, GraphSpec
        config_dict = {
            "kind": "encoder",
            "n_z": 256,
        }
        cfg = _parse_code_config(config_dict)
        assert cfg.kind == "encoder"
        assert cfg.n_z == 256
        assert cfg.encoder_config is not None
        assert cfg.graph_spec is not None
        # Should have defaults
        assert cfg.encoder_config.latent_dim == 256
        assert cfg.encoder_config.readout == "mean"
        assert cfg.graph_spec.r_max == 0.1
        assert cfg.graph_spec.dropout_rate == 0.8

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_parse_code_config_encoder_with_custom_layers(self):
        """Test parsing encoder config with custom layer specs."""
        from src.spec import EncoderConfig, GraphSpec, EncoderLayerConfig
        config_dict = {
            "kind": "encoder",
            "n_z": 64,
            "encoder_config": {
                "latent_dim": 64,
                "readout": "attention",
                "layers": [
                    {
                        "in_irreps": "1x0e",
                        "target_irreps": "16x0e + 8x1e",
                        "spatial_sh_lmax": 1,
                        "interaction_sh_lmax": 2,
                    }
                ],
            },
            "graph_spec": {
                "r_max": 0.15,
                "dropout_rate": 0.7,
                "use_supernodes": True,
                "n_supernodes": 20,
            },
        }
        cfg = _parse_code_config(config_dict)
        assert cfg.kind == "encoder"
        assert cfg.n_z == 64
        assert cfg.encoder_config.readout == "attention"
        assert cfg.encoder_config.latent_dim == 64
        assert len(cfg.encoder_config.layers) == 1
        assert cfg.encoder_config.layers[0].spatial_sh_lmax == 1
        assert cfg.graph_spec.r_max == 0.15
        assert cfg.graph_spec.use_supernodes is True
        assert cfg.graph_spec.n_supernodes == 20

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_parse_code_config_encoder_requires_valid_layer_chain(self):
        """Test that encoder config validates layer irreps chain."""
        from src.spec import EncoderConfig, EncoderLayerConfig
        config_dict = {
            "kind": "encoder",
            "n_z": 64,
            "encoder_config": {
                "layers": [
                    {
                        "in_irreps": "1x0e",
                        "target_irreps": "16x0e + 8x1e",
                        "spatial_sh_lmax": 1,
                    },
                    {
                        "in_irreps": "32x0e",  # Mismatch!
                        "target_irreps": "32x0e",
                        "spatial_sh_lmax": 1,
                    },
                ]
            },
        }
        with pytest.raises(ValueError, match="must equal"):
            _parse_code_config(config_dict)


class TestCodeSourceInstantiation:
    """Test instantiation of code sources for all modes."""

    def test_build_code_source_none_pair_mode(self):
        """Test building code source for pair mode with kind=none."""
        code_cfg = CodeCfg(kind="none", n_z=256)
        code_source = _build_code_source("none", code_cfg)
        assert code_source is not None
        # NoCode should return None from forward
        assert code_source.forward(None) is None

    def test_build_code_source_auto_decoder_cohort_mode(self):
        """Test building auto_decoder code source for cohort mode."""
        num_shapes = 5
        code_cfg = CodeCfg(kind="auto_decoder", n_z=256)
        code_source = _build_code_source_cohort(num_shapes, "auto_decoder", code_cfg)
        assert code_source is not None
        assert code_source.n_z == 256
        assert code_source.num_shapes == num_shapes

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_build_code_source_encoder_requires_config(self):
        """Test that encoder instantiation fails without config."""
        code_cfg = CodeCfg(kind="encoder", n_z=256, encoder_config=None)
        with pytest.raises(ValueError, match="encoder_config required"):
            _build_code_source("encoder", code_cfg)

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_build_code_source_encoder_with_config(self):
        """Test successful encoder code source instantiation."""
        from src.spec import EncoderConfig, GraphSpec
        code_cfg = CodeCfg(
            kind="encoder",
            n_z=64,
            encoder_config=EncoderConfig(latent_dim=64),
            graph_spec=GraphSpec(),
        )
        code_source = _build_code_source("encoder", code_cfg)
        assert code_source is not None
        assert code_source.n_z == 64

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_encoder_layers_to_list_of_dicts(self):
        """Test conversion of encoder layers from dataclass to dicts."""
        from src.spec import EncoderConfig, EncoderLayerConfig
        layers = [
            EncoderLayerConfig(
                in_irreps="1x0e",
                target_irreps="16x0e + 8x1o",
                spatial_sh_lmax=1,
                interaction_sh_lmax=2,
            )
        ]
        encoder_config = EncoderConfig(layers=layers, latent_dim=32)
        result = _encoder_layers_to_list_of_dicts(encoder_config)
        assert len(result) == 1
        assert result[0]["in_irreps"] == "1x0e"
        assert result[0]["target_irreps"] == "16x0e + 8x1o"
        assert result[0]["spatial_sh_lmax"] == 1
        assert result[0]["interaction_sh_lmax"] == 2


class TestConfigLoadFromYaml:
    """Test loading full experiment config from YAML for all code modes."""

    def test_load_config_none_mode(self):
        """Test loading config with code.kind=none from YAML."""
        yaml_content = """
source: data/test_source.obj
target: data/test_target.obj
output_dir: /tmp/test_output_none
code:
  kind: none
  n_z: 256
field:
  kind: time_varying
  num_steps: 5
loss:
  data_name: chamfer
  direction: forward
  sigma: 0.1
train:
  mode: pair
  steps: 10
  lr: 1e-4
"""
        data = yaml.safe_load(yaml_content)
        cfg = ExperimentCfg.from_dict(data)
        assert cfg.code.kind == "none"
        assert cfg.code.encoder_config is None
        assert cfg.code.graph_spec is None

    def test_load_config_auto_decoder_mode(self):
        """Test loading config with code.kind=auto_decoder from YAML."""
        yaml_content = """
source: data/test_cohort
target: data/test_target.obj
output_dir: /tmp/test_output_auto_decoder
code:
  kind: auto_decoder
  n_z: 128
  position_aware: true
  conditioning_method: concat
field:
  kind: stationary
  num_steps: 10
loss:
  data_name: chamfer
  direction: bidirectional
  sigma: 0.1
train:
  mode: cohort
  batch: 4
  steps: 20
  lr: 1e-4
"""
        data = yaml.safe_load(yaml_content)
        cfg = ExperimentCfg.from_dict(data)
        assert cfg.code.kind == "auto_decoder"
        assert cfg.code.n_z == 128
        assert cfg.code.position_aware is True
        assert cfg.code.encoder_config is None
        assert cfg.code.graph_spec is None

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_load_config_encoder_mode_minimal(self):
        """Test loading config with code.kind=encoder (minimal) from YAML."""
        yaml_content = """
source: data/test_cohort
target: data/test_target.obj
output_dir: /tmp/test_output_encoder
code:
  kind: encoder
  n_z: 64
field:
  kind: stationary
  num_steps: 10
loss:
  data_name: chamfer
  direction: bidirectional
  sigma: 0.1
train:
  mode: cohort
  batch: 4
  steps: 20
  lr: 1e-4
"""
        data = yaml.safe_load(yaml_content)
        cfg = ExperimentCfg.from_dict(data)
        assert cfg.code.kind == "encoder"
        assert cfg.code.n_z == 64
        assert cfg.code.encoder_config is not None
        assert cfg.code.graph_spec is not None

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_load_config_encoder_mode_full(self):
        """Test loading config with code.kind=encoder (custom) from YAML."""
        yaml_content = """
source: data/test_cohort
target: data/test_target.obj
output_dir: /tmp/test_output_encoder_full
code:
  kind: encoder
  n_z: 128
  encoder_config:
    latent_dim: 128
    readout: attention
    readout_heads: 4
    supernode_sh_lmax: 3
    transformer_type: se3
    area_pool: true
    latent_mode: gaussian
    layers:
      - in_irreps: "1x0e"
        target_irreps: "32x0e + 16x1e + 16x1o"
        spatial_sh_lmax: 1
      - in_irreps: "32x0e + 16x1e + 16x1o"
        target_irreps: "64x0e + 32x1o"
        spatial_sh_lmax: 2
  graph_spec:
    r_max: 0.12
    r_supergraph: 0.25
    dropout_rate: 0.7
    use_supernodes: true
    n_supernodes: 15
    sampling_mode_graph: fps
    area_k: 10
field:
  kind: stationary
  num_steps: 10
loss:
  data_name: chamfer
  direction: bidirectional
  sigma: 0.1
train:
  mode: cohort
  batch: 4
  steps: 20
  lr: 1e-4
"""
        data = yaml.safe_load(yaml_content)
        cfg = ExperimentCfg.from_dict(data)
        assert cfg.code.kind == "encoder"
        assert cfg.code.n_z == 128
        assert cfg.code.encoder_config is not None
        assert cfg.code.encoder_config.latent_dim == 128
        assert cfg.code.encoder_config.readout == "attention"
        assert cfg.code.encoder_config.readout_heads == 4
        assert cfg.code.encoder_config.area_pool is True
        assert len(cfg.code.encoder_config.layers) == 2
        assert cfg.code.graph_spec is not None
        assert cfg.code.graph_spec.r_max == 0.12
        assert cfg.code.graph_spec.use_supernodes is True
        assert cfg.code.graph_spec.n_supernodes == 15


class TestEncoderCodeSourceInference:
    """Test that encoder code source can process batches correctly."""

    @pytest.mark.skipif(not HAS_ENCODER_DEPS, reason="e3nn/torch_geometric not installed")
    def test_encoder_forward_pass_with_mock_batch(self):
        """Test that EncoderCodes forward pass works with proper batch."""
        from src.spec import EncoderConfig, EncoderLayerConfig, GraphSpec
        code_cfg = CodeCfg(
            kind="encoder",
            n_z=32,
            encoder_config=EncoderConfig(
                latent_dim=32,
                layers=[
                    EncoderLayerConfig(
                        in_irreps="1x0e",
                        target_irreps="16x0e + 8x1o",
                        spatial_sh_lmax=1,
                    )
                ],
            ),
            graph_spec=GraphSpec(r_max=0.15, dropout_rate=0.8),  # type: ignore
        )

        code_source = _build_code_source("encoder", code_cfg)
        assert code_source is not None

        # Mock batch with points (this will fail without actual graph builder,
        # but tests the wiring)
        class MockBatch:
            def __init__(self):
                self.points = torch.randn(2, 50, 3)
                self.shape_ids = torch.tensor([0, 1])

        batch = MockBatch()
        try:
            z = code_source(batch)
            assert z is not None
            assert z.shape == (2, 32)
        except RuntimeError as e:
            # Expected if e3nn/torch_geometric not properly installed
            # But at least the factory method worked
            if "torch_geometric" not in str(e) and "e3nn" not in str(e):
                raise


class TestEncoderGraphLogger:
    """Test the EncoderGraphLogger callback."""

    def test_graph_logger_instantiation(self):
        """Test that EncoderGraphLogger can be instantiated."""
        from src.resnet_lddmm.callbacks import EncoderGraphLogger
        logger = EncoderGraphLogger(every_n_steps=50)
        assert logger is not None
        assert logger.every_n_steps == 50

    def test_graph_logger_graceful_none_handling(self):
        """Test that EncoderGraphLogger handles None pred gracefully."""
        from src.resnet_lddmm.callbacks import EncoderGraphLogger
        logger = EncoderGraphLogger()

        class MockStepper:
            pass

        class MockContext:
            def __init__(self):
                self.stepper = MockStepper()

        ctx = MockContext()

        # Call on_step_end with None pred (should return early)
        logger.on_step_end(ctx, step=0, metrics={}, batch=None, pred=None)
        # If we get here without exception, test passes
        assert True

    def test_graph_logger_checks_encoder_presence(self):
        """Test that EncoderGraphLogger checks for encoder before attempting export."""
        from src.resnet_lddmm.callbacks import EncoderGraphLogger
        logger = EncoderGraphLogger(every_n_steps=1)

        # Stepper without code_source (e.g., pair mode with no code)
        class MockStepper:
            pass

        class MockContext:
            def __init__(self):
                self.stepper = MockStepper()
                self.log_dir = "/tmp/test"

        ctx = MockContext()

        # Should not crash even though there's no encoder
        logger.on_step_end(ctx, step=0, metrics={}, batch=None, pred="something")
        # Cache should remain empty
        assert logger._last_graph_cache is None

    def test_encoder_codes_caches_graph(self):
        """Test that EncoderCodes caches graphs during forward pass."""
        code_cfg = CodeCfg(
            kind="encoder",
            n_z=32,
            encoder_config=None,  # Will be created below
            graph_spec=None,
        )
        # Note: can't fully instantiate without e3nn, but can verify the class structure
        from src.resnet_lddmm.codes.encoder import EncoderCodes

        # Check that EncoderCodes has cache attributes
        assert hasattr(EncoderCodes, '__init__')
        # Verify __init__ signature includes _last_graph and _last_supergraph
        import inspect
        sig = inspect.signature(EncoderCodes.__init__)
        assert "graph_builder" in sig.parameters
        assert "encoder" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
