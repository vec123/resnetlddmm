"""Test EquivariantStationaryField integration with config system and runner."""

import torch
import pytest
from src.resnet_lddmm.config import FieldCfg, ExperimentCfg, CodeCfg, LossCfg, TrainCfg, AugmentationCfg
from src.resnet_lddmm import registrations  # Side effect: register components
from src.learning.registry import Registry
from src.resnet_lddmm.conditioning.base import NoConditioning


class TestEquivariantFieldConfig:
    """Test EquivariantStationaryField in config system."""

    def test_field_config_defaults(self):
        """FieldCfg should have equivariant defaults."""
        cfg = FieldCfg(kind="equivariant_stationary")
        assert cfg.kind == "equivariant_stationary"
        assert cfg.hidden_irreps == "4x0e + 2x1o"
        assert cfg.gate_hidden_dim == 64
        assert cfg.use_tensor_product_self is True

    def test_field_config_custom(self):
        """FieldCfg should accept custom equivariant parameters."""
        cfg = FieldCfg(
            kind="equivariant_stationary",
            hidden_irreps="6x0e + 3x1o",
            gate_hidden_dim=128,
            use_tensor_product_self=False
        )
        assert cfg.hidden_irreps == "6x0e + 3x1o"
        assert cfg.gate_hidden_dim == 128
        assert cfg.use_tensor_product_self is False

    def test_registry_resolves_equivariant_field(self):
        """Registry should resolve equivariant_stationary to the correct class."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        assert field is not None
        assert hasattr(field, "forward")
        # Should be EquivariantStationaryField instance
        assert field.__class__.__name__ == "EquivariantStationaryField"

    def test_registry_with_config_params(self):
        """Registry should accept equivariant config parameters."""
        field = Registry.create(
            "field", "equivariant_stationary",
            hidden_irreps="6x0e + 3x1o",
            gate_hidden_dim=128,
            use_tensor_product_self=True,
            conditioning=NoConditioning()
        )
        assert field is not None
        # Verify it can run forward pass
        x = torch.randn(2, 50, 3)
        v = field(x)
        assert v.shape == (2, 50, 3)

    def test_experiment_config_with_equivariant_field(self):
        """ExperimentCfg should accept equivariant_stationary field config."""
        cfg_dict = {
            "source": "dummy.ply",
            "target": "dummy.ply",
            "output_dir": "outputs/test",
            "field": {
                "kind": "equivariant_stationary",
                "hidden_irreps": "4x0e + 2x1o",
                "gate_hidden_dim": 64,
            },
            "train": {
                "mode": "pair"
            }
        }
        cfg = ExperimentCfg.from_dict(cfg_dict)
        assert cfg.field.kind == "equivariant_stationary"
        assert cfg.field.hidden_irreps == "4x0e + 2x1o"
        assert cfg.field.gate_hidden_dim == 64
        assert cfg.train.mode == "pair"


class TestEquivariantFieldForwardPass:
    """Test EquivariantStationaryField forward pass with various input configurations."""

    def test_forward_no_conditioning(self):
        """Forward pass without conditioning."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        x = torch.randn(4, 50, 3)
        v = field(x)
        assert v.shape == (4, 50, 3)
        assert not torch.isnan(v).any()

    def test_forward_multiple_batch_sizes(self):
        """Forward pass with different batch sizes."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        for B in [1, 2, 8]:
            x = torch.randn(B, 100, 3)
            v = field(x)
            assert v.shape == (B, 100, 3)

    def test_forward_large_hidden_irreps(self):
        """Forward pass with larger hidden irreps."""
        field = Registry.create(
            "field", "equivariant_stationary",
            hidden_irreps="8x0e + 4x1o",
            gate_hidden_dim=128,
            conditioning=NoConditioning()
        )
        x = torch.randn(2, 50, 3)
        v = field(x)
        assert v.shape == (2, 50, 3)

    def test_forward_without_self_interaction(self):
        """Forward pass with use_tensor_product_self=False."""
        field = Registry.create(
            "field", "equivariant_stationary",
            use_tensor_product_self=False,
            conditioning=NoConditioning()
        )
        x = torch.randn(2, 50, 3)
        v = field(x)
        assert v.shape == (2, 50, 3)
        assert not torch.isnan(v).any()

    def test_forward_step_ignored(self):
        """Step parameter should be ignored (stationary field)."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        x = torch.randn(2, 50, 3)
        v1 = field(x, step=0)
        v2 = field(x, step=5)
        v3 = field(x, step=None)
        assert torch.allclose(v1, v2) and torch.allclose(v2, v3)


class TestEquivariantFieldGradients:
    """Test gradient flow through equivariant field."""

    def test_gradient_flow_to_parameters(self):
        """Gradients should flow to field parameters."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        x = torch.randn(2, 50, 3, requires_grad=True)

        v = field(x)
        loss = v.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert any(p.grad is not None for p in field.parameters())

    def test_training_step(self):
        """Field should support a training step."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        optimizer = torch.optim.Adam(field.parameters(), lr=1e-4)

        for step in range(3):
            x = torch.randn(2, 50, 3)
            v = field(x)
            loss = v.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_device_placement(self):
        """Field should support device placement."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        x = torch.randn(2, 50, 3)

        # CPU
        v_cpu = field(x)
        assert v_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            field_cuda = field.cuda()
            x_cuda = x.cuda()
            v_cuda = field_cuda(x_cuda)
            assert v_cuda.device.type == "cuda"


class TestEquivariantFieldEvalMode:
    """Test equivariant field in evaluation mode."""

    def test_eval_mode_deterministic(self):
        """Eval mode should be deterministic."""
        field = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        field.eval()

        x = torch.randn(2, 50, 3)

        with torch.no_grad():
            v1 = field(x)
            v2 = field(x)

        assert torch.allclose(v1, v2)

    def test_state_dict_roundtrip(self):
        """Field should be serializable."""
        field1 = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        x = torch.randn(2, 50, 3)
        v1 = field1(x)

        # Save and load
        state = field1.state_dict()
        field2 = Registry.create(
            "field", "equivariant_stationary",
            conditioning=NoConditioning()
        )
        field2.load_state_dict(state)
        v2 = field2(x)

        assert torch.allclose(v1, v2)


# Integration with runner (mock test without actual file I/O)
class TestEquivariantFieldRunnerIntegration:
    """Test EquivariantStationaryField integration with runner.build."""

    def test_config_can_specify_equivariant_field(self):
        """ExperimentCfg should support equivariant_stationary specification."""
        cfg_dict = {
            "source": "dummy.ply",
            "target": "dummy.ply",
            "output_dir": "outputs/test",
            "field": {
                "kind": "equivariant_stationary",
                "num_steps": 10,
                "hidden_irreps": "4x0e + 2x1o",
                "gate_hidden_dim": 64,
                "use_tensor_product_self": True,
            },
            "code": {
                "kind": "none"
            },
            "loss": {
                "data_name": "chamfer",
                "direction": "forward",
            },
            "train": {
                "mode": "pair",
                "steps": 100,
                "batch": 1,
            }
        }
        cfg = ExperimentCfg.from_dict(cfg_dict)

        # Verify all equivariant parameters are in config
        assert cfg.field.kind == "equivariant_stationary"
        assert cfg.field.hidden_irreps == "4x0e + 2x1o"
        assert cfg.field.gate_hidden_dim == 64
        assert cfg.field.use_tensor_product_self is True

    def test_all_field_kinds_resolve(self):
        """All field kinds should resolve via Registry."""
        for kind in ["time_varying", "stationary", "equivariant_stationary"]:
            if kind == "time_varying":
                field = Registry.create(
                    "field", kind,
                    num_blocks=10,
                    width=512,
                    conditioning=NoConditioning()
                )
            elif kind == "stationary":
                field = Registry.create(
                    "field", kind,
                    conditioning=NoConditioning()
                )
            elif kind == "equivariant_stationary":
                field = Registry.create(
                    "field", kind,
                    conditioning=NoConditioning()
                )

            assert field is not None
            assert hasattr(field, "forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
