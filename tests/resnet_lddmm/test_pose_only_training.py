"""Tests for pose-only training mode (freeze_flow_at_identity feature)."""

import os
import pytest
import torch
import torch.nn as nn
from src.resnet_lddmm.config import ExperimentCfg
from src.resnet_lddmm.flow import NeuralODEFlow, IdentityFlowWrapper
from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.integrators import ForwardEuler


class DummyVelocityField(VelocityField):
    """Minimal velocity field for testing (always returns zeros)."""
    def forward(self, x, step=None, code=None):
        return torch.zeros_like(x)


# ============================================================================
# Test Config Parsing (Phase 1)
# ============================================================================
def test_config_freeze_flow_defaults_to_false():
    """Config flag should default to False for backward compatibility."""
    cfg_dict = {
        "source": "dummy_source.obj",
        "target": "dummy_target.obj",
        "output_dir": "dummy_output",
    }
    cfg = ExperimentCfg.from_dict(cfg_dict)
    assert cfg.freeze_flow_at_identity == False


def test_config_freeze_flow_can_be_set_true():
    """Config should accept and parse freeze_flow_at_identity=true."""
    cfg_dict = {
        "source": "dummy_source.obj",
        "target": "dummy_target.obj",
        "output_dir": "dummy_output",
        "freeze_flow_at_identity": True,
    }
    cfg = ExperimentCfg.from_dict(cfg_dict)
    assert cfg.freeze_flow_at_identity == True


# ============================================================================
# Test IdentityFlowWrapper (Phase 2)
# ============================================================================
def test_identity_wrapper_forward_returns_unchanged_points():
    """Forward pass should return input points unchanged."""
    # Build a minimal flow
    field = DummyVelocityField()
    integrator = ForwardEuler()
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)

    # Wrap it
    wrapper = IdentityFlowWrapper(flow)

    # Test forward
    points = torch.randn(2, 8, 3)  # [B=2, N=8, D=3]
    traj = wrapper.forward(points, code=None)

    # Check end point equals input
    assert torch.allclose(traj.end, points), "Forward should return unchanged points"

    # Check shape
    assert traj.end.shape == points.shape

    # Check all trajectory points are the same
    assert torch.allclose(traj.points[0], traj.points[-1]), "All trajectory points should be identical"


def test_identity_wrapper_kinetic_energy_is_zero():
    """Kinetic energy should be exactly 0 for identity flow."""
    field = DummyVelocityField()
    integrator = ForwardEuler()
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)

    wrapper = IdentityFlowWrapper(flow)

    points = torch.randn(2, 8, 3)
    traj = wrapper.forward(points, code=None)

    ke = traj.kinetic_energy()
    assert ke.item() == 0.0, "Identity flow kinetic energy should be exactly 0"


def test_identity_wrapper_inverse_is_same_as_forward():
    """Inverse should equal forward (identity is self-inverse)."""
    field = DummyVelocityField()
    integrator = ForwardEuler()
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)

    wrapper = IdentityFlowWrapper(flow)

    points = torch.randn(2, 8, 3)
    traj_fwd = wrapper.forward(points, code=None)
    traj_inv = wrapper.inverse(points, code=None)

    # Inverse should also return unchanged points
    assert torch.allclose(traj_inv.end, points), "Inverse should also return unchanged points"
    assert torch.allclose(traj_fwd.end, traj_inv.end), "Inverse should match forward for identity"


def test_identity_wrapper_exposes_field():
    """Wrapper should expose wrapped flow's field for callbacks."""
    field = DummyVelocityField()
    integrator = ForwardEuler()
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)

    wrapper = IdentityFlowWrapper(flow)

    assert hasattr(wrapper, 'field'), "Wrapper should expose field attribute"
    assert wrapper.field is field, "Wrapper.field should be the same as wrapped flow's field"


def test_identity_wrapper_state_dict_roundtrip():
    """State dict should save/load wrapped flow state correctly."""
    field = DummyVelocityField()
    integrator = ForwardEuler()
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)

    wrapper1 = IdentityFlowWrapper(flow)
    state = wrapper1.state_dict()

    # State dict should be callable without error (may be empty if no params)
    assert isinstance(state, dict), "state_dict should return a dict"

    # Create new wrapper and load state
    flow2 = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)
    wrapper2 = IdentityFlowWrapper(flow2)
    result = wrapper2.load_state_dict(state)

    # Should complete without error
    assert result is not None, "load_state_dict should return result"


# ============================================================================
# Test Integration: Config Validation (Phase 3)
# ============================================================================
def test_runner_rejects_freeze_without_encoder_pose():
    """Runner should reject freeze_flow_at_identity without use_encoder_pose."""
    from src.resnet_lddmm.runner import build

    cfg_dict = {
        "source": "dummy_source.obj",
        "target": "dummy_target.obj",
        "output_dir": "dummy_output",
        "freeze_flow_at_identity": True,
        "use_encoder_pose": False,  # <-- Missing!
    }
    cfg = ExperimentCfg.from_dict(cfg_dict)

    # Should raise ValueError on build
    with pytest.raises(ValueError, match="freeze_flow_at_identity requires use_encoder_pose"):
        build(cfg)


def test_runner_rejects_freeze_with_bidirectional():
    """Runner should reject freeze_flow_at_identity with bidirectional flow."""
    from src.resnet_lddmm.runner import build

    cfg_dict = {
        "source": "dummy_source.obj",
        "target": "dummy_target.obj",
        "output_dir": "dummy_output",
        "freeze_flow_at_identity": True,
        "use_encoder_pose": True,
        "loss": {
            "direction": "bidirectional",  # <-- Conflict!
        },
    }
    cfg = ExperimentCfg.from_dict(cfg_dict)

    # Should raise ValueError on build
    with pytest.raises(ValueError, match="freeze_flow_at_identity requires unidirectional flow"):
        build(cfg)


# ============================================================================
# Test Trajectory Properties (Phase 4 - Validation)
# ============================================================================
def test_identity_wrapper_trajectory_has_correct_shape():
    """Trajectory should have correct shape: points [K+1,B,N,3], velocities [K,B,N,3]."""
    field = DummyVelocityField()
    integrator = ForwardEuler()
    K = 10
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=K)

    wrapper = IdentityFlowWrapper(flow)

    B, N, D = 2, 8, 3
    points = torch.randn(B, N, D)
    traj = wrapper.forward(points, code=None)

    assert traj.points.shape == (K+1, B, N, D), f"Points shape should be ({K+1},{B},{N},{D})"
    assert traj.velocities.shape == (K, B, N, D), f"Velocities shape should be ({K},{B},{N},{D})"


def test_identity_wrapper_trajectory_all_velocities_zero():
    """All velocities should be exactly zero (no motion)."""
    field = DummyVelocityField()
    integrator = ForwardEuler()
    flow = NeuralODEFlow(field=field, direct=integrator, inverse=integrator, num_steps=10)

    wrapper = IdentityFlowWrapper(flow)

    points = torch.randn(2, 8, 3)
    traj = wrapper.forward(points, code=None)

    assert torch.all(traj.velocities == 0.0), "All velocities should be exactly zero"


# ============================================================================
# Manual runner for pytest/direct execution
# ============================================================================
def _run_all():
    """Run all tests manually."""
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {str(exc)[:160]}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")


if __name__ == '__main__':
    _run_all()
