"""Tests for config-selected loss terms (losses/terms.py) and their wiring."""

import pytest
import torch
from types import SimpleNamespace

from src.resnet_lddmm.losses.mapping_error import UnidirectionalMappingError
from src.resnet_lddmm.losses.terms import LossContext, FlowEquivarianceTerm, apply_pose
from src.resnet_lddmm.losses.data_terms import L2Data
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.flow import NeuralODEFlow, IdentityFlowWrapper
from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.registration.pair import PairRegistration
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.registry import Registry
from src.resnet_lddmm import registrations  # noqa: F401  (side effect: registers loss_term)


def _rotation(angle, axis=2):
    """A proper rotation about one axis, as [1, 3, 3]."""
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    R = torch.eye(3)
    a, b = [i for i in range(3) if i != axis]
    R[a, a], R[a, b], R[b, a], R[b, b] = c, -s, s, c
    return R.unsqueeze(0)


def _flow(num_steps=3, active=False):
    """A flow, optionally perturbed away from the identity.

    Fields zero-init their output projection (blocks.py:23, time_varying.py:91), so
    an UNTRAINED flow is exactly the identity -- which commutes with every group
    element and makes any equivariance penalty trivially zero. Tests that need a
    genuinely non-equivariant map must ask for active=True.
    """
    field = TimeVaryingField(num_blocks=num_steps, width=32)
    if active:
        with torch.no_grad():
            for param in field.parameters():
                param.add_(torch.randn_like(param) * 0.5)
    return NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=num_steps)


def _context(flow, points, rotation, translation=None):
    """Run the flow and package the result the way a stepper would."""
    traj = flow(points, None)
    return LossContext(
        flow=flow,
        code=None,
        template_points=points,
        fwd_traj=traj,
        encoder_pose=(rotation, translation),
    )


class TestFlowEquivarianceTerm:
    """L2 between φ(g·T) and g·φ(T)."""

    def test_none_without_pose(self):
        """No group element means nothing to be equivariant to -- skipped, not zero."""
        flow = _flow()
        points = torch.randn(1, 20, 3)
        assert FlowEquivarianceTerm()(_context(flow, points, None, None)) is None

    def test_zero_under_identity_flow(self):
        """Identity commutes with everything, so pose-only training scores exactly 0."""
        flow = IdentityFlowWrapper(_flow())
        points = torch.randn(1, 20, 3)

        value = FlowEquivarianceTerm()(_context(flow, points, _rotation(0.7)))

        assert value.item() == pytest.approx(0.0, abs=1e-12)

    def test_zero_for_an_equivariant_flow(self):
        """A flow whose field commutes with rotation scores ~0.

        Catches a flipped rotation convention: applying R·x where the data term
        applies x·R makes this fail while every shape stays valid.
        """
        class RadialField(TimeVaryingField):
            """v(x) = 0.1 * x — commutes with any rotation about the origin."""
            def forward(self, x, step=None, code=None):
                return 0.1 * x

        flow = NeuralODEFlow(RadialField(num_blocks=3, width=32),
                             direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        points = torch.randn(1, 20, 3)

        value = FlowEquivarianceTerm()(_context(flow, points, _rotation(0.7)))

        assert value.item() == pytest.approx(0.0, abs=1e-10)

    def test_zero_for_an_untrained_field(self):
        """Zero-init output projection => identity flow => trivially equivariant."""
        torch.manual_seed(0)
        value = FlowEquivarianceTerm()(_context(_flow(), torch.randn(1, 64, 3), _rotation(1.1)))
        assert value.item() == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_a_non_equivariant_field(self):
        """A generic MLP field is not equivariant, so the constraint has work to do."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        points = torch.randn(1, 64, 3)

        value = FlowEquivarianceTerm()(_context(flow, points, _rotation(1.1)))

        assert value.item() > 0

    def test_detach_group_keeps_gradient_off_the_pose(self):
        """detach_group=True trains the field without pushing the pose head."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        points = torch.randn(1, 32, 3)
        rotation = _rotation(0.9).requires_grad_(True)

        FlowEquivarianceTerm(detach_group=True)(_context(flow, points, rotation)).backward()
        assert rotation.grad is None
        assert any(p.grad is not None for p in flow.field.parameters())

        rotation2 = _rotation(0.9).requires_grad_(True)
        FlowEquivarianceTerm(detach_group=False)(_context(flow, points, rotation2)).backward()
        assert rotation2.grad is not None

    def test_translation_can_be_excluded(self):
        """translation=False isolates the SO(3) part of the constraint."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        points = torch.randn(1, 32, 3)
        t = torch.tensor([[0.3, -0.2, 0.1]])

        with_t = FlowEquivarianceTerm(translation=True)(_context(flow, points, None, t))
        without_t = FlowEquivarianceTerm(translation=False)(_context(flow, points, None, t))

        assert with_t.item() > 0
        assert without_t is None  # rotation None + translation excluded -> nothing to test

    def test_uses_the_points_the_flow_actually_ran_on(self):
        """The term reads back the subsampled tensor, keeping lhs/rhs corresponded.

        With subsample_M active the mapping error draws a random subset; a term that
        re-drew its own would compare unrelated vertices and never reach zero.
        """
        flow = IdentityFlowWrapper(_flow())
        error = UnidirectionalMappingError(subsample_n=10)
        template = SimpleNamespace(points=torch.randn(1, 50, 3), weights=None)
        sample = SimpleNamespace(points=torch.randn(1, 40, 3), weights=None)
        rotation = _rotation(0.5)

        error(flow, L2Data(), template, sample, None, encoder_pose=(rotation, None))

        assert error.last_template_points.shape == (1, 10, 3)
        assert error.last_effective_pose[0] is rotation

        ctx = LossContext(flow=flow, code=None,
                          template_points=error.last_template_points,
                          fwd_traj=error.last_fwd_traj,
                          encoder_pose=error.last_effective_pose)
        assert FlowEquivarianceTerm()(ctx).item() == pytest.approx(0.0, abs=1e-12)


class TestApplyPose:
    """apply_pose must agree with mapping_error's convention."""

    def test_matches_mapping_error(self):
        points = torch.randn(2, 15, 3)
        rotation = torch.cat([_rotation(0.4), _rotation(-1.2)])
        translation = torch.randn(2, 3)

        assert torch.allclose(
            apply_pose(points, rotation, translation),
            UnidirectionalMappingError._apply_encoder_pose(points, rotation, translation),
        )

    def test_identity_when_both_none(self):
        points = torch.randn(1, 5, 3)
        assert apply_pose(points, None, None) is points


class TestLossTermWiring:
    """cfg.loss.terms -> composer entries + stepper-visible modules."""

    def test_registry_resolves_the_term(self):
        term = Registry.create("loss_term", "flow_equivariance", detach_group=False)
        assert isinstance(term, FlowEquivarianceTerm)
        assert term.detach_group is False

    def test_build_loss_terms(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[
            {"kind": "flow_equivariance", "weight": 2.5, "kwargs": {"detach_group": False}},
            {"kind": "flow_equivariance", "name": "equiv_so3", "weight": 1.0,
             "kwargs": {"translation": False}},
            {"kind": "flow_equivariance", "name": "disabled", "weight": 0.0},
        ])
        entries, modules = _build_loss_terms(cfg)

        assert [e.name for e in entries] == ["flow_equivariance", "equiv_so3"]
        assert [e.weight for e in entries] == [2.5, 1.0]
        assert set(modules) == {"flow_equivariance", "equiv_so3"}   # weight 0 dropped
        assert modules["flow_equivariance"].detach_group is False
        assert modules["equiv_so3"].translation is False

    def test_empty_terms_is_inert(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        assert _build_loss_terms(SimpleNamespace(terms=[])) == ([], {})

    def test_unknown_kind_lists_valid_names(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[{"kind": "nope", "weight": 1.0}])
        with pytest.raises(ValueError, match="flow_equivariance"):
            _build_loss_terms(cfg)

    def test_duplicate_names_rejected(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[
            {"kind": "flow_equivariance", "weight": 1.0},
            {"kind": "flow_equivariance", "weight": 2.0},
        ])
        with pytest.raises(ValueError, match="duplicate loss term name"):
            _build_loss_terms(cfg)

    def test_missing_kind_rejected(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        with pytest.raises(ValueError, match="needs a 'kind'"):
            _build_loss_terms(SimpleNamespace(terms=[{"weight": 1.0}]))

    def test_term_reaches_the_breakdown(self):
        """A configured term appears in the stepper's per-term breakdown."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        composer = LossComposer([LossTerm("data", 1.0), LossTerm("kinetic", 0.0),
                                 LossTerm("flow_equivariance", 1.0)])
        code_source = SimpleNamespace(
            __call__=lambda batch: None, penalty=lambda: None,
            get_pose=lambda: (_rotation(0.6), None),
        )
        stepper = PairRegistration(
            flow, _StubCode(), L2Data(), UnidirectionalMappingError(), composer,
            torch.optim.Adam(flow.parameters(), lr=1e-4),
            use_encoder_pose=True,
            loss_terms={"flow_equivariance": FlowEquivarianceTerm()},
        )
        batch = SimpleNamespace(points=torch.randn(1, 20, 3), weights=None, faces=None)

        _, _, breakdown = stepper.train_step(batch, batch)

        assert "flow_equivariance" in breakdown
        assert breakdown["flow_equivariance"] > 0


class _StubCode(torch.nn.Module):
    """Code source that predicts a fixed, differentiable pose and no code."""

    def __init__(self):
        super().__init__()
        self.rotation = torch.nn.Parameter(_rotation(0.6).clone())

    def forward(self, batch):
        return None

    def penalty(self):
        return None

    def get_pose(self):
        return self.rotation, None
