"""Tests for config-selected loss terms (losses/terms.py) and their wiring."""

import math

import pytest
import torch
from types import SimpleNamespace

from src.resnet_lddmm.losses.mapping_error import UnidirectionalMappingError
from src.resnet_lddmm.losses.terms import (
    LossContext, EquivariantDeformationLoss, PoseSupervisionLoss,
    FlowRotationPenalty, apply_pose, procrustes_rotation,
)
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


class TestEquivariantDeformationLoss:
    """L2 between φ(g·T) and g·φ(T)."""

    def test_none_without_pose(self):
        """No group element means nothing to be equivariant to -- skipped, not zero."""
        flow = _flow()
        points = torch.randn(1, 20, 3)
        assert EquivariantDeformationLoss()(_context(flow, points, None, None)) is None

    def test_zero_under_identity_flow(self):
        """Identity commutes with everything, so pose-only training scores exactly 0."""
        flow = IdentityFlowWrapper(_flow())
        points = torch.randn(1, 20, 3)

        value = EquivariantDeformationLoss()(_context(flow, points, _rotation(0.7)))

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

        value = EquivariantDeformationLoss()(_context(flow, points, _rotation(0.7)))

        assert value.item() == pytest.approx(0.0, abs=1e-10)

    def test_zero_for_an_untrained_field(self):
        """Zero-init output projection => identity flow => trivially equivariant."""
        torch.manual_seed(0)
        value = EquivariantDeformationLoss()(_context(_flow(), torch.randn(1, 64, 3), _rotation(1.1)))
        assert value.item() == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_a_non_equivariant_field(self):
        """A generic MLP field is not equivariant, so the constraint has work to do."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        points = torch.randn(1, 64, 3)

        value = EquivariantDeformationLoss()(_context(flow, points, _rotation(1.1)))

        assert value.item() > 0

    def test_field_only_keeps_gradient_off_the_pose(self):
        """field_only=True trains the field without pushing the pose head."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        points = torch.randn(1, 32, 3)
        rotation = _rotation(0.9).requires_grad_(True)

        EquivariantDeformationLoss(field_only=True)(_context(flow, points, rotation)).backward()
        assert rotation.grad is None
        assert any(p.grad is not None for p in flow.field.parameters())

        rotation2 = _rotation(0.9).requires_grad_(True)
        EquivariantDeformationLoss(field_only=False)(_context(flow, points, rotation2)).backward()
        assert rotation2.grad is not None

    def test_field_only_keeps_gradient_off_the_latent(self):
        """field_only=True also blocks the encoder's LATENT path, not just the pose.

        Needs a conditioned field, since the default field ignores the code entirely
        and the test would pass vacuously.
        """
        from src.resnet_lddmm.conditioning.film import FiLMConditioning

        torch.manual_seed(0)
        field = TimeVaryingField(num_blocks=3, width=32,
                                 conditioning=FiLMConditioning(n_z=8, output_dim=3))
        with torch.no_grad():
            for param in field.parameters():
                param.add_(torch.randn_like(param) * 0.5)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        points = torch.randn(1, 32, 3)

        base_code = torch.randn(1, 8)   # SAME code both ways, else the values differ

        def run(field_only):
            code = base_code.clone().requires_grad_(True)
            traj = flow(points, code)
            ctx = LossContext(flow=flow, code=code, template_points=points,
                              fwd_traj=traj, encoder_pose=(_rotation(0.9), None))
            value = EquivariantDeformationLoss(field_only=field_only)(ctx)
            value.backward()
            return code.grad, value.item()

        grad_blocked, value_blocked = run(True)
        grad_open, value_open = run(False)

        assert grad_blocked is None       # nothing reaches the encoder
        assert grad_open is not None      # the cheap path does
        # Same objective either way -- only the routing differs.
        assert value_blocked == pytest.approx(value_open, rel=1e-6)

    def test_translation_can_be_excluded(self):
        """translation=False isolates the SO(3) part of the constraint."""
        torch.manual_seed(0)
        flow = _flow(active=True)
        points = torch.randn(1, 32, 3)
        t = torch.tensor([[0.3, -0.2, 0.1]])

        with_t = EquivariantDeformationLoss(translation=True)(_context(flow, points, None, t))
        without_t = EquivariantDeformationLoss(translation=False)(_context(flow, points, None, t))

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
        assert EquivariantDeformationLoss()(ctx).item() == pytest.approx(0.0, abs=1e-12)


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
        term = Registry.create("loss_term", "equivariant_deformation_loss", field_only=False)
        assert isinstance(term, EquivariantDeformationLoss)
        assert term.field_only is False

    def test_build_loss_terms(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[
            {"kind": "equivariant_deformation_loss", "weight": 2.5, "kwargs": {"field_only": False}},
            {"kind": "equivariant_deformation_loss", "name": "equiv_so3", "weight": 1.0,
             "kwargs": {"translation": False}},
            {"kind": "equivariant_deformation_loss", "name": "disabled", "weight": 0.0},
        ])
        entries, modules = _build_loss_terms(cfg)

        assert [e.name for e in entries] == ["equivariant_deformation_loss", "equiv_so3"]
        assert [e.weight for e in entries] == [2.5, 1.0]
        assert set(modules) == {"equivariant_deformation_loss", "equiv_so3"}   # weight 0 dropped
        assert modules["equivariant_deformation_loss"].field_only is False
        assert modules["equiv_so3"].translation is False

    def test_empty_terms_is_inert(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        assert _build_loss_terms(SimpleNamespace(terms=[])) == ([], {})

    def test_weight_gates_the_equivariant_deformation_loss(self):
        """weight > 0 builds it; 0 means it is never instantiated at all.

        Not "computed then scaled by zero": at 0 the term does not exist, so its
        extra forward integration never runs and it stays out of the breakdown.
        """
        from src.resnet_lddmm.runner import _build_loss_terms

        off = SimpleNamespace(terms=[], equivariant_deformation_weight=0.0,
                              equivariant_deformation_kwargs={})
        assert _build_loss_terms(off) == ([], {})

        on = SimpleNamespace(terms=[], equivariant_deformation_weight=0.25,
                             equivariant_deformation_kwargs={"field_only": False})
        entries, modules = _build_loss_terms(on)

        assert [e.name for e in entries] == ["equivariant_deformation_loss"]
        assert entries[0].weight == 0.25
        assert modules["equivariant_deformation_loss"].field_only is False

    def test_negative_weight_also_gates(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[], equivariant_deformation_weight=-1.0,
                              equivariant_deformation_kwargs={})
        assert _build_loss_terms(cfg) == ([], {})

    def test_weight_absent_from_config_is_inert(self):
        """A LossCfg without the field at all (older configs) still builds."""
        from src.resnet_lddmm.runner import _build_loss_terms

        assert _build_loss_terms(SimpleNamespace(terms=[])) == ([], {})

    def test_unknown_kind_lists_valid_names(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[{"kind": "nope", "weight": 1.0}])
        with pytest.raises(ValueError, match="equivariant_deformation_loss"):
            _build_loss_terms(cfg)

    def test_duplicate_names_rejected(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        cfg = SimpleNamespace(terms=[
            {"kind": "equivariant_deformation_loss", "weight": 1.0},
            {"kind": "equivariant_deformation_loss", "weight": 2.0},
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
                                 LossTerm("equivariant_deformation_loss", 1.0)])
        code_source = SimpleNamespace(
            __call__=lambda batch: None, penalty=lambda: None,
            get_pose=lambda: (_rotation(0.6), None),
        )
        stepper = PairRegistration(
            flow, _StubCode(), L2Data(), UnidirectionalMappingError(), composer,
            torch.optim.Adam(flow.parameters(), lr=1e-4),
            use_encoder_pose=True,
            loss_terms={"equivariant_deformation_loss": EquivariantDeformationLoss()},
        )
        batch = SimpleNamespace(points=torch.randn(1, 20, 3), weights=None, faces=None)

        _, _, breakdown = stepper.train_step(batch, batch)

        assert "equivariant_deformation_loss" in breakdown
        assert breakdown["equivariant_deformation_loss"] > 0


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


class TestAugmentationRecordsItsElement:
    """Augmentations expose the group element they drew, for supervision."""

    def test_so3_records_rotation(self):
        from src.resnet_lddmm.augmentation.so3 import SO3Augmentation

        aug = SO3Augmentation(seed=0)
        points = torch.randn(3, 20, 3)
        out = aug(points)

        R, t = aug.last_element()
        assert R.shape == (3, 3, 3)
        assert t is None
        # The recorded element is exactly what was applied, in the pipeline's convention.
        assert torch.allclose(apply_pose(points, R, None), out, atol=1e-5)

    def test_se3_records_rotation_and_translation(self):
        from src.resnet_lddmm.augmentation.se3 import SE3Augmentation

        aug = SE3Augmentation(translation_scale=0.3, seed=0)
        points = torch.randn(2, 20, 3)
        out = aug(points)

        R, t = aug.last_element()
        assert R.shape == (2, 3, 3) and t.shape == (2, 3)
        assert torch.allclose(apply_pose(points, R, t), out, atol=1e-5)

    def test_none_records_nothing(self):
        from src.resnet_lddmm.augmentation.none import NoAugmentation

        aug = NoAugmentation()
        aug(torch.randn(2, 10, 3))
        assert aug.last_element() == (None, None)

    def test_element_refreshes_each_call(self):
        from src.resnet_lddmm.augmentation.so3 import SO3Augmentation

        aug = SO3Augmentation(seed=0)
        points = torch.randn(2, 10, 3)
        aug(points); first = aug.last_element()[0].clone()
        aug(points); second = aug.last_element()[0]

        assert not torch.allclose(first, second)


class TestPoseSupervisionLoss:
    """Supervising R̂ against the rotation the augmenter actually drew."""

    @staticmethod
    def _ctx(pred_R, true_R, pred_t=None, true_t=None):
        return LossContext(encoder_pose=(pred_R, pred_t), augmentation_pose=(true_R, true_t))

    def test_zero_when_prediction_matches(self):
        R = _rotation(0.7)
        assert PoseSupervisionLoss()(self._ctx(R, R.clone())).item() == pytest.approx(0.0, abs=1e-12)

    def test_positive_when_prediction_is_wrong(self):
        assert PoseSupervisionLoss()(self._ctx(_rotation(0.7), _rotation(-0.9))).item() > 0.1

    def test_none_without_a_predicted_rotation(self):
        assert PoseSupervisionLoss()(self._ctx(None, _rotation(0.5))) is None

    def test_none_without_an_augmentation(self):
        """kind: none draws no element, so the term vanishes rather than erroring."""
        assert PoseSupervisionLoss()(self._ctx(_rotation(0.5), None)) is None

    def test_gradient_reaches_the_prediction_only(self):
        pred = _rotation(-0.4).requires_grad_(True)
        true = _rotation(0.7).requires_grad_(True)

        PoseSupervisionLoss()(self._ctx(pred, true)).backward()

        assert pred.grad is not None
        assert true.grad is None      # the target is data, never a variable

    def test_translation_off_by_default(self):
        """Rotation-only unless asked, since SO(3) augmentation has no translation."""
        R = _rotation(0.5)
        ctx = self._ctx(R, R.clone(), pred_t=torch.zeros(1, 3), true_t=torch.ones(1, 3))

        assert PoseSupervisionLoss(translation=False)(ctx).item() == pytest.approx(0.0, abs=1e-12)
        assert PoseSupervisionLoss(translation=True)(ctx).item() == pytest.approx(3.0, rel=1e-5)

    def test_descends_to_the_true_rotation(self):
        """Plain SGD on this term drives R̂ onto the drawn rotation."""
        true = _rotation(0.9)
        pred = _rotation(-0.5).clone().requires_grad_(True)
        opt = torch.optim.SGD([pred], lr=0.2)

        for _ in range(60):
            opt.zero_grad()
            PoseSupervisionLoss()(self._ctx(pred, true)).backward()
            opt.step()

        assert (pred.detach() - true).abs().max() < 0.02

    def test_registry_and_gating(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        term = Registry.create("loss_term", "pose_supervision_loss", translation=True)
        assert isinstance(term, PoseSupervisionLoss) and term.translation is True

        off = SimpleNamespace(terms=[], pose_supervision_weight=0.0, pose_supervision_kwargs={})
        assert _build_loss_terms(off) == ([], {})

        on = SimpleNamespace(terms=[], pose_supervision_weight=3.0, pose_supervision_kwargs={})
        entries, modules = _build_loss_terms(on)
        assert [e.name for e in entries] == ["pose_supervision_loss"]
        assert entries[0].weight == 3.0


class TestFlowRotationPenalty:
    """||R_flow - I||_F^2 -- fixes the pose/deformation gauge."""

    @staticmethod
    def _ctx(template, deformed):
        return LossContext(template_points=template,
                           fwd_traj=SimpleNamespace(end=deformed))

    def test_zero_for_the_identity_deformation(self):
        pts = torch.randn(1, 60, 3)
        value = FlowRotationPenalty()(self._ctx(pts, pts.clone()))
        assert value.item() == pytest.approx(0.0, abs=1e-9)

    def test_zero_for_a_pure_translation(self):
        """Procrustes mean-centres, so a net translation is not penalised.

        Deliberate: R_hat cannot absorb a translation, so it is not gauge -- the flow
        genuinely needs it to match shapes whose centroids differ.
        """
        pts = torch.randn(1, 60, 3)
        moved = pts + torch.tensor([[0.4, -0.3, 0.2]])
        assert FlowRotationPenalty()(self._ctx(pts, moved)).item() == pytest.approx(0.0, abs=1e-9)

    def test_matches_the_closed_form_for_a_pure_rotation(self):
        """||R - I||_F^2 = 8 sin^2(theta/2) -- the value reads directly as an angle."""
        pts = torch.randn(1, 200, 3)
        for angle in (0.3, 0.9, 2.0):
            rotated = apply_pose(pts, _rotation(angle), None)
            expected = 8 * math.sin(angle / 2) ** 2
            got = FlowRotationPenalty()(self._ctx(pts, rotated)).item()
            assert got == pytest.approx(expected, rel=1e-3), f"angle={angle}"

    def test_zero_for_a_rotation_free_deformation(self):
        """A deformation carrying no net rotation is not penalised.

        Uniform scaling gives H = c*X^T X, symmetric PSD, so Procrustes returns exactly
        I. Note an ANISOTROPIC stretch would not: H = X^T X diag(s) is asymmetric unless
        the stretch axes are the data's principal axes, and the term correctly reports
        the small net rotation such a warp really does carry.
        """
        torch.manual_seed(0)
        pts = torch.randn(1, 300, 3)
        scaled = 1.3 * pts + torch.tensor([[0.2, -0.1, 0.4]])   # scale + shift, no rotation

        assert FlowRotationPenalty()(self._ctx(pts, scaled)).item() < 1e-6

    def test_gradient_reaches_the_deformation(self):
        torch.manual_seed(0)
        pts = torch.randn(1, 80, 3)
        deformed = apply_pose(pts, _rotation(0.8), None).clone().requires_grad_(True)

        FlowRotationPenalty()(self._ctx(pts, deformed)).backward()

        assert deformed.grad is not None and deformed.grad.abs().max() > 0

    def test_none_without_a_trajectory(self):
        assert FlowRotationPenalty()(LossContext()) is None

    def test_registry_and_gating(self):
        from src.resnet_lddmm.runner import _build_loss_terms

        assert isinstance(Registry.create("loss_term", "flow_rotation_penalty"),
                          FlowRotationPenalty)

        off = SimpleNamespace(terms=[], flow_rotation_penalty_weight=0.0,
                              flow_rotation_penalty_kwargs={})
        assert _build_loss_terms(off) == ([], {})

        on = SimpleNamespace(terms=[], flow_rotation_penalty_weight=0.1,
                             flow_rotation_penalty_kwargs={})
        entries, modules = _build_loss_terms(on)
        assert [e.name for e in entries] == ["flow_rotation_penalty"]
        assert entries[0].weight == 0.1

    def test_identifies_the_gauge_it_exists_to_remove(self):
        """The scenario from the analysis: flow absorbs A, R_hat compensates.

        The data term is blind to this (predictions identical); only this term sees it.
        """
        torch.manual_seed(0)
        template = torch.randn(1, 200, 3)
        A = _rotation(0.7)
        canonical = 1.2 * template + torch.tensor([[0.3, 0.0, -0.2]])   # honest deformation

        # config 2: flow does the deformation only -> no net rotation
        assert FlowRotationPenalty()(self._ctx(template, canonical)).item() < 1e-6
        # config 1: flow also absorbs A -> penalised by exactly ||A - I||_F^2
        absorbed = apply_pose(canonical, A, None)
        expected = 8 * math.sin(0.7 / 2) ** 2
        assert FlowRotationPenalty()(self._ctx(template, absorbed)).item() == pytest.approx(
            expected, rel=1e-2)


class TestProcrustesConditioningGuard:
    """The SVD backward carries 1/(s_i - s_j); warn before it becomes garbage."""

    @staticmethod
    def _reset():
        import src.resnet_lddmm.losses.terms as terms_mod
        terms_mod._procrustes_warnings = 0

    @staticmethod
    def _isotropic():
        """Six axis points: covariance exactly isotropic, so all singular values tie."""
        return torch.tensor([[[1., 0., 0.], [-1., 0., 0.],
                              [0., 1., 0.], [0., -1., 0.],
                              [0., 0., 1.], [0., 0., -1.]]])

    def test_warns_on_a_degenerate_shape(self, capsys):
        self._reset()
        pts = self._isotropic()

        procrustes_rotation(pts, pts.clone())

        out = capsys.readouterr().out
        assert "PROCRUSTES_ILL_CONDITIONED" in out
        assert "near-degenerate" in out

    def test_silent_on_a_well_conditioned_shape(self, capsys):
        self._reset()
        torch.manual_seed(0)
        pts = torch.randn(1, 200, 3) * torch.tensor([3.0, 1.5, 0.5])   # distinct axes

        procrustes_rotation(pts, apply_pose(pts, _rotation(0.6), None))

        assert "PROCRUSTES_ILL_CONDITIONED" not in capsys.readouterr().out

    def test_warnings_are_rate_limited(self, capsys):
        import src.resnet_lddmm.losses.terms as terms_mod
        self._reset()
        pts = self._isotropic()

        for _ in range(terms_mod._PROCRUSTES_WARN_LIMIT + 4):
            procrustes_rotation(pts, pts.clone())

        out = capsys.readouterr().out          # readouterr CLEARS the buffer: read once
        assert out.count("PROCRUSTES_ILL_CONDITIONED") == terms_mod._PROCRUSTES_WARN_LIMIT
        assert "further warnings suppressed" in out

    def test_the_guard_does_not_change_the_result(self, capsys):
        """Diagnostics only -- the rotation itself must be untouched."""
        self._reset()
        torch.manual_seed(0)
        pts = torch.randn(1, 120, 3)
        R = _rotation(0.8)

        assert torch.allclose(procrustes_rotation(pts, apply_pose(pts, R, None)), R, atol=1e-5)

    def test_nonfinite_gradient_names_the_cause(self):
        """A NaN through the rotation must not surface as an anonymous 'non-finite loss'."""
        from src.resnet_lddmm.losses.terms import _raise_on_nonfinite_grad

        with pytest.raises(FloatingPointError, match="procrustes_rotation"):
            _raise_on_nonfinite_grad(torch.tensor([float("nan"), 1.0]))

        clean = torch.tensor([1.0, 2.0])
        assert _raise_on_nonfinite_grad(clean) is clean
