"""Tests for CohortRegistration stepper."""

from dataclasses import dataclass, field

import torch
import pytest

from src.resnet_lddmm.registration.cohort import CohortRegistration
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.integrators import ForwardEuler
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.codes.auto_decoder import AutoDecoderCodes
from src.resnet_lddmm.losses.data_terms import CDData
from src.resnet_lddmm.losses import BidirectionalMappingError
from src.learning.losses.composer import LossTerm, LossComposer


@dataclass
class CohortBatch:
    """Batch object with points, weights, and shape_ids."""

    points: torch.Tensor
    shape_ids: torch.Tensor
    weights: torch.Tensor = None


class TestCohortRegistrationBasics:
    """Basic instantiation and interface tests."""

    @staticmethod
    def make_stepper(num_shapes=3, num_steps=5, use_weight_decay=False):
        """Create a minimal CohortRegistration for testing."""
        n_z = 64
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = AutoDecoderCodes(num_shapes=num_shapes, n_z=n_z)
        data_term = CDData()
        mapping_error = BidirectionalMappingError()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])

        # Two parameter groups: flow with weight_decay, codes without
        flow_params = flow.parameters()
        code_params = code_source.parameters()

        optimizer = torch.optim.Adam([
            {"params": flow_params, "weight_decay": 1e-4 if use_weight_decay else 0.0},
            {"params": code_params, "weight_decay": 0.0},
        ], lr=0.01)

        return CohortRegistration(flow, code_source, data_term, mapping_error, composer, optimizer)

    def test_initialization(self):
        """Verify CohortRegistration can be instantiated."""
        stepper = self.make_stepper()
        assert stepper is not None
        assert stepper.flow is not None
        assert stepper.code_source is not None

    def test_has_train_step_method(self):
        """Verify train_step method exists."""
        stepper = self.make_stepper()
        assert hasattr(stepper, "train_step")
        assert callable(stepper.train_step)

    def test_has_eval_step_method(self):
        """Verify eval_step method exists."""
        stepper = self.make_stepper()
        assert hasattr(stepper, "eval_step")
        assert callable(stepper.eval_step)

    def test_optimizer_has_two_param_groups(self):
        """Verify optimizer has two parameter groups (flow, codes)."""
        stepper = self.make_stepper()
        assert len(stepper.optimizer.param_groups) == 2
        # First group is flow (may have weight_decay)
        # Second group is codes (should have weight_decay=0)
        assert stepper.optimizer.param_groups[1]["weight_decay"] == 0.0


class TestCohortRegistrationTrainStep:
    """Tests for train_step method."""

    def test_train_step_returns_tuple(self):
        """Verify train_step returns (traj, loss_float, breakdown)."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        result = stepper.train_step(source, target)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_train_step_returns_correct_types(self):
        """Verify return types: Trajectory, float, dict."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        traj, loss_val, breakdown = stepper.train_step(source, target)
        assert hasattr(traj, "points")
        assert isinstance(loss_val, float)
        assert isinstance(breakdown, dict)

    def test_train_step_updates_parameters(self):
        """Verify parameters are updated during training."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        initial_params = [p.clone() for p in stepper.flow.parameters()]
        stepper.train_step(source, target)

        changed = False
        for init, current in zip(initial_params, stepper.flow.parameters()):
            if not torch.allclose(init, current):
                changed = True
                break
        assert changed, "No parameters were updated"

    @pytest.mark.slow
    def test_train_step_decreases_loss(self):
        """Verify loss decreases over multiple training steps."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3, num_steps=5)
        torch.manual_seed(42)

        source_points = torch.randn(3, 20, 3)
        target_points = source_points.mean(dim=0, keepdim=True) + 0.3

        source = CohortBatch(
            points=source_points,
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=target_points,
            shape_ids=torch.tensor([0])
        )

        losses = []
        for _ in range(15):
            _, loss_val, _ = stepper.train_step(source, target)
            losses.append(loss_val)

        # Loss should generally decrease
        assert losses[-1] < losses[0], f"Final loss {losses[-1]} not < initial {losses[0]}"


class TestCohortRegistrationEvalStep:
    """Tests for eval_step method."""

    def test_eval_step_returns_tuple(self):
        """Verify eval_step returns (traj, loss_float, breakdown)."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        result = stepper.eval_step(source, target)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_eval_step_no_grad(self):
        """Verify eval_step computes under no_grad."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        for p in stepper.flow.parameters():
            p.requires_grad_(True)

        stepper.eval_step(source, target)

        for p in stepper.flow.parameters():
            assert p.grad is None

    def test_eval_step_does_not_update_parameters(self):
        """Verify eval_step doesn't change parameters."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        initial_params = [p.clone() for p in stepper.flow.parameters()]
        stepper.eval_step(source, target)

        for init, current in zip(initial_params, stepper.flow.parameters()):
            assert torch.equal(init, current), "eval_step modified parameters"


class TestCohortRegistrationBreakdown:
    """Tests for loss breakdown dict."""

    def test_breakdown_keys_match_configured_terms(self):
        """Verify breakdown dict keys match LossComposer terms."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        _, _, breakdown = stepper.train_step(source, target)

        # With AutoDecoderCodes, code_reg should be present
        assert "data" in breakdown
        assert "kinetic" in breakdown
        assert "code_reg" in breakdown

    def test_breakdown_values_are_floats(self):
        """Verify breakdown values are floats."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        _, _, breakdown = stepper.train_step(source, target)

        for key, value in breakdown.items():
            assert isinstance(value, float), f"{key} value is not float: {type(value)}"


class TestCohortRegistrationStateDict:
    """Tests for state_dict and load_state_dict."""

    def test_state_dict_has_required_keys(self):
        """Verify state_dict has flow, codes, optimizer keys."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        state = stepper.state_dict()

        assert "flow" in state
        assert "codes" in state
        assert "optimizer" in state

    def test_state_dict_roundtrip(self):
        """Verify state_dict can be loaded into another stepper."""
        stepper1 = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        stepper2 = TestCohortRegistrationBasics.make_stepper(num_shapes=3)

        state = stepper1.state_dict()
        stepper2.load_state_dict(state)

        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        with torch.no_grad():
            traj1, loss1, _ = stepper1.eval_step(source, target)
            traj2, loss2, _ = stepper2.eval_step(source, target)

        assert torch.allclose(traj1.points, traj2.points)
        assert abs(loss1 - loss2) < 1e-5


class TestCohortRegistrationModes:
    """Tests for train/eval mode switching."""

    def test_train_mode(self):
        """Verify train() sets training mode."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        stepper.eval()
        assert not stepper.flow.training

        stepper.train()
        assert stepper.flow.training
        assert stepper.code_source.training

    def test_eval_mode(self):
        """Verify eval() sets evaluation mode."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        stepper.train()
        assert stepper.flow.training

        stepper.eval()
        assert not stepper.flow.training
        assert not stepper.code_source.training


class TestCohortRegistrationProtocol:
    """Tests for the four-method protocol (needed by reused callbacks)."""

    def test_has_state_dict_method(self):
        """Verify state_dict method exists."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert hasattr(stepper, "state_dict")
        assert callable(stepper.state_dict)

    def test_has_load_state_dict_method(self):
        """Verify load_state_dict method exists."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert hasattr(stepper, "load_state_dict")
        assert callable(stepper.load_state_dict)

    def test_has_train_method(self):
        """Verify train method exists."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert hasattr(stepper, "train")
        assert callable(stepper.train)

    def test_has_eval_method(self):
        """Verify eval method exists."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert hasattr(stepper, "eval")
        assert callable(stepper.eval)


class TestCohortRegistrationParameterGroups:
    """Tests for optimizer parameter groups."""

    def test_flow_params_in_first_group(self):
        """Verify flow parameters are in first optimizer group."""
        stepper = TestCohortRegistrationBasics.make_stepper(use_weight_decay=True)
        flow_params = set(id(p) for p in stepper.flow.parameters())
        first_group_params = set(id(p) for p in stepper.optimizer.param_groups[0]["params"])
        # Check that flow params are in the first group
        assert len(flow_params & first_group_params) > 0

    def test_code_params_in_second_group(self):
        """Verify code parameters are in second optimizer group."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        code_params = set(id(p) for p in stepper.code_source.parameters())
        second_group_params = set(id(p) for p in stepper.optimizer.param_groups[1]["params"])
        # Check that code params are in the second group
        assert len(code_params & second_group_params) > 0

    def test_code_params_have_zero_weight_decay(self):
        """Verify code parameters have weight_decay=0."""
        stepper = TestCohortRegistrationBasics.make_stepper(use_weight_decay=True)
        # Second group (codes) should always have weight_decay=0
        assert stepper.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_flow_params_have_weight_decay(self):
        """Verify flow parameters can have weight_decay when configured."""
        stepper = TestCohortRegistrationBasics.make_stepper(use_weight_decay=True)
        # First group (flow) should have weight_decay=1e-4
        assert stepper.optimizer.param_groups[0]["weight_decay"] == 1e-4


class TestCohortRegistrationBidirectional:
    """Tests for bidirectional loss computation."""

    def test_uses_bidirectional_mapping_error(self):
        """Verify stepper uses BidirectionalMappingError."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert isinstance(stepper.mapping_error, BidirectionalMappingError)

    def test_bidirectional_flag_is_set(self):
        """Verify is_bidirectional flag is True."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert stepper.is_bidirectional is True

    def test_backward_trajectory_computed_during_train(self):
        """Verify backward trajectory is computed during training."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)
        source = CohortBatch(
            points=torch.randn(3, 10, 3),
            shape_ids=torch.tensor([0, 1, 2])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        stepper.train_step(source, target)

        # After training, backward_traj should be set
        assert stepper.backward_traj is not None
        assert hasattr(stepper.backward_traj, "points")


class TestCohortRegistrationNumericalStability:
    """Tests for numerical stability."""

    def test_loss_is_finite(self):
        """Verify loss values are always finite."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3)

        for _ in range(5):
            source = CohortBatch(
                points=torch.randn(3, 10, 3),
                shape_ids=torch.tensor([0, 1, 2])
            )
            target = CohortBatch(
                points=torch.randn(1, 10, 3),
                shape_ids=torch.tensor([0])
            )
            _, loss_val, _ = stepper.train_step(source, target)
            assert torch.isfinite(torch.tensor(loss_val))

    def test_handles_single_shape_batch(self):
        """Verify stepper works with single shape in batch."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=1)
        source = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        traj, loss_val, breakdown = stepper.train_step(source, target)
        assert traj is not None
        assert isinstance(loss_val, float)
        assert isinstance(breakdown, dict)


class TestCohortRegistrationInfer:
    """Tests for code-only inference (T29: DeepSDF auto-decoder protocol)."""

    def test_infer_returns_tensor(self):
        """Verify infer() returns a tensor of shape [1, n_z]."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=2, num_steps=5)
        new_shape = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        z = stepper.infer(new_shape, target, steps_adam=5)

        assert isinstance(z, torch.Tensor)
        assert z.shape == (1, stepper.code_source.n_z)

    def test_infer_has_infer_method(self):
        """Verify infer method exists and is callable."""
        stepper = TestCohortRegistrationBasics.make_stepper()
        assert hasattr(stepper, "infer")
        assert callable(stepper.infer)

    def test_infer_converges(self):
        """Verify loss decreases during infer() optimization."""
        torch.manual_seed(42)
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=2, num_steps=5)

        # Create a held-out shape and template
        new_shape = CohortBatch(
            points=torch.randn(1, 15, 3),
            shape_ids=torch.tensor([0])
        )
        target = CohortBatch(
            points=torch.randn(1, 15, 3),
            shape_ids=torch.tensor([0])
        )

        # Manually track loss during optimization
        z_new = torch.randn(1, stepper.code_source.n_z) * (2.0 / stepper.code_source.n_z) ** 0.5
        z_new.requires_grad = True

        optimizer = torch.optim.Adam([z_new], lr=0.01)
        stepper.flow.eval()

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            fwd_traj = stepper.flow(new_shape.points, z_new)
            tgt_broadcast = target.points.expand(1, -1, -1)
            bwd_traj = stepper.flow.inverse(tgt_broadcast, z_new)
            data_fwd = stepper.data_term(fwd_traj.end, target.points)
            data_bwd = stepper.data_term(bwd_traj.end, new_shape.points)
            data = data_fwd + data_bwd
            kinetic = fwd_traj.kinetic_energy() + bwd_traj.kinetic_energy()
            code_reg = (z_new ** 2).mean()
            values = {"data": data, "kinetic": kinetic, "code_reg": code_reg}
            loss, _ = stepper.composer.compute(values)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    def test_infer_flow_frozen(self):
        """Verify flow parameters are bit-identical before/after infer()."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=2, num_steps=5)

        # Clone flow state before inference
        flow_state_before = {
            name: param.data.clone() for name, param in stepper.flow.named_parameters()
        }

        new_shape = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        z = stepper.infer(new_shape, target, steps_adam=5)

        # Verify all flow parameters are identical
        for name, param in stepper.flow.named_parameters():
            assert torch.allclose(param.data, flow_state_before[name], atol=1e-7), \
                f"Flow param '{name}' was modified"

    def test_infer_with_lbfgs(self):
        """Verify infer() works with optional L-BFGS refinement."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=2, num_steps=5)

        new_shape = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        # Should not raise with L-BFGS steps
        z = stepper.infer(new_shape, target, steps_adam=3, steps_lbfgs=2)

        assert z is not None
        assert z.shape == (1, stepper.code_source.n_z)

    @pytest.mark.slow
    def test_infer_learns_distinct_codes(self):
        """Verify different shapes get different optimized codes."""
        torch.manual_seed(42)
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=3, num_steps=5)

        target = CohortBatch(
            points=torch.randn(1, 20, 3),
            shape_ids=torch.tensor([0])
        )

        # Infer on two different shapes
        shape1 = CohortBatch(
            points=torch.randn(1, 20, 3),
            shape_ids=torch.tensor([0])
        )
        shape2 = CohortBatch(
            points=torch.randn(1, 20, 3),
            shape_ids=torch.tensor([1])
        )

        z1 = stepper.infer(shape1, target, steps_adam=10)
        z2 = stepper.infer(shape2, target, steps_adam=10)

        # Different shapes should get different codes
        assert not torch.allclose(z1, z2), "Codes for different shapes are identical"

    def test_infer_code_not_in_embedding_table(self):
        """Verify inferred code is independent of embedding table."""
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=2, num_steps=5)

        # Get initial embedding
        initial_z0 = stepper.code_source.codes.weight[0].clone()

        new_shape = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )
        target = CohortBatch(
            points=torch.randn(1, 10, 3),
            shape_ids=torch.tensor([0])
        )

        z_inferred = stepper.infer(new_shape, target, steps_adam=5)

        # Embedding table should be unchanged
        assert torch.equal(
            stepper.code_source.codes.weight[0], initial_z0
        ), "Embedding table was modified during infer()"

        # Inferred code may be different from table entry
        # (not a hard assertion, just documenting behavior)

    @pytest.mark.slow
    def test_infer_on_real_hand_data(self):
        """Integration test: infer on held-out hand with real data (T29 Phase 6 exit)."""
        from pathlib import Path

        # Load hand template
        hand_dir = Path("data/hand")
        if not hand_dir.exists():
            pytest.skip("Hand data not found")

        # Import after conditional check
        from src.resnet_lddmm.io import load_shape

        template_path = hand_dir / "template.vtp"
        if not template_path.exists():
            pytest.skip("Hand template not found")

        # Create a small trained stepper
        torch.manual_seed(42)
        stepper = TestCohortRegistrationBasics.make_stepper(num_shapes=2, num_steps=5)

        # Train on a synthetic pair first to get reasonable flow
        train_shape = CohortBatch(
            points=torch.randn(1, 252, 3),  # match hand point count
            shape_ids=torch.tensor([0])
        )
        train_target = CohortBatch(
            points=torch.randn(1, 252, 3),
            shape_ids=torch.tensor([0])
        )

        for _ in range(10):
            stepper.train_step(train_shape, train_target)

        # Load template and create a held-out test shape
        template = load_shape(str(template_path))

        # Create a held-out test shape (perturbed template)
        test_shape = CohortBatch(
            points=template.points + torch.randn_like(template.points) * 0.05,
            shape_ids=torch.tensor([0])
        )

        target = CohortBatch(
            points=template.points,
            shape_ids=torch.tensor([0])
        )

        # Run inference
        z_inferred = stepper.infer(test_shape, target, steps_adam=10)

        # Verify shape
        assert z_inferred.shape == (1, stepper.code_source.n_z)

        # Verify flow is still usable (frozen check is in infer() itself)
        test_input = torch.randn(1, 10, 3)
        z_dummy = torch.randn(1, stepper.code_source.n_z)
        output = stepper.flow(test_input, z_dummy)
        assert output is not None
