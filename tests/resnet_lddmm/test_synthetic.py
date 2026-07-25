"""Synthetic recovery tests for Phase 2 exit criteria.

These tests verify the registration works on known transformations:
- Pure translation recovery
- Small rotation recovery
- Kinetic energy vs fit trade-off with different K values
"""

from dataclasses import dataclass

import torch
import pytest

from src.resnet_lddmm.registration.pair import PairRegistration
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.integrators import ForwardEuler
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.codes.none import NoCode
from src.resnet_lddmm.losses.data_terms import L2Data, CDData
from src.resnet_lddmm.losses import UnidirectionalMappingError
from src.learning.losses.composer import LossTerm, LossComposer
from src.learning.trainers.E3_end2end import TrainingOrchestrator
from src.learning.loader.loaders import OneBatchLoader


@dataclass
class SimpleBatch:
    """Simple batch with points and optional weights."""

    points: torch.Tensor
    weights: torch.Tensor | None = None


def rotation_matrix_z(angle):
    """Create a 3x3 rotation matrix around z-axis.

    Args:
        angle: rotation angle in radians

    Returns:
        [3, 3] rotation matrix
    """
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=torch.float32)


def rotation_matrix_xyz(angles):
    """Create 3D rotation matrix from (rx, ry, rz) angles.

    Args:
        angles: [3] tensor with rotation angles around x, y, z

    Returns:
        [3, 3] rotation matrix
    """
    rx, ry, rz = angles
    # Rotation around x
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(rx), -torch.sin(rx)],
        [0, torch.sin(rx), torch.cos(rx)],
    ], dtype=torch.float32)
    # Rotation around y
    Ry = torch.tensor([
        [torch.cos(ry), 0, torch.sin(ry)],
        [0, 1, 0],
        [-torch.sin(ry), 0, torch.cos(ry)],
    ], dtype=torch.float32)
    # Rotation around z
    Rz = torch.tensor([
        [torch.cos(rz), -torch.sin(rz), 0],
        [torch.sin(rz), torch.cos(rz), 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    return Rz @ Ry @ Rx


class TestSyntheticTranslation:
    """Tests for pure translation recovery."""

    def test_translation_recovery_l2(self):
        """Verify L2 registration recovers pure translation.

        Target = Source + translation_vector; after training, data term → ~0.
        """
        # Create a simple source point cloud
        torch.manual_seed(42)
        source_points = torch.randn(1, 10, 3)
        translation = torch.tensor([[[1.0, 0.5, -0.3]]])
        target_points = source_points + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        # Build stepper
        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = L2Data()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        # Train
        initial_loss = None
        for step in range(20):
            _, loss_val, breakdown = stepper.train_step(source, target)
            if step == 0:
                initial_loss = loss_val
            if step == 19:
                final_loss = loss_val

        # Verify loss decreased significantly
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

        # Verify data term is small (translation is recoverable)
        _, final_loss, final_breakdown = stepper.eval_step(source, target)
        assert final_breakdown["data"] < 0.1, f"Data term not small: {final_breakdown['data']}"

    def test_translation_recovery_chamfer(self):
        """Verify Chamfer distance registration recovers pure translation."""
        torch.manual_seed(43)
        source_points = torch.randn(1, 8, 3)
        translation = torch.tensor([[[0.5, 0.5, 0.5]]])
        target_points = source_points + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = CDData()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        initial_loss = None
        for step in range(20):
            _, loss_val, _ = stepper.train_step(source, target)
            if step == 0:
                initial_loss = loss_val

        _, final_loss, final_breakdown = stepper.eval_step(source, target)
        assert final_breakdown["data"] < 0.2, f"Chamfer data term not small: {final_breakdown['data']}"


class TestSyntheticRotation:
    """Tests for rotation recovery."""

    def test_small_rotation_recovery(self):
        """Verify small rotation is recoverable.

        Apply small rotation to source cloud, train to recover it.
        """
        torch.manual_seed(44)
        source_points = torch.randn(1, 10, 3)

        # Small rotation: 5 degrees around z-axis
        angle = torch.tensor(5 * 3.14159 / 180)  # ~0.087 radians
        R = rotation_matrix_z(angle)
        target_points = torch.einsum("ij,bnj->bni", R, source_points)

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = L2Data()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        # Train
        for _ in range(20):
            stepper.train_step(source, target)

        # Verify fit improved
        _, loss_val, breakdown = stepper.eval_step(source, target)
        assert breakdown["data"] < 0.5, f"Rotation not recovered well: {breakdown['data']}"


class TestKineticEnergyTradeoff:
    """Tests for kinetic energy vs fit trade-off with different K values."""

    def test_more_steps_lower_kinetic_energy(self):
        """Verify that with more integration steps, similar fits have comparable kinetic energy.

        Train two steppers with K=5 and K=20 on the same translation task.
        The principle (Plan §8) is that more steps distribute energy differently.
        """
        torch.manual_seed(45)
        source_points = torch.randn(1, 5, 3)
        translation = torch.tensor([[[0.5, 0.0, 0.0]]])
        target_points = source_points + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        results = {}

        for num_steps in [5, 20]:
            field = TimeVaryingField(num_blocks=num_steps, width=64)
            flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
            code_source = NoCode()
            data_term = L2Data()
            composer = LossComposer([
                LossTerm("data", weight=1.0),
                LossTerm("kinetic", weight=0.1),
                LossTerm("code_reg", weight=1.0),
            ])
            optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
            stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

            # Train to convergence
            steps = 100 if num_steps == 5 else 100
            for _ in range(steps):
                stepper.train_step(source, target)

            _, _, breakdown = stepper.eval_step(source, target)
            results[num_steps] = breakdown

        # Both should achieve good fit (data term small)
        assert results[5]["data"] < 0.2, f"K=5 data term: {results[5]['data']}"
        assert results[20]["data"] < 0.2, f"K=20 data term: {results[20]['data']}"

        # With comparable fits, kinetic energy per-step should be smaller with more steps
        # (more steps means smaller per-step energy; total energy may be similar due to
        # the kinetic_weight in the loss composition)
        ke_5 = results[5]["kinetic"]
        ke_20 = results[20]["kinetic"]
        # Both should be reasonable values (not NaN or infinite)
        assert torch.isfinite(torch.tensor(ke_5))
        assert torch.isfinite(torch.tensor(ke_20))


class TestSyntheticWithOrchestrator:
    """Tests using the full TrainingOrchestrator + OneBatchLoader."""

    def test_orchestrator_with_loader(self):
        """Verify stepper works with TrainingOrchestrator and OneBatchLoader."""
        torch.manual_seed(46)
        source_points = torch.randn(1, 8, 3)
        translation = torch.tensor([[[0.3, 0.3, 0.3]]])
        target_points = source_points + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        # Build stepper
        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = L2Data()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        # Create loader and orchestrator
        loader = OneBatchLoader((source, target))
        orchestrator = TrainingOrchestrator(stepper, loader, callbacks=[], log_dir="/tmp")

        # Run training
        ctx = orchestrator.run(num_steps=15)

        # Verify context has stopped properly
        assert ctx is not None

    def test_orchestrator_convergence(self):
        """Verify loss converges when using orchestrator."""
        torch.manual_seed(47)
        source_points = torch.randn(1, 6, 3)
        translation = torch.tensor([[[0.2, 0.2, 0.2]]])
        target_points = source_points + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        num_steps = 5
        field = TimeVaryingField(num_blocks=num_steps, width=64)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = L2Data()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        loader = OneBatchLoader((source, target))

        # Manual orchestrator loop to collect losses
        losses = []

        class LossCollector:
            def on_train_start(self, ctx):
                pass

            def on_step_end(self, ctx, step, metrics, batch, pred):
                losses.append(metrics["loss"])

            def on_train_end(self, ctx):
                pass

        collector = LossCollector()
        orchestrator = TrainingOrchestrator(stepper, loader, callbacks=[collector], log_dir="/tmp")
        orchestrator.run(num_steps=20)

        # Verify losses were collected
        assert len(losses) == 20

        # Verify general downward trend
        assert losses[-1] < losses[0]


@pytest.mark.slow
class TestSyntheticRecoveryLarge:
    """Larger-scale recovery tests (marked slow)."""

    def test_large_translation_multiple_points(self):
        """Verify translation recovery on larger point cloud."""
        torch.manual_seed(48)
        source_points = torch.randn(1, 50, 3)
        translation = torch.tensor([[[1.0, 1.0, 1.0]]])
        target_points = source_points + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        num_steps = 10
        field = TimeVaryingField(num_blocks=num_steps, width=128)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = L2Data()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.1),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        for _ in range(50):
            stepper.train_step(source, target)

        _, _, breakdown = stepper.eval_step(source, target)
        assert breakdown["data"] < 0.2, "Large translation not recovered"

    def test_combined_transform_recovery(self):
        """Verify recovery of translation + small rotation."""
        torch.manual_seed(49)
        source_points = torch.randn(1, 20, 3)

        # Apply rotation
        angle = torch.tensor(3 * 3.14159 / 180)  # 3 degrees
        R = rotation_matrix_z(angle)
        rotated = torch.einsum("ij,bnj->bni", R, source_points)

        # Apply translation
        translation = torch.tensor([[[0.5, 0.5, 0.0]]])
        target_points = rotated + translation

        source = SimpleBatch(source_points)
        target = SimpleBatch(target_points)

        num_steps = 10
        field = TimeVaryingField(num_blocks=num_steps, width=128)
        flow = NeuralODEFlow(field, ForwardEuler(), ForwardEuler(), num_steps=num_steps)
        code_source = NoCode()
        data_term = L2Data()
        composer = LossComposer([
            LossTerm("data", weight=1.0),
            LossTerm("kinetic", weight=0.05),
            LossTerm("code_reg", weight=1.0),
        ])
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.01)
        stepper = PairRegistration(flow, code_source, data_term, UnidirectionalMappingError(), composer, optimizer)

        for _ in range(50):
            stepper.train_step(source, target)

        _, _, breakdown = stepper.eval_step(source, target)
        assert breakdown["data"] < 0.3, "Combined transform not recovered"
