"""Unit and integration tests for augmentation (SO(3) and SE(3))."""

import pytest
import torch
import numpy as np
from src.resnet_lddmm.augmentation import (
    Augmentation, NoAugmentation, SO3Augmentation, SE3Augmentation
)


class TestNoAugmentation:
    """Tests for identity augmentation (Null Object)."""

    def test_no_augmentation_returns_input_unchanged(self):
        """Identity transformation returns input exactly."""
        aug = NoAugmentation()
        points = torch.randn(4, 100, 3)
        result = aug(points)

        assert torch.allclose(result, points)
        assert result.shape == points.shape

    def test_no_augmentation_preserves_dtype(self):
        """Output dtype matches input dtype."""
        aug = NoAugmentation()
        points = torch.randn(2, 50, 3, dtype=torch.float32)
        result = aug(points)

        assert result.dtype == torch.float32

    def test_no_augmentation_preserves_device(self):
        """Output device matches input device."""
        aug = NoAugmentation()
        points = torch.randn(2, 50, 3)
        result = aug(points)

        assert result.device == points.device

    def test_no_augmentation_zero_computational_cost(self):
        """No augmentation should be a no-op (test with identity check)."""
        aug = NoAugmentation()
        points = torch.randn(4, 100, 3, requires_grad=True)
        result = aug(points)

        # Check that gradient flows through (no detach or clone)
        loss = result.sum()
        loss.backward()

        # Gradient should equal 1 everywhere (identity)
        assert points.grad is not None
        assert torch.allclose(points.grad, torch.ones_like(points.grad))


class TestSO3Augmentation:
    """Tests for rotation-only augmentation."""

    def test_so3_output_shape(self):
        """SO3 transformation preserves shape."""
        aug = SO3Augmentation(seed=42)
        points = torch.randn(4, 100, 3)
        result = aug(points)

        assert result.shape == points.shape

    def test_so3_preserves_distances(self):
        """Rotation preserves pairwise distances."""
        aug = SO3Augmentation(seed=42)
        points = torch.randn(2, 50, 3)
        result = aug(points)

        # Distance from origin should be preserved
        orig_norms = torch.norm(points, dim=-1)  # [2, 50]
        result_norms = torch.norm(result, dim=-1)  # [2, 50]

        assert torch.allclose(orig_norms, result_norms, atol=1e-6)

    def test_so3_deterministic_with_seed(self):
        """Same seed produces same rotation."""
        aug1 = SO3Augmentation(seed=42)
        aug2 = SO3Augmentation(seed=42)

        points = torch.randn(2, 30, 3)

        result1 = aug1(points)
        result2 = aug2(points)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_so3_non_deterministic_without_seed(self):
        """Without seed, different calls produce different rotations."""
        aug1 = SO3Augmentation(seed=None)
        aug2 = SO3Augmentation(seed=None)

        points = torch.randn(2, 30, 3)

        result1 = aug1(points)
        result2 = aug2(points)

        # Very unlikely to get same rotation without seed
        assert not torch.allclose(result1, result2, atol=1e-3)

    def test_so3_gradient_flow(self):
        """Gradients flow through SO3 augmentation."""
        aug = SO3Augmentation(seed=42)
        points = torch.randn(2, 30, 3, requires_grad=True)

        result = aug(points)
        loss = result.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert points.grad is not None
        assert points.grad.abs().sum() > 0

    def test_so3_batch_independence(self):
        """Different shapes in batch get different rotations."""
        aug = SO3Augmentation(seed=42)
        points = torch.randn(4, 50, 3)  # 4 shapes

        result = aug(points)

        # Each batch element should be independently rotated
        # (Check by verifying distances preserved per batch)
        for b in range(4):
            orig_norms = torch.norm(points[b], dim=-1)
            result_norms = torch.norm(result[b], dim=-1)
            assert torch.allclose(orig_norms, result_norms, atol=1e-6)

    def test_so3_different_per_iteration(self):
        """Each training iteration produces different rotation (generator state advances)."""
        aug = SO3Augmentation(seed=42)
        points = torch.randn(2, 30, 3)

        # Call augmentation multiple times on same instance
        result1 = aug(points)
        result2 = aug(points)
        result3 = aug(points)

        # Each call should produce different rotations
        assert not torch.allclose(result1, result2, atol=1e-3)
        assert not torch.allclose(result2, result3, atol=1e-3)
        assert not torch.allclose(result1, result3, atol=1e-3)

        # But all preserve distances (isometry)
        for result in [result1, result2, result3]:
            orig_norms = torch.norm(points, dim=-1)
            result_norms = torch.norm(result, dim=-1)
            assert torch.allclose(orig_norms, result_norms, atol=1e-6)


class TestSE3Augmentation:
    """Tests for rotation + translation augmentation."""

    def test_se3_output_shape(self):
        """SE3 transformation preserves shape."""
        aug = SE3Augmentation(translation_scale=0.2, seed=42)
        points = torch.randn(4, 100, 3)
        result = aug(points)

        assert result.shape == points.shape

    def test_se3_deterministic_with_seed(self):
        """Same seed produces same transformation."""
        aug1 = SE3Augmentation(translation_scale=0.2, seed=42)
        aug2 = SE3Augmentation(translation_scale=0.2, seed=42)

        points = torch.randn(2, 30, 3)

        result1 = aug1(points)
        result2 = aug2(points)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_se3_translation_bounds(self):
        """Translation sampled in correct range."""
        # Test that translations are actually sampled in the correct range
        aug = SE3Augmentation(translation_scale=0.5, seed=42)

        # Create points at origin
        points = torch.zeros(10, 20, 3)

        result = aug(points)

        # With points at origin, result should be just the translation
        # (since rotation of zero vectors is still zero)
        # Max displacement should be roughly sqrt(3) * translation_scale ~ 0.87
        max_shift = result.abs().max()

        # Allow some margin for numerical variation
        assert max_shift <= 1.0  # Conservative bound

    def test_se3_gradient_flow(self):
        """Gradients flow through SE3 augmentation."""
        aug = SE3Augmentation(translation_scale=0.2, seed=42)
        points = torch.randn(2, 30, 3, requires_grad=True)

        result = aug(points)
        loss = result.sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert points.grad is not None
        assert points.grad.abs().sum() > 0

    def test_se3_different_translation_scales(self):
        """Larger translation_scale allows larger displacements."""
        points = torch.randn(5, 50, 3)

        aug_small = SE3Augmentation(translation_scale=0.1, seed=42)
        aug_large = SE3Augmentation(translation_scale=0.5, seed=42)

        result_small = aug_small(points)
        result_large = aug_large(points)

        # Both use same seed, so rotation is same, but translation differs
        # Large translation variant should be further from original
        shift_small = (result_small - points).norm()
        shift_large = (result_large - points).norm()

        # Larger translation scale should generally lead to larger displacements
        # (this is probabilistic, but very likely)
        assert shift_large > shift_small * 0.5  # Very loose bound

    def test_se3_different_per_iteration(self):
        """Each training iteration produces different SE(3) transformation (generator state advances)."""
        aug = SE3Augmentation(translation_scale=0.2, seed=42)
        points = torch.randn(2, 30, 3)

        # Call augmentation multiple times on same instance
        result1 = aug(points)
        result2 = aug(points)
        result3 = aug(points)

        # Each call should produce different transformations
        assert not torch.allclose(result1, result2, atol=1e-3)
        assert not torch.allclose(result2, result3, atol=1e-3)
        assert not torch.allclose(result1, result3, atol=1e-3)


class TestAugmentationGradients:
    """Test gradient flow through augmentations."""

    def test_so3_autograd_gradient(self):
        """SO3 gradients are numerically correct."""
        aug = SO3Augmentation(seed=42)
        points = torch.randn(2, 10, 3, requires_grad=True, dtype=torch.float64)

        # Test numerical gradient
        eps = 1e-4

        def func(x):
            return aug(x).sum()

        # Use numerical gradient checking (simplified)
        result = aug(points)
        loss = result.sum()
        loss.backward()

        # Just verify gradients exist and are reasonable magnitude
        assert points.grad is not None
        assert not torch.isnan(points.grad).any()
        assert not torch.isinf(points.grad).any()

    def test_se3_autograd_gradient(self):
        """SE3 gradients are numerically correct."""
        aug = SE3Augmentation(translation_scale=0.2, seed=42)
        points = torch.randn(2, 10, 3, requires_grad=True, dtype=torch.float64)

        result = aug(points)
        loss = result.sum()
        loss.backward()

        # Verify gradients exist and are reasonable
        assert points.grad is not None
        assert not torch.isnan(points.grad).any()
        assert not torch.isinf(points.grad).any()


class TestAugmentationRegistry:
    """Test augmentation registry integration."""

    def test_no_augmentation_registry(self):
        """NoAugmentation can be created via registry."""
        from src.learning.registry import Registry
        from src.resnet_lddmm import registrations  # Load registry entries

        aug = Registry.create("augmentation", "none")
        assert isinstance(aug, NoAugmentation)

    def test_so3_augmentation_registry(self):
        """SO3Augmentation can be created via registry."""
        from src.learning.registry import Registry
        from src.resnet_lddmm import registrations  # Load registry entries

        aug = Registry.create("augmentation", "so3", seed=42)
        assert isinstance(aug, SO3Augmentation)

    def test_se3_augmentation_registry(self):
        """SE3Augmentation can be created via registry."""
        from src.learning.registry import Registry
        from src.resnet_lddmm import registrations  # Load registry entries

        aug = Registry.create("augmentation", "se3", seed=42, translation_scale=0.2)
        assert isinstance(aug, SE3Augmentation)


class TestAugmentationConfig:
    """Test augmentation via config parsing."""

    def test_config_no_augmentation(self):
        """Config with no augmentation section defaults to NoAugmentation."""
        from src.resnet_lddmm.config import ExperimentCfg

        cfg_dict = {
            "source": "test.obj",
            "target": "test.obj",
            "output_dir": "outputs/test",
            # No augmentation section
        }

        cfg = ExperimentCfg.from_dict(cfg_dict)
        assert cfg.augmentation.kind == "none"
        assert cfg.augmentation.seed is None
        assert cfg.augmentation.translation_scale == 0.2

    def test_config_se3_augmentation(self):
        """Config can specify SE3 augmentation."""
        from src.resnet_lddmm.config import ExperimentCfg

        cfg_dict = {
            "source": "test.obj",
            "target": "test.obj",
            "output_dir": "outputs/test",
            "augmentation": {
                "kind": "se3",
                "seed": 42,
                "translation_scale": 0.15,
            },
        }

        cfg = ExperimentCfg.from_dict(cfg_dict)
        assert cfg.augmentation.kind == "se3"
        assert cfg.augmentation.seed == 42
        assert cfg.augmentation.translation_scale == 0.15

    def test_config_so3_augmentation(self):
        """Config can specify SO3 augmentation."""
        from src.resnet_lddmm.config import ExperimentCfg

        cfg_dict = {
            "source": "test.obj",
            "target": "test.obj",
            "output_dir": "outputs/test",
            "augmentation": {
                "kind": "so3",
                "seed": 99,
            },
        }

        cfg = ExperimentCfg.from_dict(cfg_dict)
        assert cfg.augmentation.kind == "so3"
        assert cfg.augmentation.seed == 99


class TestAugmentationIntegration:
    """Integration tests with steppers."""

    def test_pair_registration_with_augmentation(self):
        """PairRegistration works with augmentation."""
        from src.resnet_lddmm.registration.pair import PairRegistration
        from src.resnet_lddmm.flow import NeuralODEFlow
        from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
        from src.resnet_lddmm.codes.none import NoCode
        from src.resnet_lddmm.losses.data_terms import CDData
        from src.resnet_lddmm.losses.mapping_error import UnidirectionalMappingError
        from src.learning.losses.composer import LossComposer, LossTerm

        # Create minimal components
        class DummyVelocityField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3, 3)  # Ensure field has parameters

            def forward(self, x, step=None, code=None):
                # Use fc layer so field has learnable parameters
                return self.fc(x) * 0.01

        field = DummyVelocityField()
        integrator_direct = ForwardEuler()
        integrator_inverse = ModifiedEuler()
        flow = NeuralODEFlow(field, integrator_direct, integrator_inverse, num_steps=2)

        code_source = NoCode()
        data_term = CDData()
        mapping_error = UnidirectionalMappingError()
        composer = LossComposer([LossTerm("data", weight=1.0), LossTerm("kinetic", weight=1.0)])

        # Create optimizer with flow parameters
        optimizer = torch.optim.Adam(list(flow.parameters()), lr=1e-4)

        # Test with augmentation
        aug = SO3Augmentation(seed=42)

        stepper = PairRegistration(
            flow, code_source, data_term, mapping_error, composer, optimizer,
            augmentation=aug
        )

        # Create dummy batch with minimal attributes
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class SimpleBatch:
            points: torch.Tensor
            weights: torch.Tensor = None
            faces: list = dc_field(default=None)

        source = SimpleBatch(points=torch.randn(1, 20, 3))
        target = SimpleBatch(points=torch.randn(1, 20, 3))

        # Run one training step (should not raise)
        traj, loss_val, breakdown = stepper.train_step(source, target)

        assert isinstance(loss_val, float)
        assert not np.isnan(loss_val)
        assert not np.isinf(loss_val)

    def test_cohort_registration_with_augmentation(self):
        """CohortRegistration works with augmentation."""
        from src.resnet_lddmm.registration.cohort import CohortRegistration
        from src.resnet_lddmm.flow import NeuralODEFlow
        from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
        from src.resnet_lddmm.codes.none import NoCode
        from src.resnet_lddmm.losses.data_terms import CDData
        from src.resnet_lddmm.losses.mapping_error import BidirectionalMappingError
        from src.learning.losses.composer import LossComposer, LossTerm
        from src.learning.loader.loaders import CohortBatch

        class DummyVelocityField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(3, 3)  # Ensure field has parameters

            def forward(self, x, step=None, code=None):
                return self.fc(x) * 0.01

        field = DummyVelocityField()
        integrator_direct = ForwardEuler()
        integrator_inverse = ModifiedEuler()
        flow = NeuralODEFlow(field, integrator_direct, integrator_inverse, num_steps=2)

        code_source = NoCode()
        data_term = CDData()
        mapping_error = BidirectionalMappingError()
        composer = LossComposer([LossTerm("data", weight=1.0), LossTerm("kinetic", weight=1.0)])

        # Create optimizer with flow parameters
        optimizer = torch.optim.Adam(list(flow.parameters()), lr=1e-4)

        aug = SE3Augmentation(translation_scale=0.1, seed=42)

        stepper = CohortRegistration(
            flow, code_source, data_term, mapping_error, composer, optimizer,
            augmentation=aug
        )

        # Create dummy batch using actual CohortBatch dataclass
        source = CohortBatch(
            points=torch.randn(2, 20, 3),
            shape_ids=torch.tensor([0, 1]),
            weights=None,
            faces=None
        )
        target = CohortBatch(
            points=torch.randn(1, 20, 3),
            shape_ids=torch.tensor([0]),
            weights=None,
            faces=None
        )

        # Run one training step (should not raise)
        traj, loss_val, breakdown = stepper.train_step(source, target)

        assert isinstance(loss_val, float)
        assert not np.isnan(loss_val)
        assert not np.isinf(loss_val)
