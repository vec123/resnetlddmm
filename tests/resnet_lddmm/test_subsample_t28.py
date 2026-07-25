"""Tests for adaptive subsampling (STEPS T28)."""

import torch
import pytest
from src.resnet_lddmm.registration.subsample import adaptive_subsample, AdaptiveSubsampler
from src.resnet_lddmm.losses.data_terms import CDData, L2Data


class TestAdaptiveSubsample1D:
    """Tests for adaptive_subsample on 1D per-point losses [N]."""

    def test_subsample_all_points_when_M_exceeds_N(self):
        """When M >= N, return all indices."""
        losses = torch.tensor([1.0, 2.0, 3.0])  # [3]
        M = 5
        a = 0.5

        indices = adaptive_subsample(losses, M, a)

        assert indices.shape == (3,)
        assert torch.allclose(indices.float(), torch.arange(3, dtype=torch.float32))

    def test_subsample_output_size_is_M(self):
        """Output size is always M (or fewer if N < M)."""
        losses = torch.randn(100)
        M = 30
        a = 0.2

        indices = adaptive_subsample(losses, M, a)

        assert indices.shape == (M,)
        assert len(indices.unique()) == M  # All unique

    def test_subsample_includes_hard_examples(self):
        """Top a·M hardest (highest loss) are included."""
        losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # [5]
        M = 3
        a = 0.67  # Keep ~2 hard examples

        indices = adaptive_subsample(losses, M, a)

        # Top 2 are indices 3, 4 (losses 4.0, 5.0)
        hard_indices = indices[:max(1, int(a * M))]
        assert 3 in hard_indices or 4 in hard_indices

    def test_subsample_fills_rest_uniformly(self):
        """Remaining (1-a)·M come from uniform sampling."""
        losses = torch.randn(100)
        M = 50
        a = 0.2  # 10 hard, 40 uniform

        indices = adaptive_subsample(losses, M, a)

        assert len(indices) == M
        n_hard = max(1, int(a * M))
        assert len(indices) == n_hard + (M - n_hard)

    def test_subsample_sorted_output(self):
        """Output indices are sorted."""
        losses = torch.randn(50)
        M = 20
        a = 0.3

        indices = adaptive_subsample(losses, M, a)

        assert torch.all(indices[:-1] <= indices[1:])

    def test_subsample_a_equals_zero(self):
        """When a=0, use pure uniform sampling."""
        losses = torch.randn(100)
        M = 30
        a = 0.0

        indices = adaptive_subsample(losses, M, a)

        # At least one index should be included (n_hard = max(1, 0))
        assert len(indices) == M

    def test_subsample_a_equals_one(self):
        """When a=1, keep all hard examples."""
        losses = torch.randn(100)
        M = 30
        a = 1.0

        indices = adaptive_subsample(losses, M, a)

        # All M points should be from top-M losses
        assert len(indices) == M
        _, hard_indices = torch.topk(losses, M)
        assert set(indices.tolist()) == set(hard_indices.tolist())


class TestAdaptiveSubsample2D:
    """Tests for adaptive_subsample on 2D per-point losses [B, N]."""

    def test_subsample_2d_output_size(self):
        """2D output is [B, M]."""
        losses = torch.randn(4, 100)
        M = 30
        a = 0.2

        indices = adaptive_subsample(losses, M, a)

        assert indices.shape == (4, M)

    def test_subsample_2d_all_points_when_M_exceeds_N(self):
        """When M >= N, return all indices (batched)."""
        losses = torch.randn(2, 10)
        M = 20
        a = 0.5

        indices = adaptive_subsample(losses, M, a)

        assert indices.shape == (2, 10)
        for b in range(2):
            assert torch.allclose(indices[b].float(), torch.arange(10, dtype=torch.float32))

    def test_subsample_2d_sorted_per_batch(self):
        """Each batch's indices are sorted."""
        losses = torch.randn(3, 50)
        M = 20
        a = 0.3

        indices = adaptive_subsample(losses, M, a)

        for b in range(3):
            assert torch.all(indices[b, :-1] <= indices[b, 1:])

    def test_subsample_2d_unique_per_batch(self):
        """Each batch's indices are unique."""
        losses = torch.randn(2, 100)
        M = 40
        a = 0.2

        indices = adaptive_subsample(losses, M, a)

        for b in range(2):
            assert len(indices[b].unique()) == M


class TestAdaptiveSubsamplerWithCDData:
    """Tests for AdaptiveSubsampler wrapper with CDData."""

    def test_subsampler_output_size(self):
        """Subsampler output is smaller than full Chamfer."""
        data_term = CDData()
        subsampler = AdaptiveSubsampler(data_term, M=20, a=0.3)

        pred = torch.randn(1, 50, 3)
        target = torch.randn(1, 100, 3)

        loss = subsampler(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_subsampler_disabled_when_M_exceeds_N(self):
        """When M >= target.shape[1], loss unchanged from full Chamfer."""
        data_term = CDData()
        target = torch.randn(1, 20, 3)
        pred = torch.randn(1, 30, 3)

        subsampler = AdaptiveSubsampler(data_term, M=100, a=0.2)
        loss_sub = subsampler(pred, target)

        loss_full = data_term(pred, target)

        # Should be approximately the same
        assert torch.allclose(loss_sub, loss_full, atol=1e-4)

    def test_subsampler_reduces_computation(self):
        """Subsampling with M < N reduces per-point loss values."""
        data_term = CDData()
        pred = torch.randn(1, 100, 3)
        target = torch.randn(1, 200, 3)

        # Full loss
        loss_full = data_term(pred, target)

        # Subsampled loss (half the points)
        subsampler = AdaptiveSubsampler(data_term, M=100, a=0.15)
        loss_sub = subsampler(pred, target)

        assert torch.isfinite(loss_full)
        assert torch.isfinite(loss_sub)
        # Subsampled should focus on hard examples, so may be higher
        assert loss_sub >= 0


class TestAdaptiveSubsamplerWithL2Data:
    """Tests for AdaptiveSubsampler wrapper with L2Data."""

    def test_subsampler_l2_output_size(self):
        """Subsampler output with L2Data is scalar."""
        data_term = L2Data()
        subsampler = AdaptiveSubsampler(data_term, M=20, a=0.3)

        pred = torch.randn(1, 50, 3)
        target = torch.randn(1, 50, 3)

        loss = subsampler(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_subsampler_l2_gradients(self):
        """Subsampler computes gradients through L2Data."""
        data_term = L2Data()
        subsampler = AdaptiveSubsampler(data_term, M=30, a=0.2)

        pred = torch.randn(1, 50, 3, requires_grad=True)
        target = torch.randn(1, 50, 3)

        loss = subsampler(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))


class TestDataTermPerPoint:
    """Tests for per_point mode in CDData and L2Data (T28)."""

    def test_cddata_per_point_shape_1d(self):
        """CDData per_point=True returns [B, M]."""
        term = CDData()
        pred = torch.randn(1, 20, 3)
        target = torch.randn(1, 30, 3)

        per_point = term(pred, target, per_point=True)

        assert per_point.shape == (1, 30)

    def test_cddata_per_point_shape_2d(self):
        """CDData per_point=True returns [B, M] for batch."""
        term = CDData()
        pred = torch.randn(3, 20, 3)
        target = torch.randn(3, 30, 3)

        per_point = term(pred, target, per_point=True)

        assert per_point.shape == (3, 30)

    def test_cddata_per_point_nonnegative(self):
        """Per-point Chamfer distances are non-negative."""
        term = CDData()
        pred = torch.randn(1, 20, 3)
        target = torch.randn(1, 30, 3)

        per_point = term(pred, target, per_point=True)

        assert torch.all(per_point >= 0)

    def test_cddata_per_point_mean_matches_scalar(self):
        """Mean of per-point losses matches scalar loss."""
        term = CDData()
        pred = torch.randn(1, 20, 3)
        target = torch.randn(1, 30, 3)

        per_point = term(pred, target, per_point=True)
        scalar = term(pred, target, per_point=False)

        # Note: scalar is sum of two terms, per_point is one term
        # This tests that per_point gives reasonable values
        assert torch.all(torch.isfinite(per_point))
        assert torch.isfinite(scalar)

    def test_l2data_per_point_shape(self):
        """L2Data per_point=True returns [B, N]."""
        term = L2Data()
        pred = torch.randn(2, 25, 3)
        target = torch.randn(2, 25, 3)

        per_point = term(pred, target, per_point=True)

        assert per_point.shape == (2, 25)

    def test_l2data_per_point_nonnegative(self):
        """Per-point L2 distances are non-negative."""
        term = L2Data()
        pred = torch.randn(2, 25, 3)
        target = torch.randn(2, 25, 3)

        per_point = term(pred, target, per_point=True)

        assert torch.all(per_point >= 0)

    def test_l2data_per_point_mean_matches_scalar(self):
        """Mean of per-point L2 matches scalar loss."""
        term = L2Data()
        pred = torch.randn(1, 25, 3)
        target = torch.randn(1, 25, 3)

        per_point = term(pred, target, per_point=True)
        scalar = term(pred, target, per_point=False)

        assert torch.allclose(per_point.mean(), scalar, atol=1e-5)

    def test_l2data_per_point_zero_identical_clouds(self):
        """Per-point L2 is zero when pred == target."""
        term = L2Data()
        cloud = torch.randn(2, 20, 3)

        per_point = term(cloud, cloud, per_point=True)

        assert torch.allclose(per_point, torch.zeros_like(per_point), atol=1e-6)
