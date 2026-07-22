"""Tests for DataTerm implementations."""

import torch
import pytest
from src.resnet_lddmm.losses.data_terms import DataTerm, CDData, L2Data


class TestDataTermABC:
    """Tests for DataTerm abstract base class."""

    def test_data_term_is_abstract(self):
        """Verify DataTerm cannot be instantiated."""
        with pytest.raises(TypeError):
            DataTerm()


class TestCDDataBasics:
    """Basic tests for CDData Chamfer distance."""

    def test_cddata_is_data_term(self):
        """Verify CDData is a DataTerm."""
        assert issubclass(CDData, DataTerm)

    def test_cddata_callable(self):
        """Verify CDData is callable."""
        term = CDData()
        assert callable(term)

    def test_cddata_returns_scalar(self):
        """Verify CDData returns a scalar tensor."""
        term = CDData()
        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        loss = term(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32


class TestCDDataIdentical:
    """Tests for CDData on identical clouds."""

    def test_cddata_zero_on_identical(self):
        """Verify CDData is zero when pred == target."""
        term = CDData()
        cloud = torch.randn(2, 20, 3)

        loss = term(cloud, cloud)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_cddata_zero_on_identical_various_sizes(self):
        """Verify CDData is zero for various cloud sizes."""
        term = CDData()

        for B in [1, 2, 4]:
            for N in [5, 10, 20]:
                cloud = torch.randn(B, N, 3)
                loss = term(cloud, cloud)
                assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


class TestCDDataSymmetry:
    """Tests for CDData symmetry property."""

    def test_cddata_symmetric(self):
        """Verify CDData(pred, target) == CDData(target, pred)."""
        term = CDData()
        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 15, 3)

        loss_12 = term(pred, target)
        loss_21 = term(target, pred)

        assert torch.allclose(loss_12, loss_21, atol=1e-5)

    def test_cddata_symmetric_various_sizes(self):
        """Verify symmetry holds for various cloud sizes."""
        term = CDData()

        for _ in range(5):
            N = torch.randint(5, 30, (1,)).item()
            M = torch.randint(5, 30, (1,)).item()
            pred = torch.randn(1, N, 3)
            target = torch.randn(1, M, 3)

            loss_12 = term(pred, target)
            loss_21 = term(target, pred)
            assert torch.allclose(loss_12, loss_21, atol=1e-5)


class TestCDDataManualFixture:
    """Test CDData against hand-computed Chamfer on tiny fixture."""

    def test_cddata_manual_2point_fixture(self):
        """Verify CDData matches manual O(N²) computation on 2-point clouds.

        Points:
        pred = [[0, 0, 0], [1, 0, 0]]
        target = [[0.5, 0, 0], [1.5, 0, 0]]

        Distances:
        pred[0] to target: [0.5, 1.5] -> min 0.5, dist² 0.25
        pred[1] to target: [0.5, 0.5] -> min 0.5, dist² 0.25

        target[0] to pred: [0.5, 1.5] -> min 0.5, dist² 0.25
        target[1] to pred: [0.5, 0.5] -> min 0.5, dist² 0.25

        Chamfer = mean(0.25, 0.25) + mean(0.25, 0.25) = 0.5
        """
        term = CDData()
        pred = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])  # [1, 2, 3]
        target = torch.tensor([[[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]])  # [1, 2, 3]

        loss = term(pred, target)
        expected = torch.tensor(0.5)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_cddata_single_point_fixture(self):
        """Verify CDData on single-point clouds.

        pred = [[0, 0, 0]]
        target = [[1, 1, 1]]

        Distance² = 1 + 1 + 1 = 3
        Chamfer = 3 + 3 = 6
        """
        term = CDData()
        pred = torch.tensor([[[0.0, 0.0, 0.0]]])  # [1, 1, 3]
        target = torch.tensor([[[1.0, 1.0, 1.0]]])  # [1, 1, 3]

        loss = term(pred, target)
        expected = torch.tensor(6.0)
        assert torch.allclose(loss, expected, atol=1e-5)


class TestCDDataNonzero:
    """Tests for CDData on different clouds."""

    def test_cddata_positive_on_different_clouds(self):
        """Verify CDData > 0 when pred != target."""
        term = CDData()
        pred = torch.zeros(1, 5, 3)
        target = torch.ones(1, 5, 3)

        loss = term(pred, target)
        assert loss > 0

    def test_cddata_gradient_flow(self):
        """Verify gradients flow through CDData."""
        term = CDData()
        pred = torch.randn(1, 5, 3, requires_grad=True)
        target = torch.randn(1, 5, 3)

        loss = term(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))


class TestL2DataBasics:
    """Basic tests for L2Data."""

    def test_l2data_is_data_term(self):
        """Verify L2Data is a DataTerm."""
        assert issubclass(L2Data, DataTerm)

    def test_l2data_callable(self):
        """Verify L2Data is callable."""
        term = L2Data()
        assert callable(term)

    def test_l2data_returns_scalar(self):
        """Verify L2Data returns a scalar tensor."""
        term = L2Data()
        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 10, 3)

        loss = term(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32


class TestL2DataIdentical:
    """Tests for L2Data on identical clouds."""

    def test_l2data_zero_on_identical(self):
        """Verify L2Data is zero when pred == target."""
        term = L2Data()
        cloud = torch.randn(2, 20, 3)

        loss = term(cloud, cloud)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_l2data_zero_on_identical_various_sizes(self):
        """Verify L2Data is zero for various cloud sizes."""
        term = L2Data()

        for B in [1, 2, 4]:
            for N in [5, 10, 20]:
                cloud = torch.randn(B, N, 3)
                loss = term(cloud, cloud)
                assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


class TestL2DataManualComputation:
    """Test L2Data against hand-computed values."""

    def test_l2data_single_point_fixture(self):
        """Verify L2Data on single point.

        pred = [[1, 2, 3]]
        target = [[0, 0, 0]]

        Error² per point: (1² + 2² + 3²) = 14
        MSE = 14 / 1 = 14
        """
        term = L2Data()
        pred = torch.tensor([[[1.0, 2.0, 3.0]]])  # [1, 1, 3]
        target = torch.tensor([[[0.0, 0.0, 0.0]]])  # [1, 1, 3]

        loss = term(pred, target)
        expected = torch.tensor(14.0)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_l2data_two_point_fixture(self):
        """Verify L2Data on two points.

        Batch of 1, 2 points each:
        pred = [[0, 0, 0], [1, 0, 0]]
        target = [[1, 0, 0], [0, 0, 0]]

        Error² per point: [1, 1]
        MSE = (1 + 1) / 2 = 1
        """
        term = L2Data()
        pred = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )  # [1, 2, 3]
        target = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )  # [1, 2, 3]

        loss = term(pred, target)
        expected = torch.tensor(1.0)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_l2data_multi_batch_fixture(self):
        """Verify L2Data averages over batch correctly.

        Batch 0: pred [0, 0, 0], target [1, 0, 0] -> error² = 1
        Batch 1: pred [0, 0, 0], target [2, 0, 0] -> error² = 4

        Mean error² = (1 + 4) / 2 = 2.5
        """
        term = L2Data()
        # Two batches, one point each
        pred = torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])  # [2, 1, 3]
        target = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])  # [2, 1, 3]

        loss = term(pred, target)
        expected = torch.tensor(2.5)
        assert torch.allclose(loss, expected, atol=1e-5)


class TestL2DataGradients:
    """Tests for L2Data gradients."""

    def test_l2data_gradient_flow(self):
        """Verify gradients flow through L2Data."""
        term = L2Data()
        pred = torch.randn(2, 10, 3, requires_grad=True)
        target = torch.randn(2, 10, 3)

        loss = term(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_l2data_positive_nonzero_gradient(self):
        """Verify nonzero pred yields nonzero gradient."""
        term = L2Data()
        pred = torch.ones(1, 1, 3, requires_grad=True)
        target = torch.zeros(1, 1, 3)

        loss = term(pred, target)
        loss.backward()

        # Gradient should be nonzero and match manual derivative
        # d(L2) / d(pred) = 2 * (pred - target) / (B*N)
        # = 2 * 1 / (1*1) = 2 (per component before averaging)
        # After mean: 2 * 1 / (1 * 1) = 2
        expected_grad = torch.ones(1, 1, 3) * 2
        assert torch.allclose(pred.grad, expected_grad, atol=1e-5)


class TestDataTermIgnoresOptionalArgs:
    """Tests that basic terms ignore optional arguments correctly."""

    def test_cddata_ignores_weights(self):
        """Verify CDData ignores pred_w, tgt_w arguments."""
        term = CDData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)
        pred_w = torch.ones(1, 5)
        tgt_w = torch.ones(1, 5)

        loss_with_w = term(pred, target, pred_w=pred_w, tgt_w=tgt_w)
        loss_without_w = term(pred, target)

        assert torch.allclose(loss_with_w, loss_without_w, atol=1e-6)

    def test_cddata_ignores_normals(self):
        """Verify CDData ignores normals argument."""
        term = CDData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)
        normals = torch.randn(1, 5, 3)

        loss_with_n = term(pred, target, normals=normals)
        loss_without_n = term(pred, target)

        assert torch.allclose(loss_with_n, loss_without_n, atol=1e-6)

    def test_l2data_ignores_weights(self):
        """Verify L2Data ignores weight arguments."""
        term = L2Data()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)
        pred_w = torch.ones(1, 5)
        tgt_w = torch.ones(1, 5)

        loss_with_w = term(pred, target, pred_w=pred_w, tgt_w=tgt_w)
        loss_without_w = term(pred, target)

        assert torch.allclose(loss_with_w, loss_without_w, atol=1e-6)

    def test_l2data_ignores_normals(self):
        """Verify L2Data ignores normals argument."""
        term = L2Data()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)
        normals = torch.randn(1, 5, 3)

        loss_with_n = term(pred, target, normals=normals)
        loss_without_n = term(pred, target)

        assert torch.allclose(loss_with_n, loss_without_n, atol=1e-6)
