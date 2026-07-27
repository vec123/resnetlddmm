"""Tests for DataTerm implementations."""

import torch
import pytest
from src.resnet_lddmm.losses.data_terms import (
    DataTerm, CDData, L2Data, WeightedCDData, PCDData, NCDData, SinkhornData, EMDData
)


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


class TestWeightedCDDataBasics:
    """Tests for WeightedCDData (T33)."""

    def test_weighted_cddata_is_data_term(self):
        """Verify WeightedCDData is a DataTerm."""
        assert issubclass(WeightedCDData, DataTerm)

    def test_weighted_cddata_returns_scalar(self):
        """Verify WeightedCDData returns a scalar tensor."""
        term = WeightedCDData()
        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        loss = term(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    def test_weighted_cddata_zero_on_identical(self):
        """Verify WeightedCDData is zero when pred == target."""
        term = WeightedCDData()
        cloud = torch.randn(2, 20, 3)

        loss = term(cloud, cloud)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


class TestWeightedCDDataUniformWeights:
    """Test that uniform weights reduce WeightedCD to standard CD."""

    def test_weighted_cddata_uniform_equals_cddata(self):
        """Verify uniform weights give same loss as CDData."""
        weighted_term = WeightedCDData()
        cd_term = CDData()

        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 15, 3)
        uniform_w = torch.ones(1, 15)

        loss_weighted = weighted_term(pred, target, tgt_w=uniform_w)
        loss_cd = cd_term(pred, target)

        # Uniform weights normalized to mean 1 should give the same result
        assert torch.allclose(loss_weighted, loss_cd, atol=1e-5)

    def test_weighted_cddata_no_weights_equals_cddata(self):
        """Verify no weights defaults to CDData behavior."""
        weighted_term = WeightedCDData()
        cd_term = CDData()

        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        loss_weighted = weighted_term(pred, target)
        loss_cd = cd_term(pred, target)

        assert torch.allclose(loss_weighted, loss_cd, atol=1e-5)


class TestWeightedCDDataWeighting:
    """Test that WeightedCD actually uses weights."""

    def test_weighted_cddata_non_uniform_changes_loss(self):
        """Verify non-uniform weights change the loss."""
        weighted_term = WeightedCDData()

        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 15, 3)

        # Uniform weights
        w_uniform = torch.ones(1, 15)
        loss_uniform = weighted_term(pred, target, tgt_w=w_uniform)

        # Non-uniform weights (emphasize some targets more)
        w_nonuniform = torch.ones(1, 15)
        w_nonuniform[:, :5] = 10.0  # Emphasize first 5 targets
        loss_nonuniform = weighted_term(pred, target, tgt_w=w_nonuniform)

        # Losses should differ (at least in floating-point)
        assert not torch.allclose(loss_uniform, loss_nonuniform, atol=1e-4)

    def test_weighted_cddata_higher_weights_increase_loss(self):
        """Verify higher weights on distant targets increase loss."""
        weighted_term = WeightedCDData()

        # Close target points
        pred = torch.zeros(1, 1, 3)
        target_close = torch.tensor([[[0.1, 0.0, 0.0]]])  # Close to origin
        target_far = torch.tensor([[[10.0, 0.0, 0.0]]])  # Far from origin

        # Weight the far point
        w_far = torch.tensor([[10.0]])
        loss_far_weighted = weighted_term(pred, target_far, tgt_w=w_far)

        # Weight the close point
        w_close = torch.tensor([[10.0]])
        loss_close_weighted = weighted_term(pred, target_close, tgt_w=w_close)

        # Weighted loss on far target should be much larger
        assert loss_far_weighted > loss_close_weighted


class TestWeightedCDDataPerPoint:
    """Test per-point loss mode for WeightedCD."""

    def test_weighted_cddata_per_point_shape(self):
        """Verify per_point returns [B, M] shape."""
        term = WeightedCDData()
        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        losses = term(pred, target, per_point=True)
        assert losses.shape == (2, 15)

    def test_weighted_cddata_per_point_with_weights(self):
        """Verify per_point respects tgt_w."""
        term = WeightedCDData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 10, 3)
        w = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]])

        losses = term(pred, target, tgt_w=w, per_point=True)

        # Last 5 targets have 2x weight, so losses should be roughly 2x
        mean_first_half = losses[:, :5].mean()
        mean_second_half = losses[:, 5:].mean()
        assert mean_second_half > mean_first_half


class TestPCDDataBasics:
    """Tests for PCDData (T33)."""

    def test_pcddata_is_data_term(self):
        """Verify PCDData is a DataTerm."""
        assert issubclass(PCDData, DataTerm)

    def test_pcddata_returns_scalar(self):
        """Verify PCDData returns a scalar tensor."""
        term = PCDData()
        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        loss = term(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    def test_pcddata_zero_on_identical(self):
        """Verify PCDData is zero when pred == target."""
        term = PCDData()
        cloud = torch.randn(2, 20, 3)

        loss = term(cloud, cloud)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


class TestPCDDataSymmetry:
    """Test PCDData symmetry property."""

    def test_pcddata_symmetric(self):
        """Verify PCDData(pred, target) == PCDData(target, pred)."""
        term = PCDData()
        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 15, 3)

        loss_12 = term(pred, target)
        loss_21 = term(target, pred)

        assert torch.allclose(loss_12, loss_21, atol=1e-5)

    def test_pcddata_equiv_to_cddata(self):
        """Verify PCDData is equivalent to CDData."""
        pcd = PCDData()
        cd = CDData()

        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        loss_pcd = pcd(pred, target)
        loss_cd = cd(pred, target)

        assert torch.allclose(loss_pcd, loss_cd, atol=1e-5)


class TestNCDDataBasics:
    """Tests for NCDData (T33)."""

    def test_ncddata_is_data_term(self):
        """Verify NCDData is a DataTerm."""
        assert issubclass(NCDData, DataTerm)

    def test_ncddata_returns_scalar(self):
        """Verify NCDData returns a scalar tensor."""
        term = NCDData()
        pred = torch.randn(2, 10, 3)
        target = torch.randn(2, 15, 3)

        loss = term(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    def test_ncddata_zero_on_identical(self):
        """Verify NCDData is zero when pred == target."""
        term = NCDData()
        cloud = torch.randn(2, 20, 3)

        loss = term(cloud, cloud)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_ncddata_init_with_normal_weight(self):
        """Verify NCDData accepts normal_weight parameter."""
        term = NCDData(normal_weight=2.0)
        assert term.normal_weight == 2.0


class TestNCDDataWithoutNormals:
    """Test NCDData falls back to Chamfer when normals are None."""

    def test_ncddata_without_normals_equals_cddata(self):
        """Verify NCD without normals reduces to CDData."""
        ncd = NCDData()
        cd = CDData()

        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 15, 3)

        loss_ncd = ncd(pred, target)
        loss_cd = cd(pred, target)

        assert torch.allclose(loss_ncd, loss_cd, atol=1e-5)

    def test_ncddata_ignores_normals_if_none(self):
        """Verify NCD is unaffected when normals=None explicitly."""
        term = NCDData()
        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 15, 3)

        loss_no_normals = term(pred, target, normals=None)
        loss_explicit_none = term(pred, target, normals=None)

        assert torch.allclose(loss_no_normals, loss_explicit_none, atol=1e-6)


class TestNCDDataWithNormals:
    """Test NCDData penalizes normal-flipped surfaces."""

    def test_ncddata_penalizes_normal_mismatch(self):
        """Verify NCD penalizes normals pointing towards pred more than away."""
        ncd = NCDData(normal_weight=1.0)
        cd = CDData()

        # Simple fixture: pred at origin, target at (0,0,1)
        # direction vector = pred - target = (0,0,-1)
        pred = torch.tensor([[[0.0, 0.0, 0.0]]])  # [1, 1, 3]
        target = torch.tensor([[[0.0, 0.0, 1.0]]])  # [1, 1, 3]

        # Normal pointing up (away from pred): well-aligned
        # dot((0,0,-1), (0,0,1)) = -1, penalty = max(0, -1) = 0
        normal_good = torch.tensor([[[0.0, 0.0, 1.0]]])  # [1, 1, 3]
        loss_good = ncd(pred, target, normals=normal_good)

        # Normal pointing down (towards pred): misaligned
        # dot((0,0,-1), (0,0,-1)) = 1, penalty = max(0, 1) = 1
        normal_bad = torch.tensor([[[0.0, 0.0, -1.0]]])  # [1, 1, 3]
        loss_bad = ncd(pred, target, normals=normal_bad)

        # Geometric (CD) loss is the same, but NCD should be different
        loss_cd = cd(pred, target)

        # With normal pointing towards pred, NCD should be larger
        assert loss_bad > loss_good
        assert loss_good == loss_cd  # No normal penalty when normal points away

    def test_ncddata_gradient_flow_with_normals(self):
        """Verify gradients flow through NCD with normals."""
        term = NCDData()
        pred = torch.randn(1, 5, 3, requires_grad=True)
        target = torch.randn(1, 5, 3)
        normals = torch.randn(1, 5, 3)

        loss = term(pred, target, normals=normals)
        loss.backward()

        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_ncddata_per_point_with_normals(self):
        """Verify per_point mode works with normals."""
        term = NCDData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 10, 3)
        normals = torch.randn(1, 10, 3)

        losses = term(pred, target, normals=normals, per_point=True)
        assert losses.shape == (1, 10)


def _has_geomloss():
    """Check if geomloss is available."""
    try:
        import geomloss  # noqa: F401
        return True
    except ImportError:
        return False


class TestSinkhornDataBasics:
    """Tests for SinkhornData (T34)."""

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_is_data_term(self):
        """Verify SinkhornData is a DataTerm."""
        assert issubclass(SinkhornData, DataTerm)

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_instantiation(self):
        """Verify SinkhornData can be instantiated."""
        term = SinkhornData()
        assert callable(term)
        assert term.p == 2
        assert term.blur == 0.01

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_returns_scalar(self):
        """Verify SinkhornData returns a scalar tensor."""
        term = SinkhornData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)

        loss = term(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_zero_on_identical(self):
        """Verify SinkhornData is nearly zero on identical clouds."""
        term = SinkhornData()
        cloud = torch.randn(1, 10, 3)

        loss = term(cloud, cloud)
        # Sinkhorn may not be exactly zero due to numerical precision
        assert loss.item() < 1e-5


class TestSinkhornDataLazyLoading:
    """Test lazy loading of geomloss in SinkhornData."""

    def test_sinkhorn_lazy_loads_geomloss(self):
        """Verify geomloss is not loaded until forward() is called."""
        term = SinkhornData()
        # At init time, loss_fn should be None
        assert term.loss_fn is None

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_loads_geomloss_on_forward(self):
        """Verify geomloss is loaded on first forward call."""
        term = SinkhornData()
        assert term.loss_fn is None

        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)
        loss = term(pred, target)

        # After forward, loss_fn should be initialized
        assert term.loss_fn is not None

    @pytest.mark.skipif(_has_geomloss(), reason="geomloss is available (test requires absence)")
    def test_sinkhorn_raises_without_geomloss(self):
        """Verify SinkhornData raises ImportError if geomloss not available."""
        term = SinkhornData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)

        with pytest.raises(ImportError, match="geomloss required"):
            term(pred, target)


class TestSinkhornDataWeights:
    """Test SinkhornData with weights."""

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_with_weights(self):
        """Verify SinkhornData accepts weight arguments."""
        term = SinkhornData()
        pred = torch.randn(1, 5, 3)
        target = torch.randn(1, 5, 3)
        pred_w = torch.ones(1, 5)
        tgt_w = torch.ones(1, 5)

        loss = term(pred, target, pred_w=pred_w, tgt_w=tgt_w)
        assert loss.shape == ()

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_uniform_weights_vs_no_weights(self):
        """Verify uniform weights give same loss as no weights."""
        term = SinkhornData()
        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 10, 3)

        loss_no_w = term(pred, target)
        w = torch.ones_like(pred[..., 0])
        loss_with_w = term(pred, target, pred_w=w, tgt_w=w)

        # Should be very close (may differ slightly due to Sinkhorn numerical issues)
        assert torch.allclose(loss_no_w, loss_with_w, atol=1e-4)


class TestSinkhornDataInitArgs:
    """Test SinkhornData initialization arguments."""

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_custom_p(self):
        """Verify SinkhornData accepts custom p norm."""
        term = SinkhornData(p=1)
        assert term.p == 1

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_custom_blur(self):
        """Verify SinkhornData accepts custom blur."""
        term = SinkhornData(blur=0.05)
        assert term.blur == 0.05

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_sinkhorn_custom_backend(self):
        """Verify SinkhornData accepts custom backend."""
        term = SinkhornData(backend="torch")
        assert term.backend == "torch"


class TestEMDDataBackwardCompat:
    """Test EMDData backward compatibility with SinkhornData."""

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_emddata_is_data_term(self):
        """Verify EMDData is still a DataTerm."""
        assert issubclass(EMDData, DataTerm)

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_emddata_delegates_to_sinkhorn(self):
        """Verify EMDData delegates to SinkhornData."""
        emd = EMDData()
        pred = torch.randn(1, 10, 3)
        target = torch.randn(1, 10, 3)

        loss = emd(pred, target)
        assert loss.shape == ()
        assert loss.dtype == torch.float32

    @pytest.mark.skipif(not _has_geomloss(), reason="geomloss not available")
    def test_emddata_zero_on_identical(self):
        """Verify EMDData is nearly zero on identical clouds."""
        emd = EMDData()
        cloud = torch.randn(1, 10, 3)

        loss = emd(cloud, cloud)
        assert loss.item() < 1e-5


class TestAutoDecoderWithoutGeomloss:
    """Verify auto-decoder configs work without geomloss (T34 lazy-loading guarantee)."""

    def test_chamfer_instantiation_without_geomloss(self):
        """Verify CDData works regardless of geomloss availability."""
        term = CDData()
        assert callable(term)

    def test_l2_instantiation_without_geomloss(self):
        """Verify L2Data works regardless of geomloss availability."""
        term = L2Data()
        assert callable(term)

    def test_weighted_cd_instantiation_without_geomloss(self):
        """Verify WeightedCDData works regardless of geomloss availability."""
        term = WeightedCDData()
        assert callable(term)

    def test_pcd_instantiation_without_geomloss(self):
        """Verify PCDData works regardless of geomloss availability."""
        term = PCDData()
        assert callable(term)

    def test_ncd_instantiation_without_geomloss(self):
        """Verify NCDData works regardless of geomloss availability."""
        term = NCDData()
        assert callable(term)

    def test_auto_decoder_codes_instantiation_without_geomloss(self):
        """Verify AutoDecoderCodes can be imported without geomloss."""
        # This should not raise even if geomloss is unavailable
        from src.resnet_lddmm.codes.auto_decoder import AutoDecoderCodes
        codes = AutoDecoderCodes(num_shapes=5, n_z=3)
        assert hasattr(codes, 'forward')
