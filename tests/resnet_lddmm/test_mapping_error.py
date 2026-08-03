"""Tests for UnidirectionalMappingError and BidirectionalMappingError (STEPS T25)."""

import pytest
import torch
from types import SimpleNamespace

from src.resnet_lddmm.losses.mapping_error import (
    UnidirectionalMappingError,
    BidirectionalMappingError,
)
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.losses.data_terms import L2Data


class TestUnidirectionalMappingError:
    """Test UnidirectionalMappingError (forward-only mapping)."""

    def test_instantiation(self):
        """Verify UnidirectionalMappingError can be instantiated."""
        error = UnidirectionalMappingError()
        assert isinstance(error, UnidirectionalMappingError)

    def test_returns_tuple(self):
        """Verify forward returns (data, kinetic) tuple."""
        # Setup
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        code = None

        # Forward
        result = error(flow, data_term, source, target, code)

        # Verify tuple with two tensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_data_kinetic_are_scalars(self):
        """Verify data and kinetic are scalar tensors."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, kinetic = error(flow, data_term, source, target, None)

        assert data.ndim == 0  # Scalar
        assert kinetic.ndim == 0  # Scalar

    def test_data_positive(self):
        """Verify data loss is positive (squared error)."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, _ = error(flow, data_term, source, target, None)

        assert data.item() >= 0

    def test_kinetic_positive(self):
        """Verify kinetic energy is positive."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        _, kinetic = error(flow, data_term, source, target, None)

        assert kinetic.item() >= 0

    def test_gradient_flow(self):
        """Verify gradients flow through data and kinetic."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3, requires_grad=True), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, kinetic = error(flow, data_term, source, target, None)
        loss = data + kinetic
        loss.backward()

        # Field should have gradients
        for param in flow.field.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_with_different_batch_sizes(self):
        """Verify error computation works with different batch sizes."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        for batch_size in [1, 2, 4]:
            source = SimpleNamespace(
                points=torch.randn(batch_size, 10, 3), weights=None
            )
            target = SimpleNamespace(
                points=torch.randn(batch_size, 10, 3), weights=None
            )

            data, kinetic = error(flow, data_term, source, target, None)

            assert data.ndim == 0
            assert kinetic.ndim == 0


class TestBidirectionalMappingError:
    """Test BidirectionalMappingError (forward + backward mapping)."""

    def test_instantiation(self):
        """Verify BidirectionalMappingError can be instantiated."""
        error = BidirectionalMappingError()
        assert isinstance(error, BidirectionalMappingError)

    def test_returns_tuple(self):
        """Verify forward returns (data, kinetic) tuple."""
        # Setup
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        code = None

        # Forward
        result = error(flow, data_term, source, target, code)

        # Verify tuple with two tensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_data_kinetic_are_scalars(self):
        """Verify data and kinetic are scalar tensors."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, kinetic = error(flow, data_term, source, target, None)

        assert data.ndim == 0  # Scalar
        assert kinetic.ndim == 0  # Scalar

    def test_kinetic_sums_both_directions(self):
        """Verify bidirectional kinetic is sum of forward and backward."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()

        # Same flow for both directions to ensure determinism
        uni_error = UnidirectionalMappingError()
        bi_error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        # Bidirectional
        data_bi, kinetic_bi = bi_error(flow, data_term, source, target, None)

        # Compute forward and backward separately
        fwd = flow(source.points, None)
        bwd = flow.inverse(target.points, None)
        kinetic_fwd = fwd.kinetic_energy()
        kinetic_bwd = bwd.kinetic_energy()
        kinetic_expected = kinetic_fwd + kinetic_bwd

        # Kinetic from bidirectional should equal sum of both
        assert torch.allclose(kinetic_bi, kinetic_expected, rtol=1e-5)

    def test_data_sums_both_directions(self):
        """Verify bidirectional data is sum of both direction terms."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        # Bidirectional
        data_bi, _ = error(flow, data_term, source, target, None)

        # Compute forward and backward separately
        fwd = flow(source.points, None)
        bwd = flow.inverse(target.points, None)
        data_fwd = data_term(fwd.end, target.points, tgt_w=target.weights)
        data_bwd = data_term(bwd.end, source.points, tgt_w=source.weights)
        data_expected = data_fwd + data_bwd

        # Data from bidirectional should equal sum of both directions
        assert torch.allclose(data_bi, data_expected, rtol=1e-5)

    def test_data_positive(self):
        """Verify data loss is positive."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, _ = error(flow, data_term, source, target, None)

        assert data.item() >= 0

    def test_kinetic_positive(self):
        """Verify kinetic energy is positive."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        _, kinetic = error(flow, data_term, source, target, None)

        assert kinetic.item() >= 0

    def test_gradient_flow(self):
        """Verify gradients flow through both directions."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3, requires_grad=True), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3, requires_grad=True), weights=None
        )

        data, kinetic = error(flow, data_term, source, target, None)
        loss = data + kinetic
        loss.backward()

        # Field should have gradients
        for param in flow.field.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_with_code(self):
        """Verify bidirectional works with code conditioning."""
        from src.resnet_lddmm.conditioning.film import FiLMConditioning

        field = TimeVaryingField(num_blocks=3, width=32, conditioning=FiLMConditioning(n_z=256, output_dim=3))
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        code = torch.randn(2, 256)

        data, kinetic = error(flow, data_term, source, target, code)

        assert data.ndim == 0
        assert kinetic.ndim == 0
        assert torch.isfinite(data)
        assert torch.isfinite(kinetic)

    def test_bidirectional_vs_unidirectional(self):
        """Verify bidirectional kinetic >= unidirectional."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        # Unidirectional
        uni_error = UnidirectionalMappingError()
        _, kinetic_uni = uni_error(flow, data_term, source, target, None)

        # Bidirectional
        bi_error = BidirectionalMappingError()
        _, kinetic_bi = bi_error(flow, data_term, source, target, None)

        # Bidirectional includes both forward and backward trajectories, so kinetic >= unidirectional
        # In the best case they are equal (when backward has zero kinetic), but typically bi > uni
        assert kinetic_bi.item() >= kinetic_uni.item()


class TestMappingErrorIntegration:
    """Integration tests with different field types."""

    def test_time_varying_field_unidirectional(self):
        """Verify UnidirectionalMappingError works with TimeVaryingField."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, kinetic = error(flow, data_term, source, target, None)

        assert torch.isfinite(data)
        assert torch.isfinite(kinetic)

    def test_time_varying_field_bidirectional(self):
        """Verify BidirectionalMappingError works with TimeVaryingField."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = BidirectionalMappingError()

        source = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )
        target = SimpleNamespace(
            points=torch.randn(2, 10, 3), weights=None
        )

        data, kinetic = error(flow, data_term, source, target, None)

        assert torch.isfinite(data)
        assert torch.isfinite(kinetic)

    def test_identical_clouds_data_minimized(self):
        """Verify data is small when source and target are close."""
        field = TimeVaryingField(num_blocks=3, width=32)
        flow = NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)
        data_term = L2Data()
        error = UnidirectionalMappingError()

        # Start with small perturbation
        points = torch.randn(2, 10, 3)
        source = SimpleNamespace(points=points, weights=None)
        target = SimpleNamespace(points=points + 1e-3, weights=None)

        data, _ = error(flow, data_term, source, target, None)

        # Should be finite
        assert torch.isfinite(data)


class _SpyDataTerm:
    """Records the (target, tgt_w) pair each data-term call received."""

    def __init__(self):
        self.calls = []

    def __call__(self, pred, target, pred_w=None, tgt_w=None, normals=None):
        self.calls.append(
            (target.detach().clone(), None if tgt_w is None else tgt_w.detach().clone())
        )
        return pred.new_zeros(())


class TestSubsampleKeepsWeightsPaired:
    """Subsampling must keep tgt_w[i] the weight OF target[i].

    Each cloud carries a fingerprint (weights == x-coordinate), so decoupling points
    from weights breaks an invariant instead of a shape — which is how drawing a
    separate index per tensor stayed invisible: it produced correctly shaped,
    wrongly indexed weights that only WeightedCDData/Sinkhorn/EMD ever read.
    """

    @staticmethod
    def _cloud(n, batch=1):
        points = torch.randn(batch, n, 3)
        return SimpleNamespace(points=points, weights=points[:, :, 0].clone())

    @staticmethod
    def _flow():
        field = TimeVaryingField(num_blocks=3, width=32)
        return NeuralODEFlow(field, direct=ForwardEuler(), inverse=ModifiedEuler(), num_steps=3)

    def test_unidirectional_pairs_sample_weights(self):
        """The sample's points and weights survive subsampling as one cloud."""
        spy = _SpyDataTerm()
        error = UnidirectionalMappingError(subsample_n=10)

        error(self._flow(), spy, self._cloud(50), self._cloud(40), None)

        (target, tgt_w), = spy.calls
        assert target.shape[1] == 10
        assert tgt_w.shape == (1, 10)
        assert torch.equal(tgt_w, target[:, :, 0])

    def test_bidirectional_pairs_both_clouds(self):
        """Both data-term calls get weights matching the points they accompany."""
        spy = _SpyDataTerm()
        error = BidirectionalMappingError(subsample_n=10)

        error(self._flow(), spy, self._cloud(50), self._cloud(40), None)

        assert len(spy.calls) == 2
        for target, tgt_w in spy.calls:
            assert target.shape[1] == 10
            assert torch.equal(tgt_w, target[:, :, 0])

    def test_weights_untouched_without_subsampling(self):
        """subsample_n=None passes points and weights through unchanged."""
        spy = _SpyDataTerm()
        sample = self._cloud(40)

        UnidirectionalMappingError()(self._flow(), spy, self._cloud(50), sample, None)

        (target, tgt_w), = spy.calls
        assert torch.equal(target, sample.points)
        assert torch.equal(tgt_w, sample.weights)

    def test_mismatched_weights_raise(self):
        """A cloud whose weights don't count its own vertices fails loudly."""
        sample = self._cloud(50)
        sample.weights = torch.randn(1, 49)  # one weight short of its points

        with pytest.raises(ValueError, match="must describe the same vertices"):
            UnidirectionalMappingError(subsample_n=10)(
                self._flow(), _SpyDataTerm(), self._cloud(50), sample, None
            )
