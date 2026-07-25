"""Tests for AutoDecoderCodes (STEPS T24)."""

import pytest
import torch
from types import SimpleNamespace

from src.resnet_lddmm.codes.auto_decoder import AutoDecoderCodes
from src.resnet_lddmm.codes.base import ShapeCode


class TestAutoDecoderBasics:
    """Basic functionality tests for AutoDecoderCodes."""

    def test_is_shape_code(self):
        """Verify AutoDecoderCodes is a ShapeCode."""
        assert issubclass(AutoDecoderCodes, ShapeCode)

    def test_instantiation(self):
        """Verify AutoDecoderCodes can be instantiated."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        assert isinstance(codes, AutoDecoderCodes)
        assert codes.num_shapes == 10
        assert codes.n_z == 256

    def test_default_regularization_weight(self):
        """Verify default regularization weight is 1.0."""
        codes = AutoDecoderCodes(num_shapes=10)
        assert codes.regularization_weight == 1.0

    def test_custom_regularization_weight(self):
        """Verify custom regularization weight is stored."""
        codes = AutoDecoderCodes(num_shapes=10, regularization_weight=0.5)
        assert codes.regularization_weight == 0.5


class TestAutoDecoderForward:
    """Tests for AutoDecoderCodes.forward()."""

    def test_forward_returns_tensor(self):
        """Verify forward returns a tensor."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1, 2]))

        result = codes(batch)

        assert isinstance(result, torch.Tensor)

    def test_forward_output_shape(self):
        """Verify output shape is [B, n_z]."""
        B, num_shapes, n_z = 4, 10, 256
        codes = AutoDecoderCodes(num_shapes=num_shapes, n_z=n_z)
        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1, 2, 3]))

        result = codes(batch)

        assert result.shape == (B, n_z)

    def test_forward_different_batch_sizes(self):
        """Test forward with different batch sizes."""
        num_shapes, n_z = 20, 128
        codes = AutoDecoderCodes(num_shapes=num_shapes, n_z=n_z)

        for B in [1, 2, 4, 8]:
            shape_ids = torch.randint(0, num_shapes, (B,))
            batch = SimpleNamespace(shape_ids=shape_ids)
            result = codes(batch)
            assert result.shape == (B, n_z)

    def test_forward_no_shape_ids(self):
        """Verify forward returns None when batch has no shape_ids."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace()  # No shape_ids attribute

        result = codes(batch)

        assert result is None

    def test_forward_none_shape_ids(self):
        """Verify forward returns None when shape_ids is None."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace(shape_ids=None)

        result = codes(batch)

        assert result is None

    def test_forward_same_ids_same_codes(self):
        """Verify same shape_id returns same code."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch1 = SimpleNamespace(shape_ids=torch.tensor([0, 1, 0]))
        batch2 = SimpleNamespace(shape_ids=torch.tensor([0, 0]))

        result1 = codes(batch1)
        result2 = codes(batch2)

        # result1[0] (shape_id 0) should equal result1[2] (shape_id 0)
        assert torch.allclose(result1[0], result1[2])
        # result1[0] should equal result2[0] (both shape_id 0)
        assert torch.allclose(result1[0], result2[0])

    def test_forward_different_ids_different_codes(self):
        """Verify different shape_ids usually produce different codes."""
        torch.manual_seed(42)
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1, 2]))

        result = codes(batch)

        # Different shape IDs should produce different codes (with high probability)
        assert not torch.allclose(result[0], result[1])
        assert not torch.allclose(result[1], result[2])


class TestAutoDecoderPenalty:
    """Tests for AutoDecoderCodes.penalty()."""

    def test_penalty_returns_tensor(self):
        """Verify penalty returns a scalar tensor."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        penalty = codes.penalty()

        assert isinstance(penalty, torch.Tensor)
        assert penalty.ndim == 0  # Scalar

    def test_penalty_is_positive(self):
        """Verify penalty is non-negative."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        penalty = codes.penalty()

        assert penalty.item() >= 0

    def test_penalty_zero_at_init_with_zero_init(self):
        """Verify penalty is near zero with zero initialization."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        nn_init = torch.nn.init.zeros_
        nn_init(codes.codes.weight)

        penalty = codes.penalty()

        assert penalty.item() < 1e-6

    def test_penalty_depends_on_weight(self):
        """Verify penalty increases with larger codes."""
        codes1 = AutoDecoderCodes(num_shapes=10, n_z=256)
        codes2 = AutoDecoderCodes(num_shapes=10, n_z=256)

        # Scale codes2 to be larger
        codes2.codes.weight.data.mul_(10.0)

        penalty1 = codes1.penalty()
        penalty2 = codes2.penalty()

        assert penalty2.item() > penalty1.item()

    def test_penalty_scales_with_regularization_weight(self):
        """Verify penalty scales with regularization_weight."""
        codes1 = AutoDecoderCodes(num_shapes=10, n_z=256, regularization_weight=1.0)
        codes2 = AutoDecoderCodes(num_shapes=10, n_z=256, regularization_weight=2.0)

        # Copy same weights
        codes2.codes.weight.data.copy_(codes1.codes.weight.data)

        penalty1 = codes1.penalty()
        penalty2 = codes2.penalty()

        assert abs(penalty2.item() - 2 * penalty1.item()) < 1e-5


class TestAutoDecoderGradients:
    """Tests for gradient flow through AutoDecoderCodes."""

    def test_gradient_flow_forward(self):
        """Verify gradients flow through forward pass."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1, 2]))

        result = codes(batch)
        loss = result.sum()
        loss.backward()

        # Codes embedding should have gradients
        assert codes.codes.weight.grad is not None
        assert codes.codes.weight.grad.abs().sum().item() > 0

    def test_gradient_flow_penalty(self):
        """Verify gradients flow through penalty."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)

        penalty = codes.penalty()
        penalty.backward()

        assert codes.codes.weight.grad is not None
        assert codes.codes.weight.grad.abs().sum().item() > 0

    def test_parameters_trainable(self):
        """Verify embedding parameters are trainable."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)

        params = list(codes.parameters())
        assert len(params) == 1
        assert params[0].requires_grad


class TestAutoDecoderStateDictProtocol:
    """Tests for state_dict protocol."""

    def test_state_dict_has_codes(self):
        """Verify state_dict contains codes."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        state = codes.state_dict()

        assert "codes.weight" in state

    def test_load_state_dict(self):
        """Verify load_state_dict works."""
        codes1 = AutoDecoderCodes(num_shapes=10, n_z=256)
        codes2 = AutoDecoderCodes(num_shapes=10, n_z=256)

        state = codes1.state_dict()
        codes2.load_state_dict(state)

        # Should have same codes
        assert torch.allclose(codes1.codes.weight, codes2.codes.weight)

    def test_state_dict_roundtrip(self):
        """Verify state_dict roundtrip preserves values."""
        codes1 = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1, 2]))

        result1 = codes1(batch)

        state = codes1.state_dict()
        codes2 = AutoDecoderCodes(num_shapes=10, n_z=256)
        codes2.load_state_dict(state)

        result2 = codes2(batch)

        assert torch.allclose(result1, result2)


class TestAutoDecoderDevice:
    """Tests for device movement."""

    def test_cpu_device(self):
        """Verify .cpu() works."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        codes.cpu()
        # Should not raise

    def test_device_movement(self):
        """Verify codes move with device."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        codes.cpu()

        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1]))
        result = codes(batch)

        assert result.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Verify .cuda() works if available."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        codes.cuda()

        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1]).cuda())
        result = codes(batch)

        assert result.device.type == "cuda"


class TestAutoDecoderIntegration:
    """Integration tests with typical usage patterns."""

    def test_multiple_shapes_multiple_ids(self):
        """Test with multiple shapes and repeated IDs."""
        torch.manual_seed(42)
        num_shapes, n_z = 50, 128
        codes = AutoDecoderCodes(num_shapes=num_shapes, n_z=n_z)

        # Batch with repeated shape IDs
        shape_ids = torch.tensor([0, 1, 2, 0, 1, 5, 10, 0])
        batch = SimpleNamespace(shape_ids=shape_ids)

        result = codes(batch)

        assert result.shape == (8, n_z)
        # Check that repeated IDs return identical codes
        assert torch.allclose(result[0], result[3])
        assert torch.allclose(result[0], result[7])
        assert torch.allclose(result[1], result[4])

    def test_forward_and_penalty_together(self):
        """Test typical training loop pattern."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        batch = SimpleNamespace(shape_ids=torch.tensor([0, 1, 2]))

        # Forward
        code_tensor = codes(batch)
        assert code_tensor is not None

        # Penalty
        penalty = codes.penalty()
        assert penalty.item() > 0

        # Typical loss: use codes + regularize
        loss = code_tensor.sum() + penalty
        loss.backward()

        assert codes.codes.weight.grad is not None

    def test_handles_2d_shape_ids(self):
        """Test that 2D shape_ids are flattened correctly."""
        codes = AutoDecoderCodes(num_shapes=10, n_z=256)
        shape_ids = torch.tensor([[0, 1], [2, 3]])
        batch = SimpleNamespace(shape_ids=shape_ids)

        result = codes(batch)

        assert result.shape == (4, 256)
