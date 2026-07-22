"""Tests for ShapeCode implementations."""

import torch
import pytest
from src.resnet_lddmm.codes.base import ShapeCode
from src.resnet_lddmm.codes.none import NoCode


class TestShapeCodeABC:
    """Tests for ShapeCode abstract base class."""

    def test_shape_code_is_abstract(self):
        """Verify ShapeCode cannot be instantiated."""
        with pytest.raises(TypeError):
            ShapeCode()


class TestNoCodeBasics:
    """Basic tests for NoCode."""

    def test_no_code_is_shape_code(self):
        """Verify NoCode is a ShapeCode."""
        assert issubclass(NoCode, ShapeCode)

    def test_no_code_is_nn_module(self):
        """Verify NoCode is an nn.Module."""
        code = NoCode()
        assert isinstance(code, torch.nn.Module)


class TestNoCodeForward:
    """Tests for NoCode.forward()."""

    def test_forward_returns_none(self):
        """Verify forward() returns None."""
        code = NoCode()
        batch = None
        result = code.forward(batch)
        assert result is None

    def test_forward_ignores_batch(self):
        """Verify forward() ignores any batch argument."""
        code = NoCode()

        for batch in [None, {}, "ignored", torch.randn(2, 10, 3), 42]:
            result = code.forward(batch)
            assert result is None

    def test_call_operator_returns_none(self):
        """Verify calling code() returns None (nn.Module forward)."""
        code = NoCode()
        result = code(None)
        assert result is None


class TestNoCodePenalty:
    """Tests for NoCode.penalty()."""

    def test_penalty_returns_none(self):
        """Verify penalty() returns None."""
        code = NoCode()
        result = code.penalty()
        assert result is None

    def test_penalty_is_callable(self):
        """Verify penalty() can be called multiple times."""
        code = NoCode()
        for _ in range(5):
            result = code.penalty()
            assert result is None


class TestNoCodeParameters:
    """Tests verifying NoCode has no learnable parameters."""

    def test_no_parameters(self):
        """Verify NoCode has no parameters."""
        code = NoCode()
        params = list(code.parameters())
        assert len(params) == 0

    def test_no_buffers(self):
        """Verify NoCode has no registered buffers."""
        code = NoCode()
        buffers = list(code.buffers())
        assert len(buffers) == 0

    def test_no_named_parameters(self):
        """Verify no named parameters exist."""
        code = NoCode()
        named_params = list(code.named_parameters())
        assert len(named_params) == 0


class TestNoCodeGradientBehavior:
    """Tests for gradient flow with NoCode."""

    def test_no_gradients_accumulate(self):
        """Verify no gradients can accumulate (no parameters)."""
        code = NoCode()
        # Try backward on a tensor that used code
        x = torch.randn(2, 5, requires_grad=True)
        result = code(None)
        # result is None, so can't backprop through it
        assert result is None


class TestNoCodeTrainEval:
    """Tests for train/eval mode switching."""

    def test_train_mode(self):
        """Verify train() works."""
        code = NoCode()
        code.train()
        assert code.training

    def test_eval_mode(self):
        """Verify eval() works."""
        code = NoCode()
        code.eval()
        assert not code.training


class TestNoCodeStateDictProtocol:
    """Tests for state_dict protocol (should be empty)."""

    def test_state_dict_empty(self):
        """Verify state_dict() is empty."""
        code = NoCode()
        state = code.state_dict()
        assert len(state) == 0

    def test_load_state_dict_empty(self):
        """Verify load_state_dict() accepts empty dict."""
        code = NoCode()
        code.load_state_dict({})
        # Should not raise

    def test_state_dict_roundtrip(self):
        """Verify state_dict roundtrip on another NoCode instance."""
        code1 = NoCode()
        state = code1.state_dict()

        code2 = NoCode()
        code2.load_state_dict(state)

        assert list(code2.state_dict().keys()) == list(code1.state_dict().keys())


class TestNoCodeDevice:
    """Tests for device movement (should be no-op)."""

    def test_cpu_device(self):
        """Verify .cpu() works."""
        code = NoCode()
        code.cpu()
        # No parameters, so no-op but should not raise

    def test_device_movement_no_op(self):
        """Verify device movement has no effect (no parameters)."""
        code = NoCode()
        code1_state = str(code.state_dict())

        code.cuda() if torch.cuda.is_available() else code.cpu()
        code2_state = str(code.state_dict())

        # State unchanged since no parameters
        assert code1_state == code2_state


class TestNoCodeDocstring:
    """Verify interface documentation."""

    def test_forward_has_docstring(self):
        """Verify forward() is documented."""
        assert NoCode.forward.__doc__ is not None

    def test_penalty_has_docstring(self):
        """Verify penalty() is documented."""
        assert NoCode.penalty.__doc__ is not None
