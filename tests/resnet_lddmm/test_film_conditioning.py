"""Tests for FiLMConditioning."""

import pytest
import torch
import torch.nn as nn

from src.resnet_lddmm.conditioning.film import FiLMConditioning
from src.resnet_lddmm.conditioning.factory import create_conditioning


class TestFiLMConditioning:
    """Tests for FiLMConditioning."""

    def test_output_shape(self):
        """Verify output shape is [B, N, output_dim]."""
        B, N, n_z, out_dim = 2, 10, 256, 3
        cond = FiLMConditioning(n_z=n_z, output_dim=out_dim)

        x = torch.randn(B, N, 3)
        code = torch.randn(B, n_z)

        out = cond(x, code)

        assert out.shape == (B, N, out_dim)

    def test_none_code_returns_none(self):
        """Verify forward returns None when code is None."""
        cond = FiLMConditioning(n_z=256)
        x = torch.randn(2, 10, 3)
        out = cond(x, None)
        assert out is None

    def test_dim_equals_output_dim(self):
        """Verify dim matches output_dim."""
        for out_dim in [3, 16, 32, 64]:
            cond = FiLMConditioning(n_z=256, output_dim=out_dim)
            assert cond.dim == out_dim

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        n_z, out_dim = 256, 3
        cond = FiLMConditioning(n_z=n_z, output_dim=out_dim)

        for B in [1, 2, 4, 8]:
            x = torch.randn(B, 10, 3)
            code = torch.randn(B, n_z)
            out = cond(x, code)
            assert out.shape == (B, 10, out_dim)

    def test_different_point_counts(self):
        """Test with different point counts."""
        B, n_z, out_dim = 2, 256, 3
        cond = FiLMConditioning(n_z=n_z, output_dim=out_dim)

        for N in [5, 10, 50, 100]:
            x = torch.randn(B, N, 3)
            code = torch.randn(B, n_z)
            out = cond(x, code)
            assert out.shape == (B, N, out_dim)

    def test_gradient_flow(self):
        """Verify gradients flow through conditioning."""
        cond = FiLMConditioning(n_z=256, output_dim=3)
        x = torch.randn(2, 10, 3, requires_grad=True)
        code = torch.randn(2, 256, requires_grad=True)

        out = cond(x, code)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert code.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(code.grad).all()

    def test_different_codes_produce_different_outputs(self):
        """Verify that different codes produce different outputs."""
        cond = FiLMConditioning(n_z=256)
        x = torch.randn(1, 10, 3)

        code1 = torch.randn(1, 256)
        code2 = torch.randn(1, 256)

        out1 = cond(x, code1)
        out2 = cond(x, code2)

        assert not torch.allclose(out1, out2)

    def test_parameters_are_trainable(self):
        """Verify gamma and beta networks have trainable parameters."""
        cond = FiLMConditioning(n_z=256)

        for name, param in cond.named_parameters():
            assert param.requires_grad, f"{name} is not trainable"

    def test_modulation_form(self):
        """Verify that output follows γ ⊙ x + β form."""
        torch.manual_seed(42)
        n_z, out_dim = 16, 3
        cond = FiLMConditioning(n_z=n_z, output_dim=out_dim)
        x = torch.randn(1, 5, 3)
        code = torch.randn(1, n_z)

        # Manually compute expected output
        code_exp = code.unsqueeze(1).expand(1, 5, -1).reshape(5, -1)
        gamma_expected = cond.gamma_net(code_exp).view(1, 5, out_dim)
        beta_expected = cond.beta_net(code_exp).view(1, 5, out_dim)
        expected = gamma_expected * x + beta_expected

        out = cond(x, code)

        assert torch.allclose(out, expected, atol=1e-6)


class TestConditioningFactory:
    """Tests for conditioning factory."""

    def test_factory_concat(self):
        """Test factory creates ConcatConditioning."""
        config = {"method": "concat", "n_z": 256}
        cond = create_conditioning(config)

        assert cond.dim == 256
        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)
        out = cond(x, code)
        assert out.shape == (2, 10, 256)

    def test_factory_film(self):
        """Test factory creates FiLMConditioning."""
        config = {"method": "film", "n_z": 256, "output_dim": 3}
        cond = create_conditioning(config)

        assert isinstance(cond, FiLMConditioning)
        assert cond.dim == 3
        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)
        out = cond(x, code)
        assert out.shape == (2, 10, 3)

    def test_factory_none(self):
        """Test factory creates NoConditioning."""
        config = {"method": "none"}
        cond = create_conditioning(config)

        assert cond.dim == 0
        x = torch.randn(2, 10, 3)
        out = cond(x, None)
        assert out is None

    def test_factory_none_config(self):
        """Test factory with None config."""
        cond = create_conditioning(None)

        assert cond.dim == 0
        x = torch.randn(2, 10, 3)
        out = cond(x, None)
        assert out is None

    def test_factory_empty_config(self):
        """Test factory with empty config."""
        cond = create_conditioning({})

        assert cond.dim == 0

    def test_factory_position_aware(self):
        """Test factory creates PositionAware."""
        config = {"method": "position_aware", "n_z": 256, "g": 2, "channels": 32}
        cond = create_conditioning(config)

        assert cond.dim == 32
        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)
        out = cond(x, code)
        assert out.shape == (2, 10, 32)

    def test_factory_invalid_method(self):
        """Test factory with invalid method."""
        config = {"method": "invalid_method"}
        with pytest.raises(ValueError):
            create_conditioning(config)
