"""Tests for VelocityBlock and FourierFeatures."""

import torch
import pytest
from src.resnet_lddmm.fields.blocks import VelocityBlock, FourierFeatures


class TestVelocityBlock:
    """Tests for VelocityBlock."""

    def test_output_dim(self):
        """Verify output is always [B, N, 3]."""
        block = VelocityBlock(in_dim=10, width=64)
        h = torch.randn(2, 20, 10)  # [B=2, N=20, in_dim=10]
        output = block(h)
        assert output.shape == (2, 20, 3)

    def test_proj_no_bias(self):
        """Verify proj layer has no bias."""
        block = VelocityBlock(in_dim=10, width=64)
        assert block.proj.bias is None

    def test_zero_init_output_is_zero(self):
        """Verify zero-initialized block outputs zero velocity."""
        block = VelocityBlock(in_dim=5, width=32)
        # After zero-init of proj.weight, the output should be ≡ 0
        h = torch.randn(4, 15, 5)
        output = block(h)
        # Allow small floating point tolerance
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_relu_activation(self):
        """Verify ReLU activation works."""
        block = VelocityBlock(in_dim=5, width=32, activation="relu")
        h = torch.randn(2, 10, 5)
        output = block(h)
        assert output.shape == (2, 10, 3)

    def test_leaky_relu_activation(self):
        """Verify LeakyReLU activation works."""
        block = VelocityBlock(in_dim=5, width=32, activation="leaky_relu")
        h = torch.randn(2, 10, 5)
        output = block(h)
        assert output.shape == (2, 10, 3)

    def test_gradients_flow(self):
        """Verify gradients flow through the block."""
        block = VelocityBlock(in_dim=5, width=32)
        h = torch.randn(2, 10, 5, requires_grad=True)
        output = block(h)
        loss = output.sum()
        loss.backward()
        assert h.grad is not None
        assert block.lift.weight.grad is not None
        assert block.mix.weight.grad is not None
        assert block.proj.weight.grad is not None


class TestFourierFeatures:
    """Tests for FourierFeatures."""

    def test_output_dim_formula(self):
        """Verify output dim = (2*n_e + 1)*d."""
        n_e = 3
        d = 64
        fourier = FourierFeatures(n_e=n_e)
        h = torch.randn(2, 10, d)  # [..., d]
        output = fourier(h)
        expected_dim = (2 * n_e + 1) * d
        assert output.shape[-1] == expected_dim
        assert output.shape[:-1] == h.shape[:-1]

    def test_various_n_e(self):
        """Test with different n_e values."""
        for n_e in [1, 2, 3, 4, 5]:
            fourier = FourierFeatures(n_e=n_e)
            h = torch.randn(1, 5, 32)
            output = fourier(h)
            expected_dim = (2 * n_e + 1) * 32
            assert output.shape[-1] == expected_dim

    def test_identity_passthrough(self):
        """Verify identity passthrough is included in output."""
        fourier = FourierFeatures(n_e=3)
        h = torch.randn(2, 10, 5)
        output = fourier(h)
        # First d elements should be the original h
        d = h.shape[-1]
        output_identity = output[..., :d]
        assert torch.allclose(output_identity, h)

    def test_matches_manual_computation(self):
        """Verify Fourier output matches manual sin/cos computation."""
        n_e = 3
        d = 4
        fourier = FourierFeatures(n_e=n_e)
        h = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]

        output = fourier(h)

        # Manual computation
        freqs = torch.pi * 2.0 ** torch.arange(n_e)
        angles = h.unsqueeze(-1) * freqs  # [1, 4, 3]
        sin_parts = angles.sin().flatten(-2)  # [1, 12]
        cos_parts = angles.cos().flatten(-2)  # [1, 12]
        expected = torch.cat([h, sin_parts, cos_parts], dim=-1)  # [1, 28]

        assert torch.allclose(output, expected, atol=1e-6)

    def test_batch_independence(self):
        """Verify Fourier features are computed independently per batch element."""
        fourier = FourierFeatures(n_e=2)
        h1 = torch.randn(1, 3, 5)
        h2 = torch.randn(1, 3, 5)
        h_batch = torch.cat([h1, h2], dim=0)  # [2, 3, 5]

        out1 = fourier(h1)
        out2 = fourier(h2)
        out_batch = fourier(h_batch)

        assert torch.allclose(out_batch[0], out1[0])
        assert torch.allclose(out_batch[1], out2[0])

    def test_differentiability(self):
        """Verify Fourier features are differentiable."""
        fourier = FourierFeatures(n_e=3)
        h = torch.randn(2, 5, 8, requires_grad=True)
        output = fourier(h)
        loss = output.sum()
        loss.backward()
        assert h.grad is not None
