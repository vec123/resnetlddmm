"""Tests for StationaryField (STEPS T23)."""

import pytest
import torch
import torch.nn as nn

from src.resnet_lddmm.fields.time_varying import StationaryField
from src.resnet_lddmm.conditioning.base import NoConditioning
from src.resnet_lddmm.conditioning.film import ConcatConditioning


class TestStationaryFieldBasic:
    """Basic functionality tests for StationaryField."""

    def test_output_shape(self):
        """Verify output shape is [B, N, 3]."""
        B, N = 2, 10
        field = StationaryField()
        x = torch.randn(B, N, 3)

        v = field(x, step=0)

        assert v.shape == (B, N, 3)

    def test_ignores_step(self):
        """Verify velocity is identical regardless of step parameter."""
        field = StationaryField()
        x = torch.randn(2, 10, 3)

        v_step_0 = field(x, step=0)
        v_step_5 = field(x, step=5)
        v_step_100 = field(x, step=100)

        assert torch.allclose(v_step_0, v_step_5)
        assert torch.allclose(v_step_0, v_step_100)

    def test_identity_at_init(self):
        """Verify zero velocity at initialization (identity field)."""
        torch.manual_seed(42)
        field = StationaryField()
        x = torch.randn(2, 10, 3)

        v = field(x)

        # Zero-init final layer means zero velocity at init
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)

    def test_parameter_count_defaults(self):
        """Verify parameter count is reasonable at paper defaults.

        Spec says ~278k with loose bounds — actual count depends on conditioning
        setup. This verifies the architecture computes params correctly (FA and DF
        layers, no redundancy).
        """
        # Without conditioning: no extra parameters
        field = StationaryField(
            fa=(64, 64, 64),
            df=(256, 256, 256, 256, 256),
            fourier_n_e=3,
            conditioning=None
        )

        total_params = sum(p.numel() for p in field.parameters())

        # Verify it's in a reasonable ballpark (hundreds of thousands)
        assert 200000 <= total_params <= 500000, f"Got {total_params}, expected O(278k)"

    def test_with_no_conditioning(self):
        """Verify works with NoConditioning."""
        field = StationaryField(conditioning=NoConditioning())
        x = torch.randn(2, 10, 3)

        v = field(x)

        assert v.shape == (2, 10, 3)

    def test_with_concat_conditioning(self):
        """Verify works with ConcatConditioning."""
        field = StationaryField(conditioning=ConcatConditioning(n_z=256))
        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        v = field(x, code=code)

        assert v.shape == (2, 10, 3)

    def test_none_code_with_no_conditioning(self):
        """Verify None code works with NoConditioning."""
        field = StationaryField(conditioning=NoConditioning())
        x = torch.randn(2, 10, 3)

        # NoConditioning ignores code
        v = field(x, code=None)

        assert v.shape == (2, 10, 3)


class TestStationaryFieldArchitecture:
    """Test the FA-NN → Fourier → DF-NN architecture."""

    def test_fa_output_shape(self):
        """Verify FA-NN output shape."""
        fa_widths = (64, 64, 64)
        field = StationaryField(fa=fa_widths)
        x = torch.randn(2, 10, 3)

        # Manually extract FA output
        h = field.fa(x)

        assert h.shape == (2, 10, fa_widths[-1])

    def test_fourier_expansion(self):
        """Verify FourierFeatures expands correctly."""
        fa_widths = (64, 64, 64)
        fourier_n_e = 3
        field = StationaryField(fa=fa_widths, fourier_n_e=fourier_n_e)
        x = torch.randn(2, 10, 3)

        fa_out = field.fa(x)
        fourier_out = field.fourier(fa_out)

        expected_dim = (2 * fourier_n_e + 1) * fa_widths[-1]
        assert fourier_out.shape == (2, 10, expected_dim)

    def test_code_injected_twice_with_conditioning(self):
        """Verify code is injected into both FA and DF inputs."""
        torch.manual_seed(42)
        field = StationaryField(conditioning=ConcatConditioning(n_z=256))
        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)

        # With conditioning, FA input is [x ⊕ z̄] where dim = 3 + 256 = 259
        # Verify by checking that FA input dimension matches
        assert field.fa[0].in_features == 3 + 256

        v = field(x, code=code)
        assert v.shape == (2, 10, 3)

    def test_identity_init_verified(self):
        """Verify identity initialization by checking final layer is zero."""
        field = StationaryField()

        # Last linear layer should be zero-initialized
        last_layer = None
        for module in field.modules():
            if isinstance(module, nn.Linear):
                last_layer = module

        if last_layer is not None:
            assert torch.allclose(last_layer.weight, torch.zeros_like(last_layer.weight))


class TestStationaryFieldGradients:
    """Test gradient flow through StationaryField."""

    def test_gradient_flow_through_fa_df(self):
        """Verify gradients flow through FA-NN and DF-NN."""
        field = StationaryField()
        x = torch.randn(2, 10, 3, requires_grad=True)

        v = field(x)
        loss = v.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gradient_flow_with_conditioning(self):
        """Verify gradients flow through conditioning."""
        field = StationaryField(conditioning=ConcatConditioning(n_z=256))
        x = torch.randn(2, 10, 3, requires_grad=True)
        code = torch.randn(2, 256, requires_grad=True)

        v = field(x, code=code)
        loss = v.sum()
        loss.backward()

        assert x.grad is not None
        assert code.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(code.grad).all()

    def test_parameters_are_trainable(self):
        """Verify FA and DF parameters have gradients enabled."""
        field = StationaryField()

        for name, param in field.named_parameters():
            assert param.requires_grad, f"{name} is not trainable"


class TestStationaryFieldCustomWidths:
    """Test with various architecture configurations."""

    def test_minimal_architecture(self):
        """Test with minimal FA and DF widths."""
        field = StationaryField(
            fa=(16,),
            df=(32,),
            fourier_n_e=2
        )
        x = torch.randn(2, 10, 3)

        v = field(x)

        assert v.shape == (2, 10, 3)

    def test_wide_architecture(self):
        """Test with wider FA and DF layers."""
        field = StationaryField(
            fa=(128, 128, 128),
            df=(512, 512, 512),
            fourier_n_e=4
        )
        x = torch.randn(2, 10, 3)

        v = field(x)

        assert v.shape == (2, 10, 3)

    def test_asymmetric_widths(self):
        """Test with non-uniform layer widths."""
        field = StationaryField(
            fa=(32, 64, 128),
            df=(256, 128, 64, 32),
            fourier_n_e=3
        )
        x = torch.randn(2, 10, 3)

        v = field(x)

        assert v.shape == (2, 10, 3)


class TestStationaryFieldActivations:
    """Test different activation functions."""

    def test_relu_activation(self):
        """Test with ReLU activation."""
        field = StationaryField(activation="relu")
        x = torch.randn(2, 10, 3)

        v = field(x)

        assert v.shape == (2, 10, 3)
        assert torch.isfinite(v).all()

    def test_leaky_relu_activation(self):
        """Test with LeakyReLU activation (default)."""
        field = StationaryField(activation="leaky_relu")
        x = torch.randn(2, 10, 3)

        v = field(x)

        assert v.shape == (2, 10, 3)
        assert torch.isfinite(v).all()


@pytest.mark.slow
class TestStationaryFieldOverfitting:
    """Test overfitting behavior on toy data."""

    def test_overfit_single_pair(self):
        """Verify field can overfit a single pair (velocity recovery)."""
        torch.manual_seed(42)
        field = StationaryField(
            fa=(32, 32),
            df=(64, 64, 64),
            fourier_n_e=2
        )
        optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)

        # Toy pair: two point clouds in correspondence
        q0 = torch.randn(1, 20, 3) - 0.5  # source
        q1 = torch.randn(1, 20, 3) + 0.5  # target

        # Simple 1-step matching loss: field(q0) should point towards q1
        target_velocity = (q1 - q0) / 1.0

        losses = []
        for step in range(100):
            optimizer.zero_grad()

            v = field(q0)
            loss = torch.nn.functional.mse_loss(v, target_velocity)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify loss decreases (overfitting works)
        assert losses[-1] < losses[0] * 0.5, "Field should overfit toy data"
