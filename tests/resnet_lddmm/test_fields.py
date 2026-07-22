"""Tests for VelocityField and TimeVaryingField."""

import torch
import pytest
from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.fields.time_varying import TimeVaryingField
from src.resnet_lddmm.conditioning.base import NoConditioning


class TestVelocityFieldABC:
    """Tests for VelocityField ABC."""

    def test_velocity_field_is_abstract(self):
        """Verify VelocityField is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            VelocityField()

    def test_velocity_field_is_nn_module(self):
        """Verify VelocityField is an nn.Module."""
        import torch.nn as nn

        assert issubclass(VelocityField, nn.Module)


class TestTimeVaryingField:
    """Tests for TimeVaryingField."""

    def test_block_count(self):
        """Verify the field has the correct number of blocks."""
        num_blocks = 15
        field = TimeVaryingField(num_blocks=num_blocks, width=256)
        assert len(field.blocks) == num_blocks

    def test_default_num_blocks(self):
        """Verify default num_blocks is 10."""
        field = TimeVaryingField()
        assert len(field.blocks) == 10

    def test_output_shape(self):
        """Verify output shape is [B, N, 3]."""
        field = TimeVaryingField(num_blocks=10, width=256)
        x = torch.randn(4, 20, 3)  # [B=4, N=20, 3]
        output = field(x, step=0)
        assert output.shape == (4, 20, 3)

    def test_distinct_parameters_per_block(self):
        """Verify each block has distinct parameters (different objects)."""
        field = TimeVaryingField(num_blocks=5, width=64)
        # Get parameters of block 0 and block 1
        block0_params = list(field.blocks[0].parameters())
        block1_params = list(field.blocks[1].parameters())

        # Check that they are different objects
        for p0, p1 in zip(block0_params, block1_params):
            # Different object IDs means they are distinct parameter tensors
            assert id(p0) != id(p1)

    def test_step_selection(self):
        """Verify different steps can be selected and produce valid outputs."""
        torch.manual_seed(42)
        field = TimeVaryingField(num_blocks=5, width=64)
        x = torch.randn(2, 10, 3)

        # Forward pass through different steps should all work
        for step in range(5):
            output = field(x, step=step)
            assert output.shape == (2, 10, 3)
            # All outputs should be zero due to zero-init of proj.weight
            assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)

    def test_works_with_no_conditioning(self):
        """Verify field works with NoConditioning."""
        conditioning = NoConditioning()
        field = TimeVaryingField(num_blocks=5, width=64, conditioning=conditioning)

        x = torch.randn(2, 10, 3)
        code = torch.randn(2, 256)  # Code is ignored
        output = field(x, step=0, code=code)

        assert output.shape == (2, 10, 3)

    def test_default_conditioning(self):
        """Verify default conditioning is NoConditioning."""
        field = TimeVaryingField(num_blocks=5, width=64)
        assert isinstance(field.conditioning, NoConditioning)

    def test_input_dim_with_conditioning(self):
        """Verify input dimension accounts for conditioning."""
        # Create a mock conditioning with dim > 0
        class MockConditioning(NoConditioning):
            def __init__(self):
                super().__init__()
                self.dim = 8

        conditioning = MockConditioning()
        field = TimeVaryingField(
            num_blocks=3, width=64, conditioning=conditioning
        )

        # The input to blocks should be 3 + 8 = 11
        # Check that the first block's lift layer has the right input dimension
        assert field.blocks[0].lift.in_features == 3 + 8

    def test_different_activations(self):
        """Verify field accepts different activation functions."""
        for activation in ["relu", "leaky_relu"]:
            field = TimeVaryingField(
                num_blocks=5, width=64, activation=activation
            )
            x = torch.randn(2, 10, 3)
            output = field(x, step=0)
            assert output.shape == (2, 10, 3)

    def test_gradients_flow_through_field(self):
        """Verify gradients flow through the field."""
        field = TimeVaryingField(num_blocks=5, width=64)
        x = torch.randn(2, 10, 3, requires_grad=True)
        output = field(x, step=0)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check that block parameters have gradients
        for param in field.blocks[0].parameters():
            assert param.grad is not None

    def test_various_widths(self):
        """Verify field works with different widths."""
        for width in [32, 64, 128, 256, 512]:
            field = TimeVaryingField(num_blocks=3, width=width)
            x = torch.randn(1, 5, 3)
            output = field(x, step=0)
            assert output.shape == (1, 5, 3)

    def test_step_out_of_bounds(self):
        """Verify accessing invalid step raises an error."""
        field = TimeVaryingField(num_blocks=5, width=64)
        x = torch.randn(2, 10, 3)

        with pytest.raises(IndexError):
            field(x, step=10)  # num_blocks=5, so valid steps are 0-4

    def test_batch_independence(self):
        """Verify batch elements are processed independently."""
        torch.manual_seed(42)
        field = TimeVaryingField(num_blocks=5, width=64)

        x1 = torch.randn(1, 10, 3)
        x2 = torch.randn(1, 10, 3)
        x_batch = torch.cat([x1, x2], dim=0)  # [2, 10, 3]

        out1 = field(x1, step=0)
        out2 = field(x2, step=0)
        out_batch = field(x_batch, step=0)

        assert torch.allclose(out_batch[0], out1[0])
        assert torch.allclose(out_batch[1], out2[0])
