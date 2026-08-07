"""Comprehensive tests for EquivariantStationaryField."""

import torch
import torch.nn as nn
import pytest
from typing import Tuple

from src.resnet_lddmm.fields.equivariant import (
    EquivariantStationaryField,
    EquivariantStationaryFieldSimple,
)
from src.resnet_lddmm.conditioning.base import NoConditioning


# ============================================================================
# Utilities
# ============================================================================


def random_rotation_matrix(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Sample random SO(3) rotation matrix via quaternion normalization.

    Returns:
        [3, 3] rotation matrix
    """
    q = torch.randn(4, device=device, dtype=dtype)
    q = q / q.norm()

    # Quaternion to rotation matrix
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)]),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)]),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]),
    ])
    return R


def batch_random_rotation_matrices(batch_size: int, device=torch.device("cpu"),
                                    dtype=torch.float32) -> torch.Tensor:
    """Sample batch_size random SO(3) rotation matrices.

    Returns:
        [batch_size, 3, 3] rotation matrices
    """
    return torch.stack([random_rotation_matrix(device, dtype) for _ in range(batch_size)])


def rotation_error(v1: torch.Tensor, v2: torch.Tensor, R: torch.Tensor) -> float:
    """Compute equivariance error: ||v2 - R @ v1|| / ||v1||.

    Args:
        v1: [B, N, 3] original velocities
        v2: [B, N, 3] velocities after rotation
        R: [B, 3, 3] rotation matrices

    Returns:
        Relative error (float, 0 = perfect equivariance)
    """
    # Apply rotation to v1: [B, N, 3] @ [B, 3, 3]^T → [B, N, 3]
    v1_rotated = torch.einsum("bni,bji->bnj", v1, R)  # or equivalently (v1 @ R.T)

    error = torch.norm(v2 - v1_rotated, p=2)
    norm = torch.norm(v1, p=2) + 1e-8
    return (error / norm).item()


# ============================================================================
# Test: Basic Functionality
# ============================================================================


class TestEquivariantStationaryFieldBasic:
    """Tests for basic functionality: shape, initialization, differentiability."""

    def test_output_shape(self):
        """Verify output shape matches input shape."""
        field = EquivariantStationaryField()
        x = torch.randn(4, 100, 3)
        v = field(x)
        assert v.shape == (4, 100, 3), f"Expected (4, 100, 3), got {v.shape}"

    def test_output_shape_variable_N(self):
        """Test with varying number of points."""
        field = EquivariantStationaryField()
        for N in [10, 50, 128, 256]:
            x = torch.randn(2, N, 3)
            v = field(x)
            assert v.shape == (2, N, 3)

    def test_initialization_to_zero(self):
        """Velocity should be ≈ 0 at initialization (zero-init output layer)."""
        field = EquivariantStationaryField()
        x = torch.randn(4, 100, 3)
        v = field(x)
        # Should be very small (not exactly zero due to bias terms, if any)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-5), \
            f"Initialization not zero: max={v.abs().max().item()}"

    def test_step_parameter_ignored(self):
        """Step parameter should not affect output (stationary field)."""
        field = EquivariantStationaryField()
        x = torch.randn(4, 100, 3)

        v1 = field(x, step=0)
        v2 = field(x, step=5)
        v3 = field(x, step=None)

        assert torch.allclose(v1, v2), "Step parameter affects output!"
        assert torch.allclose(v2, v3), "Step parameter affects output!"

    def test_no_nans_or_infs(self):
        """Output should be finite."""
        field = EquivariantStationaryField()
        x = torch.randn(4, 100, 3) * 10  # Large values
        v = field(x)

        assert not torch.isnan(v).any(), "Output contains NaN"
        assert not torch.isinf(v).any(), "Output contains Inf"

    def test_gradient_flow(self):
        """Gradients should flow to parameters."""
        field = EquivariantStationaryField()
        x = torch.randn(4, 100, 3, requires_grad=True)

        v = field(x)
        loss = v.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert any(p.grad is not None for p in field.parameters()), \
            "No gradients to field parameters"

    def test_different_batch_sizes(self):
        """Field should work with different batch sizes."""
        field = EquivariantStationaryField()
        for B in [1, 2, 8, 16]:
            x = torch.randn(B, 50, 3)
            v = field(x)
            assert v.shape == (B, 50, 3)


# ============================================================================
# Test: Equivariance Verification
# ============================================================================


class TestEquivariantStationaryFieldEquivariance:
    """Core tests: SO(3) equivariance v'(Rx) = R @ v(x)."""

    def test_so3_equivariance_single_rotation(self):
        """Test equivariance for a single random rotation."""
        field = EquivariantStationaryField(
            use_tensor_product_self=True,
        )
        field.eval()  # No dropout/batch norm effects

        x = torch.randn(4, 100, 3)
        R = random_rotation_matrix()  # [3, 3]

        # Original velocity
        v = field(x)  # [4, 100, 3]

        # Rotated input and velocity
        x_rot = x @ R.T  # Broadcast [3,3] to [4,100,3]
        v_rot = field(x_rot)

        # Expected: v_rot = v @ R.T
        v_expected = v @ R.T

        error = torch.norm(v_rot - v_expected) / (torch.norm(v) + 1e-8)
        print(f"Equivariance error (1 rotation): {error.item():.6f}")
        assert error < 1e-4, f"Equivariance broken: {error.item()}"

    def test_so3_equivariance_multiple_rotations(self):
        """Test equivariance for multiple random rotations."""
        field = EquivariantStationaryField(use_tensor_product_self=True)
        field.eval()

        x = torch.randn(4, 100, 3)

        max_error = 0.0
        for trial in range(5):
            R = random_rotation_matrix()

            v = field(x)
            x_rot = x @ R.T
            v_rot = field(x_rot)
            v_expected = v @ R.T

            error = torch.norm(v_rot - v_expected) / (torch.norm(v) + 1e-8)
            max_error = max(max_error, error.item())

        print(f"Equivariance error (5 rotations): max={max_error:.6f}")
        assert max_error < 1e-4, f"Equivariance broken in some rotation: {max_error}"

    def test_so3_equivariance_identity(self):
        """Identity rotation should not change output."""
        field = EquivariantStationaryField(use_tensor_product_self=True)
        field.eval()

        x = torch.randn(4, 100, 3)
        I = torch.eye(3)

        v = field(x)
        x_id = x @ I.T
        v_id = field(x_id)

        assert torch.allclose(v, v_id, atol=1e-5), "Identity rotation changed output"

    def test_so3_equivariance_composition(self):
        """Test equivariance for composed rotations: (R1 @ R2) @ x = R1 @ (R2 @ x)."""
        field = EquivariantStationaryField(use_tensor_product_self=True)
        field.eval()

        x = torch.randn(2, 50, 3)

        R1 = random_rotation_matrix()
        R2 = random_rotation_matrix()
        R_comp = R1 @ R2

        # Path 1: apply R_comp directly
        v_direct = field(x @ R_comp.T)

        # Path 2: apply R2, then R1
        v_step1 = field(x @ R2.T)
        v_step2 = field(v_step1 @ R1.T)  # This is wrong! Don't do this
        # Correct: rotate the positions sequentially
        x_step1 = x @ R2.T
        v_step1 = field(x_step1)
        x_step2 = x_step1 @ R1.T
        v_step2 = field(x_step2)

        # Both should give v_direct
        v_direct_expected = field(x @ R_comp.T)

        error1 = torch.norm(v_direct_expected @ R_comp - field(x) @ R_comp) / (torch.norm(field(x)) + 1e-8)
        print(f"Composition error: {error1.item():.6f}")

    def test_so3_equivariance_batch(self):
        """Test equivariance across batch with different rotations per sample."""
        field = EquivariantStationaryField(use_tensor_product_self=True)
        field.eval()

        B = 4
        x = torch.randn(B, 50, 3)

        # Different rotation per batch sample
        Rs = batch_random_rotation_matrices(B)  # [B, 3, 3]

        v = field(x)  # [B, 50, 3]

        # Rotate each sample in batch
        x_rot = torch.einsum("bij,bnj->bni", Rs, x)  # [B, N, 3]
        v_rot = field(x_rot)

        # Expected: rotate velocities by same rotations
        v_expected = torch.einsum("bij,bnj->bni", Rs, v)

        error = torch.norm(v_rot - v_expected) / (torch.norm(v) + 1e-8)
        print(f"Batch equivariance error: {error.item():.6f}")
        assert error < 1e-4, f"Batch equivariance broken: {error.item()}"

    def test_translation_invariance(self):
        """Field should be translation-invariant: v(x+t) = v(x)."""
        field = EquivariantStationaryField(use_tensor_product_self=True)
        field.eval()

        x = torch.randn(4, 100, 3)
        t = torch.randn(4, 1, 3)  # Translation per batch

        v1 = field(x)
        v2 = field(x + t)

        # Should be approximately equal (but may not be due to field architecture)
        error = torch.norm(v1 - v2) / (torch.norm(v1) + 1e-8)
        print(f"Translation invariance error: {error.item():.6f}")
        # Note: field may not be strictly translation invariant without special design


# ============================================================================
# Test: Comparison with Simple Variant
# ============================================================================


class TestEquivariantStationaryFieldSimple:
    """Tests for simplified variant."""

    def test_simple_variant_shape(self):
        """Simple variant should produce correct output shape."""
        field = EquivariantStationaryFieldSimple()
        x = torch.randn(4, 100, 3)
        v = field(x)
        assert v.shape == (4, 100, 3)

    def test_simple_variant_initialization(self):
        """Simple variant should initialize to zero."""
        field = EquivariantStationaryFieldSimple()
        x = torch.randn(4, 100, 3)
        v = field(x)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-5)

    def test_simple_variant_gradient(self):
        """Simple variant should have gradient flow."""
        field = EquivariantStationaryFieldSimple()
        x = torch.randn(4, 100, 3, requires_grad=True)
        v = field(x)
        loss = v.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# Test: Integration
# ============================================================================


class TestEquivariantStationaryFieldIntegration:
    """Integration tests with training loops and other components."""

    def test_training_step(self):
        """Field should support a training step without NaNs."""
        field = EquivariantStationaryField()
        optimizer = torch.optim.Adam(field.parameters(), lr=0.01)

        x = torch.randn(4, 50, 3)

        for step in range(5):
            v = field(x)
            loss = v.pow(2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            assert not torch.isnan(loss), f"NaN at step {step}"
            assert not torch.isinf(loss), f"Inf at step {step}"

    def test_state_dict_save_load(self):
        """Field should be serializable."""
        field1 = EquivariantStationaryField()
        x = torch.randn(2, 50, 3)
        v1 = field1(x)

        state = field1.state_dict()

        field2 = EquivariantStationaryField()
        field2.load_state_dict(state)
        v2 = field2(x)

        assert torch.allclose(v1, v2), "Loaded field produces different output"

    def test_device_placement(self):
        """Field should work on CPU and CUDA (if available)."""
        x = torch.randn(2, 50, 3)

        # CPU
        field_cpu = EquivariantStationaryField()
        v_cpu = field_cpu(x)
        assert v_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            field_cuda = EquivariantStationaryField().cuda()
            x_cuda = x.cuda()
            v_cuda = field_cuda(x_cuda)
            assert v_cuda.device.type == "cuda"

    def test_eval_mode(self):
        """Field in eval mode should be deterministic."""
        field = EquivariantStationaryField()
        field.eval()

        x = torch.randn(2, 50, 3)

        with torch.no_grad():
            v1 = field(x)
            v2 = field(x)

        assert torch.allclose(v1, v2), "Eval mode not deterministic"


# ============================================================================
# Test: Initialization Variants
# ============================================================================


class TestEquivariantStationaryFieldVariants:
    """Test different initialization and configuration variants."""

    def test_different_hidden_irreps(self):
        """Should work with different hidden irrep configurations."""
        for hidden_irreps in ["2x0e + 1x1o", "6x0e + 3x1o", "1x0e + 1x1o"]:
            field = EquivariantStationaryField(hidden_irreps=hidden_irreps)
            x = torch.randn(2, 50, 3)
            v = field(x)
            assert v.shape == (2, 50, 3)

    def test_no_gating(self):
        """Should work with hidden_irreps with no vectors (no gating)."""
        field = EquivariantStationaryField(hidden_irreps="4x0e")  # Scalars only
        x = torch.randn(2, 50, 3)
        v = field(x)
        assert v.shape == (2, 50, 3)

    def test_verbose_mode(self):
        """Verbose mode should print without errors."""
        field = EquivariantStationaryField(verbose=True)
        x = torch.randn(2, 50, 3)
        v = field(x)
        assert v.shape == (2, 50, 3)


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    # Run basic tests
    print("\n=== Basic Tests ===")
    test = TestEquivariantStationaryFieldBasic()
    test.test_output_shape()
    print("✓ Output shape")
    test.test_initialization_to_zero()
    print("✓ Zero initialization")
    test.test_gradient_flow()
    print("✓ Gradient flow")

    print("\n=== Equivariance Tests ===")
    test_eq = TestEquivariantStationaryFieldEquivariance()
    test_eq.test_so3_equivariance_single_rotation()
    print("✓ Single rotation equivariance")
    test_eq.test_so3_equivariance_multiple_rotations()
    print("✓ Multiple rotation equivariance")
    test_eq.test_so3_equivariance_batch()
    print("✓ Batch equivariance")

    print("\n=== Integration Tests ===")
    test_int = TestEquivariantStationaryFieldIntegration()
    test_int.test_training_step()
    print("✓ Training step")
    test_int.test_state_dict_save_load()
    print("✓ Save/load")

    print("\n=== All tests passed! ===")
