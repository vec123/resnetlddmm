"""Tests for Conditioning ABC and implementations."""

import torch
import pytest
from src.resnet_lddmm.conditioning.base import NoConditioning


def test_no_conditioning_dim():
    """Verify NoConditioning has dim == 0."""
    cond = NoConditioning()
    assert cond.dim == 0


def test_no_conditioning_returns_none():
    """Verify NoConditioning returns None for any input."""
    cond = NoConditioning()

    # Test with various input shapes and codes
    x = torch.randn(2, 10, 3)  # [B=2, N=10, 3]
    code = torch.randn(2, 256)  # [B=2, N_z=256]

    result = cond(x, code)
    assert result is None

    # Test with None code
    result = cond(x, None)
    assert result is None

    # Test with different batch sizes
    x_large = torch.randn(8, 1000, 3)
    code_large = torch.randn(8, 128)
    result = cond(x_large, code_large)
    assert result is None


def test_no_conditioning_forward_is_none():
    """Verify .forward() explicitly returns None."""
    cond = NoConditioning()
    x = torch.randn(1, 5, 3)
    result = cond.forward(x, None)
    assert result is None
