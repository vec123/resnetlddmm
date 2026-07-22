"""Acceptance tests for the equivariant modules (contract for the jax -> torch port).

These tests define the behaviour every module MUST satisfy once ported to PyTorch
e3nn 0.6.0. They intentionally assume the idiomatic torch-e3nn convention:

    * features are plain ``torch.Tensor`` of shape ``[N, irreps.dim]``
    * the irreps are tracked by the module, not carried on the tensor
    * a rotation acts on features as ``x @ D.T`` where ``D = irreps.D_from_matrix(R)``
      and on positions as ``pos @ R.T``

Against the current (e3nn-jax-style) modules these tests FAIL — that is the point:
they are the red baseline that the port must turn green. Run with pytest, or
directly:  python tests/learning/test_equivariant_modules.py
"""

import os
import sys

import torch
from e3nn import o3


from src.learning.modules.equivariant.layer_norm import EquivariantLayerNorm
from src.learning.modules.equivariant.interaction import SelfInteraction, SpatialConvolution
from src.learning.modules.equivariant.attention import EquivariantAttention
from src.learning.layers.equivariant.Self_Spatial_layer import EquiLayer
from src.learning.models.group_encoder import GroupEncoder

torch.manual_seed(0)

ATOL = 1e-3  # float32 message passing accumulates error; keep a sane tolerance.


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def rotate_features(x, irreps, R):
    """Apply the Wigner-D of rotation ``R`` to a feature tensor laid out per ``irreps``."""
    D = o3.Irreps(irreps).D_from_matrix(R)
    return x @ D.T


def max_err(a, b):
    return (a - b).abs().max().item()


def assert_close(a, b, msg, atol=ATOL):
    assert torch.allclose(a, b, atol=atol, rtol=1e-3), f"{msg} (max err {max_err(a, b):.2e})"


def ring_edges(n):
    """A fixed directed ring + reverse edges — deterministic topology for conv tests."""
    src = torch.arange(n)
    dst = (src + 1) % n
    fwd = torch.stack([src, dst])
    bwd = torch.stack([dst, src])
    return torch.cat([fwd, bwd], dim=1)


# --------------------------------------------------------------------------- #
# EquivariantLayerNorm
# --------------------------------------------------------------------------- #
def test_layer_norm_preserves_shape():
    irreps = '4x0e + 2x1o'
    ln = EquivariantLayerNorm(irreps, affine=True, verbose=False)
    x = torch.randn(5, o3.Irreps(irreps).dim)
    out = ln(x)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"


def test_layer_norm_is_equivariant():
    """LN(D x) == D LN(x): normalization commutes with rotation."""
    irreps = '4x0e + 2x1o'
    ln = EquivariantLayerNorm(irreps, affine=True, verbose=False)
    x = torch.randn(5, o3.Irreps(irreps).dim)
    R = o3.rand_matrix()

    out_then_rot = rotate_features(ln(x), irreps, R)
    rot_then_out = ln(rotate_features(x, irreps, R))
    assert_close(out_then_rot, rot_then_out, "EquivariantLayerNorm is not rotation-equivariant")


def test_layer_norm_output_is_finite():
    irreps = '4x0e + 2x1o'
    ln = EquivariantLayerNorm(irreps, affine=True, verbose=False)
    out = ln(torch.randn(5, o3.Irreps(irreps).dim))
    assert torch.isfinite(out).all(), "EquivariantLayerNorm produced non-finite values"


# --------------------------------------------------------------------------- #
# SelfInteraction
# --------------------------------------------------------------------------- #
def test_self_interaction_shape():
    in_irreps, target = '4x0e + 2x1o', '4x0e + 2x1o'
    si = SelfInteraction(in_irreps, target, sh_lmax=1, verbose=False)
    x = torch.randn(6, o3.Irreps(in_irreps).dim)
    out = si(x)
    assert out.shape == (6, o3.Irreps(target).dim)


def test_self_interaction_is_equivariant():
    """SI(D_in x) == D_out SI(x)."""
    in_irreps, target = '4x0e + 2x1o', '4x0e + 2x1o'
    si = SelfInteraction(in_irreps, target, sh_lmax=1, verbose=False)
    x = torch.randn(6, o3.Irreps(in_irreps).dim)
    R = o3.rand_matrix()

    out_then_rot = rotate_features(si(x), target, R)
    rot_then_out = si(rotate_features(x, in_irreps, R))
    assert_close(out_then_rot, rot_then_out, "SelfInteraction is not rotation-equivariant")


# --------------------------------------------------------------------------- #
# SpatialConvolution (message passing over positions)
# --------------------------------------------------------------------------- #
def test_spatial_conv_shape():
    in_irreps, target = '4x0e + 2x1o', '4x0e + 2x1o'
    conv = SpatialConvolution(in_irreps, target, sh_lmax=1, verbose=False)
    n = 6
    x = torch.randn(n, o3.Irreps(in_irreps).dim)
    pos = torch.randn(n, 3)
    edge_index = ring_edges(n)
    out = conv(x, pos, edge_index)
    assert out.shape == (n, o3.Irreps(target).dim)


def test_spatial_conv_rotation_equivariance():
    """Conv(D_in x, R pos, E) == D_out Conv(x, pos, E): messages built from
    spherical harmonics of relative positions rotate with the geometry."""
    in_irreps, target = '4x0e + 2x1o', '4x0e + 2x1o'
    conv = SpatialConvolution(in_irreps, target, sh_lmax=1, verbose=False)
    n = 6
    x = torch.randn(n, o3.Irreps(in_irreps).dim)
    pos = torch.randn(n, 3)
    edge_index = ring_edges(n)
    R = o3.rand_matrix()

    out_then_rot = rotate_features(conv(x, pos, edge_index), target, R)
    rot_then_out = conv(rotate_features(x, in_irreps, R), pos @ R.T, edge_index)
    assert_close(out_then_rot, rot_then_out, "SpatialConvolution is not rotation-equivariant")


def test_spatial_conv_translation_invariance():
    """Conv(x, pos + t, E) == Conv(x, pos, E): only relative positions enter."""
    in_irreps, target = '4x0e + 2x1o', '4x0e + 2x1o'
    conv = SpatialConvolution(in_irreps, target, sh_lmax=1, verbose=False)
    n = 6
    x = torch.randn(n, o3.Irreps(in_irreps).dim)
    pos = torch.randn(n, 3)
    edge_index = ring_edges(n)
    t = torch.randn(3)

    base = conv(x, pos, edge_index)
    shifted = conv(x, pos + t, edge_index)
    assert_close(base, shifted, "SpatialConvolution is not translation-invariant")


# --------------------------------------------------------------------------- #
# EquivariantAttention
# --------------------------------------------------------------------------- #
def _attention_graph(irreps_in, n=4):
    x = torch.randn(n, o3.Irreps(irreps_in).dim)
    pos = torch.randn(n, 3)
    senders = torch.tensor([0, 1, 2, 3, 0, 2])
    receivers = torch.tensor([1, 2, 3, 0, 2, 0])
    return x, pos, senders, receivers, n


def test_attention_shape():
    irreps_in, irreps_out = '1x0e + 1x1o', '1x0e + 1x1o'
    attn = EquivariantAttention(irreps_in, irreps_out, sh_lmax=1, verbose=False)
    x, pos, senders, receivers, n = _attention_graph(irreps_in)
    out, alpha = attn(x, pos, senders, receivers, n)
    assert out.shape == (n, o3.Irreps(irreps_out).dim)
    assert alpha.shape[0] == senders.shape[0]


def test_attention_rotation_equivariance():
    """Attention output rotates with the input geometry."""
    irreps_in, irreps_out = '1x0e + 1x1o', '1x0e + 1x1o'
    attn = EquivariantAttention(irreps_in, irreps_out, sh_lmax=1, verbose=False)
    x, pos, senders, receivers, n = _attention_graph(irreps_in)
    R = o3.rand_matrix()

    out, _ = attn(x, pos, senders, receivers, n)
    out_rot, _ = attn(rotate_features(x, irreps_in, R), pos @ R.T, senders, receivers, n)
    assert_close(rotate_features(out, irreps_out, R), out_rot,
                 "EquivariantAttention output is not rotation-equivariant")


def test_attention_weights_are_invariant():
    """Attention scores are scalars (0e) and must be rotation-invariant."""
    irreps_in, irreps_out = '1x0e + 1x1o', '1x0e + 1x1o'
    attn = EquivariantAttention(irreps_in, irreps_out, sh_lmax=1, verbose=False)
    x, pos, senders, receivers, n = _attention_graph(irreps_in)
    R = o3.rand_matrix()

    _, alpha = attn(x, pos, senders, receivers, n)
    _, alpha_rot = attn(rotate_features(x, irreps_in, R), pos @ R.T, senders, receivers, n)
    assert_close(alpha, alpha_rot, "EquivariantAttention weights are not rotation-invariant")


# --------------------------------------------------------------------------- #
# EquiLayer (SelfInteraction + SpatialConvolution + LayerNorm)
# --------------------------------------------------------------------------- #
def test_equi_layer_shape_and_equivariance():
    in_irreps, target = '1x0e + 1x1o', '4x0e + 2x1o'
    layer = EquiLayer(in_irreps, target, verbose=False)
    n = 6
    x = torch.randn(n, o3.Irreps(in_irreps).dim)
    pos = torch.randn(n, 3)
    edge_index = ring_edges(n)
    R = o3.rand_matrix()

    out = layer(x, pos, edge_index)
    assert out.shape == (n, o3.Irreps(target).dim)

    out_rot = layer(rotate_features(x, in_irreps, R), pos @ R.T, edge_index)
    assert_close(rotate_features(out, target, R), out_rot,
                 "EquiLayer is not rotation-equivariant")


# --------------------------------------------------------------------------- #
# GroupEncoder (end-to-end: invariant latent + equivariant pose)
# --------------------------------------------------------------------------- #
def _encoder_inputs(n=8):
    layers_cfg = [{
        'in_irreps': '1x0e',
        'target_irreps': '4x0e + 2x1o',
        'spatial_sh_lmax': 1,
        'interaction_sh_lmax': 4,
    }]
    enc = GroupEncoder(layers_cfg=layers_cfg, latent_dim=4,
                        output_irreps='4x0e + 2x1o', verbose=False)
    x = torch.randn(n, 1)                 # 1x0e scalar per node
    pos = torch.randn(n, 3)
    edge_index = ring_edges(n)
    batch = torch.zeros(n, dtype=torch.long)
    return enc, x, pos, edge_index, batch


def _encoder_graph(x, pos, edge_index, batch):
    """Wrap the mock inputs in the PyG ``Data`` graph the encoder now consumes."""
    from torch_geometric.data import Data
    return Data(x=x, pos=pos, edge_index=edge_index, batch=batch)


def test_group_encoder_latent_is_invariant():
    """The scalar VAE latent (mu) must not change when the input is rotated."""
    enc, x, pos, edge_index, batch = _encoder_inputs()
    R = o3.rand_matrix()

    mu = enc(_encoder_graph(x, pos, edge_index, batch), None).mu
    mu_rot = enc(_encoder_graph(x, pos @ R.T, edge_index, batch), None).mu
    assert_close(mu, mu_rot, "GroupEncoder latent mu is not rotation-invariant")


def test_group_encoder_frame_is_equivariant_and_orthonormal():
    """The predicted rotation frame must be orthonormal and rotate with the input."""
    enc, x, pos, edge_index, batch = _encoder_inputs()
    R = o3.rand_matrix()

    R_pred = enc(_encoder_graph(x, pos, edge_index, batch), None).rotation
    R_pred_rot = enc(_encoder_graph(x, pos @ R.T, edge_index, batch), None).rotation

    # Orthonormal: R Rᵀ = I
    eye = torch.eye(3).expand_as(R_pred @ R_pred.transpose(-2, -1))
    assert_close(R_pred @ R_pred.transpose(-2, -1), eye, "predicted frame is not orthonormal")

    # Equivariant: rotating the input premultiplies the frame by R.
    assert_close(R @ R_pred, R_pred_rot, "predicted frame is not rotation-equivariant")


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {str(exc)[:120]}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")


if __name__ == '__main__':
    _run_all()
