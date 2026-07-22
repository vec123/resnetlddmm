"""SE(3) equivariance / invariance of the GroupEncoder outputs.

The whole point of the equivariant encoder is that it factorizes a shape into

  * ``mu``          -- the INVARIANT scalar latent (0e): must NOT change under any
                       rotation/translation of the input;
  * ``rotation``    -- an EQUIVARIANT frame built from the encoder's 1o vectors: must
                       rotate WITH the input, i.e. ``R_out(R x + t) == R @ R_out(x)``;
  * ``translation`` -- the shape's centre of mass (a 1o vector): must transform as
                       ``t_out(R x + t) == R @ t_out(x) + t``.

We build ONE fixed graph, then feed the encoder a rotated + translated copy with the
SAME edges and SAME node order (only the positions move), so the ONLY difference is the
SE(3) action. e3nn tensor products make these relations exact up to float32 tolerance.

Run with pytest, or:  python tests/learning/test_encoder_equivariance.py
"""

import os
import sys

import torch
from e3nn import o3


from torch_geometric.data import Data

from src.learning.models.group_encoder import GroupEncoder


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_graph(seed=0):
    """A small 2-graph batch (two 6-node rings) with a 1x0e node feature."""
    torch.manual_seed(seed)

    def ring(offset, n):
        s = torch.arange(n) + offset
        d = (torch.arange(n) + 1) % n + offset
        return torch.stack([torch.cat([s, d]), torch.cat([d, s])])

    edge_index = torch.cat([ring(0, 6), ring(6, 6)], dim=1)
    pos = torch.randn(12, 3)
    batch = torch.tensor([0] * 6 + [1] * 6)
    x = torch.ones(12, 1)  # 1x0e input scalar per node
    return Data(x=x, pos=pos, edge_index=edge_index, batch=batch)


def make_encoder(seed=0):
    torch.manual_seed(seed)
    layers_cfg = [{
        'in_irreps': '1x0e',
        'target_irreps': '8x0e + 4x1o + 2x1e',
        'spatial_sh_lmax': 2,
        'interaction_sh_lmax': 4,
    }]
    enc = GroupEncoder(layers_cfg=layers_cfg, latent_dim=4,
                        output_irreps='4x0e + 2x1o',   # 4 invariant scalars + 2 equivariant vectors
                        readout='mean', verbose=False)
    enc.eval()  # deterministic (no dropout / batchnorm active) so equivariance is exact
    return enc


def transform_graph(graph, R, t):
    """Apply x -> R x + t to every node position (rows are points -> ``pos @ R.T + t``);
    edges, node order and features are untouched, so only the SE(3) action changes."""
    new_pos = graph.pos @ R.T + t
    return Data(x=graph.x, pos=new_pos, edge_index=graph.edge_index, batch=graph.batch)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_scalar_latent_is_SE3_invariant():
    """mu (the 0e latent) must be identical before and after an SE(3) transform."""
    enc = make_encoder()
    graph = make_graph()
    torch.manual_seed(1)
    R, t = o3.rand_matrix(), torch.randn(3)   # proper rotation (det=+1) + translation

    with torch.no_grad():
        mu = enc(graph, None).mu
        mu_t = enc(transform_graph(graph, R, t), None).mu

    err = (mu - mu_t).abs().max().item()
    print(f"[invariance] max |mu - mu(Rx+t)| = {err:.2e}")
    assert torch.allclose(mu, mu_t, atol=1e-4), f"mu is NOT SE(3)-invariant (max err {err:.2e})"


def test_rotation_frame_is_equivariant():
    """The 1o-derived frame must satisfy R_out(R x + t) == R @ R_out(x)."""
    enc = make_encoder()
    graph = make_graph()
    torch.manual_seed(2)
    R, t = o3.rand_matrix(), torch.randn(3)

    with torch.no_grad():
        rot = enc(graph, None).rotation                    # [B, 3, 3]
        rot_t = enc(transform_graph(graph, R, t), None).rotation

    expected = torch.einsum('ij,bjk->bik', R, rot)         # R @ R_out(x), per graph
    err = (rot_t - expected).abs().max().item()
    print(f"[equivariance:frame] max |R_out(Rx+t) - R R_out(x)| = {err:.2e}")
    assert torch.allclose(rot_t, expected, atol=1e-3), \
        f"rotation frame is NOT equivariant (max err {err:.2e})"


def test_translation_is_equivariant():
    """The translation (centre of mass, a 1o vector) must map as R @ t_out + t."""
    enc = make_encoder()
    graph = make_graph()
    torch.manual_seed(3)
    R, t = o3.rand_matrix(), torch.randn(3)

    with torch.no_grad():
        transl = enc(graph, None).translation              # [B, 3]
        transl_t = enc(transform_graph(graph, R, t), None).translation

    expected = transl @ R.T + t                            # R @ t_out(x) + t (row convention)
    err = (transl_t - expected).abs().max().item()
    print(f"[equivariance:translation] max |t_out(Rx+t) - (R t_out + t)| = {err:.2e}")
    assert torch.allclose(transl_t, expected, atol=1e-4), \
        f"translation is NOT equivariant (max err {err:.2e})"


def test_rotation_output_is_a_valid_rotation():
    """Sanity: the frame the encoder emits is a proper rotation (orthonormal, det +1)."""
    enc = make_encoder()
    with torch.no_grad():
        rot = enc(make_graph(), None).rotation             # [B, 3, 3]

    eye = torch.eye(3).expand_as(rot)
    orth_err = (torch.einsum('bij,bik->bjk', rot, rot) - eye).abs().max().item()
    dets = torch.det(rot)
    print(f"[frame] max |R^T R - I| = {orth_err:.2e} | dets = {dets.tolist()}")
    assert orth_err < 1e-4, f"frame is not orthonormal (err {orth_err:.2e})"
    assert torch.all(dets > 0), f"frame is not a proper rotation (dets {dets.tolist()})"


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
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {str(exc)[:160]}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")


if __name__ == '__main__':
    _run_all()
