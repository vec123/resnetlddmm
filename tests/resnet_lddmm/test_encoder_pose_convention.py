"""The handedness contract between GroupEncoder's rotation and the pipeline's.

GroupEncoder builds its rotation from type-1 features, making it equivariant as
``R(x·Q) = Qᵀ·R(x)`` for EVERY weight setting — architecture, not training. Poses are
applied downstream as ``points @ R``, which needs ``P(x·Q) = P(x)·Q``. Requiring both
of an unmodified R gives ``Qᵀ = A Q A⁻¹``: inversion is an anti-automorphism,
conjugation an automorphism, and on non-abelian SO(3) no such A exists — so the pose
objective would be unsatisfiable rather than merely hard. EncoderCodes.get_pose
transposes to close the gap. These tests pin that.
"""

import torch
from types import SimpleNamespace

from src.resnet_lddmm.codes.encoder import EncoderCodes


def _rotation(angle, axis=2):
    """A proper rotation about one axis, as [1, 3, 3]."""
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    R = torch.eye(3)
    a, b = [i for i in range(3) if i != axis]
    R[a, a], R[a, b], R[b, a], R[b, b] = c, -s, s, c
    return R.unsqueeze(0)


def _codes():
    """EncoderCodes with no graph builder or encoder: get_pose reads _last only."""
    return EncoderCodes(graph_builder=None, encoder=None, n_z=4)


class TestEncoderPoseConvention:

    def test_get_pose_transposes(self):
        codes = _codes()
        R = _rotation(0.7)
        codes._last = SimpleNamespace(rotation=R, translation=None)

        got, _ = codes.get_pose()

        assert torch.allclose(got, R.transpose(1, 2))

    def test_left_equivariant_encoder_yields_right_equivariant_pose(self):
        """The property the whole pipeline depends on.

        Feeding a rotated input left-multiplies the encoder's raw rotation by Qᵀ.
        After the transpose, get_pose must instead RIGHT-multiply by Q, which is what
        ``points @ P`` needs in order to track a rotated input.
        """
        codes = _codes()
        canonical = _rotation(0.3)
        Q = _rotation(1.1)

        codes._last = SimpleNamespace(rotation=canonical, translation=None)
        pose_x, _ = codes.get_pose()

        # what the architecture produces for the rotated input
        codes._last = SimpleNamespace(rotation=Q.transpose(1, 2) @ canonical, translation=None)
        pose_xq, _ = codes.get_pose()

        assert torch.allclose(pose_xq, pose_x @ Q, atol=1e-6)

    def test_translation_passes_through_untouched(self):
        codes = _codes()
        t = torch.tensor([[0.1, -0.2, 0.3]])
        codes._last = SimpleNamespace(rotation=_rotation(0.5), translation=t)

        _, got = codes.get_pose()

        assert got is t

    def test_none_rotation_survives(self):
        codes = _codes()
        codes._last = SimpleNamespace(rotation=None, translation=None)
        assert codes.get_pose() == (None, None)

    def test_no_output_yet(self):
        assert _codes().get_pose() == (None, None)

    def test_frame_transform_uses_the_same_convention(self):
        """with_pose_transform must not reintroduce the raw, untransposed rotation."""
        from src.resnet_lddmm.io import FrameTransform

        codes = _codes()
        R = _rotation(0.9)
        codes._last = SimpleNamespace(rotation=R, translation=None)

        out = codes.with_pose_transform(FrameTransform(center=torch.zeros(3), scale=1.0))

        assert torch.allclose(out.rotation, R.transpose(1, 2))
