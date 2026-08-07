"""Frame parameterisation: the orthogonalisation step of the encoder's pose head.

``GroupEncoder`` builds its rotation from two 1o vectors. How those two become an
orthonormal frame is ``pose_mode``:

    first_moment / second_moment    Gram-Schmidt
    polar                           symmetric (Lowdin) orthogonalisation

These tests pin the properties that make a frame usable at all -- proper rotation,
left-equivariance, scale invariance -- and the one property that distinguishes polar
from Gram-Schmidt: it treats its two inputs alike.

Float64 throughout. The geodesic angle via ``arccos((tr-1)/2)`` is catastrophically
ill-conditioned near identity, which is exactly where a sensitivity test lives, so the
chordal form is used instead (see ``_angle``).
"""

import math

import pytest
import torch

from src.learning.models.group_encoder import GroupEncoder
from src.spec import EncoderConfig, EncoderLayerConfig


gram_schmidt = GroupEncoder._Gram_Schmidt_frame
polar = GroupEncoder._polar_frame


def _angle(A, B):
    """Geodesic angle in degrees, via ||A - B||_F = 2 sqrt(2) sin(theta/2).

    Well-conditioned near identity, unlike arccos((tr - 1) / 2), whose derivative
    is unbounded there -- in float32 that metric reports pure rounding noise for
    the perturbation sizes these tests use.
    """
    chord = (A - B).reshape(*A.shape[:-2], 9).norm(dim=-1) / (2 * math.sqrt(2))
    return torch.rad2deg(2 * torch.arcsin(chord.clamp(max=1.0)))


def _pair(batch, theta_deg, generator):
    """Two unit vectors separated by exactly ``theta_deg``, uniformly oriented."""
    theta = math.radians(theta_deg)
    a = torch.randn(batch, 3, generator=generator, dtype=torch.float64)
    a = a / a.norm(dim=-1, keepdim=True)
    h = torch.randn(batch, 3, generator=generator, dtype=torch.float64)
    h = h - (h * a).sum(-1, keepdim=True) * a
    h = h / h.norm(dim=-1, keepdim=True)
    return a, math.cos(theta) * a + math.sin(theta) * h


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)


@pytest.mark.parametrize("frame", [gram_schmidt, polar], ids=["gram_schmidt", "polar"])
class TestFrameValidity:
    """Properties both parameterisations must satisfy to be usable at all."""

    @pytest.mark.parametrize("theta_deg", [90.0, 45.0, 16.0])
    def test_is_a_proper_rotation(self, frame, theta_deg, rng):
        """det = +1 and orthonormal columns -- a reflection would not be in SO(3).

        At CONTROLLED angles rather than fully random pairs. Both implementations
        normalise as ``v / (||v|| + eps)``, and eps is absolute while the quantity it
        guards scales with ``||v|| sin(theta)``, so its relative weight blows up as a
        pair approaches (anti)parallel: a random batch of 256 reliably contains a pair
        near 178 degrees, where orthonormality degrades to ~1e-6. That is the
        documented degeneracy, not a property worth asserting against here.
        """
        v1, v2 = _pair(256, theta_deg, rng)

        R = frame(v1, v2)
        identity = torch.eye(3, dtype=torch.float64)

        assert torch.allclose(R.det(), torch.ones(256, dtype=torch.float64), atol=1e-7)
        assert torch.allclose(R.transpose(-2, -1) @ R, identity, atol=1e-7)

    def test_is_left_equivariant(self, frame, rng):
        """f(Qv) == Q f(v).

        The encoder is LEFT-equivariant by construction and ``EncoderCodes.get_pose``
        transposes to convert. A parameterisation that broke this convention would
        silently disagree with the data term about which element was predicted.
        """
        Q = torch.linalg.qr(torch.randn(3, 3, generator=rng, dtype=torch.float64))[0]
        Q = Q * Q.det().sign()                       # into SO(3), not just O(3)
        v1 = torch.randn(128, 3, generator=rng, dtype=torch.float64)
        v2 = torch.randn(128, 3, generator=rng, dtype=torch.float64)

        rotated = frame(v1 @ Q.T, v2 @ Q.T)          # x -> Qx, column convention

        assert torch.allclose(Q @ frame(v1, v2), rotated, atol=1e-10)

    def test_is_scale_invariant(self, frame, rng):
        """Only the DIRECTIONS carry pose, so the frame must ignore the norms.

        This matters here: the pose vectors are measured at norms ~0.02, because the
        first moment nearly cancels over a closed surface. A parameterisation that
        was norm-sensitive would confound "small" with "uncertain".

        Held to 1e-6, and the down-scaling kept mild: the eps in ``v / (||v|| + eps)``
        is a fixed absolute quantity, so its relative weight grows as the vectors
        shrink. Gram-Schmidt also hard-errors below ||v|| = 1e-4.
        """
        v1 = torch.randn(128, 3, generator=rng, dtype=torch.float64)
        v2 = torch.randn(128, 3, generator=rng, dtype=torch.float64)

        assert torch.allclose(frame(v1, v2), frame(37.0 * v1, 0.05 * v2), atol=1e-6)


class TestPolarSpecific:
    """What polar orthogonalisation buys over Gram-Schmidt, and what it does not."""

    def test_treats_its_two_inputs_alike(self, rng):
        """The defining property: equal input error produces equal output error.

        Gram-Schmidt keeps v1 EXACTLY and forces the whole correction onto v2, so
        v1's error reaches the frame unattenuated while v2's is partly repaired.
        Polar splits it. Measured at the encoder's operating angle, the asymmetry
        Gram-Schmidt shows is ~8%; polar's is zero to numerical precision.
        """
        batch, epsilon = 20000, 1e-6
        v1, v2 = _pair(batch, 16.0, rng)
        zero = torch.zeros(batch, 3, dtype=torch.float64)
        noise = lambda: torch.randn(batch, 3, generator=rng, dtype=torch.float64) * epsilon

        moved = {}
        for label, (p1, p2) in (("v1", (noise(), zero)), ("v2", (zero, noise()))):
            moved[label] = {
                "gram_schmidt": _angle(gram_schmidt(v1, v2),
                                       gram_schmidt(v1 + p1, v2 + p2)).mean().item(),
                "polar": _angle(polar(v1, v2), polar(v1 + p1, v2 + p2)).mean().item(),
            }

        polar_gap = abs(moved["v1"]["polar"] - moved["v2"]["polar"]) / moved["v1"]["polar"]
        gs_gap = abs(moved["v1"]["gram_schmidt"]
                     - moved["v2"]["gram_schmidt"]) / moved["v1"]["gram_schmidt"]

        assert polar_gap < 0.01, f"polar should be symmetric, got {polar_gap:.3f}"
        assert gs_gap > polar_gap

    def test_is_the_identity_map_at_ninety_degrees(self, rng):
        """Orthogonal inputs are already a frame, so nothing should move.

        This is the case a naive ``torch.linalg.svd`` implementation gets WRONG: all
        three singular values coincide, U and V are individually arbitrary, and while
        their product is right in exact arithmetic it is noisy in floating point. The
        closed form has no such failure.
        """
        v1, v2 = _pair(64, 90.0, rng)

        R = polar(v1, v2)

        assert torch.allclose(R[:, :, 0], v1, atol=1e-12)
        assert torch.allclose(R[:, :, 1], v2, atol=1e-12)

    def test_does_not_rescue_near_parallel_inputs(self, rng):
        """Both methods amplify input noise the same way; polar is not a fix.

        Worth pinning as a NEGATIVE result, because it is the whole reason polar is a
        refinement rather than a solution: amplification is governed by the ANGLE
        between the two pose vectors -- it grows like 1/sin(theta) -- and not by how
        they are orthogonalised. Reducing it needs different POOLING
        (``second_moment`` returns orthonormal axes, i.e. theta = 90 degrees), not a
        different orthogonalisation.
        """
        batch, epsilon = 20000, 1e-6
        amplification = {}
        for theta_deg in (16.0, 8.0):
            v1, v2 = _pair(batch, theta_deg, rng)
            d1 = torch.randn(batch, 3, generator=rng, dtype=torch.float64) * epsilon
            d2 = torch.randn(batch, 3, generator=rng, dtype=torch.float64) * epsilon
            amplification[theta_deg] = {
                "gram_schmidt": _angle(gram_schmidt(v1, v2),
                                       gram_schmidt(v1 + d1, v2 + d2)).mean().item(),
                "polar": _angle(polar(v1, v2), polar(v1 + d1, v2 + d2)).mean().item(),
            }

        # The two methods are within a few percent of each other at both angles.
        for theta_deg, moved in amplification.items():
            ratio = moved["gram_schmidt"] / moved["polar"]
            assert ratio == pytest.approx(1.0, abs=0.1), (
                f"at {theta_deg} deg the two differ by {ratio:.2f}x -- if this grows, "
                f"the choice of orthogonalisation has started to matter after all"
            )

        # Halving sin(theta) roughly doubles the amplification, for both.
        expected = math.sin(math.radians(16.0)) / math.sin(math.radians(8.0))
        for method in ("gram_schmidt", "polar"):
            grew = amplification[8.0][method] / amplification[16.0][method]
            assert grew == pytest.approx(expected, rel=0.1)


class TestPoseModeIsReachable:
    """pose_mode existed on GroupEncoder but no config path passed it (until now)."""

    def test_encoder_config_carries_and_validates_it(self):
        assert EncoderConfig().pose_mode == "first_moment"
        assert EncoderConfig(pose_mode="polar").pose_mode == "polar"

        with pytest.raises(ValueError, match="unknown pose_mode"):
            EncoderConfig(pose_mode="gram_schmidt")

    @pytest.mark.parametrize("mode", ["first_moment", "second_moment", "polar"])
    def test_encoder_accepts_every_configured_mode(self, mode):
        """Every value EncoderConfig admits must be one GroupEncoder admits."""
        layers = [{"in_irreps": "1x0e", "target_irreps": "8x0e + 2x1o",
                   "spatial_sh_lmax": 1, "interaction_sh_lmax": 2}]

        encoder = GroupEncoder(layers, latent_dim=4, pose_mode=mode,
                               transformer_type=None)

        assert encoder.pose_mode == mode

    def test_encoder_rejects_an_unknown_mode(self):
        layers = [{"in_irreps": "1x0e", "target_irreps": "8x0e + 2x1o",
                   "spatial_sh_lmax": 1, "interaction_sh_lmax": 2}]

        with pytest.raises(ValueError, match="pose_mode must be"):
            GroupEncoder(layers, latent_dim=4, pose_mode="nope")

    def test_config_and_encoder_agree_on_the_valid_set(self):
        """A mode EncoderConfig allows but GroupEncoder rejects would fail at build
        time, several seconds into a run, which is the failure this pairing avoids."""
        assert set(EncoderConfig.POSE_MODES) == {"first_moment", "second_moment", "polar"}
