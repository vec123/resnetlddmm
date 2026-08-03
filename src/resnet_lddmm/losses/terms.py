"""Config-selected loss terms.

A term is any module taking a :class:`LossContext` and returning a scalar Tensor,
or ``None`` when it does not apply to the current step or mode. ``None`` is already
a first-class skip in ``LossComposer.compute``, so mode dependence (bidirectional
only, encoder only, ...) costs zero branches in the registration steppers.

Terms are selected by ``loss.terms`` in the YAML and resolved through the
``loss_term`` Registry category, so adding one is a new class plus a registry line
-- never an edit to the steppers.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class LossContext:
    """One training step's inputs, as seen by every configured term.

    Every field defaults to None so that adding a field here never breaks a term
    that does not read it.

    ``template_points`` is the load-bearing one: it is the tensor that was actually
    fed to the flow, AFTER subsampling and the cohort broadcast. Any term that
    instead re-derived its own subset would lose index correspondence with
    ``fwd_traj`` -- silently, since the shapes would still line up.
    """

    flow: Any = None                                   # NeuralODEFlow | IdentityFlowWrapper
    code: Optional[Tensor] = None                      # [B, n_z] or None
    template_points: Optional[Tensor] = None           # [B, N, 3] exactly as flowed
    sample_points: Optional[Tensor] = None             # [B, M, 3] exactly as compared against
    fwd_traj: Any = None                               # Trajectory
    bwd_traj: Any = None                               # Trajectory, bidirectional only
    pred: Optional[Tensor] = None                      # posed endpoint the data term saw
    encoder_pose: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None)
    data_term: Any = None                              # configured DataTerm, reusable
    code_source: Any = None
    template: Any = None                               # unsubsampled batch (faces, weights)
    sample: Any = None                                 # augmented batch

    @property
    def field(self):
        """The velocity field, through whichever flow wrapper is in play."""
        return None if self.flow is None else self.flow.field


def apply_pose(points: Tensor, rotation: Optional[Tensor],
               translation: Optional[Tensor]) -> Tensor:
    """Apply a group element to batched points: ``x·R + t``.

    Deliberately mirrors ``mapping_error._apply_encoder_pose`` including its
    right-multiplication convention. A term that applied ``R·x`` instead would
    disagree with the data term about which element the encoder predicted, and the
    two objectives would pull against each other.

    Args:
        points: [B, N, 3]
        rotation: [B, 3, 3] or None
        translation: [B, 3] or None

    Returns:
        [B, N, 3] transformed points (the input tensor when both are None)
    """
    if rotation is not None:
        points = torch.einsum('bij,bjk->bik', points, rotation)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    return points


class EquivariantDeformationLoss(nn.Module):
    """L2 between ``φ(g·T)`` and ``g·φ(T)``: the deformation must commute with the pose.

    The deformation of a posed template should equal the posed deformation of the
    template. Both sides descend from the SAME ``ctx.template_points`` and the
    integrator is elementwise, so the two clouds are in exact pointwise
    correspondence -- which is why this is a plain L2 and not a Chamfer distance.

    Costs one extra forward integration per step, on the subsampled template, so
    ``loss.subsample_M`` sets its price directly.

    Expected values, useful as sanity checks: exactly 0 under
    ``freeze_flow_at_identity`` (identity commutes with everything), ~0 for an e3nn
    equivariant field (where the property holds by construction), and strictly
    positive for an MLP field, where it is a genuine soft constraint.
    """

    def __init__(self, field_only: bool = True, translation: bool = True):
        """Initialize the equivariance term.

        Args:
            field_only: route the gradient to the VELOCITY FIELD alone. Both the
                group element g and the latent code z are treated as constants, so
                nothing reaches the encoder — neither its pose head via g nor its
                latent path via z. This costs one extra integration: the forward
                trajectory the data term computed carries encoder gradient through
                z, so it cannot be reused for the right-hand side and is recomputed
                against the detached code. The loss VALUE is identical either way;
                only gradient routing and cost change.

                False keeps the cheap path (reuse that trajectory, detach nothing),
                letting the constraint reach the field, the latent, and the pose
                head. Beware: the term is trivially zero at g = identity, so it can
                then be minimised by collapsing the predicted rotation rather than
                by making the flow equivariant.
            translation: include the predicted translation in g. Rotation-only
                (False) isolates the SO(3) part of the constraint.
        """
        super().__init__()
        self.field_only = field_only
        self.translation = translation

    def forward(self, ctx: LossContext) -> Optional[Tensor]:
        """Compute ‖φ(g·T) − g·φ(T)‖², or None when there is no pose to test.

        Args:
            ctx: the step's LossContext

        Returns:
            Scalar tensor, or None when no group element is available (no encoder
            pose predicted, so nothing to be equivariant to)
        """
        rotation, translation = ctx.encoder_pose
        if not self.translation:
            translation = None
        if rotation is None and translation is None:
            return None

        code = ctx.code
        if self.field_only:
            rotation = None if rotation is None else rotation.detach()
            translation = None if translation is None else translation.detach()
            code = None if code is None else code.detach()
            # ctx.fwd_traj was built with the LIVE code, so reusing it would leak
            # gradient into the encoder. Recompute against the detached code: same
            # numbers, but a graph that stops at the field.
            deformed = ctx.flow(ctx.template_points, code).end
        else:
            deformed = ctx.fwd_traj.end

        # Flow the POSED template. Same tensor the forward trajectory started from,
        # so lhs[i] and rhs[i] describe the same vertex.
        posed_template = apply_pose(ctx.template_points, rotation, translation)
        lhs = ctx.flow(posed_template, code).end              # φ(g·T)

        rhs = apply_pose(deformed, rotation, translation)     # g·φ(T)

        return (lhs - rhs).pow(2).sum(-1).mean()
