"""Loss terms for shape registration: data fitting and flow regularization."""

import abc
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.learning.losses.losses import chamfer_loss
from src.resnet_lddmm.trajectory import Trajectory
from src.resnet_lddmm.fields.base import VelocityField

try:
    import geomloss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False


class RegistrationLoss(nn.Module, abc.ABC):
    """Abstract base for all registration loss terms.

    Subclasses: DataTerm (data fitting), FlowTerm (velocity field regularization).
    """

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Compute loss. Signature varies by subclass."""


class DataTerm(RegistrationLoss):
    """Measures mismatch between predicted and target point clouds.

    Maps [B,N,3] predicted points and [B,M,3] target points to a scalar loss.
    Subclasses decide how to combine these with optional weights and normals.
    """

    @abc.abstractmethod
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute data term loss.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w: [B, N] per-point weights for pred (unused by basic terms)
            tgt_w: [B, M] per-point weights for target (unused by basic terms)
            normals: [B, M, 3] target surface normals (unused by basic terms)

        Returns:
            Scalar loss tensor
        """


class CDData(DataTerm):
    """Chamfer distance on unpadded point clouds.

    Wraps the reused chamfer_loss with an all-ones mask, since our inputs
    have no padding. Symmetric and differentiable.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Chamfer distance with all-ones mask.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar Chamfer loss
        """
        # Create all-ones mask for unpadded target cloud
        mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )
        return chamfer_loss(pred, target, mask)


class L2Data(DataTerm):
    """Mean squared pointwise error for corresponded point clouds.

    Assumes points are in correspondence by index (e.g., template points
    matched to target points at the same position). Simpler than Chamfer
    but requires pre-established correspondence.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute mean squared error between matched clouds.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, N, 3] target positions (same N as pred)
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar MSE loss
        """
        return (pred - target).pow(2).sum(-1).mean()


class EMDData(DataTerm):
    """Earth Mover's Distance via Sinkhorn's algorithm (geomloss).

    Recommended for articulated shapes (e.g., hands) where Chamfer can
    produce incorrect finger-to-finger matchings. Solves optimal transport
    problem: min over permutations of sum of pairwise distances.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto"):
        """Initialize EMD loss.

        Args:
            p: Distance norm (default 2 for L2)
            blur: Sinkhorn blur radius (higher = smoother gradient, default 0.01)
            backend: geomloss backend ("auto", "keops", "torch", default "auto")
        """
        super().__init__()
        if not HAS_GEOMLOSS:
            raise ImportError(
                "geomloss required for EMD. Install with: pip install geomloss"
            )
        self.loss_fn = geomloss.SamplesLoss(
            "sinkhorn", p=p, blur=blur, backend=backend
        )

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Earth Mover's Distance via Sinkhorn.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w: [B, N] weights for pred (optional, passed to geomloss)
            tgt_w: [B, M] weights for target (optional, passed to geomloss)
            normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar EMD loss
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape
        assert B == B_tgt, f"Batch size mismatch: {B} vs {B_tgt}"

        loss_total = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        for b in range(B):
            p_b = pred[b]  # [N, 3]
            t_b = target[b]  # [M, 3]

            # Compute Sinkhorn loss; weights can be None
            loss_b = self.loss_fn(
                pred_w[b] if pred_w is not None else torch.ones(N, device=pred.device),
                p_b,
                tgt_w[b] if tgt_w is not None else torch.ones(M, device=target.device),
                t_b,
            )
            loss_total = loss_total + loss_b

        return loss_total / B


class FlowTerm(RegistrationLoss):
    """Regularizes velocity field properties during integration.

    Subclasses: IsometryLoss (shape-preservation), and future terms.
    """

    @abc.abstractmethod
    def forward(self, trajectory: Trajectory, field: VelocityField) -> Tensor:
        """Compute regularization loss on trajectory and field."""


class IsometryLoss(FlowTerm):
    """Penalizes Jacobian distortion to encourage shape-preserving deformations.

    Supported loss types:
    - "strain": mean((singular_values - 1)^2), penalizes stretch/compression
    - "det": mean((det(J) - 1)^2), penalizes volume change
    - "orthogonal": mean(||J - R_closest||_F^2), encourages rigid (SO(3)) maps
    """

    def __init__(self, loss_type: str = "strain", sample_points: int = 64):
        super().__init__()
        if loss_type not in ("strain", "det", "orthogonal"):
            raise ValueError(f"loss_type must be 'strain', 'det', or 'orthogonal', got {loss_type}")
        self.loss_type = loss_type
        self.sample_points = sample_points

    def forward(self, trajectory: Trajectory, field: VelocityField) -> Tensor:
        """Compute isometry penalty using finite differences for jacobian.

        Samples a subset of points to reduce computational cost.
        """
        points = trajectory.points.detach()
        K = points.shape[0] - 1
        B, N = points.shape[1], points.shape[2]

        # Sample points for efficiency
        n_sample = min(self.sample_points, N)
        indices = torch.randperm(N, device=points.device)[:n_sample]

        all_losses = []

        # For each step
        for k in range(K):
            x = points[k]  # [B, N, 3]

            # Process each batch and sampled point
            for b in range(B):
                for idx in indices:
                    x_bn = x[b, idx:idx+1, :].clone()  # [1, 3]
                    x_bn.requires_grad_(True)

                    # Create a wrapper for this point
                    def field_fn(x_):
                        return field(x_, code=None, step=k)

                    # Compute loss using jacobian
                    if self.loss_type == "det":
                        loss_bn = self._det_from_velocity(field_fn, x_bn)
                    elif self.loss_type == "strain":
                        loss_bn = self._strain_from_velocity(field_fn, x_bn)
                    elif self.loss_type == "orthogonal":
                        loss_bn = self._orthogonal_from_velocity(field_fn, x_bn)

                    all_losses.append(loss_bn)

        # Compute final loss as mean of all point losses
        return torch.stack(all_losses).mean()

    def _det_from_velocity(self, field_fn, x: Tensor) -> Tensor:
        """Compute det(J) by differentiating field output."""
        v = field_fn(x)  # [1, 3]

        # Compute jacobian via finite differences without in-place ops
        eps = 1e-4
        jac_cols = []

        for j in range(3):
            # Create perturbation tensor (non-in-place)
            delta = torch.zeros_like(x)
            delta[0, j] = eps
            x_pert = x + delta
            v_pert = field_fn(x_pert)
            jac_col = (v_pert[0] - v[0]) / eps  # [3]
            jac_cols.append(jac_col)

        # Stack columns into jacobian: [3, 3]
        jac = torch.stack(jac_cols, dim=1)

        det_val = torch.det(jac)
        return (det_val - 1.0) ** 2

    def _strain_from_velocity(self, field_fn, x: Tensor) -> Tensor:
        """Compute strain penalty via finite differences."""
        v = field_fn(x)  # [1, 3]
        eps = 1e-4
        jac_cols = []

        for j in range(3):
            delta = torch.zeros_like(x)
            delta[0, j] = eps
            x_pert = x + delta
            v_pert = field_fn(x_pert)
            jac_col = (v_pert[0] - v[0]) / eps  # [3]
            jac_cols.append(jac_col)

        jac = torch.stack(jac_cols, dim=1)  # [3, 3]
        _, S, _ = torch.svd(jac)
        return torch.mean((S - 1.0) ** 2)

    def _orthogonal_from_velocity(self, field_fn, x: Tensor) -> Tensor:
        """Compute orthogonal penalty via finite differences."""
        v = field_fn(x)  # [1, 3]
        eps = 1e-4
        jac_cols = []

        for j in range(3):
            delta = torch.zeros_like(x)
            delta[0, j] = eps
            x_pert = x + delta
            v_pert = field_fn(x_pert)
            jac_col = (v_pert[0] - v[0]) / eps  # [3]
            jac_cols.append(jac_col)

        jac = torch.stack(jac_cols, dim=1)  # [3, 3]
        U, _, Vt = torch.svd(jac)
        R = U @ Vt
        return torch.sum((jac - R) ** 2)


    def _strain_loss(self, jacobians: Tensor) -> Tensor:
        """Penalize singular values deviating from 1 (stretch/compression).

        jacobians: [K, B, N, 3, 3]
        """
        K = jacobians.shape[0]
        B = jacobians.shape[1]
        N = jacobians.shape[2]
        jacobians_flat = jacobians.reshape(K * B * N, 3, 3)

        # Compute singular values via SVD
        _, S, _ = torch.svd(jacobians_flat)  # S: [K*B*N, 3]

        return torch.mean((S - 1.0) ** 2)

    def _det_loss(self, jacobians: Tensor) -> Tensor:
        """Penalize determinant deviating from 1 (volume change).

        jacobians: [K, B, N, 3, 3]
        """
        K = jacobians.shape[0]
        B = jacobians.shape[1]
        N = jacobians.shape[2]
        jacobians_flat = jacobians.reshape(K * B * N, 3, 3)

        det_j = torch.det(jacobians_flat)  # [K*B*N]

        return torch.mean((det_j - 1.0) ** 2)

    def _orthogonal_loss(self, jacobians: Tensor) -> Tensor:
        """Penalize Jacobian deviating from orthogonal (rigid maps).

        Finds nearest orthogonal matrix R via Procrustes and measures ||J - R||_F^2.
        jacobians: [K, B, N, 3, 3]
        """
        K = jacobians.shape[0]
        B = jacobians.shape[1]
        N = jacobians.shape[2]
        jacobians_flat = jacobians.reshape(K * B * N, 3, 3)

        # Nearest orthogonal matrix via SVD: R = U @ V^T
        U, _, Vt = torch.svd(jacobians_flat)
        R_closest = U @ Vt  # [K*B*N, 3, 3]

        # Frobenius norm of difference
        return torch.mean(torch.sum((jacobians_flat - R_closest) ** 2, dim=(1, 2)))
