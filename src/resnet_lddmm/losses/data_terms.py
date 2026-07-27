"""Loss terms for shape registration: data fitting and flow regularization."""

import abc
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.learning.losses.losses import chamfer_loss
from src.resnet_lddmm.trajectory import Trajectory
from src.resnet_lddmm.fields.base import VelocityField


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

    Supports both scalar loss and per-point losses (for adaptive subsampling in T28).
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute Chamfer distance with all-ones mask.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar Chamfer loss if per_point=False, else per-point distances [B, M]
        """
        # Create all-ones mask for unpadded target cloud
        mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )

        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)
        dist_sq = dist_sq.masked_fill(~mask.unsqueeze(1), 1e6)

        if per_point:
            # Return per-point losses from target to pred (term2 of Chamfer)
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            return per_point_losses

        # Standard scalar loss
        term1 = dist_sq.min(dim=2)[0].mean()
        dist_t_to_p = dist_sq.min(dim=1)[0]
        term2 = (dist_t_to_p * mask).sum() / (mask.sum() + 1e-8)
        return term1 + term2


class L2Data(DataTerm):
    """Mean squared pointwise error for corresponded point clouds.

    Assumes points are in correspondence by index (e.g., template points
    matched to target points at the same position). Simpler than Chamfer
    but requires pre-established correspondence.

    Supports both scalar loss and per-point losses (for adaptive subsampling in T28).
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute mean squared error between matched clouds.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, N, 3] target positions (same N as pred)
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, N]; if False, return scalar

        Returns:
            Scalar MSE loss if per_point=False, else per-point MSE [B, N]
        """
        per_point_losses = (pred - target).pow(2).sum(-1)  # [B, N]

        if per_point:
            return per_point_losses

        return per_point_losses.mean()


class WeightedCDData(DataTerm):
    """Weighted Chamfer distance using per-point target areas (T33).

    Similar to CDData but weights the target→pred term by target point weights
    (typically surface areas). Reduces to standard Chamfer with uniform weights.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute weighted Chamfer distance.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w: [B, N] weights for pred (ignored)
            tgt_w: [B, M] per-point target weights (surface areas); uniform if None
            normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar weighted Chamfer loss if per_point=False, else per-point losses [B, M]
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape

        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            # Return per-point losses from target to pred, weighted by tgt_w
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            if tgt_w is not None:
                per_point_losses = per_point_losses * tgt_w
            return per_point_losses

        # Term 1: pred → target (unweighted, average nearest target distance)
        term1 = dist_sq.min(dim=2)[0].mean()

        # Term 2: target → pred (weighted by tgt_w if provided)
        tgt_to_pred = dist_sq.min(dim=1)[0]  # [B, M]
        if tgt_w is not None:
            # Normalize weights to avoid changing overall scale
            w_normalized = tgt_w / (tgt_w.mean(dim=1, keepdim=True) + 1e-8)
            term2 = (tgt_to_pred * w_normalized).mean()
        else:
            term2 = tgt_to_pred.mean()

        return term1 + term2


class PCDData(DataTerm):
    """Point Cloud Distance: symmetric Chamfer-like metric without explicit masking (T33).

    Similar to CDData but computed more directly without mask operations.
    Equivalent to CDData on unpadded point clouds.
    """

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute point cloud distance (symmetric).

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar PCD loss if per_point=False, else per-point distances [B, M]
        """
        # Compute pairwise distances [B, N, M]
        dist_sq = torch.cdist(pred, target, p=2).pow(2)

        if per_point:
            # Return per-point losses from target to pred
            per_point_losses = dist_sq.min(dim=1)[0]  # [B, M]
            return per_point_losses

        # Symmetric: pred→target + target→pred
        term1 = dist_sq.min(dim=2)[0].mean()
        term2 = dist_sq.min(dim=1)[0].mean()
        return term1 + term2


class NCDData(DataTerm):
    """Normal-aware Chamfer distance: penalizes surfaces with wrong orientation (T33).

    Combines point-to-point distance with normal alignment penalty.
    Encourages normals to be oriented away from the matching pred points.
    """

    def __init__(self, normal_weight: float = 1.0):
        """Initialize NCD loss.

        Args:
            normal_weight: balance between geometric and normal penalties (default 1.0)
        """
        super().__init__()
        self.normal_weight = normal_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
        per_point: bool = False,
    ) -> Tensor:
        """Compute normal-aware Chamfer distance.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w, tgt_w: ignored
            normals: [B, M, 3] target surface normals; if None, reduces to CDData
            per_point: if True, return per-point losses [B, M]; if False, return scalar

        Returns:
            Scalar NCD loss if per_point=False, else per-point losses [B, M]
        """
        B, N, D = pred.shape
        B_tgt, M, D = target.shape

        # Compute pairwise distances [B, N, M]
        dist_vec = pred.unsqueeze(2) - target.unsqueeze(1)  # [B, N, M, 3]
        dist_sq = (dist_vec ** 2).sum(dim=3)  # [B, N, M]

        # If no normals, fall back to standard Chamfer
        if normals is None:
            if per_point:
                return dist_sq.min(dim=1)[0]
            term1 = dist_sq.min(dim=2)[0].mean()
            term2 = dist_sq.min(dim=1)[0].mean()
            return term1 + term2

        # Compute normal alignment penalty: penalizes normals pointing towards pred
        # For each pred point, find nearest target and its normal
        pred_to_tgt_indices = dist_sq.min(dim=2)[1]  # [B, N]
        pred_to_tgt_dists = dist_sq.min(dim=2)[0]  # [B, N]

        # For normal penalty, compute alignment with nearest target normal
        normal_penalty = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

        for b in range(B):
            for n in range(N):
                m_idx = pred_to_tgt_indices[b, n]
                direction = dist_vec[b, n, m_idx]  # [3], points from target to pred
                normal = normals[b, m_idx]  # [3], should point away from pred

                # Normalize for stable dot product
                dir_norm = torch.norm(direction) + 1e-8
                normal_norm = torch.norm(normal) + 1e-8
                direction_normalized = direction / dir_norm
                normal_normalized = normal / normal_norm

                # Penalty: max(0, direction · normal) penalizes when normal points towards pred
                # We want normal to point AWAY from pred, so dot product should be negative
                dot_product = torch.dot(direction_normalized, normal_normalized)
                normal_penalty = normal_penalty + torch.clamp(dot_product, min=0.0)

        normal_penalty = normal_penalty / (B * N) if B * N > 0 else normal_penalty

        # Term 1: pred → target (point distance + normal penalty)
        term1 = (pred_to_tgt_dists.mean() + self.normal_weight * normal_penalty)

        # Term 2: target → pred (point distance only)
        tgt_to_pred_dists = dist_sq.min(dim=1)[0]  # [B, M]
        term2 = tgt_to_pred_dists.mean()

        if per_point:
            # Return per-point losses from target to pred (geometric only for per_point)
            return tgt_to_pred_dists

        return term1 + term2


class SinkhornData(DataTerm):
    """Sinkhorn optimal transport distance via geomloss (T34).

    Solves the optimal transport problem using Sinkhorn's algorithm.
    Recommended for articulated shapes (e.g., hands) where Chamfer can
    produce incorrect finger-to-finger matchings.

    Lazy-loads geomloss: auto-decoder configs work even without it installed.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto", **kwargs):
        """Initialize Sinkhorn loss.

        Args:
            p: Distance norm (default 2 for L2)
            blur: Sinkhorn blur radius (higher = smoother gradient, default 0.01)
            backend: geomloss backend ("auto", "keops", "torch", default "auto")
            **kwargs: ignored (compatibility with other data terms)
        """
        super().__init__()
        self.p = p
        self.blur = blur
        self.backend = backend
        self.loss_fn = None  # Lazy initialized

    def _get_loss_fn(self):
        """Lazy-load geomloss on first call."""
        if self.loss_fn is None:
            try:
                import geomloss
            except ImportError:
                raise ImportError(
                    "geomloss required for Sinkhorn distance. "
                    "Install with: pip install geomloss"
                )
            self.loss_fn = geomloss.SamplesLoss(
                "sinkhorn", p=self.p, blur=self.blur, backend=self.backend
            )
        return self.loss_fn

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Sinkhorn optimal transport distance.

        Args:
            pred: [B, N, 3] predicted point positions
            target: [B, M, 3] target point positions
            pred_w: [B, N] weights for pred (optional, passed to geomloss)
            tgt_w: [B, M] weights for target (optional, passed to geomloss)
            normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar Sinkhorn loss
        """
        loss_fn = self._get_loss_fn()

        B, N, D = pred.shape
        B_tgt, M, D = target.shape
        assert B == B_tgt, f"Batch size mismatch: {B} vs {B_tgt}"

        loss_total = torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
        for b in range(B):
            p_b = pred[b]  # [N, 3]
            t_b = target[b]  # [M, 3]

            # Compute Sinkhorn loss; weights can be None
            loss_b = loss_fn(
                pred_w[b] if pred_w is not None else torch.ones(N, device=pred.device),
                p_b,
                tgt_w[b] if tgt_w is not None else torch.ones(M, device=target.device),
                t_b,
            )
            loss_total = loss_total + loss_b

        return loss_total / B


class EMDData(DataTerm):
    """Earth Mover's Distance via Sinkhorn's algorithm (legacy alias for SinkhornData).

    Deprecated: use SinkhornData instead. Kept for backward compatibility.
    """

    def __init__(self, p: float = 2, blur: float = 0.01, backend: str = "auto"):
        """Initialize EMD loss (delegates to SinkhornData)."""
        super().__init__()
        self._sinkhorn = SinkhornData(p=p, blur=blur, backend=backend)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Optional[Tensor] = None,
        tgt_w: Optional[Tensor] = None,
        normals: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Earth Mover's Distance via Sinkhorn."""
        return self._sinkhorn(pred, target, pred_w=pred_w, tgt_w=tgt_w, normals=normals)


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
