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

        # Vectorized normal penalty: gather directions and normals, compute dot products in parallel
        # direction: [B, N, 3] - vector from target to pred
        b_idx = torch.arange(B, device=pred.device).unsqueeze(1)  # [B, 1]
        n_idx = torch.arange(N, device=pred.device).unsqueeze(0)  # [1, N]
        direction = dist_vec[b_idx, n_idx, pred_to_tgt_indices]  # [B, N, 3]

        # normal: [B, N, 3] - normals at nearest target points
        normal = normals[b_idx, pred_to_tgt_indices]  # [B, N, 3]

        # Normalize both
        dir_norm = torch.norm(direction, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
        normal_norm = torch.norm(normal, dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
        direction_normalized = direction / dir_norm  # [B, N, 3]
        normal_normalized = normal / normal_norm  # [B, N, 3]

        # Compute dot products and penalty
        dot_products = (direction_normalized * normal_normalized).sum(dim=-1)  # [B, N]
        normal_penalty = torch.clamp(dot_products, min=0.0).mean()  # scalar

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

    Point weights are normalised to probability measures before being handed to
    geomloss, which solves BALANCED transport and therefore needs both sides to
    carry equal mass; see _as_measure.

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

            loss_b = loss_fn(
                self._as_measure(None if pred_w is None else pred_w[b], N, p_b),
                p_b,
                self._as_measure(None if tgt_w is None else tgt_w[b], M, t_b),
                t_b,
            )
            loss_total = loss_total + loss_b

        return loss_total / B

    @staticmethod
    def _as_measure(weights, count, like):
        """Per-point weights as a probability measure: non-negative, summing to one.

        geomloss solves BALANCED optimal transport, so both arguments must carry the
        same total mass. Passing raw ``ones(N)`` gives mass N rather than 1, which
        rescales the loss by that mass -- and is outright ill-posed once N != M, as
        happens whenever the two clouds are subsampled to different sizes.
        Normalising also makes the value independent of the point count, so a run is
        comparable across subsample_M settings.

        Args:
            weights: [K] per-point weights, or None for uniform
            count: K, the number of points
            like: tensor supplying device and dtype

        Returns:
            [K] non-negative weights summing to 1
        """
        uniform = torch.full((count,), 1.0 / count, device=like.device, dtype=like.dtype)
        if weights is None:
            return uniform

        w = weights.to(dtype=like.dtype).clamp_min(0)
        total = w.sum()
        # Degenerate input (all zero, or negative before clamping) would divide by ~0
        # and hand geomloss a non-finite measure; fall back rather than propagate it.
        if not torch.isfinite(total) or total <= 0:
            return uniform
        return w / total


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
    """Penalizes non-isometric deformation, per Euler step, along the trajectory.

    The quantity being judged is the jacobian of the STEP MAP the integrator
    actually applies, ``F = I + dt J``, where ``J = dv/dx``. Not J itself: the
    deformation is ``x + dt v(x)``, so an isometric flow is one where F is a
    rotation, which for small dt means the SYMMETRIC part of J vanishes (the
    Killing-field condition). Penalizing J directly would have the objective
    backwards -- it is minimized by a large velocity gradient, and maximal at the
    identity deformation.

    Every penalty is divided by dt^2, which makes it a RATE and therefore
    independent of ``field.num_steps``: the same weight means the same thing when
    K changes.

    Supported loss types (rate form, up to O(dt) corrections):
    - "strain":     ||sym(J)||_F^2 -- the strain-rate tensor; the cheapest, and
                    the linearization of the one below (which it equals times 4,
                    as dt -> 0). EXACTLY zero for any rigid flow, which makes it
                    the default choice.
    - "orthogonal": ||F^T F - I||_F^2 / dt^2 -- the exact condition on the
                    discrete step map, but see the floor below.
    - "det":        ((det F - 1)/dt)^2 -- volume only; permits shear. Same floor.

    The floor: an explicit Euler step of a genuine ROTATION is not exactly
    isometric. For a rotation generator W, F^T F = I - dt^2 W^2 and det F =
    1 + dt^2 |w|^2, so "orthogonal" and "det" both charge a rigid flow for the
    integrator's error -- dt^2 ||W^T W||_F^2 and (dt |w|^2)^2 respectively. That
    is not always negligible: at dt = 0.1 and |w| = 3.7 it measures 3.92 and 1.96,
    which is MORE than those same terms report for a 50%-per-unit-time uniform
    expansion (3.15 and 2.48). Under those two settings a fast rigid rotation
    looks worse than a real distortion. "strain" has no such term.

    NO singular-value decomposition anywhere, deliberately. The natural way to
    write "distance to the closest rotation" is ||F - UV^T||_F, but the SVD
    backward carries 1/(s_i^2 - s_j^2) terms that are infinite when singular
    values repeat -- which is not a rare edge case here but the TARGET state: a
    rigid F has s = (1,1,1), and a zero-initialized field gives s = (0,0,0). That
    formulation returns a NaN gradient exactly where the loss is minimal, which
    then propagates into the field weights and surfaces one step later as
    "linalg.svd: input contained non-finite values". The forms above are
    polynomial in F, so they are smooth everywhere including at the optimum.
    """

    def __init__(self, loss_type: str = "strain", sample_points: int = 64,
                 detach_trajectory: bool = True):
        super().__init__()
        if loss_type not in ("strain", "det", "orthogonal"):
            raise ValueError(f"loss_type must be 'strain', 'det', or 'orthogonal', got {loss_type}")
        self.loss_type = loss_type
        self.sample_points = sample_points
        self.detach_trajectory = detach_trajectory

    def forward(self, trajectory: Trajectory, field: VelocityField,
                code: Optional[Tensor] = None) -> Tensor:
        """Compute isometry penalty using finite differences for jacobian.

        Samples a subset of points to reduce computational cost.

        ``code`` must be the SAME per-shape code the trajectory was integrated
        with. A conditioned field's first layer takes [3 + n_z] inputs, so
        omitting it is a shape error rather than a milder unconditioned
        approximation.

        ``detach_trajectory`` decides how far the gradient reaches. True (default)
        detaches both the trajectory and the code, leaving this a pure FIELD
        regularizer. False lets it reach whatever produced the trajectory -- which
        under ``pose_application="pre"`` includes the POSE, and is the only way the
        "least non-rigid flow" criterion can actually train it. Detaching here is
        not a free choice under that wiring: it silently severs the coupling while
        the loss VALUE still appears to respond to the pose.
        """
        points = trajectory.points.detach() if self.detach_trajectory else trajectory.points
        K = points.shape[0] - 1
        N = points.shape[2]
        if code is not None and self.detach_trajectory:
            code = code.detach()

        # Sample points for efficiency
        n_sample = min(self.sample_points, N)
        indices = torch.randperm(N, device=points.device)[:n_sample]

        all_losses = []

        # For each step
        for k in range(K):
            # [B, n_sample, 3]. The batch dim is KEPT: conditioning broadcasts a
            # per-shape code along it and cannot recover it once flattened away.
            x_sampled = points[k][:, indices, :]

            # Create a wrapper for this step
            def field_fn(x_):
                return field(x_, code=code, step=k)

            # Compute loss for all sampled points at once (vectorized)
            all_losses.append(self._step_penalty(field_fn, x_sampled, trajectory.dt).reshape(-1))

        # Compute final loss as mean of all point losses
        return torch.cat(all_losses).mean()

    def _step_penalty(self, field_fn, x: Tensor, dt: float) -> Tensor:
        """Per-point isometry penalty for ONE Euler step.

        Args:
            x: [B, M, 3] points
            dt: integrator step size, so ``F = I + dt J`` is the step map's jacobian
        Returns:
            [B, M] loss values
        """
        jac = self._jacobians(field_fn, x)                     # [B, M, 3, 3]
        eye = torch.eye(3, device=x.device, dtype=x.dtype)

        # Every statistic below is invariant to transposing F, which is what lets
        # ``_jacobians`` return the transposed layout it finds convenient:
        # ||F^T F - I||_F = ||F F^T - I||_F (same eigenvalues), det F = det F^T,
        # and sym(J) is unchanged by construction.
        if self.loss_type == "strain":
            # ||sym(J)||_F^2 -- no matmul, no dt: already a rate.
            sym = 0.5 * (jac + jac.transpose(-1, -2))
            return (sym ** 2).sum(dim=(-2, -1))

        F = eye + dt * jac
        if self.loss_type == "det":
            return ((torch.det(F) - 1.0) / dt) ** 2

        # "orthogonal": the right Cauchy-Green deviation C - I, which vanishes iff
        # F is orthogonal. One 3x3 matmul per point -- cheaper than an SVD, and
        # differentiable at the optimum, which the SVD form was not.
        C = F.transpose(-1, -2) @ F
        return ((C - eye) ** 2).sum(dim=(-2, -1)) / (dt ** 2)

    @staticmethod
    def _jacobians(field_fn, x: Tensor) -> Tensor:
        """Finite-difference Jacobians of the velocity field at every point.

        Forward differences: 4 field evaluations per point (one base, three
        perturbed), issued as a SINGLE batched call. At these point counts the
        field is launch-bound rather than FLOP-bound -- the [B, M, 3] base call
        costs about as much as the [B, 3M, 3] perturbed one -- so fusing the two
        into [B, 4M, 3] is worth close to a factor of 2 on the whole term.

        The step is sqrt(machine eps), the textbook optimum for a forward
        difference -- ~1.5e-8 in float64, ~3.4e-4 in float32 -- which assumes
        points of order 1, as the normalized frame gives.

        Args:
            x: [B, M, 3] points, the batch dim being the SHAPE index
        Returns:
            [B, M, 3, 3] jacobians. Rows hold the jacobian COLUMNS (dv/dx_j), i.e.
            the block is J^T; every statistic taken from it is transpose-invariant
            (see _step_penalty), so the layout does not matter.
        """
        eps = torch.finfo(x.dtype).eps ** 0.5
        B, M, _ = x.shape

        eye = torch.eye(3, device=x.device, dtype=x.dtype)     # [3, 3]
        # [B, M, 4, 3]: the base point followed by its three perturbations.
        stacked = torch.cat([x.unsqueeze(-2), x.unsqueeze(-2) + eps * eye], dim=-2)

        # Flattened to [B, 4M, 3], NOT [B*4M, 3]: the leading dim has to stay the
        # shape index or a conditioned field broadcasts the wrong code.
        v = field_fn(stacked.reshape(B, 4 * M, 3)).reshape(B, M, 4, 3)
        return (v[..., 1:, :] - v[..., :1, :]) / eps


