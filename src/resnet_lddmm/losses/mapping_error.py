"""Mapping error strategies: unidirectional (forward-only) and bidirectional."""

from typing import Tuple, Optional
import torch
from torch import Tensor
from src.transforms.group_transforms import SE3_transform


class UnidirectionalMappingError:
    """Forward-only mapping error: D(φ(T), S) and kinetic energy of forward trajectory.

    Used for per-pair registration (Milestone A). Computes data term and kinetic
    energy over the forward trajectory only. Flow deforms template to match sample.

    Supports optional random subsampling for computational efficiency on high-density shapes:
    both template and sample are subsampled to subsample_n points before data-term computation,
    but flow is only computed on subsampled template. Tracks full and subsampled vertex counts.
    Optionally computes and stores full trajectory when save_full=True.
    """

    def __init__(self, subsample_n: Optional[int] = None, save_full: bool = False):
        """Initialize mapping error.

        Args:
            subsample_n: if not None, randomly subsample source to this many points before flow.
                        Target shape is kept at full resolution.
            save_full: if True, also compute and store trajectory on full source points
        """
        self.subsample_n = subsample_n
        self.save_full = save_full
        self.last_full_vertices = None  # Track full vertex count for logging
        self.last_subsample_vertices = None
        self.last_fwd_traj = None  # Store trajectory to avoid recomputation
        self.last_fwd_traj_full = None  # Store full trajectory if save_full=True

    def __call__(self, flow, data_term, template, sample, code, encoder_pose=None) -> Tuple[Tensor, Tensor]:
        """Compute unidirectional mapping error.

        Args:
            flow: NeuralODEFlow instance
            data_term: DataTerm instance
            template: batch with .points [B,N,3] (canonical reference, where flow starts)
            sample: batch with .points [B,M,3] (augmented input, encoder input)
            code: [B, n_z] or None
            encoder_pose: Optional (rotation [B,3,3], translation [B,3]) tuple, or None

        Returns:
            (data_loss, kinetic_energy) tuple of scalars
        """
        # Track full template vertices
        self.last_full_vertices = template.points.shape[1]

        # Subsample template points before flow if configured
        template_points = template.points
        if self.subsample_n is not None and self.subsample_n > 0:
            template_points = self._subsample_points(template_points, self.subsample_n)
            self.last_subsample_vertices = template_points.shape[1]
        else:
            self.last_subsample_vertices = self.last_full_vertices

        # Broadcast template to match code batch size (cohort: template [1,N,3], code [B,n_z])
        if template_points.shape[0] == 1 and code is not None and code.shape[0] > 1:
            template_points = template_points.expand(code.shape[0], -1, -1)

        # Flow on (possibly subsampled) template points
        fwd = flow(template_points, code)
        self.last_fwd_traj = fwd  # Store for reuse (avoid double-computation in pair.py)
        pred = fwd.end

        # Apply encoder pose if provided (transforms pred to augmented frame for comparison)
        if encoder_pose is not None and (encoder_pose[0] is not None or encoder_pose[1] is not None):
            rotation, translation = encoder_pose
            pred = self._apply_encoder_pose(pred, rotation, translation)

        # Optionally compute full trajectory for export if save_full=True
        if self.save_full and self.subsample_n is not None and self.subsample_n > 0:
            template_full = template.points
            if template_full.shape[0] == 1 and code is not None and code.shape[0] > 1:
                template_full = template_full.expand(code.shape[0], -1, -1)
            self.last_fwd_traj_full = flow(template_full, code)
        else:
            self.last_fwd_traj_full = None

        # Subsample sample points for data-term computation (faster Chamfer on dense clouds)
        sample_points = sample.points
        sample_weights = sample.weights
        if self.subsample_n is not None and self.subsample_n > 0:
            sample_points = self._subsample_points(sample_points, self.subsample_n)
            # Subsample weights if present
            if sample_weights is not None:
                sample_weights = self._subsample_points(sample_weights.unsqueeze(-1), self.subsample_n).squeeze(-1)

        data = data_term(pred, sample_points, tgt_w=sample_weights)
        kinetic = fwd.kinetic_energy()
        return data, kinetic

    @staticmethod
    def _subsample_points(points: Tensor, n_subsample: int) -> Tensor:
        """Randomly subsample points.

        Args:
            points: [B, N, 3] point tensor
            n_subsample: number of points to keep

        Returns:
            Subsampled [B, n_subsample, 3] tensor (or input if N <= n_subsample)
        """
        B, N, D = points.shape
        if N <= n_subsample:
            return points

        # Randomly select indices (same for all batch elements)
        indices = torch.randperm(N, device=points.device)[:n_subsample]
        return points[:, indices, :]

    @staticmethod
    def _apply_encoder_pose(points: Tensor, rotation: Optional[Tensor],
                            translation: Optional[Tensor]) -> Tensor:
        """Apply SE(3) transformation: points_out = R @ points + t.

        Args:
            points: [B, N, 3] or [B*N, 3]
            rotation: [B, 3, 3] or None
            translation: [B, 3] or None

        Returns:
            [B, N, 3] or [B*N, 3] transformed points
        """
        is_batched = points.ndim == 3
        if is_batched:
            B, N, D = points.shape
            flat_points = points.reshape(B * N, 3)
        else:
            flat_points = points

        if rotation is None and translation is None:
            return points

        # Infer batch size from rotation if available
        if rotation is not None:
            B = rotation.shape[0]
        else:
            B = translation.shape[0]

        n_node = torch.full((B,), flat_points.shape[0] // B, dtype=torch.long, device=points.device)

        rot = rotation if rotation is not None else torch.eye(3, device=points.device).unsqueeze(0).expand(B, -1, -1)
        trans = translation if translation is not None else torch.zeros(B, 3, device=points.device)

        transformed = SE3_transform(flat_points, n_node, rot, trans)

        if is_batched:
            transformed = transformed.reshape(B, N, 3)
        return transformed


class BidirectionalMappingError:
    """Bidirectional mapping error: D(φ(T), S) + D(φ⁻¹(S), T).

    Used for cohort registration (Milestone B). Computes data term and kinetic
    energy summed over both forward and backward trajectories (AD-SVFD Eq. loss function).
    Forward: deform template to match sample. Backward: inverse flow from sample.

    Supports optional random subsampling for computational efficiency on high-density shapes:
    both template and sample are subsampled to subsample_n points before data-term computation,
    but flows are computed on subsampled points. Tracks full and subsampled vertex counts.
    Optionally computes and stores full trajectories when save_full=True.
    """

    def __init__(self, subsample_n: Optional[int] = None, save_full: bool = False):
        """Initialize mapping error.

        Args:
            subsample_n: if not None, randomly subsample source and target to this many points each call
                        before their respective flows.
            save_full: if True, also compute and store trajectories on full source/target points
        """
        self.subsample_n = subsample_n
        self.save_full = save_full
        self.last_full_source_vertices = None
        self.last_full_target_vertices = None
        self.last_subsample_vertices = None
        self.last_fwd_traj = None  # Store trajectory to avoid recomputation
        self.last_bwd_traj = None
        self.last_fwd_traj_full = None  # Store full trajectories if save_full=True
        self.last_bwd_traj_full = None

    def __call__(self, flow, data_term, template, sample, code) -> Tuple[Tensor, Tensor]:
        """Compute bidirectional mapping error.

        Args:
            flow: NeuralODEFlow instance (must support .inverse())
            data_term: DataTerm instance
            template: batch with .points [B,N,3] (canonical reference)
            sample: batch with .points [B,M,3] (augmented input)
            code: [B, n_z] or None

        Returns:
            (data_loss, kinetic_energy) tuple where:
            - data_loss = D(φ(T), S) + D(φ⁻¹(S), T)
            - kinetic_energy = KE(forward) + KE(backward)
        """
        # Track full vertex counts
        self.last_full_source_vertices = template.points.shape[1]
        self.last_full_target_vertices = sample.points.shape[1]

        # Subsample template and sample before flow if configured
        template_points = template.points
        sample_points = sample.points
        template_weights = template.weights
        sample_weights = sample.weights

        if self.subsample_n is not None and self.subsample_n > 0:
            template_points = self._subsample_points(template_points, self.subsample_n)
            sample_points = self._subsample_points(sample_points, self.subsample_n)
            # Subsample weights if present
            if template_weights is not None:
                template_weights = self._subsample_points(template_weights.unsqueeze(-1), self.subsample_n).squeeze(-1)
            if sample_weights is not None:
                sample_weights = self._subsample_points(sample_weights.unsqueeze(-1), self.subsample_n).squeeze(-1)
            self.last_subsample_vertices = self.subsample_n
        else:
            self.last_subsample_vertices = None

        # Broadcast template/sample to match code batch size (cohort: one is [1,N,3], code is [B,n_z])
        if template_points.shape[0] == 1 and code is not None and code.shape[0] > 1:
            template_points = template_points.expand(code.shape[0], -1, -1)
        if sample_points.shape[0] == 1 and code is not None and code.shape[0] > 1:
            sample_points = sample_points.expand(code.shape[0], -1, -1)

        # Flow on (possibly subsampled) template and sample
        fwd = flow(template_points, code)
        bwd = flow.inverse(sample_points, code)
        self.last_fwd_traj = fwd  # Store for reuse (avoid double-computation in cohort.py)
        self.last_bwd_traj = bwd

        # Optionally compute full trajectories for export if save_full=True
        if self.save_full and self.subsample_n is not None and self.subsample_n > 0:
            template_full = template.points
            sample_full = sample.points
            if template_full.shape[0] == 1 and code is not None and code.shape[0] > 1:
                template_full = template_full.expand(code.shape[0], -1, -1)
            if sample_full.shape[0] == 1 and code is not None and code.shape[0] > 1:
                sample_full = sample_full.expand(code.shape[0], -1, -1)
            self.last_fwd_traj_full = flow(template_full, code)
            self.last_bwd_traj_full = flow.inverse(sample_full, code)
        else:
            self.last_fwd_traj_full = None
            self.last_bwd_traj_full = None

        # Data term: forward distance + backward distance (both on subsampled points)
        data = (
            data_term(fwd.end, sample_points, tgt_w=sample_weights)
            + data_term(bwd.end, template_points, tgt_w=template_weights)
        )

        # Kinetic energy: sum over both trajectories
        kinetic = fwd.kinetic_energy() + bwd.kinetic_energy()

        return data, kinetic

    @staticmethod
    def _subsample_points(points: Tensor, n_subsample: int) -> Tensor:
        """Randomly subsample points.

        Args:
            points: [B, N, 3] point tensor
            n_subsample: number of points to keep

        Returns:
            Subsampled [B, n_subsample, 3] tensor (or input if N <= n_subsample)
        """
        B, N, D = points.shape
        if N <= n_subsample:
            return points

        # Randomly select indices (same for all batch elements)
        indices = torch.randperm(N, device=points.device)[:n_subsample]
        return points[:, indices, :]
