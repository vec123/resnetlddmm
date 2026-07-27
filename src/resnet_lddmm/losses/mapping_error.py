"""Mapping error strategies: unidirectional (forward-only) and bidirectional."""

from typing import Tuple, Optional
import torch
from torch import Tensor


class UnidirectionalMappingError:
    """Forward-only mapping error: D(φ(S), T) and kinetic energy of forward trajectory.

    Used for per-pair registration (Milestone A). Computes data term and kinetic
    energy over the forward trajectory only.

    Supports optional random subsampling for computational efficiency on high-density shapes:
    both source and target are subsampled to subsample_n points before data-term computation,
    but flow is only computed on subsampled source. Tracks full and subsampled vertex counts.
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

    def __call__(self, flow, data_term, source, target, code) -> Tuple[Tensor, Tensor]:
        """Compute unidirectional mapping error.

        Args:
            flow: NeuralODEFlow instance
            data_term: DataTerm instance
            source: batch with .points [B,N,3]
            target: batch with .points [B,M,3]
            code: [B, n_z] or None

        Returns:
            (data_loss, kinetic_energy) tuple of scalars
        """
        # Track full source vertices
        self.last_full_vertices = source.points.shape[1]

        # Subsample source points before flow if configured
        source_points = source.points
        if self.subsample_n is not None and self.subsample_n > 0:
            source_points = self._subsample_points(source_points, self.subsample_n)
            self.last_subsample_vertices = source_points.shape[1]
        else:
            self.last_subsample_vertices = self.last_full_vertices

        # Flow on (possibly subsampled) source
        fwd = flow(source_points, code)
        self.last_fwd_traj = fwd  # Store for reuse (avoid double-computation in pair.py)
        pred = fwd.end

        # Optionally compute full trajectory for export if save_full=True
        if self.save_full and self.subsample_n is not None and self.subsample_n > 0:
            self.last_fwd_traj_full = flow(source.points, code)
        else:
            self.last_fwd_traj_full = None

        # Subsample target for data-term computation (faster Chamfer on dense clouds)
        target_points = target.points
        target_weights = target.weights
        if self.subsample_n is not None and self.subsample_n > 0:
            target_points = self._subsample_points(target_points, self.subsample_n)
            # Subsample weights if present
            if target_weights is not None:
                target_weights = self._subsample_points(target_weights.unsqueeze(-1), self.subsample_n).squeeze(-1)

        data = data_term(pred, target_points, tgt_w=target_weights)
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


class BidirectionalMappingError:
    """Bidirectional mapping error: D(φ(S), T) + D(φ⁻¹(T), S).

    Used for cohort registration (Milestone B). Computes data term and kinetic
    energy summed over both forward and backward trajectories (AD-SVFD Eq. loss function).

    Supports optional random subsampling for computational efficiency on high-density shapes:
    both source and target are subsampled to subsample_n points before data-term computation,
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

    def __call__(self, flow, data_term, source, target, code) -> Tuple[Tensor, Tensor]:
        """Compute bidirectional mapping error.

        Args:
            flow: NeuralODEFlow instance (must support .inverse())
            data_term: DataTerm instance
            source: batch with .points [B,N,3]
            target: batch with .points [B,M,3]
            code: [B, n_z] or None

        Returns:
            (data_loss, kinetic_energy) tuple where:
            - data_loss = D(φ(S), T) + D(φ⁻¹(T), S)
            - kinetic_energy = KE(forward) + KE(backward)
        """
        # Track full vertex counts
        self.last_full_source_vertices = source.points.shape[1]
        self.last_full_target_vertices = target.points.shape[1]

        # Subsample source and target before flow if configured
        source_points = source.points
        target_points = target.points
        source_weights = source.weights
        target_weights = target.weights

        if self.subsample_n is not None and self.subsample_n > 0:
            source_points = self._subsample_points(source_points, self.subsample_n)
            target_points = self._subsample_points(target_points, self.subsample_n)
            # Subsample weights if present
            if source_weights is not None:
                source_weights = self._subsample_points(source_weights.unsqueeze(-1), self.subsample_n).squeeze(-1)
            if target_weights is not None:
                target_weights = self._subsample_points(target_weights.unsqueeze(-1), self.subsample_n).squeeze(-1)
            self.last_subsample_vertices = self.subsample_n
        else:
            self.last_subsample_vertices = None

        # Flow on (possibly subsampled) source and target
        fwd = flow(source_points, code)
        bwd = flow.inverse(target_points, code)
        self.last_fwd_traj = fwd  # Store for reuse (avoid double-computation in cohort.py)
        self.last_bwd_traj = bwd

        # Optionally compute full trajectories for export if save_full=True
        if self.save_full and self.subsample_n is not None and self.subsample_n > 0:
            self.last_fwd_traj_full = flow(source.points, code)
            self.last_bwd_traj_full = flow.inverse(target.points, code)
        else:
            self.last_fwd_traj_full = None
            self.last_bwd_traj_full = None

        # Data term: forward distance + backward distance (both on subsampled points)
        data = (
            data_term(fwd.end, target_points, tgt_w=target_weights)
            + data_term(bwd.end, source_points, tgt_w=source_weights)
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
