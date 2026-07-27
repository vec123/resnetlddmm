"""Mapping error strategies: unidirectional (forward-only) and bidirectional."""

from typing import Tuple, Optional
import torch
from torch import Tensor


class UnidirectionalMappingError:
    """Forward-only mapping error: D(φ(S), T) and kinetic energy of forward trajectory.

    Used for per-pair registration (Milestone A). Computes data term and kinetic
    energy over the forward trajectory only.

    Supports optional random subsampling of predicted points for computational efficiency
    on high-density shapes: predicted shape subsampled to subsample_pred_n points,
    target shape kept at full resolution.
    """

    def __init__(self, subsample_pred_n: Optional[int] = None):
        """Initialize mapping error.

        Args:
            subsample_pred_n: if not None, randomly subsample pred to this many points each call.
                             Target shape is kept at full resolution.
        """
        self.subsample_pred_n = subsample_pred_n

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
        fwd = flow(source.points, code)
        pred = fwd.end

        # Subsample predicted points if configured
        if self.subsample_pred_n is not None and self.subsample_pred_n > 0:
            pred = self._subsample_points(pred, self.subsample_pred_n)

        data = data_term(pred, target.points, tgt_w=target.weights)
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

    Supports optional random subsampling of predicted points for computational efficiency
    on high-density shapes: both forward and backward predicted shapes are subsampled
    independently, targets kept at full resolution.
    """

    def __init__(self, subsample_pred_n: Optional[int] = None):
        """Initialize mapping error.

        Args:
            subsample_pred_n: if not None, randomly subsample pred to this many points each call.
                             Target shapes are kept at full resolution.
        """
        self.subsample_pred_n = subsample_pred_n

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
        fwd = flow(source.points, code)
        bwd = flow.inverse(target.points, code)

        # Subsample predicted points if configured (forward and backward independently)
        pred_fwd = fwd.end
        pred_bwd = bwd.end
        if self.subsample_pred_n is not None and self.subsample_pred_n > 0:
            pred_fwd = self._subsample_points(pred_fwd, self.subsample_pred_n)
            pred_bwd = self._subsample_points(pred_bwd, self.subsample_pred_n)

        # Data term: forward distance + backward distance
        data = (
            data_term(pred_fwd, target.points, tgt_w=target.weights)
            + data_term(pred_bwd, source.points, tgt_w=source.weights)
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
