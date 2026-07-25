"""Mapping error strategies: unidirectional (forward-only) and bidirectional."""

from typing import Tuple
from torch import Tensor


class UnidirectionalMappingError:
    """Forward-only mapping error: D(φ(S), T) and kinetic energy of forward trajectory.

    Used for per-pair registration (Milestone A). Computes data term and kinetic
    energy over the forward trajectory only.
    """

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
        data = data_term(fwd.end, target.points, tgt_w=target.weights)
        kinetic = fwd.kinetic_energy()
        return data, kinetic


class BidirectionalMappingError:
    """Bidirectional mapping error: D(φ(S), T) + D(φ⁻¹(T), S).

    Used for cohort registration (Milestone B). Computes data term and kinetic
    energy summed over both forward and backward trajectories (AD-SVFD Eq. loss function).
    """

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

        # Data term: forward distance + backward distance
        data = (
            data_term(fwd.end, target.points, tgt_w=target.weights)
            + data_term(bwd.end, source.points, tgt_w=source.weights)
        )

        # Kinetic energy: sum over both trajectories
        kinetic = fwd.kinetic_energy() + bwd.kinetic_energy()

        return data, kinetic
