"""ODE integrators for neural flows."""

import abc
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.trajectory import Trajectory


class Integrator(abc.ABC):
    """Abstract integrator: steps a velocity field to produce a trajectory."""

    @abc.abstractmethod
    def integrate(
        self,
        q0: Tensor,
        field: VelocityField,
        code: Optional[Tensor],
        num_steps: int,
    ) -> Trajectory:
        """Integrate the velocity field from initial position.

        Args:
            q0: [B, N, 3] initial positions
            field: velocity field
            code: [B, N_z] per-shape code (or None)
            num_steps: number of Euler steps

        Returns:
            Trajectory with points [K+1, B, N, 3] and velocities [K, B, N, 3]
        """


class ForwardEuler(Integrator):
    """Forward Euler integration: q_{k+1} = q_k + dt * v_k.

    Time step dt = 1/num_steps is computed here — the sole owner of this invariant.
    """

    def integrate(
        self,
        q0: Tensor,
        field: VelocityField,
        code: Optional[Tensor],
        num_steps: int,
    ) -> Trajectory:
        """Integrate field from q0 through num_steps forward Euler steps.

        Args:
            q0: [B, N, 3] initial positions
            field: velocity field
            code: [B, N_z] per-shape code (or None)
            num_steps: number of steps; dt = 1/num_steps

        Returns:
            Trajectory containing integrated points and velocities
        """
        dt = 1.0 / num_steps
        q = q0
        points = [q0]
        velocities = []

        for k in range(num_steps):
            v = field(q, step=k, code=code)
            q = q + dt * v
            points.append(q)
            velocities.append(v)

        return Trajectory(
            torch.stack(points),
            torch.stack(velocities),
            dt,
        )
