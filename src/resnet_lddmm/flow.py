"""NeuralODEFlow: facade combining a velocity field with forward and inverse integrators."""

from typing import Optional
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.integrators import Integrator
from src.resnet_lddmm.trajectory import Trajectory


class NeuralODEFlow(nn.Module):
    """Deformation flow: field + two integrators (forward and inverse).

    Integrates a velocity field to produce a trajectory. Supports both forward
    deformation (source → target) and inverse (target → source).
    """

    def __init__(
        self,
        field: VelocityField,
        direct: Integrator,
        inverse: Integrator,
        num_steps: int = 10,
    ):
        """Initialize flow with field and integrators.

        Args:
            field: velocity field that maps positions to velocities
            direct: integrator for forward pass (source → target)
            inverse: integrator for inverse pass (target → source)
            num_steps: number of Euler steps (default 10)
        """
        super().__init__()
        self.field = field
        self.direct = direct
        self.inv = inverse
        self.num_steps = num_steps

    def forward(self, q0: Tensor, code: Optional[Tensor] = None) -> Trajectory:
        """Integrate forward: initial → deformed.

        Args:
            q0: [B, N, 3] initial positions
            code: [B, N_z] per-shape code (optional)

        Returns:
            Trajectory of forward deformation
        """
        return self.direct.integrate(q0, self.field, code, self.num_steps)

    def inverse(self, qT: Tensor, code: Optional[Tensor] = None) -> Trajectory:
        """Integrate backward: deformed → initial.

        Args:
            qT: [B, N, 3] deformed positions
            code: [B, N_z] per-shape code (optional)

        Returns:
            Trajectory of inverse deformation
        """
        return self.inv.integrate(qT, self.field, code, self.num_steps)
