"""TimeVaryingField: ResNet-LDDMM velocity field (one block per time step)."""

import torch
import torch.nn as nn
from torch import Tensor

from src.resnet_lddmm.fields.base import VelocityField
from src.resnet_lddmm.fields.blocks import VelocityBlock
from src.resnet_lddmm.conditioning.base import Conditioning, NoConditioning


class TimeVaryingField(VelocityField):
    """ResNet-LDDMM: block l is time step l.
    """

    def __init__(
        self,
        num_blocks: int = 10,
        width: int = 512,
        activation: str = "relu",
        conditioning: Conditioning | None = None,
    ):
        super().__init__()
        self.conditioning = conditioning or NoConditioning()
        in_dim = 3 + self.conditioning.dim
        self.blocks = nn.ModuleList(
            VelocityBlock(in_dim, width, activation) for _ in range(num_blocks)
        )

    def forward(
        self, x: Tensor, step: int | None = None, code: Tensor | None = None
    ) -> Tensor:
        """Compute velocity using the block for the given step.

        Args:
            x: [B, N, 3] point positions
            step: which block to use (must be provided for TimeVaryingField)
            code: [B, N_z] per-shape code (ignored by NoConditioning)

        Returns:
            [B, N, 3] velocity
        """
        # Get per-point conditioning features (None under NoConditioning)
        zbar = self.conditioning(x, code)
        # Concatenate position with conditioning if present
        h = x if zbar is None else torch.cat([x, zbar], dim=-1)
        # Apply the block for this step
        return self.blocks[step](h)
