"""Shape code abstractions for amortised vs per-shape inference."""

import abc
from typing import Optional
import torch.nn as nn
from torch import Tensor


class ShapeCode(nn.Module, abc.ABC):
    """Provides per-shape conditioning codes.

    Maps a batch object to per-shape latent codes (None if no codes).
    The code_source decides HOW shapes get codes (auto-decoder table lookup,
    encoder forward pass, or none); the stepper never knows which.
    """

    @abc.abstractmethod
    def forward(self, batch) -> Optional[Tensor]:
        """Extract codes from batch.

        Args:
            batch: batch-like object with shape information (e.g., shape_ids)

        Returns:
            [B, n_z] tensor of per-shape codes, or None if no codes
        """

    @abc.abstractmethod
    def penalty(self) -> Optional[Tensor]:
        """Compute code regularisation term (e.g., L2 on embeddings).

        Returns:
            Scalar loss tensor, or None if no regularisation applies
        """
