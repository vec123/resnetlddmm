"""Data term abstractions for shape registration loss."""

import abc
import torch
import torch.nn as nn
from torch import Tensor

from src.learning.losses.losses import chamfer_loss


class DataTerm(abc.ABC):
    """Measures mismatch between predicted and target point clouds.

    Maps [B,N,3] predicted points and [B,M,3] target points to a scalar loss.
    Subclasses decide how to combine these with optional weights and normals.
    """

    @abc.abstractmethod
    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Tensor | None = None,
        tgt_w: Tensor | None = None,
        normals: Tensor | None = None,
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
    """

    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Tensor | None = None,
        tgt_w: Tensor | None = None,
        normals: Tensor | None = None,
    ) -> Tensor:
        """Compute Chamfer distance with all-ones mask.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar Chamfer loss
        """
        # Create all-ones mask for unpadded target cloud
        mask = torch.ones(
            target.shape[:2], dtype=torch.bool, device=target.device
        )
        return chamfer_loss(pred, target, mask)


class L2Data(DataTerm):
    """Mean squared pointwise error for corresponded point clouds.

    Assumes points are in correspondence by index (e.g., template points
    matched to target points at the same position). Simpler than Chamfer
    but requires pre-established correspondence.
    """

    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w: Tensor | None = None,
        tgt_w: Tensor | None = None,
        normals: Tensor | None = None,
    ) -> Tensor:
        """Compute mean squared error between matched clouds.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, N, 3] target positions (same N as pred)
            pred_w, tgt_w, normals: ignored (kept for protocol compatibility)

        Returns:
            Scalar MSE loss
        """
        return (pred - target).pow(2).sum(-1).mean()
