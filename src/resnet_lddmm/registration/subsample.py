"""Adaptive subsampling of point correspondences based on loss (STEPS T28)."""

import torch
from torch import Tensor


def adaptive_subsample(per_point_losses: Tensor, M: int, a: float) -> Tensor:
    """Adaptive subsampling: keep top-a·M hard examples + uniform rest (STEPS T28).

    Selects point indices by combining:
    - Top a·M indices with highest per-point loss (hard examples)
    - Remaining (1-a)·M indices drawn uniformly at random

    Args:
        per_point_losses: [B, N] or [N] per-point losses
        M: total number of points to return
        a: fraction (0 ≤ a ≤ 1) of hard examples to keep

    Returns:
        Indices [M] (if input is [N]) or [B, M] (if input is [B, N])
        Sorted for stable subsampling.
    """
    if per_point_losses.dim() == 1:
        # Single batch case: [N]
        N = per_point_losses.shape[0]
        if M >= N:
            return torch.arange(N, device=per_point_losses.device)

        # Top-a·M hard examples
        n_hard = max(1, int(a * M))
        _, hard_indices = torch.topk(per_point_losses, n_hard)

        # Uniform rest (excluding hard indices)
        n_rest = M - n_hard
        hard_set = set(hard_indices.tolist())
        available = torch.tensor([i for i in range(N) if i not in hard_set], device=per_point_losses.device)
        perm = torch.randperm(available.shape[0], device=per_point_losses.device)
        rest_indices = available[perm[:n_rest]]

        # Combine and sort
        indices = torch.cat([hard_indices, rest_indices])
        return indices.sort()[0]

    elif per_point_losses.dim() == 2:
        # Batched case: [B, N]
        B, N = per_point_losses.shape
        if M >= N:
            return torch.arange(N, device=per_point_losses.device).unsqueeze(0).expand(B, -1)

        # Top-a·M hard examples per batch
        n_hard = max(1, int(a * M))
        _, hard_indices = torch.topk(per_point_losses, n_hard, dim=1)  # [B, n_hard]

        # Uniform rest per batch
        n_rest = M - n_hard
        all_indices = torch.arange(N, device=per_point_losses.device).unsqueeze(0).expand(B, -1)
        rest_batch = []
        for b in range(B):
            # Exclude hard indices for this batch
            mask = torch.ones(N, dtype=torch.bool, device=per_point_losses.device)
            mask[hard_indices[b]] = False
            available = all_indices[b, mask]
            # Sample n_rest from available
            perm = torch.randperm(available.shape[0], device=per_point_losses.device)
            sampled = available[perm[:n_rest]]
            rest_batch.append(sampled)
        rest_indices = torch.stack(rest_batch)  # [B, n_rest]

        # Combine and sort per batch
        indices = torch.cat([hard_indices, rest_indices], dim=1)  # [B, M]
        indices = indices.sort(dim=1)[0]
        return indices

    else:
        raise ValueError(f"per_point_losses must be 1D or 2D, got {per_point_losses.dim()}D")


class AdaptiveSubsampler:
    """Wraps a DataTerm to apply adaptive subsampling (STEPS T28).

    Usage:
        subsampler = AdaptiveSubsampler(data_term, M=2000, a=0.15)
        loss = subsampler(pred, target, ...)

    The subsampler extracts per-point losses, subsamples target points,
    and recomputes the scalar loss on the subsampled subset.
    """

    def __init__(self, data_term, M: int = 2000, a: float = 0.15):
        """Initialize subsampler.

        Args:
            data_term: DataTerm instance (must support per_point=True)
            M: target number of points after subsampling
            a: fraction of hard examples to keep (0 ≤ a ≤ 1)
        """
        self.data_term = data_term
        self.M = M
        self.a = a

    def __call__(
        self,
        pred: Tensor,
        target: Tensor,
        pred_w=None,
        tgt_w=None,
        normals=None,
    ) -> Tensor:
        """Apply adaptive subsampling and compute loss.

        Args:
            pred: [B, N, 3] predicted positions
            target: [B, M, 3] target positions
            pred_w, tgt_w, normals: optional weights/normals

        Returns:
            Scalar loss computed on subsampled points
        """
        # Get per-point losses
        per_point_losses = self.data_term(pred, target, pred_w, tgt_w, normals, per_point=True)

        # Subsample indices
        indices = adaptive_subsample(per_point_losses, self.M, self.a)

        # Subsample target and (if correspondence) pred
        if per_point_losses.dim() == 1:
            # Single batch: [N] -> subsample
            target_sub = target[indices]  # Assumes target is [N, 3] or [1, N, 3]
            pred_sub = pred
            if target.dim() == 3:
                target_sub = target[:, indices, :]
                # For correspondence-based losses (L2), subsample pred too
                if pred.shape[1] == target.shape[1]:
                    pred_sub = pred[:, indices, :]
        else:
            # Batched: [B, N] -> subsample per batch
            B = target.shape[0]
            target_sub = torch.stack([target[b, indices[b]] for b in range(B)])
            pred_sub = pred
            # For correspondence-based losses (L2), subsample pred too
            if pred.shape[1] == target.shape[1]:
                pred_sub = torch.stack([pred[b, indices[b]] for b in range(B)])

        # Subsample pred weights if present
        pred_w_sub = None
        if pred_w is not None:
            if pred_w.dim() == 1:
                pred_w_sub = pred_w[indices]
            else:
                B = pred_w.shape[0]
                pred_w_sub = torch.stack([pred_w[b, indices[b]] for b in range(B)])

        # Subsample normals if present
        normals_sub = None
        if normals is not None:
            if normals.dim() == 2:
                normals_sub = normals[indices]
            else:
                B = normals.shape[0]
                normals_sub = torch.stack([normals[b, indices[b]] for b in range(B)])

        # Recompute scalar loss on subsampled points
        return self.data_term(pred_sub, target_sub, pred_w_sub, None, normals_sub, per_point=False)
