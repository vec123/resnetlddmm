import os
import sys
import torch

from src.learning.losses.losses import (
    kl_divergence_loss,
    geometric_clustering_loss,
    laplacian_loss,
    chamfer_loss,
    combined_surface_loss,
)


def test_losses():
    print('=== test_losses ===')
    torch.manual_seed(0)

    mean = torch.zeros(2, 3)
    logvar = torch.zeros(2, 3)
    kl = kl_divergence_loss(mean, logvar)
    print('kl_divergence_loss:', kl.item())
    assert torch.isfinite(kl)
    assert abs(kl.item()) < 1e-6

    logits = torch.tensor([[2.0, -1.0], [-0.5, 0.5]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.long)
    gc = geometric_clustering_loss(logits, edge_index, smoothness_weight=1.0, balance_weight=0.1, entropy_weight=0.1)
    print('geometric_clustering_loss:', gc.item())
    assert gc.item() > 0

    # laplacian_loss takes a flat [B, N, 3] cloud (N a perfect square) and grids it
    # internally; N=4 -> a 2x2 grid.
    pred = torch.arange(12.0, dtype=torch.float32).view(1, 4, 3)
    lap = laplacian_loss(pred)
    print('laplacian_loss:', lap.item())
    assert lap.item() >= 0

    # pred has a SQUARE number of points (4 -> 2x2 grid): combined_surface_loss adds
    # laplacian_loss, which grids the prediction and requires N to be a perfect square
    # (as the FoldingDecoder always emits).
    pred_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=torch.float32)
    target_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32)
    target_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    ch = chamfer_loss(pred_pos, target_pos, target_mask)
    print('chamfer_loss:', ch.item())
    assert ch.item() >= 0

    combined = combined_surface_loss(pred_pos, target_pos, target_mask, laplacian_weight=0.1)
    print('combined_surface_loss:', combined.item())
    assert combined.item() >= ch.item()


if __name__ == '__main__':
    test_losses()
    print('test_losses.py completed successfully.')
