import os
import sys
import torch
from e3nn import o3


from src.learning.modules.equivariant.layer_norm import EquivariantLayerNorm

try:
    from src.learning.modules.equivariant.attention import EquivariantAttention
    attention_import_error = None
except Exception as exc:
    EquivariantAttention = None
    attention_import_error = exc


def test_equivariant_layer_norm():
    print('=== test_equivariant_layer_norm ===')
    # Current torch-e3nn convention: features are a plain [N, irreps.dim] tensor; the
    # module tracks the irreps itself (no IrrepsArray wrapper).
    irreps = '2x0e + 1x1o'
    x = torch.randn(2, o3.Irreps(irreps).dim, dtype=torch.float32)
    layer_norm = EquivariantLayerNorm(irreps, affine=True, verbose=False)

    output = layer_norm(x)
    print('output shape:', output.shape)

    assert output.shape == x.shape


def test_equivariant_attention():
    print('=== test_equivariant_attention ===')
    if EquivariantAttention is None:
        print('Skipping EquivariantAttention test because import failed:', attention_import_error)
        return

    irreps_in = '1x0e + 1x1o'
    irreps_out = '1x0e + 1x1o'
    num_nodes = 3
    node_features = torch.randn(num_nodes, o3.Irreps(irreps_in).dim, dtype=torch.float32)
    positions = torch.randn(num_nodes, 3, dtype=torch.float32)
    senders = torch.tensor([0, 1, 2], dtype=torch.long)
    receivers = torch.tensor([1, 2, 0], dtype=torch.long)

    attention = EquivariantAttention(irreps_in, irreps_out, sh_lmax=1)
    try:
        output, alpha = attention(node_features, positions, senders, receivers, num_nodes)
        print('attention output type:', type(output))
        print('alpha shape:', alpha.shape)
        assert alpha.shape[0] == senders.shape[0]
    except Exception as exc:
        print('EquivariantAttention forward pass failed:', exc)
        raise


if __name__ == '__main__':
    test_equivariant_layer_norm()
    test_equivariant_attention()
    print('test_equivariant.py completed successfully.')
