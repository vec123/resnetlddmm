import os
import sys
import torch


from src.learning.models.folding_decoder import FoldingDecoder
from src.learning.models.group_encoder import GroupEncoder


def test_folding_decoder():
    print('=== test_folding_decoder ===')
    batch_size = 2
    latent_dim = 4
    num_samples = 16

    decoder = FoldingDecoder(num_samples=num_samples, latent_dim=latent_dim, n_freqs=2, verbose=False)
    latent = torch.randn(batch_size, latent_dim, dtype=torch.float32)

    output = decoder(latent)
    print('folding_decoder output shape:', output.shape)
    assert output.shape == (batch_size, num_samples, 3)

    output_manual = decoder(latent, manual_inv=True)
    print('folding_decoder manual_inv output shape:', output_manual.shape)
    assert output_manual.shape == output.shape


def test_group_encoder_rotation():
    print('=== test_group_encoder_rotation ===')
    layers_cfg = [{
        'in_irreps': '1x0e',
        'target_irreps': '1x0e + 1x1o',
        'spatial_sh_lmax': 1,
        'interaction_sh_lmax': 4,
    }]
    encoder = GroupEncoder(layers_cfg=layers_cfg, latent_dim=4,
                            output_irreps='4x0e + 2x1o', verbose=False)

    v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    v2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    rotation_matrix = encoder.get_rotation_matrix_from_two_vectors(v1, v2)

    print('rotation_matrix shape:', rotation_matrix.shape)
    assert rotation_matrix.shape == (2, 3, 3)

    identity = torch.eye(3, dtype=rotation_matrix.dtype, device=rotation_matrix.device).unsqueeze(0)
    product = torch.matmul(rotation_matrix, rotation_matrix.transpose(-2, -1))
    print('rotation_matrix orthonormal check:', product)
    assert torch.allclose(product, identity, atol=1e-5)

def test_group_encoder_forward():
    print('\n=== test_group_encoder_forward ===')
    from torch_geometric.data import Data

    # 1. Setup Configuration
    latent_dim = 4
    batch_size = 2
    num_nodes_per_graph = 10
    total_nodes = batch_size * num_nodes_per_graph

    layers_cfg = [{
        'in_irreps': '1x0e',
        'target_irreps': '2x0e + 2x1o',
        'spatial_sh_lmax': 1,
        'interaction_sh_lmax': 4,
    }]

    encoder = GroupEncoder(layers_cfg=layers_cfg, latent_dim=latent_dim,
                            output_irreps=f'{latent_dim}x0e + 2x1o', verbose=False)
    encoder.eval()  # Set to evaluation mode

    # 2. Mock Data assembled into the PyG ``Data`` graph the encoder now consumes.
    x = torch.randn(total_nodes, 1)                       # 1x0e node feature
    pos = torch.randn(total_nodes, 3)                     # xyz coordinates
    batch_idx = torch.cat([torch.zeros(num_nodes_per_graph),
                           torch.ones(num_nodes_per_graph)]).long()
    edge_index = torch.stack([
        torch.arange(total_nodes),
        torch.roll(torch.arange(total_nodes), 1),
    ], dim=0)
    graph = Data(x=x, pos=pos, edge_index=edge_index, batch=batch_idx)

    # 3. Forward Pass -- current API: encoder(graph, supergraph) -> EncoderOutput.
    with torch.no_grad():
        out = encoder(graph, None)

    # 4. Assertions on the EncoderOutput fields.
    print(f"mu shape: {out.mu.shape}")
    print(f"rotation shape: {out.rotation.shape}")
    print(f"translation shape: {out.translation.shape}")

    assert out.mu.shape == (batch_size, latent_dim)
    assert out.logvar.shape == (batch_size, latent_dim)
    assert out.rotation.shape == (batch_size, 3, 3)
    assert out.translation.shape == (batch_size, 3)

    print('test_group_encoder_forward passed successfully.')


if __name__ == '__main__':
    test_folding_decoder()
    test_group_encoder_rotation()
    test_group_encoder_forward()
    print('test_models.py completed successfully.')
