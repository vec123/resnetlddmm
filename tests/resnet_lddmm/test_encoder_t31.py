"""Tests for EncoderCodes adapter (T31)."""

import sys
import pytest
import torch

from src.learning.registry import Registry


def _deps_available():
    """Check if encoder dependencies are available."""
    try:
        import e3nn  # noqa: F401
        import torch_geometric  # noqa: F401
        from src.learning.models.group_encoder import GroupEncoder  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


class TestLazyImportGuarantee:
    """Verify lazy-import guarantee: auto_decoder works even without e3nn.

    This test poisons sys.modules so e3nn/torch_geometric cannot be imported,
    then verifies auto_decoder still works. This ensures the registry stores
    strings (lazy) and only loads on demand.
    """

    def test_auto_decoder_works_without_e3nn(self):
        """Auto-decoder creation should not fail even if e3nn is 'missing'."""
        # Save original modules
        e3nn_backup = sys.modules.get('e3nn')
        torch_geo_backup = sys.modules.get('torch_geometric')

        try:
            # Poison e3nn/torch_geometric by making them raise on import
            class FakeModule:
                def __getattr__(self, name):
                    raise ImportError("e3nn/torch_geometric not available (poisoned for testing)")

            sys.modules['e3nn'] = None  # type: ignore
            sys.modules['torch_geometric'] = None  # type: ignore

            # Auto-decoder should still work (no import of e3nn)
            from src.resnet_lddmm.codes.auto_decoder import AutoDecoderCodes
            codes = AutoDecoderCodes(num_shapes=5, n_z=3)

            # Verify it's an nn.Module and works
            assert hasattr(codes, 'forward')
            assert hasattr(codes, 'penalty')
            assert hasattr(codes, 'codes')  # The embedding table

        finally:
            # Restore modules
            if e3nn_backup is None:
                sys.modules.pop('e3nn', None)
            else:
                sys.modules['e3nn'] = e3nn_backup

            if torch_geo_backup is None:
                sys.modules.pop('torch_geometric', None)
            else:
                sys.modules['torch_geometric'] = torch_geo_backup

    def test_registry_lazy_registration(self):
        """Verify registry stores strings (lazy) for encoder codes."""
        # Import registrations to populate the registry
        import src.resnet_lddmm.registrations  # noqa: F401

        # Check that auto_decoder and encoder are registered as strings
        assert ("code", "auto_decoder") in Registry._entries
        assert ("code", "encoder") in Registry._entries

        # The important part: entries are strings, not imported classes
        auto_decoder_target = Registry._entries[("code", "auto_decoder")]
        encoder_target = Registry._entries[("code", "encoder")]

        assert isinstance(auto_decoder_target, str)
        assert isinstance(encoder_target, str)
        assert auto_decoder_target.endswith(":AutoDecoderCodes")
        assert encoder_target.endswith(":EncoderCodes")


class TestEncoderCodesImportSkip:
    """Tests for EncoderCodes that require e3nn/torch_geometric.

    These tests are skipped if dependencies are missing.
    """

    @pytest.mark.skipif(
        not _deps_available(),
        reason="e3nn, torch_geometric, or GroupEncoder not available"
    )
    def test_encoder_codes_instantiation(self):
        """Verify EncoderCodes can be instantiated with mocked dependencies."""
        from src.resnet_lddmm.codes.encoder import EncoderCodes
        from tests.resnet_lddmm.test_encoder_t31 import MockGraphBuilder, MockEncoder

        graph_builder = MockGraphBuilder()
        encoder = MockEncoder(latent_dim=3)

        codes = EncoderCodes(graph_builder=graph_builder, encoder=encoder, n_z=3)

        assert codes.n_z == 3
        assert hasattr(codes, 'forward')
        assert hasattr(codes, 'penalty')

    @pytest.mark.skipif(
        not _deps_available(),
        reason="e3nn, torch_geometric, or GroupEncoder not available"
    )
    def test_encoder_codes_output_shape(self):
        """Verify EncoderCodes produces [B, n_z] output."""
        from src.resnet_lddmm.codes.encoder import EncoderCodes
        from src.learning.loader.loaders import CohortBatch
        from tests.resnet_lddmm.test_encoder_t31 import MockGraphBuilder, MockEncoder

        graph_builder = MockGraphBuilder()
        encoder = MockEncoder(latent_dim=5)
        codes = EncoderCodes(graph_builder=graph_builder, encoder=encoder, n_z=5)

        # Create a mock batch
        B, N = 3, 10
        batch = CohortBatch(
            points=torch.randn(B, N, 3),
            shape_ids=torch.tensor([0, 1, 2], dtype=torch.long),
            weights=None,
            faces=[None] * B,
        )

        # Forward pass
        z = codes(batch)

        assert z.shape == (B, 5)
        assert z.dtype == torch.float32

    @pytest.mark.skipif(
        not _deps_available(),
        reason="e3nn, torch_geometric, or GroupEncoder not available"
    )
    def test_encoder_codes_penalty_is_none(self):
        """Verify EncoderCodes.penalty() returns None (no embedding table)."""
        from src.resnet_lddmm.codes.encoder import EncoderCodes
        from tests.resnet_lddmm.test_encoder_t31 import MockGraphBuilder, MockEncoder

        graph_builder = MockGraphBuilder()
        encoder = MockEncoder(latent_dim=3)
        codes = EncoderCodes(graph_builder=graph_builder, encoder=encoder)

        penalty = codes.penalty()
        assert penalty is None

    @pytest.mark.skipif(
        not _deps_available(),
        reason="e3nn, torch_geometric, or GroupEncoder not available"
    )
    def test_encoder_codes_stores_encoder_output(self):
        """Verify EncoderCodes stores last encoder output for pose access (T32)."""
        from src.resnet_lddmm.codes.encoder import EncoderCodes
        from src.learning.loader.loaders import CohortBatch
        from tests.resnet_lddmm.test_encoder_t31 import MockGraphBuilder, MockEncoder

        graph_builder = MockGraphBuilder()
        encoder = MockEncoder(latent_dim=4)
        codes = EncoderCodes(graph_builder=graph_builder, encoder=encoder)

        batch = CohortBatch(
            points=torch.randn(2, 8, 3),
            shape_ids=torch.tensor([0, 1], dtype=torch.long),
            weights=None,
            faces=[None] * 2,
        )

        # After forward, _last should store the encoder output
        z = codes(batch)
        assert codes._last is not None
        assert hasattr(codes._last, 'rotation')  # EncoderOutput field
        assert hasattr(codes._last, 'translation')  # EncoderOutput field


# Mock classes for testing without full dependencies
class MockGraphBuilder:
    """Stub graph builder for testing."""

    def build(self, vertices, mask, rng, areas=None, normals=None):
        """Return dummy graph and supergraph."""
        import torch_geometric.data as tg_data

        # Create a minimal torch_geometric Data object
        num_nodes = vertices.shape[0]
        graph = tg_data.Data(
            x=torch.ones(num_nodes, 1),
            pos=vertices.clone(),
            edge_index=torch.tensor([[], []], dtype=torch.long),
            batch=mask.clone(),
        )

        supergraph = None
        return graph, supergraph


class MockEncoder:
    """Stub encoder that returns deterministic latents."""

    def __init__(self, latent_dim=5):
        self.latent_dim = latent_dim

    def __call__(self, graph, supergraph):
        """Return mock EncoderOutput with rotation/translation."""
        from src.learning.models.encoder_output import EncoderOutput

        B = int(graph.batch.max().item()) + 1 if hasattr(graph, 'batch') else 1
        z = torch.randn(B, self.latent_dim)
        rot = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        transl = torch.zeros(B, 3)

        return EncoderOutput(latent=z, rotation=rot, translation=transl)

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
