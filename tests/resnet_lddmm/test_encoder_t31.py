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


class TestPoseTransform:
    """Tests for FrameTransform with optional rotation/translation (T32)."""

    def test_frame_transform_without_pose(self):
        """Verify FrameTransform works without pose (backward compatibility)."""
        from src.resnet_lddmm.io import FrameTransform

        center = torch.tensor([0.5, 0.5, 0.5])
        scale = torch.tensor(1.0)
        transform = FrameTransform(center=center, scale=scale)

        # Normalized point
        point_norm = torch.tensor([[0.0, 0.0, 0.0]])

        # Invert: should give 0*1 + 0.5 = [0.5, 0.5, 0.5]
        point_world = transform.invert(point_norm)
        assert torch.allclose(point_world, torch.tensor([[0.5, 0.5, 0.5]]))

    def test_frame_transform_with_pose(self):
        """Verify FrameTransform applies rotation and translation correctly."""
        from src.resnet_lddmm.io import FrameTransform

        center = torch.tensor([0.0, 0.0, 0.0])
        scale = torch.tensor(1.0)

        # Identity rotation, translation [1, 0, 0]
        rotation = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
        translation = torch.tensor([[1.0, 0.0, 0.0]])  # [1, 3]

        transform = FrameTransform(center=center, scale=scale, rotation=rotation, translation=translation)

        # Normalized point [0, 0, 0]
        point_norm = torch.tensor([[0.0, 0.0, 0.0]])

        # Invert: denormalize, then translate -> [1, 0, 0]
        point_world = transform.invert(point_norm)
        assert torch.allclose(point_world, torch.tensor([[1.0, 0.0, 0.0]]))

    def test_frame_contract_with_learned_pose(self):
        """Frame contract: pose is learned with flow to improve alignment (T32)."""
        from src.resnet_lddmm.io import FrameTransform

        # Canonical points in [0, 1]: output of flow
        points_canonical = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]], dtype=torch.float32)

        # Encoder learns rotation and translation jointly with flow
        # For this test: identity (no rotation) and small translation
        rotation = torch.eye(3)
        translation = torch.tensor([0.1, 0.0, 0.0])

        center = torch.tensor([0.0, 0.0, 0.0])
        scale = torch.tensor(1.0)
        transform = FrameTransform(center=center, scale=scale, rotation=rotation, translation=translation)

        # After invert (apply learned pose), shape is transformed
        points_aligned = transform.invert(points_canonical)

        # Expected: points + translation (since rotation is identity)
        expected = points_canonical + translation
        assert torch.allclose(points_aligned, expected, atol=1e-6)


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

    @pytest.mark.skipif(
        not _deps_available(),
        reason="e3nn, torch_geometric, or GroupEncoder not available"
    )
    def test_encoder_codes_get_pose(self):
        """Verify EncoderCodes.get_pose() returns rotation/translation (T32)."""
        from src.resnet_lddmm.codes.encoder import EncoderCodes
        from src.learning.loader.loaders import CohortBatch
        from tests.resnet_lddmm.test_encoder_t31 import MockGraphBuilder, MockEncoder

        graph_builder = MockGraphBuilder()
        encoder = MockEncoder(latent_dim=3)
        codes = EncoderCodes(graph_builder=graph_builder, encoder=encoder)

        batch = CohortBatch(
            points=torch.randn(2, 8, 3),
            shape_ids=torch.tensor([0, 1], dtype=torch.long),
            weights=None,
            faces=[None] * 2,
        )

        # Before forward, get_pose() should return None, None
        rot, trans = codes.get_pose()
        assert rot is None and trans is None

        # After forward, get_pose() should return pose tensors
        z = codes(batch)
        rot, trans = codes.get_pose()
        assert rot is not None
        assert trans is not None
        assert rot.shape == (2, 3, 3)
        assert trans.shape == (2, 3)

    @pytest.mark.skipif(
        not _deps_available(),
        reason="e3nn, torch_geometric, or GroupEncoder not available"
    )
    def test_encoder_codes_with_pose_transform(self):
        """Verify EncoderCodes.with_pose_transform() folds pose into FrameTransform (T32)."""
        from src.resnet_lddmm.codes.encoder import EncoderCodes
        from src.resnet_lddmm.io import FrameTransform
        from src.learning.loader.loaders import CohortBatch
        from tests.resnet_lddmm.test_encoder_t31 import MockGraphBuilder, MockEncoder

        graph_builder = MockGraphBuilder()
        encoder = MockEncoder(latent_dim=3)
        codes = EncoderCodes(graph_builder=graph_builder, encoder=encoder)

        batch = CohortBatch(
            points=torch.randn(2, 8, 3),
            shape_ids=torch.tensor([0, 1], dtype=torch.long),
            weights=None,
            faces=[None] * 2,
        )

        # Forward to populate _last
        z = codes(batch)

        # Create base transform
        base_transform = FrameTransform(
            center=torch.tensor([0.5, 0.5, 0.5]),
            scale=torch.tensor(1.0)
        )

        # with_pose_transform should fold encoder pose in
        pose_transform = codes.with_pose_transform(base_transform)
        assert pose_transform.rotation is not None
        assert pose_transform.translation is not None
        assert pose_transform.center.equal(base_transform.center)
        assert pose_transform.scale.equal(base_transform.scale)


# Mock classes for testing without full dependencies
class MockGraphBuilder:
    """Stub graph builder honoring the real one's OUTPUT contract.

    What matters to everything downstream is ``graph.batch``: one entry per NODE
    naming the shape that node came from, over a flattened [B*N, 3] position
    array (see graphs.build_radius_graph). It is not the padding mask -- and
    storing the mask there, as this stub used to, made ``batch.max() + 1`` equal
    2 for every input, because EncoderCodes passes an all-True [B, N] mask. A
    3-shape batch then encoded as 2 shapes.
    """

    def build(self, vertices, mask, rng, areas=None, normals=None):
        """[B, N, 3] vertices + [B, N] mask -> a flat graph over the kept nodes."""
        import torch_geometric.data as tg_data

        B, N = vertices.shape[0], vertices.shape[1]
        positions = vertices.reshape(-1, 3)
        node_batch = torch.arange(B, device=vertices.device).repeat_interleave(N)

        keep = mask.reshape(-1)
        positions, node_batch = positions[keep], node_batch[keep]

        graph = tg_data.Data(
            x=torch.ones(positions.shape[0], 1),
            pos=positions.clone(),
            edge_index=torch.tensor([[], []], dtype=torch.long),
            batch=node_batch,
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
