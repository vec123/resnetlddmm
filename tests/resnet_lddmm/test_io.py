"""Tests for I/O functions (STEPS T13+)."""

import torch
import pytest

from src.resnet_lddmm.io import Shape, load_shape, FrameTransform, joint_normalize


class TestShapeDataclass:
    """Tests for Shape dataclass."""

    def test_shape_creation(self):
        """Verify Shape can be instantiated."""
        points = torch.randn(1, 10, 3)
        faces = torch.randint(0, 10, (5, 3))
        shape = Shape(points=points, faces=faces)

        assert shape.points.shape == (1, 10, 3)
        assert shape.faces.shape == (5, 3)
        assert shape.weights is None
        assert shape.normals is None

    def test_shape_with_weights_and_normals(self):
        """Verify Shape stores weights and normals."""
        points = torch.randn(1, 10, 3)
        faces = torch.randint(0, 10, (5, 3))
        weights = torch.ones(1, 10)
        normals = torch.randn(1, 10, 3)

        shape = Shape(points=points, faces=faces, weights=weights, normals=normals)

        assert shape.weights.shape == (1, 10)
        assert shape.normals.shape == (1, 10, 3)


class TestLoadShape:
    """Tests for load_shape function."""

    def test_load_template_shape(self):
        """Verify template.vtp loads with correct shape."""
        shape = load_shape("data/hand/template.vtp")

        # Expected from spec: 252 points, 500 faces
        assert shape.points.shape[1] == 252, f"Expected 252 points, got {shape.points.shape[1]}"
        assert shape.faces.shape[0] == 500, f"Expected 500 faces, got {shape.faces.shape[0]}"

    def test_load_shape_points_shape(self):
        """Verify points are [1, N, 3]."""
        shape = load_shape("data/hand/template.vtp")
        assert len(shape.points.shape) == 3
        assert shape.points.shape[0] == 1

    def test_load_shape_faces_shape(self):
        """Verify faces are [F, 3]."""
        shape = load_shape("data/hand/template.vtp")
        assert len(shape.faces.shape) == 2
        assert shape.faces.shape[1] == 3

    def test_load_shape_points_dtype(self):
        """Verify points are float32."""
        shape = load_shape("data/hand/template.vtp")
        assert shape.points.dtype == torch.float32

    def test_load_shape_faces_dtype(self):
        """Verify faces are int64."""
        shape = load_shape("data/hand/template.vtp")
        assert shape.faces.dtype == torch.int64

    def test_load_shape_weights_optional(self):
        """Verify weights are optional (can be None)."""
        shape = load_shape("data/hand/template.vtp")
        # Template may or may not have area field; just verify type
        assert shape.weights is None or isinstance(shape.weights, torch.Tensor)

    def test_load_shape_normals_optional(self):
        """Verify normals are optional (can be None)."""
        shape = load_shape("data/hand/template.vtp")
        # Template may or may not have normal field; just verify type
        assert shape.normals is None or isinstance(shape.normals, torch.Tensor)

    def test_load_shape_face_indices_valid(self):
        """Verify face indices are within valid range."""
        shape = load_shape("data/hand/template.vtp")
        num_points = shape.points.shape[1]
        assert shape.faces.max() < num_points
        assert shape.faces.min() >= 0


class TestFrameTransform:
    """Tests for FrameTransform (STEPS T14)."""

    def test_frame_transform_is_frozen(self):
        """Verify FrameTransform is immutable."""
        center = torch.zeros(3)
        scale = torch.tensor(1.0)
        transform = FrameTransform(center=center, scale=scale)

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            transform.center = torch.ones(3)

    def test_frame_transform_apply_basic(self):
        """Verify apply() centers and scales points."""
        center = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(2.0)
        transform = FrameTransform(center=center, scale=scale)

        points = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=torch.float32)
        result = transform.apply(points)

        expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(result, expected)

    def test_frame_transform_invert_basic(self):
        """Verify invert() reverses apply()."""
        center = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(2.0)
        transform = FrameTransform(center=center, scale=scale)

        points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        inverted = transform.invert(points)

        expected = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=torch.float32)
        assert torch.allclose(inverted, expected)

    def test_frame_transform_roundtrip(self):
        """Verify invert(apply(x)) == x."""
        center = torch.tensor([1.5, 2.5, 3.5])
        scale = torch.tensor(2.5)
        transform = FrameTransform(center=center, scale=scale)

        original = torch.randn(10, 3)
        transformed = transform.apply(original)
        recovered = transform.invert(transformed)

        assert torch.allclose(original, recovered, atol=1e-6)

    def test_frame_transform_batch_apply(self):
        """Verify apply() works on [B, N, 3] batches."""
        center = torch.zeros(3)
        scale = torch.tensor(1.0)
        transform = FrameTransform(center=center, scale=scale)

        batch_points = torch.randn(2, 5, 3)
        result = transform.apply(batch_points)

        assert result.shape == batch_points.shape
        assert torch.allclose(result, batch_points)

    def test_frame_transform_batch_roundtrip(self):
        """Verify roundtrip on batch."""
        center = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(1.5)
        transform = FrameTransform(center=center, scale=scale)

        original = torch.randn(3, 7, 3)
        transformed = transform.apply(original)
        recovered = transform.invert(transformed)

        assert torch.allclose(original, recovered, atol=1e-6)


class TestJointNormalize:
    """Tests for joint_normalize (STEPS T14)."""

    def test_joint_normalize_basic(self):
        """Verify joint_normalize returns shapes and transform."""
        shapes = [
            Shape(points=torch.randn(1, 5, 3), faces=torch.zeros((2, 3), dtype=torch.int64)),
            Shape(points=torch.randn(1, 5, 3), faces=torch.zeros((2, 3), dtype=torch.int64)),
        ]

        normalized, transform = joint_normalize(shapes)

        assert len(normalized) == 2
        assert isinstance(transform, FrameTransform)

    def test_joint_normalize_default_domain(self):
        """Verify default domain is (0, 1)."""
        # Create two shapes that are clearly separated
        shape1 = Shape(points=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
                       faces=torch.zeros((1, 3), dtype=torch.int64))
        shape2 = Shape(points=torch.tensor([[[2.0, 2.0, 2.0]]], dtype=torch.float32),
                       faces=torch.zeros((1, 3), dtype=torch.int64))

        normalized, _ = joint_normalize([shape1, shape2], domain=(0, 1))

        # All points should be in [0, 1]
        for shape in normalized:
            assert (shape.points >= 0.0).all()
            assert (shape.points <= 1.0).all()

    def test_joint_normalize_all_points_in_domain(self):
        """Verify all normalized points are within domain."""
        shapes = [
            Shape(points=torch.randn(1, 10, 3) * 5, faces=torch.zeros((5, 3), dtype=torch.int64)),
            Shape(points=torch.randn(1, 8, 3) * 5 + 10, faces=torch.zeros((4, 3), dtype=torch.int64)),
        ]

        normalized, _ = joint_normalize(shapes, domain=(0, 1))

        for shape in normalized:
            assert (shape.points >= -1e-5).all(), f"Min below domain: {shape.points.min()}"
            assert (shape.points <= 1.0 + 1e-5).all(), f"Max above domain: {shape.points.max()}"

    def test_joint_normalize_preserves_relative_offset(self):
        """Verify two shapes keep their relative offset after joint normalization."""
        # Create two shapes with a known relative offset
        center1 = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        center2 = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)

        shape1 = Shape(points=center1, faces=torch.zeros((1, 3), dtype=torch.int64))
        shape2 = Shape(points=center2, faces=torch.zeros((1, 3), dtype=torch.int64))

        normalized, _ = joint_normalize([shape1, shape2], domain=(0, 1))

        # Compute distance between centers before and after
        center1_norm = normalized[0].points[0, 0, :]
        center2_norm = normalized[1].points[0, 0, :]

        # Both should be within [0, 1] and preserve relative offset (direction)
        offset_original = (center2 - center1).squeeze()
        offset_normalized = (center2_norm - center1_norm)

        # Direction should be preserved (both are in x direction)
        assert torch.allclose(offset_normalized[1], torch.tensor(0.0), atol=1e-5)  # y component
        assert torch.allclose(offset_normalized[2], torch.tensor(0.0), atol=1e-5)  # z component
        # x offset should be positive and preserved
        assert float(offset_normalized[0]) > 0
        # Offset should scale uniformly (scaled down into domain)
        assert float(offset_normalized[0]) < float(offset_original[0])

    def test_joint_normalize_single_shape(self):
        """Verify joint_normalize works on a single shape."""
        shape = Shape(points=torch.randn(1, 5, 3) * 10, faces=torch.zeros((2, 3), dtype=torch.int64))
        normalized, transform = joint_normalize([shape], domain=(0, 1))

        assert len(normalized) == 1
        assert (normalized[0].points >= -1e-5).all()
        assert (normalized[0].points <= 1.0 + 1e-5).all()

    def test_joint_normalize_empty_raises(self):
        """Verify empty shapes list raises."""
        with pytest.raises(ValueError):
            joint_normalize([])

    def test_joint_normalize_preserves_faces_and_attributes(self):
        """Verify faces, weights, and normals are preserved."""
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        weights = torch.ones(1, 3)
        normals = torch.randn(1, 3, 3)

        shape = Shape(
            points=torch.randn(1, 3, 3),
            faces=faces,
            weights=weights,
            normals=normals
        )

        normalized, _ = joint_normalize([shape])

        assert torch.equal(normalized[0].faces, faces)
        assert torch.allclose(normalized[0].weights, weights)
        assert torch.allclose(normalized[0].normals, normals)

    def test_joint_normalize_custom_domain(self):
        """Verify custom domain bounds are respected."""
        shapes = [
            Shape(points=torch.randn(1, 5, 3), faces=torch.zeros((2, 3), dtype=torch.int64)),
        ]

        normalized, _ = joint_normalize(shapes, domain=(-1, 1))

        # All points should be in [-1, 1]
        assert (normalized[0].points >= -1.0 - 1e-5).all()
        assert (normalized[0].points <= 1.0 + 1e-5).all()
