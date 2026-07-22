"""Tests for I/O functions (STEPS T13+)."""

import torch
import pytest

from src.resnet_lddmm.io import Shape, load_shape


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
