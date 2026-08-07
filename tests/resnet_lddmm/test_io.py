"""Tests for I/O functions (STEPS T13+)."""

import torch
import pytest
import os
import tempfile
import numpy as np

from src.resnet_lddmm.io import Shape, load_shape, FrameTransform, joint_normalize, export_trajectory, load_cohort
from src.resnet_lddmm.trajectory import Trajectory
from src.vtk.io import load_vtp
from src.vtk.extract import extract_vtp_points_cells, extract_vtp_point_fields


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


class TestExportTrajectory:
    """Tests for export_trajectory (STEPS T15)."""

    def test_export_creates_files(self):
        """Verify export_trajectory creates VTP files."""
        # Create a simple trajectory: K=2 (3 snapshots), B=1, N=4
        points = torch.randn(3, 1, 4, 3)  # [K+1=3, B=1, N=4, 3]
        velocities = torch.randn(2, 1, 4, 3)  # [K=2, B=1, N=4, 3]
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_trajectory(traj, faces, transform, tmpdir)

            # Should create K+1=3 files
            assert len(paths) == 3
            for p in paths:
                assert os.path.exists(p)

    def test_export_file_naming(self):
        """Verify files are named step_0000.vtp, step_0001.vtp, etc."""
        points = torch.randn(3, 1, 4, 3)
        velocities = torch.randn(2, 1, 4, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_trajectory(traj, faces, transform, tmpdir)

            # Check naming
            assert paths[0].endswith("step_0000.vtp")
            assert paths[1].endswith("step_0001.vtp")
            assert paths[2].endswith("step_0002.vtp")

    def test_export_denormalizes_points(self):
        """Verify points are denormalized back to world coordinates."""
        # Create trajectory with simple normalized points
        points_norm = torch.tensor([
            [[0.0, 0.0, 0.0]],  # Step 0, batch 0, point 0
            [[0.5, 0.5, 0.5]],  # Step 1
        ], dtype=torch.float32).reshape(2, 1, 1, 3)

        velocities = torch.randn(1, 1, 1, 3)
        traj = Trajectory(points=points_norm, velocities=velocities, dt=0.5)

        # Transform that shifts center to [1, 2, 3] and scales by 2
        transform = FrameTransform(
            center=torch.tensor([1.0, 2.0, 3.0]),
            scale=torch.tensor(2.0)
        )

        faces = torch.tensor([[0, 0, 0]], dtype=torch.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_trajectory(traj, faces, transform, tmpdir)

            # Load first file and check points
            poly = load_vtp(os.path.join(tmpdir, "step_0000.vtp"))
            points_np, _ = extract_vtp_points_cells(poly)

            # Denormalized: (0 - 0) * 2 + [1, 2, 3] = [1, 2, 3]
            expected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            assert np.allclose(points_np, expected, atol=1e-5)

    def test_export_includes_velocity_field(self):
        """Verify velocity is included as a point field."""
        points = torch.randn(2, 1, 3, 3)
        velocities = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        velocities = velocities.reshape(1, 1, 3, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            export_trajectory(traj, faces, transform, tmpdir)

            # Load step 1 (has actual velocity)
            poly = load_vtp(os.path.join(tmpdir, "step_0001.vtp"))
            fields = extract_vtp_point_fields(poly, ["velocity"])

            # Velocity field should be present
            assert fields["velocity"] is not None
            assert fields["velocity"].shape == (3, 3)

    def test_export_step_0_zero_velocity(self):
        """Verify step 0 has zero velocity."""
        points = torch.randn(2, 1, 3, 3)
        velocities = torch.randn(1, 1, 3, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            export_trajectory(traj, faces, transform, tmpdir)

            # Load step 0
            poly = load_vtp(os.path.join(tmpdir, "step_0000.vtp"))
            fields = extract_vtp_point_fields(poly, ["velocity"])

            velocity = fields["velocity"]
            # All velocities should be zero
            assert np.allclose(velocity, 0.0, atol=1e-6)

    def test_export_writes_point_clouds_not_meshes(self):
        """Trajectory steps carry no connectivity, even when faces are passed.

        Deliberate, not an oversight: export_trajectory receives the template's
        faces whether or not the trajectory was subsampled, and under subsampling
        those indices name vertices that are not the exported points. Every step
        is a point cloud so the format does not depend on the run's settings.
        """
        points = torch.randn(2, 1, 4, 3)
        velocities = torch.randn(1, 1, 4, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        # Valid connectivity for these 4 points -- passed, and still not written.
        faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            export_trajectory(traj, faces, transform, tmpdir)

            poly = load_vtp(os.path.join(tmpdir, "step_0000.vtp"))
            points_np, faces_np = extract_vtp_points_cells(poly)

            assert points_np.shape == (4, 3)
            assert faces_np.shape[0] == 0, "trajectory export must not write polygons"

    def test_export_multiple_steps(self):
        """Verify export handles multiple trajectory steps."""
        # Create trajectory with K=5 (6 snapshots)
        K = 5
        points = torch.randn(K + 1, 1, 5, 3)
        velocities = torch.randn(K, 1, 5, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=1.0 / K)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_trajectory(traj, faces, transform, tmpdir)

            # Should create K+1=6 files
            assert len(paths) == K + 1

            # All files should exist
            for p in paths:
                assert os.path.exists(p)

    def test_export_creates_directory(self):
        """Verify export creates out_dir if it doesn't exist."""
        points = torch.randn(2, 1, 3, 3)
        velocities = torch.randn(1, 1, 3, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "nested", "path")
            # Directory does not exist yet
            assert not os.path.exists(outdir)

            export_trajectory(traj, faces, transform, outdir)

            # Directory should have been created
            assert os.path.exists(outdir)
            # Files should exist
            assert os.path.exists(os.path.join(outdir, "step_0000.vtp"))

    def test_export_velocity_matches_trajectory(self):
        """Verify exported velocities match trajectory velocities."""
        points = torch.randn(3, 1, 2, 3)
        velocities = torch.tensor([
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],  # Step 1
            [[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]   # Step 2
        ], dtype=torch.float32).reshape(2, 1, 2, 3)
        traj = Trajectory(points=points, velocities=velocities, dt=0.5)

        faces = torch.tensor([[0, 0, 0]], dtype=torch.int64)
        transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            export_trajectory(traj, faces, transform, tmpdir)

            # Load step 1 and check velocity
            poly = load_vtp(os.path.join(tmpdir, "step_0001.vtp"))
            fields = extract_vtp_point_fields(poly, ["velocity"])
            velocity_loaded = fields["velocity"]

            expected = velocities[0, 0, :, :].numpy()
            assert np.allclose(velocity_loaded, expected, atol=1e-6)


class TestLoadCohort:
    """Tests for load_cohort (STEPS T26)."""

    def test_load_cohort_basic(self):
        """Verify load_cohort loads shapes and returns proper structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple cohort with 2 shapes
            template = load_shape("data/hand/template.vtp")

            # Copy template to cohort folder twice
            import shutil
            for i in range(2):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            cohort, template_norm, transform = load_cohort(tmpdir, template)

            # Should load 2 shapes
            assert len(cohort) == 2
            assert isinstance(template_norm, Shape)
            assert isinstance(transform, FrameTransform)

    def test_load_cohort_assigns_shape_ids(self):
        """Verify cohort shapes are assigned sequential shape_ids."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = load_shape("data/hand/template.vtp")

            import shutil
            for i in range(3):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            cohort, _, _ = load_cohort(tmpdir, template)

            # Extract IDs
            ids = [shape_id for shape_id, _ in cohort]

            # Should have sequential IDs starting from 0
            assert ids == [0, 1, 2]

    def test_load_cohort_ids_stable(self):
        """Verify shape_ids are stable across multiple loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = load_shape("data/hand/template.vtp")

            import shutil
            for i in range(3):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            # Load twice
            cohort1, _, _ = load_cohort(tmpdir, template)
            cohort2, _, _ = load_cohort(tmpdir, template)

            ids1 = [shape_id for shape_id, _ in cohort1]
            ids2 = [shape_id for shape_id, _ in cohort2]

            # IDs should be identical
            assert ids1 == ids2

    def test_load_cohort_all_points_in_domain(self):
        """Verify all cohort points are normalized to domain [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = load_shape("data/hand/template.vtp")

            import shutil
            for i in range(2):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            cohort, template_norm, _ = load_cohort(tmpdir, template)

            # Template points should be in [0, 1]
            assert (template_norm.points >= -1e-5).all()
            assert (template_norm.points <= 1.0 + 1e-5).all()

            # Cohort points should be in [0, 1]
            for _, shape in cohort:
                assert (shape.points >= -1e-5).all(), f"Min below domain: {shape.points.min()}"
                assert (shape.points <= 1.0 + 1e-5).all(), f"Max above domain: {shape.points.max()}"

    def test_load_cohort_joint_normalization(self):
        """Verify cohort and template are normalized jointly (one frame)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Load a template
            template = load_shape("data/hand/template.vtp")
            template_unnorm_center = template.points.mean()

            import shutil
            for i in range(2):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            # Load cohort
            cohort, template_norm, _ = load_cohort(tmpdir, template)

            # All shapes (template and cohort) should be normalized jointly
            # This means they should share the same center and scale
            # A simple check: after normalization, all should be in [0, 1]
            # and the relative center position should be preserved

            # Since all shapes are identical (copies of template),
            # they should have identical normalized coordinates
            for _, cohort_shape in cohort:
                assert torch.allclose(
                    cohort_shape.points,
                    template_norm.points,
                    atol=1e-5
                ), "Cohort shapes should match template after joint normalization"

    def test_load_cohort_empty_folder_raises(self):
        """Verify empty folder raises an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = load_shape("data/hand/template.vtp")

            with pytest.raises(ValueError, match="No .vtp files found"):
                load_cohort(tmpdir, template)

    def test_load_cohort_with_template_path(self):
        """Verify load_cohort works when template is a path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import shutil
            for i in range(2):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            # Pass template as path string
            cohort, template_norm, _ = load_cohort(tmpdir, "data/hand/template.vtp")

            assert len(cohort) == 2
            assert isinstance(template_norm, Shape)

    def test_load_cohort_preserves_faces_and_attributes(self):
        """Verify faces and optional attributes are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template = load_shape("data/hand/template.vtp")

            import shutil
            for i in range(2):
                shutil.copy("data/hand/template.vtp", os.path.join(tmpdir, f"shape_{i:03d}.vtp"))

            cohort, template_norm, _ = load_cohort(tmpdir, template)

            # Faces should be preserved
            assert torch.equal(template_norm.faces, template.faces)

            for _, cohort_shape in cohort:
                assert torch.equal(cohort_shape.faces, template.faces)

                # Weights and normals should be preserved (or remain None)
                if template.weights is not None:
                    assert torch.allclose(cohort_shape.weights, template.weights, atol=1e-5)
                if template.normals is not None:
                    assert torch.allclose(cohort_shape.normals, template.normals, atol=1e-5)
