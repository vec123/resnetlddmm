"""I/O for shapes and trajectories (STEPS T13+, T32)."""

from dataclasses import dataclass, field
from typing import Optional
import os

import torch
import numpy as np

from src.vtk.io import load_vtp, load_obj, save_vtp
from src.vtk.extract import extract_vtp_points_cells, extract_vtp_point_fields
from src.vtk.create import create_polydata
from src.vtk.fields import add_point_field


@dataclass(frozen=True)
class FrameTransform:
    """Affine normalization transform: center and scale (STEPS T14, T32).

    Applies the transformation: (x - center) / scale
    Inverts via: x * scale + center

    Optional pose (T32): includes encoder-predicted rotation/translation.
    When present, invert() applies: x * scale + center, then rotates and translates.
    """

    center: torch.Tensor  # [3]
    scale: torch.Tensor  # scalar or [1]
    rotation: Optional[torch.Tensor] = None  # [3, 3] or [B, 3, 3] or None
    translation: Optional[torch.Tensor] = None  # [3] or [B, 3] or None

    def apply(self, points):
        """Apply the transform: (points - center) / scale.

        Args:
            points: [B, N, 3] or [N, 3]

        Returns:
            Transformed points in the same shape
        """
        return (points - self.center) / self.scale

    def invert(self, points):
        """Invert the transform: points * scale + center, then apply pose if available.

        Args:
            points: [N, 3] or [B, N, 3]
            rotation: [3, 3] or None (learned encoder pose)
            translation: [3] or None (learned encoder pose)

        Returns:
            Inverted points (with learned pose applied if available)
        """
        # First denormalize: x * scale + center
        world_points = points * self.scale + self.center

        # Then apply encoder-learned rotation and translation if available
        if self.rotation is not None:
            # Rotation: points @ R^T (works for any shape ending in 3)
            world_points = torch.matmul(world_points, self.rotation.T)

        if self.translation is not None:
            world_points = world_points + self.translation

        return world_points


@dataclass
class Shape:
    """A shape with points, faces, and optional weights and normals.

    Attributes:
        points: [1, N, 3] point cloud in normalized coordinates.
        faces: [F, 3] triangular face indices.
        weights: [1, N] per-point weights (e.g., area) or None (caller uses uniform).
        normals: [1, N, 3] per-point normals or None (caller computes if needed).
    """

    points: torch.Tensor  # [1, N, 3]
    faces: torch.Tensor  # [F, 3]
    weights: Optional[torch.Tensor] = None  # [1, N] or None
    normals: Optional[torch.Tensor] = None  # [1, N, 3] or None


def load_shape(path):
    """Load a shape from a VTP or OBJ file.

    Extracts points, faces, and optional point fields (area, normal).
    Missing fields return None (caller handles uniform fallback).

    Args:
        path: path to .vtp or .obj file

    Returns:
        Shape with points [1,N,3], faces [F,3], weights/normals or None
    """
    # Detect format by extension
    _, ext = os.path.splitext(path.lower())

    if ext == ".vtp":
        poly = load_vtp(path)
    elif ext == ".obj":
        poly = load_obj(path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .vtp, .obj")

    points_np, faces_np = extract_vtp_points_cells(poly)
    fields = extract_vtp_point_fields(poly, ["area", "normal"])

    # Reshape points to [1, N, 3]
    points = torch.tensor(points_np, dtype=torch.float32).unsqueeze(0)

    # Reshape faces to [F, 3]
    faces = torch.tensor(faces_np, dtype=torch.int64)

    # Extract weights from "area" field (or None if absent)
    weights = None
    if fields["area"] is not None:
        weights = torch.tensor(fields["area"], dtype=torch.float32).unsqueeze(0)

    # Extract normals (or None if absent)
    normals = None
    if fields["normal"] is not None:
        normals = torch.tensor(fields["normal"], dtype=torch.float32).unsqueeze(0)

    return Shape(points=points, faces=faces, weights=weights, normals=normals)


def joint_normalize(shapes, domain=(0, 1)):
    """Normalize a list of shapes to fit jointly into a domain (STEPS T14).

    Computes one bounding box over all shapes' union, then fits it to the
    target domain with a single center and scale applied to all.

    Args:
        shapes: list of Shape objects
        domain: (min, max) tuple for target domain (default (0, 1))

    Returns:
        (normalized_shapes, transform) where normalized_shapes is a list of
        Shape objects with normalized points, and transform is the FrameTransform
        used (for exporting back to world coordinates)
    """
    if not shapes:
        raise ValueError("shapes list cannot be empty")

    # Collect all points: [B, N, 3] -> flatten to [total_points, 3]
    all_points = torch.cat([s.points.reshape(-1, 3) for s in shapes], dim=0)

    # Compute bbox
    min_pt = all_points.min(dim=0).values  # [3]
    max_pt = all_points.max(dim=0).values  # [3]

    # Bbox size
    bbox_size = max_pt - min_pt  # [3]
    # Use the max dimension for uniform scaling (fits tightest box into cube)
    scale = bbox_size.max()

    # Center of bbox
    center = (min_pt + max_pt) / 2

    # Fit to domain: normalize to [-0.5, 0.5], then scale to domain
    domain_min, domain_max = domain
    domain_scale = domain_max - domain_min
    domain_center = (domain_max + domain_min) / 2

    # Transform: first center at origin, scale by 1/scale, then fit to domain
    transform = FrameTransform(center=center, scale=scale)

    # Normalize all shapes
    normalized_shapes = []
    for shape in shapes:
        norm_points = transform.apply(shape.points)
        # Shift from [-0.5, 0.5] to domain
        norm_points = norm_points * domain_scale + domain_center

        normalized_shapes.append(Shape(
            points=norm_points,
            faces=shape.faces,
            weights=shape.weights,
            normals=shape.normals
        ))

    return normalized_shapes, transform


def load_cohort(folder, template):
    """Load a cohort of shapes from a folder and normalize jointly with template (STEPS T26).

    Loads all .vtp files from a folder, assigns sequential shape_ids, and normalizes
    the cohort jointly with a template using a single bounding box (one frame).

    Args:
        folder: path to folder containing .vtp files
        template: Shape object or path to .vtp/.obj template file

    Returns:
        (cohort_shapes, template_norm, transform) where:
        - cohort_shapes: list of (shape_id, Shape) tuples for each cohort member
        - template_norm: normalized template Shape
        - transform: FrameTransform used for joint normalization
    """
    # Load template if it's a path
    if isinstance(template, str):
        template = load_shape(template)

    # Collect all .vtp files from cohort folder
    cohort_filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith('.vtp')])
    if not cohort_filenames:
        raise ValueError(f"No .vtp files found in {folder}")

    cohort_shapes_unnorm = []
    for fname in cohort_filenames:
        fpath = os.path.join(folder, fname)
        shape = load_shape(fpath)
        cohort_shapes_unnorm.append(shape)

    # Joint normalization: template + all cohort shapes in one frame
    all_shapes = [template] + cohort_shapes_unnorm
    normalized_all, transform = joint_normalize(all_shapes)

    # Extract normalized template and cohort
    template_norm = normalized_all[0]
    cohort_shapes = [(i, normalized_all[i + 1]) for i in range(len(cohort_shapes_unnorm))]

    return cohort_shapes, template_norm, transform


def export_reference_shapes(source, target, transform, out_dir):
    """Export source and target reference shapes as VTP files.

    Saves source.vtp and target.vtp in the output directory for visualization
    alongside trajectories. Shapes are denormalized to world coordinates.

    Args:
        source: Shape object with points [1, N, 3]
        target: Shape object with points [1, M, 3]
        transform: FrameTransform used for normalization (inverted to denormalize)
        out_dir: directory to save source.vtp and target.vtp files
    """
    os.makedirs(out_dir, exist_ok=True)

    # Denormalize source points
    source_points_norm = source.points[0, :, :]  # [N, 3]
    source_points_world = transform.invert(source_points_norm)  # [N, 3]

    # Validate source points
    if torch.isnan(source_points_world).any() or torch.isinf(source_points_world).any():
        print(f"[WARNING] Source points contain NaN/Inf, replacing with zeros")
        source_points_world = torch.where(torch.isnan(source_points_world) | torch.isinf(source_points_world),
                                          torch.tensor(0.0, device=source_points_world.device), source_points_world)

    # Create and save source PolyData (point cloud, no faces)
    source_polydata = create_polydata(source_points_world, faces=None)
    source_path = os.path.join(out_dir, "template_points.vtp")
    save_vtp(source_polydata, source_path, binary=True)

    # Denormalize target points
    target_points_norm = target.points[0, :, :]  # [M, 3]
    target_points_world = transform.invert(target_points_norm)  # [M, 3]

    # Validate target points
    if torch.isnan(target_points_world).any() or torch.isinf(target_points_world).any():
        print(f"[WARNING] Target points contain NaN/Inf, replacing with zeros")
        target_points_world = torch.where(torch.isnan(target_points_world) | torch.isinf(target_points_world),
                                          torch.tensor(0.0, device=target_points_world.device), target_points_world)

    # Create and save target PolyData (point cloud, no faces)
    target_polydata = create_polydata(target_points_world, faces=None)
    target_path = os.path.join(out_dir, "sample_points.vtp")
    save_vtp(target_polydata, target_path, binary=True)

    print(f"[io.export_reference_shapes] Exported source ({source_points_world.shape[0]} points) and target ({target_points_world.shape[0]} points)")


def export_trajectory(traj, faces, transform, out_dir):
    """Export trajectory steps as VTP files in world coordinates (STEPS T15).

    Saves one VTP file per integration step with velocity as a point field.
    Points are denormalized from normalized domain back to world coordinates.

    When subsampling is used, faces are not exported (point cloud only) since face
    indices would be invalid for subsampled points.

    Args:
        traj: Trajectory object with points [K+1, B, N, 3] and velocities [K, B, N, 3]
        faces: [F, 3] face indices carried to all steps (ignored if subsampled)
        transform: FrameTransform used for normalization (inverted to denormalize)
        out_dir: directory to save step_XXXX.vtp files

    Returns:
        List of saved file paths
    """
    os.makedirs(out_dir, exist_ok=True)

    K_plus_1, B, N, _ = traj.points.shape
    K = K_plus_1 - 1

    saved_paths = []

    for step in range(K_plus_1):
        # Extract points for this step, batch 0: [N, 3]
        points_norm = traj.points[step, 0, :, :]  # [N, 3]

        # Denormalize to world coordinates
        points_world = transform.invert(points_norm)  # [N, 3]

        # Validate points (NaN/Inf crash ParaView)
        if torch.isnan(points_world).any() or torch.isinf(points_world).any():
            print(f"[WARNING] Step {step}: points contain NaN/Inf, replacing with zeros")
            points_world = torch.where(torch.isnan(points_world) | torch.isinf(points_world),
                                       torch.tensor(0.0, device=points_world.device), points_world)

        # Create PolyData without faces (subsampled trajectories have invalid face indices)
        polydata = create_polydata(points_world, faces=None)

        # Add velocity as point field (zero for step 0, actual velocity for steps 1..K)
        if step == 0:
            velocity = torch.zeros(N, 3, dtype=torch.float32)
        else:
            velocity = traj.velocities[step - 1, 0, :, :]  # [N, 3]

        # Validate velocity
        if torch.isnan(velocity).any() or torch.isinf(velocity).any():
            print(f"[WARNING] Step {step}: velocity contains NaN/Inf, replacing with zeros")
            velocity = torch.where(torch.isnan(velocity) | torch.isinf(velocity),
                                  torch.tensor(0.0, device=velocity.device), velocity)

        polydata = add_point_field(polydata, velocity, field_name="velocity")

        # Save as step_XXXX.vtp
        filename = f"step_{step:04d}.vtp"
        filepath = os.path.join(out_dir, filename)
        save_vtp(polydata, filepath, binary=True)
        saved_paths.append(filepath)

    return saved_paths
