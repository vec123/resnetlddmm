"""I/O for shapes and trajectories (STEPS T13+)."""

from dataclasses import dataclass
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
    """Affine normalization transform: center and scale (STEPS T14).

    Applies the transformation: (x - center) / scale
    Inverts via: x * scale + center
    """

    center: torch.Tensor  # [3]
    scale: torch.Tensor  # scalar or [1]

    def apply(self, points):
        """Apply the transform: (points - center) / scale.

        Args:
            points: [B, N, 3] or [N, 3]

        Returns:
            Transformed points in the same shape
        """
        return (points - self.center) / self.scale

    def invert(self, points):
        """Invert the transform: points * scale + center.

        Args:
            points: [B, N, 3] or [N, 3]

        Returns:
            Inverted points in the same shape
        """
        return points * self.scale + self.center


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


def export_trajectory(traj, faces, transform, out_dir):
    """Export trajectory steps as VTP files in world coordinates (STEPS T15).

    Saves one VTP file per integration step with velocity as a point field.
    Points are denormalized from normalized domain back to world coordinates.

    Args:
        traj: Trajectory object with points [K+1, B, N, 3] and velocities [K, B, N, 3]
        faces: [F, 3] face indices carried to all steps
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

        # Create PolyData with points and faces
        polydata = create_polydata(points_world, faces)

        # Add velocity as point field (zero for step 0, actual velocity for steps 1..K)
        if step == 0:
            velocity = torch.zeros(N, 3, dtype=torch.float32)
        else:
            velocity = traj.velocities[step - 1, 0, :, :]  # [N, 3]

        polydata = add_point_field(polydata, velocity, field_name="velocity")

        # Save as step_XXXX.vtp
        filename = f"step_{step:04d}.vtp"
        filepath = os.path.join(out_dir, filename)
        save_vtp(polydata, filepath, binary=True)
        saved_paths.append(filepath)

    return saved_paths
