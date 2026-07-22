"""I/O for shapes and trajectories (STEPS T13+)."""

from dataclasses import dataclass

import torch
import numpy as np

from src.vtk.io import load_vtp
from src.vtk.extract import extract_vtp_points_cells, extract_vtp_point_fields


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
    weights: torch.Tensor | None = None  # [1, N] or None
    normals: torch.Tensor | None = None  # [1, N, 3] or None


def load_shape(path):
    """Load a shape from a VTP file.

    Extracts points, faces, and optional point fields (area, normal).
    Missing fields return None (caller handles uniform fallback).

    Args:
        path: path to .vtp file

    Returns:
        Shape with points [1,N,3], faces [F,3], weights/normals or None
    """
    poly = load_vtp(path)
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
