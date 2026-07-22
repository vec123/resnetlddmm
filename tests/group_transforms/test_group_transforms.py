import os
import torch
import numpy as np
from src.vtk.io import load_vtp, save_vtp
from src.vtk.create import(
    create_polydata, 
    create_polydata_w_lines
)
from src.vtk.extract import extract_vtp_points_cells
from src.transforms.group_transforms import (
    SE3_transform,
    SE3_transform_numpy
    )

from src.paths import get_project_root

Project_ROOT = get_project_root()
OUTPUT_DIR = os.path.join(Project_ROOT, "tests", "output_data")

vtp_path = os.path.join(Project_ROOT, "tests", "data", "sample_01.vtp")

vtp = load_vtp(vtp_path)
vertices, _ = extract_vtp_points_cells(vtp)
n_node = [vertices.shape[0]]

vtp_original = create_polydata(vertices)
output_path = os.path.join(OUTPUT_DIR, "original_graph_test_torch.vtp")
save_vtp(vtp_original, output_path)
#-------------NUMPY TEST
print("----------------test numpy transform-----------------")
rotation = torch.eye(3).unsqueeze(0)  
translation = torch.ones(1, 3)  
permute = True

random_matrix = np.random.randn(3, 3)
q, r = np.linalg.qr(random_matrix)
# Ensure it is a rotation matrix (det = 1) rather than a reflection (det = -1)
if np.linalg.det(q) < 0:
    q[:, 0] *= -1
rotation = q[np.newaxis, ...] # Shape [1, 3, 3]
translation = np.random.uniform(1.0, 5.0, (1, 3))

transformed_vertices = SE3_transform_numpy(
    vertices, n_node=n_node, rotations=rotation, translations=translation
    )

vtp_transformed = create_polydata(transformed_vertices)
output_path = os.path.join(OUTPUT_DIR, "transformed_graph_test_numpy.vtp")
save_vtp(vtp_transformed, output_path)




#-------------TORCH TEST
print("-------------------test torch transform-------------------")
rotation = torch.tensor(rotation, dtype=torch.float32)
translation = torch.tensor(translation, dtype=torch.float32)
vertices = torch.tensor(vertices, dtype=torch.float32)
n_node = torch.tensor(n_node, dtype=torch.int64)

transformed_vertices_torch = SE3_transform(
    vertices, n_node=n_node, rotations=rotation, translations=translation
)

transformed_vertices = transformed_vertices_torch.detach().cpu().numpy()
vtp_transformed = create_polydata(transformed_vertices)

output_path = os.path.join(OUTPUT_DIR, "transformed_graph_test_torch.vtp")
save_vtp(vtp_transformed, output_path)