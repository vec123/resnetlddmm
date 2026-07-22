import os
import torch
import numpy as np
from src.vtk.io import load_vtp, save_vtp
from src.vtk.extract import extract_vtp_points_cells
from src.vtk.create import(
    create_polydata, 
    create_polydata_w_lines
)
from src.graphs.graphs import (
    apply_noise_and_masking, 
    sample_nodes,
    build_radius_graph,
    get_graphs_from_vertices,
    get_individual_graph
    )
from src.transforms.padding import pad_vertex_list

from src.paths import get_project_root

Project_ROOT = get_project_root()
print("Project_ROOT: ", Project_ROOT)
OUTPUT_DIR = os.path.join(Project_ROOT, "tests", "output_data")


# Step 1: Load the shape vertices from the VTP files
shape_vertices = []
for name in ["nose_0", "nose_1", "nose_2", "nose_3"]:
    vtp_path = os.path.join(Project_ROOT, "tests", "data", f"{name}.vtp")
    vtp = load_vtp(vtp_path)
    vertices, cells = extract_vtp_points_cells(vtp)
    print("vertices: ", vertices.shape)
    print("cells: ", cells.shape)
    shape_vertices.append(vertices)

shape_vertices, shape_mask = pad_vertex_list(shape_vertices)
shape_mask = torch.tensor(shape_mask, dtype=torch.bool)
for i, num_samples in enumerate([None, 100]):
    key = torch.Generator(device='cpu')  # or 'cuda'
    key.manual_seed(i)

    sampling_mode = 'fps'
    print("shape_vertices.shape: ", shape_vertices.shape)
    graph = get_graphs_from_vertices(shape_vertices,
                                          masks=shape_mask,
                                          r_max=0.1, 
                                          dropout_rate=None, 
                                          noise_std=0.00,
                                          key = key,
                                          sampling_mode=sampling_mode,
                                          num_samples = num_samples)
    print("graph.batch: ", torch.unique(graph.batch))
    for j in range(len(torch.unique(graph.batch))):
        mask = (graph.batch == j)

        # 2. Extract the vertex positions for this graph
        #V= graph.pos[mask]
        #source_nodes = graph.edge_index[0]
        #edge_mask = (graph.batch[source_nodes] == 0)
        #E = graph.edge_index[:, edge_mask]
        

        V,E = get_individual_graph(graph, index=0)
        V = V.detach().cpu().numpy().astype(np.float32)
        # get_individual_graph already returns local (M, 2) edges — no transpose.
        E = E.detach().cpu().numpy().astype(np.int64)

        print("sampled vertices: ", V.shape)
        graph_vtp = create_polydata_w_lines(V, E)

        vtp_path = os.path.join(OUTPUT_DIR, f"{j}_th_graph_samples_{num_samples}.vtp")
        save_vtp(graph_vtp, vtp_path)

