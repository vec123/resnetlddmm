import os
import sys
import torch
from src.vtk.io import save_vtp, load_vtp
from src.vtk.extract import extract_vtp_points_cells
from src.vtk.fields import add_point_field
from src.vtk.create import create_polydata_w_lines, create_polydata
from src.graphs.graphs import (
    get_graphs_from_vertices,
    build_super_graph,
    get_individual_graph,
    get_bipartite_graph
)
from src.transforms.padding import pad_vertex_list
from src.paths import get_project_root


def load_shapes(names, project_root):
    """Loads VTP files and returns them as a padded [B, N, 3] tensor + validity mask."""
    shape_vertices = []
    for name in names:
        vtp = load_vtp(os.path.join(project_root, "tests", "data", f"{name}.vtp"))
        vertices, _ = extract_vtp_points_cells(vtp)
        shape_vertices.append(vertices)

    padded, mask = pad_vertex_list(shape_vertices)
    return torch.tensor(padded, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


def save_graph_as_vtp(graph, batch_idx, output_dir, filename_suffix):
    """Extracts, re-indexes, and saves one shape from a homogeneous PyG graph."""
    mask = (graph.batch == batch_idx)
    pos = graph.pos[mask].detach().cpu().numpy()

    # Normalize global edge indices to local [0, num_nodes_in_batch) via the batch start.
    start_idx = mask.nonzero(as_tuple=True)[0][0]
    edge_mask = (graph.batch[graph.edge_index[0]] == batch_idx)
    edges = (graph.edge_index[:, edge_mask] - start_idx).T.detach().cpu().numpy()

    save_path = os.path.join(output_dir, f"graph_{batch_idx}_{filename_suffix}.vtp")
    save_vtp(create_polydata_w_lines(pos, edges), save_path)






def main():
    Project_ROOT = get_project_root()
    OUTPUT_DIR = os.path.join(Project_ROOT, "tests", "output_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  Load shapes into a padded [B, N, 3] tensor + validity mask.
    samples = ["sample_01", "nose_0"]
    samples =["nose_0"]
    vertices, mask = load_shapes(samples, Project_ROOT)
    print(f"Loaded vertices shape: {vertices.shape}, mask shape: {mask.shape}")

    # Full graph: radius graph over all valid nodes. No dropout/noise here so the
    #    supernodes sampled below share the same coordinate space; pass a dropout_rate
    #    to get_graphs_from_vertices to decimate the graph before construction.
    full_graph = get_graphs_from_vertices(
        vertices, masks=mask, r_max=0.4, dropout_rate=0.0, noise_std=0.0
    )
    print(f"Full graph has {full_graph.num_nodes} nodes and {full_graph.num_edges} edges.")

    # Supernodes: draw n_s well-spread nodes per shape via farthest-point sampling.

    super_graph = build_super_graph(vertices, mask, full_graph, num_samples=50)
    print(f"Supernode graph has {super_graph.pos.shape[0]} supernodes and "
          f"{super_graph.edge_index.shape[1]} aggregation edges.")

    # Inspection: one VTP per shape for both the full graph and the supernode graph.
    #for j in range(vertices.shape[0]):

    for sample_idx in range( len(torch.unique(super_graph.batch)) ):
            pos, edges = get_individual_graph(super_graph, sample_idx)
            save_path = os.path.join(OUTPUT_DIR, f"super_graph_{sample_idx}.vtp")
            vtp = create_polydata_w_lines(pos, edges)
            save_vtp(vtp, save_path)

            pos, edges, node_field = get_bipartite_graph(super_graph, sample_idx)  
            save_path = os.path.join(OUTPUT_DIR, f"bipar_super_graph_{sample_idx}.vtp")
            vtp = create_polydata_w_lines(pos, edges)
            vtp = add_point_field(vtp, field_data=node_field,  field_name="super_node")
            save_vtp(vtp, save_path)

    for sample_idx in range( len(torch.unique(full_graph.batch)) ):
            pos, edges = get_individual_graph(full_graph, sample_idx)
            save_path = os.path.join(OUTPUT_DIR, f"full_graph_{sample_idx}.vtp")
            vtp = create_polydata(pos)
            save_vtp(vtp, save_path)

        #save_graph_as_vtp(full_graph, j, OUTPUT_DIR, "full")
        #save_bipartite_as_vtp(super_graph, j, OUTPUT_DIR, "supernodes")

    print(f"Inspection files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
