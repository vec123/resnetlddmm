"""Diagnostics and export callbacks for ResNetLDDMM (STEP T20)."""

import os
import torch
import numpy as np

from src.learning.callbacks.base import Callback
from src.resnet_lddmm.diagnostics import jacobian_determinants, triangle_flips, lipschitz_bound
from src.resnet_lddmm.io import export_trajectory, export_reference_shapes


class TrajectoryExporter(Callback):
    """Export predicted trajectories to VTP files for visualization.

    Calls io.export_trajectory() at its cadence to save deformed shapes
    at each integration step, with velocity fields attached.
    Supports selective export in cohort mode via export_shapes and export_strategy.
    """

    def __init__(self, every_n_steps=100, export_shapes=0, export_strategy="sequential", rng=None):
        super().__init__(every_n_steps)
        self.transform = None  # Set by runner after build()
        self.export_shapes = export_shapes  # 0 = all shapes
        self.export_strategy = export_strategy  # sequential | random
        self.rng = rng  # torch.Generator for random selection

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Export trajectory if due."""
        if not self._due(step):
            return

        # Trajectory is in the prediction (first return value from stepper.train_step)
        if pred is None:
            print(f"[TrajectoryExporter] Step {step}: pred is None, skipping")
            return

        # pred is a Trajectory object (from stepper.train_step return)
        traj = pred

        # Get batch geometry for faces (optional—point clouds work without faces)
        source, target = batch
        source_faces = source.faces if hasattr(source, "faces") and source.faces is not None else None
        target_faces = target.faces if hasattr(target, "faces") and target.faces is not None else None

        # Use actual transform from runner, or identity if not set
        if self.transform is None:
            from src.resnet_lddmm.io import FrameTransform
            transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))
        else:
            transform = self.transform

        base_out_dir = f"{ctx.log_dir}/trajectories/step_{step}"

        try:
            # Extract number of shapes in batch (B dimension)
            num_shapes = traj.points.shape[1]

            # Determine which shapes to export
            if self.export_shapes == 0:
                # Export all shapes
                shape_indices = list(range(num_shapes))
            else:
                # Export subset
                num_to_export = min(self.export_shapes, num_shapes)
                if self.export_strategy == "random":
                    shape_indices = torch.randperm(num_shapes, generator=self.rng)[:num_to_export].tolist()
                else:  # sequential
                    shape_indices = list(range(num_to_export))

            # Export reference shapes (source and target) once per step to base directory
            try:
                export_reference_shapes(source, target, self.transform, base_out_dir)
            except Exception as e:
                print(f"[TrajectoryExporter] Reference shape export failed: {e}")

            # Export forward trajectory for selected shapes
            for shape_idx in shape_indices:
                # Slice trajectory to single shape: [K+1, 1, N, 3]
                traj_slice = type(traj)(
                    points=traj.points[:, shape_idx:shape_idx+1, :, :],
                    velocities=traj.velocities[:, shape_idx:shape_idx+1, :, :],
                    dt=traj.dt
                )
                # Get faces for this shape (source_faces is a list)
                shape_faces = source_faces[shape_idx] if (isinstance(source_faces, list) and shape_idx < len(source_faces)) else source_faces
                fwd_dir = f"{base_out_dir}/forward/shape_{shape_idx}"
                print(f"[TrajectoryExporter] Exporting forward trajectory (shape {shape_idx}) to {fwd_dir}")
                export_trajectory(traj_slice, shape_faces, transform, fwd_dir)
            print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} forward trajectories")

            # Export full forward trajectory if save_full is enabled
            stepper = ctx.stepper
            mapping_error = stepper.mapping_error if hasattr(stepper, "mapping_error") else None
            if mapping_error is not None and hasattr(mapping_error, "last_fwd_traj_full") and mapping_error.last_fwd_traj_full is not None:
                traj_full = mapping_error.last_fwd_traj_full
                for shape_idx in shape_indices:
                    # Slice full trajectory to single shape: [K+1, 1, N, 3]
                    traj_full_slice = type(traj_full)(
                        points=traj_full.points[:, shape_idx:shape_idx+1, :, :],
                        velocities=traj_full.velocities[:, shape_idx:shape_idx+1, :, :],
                        dt=traj_full.dt
                    )
                    shape_faces = source_faces[shape_idx] if (isinstance(source_faces, list) and shape_idx < len(source_faces)) else source_faces
                    fwd_full_dir = f"{base_out_dir}/forward_full/shape_{shape_idx}"
                    print(f"[TrajectoryExporter] Exporting full forward trajectory (shape {shape_idx}) to {fwd_full_dir}")
                    export_trajectory(traj_full_slice, shape_faces, transform, fwd_full_dir)
                print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} full forward trajectories")

            # Export backward trajectory if bidirectional mode
            if hasattr(stepper, "backward_traj") and stepper.backward_traj is not None:
                bwd_traj = stepper.backward_traj
                for shape_idx in shape_indices:
                    # Slice backward trajectory: [K+1, 1, N, 3]
                    bwd_traj_slice = type(bwd_traj)(
                        points=bwd_traj.points[:, shape_idx:shape_idx+1, :, :],
                        velocities=bwd_traj.velocities[:, shape_idx:shape_idx+1, :, :],
                        dt=bwd_traj.dt
                    )
                    # Get target faces (target_faces is a list with one element)
                    tgt_faces = target_faces[0] if (isinstance(target_faces, list) and len(target_faces) > 0) else target_faces
                    bwd_dir = f"{base_out_dir}/backward/shape_{shape_idx}"
                    print(f"[TrajectoryExporter] Exporting backward trajectory (shape {shape_idx}) to {bwd_dir}")
                    export_trajectory(bwd_traj_slice, tgt_faces, transform, bwd_dir)
                print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} backward trajectories")

                # Export full backward trajectory if save_full is enabled
                if mapping_error is not None and hasattr(mapping_error, "last_bwd_traj_full") and mapping_error.last_bwd_traj_full is not None:
                    bwd_traj_full = mapping_error.last_bwd_traj_full
                    for shape_idx in shape_indices:
                        # Slice full backward trajectory: [K+1, 1, N, 3]
                        bwd_traj_full_slice = type(bwd_traj_full)(
                            points=bwd_traj_full.points[:, shape_idx:shape_idx+1, :, :],
                            velocities=bwd_traj_full.velocities[:, shape_idx:shape_idx+1, :, :],
                            dt=bwd_traj_full.dt
                        )
                        tgt_faces = target_faces[0] if (isinstance(target_faces, list) and len(target_faces) > 0) else target_faces
                        bwd_full_dir = f"{base_out_dir}/backward_full/shape_{shape_idx}"
                        print(f"[TrajectoryExporter] Exporting full backward trajectory (shape {shape_idx}) to {bwd_full_dir}")
                        export_trajectory(bwd_traj_full_slice, tgt_faces, transform, bwd_full_dir)
                    print(f"[TrajectoryExporter] Successfully exported {len(shape_indices)} full backward trajectories")
        except Exception as e:
            print(f"[TrajectoryExporter] Export failed: {e}")
            import traceback
            traceback.print_exc()


class DiagnosticsCallback(Callback):
    """Log diagnostic metrics into the training metrics dict.

    Computes and records:
    - Jacobian determinants (invertibility check)
    - Triangle flips (topology check)
    - Lipschitz bound (Lipschitz continuity check)
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Compute and log diagnostics if due."""
        if not self._due(step):
            return

        if pred is None:
            return

        traj = pred
        source, _ = batch
        faces = source.faces if hasattr(source, "faces") else None

        # Log subsampling info
        try:
            stepper = ctx.stepper
            mapping_error = stepper.mapping_error if hasattr(stepper, "mapping_error") else None
            if mapping_error is not None:
                if hasattr(mapping_error, "last_full_vertices") and mapping_error.last_full_vertices is not None:
                    metrics["diag/vertices_full"] = float(mapping_error.last_full_vertices)
                if hasattr(mapping_error, "last_subsample_vertices") and mapping_error.last_subsample_vertices is not None:
                    metrics["diag/vertices_subsampled"] = float(mapping_error.last_subsample_vertices)
        except Exception:
            pass

        # Jacobian determinants: should be ≥ 0 (positive measure for diffeomorphism)
        try:
            dets = jacobian_determinants(traj, faces)
            min_det = dets.min().item()
            metrics["diag/det_min"] = min_det
            metrics["diag/det_mean"] = dets.mean().item()
        except Exception:
            pass

        # Triangle flips: should be zero (no folds)
        if faces is not None and len(faces) > 0:
            try:
                flips = triangle_flips(traj, faces)
                num_flips = flips.sum().item()
                metrics["diag/flips_count"] = float(num_flips)
            except Exception:
                pass

        # Lipschitz bound: inferred from network weights
        try:
            flow = ctx.stepper.flow if hasattr(ctx.stepper, "flow") else None
            if flow is not None:
                bound = lipschitz_bound(flow)
                metrics["diag/lipschitz_bound"] = bound
        except Exception:
            pass


class EncoderGraphLogger(Callback):
    """Log encoder input graphs with node-type marking (full nodes vs supernodes).

    When using EncoderCodes, saves the point cloud and graph structure that the
    encoder receives at each logging step. Includes a point field "node_type" to
    distinguish regular nodes (0) from supernodes (1).

    Useful for debugging graph construction: visualize node density, dropout effects,
    supernode placement, and graph connectivity.
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)
        self._last_graph_cache = None
        self._last_supergraph_cache = None

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Cache graph and log if due."""
        if pred is None:
            return

        # Store references for potential use in on_log or explicit export
        stepper = ctx.stepper
        if hasattr(stepper, "code_source"):
            code_source = stepper.code_source
            # Check if this is an encoder (has graph_builder and encoder)
            if hasattr(code_source, "graph_builder") and hasattr(code_source, "_last"):
                # Cache for potential use later
                self._last_graph_cache = getattr(code_source, "_last_graph", None)
                self._last_supergraph_cache = getattr(code_source, "_last_supergraph", None)

        if not self._due(step):
            return

        # Export graph if available
        self._export_encoder_graph(ctx, step)

    def _export_encoder_graph(self, ctx, step):
        """Export the last encoder graph with node-type marking."""
        stepper = ctx.stepper
        if not hasattr(stepper, "code_source"):
            return

        code_source = stepper.code_source

        # Check if this is an encoder (has graph_builder)
        if not hasattr(code_source, "graph_builder"):
            return

        # Try to get the cached graph (stored during forward pass)
        graph = getattr(code_source, "_last_graph", None)
        supergraph = getattr(code_source, "_last_supergraph", None)

        if graph is None:
            print(f"[EncoderGraphLogger] Step {step}: No graph cached (encoder not called yet)")
            return

        try:
            # Create output directory
            out_dir = f"{ctx.log_dir}/encoder_graphs/step_{step}"
            os.makedirs(out_dir, exist_ok=True)

            # Export full graph with node-type marking
            self._export_graph_with_node_types(graph, supergraph, out_dir, "graph")

            # Export supergraph separately if it exists
            if supergraph is not None:
                self._export_graph_with_node_types(supergraph, None, out_dir, "supergraph")

            print(f"[EncoderGraphLogger] Exported encoder graph to {out_dir}")

        except Exception as e:
            print(f"[EncoderGraphLogger] Export failed: {e}")
            import traceback
            traceback.print_exc()

    def _export_graph_with_node_types(self, graph, supergraph, out_dir, name):
        """Export a PyG graph as VTP with node-type point field.

        Args:
            graph: PyG Data object with pos, batch, edge_index
            supergraph: PyG Data object (supergraph) or None
            out_dir: output directory
            name: base filename (e.g. "graph" or "supergraph")
        """
        try:
            from src.vtk.create import create_polydata
            from src.vtk.fields import add_point_field
            from src.vtk.io import save_vtp
        except ImportError:
            print(f"[EncoderGraphLogger] VTK utilities not available, skipping export")
            return

        pos = graph.pos.detach().cpu().numpy()
        edge_index = graph.edge_index.detach().cpu().numpy()
        batch = graph.batch.detach().cpu().numpy() if hasattr(graph, "batch") else np.zeros(pos.shape[0], dtype=np.int32)

        # Determine node types:
        # - Regular node: 0
        # - Supernode: 1 (if this graph IS a supergraph, or points to supergraph nodes)
        node_type = np.zeros(pos.shape[0], dtype=np.int32)

        # If we have a supergraph, mark supernodes as type 1
        if supergraph is not None:
            n_supernodes = supergraph.pos.shape[0]
            node_type[:n_supernodes] = 1

        # Create polydata with points and edges as lines
        polydata = create_polydata(pos, edge_index.T)

        # Add node_type as point field
        polydata = add_point_field(polydata, node_type, "node_type")

        # Add batch index as point field (for multi-shape batches)
        polydata = add_point_field(polydata, batch.astype(np.int32), "batch")

        # Add area if available
        if hasattr(graph, "area") and graph.area is not None:
            area = graph.area.detach().cpu().numpy().flatten()
            polydata = add_point_field(polydata, area.astype(np.float32), "area")

        # Save to VTP
        filepath = os.path.join(out_dir, f"{name}.vtp")
        save_vtp(filepath, polydata)
        print(f"[EncoderGraphLogger] Saved {name}: {pos.shape[0]} nodes, {edge_index.shape[1]} edges")
