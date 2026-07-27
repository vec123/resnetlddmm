"""Diagnostics and export callbacks for ResNetLDDMM (STEP T20)."""

import torch

from src.learning.callbacks.base import Callback
from src.resnet_lddmm.diagnostics import jacobian_determinants, triangle_flips, lipschitz_bound
from src.resnet_lddmm.io import export_trajectory


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

            # Export backward trajectory if bidirectional mode
            stepper = ctx.stepper
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
