"""Diagnostics and export callbacks for ResNetLDDMM (STEP T20)."""

import torch

from src.learning.callbacks.base import Callback
from src.resnet_lddmm.diagnostics import jacobian_determinants, triangle_flips, lipschitz_bound
from src.resnet_lddmm.io import export_trajectory


class TrajectoryExporter(Callback):
    """Export predicted trajectories to VTP files for visualization.

    Calls io.export_trajectory() at its cadence to save deformed shapes
    at each integration step, with velocity fields attached.
    """

    def __init__(self, every_n_steps=100):
        super().__init__(every_n_steps)
        self.transform = None  # Set by runner after build()

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

        # Get batch geometry for faces
        source, target = batch
        source_faces = source.faces if hasattr(source, "faces") else None
        target_faces = target.faces if hasattr(target, "faces") else None

        if source_faces is None:
            print(f"[TrajectoryExporter] Step {step}: no source faces, skipping")
            return

        # Use actual transform from runner, or identity if not set
        if self.transform is None:
            from src.resnet_lddmm.io import FrameTransform
            transform = FrameTransform(center=torch.zeros(3), scale=torch.tensor(1.0))
        else:
            transform = self.transform

        base_out_dir = f"{ctx.log_dir}/trajectories/step_{step}"

        try:
            # Export forward trajectory
            fwd_dir = f"{base_out_dir}/forward"
            print(f"[TrajectoryExporter] Exporting forward trajectory to {fwd_dir}")
            export_trajectory(traj, source_faces, transform, fwd_dir)
            print(f"[TrajectoryExporter] Successfully exported forward trajectory")

            # Export backward trajectory if bidirectional mode
            stepper = ctx.stepper
            if hasattr(stepper, "backward_traj") and stepper.backward_traj is not None:
                bwd_dir = f"{base_out_dir}/backward"
                print(f"[TrajectoryExporter] Exporting backward trajectory to {bwd_dir}")
                if target_faces is not None:
                    export_trajectory(stepper.backward_traj, target_faces, transform, bwd_dir)
                else:
                    export_trajectory(stepper.backward_traj, source_faces, transform, bwd_dir)
                print(f"[TrajectoryExporter] Successfully exported backward trajectory")
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
