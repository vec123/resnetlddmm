"""Cohort registration stepper for Milestone B (multi-shape training). STEPS T27 T29."""

import torch
import torch.nn as nn
from src.learning.losses.composer import LossComposer
from src.resnet_lddmm.losses import BidirectionalMappingError


class CohortRegistration:
    """Multi-shape registration with shared flow Θ and per-shape codes Z.

    Registers a cohort of shapes to a shared template using a learned flow
    and per-shape latent codes. Uses bidirectional mapping error and two
    optimizer parameter groups: flow Θ with weight decay, codes Z without
    (since code regularization flows through the composer for visibility).
    Implements the four-method protocol (state_dict, load_state_dict, train, eval).
    """

    def __init__(self, flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=None):
        """Initialize the cohort registration stepper.

        Args:
            flow: NeuralODEFlow instance
            code_source: ShapeCode instance (must be AutoDecoderCodes for T27)
            data_term: DataTerm instance
            mapping_error: MappingError strategy (BidirectionalMappingError)
            composer: LossComposer instance
            optimizer: torch optimizer with param groups [flow_params, code_params]
            iso_loss: IsometryLoss instance (optional)
        """
        self.flow = flow
        self.code_source = code_source
        self.data_term = data_term
        self.mapping_error = mapping_error
        self.composer = composer
        self.optimizer = optimizer
        self.iso_loss = iso_loss
        self.is_bidirectional = isinstance(mapping_error, BidirectionalMappingError)
        self.backward_traj = None

    def _values(self, source_batch, target):
        """Compute trajectory and per-term loss values.

        Args:
            source_batch: batch-like with .points [B,N,3] and .shape_ids [B]
            target: batch-like with .points [B,M,3]

        Returns:
            (fwd_traj, values_dict) where values_dict has keys for data, kinetic, code_reg, isometry
        """
        code = self.code_source(source_batch)

        # Use mapping error strategy (encapsulates bidirectional logic)
        data, kinetic = self.mapping_error(self.flow, self.data_term, source_batch, target, code)

        # Compute forward trajectory for isometry loss
        fwd_traj = self.flow(source_batch.points, code)

        # For bidirectional mode, also compute backward trajectory for export
        if self.is_bidirectional:
            # Broadcast target to match batch size: [1, M, 3] -> [B, M, 3]
            B = source_batch.points.shape[0]
            target_points_broadcast = target.points.expand(B, -1, -1)
            self.backward_traj = self.flow.inverse(target_points_broadcast, code)
        else:
            self.backward_traj = None

        values = {
            "data": data,
            "kinetic": kinetic,
            "code_reg": self.code_source.penalty(),
        }

        # Add isometry loss if enabled
        if self.iso_loss is not None:
            values["isometry"] = self.iso_loss(fwd_traj, self.flow.field)

        return fwd_traj, values

    def train_step(self, source_batch, target):
        """One gradient step: forward, loss, backward, optimizer step.

        Args:
            source_batch: source shape batch with shape_ids
            target: target template shape

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        self.optimizer.zero_grad()
        traj, values = self._values(source_batch, target)
        loss, breakdown = self.composer.compute(values)
        loss.backward()
        self.optimizer.step()
        return traj, loss.item(), breakdown

    @torch.no_grad()
    def eval_step(self, source_batch, target):
        """Evaluate loss without gradient accumulation.

        Args:
            source_batch: source shape batch with shape_ids
            target: target template shape

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        traj, values = self._values(source_batch, target)
        loss, breakdown = self.composer.compute(values)
        return traj, loss.item(), breakdown

    def state_dict(self):
        """Serialize flow, codes, and optimizer state.

        Returns:
            Dict with keys "flow", "codes", "optimizer"
        """
        return {
            "flow": self.flow.state_dict(),
            "codes": self.code_source.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore flow, codes, and optimizer state.

        Args:
            state: dict with keys "flow", "codes", "optimizer"
        """
        self.flow.load_state_dict(state["flow"])
        self.code_source.load_state_dict(state["codes"])
        self.optimizer.load_state_dict(state["optimizer"])

    def train(self):
        """Set flow and code source to training mode."""
        self.flow.train()
        self.code_source.train()

    def eval(self):
        """Set flow and code source to evaluation mode."""
        self.flow.eval()
        self.code_source.eval()

    def infer(self, new_shape, target, steps_adam, steps_lbfgs=0):
        """Optimize code for a new shape with frozen flow (STEPS T29).

        Args:
            new_shape: Shape object with .points [1, N, 3] and optional weights/faces
            target: target template Shape with .points [1, M, 3]
            steps_adam: number of Adam optimization steps
            steps_lbfgs: number of L-BFGS steps after Adam (optional, default 0)

        Returns:
            optimized_code: Tensor [1, n_z], the inferred code for the new shape

        This implements the DeepSDF auto-decoder test-time protocol: freeze Θ,
        optimize only a fresh z initialized from N(0, 2/N_z·I). Cheap by
        construction — the amortisation payoff. Asserts flow is bit-identical
        before/after (frozen-decoder guarantee).
        """
        from src.learning.loader.loaders import CohortBatch

        # Snapshot flow state to verify it stays frozen
        flow_state_before = {
            name: param.data.clone() for name, param in self.flow.named_parameters()
        }

        # Freeze flow: disable gradients
        for param in self.flow.parameters():
            param.requires_grad = False

        # Initialize fresh code [1, n_z] ~ N(0, 2/n_z·I) (per paper)
        n_z = self.code_source.n_z
        device = next(self.flow.parameters()).device
        z_new = torch.randn(1, n_z, device=device) * (2.0 / n_z) ** 0.5
        z_new.requires_grad = True

        # Create batch for new shape
        source_batch = CohortBatch(
            points=new_shape.points.to(device),
            shape_ids=torch.tensor([0], dtype=torch.long, device=device),
            weights=new_shape.weights.to(device) if hasattr(new_shape, "weights") and new_shape.weights is not None else None,
            faces=[new_shape.faces] if hasattr(new_shape, "faces") and new_shape.faces is not None else [None],
        )

        # Create target batch (single point)
        target_batch = CohortBatch(
            points=target.points.to(device),
            shape_ids=torch.tensor([0], dtype=torch.long, device=device),
            weights=target.weights.to(device) if hasattr(target, "weights") and target.weights is not None else None,
            faces=[target.faces] if hasattr(target, "faces") and target.faces is not None else [None],
        )

        # Use optimizer learning rate (shared across param groups)
        lr = self.optimizer.param_groups[0]["lr"]
        optimizer = torch.optim.Adam([z_new], lr=lr)

        # Set to eval mode for inference (no dropout, stable BN)
        self.flow.eval()
        self.code_source.eval()

        # Adam optimization loop
        for _ in range(steps_adam):
            optimizer.zero_grad()

            # Forward pass: compute loss with frozen flow, variable z
            with torch.enable_grad():
                # Forward trajectory
                fwd_traj = self.flow(source_batch.points, z_new)

                # Backward trajectory (single template point broadcast)
                tgt_broadcast = target_batch.points.expand(1, -1, -1)
                bwd_traj = self.flow.inverse(tgt_broadcast, z_new)

                # Data term: bidirectional (source→target and target→source)
                data_fwd = self.data_term(fwd_traj.end, target_batch.points)
                data_bwd = self.data_term(bwd_traj.end, source_batch.points)
                data = data_fwd + data_bwd

                # Kinetic energy: sum both directions
                kinetic = fwd_traj.kinetic_energy() + bwd_traj.kinetic_energy()

                # Code regularization: direct L2 on z (not through embedding table)
                code_reg = (z_new ** 2).mean()

                # Compose loss with same weights as training
                values = {
                    "data": data,
                    "kinetic": kinetic,
                    "code_reg": code_reg,
                }
                loss, _ = self.composer.compute(values)

            loss.backward()
            optimizer.step()

        # Optional L-BFGS refinement (max_iter controls steps)
        if steps_lbfgs > 0:
            optimizer_lbfgs = torch.optim.LBFGS([z_new], max_iter=steps_lbfgs)

            def closure():
                optimizer_lbfgs.zero_grad()
                fwd_traj = self.flow(source_batch.points, z_new)
                tgt_broadcast = target_batch.points.expand(1, -1, -1)
                bwd_traj = self.flow.inverse(tgt_broadcast, z_new)
                data_fwd = self.data_term(fwd_traj.end, target_batch.points)
                data_bwd = self.data_term(bwd_traj.end, source_batch.points)
                data = data_fwd + data_bwd
                kinetic = fwd_traj.kinetic_energy() + bwd_traj.kinetic_energy()
                code_reg = (z_new ** 2).mean()
                values = {"data": data, "kinetic": kinetic, "code_reg": code_reg}
                loss, _ = self.composer.compute(values)
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

        # Verify flow is frozen: bit-identical before/after
        for name, param in self.flow.named_parameters():
            assert torch.allclose(
                param.data, flow_state_before[name], atol=1e-7
            ), f"Flow param '{name}' was modified during infer()"

        # Re-enable gradients on flow for future training
        for param in self.flow.parameters():
            param.requires_grad = True

        return z_new.detach()
