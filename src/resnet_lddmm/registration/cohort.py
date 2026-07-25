"""Cohort registration stepper for Milestone B (multi-shape training)."""

import torch
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
            self.backward_traj = self.flow.inverse(target.points, code)
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
