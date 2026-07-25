"""Pair registration stepper for Milestone A (per-pair training)."""

import torch
from src.learning.losses.composer import LossComposer


class PairRegistration:
    """One-to-one shape registration with optional code regularisation.

    Composes flow, code source, data term, and loss via LossComposer.
    Implements both the training loop (train_step) and the four-method protocol
    (state_dict, load_state_dict, train, eval) for reused callbacks.
    """

    def __init__(self, flow, code_source, data_term, composer, optimizer, iso_loss=None):
        """Initialize the registration stepper.

        Args:
            flow: NeuralODEFlow instance
            code_source: ShapeCode instance (e.g., NoCode or AutoDecoderCodes)
            data_term: DataTerm instance (e.g., CDData or L2Data)
            composer: LossComposer instance
            optimizer: torch optimizer instance
            iso_loss: IsometryLoss instance (optional, created by runner if enabled)
        """
        self.flow = flow
        self.code_source = code_source
        self.data_term = data_term
        self.composer = composer
        self.optimizer = optimizer
        self.iso_loss = iso_loss

    def _values(self, source, target):
        """Compute trajectory and per-term loss values.

        Args:
            source: batch-like with .points [B,N,3] and optionally .weights
            target: batch-like with .points [B,M,3] and optionally .weights

        Returns:
            (traj, values_dict) where values_dict has keys for data, kinetic, code_reg, isometry
        """
        code = self.code_source(source)
        traj = self.flow(source.points, code)

        values = {
            "data": self.data_term(traj.end, target.points, tgt_w=target.weights),
            "kinetic": traj.kinetic_energy(),
            "code_reg": self.code_source.penalty(),
        }

        # Add isometry loss if enabled
        if self.iso_loss is not None:
            values["isometry"] = self.iso_loss(traj, self.flow.field)

        return traj, values

    def train_step(self, source, target):
        """One gradient step: forward, loss, backward, optimizer step.

        Args:
            source: source shape batch
            target: target shape batch

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        self.optimizer.zero_grad()
        traj, values = self._values(source, target)
        loss, breakdown = self.composer.compute(values)
        loss.backward()
        self.optimizer.step()
        return traj, loss.item(), breakdown

    @torch.no_grad()
    def eval_step(self, source, target):
        """Evaluate loss without gradient accumulation.

        Args:
            source: source shape batch
            target: target shape batch

        Returns:
            (trajectory, loss_float, breakdown_dict)
        """
        traj, values = self._values(source, target)
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
