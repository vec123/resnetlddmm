"""Runner: composition root for ResNetLDDMM (STEPS T16).

The ONE place where Registry.create is called and all components are assembled:
config -> shapes -> normalization -> stepper -> orchestrator -> run.

Does NOT import config directly; reads attributes dynamically (layering rule).
"""

import os
import json
import random

import numpy as np
import torch

from src.resnet_lddmm.config import ExperimentCfg
from src.resnet_lddmm.io import load_shape, joint_normalize, export_trajectory
from src.resnet_lddmm.registration.pair import PairRegistration
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.losses import UnidirectionalMappingError, BidirectionalMappingError
from src.resnet_lddmm import registrations  #  Side effect: registers all component Load component registrations
from src.learning.registry import Registry
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.loader.loaders import OneBatchLoader
from src.learning.trainers.E3_end2end import TrainingOrchestrator
from src.learning.callbacks.base import Callback
from src.resnet_lddmm.callbacks import TrajectoryExporter, DiagnosticsCallback


class VerboseCallback(Callback):
    """Log loss progression during training."""

    def __init__(self, log_every=1):
        super().__init__(every_n_steps=log_every)

    def on_step_end(self, ctx, step, metrics, batch, pred):
        """Print loss at cadence."""
        if not self._due(step):
            return
        loss = metrics.get("loss", 0.0)
        progress = f"Step {step+1:4d}/{ctx.num_steps} | loss: {loss:.6f}"

        # Add any other metrics
        for key in sorted(metrics.keys()):
            if key != "loss" and not key.startswith("diag/"):
                progress += f" | {key}: {metrics[key]:.6f}"

        print(progress)

    def _due(self, step):
        """Check if we should log this step."""
        return step % self.every_n_steps == 0 or step == 0


def seed_everything(seed):
    """Mirrors seeding: returns loader rng or None if seed is None."""
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    key = torch.Generator(device="cpu")
    key.manual_seed(seed)
    return key


def build(cfg: ExperimentCfg):
    """Assemble all components from config: the ONLY place Registry.create is called.

    Args:
        cfg: ExperimentCfg with source/target paths and component kinds

    Returns:
        PairRegistration stepper ready for training
    """
    seed_everything(cfg.train.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load and normalize shapes
    source_shape = load_shape(cfg.source)
    target_shape = load_shape(cfg.target)
    normalized, transform = joint_normalize([source_shape, target_shape])
    source_norm, target_norm = normalized

    # Create loader with normalized shapes
    # SimpleBatch duck-typing: has .points, .weights, and .faces
    class SimpleBatch:
        def __init__(self, shape):
            self.points = shape.points  # [1, N, 3]
            self.weights = shape.weights
            self.faces = shape.faces  # [F, 3] for trajectory export

    loader = OneBatchLoader((SimpleBatch(source_norm), SimpleBatch(target_norm)))

    # Registry.create: field and integrators
    if cfg.field.kind == "stationary":
        field = Registry.create("field", "stationary", activation=cfg.field.activation)
    else:
        field = Registry.create(
            "field", cfg.field.kind,
            num_blocks=cfg.field.num_steps,
            width=cfg.field.width,
            activation=cfg.field.activation
        )
    integrator_direct = ForwardEuler()
    integrator_inverse = ModifiedEuler() 

    # Build flow
    flow = NeuralODEFlow(
        field=field,
        direct=integrator_direct,
        inverse=integrator_inverse,
        num_steps=cfg.field.num_steps
    )

    # Registry.create: code source
    code_source = Registry.create("code", cfg.code.kind)

    # Registry.create: data term and isometry loss
    data_term = Registry.create(
        "data_term", cfg.loss.data_name,
        **cfg.loss.data_kwargs
    )
    iso_loss = Registry.create("iso_loss", "isometry", **cfg.loss.iso_kwargs) if cfg.loss.isometry_weight > 0 else None

    # Build loss composer
    # Data weight is 1/(2σ²) per D4 invariant
    data_weight = 1.0 / (2 * cfg.loss.sigma**2)
    terms = [
        LossTerm("data", weight=data_weight),
        LossTerm("kinetic", weight=cfg.loss.kinetic_weight),
        LossTerm("code_reg", weight=cfg.loss.code_reg_weight),
        LossTerm("isometry", weight=cfg.loss.isometry_weight),
    ]
    composer = LossComposer(terms)

    # Create optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.train.lr)

    # Create mapping error strategy based on config
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError()
    else:  # default to forward
        mapping_error = UnidirectionalMappingError()

    # Create stepper
    stepper = PairRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss)

    # Dump config to output dir for provenance
    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "source": cfg.source,
            "target": cfg.target,
            "field": {
                "kind": cfg.field.kind,
                "num_steps": cfg.field.num_steps,
                "width": cfg.field.width,
                "activation": cfg.field.activation,
            },
            "code": {"kind": cfg.code.kind},
            "loss": {
                "data_name": cfg.loss.data_name,
                "direction": cfg.loss.direction,
                "sigma": cfg.loss.sigma,
                "kinetic_weight": cfg.loss.kinetic_weight,
                "code_reg_weight": cfg.loss.code_reg_weight,
                "isometry_weight": cfg.loss.isometry_weight,
                "isometry_type": cfg.loss.isometry_type,
            },
            "train": {
                "steps": cfg.train.steps,
                "lr": cfg.train.lr,
                "seed": cfg.train.seed,
            },
        }, f, indent=2)

    return stepper, loader, transform


def run(cfg: ExperimentCfg, callbacks=None):
    """Load config, build stepper, run orchestrator.

    Args:
        cfg: ExperimentCfg instance
        callbacks: optional list of callbacks (default: VerboseCallback + TrajectoryExporter + DiagnosticsCallback)

    Returns:
        TrainingContext from orchestrator.run
    """
    if callbacks is None:
        # Default callbacks: verbose logging, trajectory export, diagnostics
        log_every = getattr(cfg.train, 'log_every', 50)
        save_every = getattr(cfg.train, 'save_every', 100)
        callbacks = [
            VerboseCallback(log_every=log_every),
            TrajectoryExporter(every_n_steps=save_every),
            DiagnosticsCallback(every_n_steps=save_every),
        ]

    stepper, loader, transform = build(cfg)

    # Pass transform to TrajectoryExporter
    for cb in callbacks:
        if isinstance(cb, TrajectoryExporter):
            cb.transform = transform

    orchestrator = TrainingOrchestrator(
        stepper=stepper,
        dataloader=loader,
        callbacks=callbacks,
        log_dir=cfg.output_dir,
    )
    return orchestrator.run(num_steps=cfg.train.steps)
