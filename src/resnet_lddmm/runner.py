"""Runner: composition root for ResNetLDDMM (STEPS T16).

The ONE place where Registry.create is called and all components are assembled:
config -> shapes -> normalization -> stepper -> orchestrator -> run.

Does NOT import config directly; reads attributes dynamically (layering rule).
"""

import os
import json
import glob
import random

import numpy as np
import torch

from src.resnet_lddmm.config import ExperimentCfg
from src.resnet_lddmm.io import load_shape, joint_normalize, export_trajectory
from src.resnet_lddmm.registration.pair import PairRegistration
from src.resnet_lddmm.registration.cohort import CohortRegistration
from src.resnet_lddmm.flow import NeuralODEFlow
from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.losses import UnidirectionalMappingError, BidirectionalMappingError
from src.resnet_lddmm import registrations  #  Side effect: registers all component Load component registrations
from src.learning.registry import Registry
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.loader.loaders import OneBatchLoader, CohortBatchLoader
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
        (stepper, loader, transform) where stepper is PairRegistration or CohortRegistration
    """
    seed_everything(cfg.train.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Registry.create: field and integrators (shared across pair/cohort)
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

    # Build flow (shared across pair/cohort)
    flow = NeuralODEFlow(
        field=field,
        direct=integrator_direct,
        inverse=integrator_inverse,
        num_steps=cfg.field.num_steps
    )

    # Registry.create: data term and isometry loss (shared)
    data_term = Registry.create(
        "data_term", cfg.loss.data_name,
        **cfg.loss.data_kwargs
    )
    iso_loss = Registry.create("iso_loss", "isometry", **cfg.loss.iso_kwargs) if cfg.loss.isometry_weight > 0 else None

    # Build loss composer (shared)
    data_weight = 1.0 / (2 * cfg.loss.sigma**2)
    terms = [
        LossTerm("data", weight=data_weight),
        LossTerm("kinetic", weight=cfg.loss.kinetic_weight),
        LossTerm("code_reg", weight=cfg.loss.code_reg_weight),
        LossTerm("isometry", weight=cfg.loss.isometry_weight),
    ]
    composer = LossComposer(terms)

    # Create mapping error strategy (shared)
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError()
    else:  # default to forward
        mapping_error = UnidirectionalMappingError()

    # Branch on training mode
    if cfg.train.mode == "cohort":
        return _build_cohort(cfg, flow, data_term, mapping_error, composer, iso_loss)
    else:
        return _build_pair(cfg, flow, data_term, mapping_error, composer, iso_loss)


def _build_pair(cfg, flow, data_term, mapping_error, composer, iso_loss):
    """Build PairRegistration stepper."""
    # Load and normalize shapes
    source_shape = load_shape(cfg.source)
    target_shape = load_shape(cfg.target)
    normalized, transform = joint_normalize([source_shape, target_shape])
    source_norm, target_norm = normalized

    # Create loader with normalized shapes
    class SimpleBatch:
        def __init__(self, shape):
            self.points = shape.points  # [1, N, 3]
            self.weights = shape.weights
            self.faces = shape.faces  # [F, 3] for trajectory export

    loader = OneBatchLoader((SimpleBatch(source_norm), SimpleBatch(target_norm)))

    # Registry.create: code source
    code_source = Registry.create("code", cfg.code.kind)

    # Create optimizer (single param group)
    optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.train.lr, weight_decay=cfg.loss.weight_decay)

    # Create stepper
    stepper = PairRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss)

    # Dump config to output dir
    _dump_config(cfg, "pair")

    return stepper, loader, transform


def _build_cohort(cfg, flow, data_term, mapping_error, composer, iso_loss):
    """Build CohortRegistration stepper."""
    # Load all cohort shapes from directory
    cohort_paths = sorted(glob.glob(os.path.join(cfg.source, "*.obj"))) + \
                   sorted(glob.glob(os.path.join(cfg.source, "*.ply")))
    if not cohort_paths:
        raise ValueError(f"No shape files found in {cfg.source}")

    cohort_shapes = [load_shape(p) for p in cohort_paths]
    target_shape = load_shape(cfg.target)

    # Normalize cohort + template together
    normalized, transform = joint_normalize(cohort_shapes + [target_shape])
    cohort_norm = normalized[:-1]
    target_norm = normalized[-1]

    # Create loader for cohort
    loader = CohortBatchLoader(cohort_norm, target_norm, batch_size=cfg.train.batch)

    # Registry.create: code source (must be AutoDecoderCodes for cohort)
    code_source = Registry.create("code", cfg.code.kind, num_shapes=len(cohort_norm), n_z=cfg.code.n_z)

    # Create optimizer with two param groups: flow with weight_decay, codes without
    optimizer = torch.optim.Adam([
        {"params": flow.parameters(), "weight_decay": cfg.loss.weight_decay},
        {"params": code_source.parameters(), "weight_decay": 0.0},
    ], lr=cfg.train.lr)

    # Create stepper
    stepper = CohortRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss)

    # Dump config to output dir
    _dump_config(cfg, "cohort")

    return stepper, loader, transform


def _dump_config(cfg, mode):
    """Dump config to output dir for provenance."""
    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "mode": mode,
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
        export_shapes = getattr(cfg.train, 'export_shapes', 0)
        export_strategy = getattr(cfg.train, 'export_strategy', 'sequential')
        # Seed rng for export strategy if seed is set
        export_rng = None
        if cfg.train.seed is not None:
            export_rng = torch.Generator(device="cpu")
            export_rng.manual_seed(cfg.train.seed)
        callbacks = [
            VerboseCallback(log_every=log_every),
            TrajectoryExporter(every_n_steps=save_every, export_shapes=export_shapes,
                             export_strategy=export_strategy, rng=export_rng),
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
