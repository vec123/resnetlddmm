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
from src.resnet_lddmm.integrators import ForwardEuler
from src.resnet_lddmm import registrations  # Load component registrations
from src.learning.registry import Registry
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.loader.loaders import OneBatchLoader
from src.learning.trainers.E3_end2end import TrainingOrchestrator


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
    # SimpleBatch duck-typing: has .points and .weights
    class SimpleBatch:
        def __init__(self, shape):
            self.points = shape.points  # [1, N, 3]
            self.weights = shape.weights

    loader = OneBatchLoader((SimpleBatch(source_norm), SimpleBatch(target_norm)))

    # Registry.create: field and integrators
    field = Registry.create(
        "field", cfg.field.kind,
        num_blocks=cfg.field.num_steps,
        width=cfg.field.width,
        activation=cfg.field.activation
    )
    integrator_direct = ForwardEuler()
    integrator_inverse = None  # T21 adds ModifiedEuler

    # Build flow
    flow = NeuralODEFlow(
        field=field,
        direct=integrator_direct,
        inverse=integrator_inverse,
        num_steps=cfg.field.num_steps
    )

    # Registry.create: code source
    code_source = Registry.create("code", cfg.code.kind)

    # Registry.create: data term
    data_term = Registry.create(
        "data_term", cfg.loss.data_name,
        **cfg.loss.data_kwargs
    )

    # Build loss composer
    # Data weight is 1/(2σ²) per D4 invariant
    data_weight = 1.0 / (2 * cfg.loss.sigma**2)
    composer = LossComposer([
        LossTerm("data", weight=data_weight),
        LossTerm("kinetic", weight=cfg.loss.kinetic_weight),
        LossTerm("code_reg", weight=cfg.loss.code_reg_weight),
    ])

    # Create optimizer
    optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.train.lr)

    # Create stepper
    stepper = PairRegistration(flow, code_source, data_term, composer, optimizer)

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
                "sigma": cfg.loss.sigma,
                "kinetic_weight": cfg.loss.kinetic_weight,
                "code_reg_weight": cfg.loss.code_reg_weight,
            },
            "train": {
                "steps": cfg.train.steps,
                "lr": cfg.train.lr,
                "seed": cfg.train.seed,
            },
        }, f, indent=2)

    return stepper, loader


def run(cfg: ExperimentCfg, callbacks=None):
    """Load config, build stepper, run orchestrator.

    Args:
        cfg: ExperimentCfg instance
        callbacks: optional list of callbacks (default: empty)

    Returns:
        TrainingContext from orchestrator.run
    """
    callbacks = callbacks or []
    stepper, loader = build(cfg)
    orchestrator = TrainingOrchestrator(
        stepper=stepper,
        dataloader=loader,
        callbacks=callbacks,
        log_dir=cfg.output_dir,
    )
    return orchestrator.run(num_steps=cfg.train.steps)
