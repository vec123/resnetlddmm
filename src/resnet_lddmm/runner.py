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

from src.resnet_lddmm.config import ExperimentCfg, CENTERING_DOMAINS
from src.resnet_lddmm.io import load_shape, joint_normalize, export_trajectory
from src.resnet_lddmm.registration.pair import PairRegistration
from src.resnet_lddmm.registration.cohort import CohortRegistration
from src.resnet_lddmm.flow import NeuralODEFlow, IdentityFlowWrapper
from src.resnet_lddmm.integrators import ForwardEuler, ModifiedEuler
from src.resnet_lddmm.losses import UnidirectionalMappingError, BidirectionalMappingError
from src.resnet_lddmm import registrations  #  Side effect: registers all component Load component registrations
from src.learning.registry import Registry
from src.learning.losses.composer import LossComposer, LossTerm
from src.learning.loader.loaders import OneBatchLoader, CohortBatchLoader
from src.learning.trainers.E3_end2end import TrainingOrchestrator
from src.learning.callbacks.base import Callback
from src.resnet_lddmm.callbacks import TrajectoryExporter, DiagnosticsCallback, EncoderGraphLogger, GradientLogger, NetworkStructureInspector, RequiresGradMonitor, PoseLogger, PoseShapeExporter


def _encoder_layers_to_list_of_dicts(encoder_config):
    """Convert EncoderConfig.layers (dataclass objects) to list of dicts for Registry.create.

    EncoderConfig stores layers as EncoderLayerConfig dataclass instances;
    Registry.create(**kwargs) expects dicts to pass as kwargs to GroupEncoder.__init__.
    """
    return [
        {
            "in_irreps": layer.in_irreps,
            "target_irreps": layer.target_irreps,
            "spatial_sh_lmax": layer.spatial_sh_lmax,
            "interaction_sh_lmax": layer.interaction_sh_lmax,
        }
        for layer in encoder_config.layers
    ]


def _build_encoder_code_source(encoder_config, graph_spec, n_z):
    """Factory to instantiate EncoderCodes with graph builder and encoder.

    Creates a GraphBuilder and GroupEncoder from spec/config, then wraps
    in EncoderCodes for training. This is only called when code.kind == "encoder".

    Args:
        encoder_config: EncoderConfig instance (from spec.py)
        graph_spec: GraphSpec instance (from spec.py)
        n_z: latent dimension (int)

    Returns:
        EncoderCodes instance ready to use in training
    """
    if encoder_config is None:
        raise ValueError("encoder_config required for code.kind='encoder'")
    if graph_spec is None:
        raise ValueError("graph_spec required for code.kind='encoder'")

    # Create GraphBuilder
    graph_builder = Registry.create("graph_builder", "radius", spec=graph_spec)

    # Create GroupEncoder with layer specs from encoder_config
    layers_as_dicts = _encoder_layers_to_list_of_dicts(encoder_config)
    encoder = Registry.create(
        "encoder", "group_encoder",
        layers_cfg=layers_as_dicts,
        latent_dim=n_z,
        readout=encoder_config.readout,
        readout_heads=encoder_config.readout_heads,
        supernode_sh_lmax=encoder_config.supernode_sh_lmax,
        transformer_type=encoder_config.transformer_type,
        transformer_cfg=encoder_config.transformer_cfg,
        supernode_samples=encoder_config.supernode_samples,
        supernode_seed=encoder_config.supernode_seed,
        area_pool=encoder_config.area_pool,
        latent_mode=encoder_config.latent_mode,
        verbose=encoder_config.verbose,
    )

    # Create EncoderCodes wrapping both
    code_source = Registry.create(
        "code", "encoder",
        graph_builder=graph_builder,
        encoder=encoder,
        n_z=n_z,
    )

    return code_source


class VerboseCallback(Callback):
    """Log loss progression during training."""

    def __init__(self, log_every=1, pose_only_mode=False):
        super().__init__(every_n_steps=log_every)
        self.pose_only_mode = pose_only_mode

    def on_train_start(self, ctx):
        """Print training mode info at startup."""
        if self.pose_only_mode:
            print("[POSE-ONLY MODE] Flow is frozen at identity; encoder learns pose only.")

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


def _normalization_domain(cfg):
    """cfg.centering -> the (min, max) domain joint_normalize places shapes in.

    getattr so a stub config without the field still builds, matching how the rest
    of this module reads optional attributes.
    """
    return CENTERING_DOMAINS[getattr(cfg, "centering", "box")]


def _build_loss_terms(loss_cfg):
    """cfg.loss.terms -> (composer entries, {name: term module}).

    Zero-weight entries are dropped rather than computed and multiplied by zero,
    matching how isometry has always been skipped when disabled.

    Args:
        loss_cfg: LossCfg with a .terms list of {kind, weight, name, kwargs} dicts

    Returns:
        (list of LossTerm, dict mapping metric name -> instantiated term)

    Raises:
        ValueError: on a missing 'kind', or a name colliding with another term
    """
    specs = list(loss_cfg.terms)

    # First-class weights, configured like every other term (isometry_weight,
    # kinetic_weight, ...). Strictly > 0 to be built at all: at 0 the term is never
    # instantiated and its extra work never runs, rather than being computed and
    # multiplied by zero. A new gated term is one entry here.
    for prefix, kind in (("equivariant_deformation", "equivariant_deformation_loss"),
                         ("pose_supervision", "pose_supervision_loss"),
                         ("flow_rotation_penalty", "flow_rotation_penalty")):
        weight = float(getattr(loss_cfg, f"{prefix}_weight", 0.0) or 0.0)
        if weight > 0:
            specs.append({
                "kind": kind,
                "weight": weight,
                "kwargs": dict(getattr(loss_cfg, f"{prefix}_kwargs", None) or {}),
            })

    entries, modules = [], {}
    for spec in specs:
        if "kind" not in spec:
            raise ValueError(f"loss term needs a 'kind': {spec}")
        name = spec.get("name", spec["kind"])
        if name in modules:
            raise ValueError(
                f"duplicate loss term name {name!r}; give one of them an explicit "
                f"'name' so their metrics stay distinguishable"
            )
        weight = float(spec.get("weight", 1.0))
        if weight <= 0:
            continue
        # Registry.create raises with the list of valid names on an unknown kind.
        modules[name] = Registry.create("loss_term", spec["kind"], **spec.get("kwargs", {}))
        entries.append(LossTerm(name, weight=weight))
    return entries, modules


def build(cfg: ExperimentCfg):
    """Assemble all components from config: the ONLY place Registry.create is called.

    Args:
        cfg: ExperimentCfg with source/target paths and component kinds

    Returns:
        (stepper, loader, transform) where stepper is PairRegistration or CohortRegistration
    """
    seed_everything(cfg.train.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Validate encoder_pose configuration
    use_encoder_pose = getattr(cfg, 'use_encoder_pose', False)
    freeze_flow_at_identity = getattr(cfg, 'freeze_flow_at_identity', False)

    # Validate freeze_flow_at_identity first (most specific)
    if freeze_flow_at_identity and not use_encoder_pose:
        raise ValueError(
            "freeze_flow_at_identity requires use_encoder_pose=true. "
            "Pose-only training only makes sense if encoder learns pose."
        )

    if freeze_flow_at_identity and cfg.loss.direction == "bidirectional":
        raise ValueError(
            "freeze_flow_at_identity requires unidirectional flow. "
            "Set loss.direction='forward' or disable freeze_flow_at_identity."
        )

    # Then validate use_encoder_pose
    if use_encoder_pose and cfg.loss.direction == "bidirectional":
        raise ValueError(
            "use_encoder_pose currently supports unidirectional flow only. "
            "Set loss.direction='forward' or disable use_encoder_pose."
        )

    # Build conditioning (shared across pair/cohort)
    # Orthogonal axes: position_aware (grid interpolation or broadcast?)
    #                  + conditioning_method (concat or FiLM modulation?)
    conditioning = None
    if cfg.code.kind not in ("none",):
        # Build conditioning based on position_aware + conditioning_method
        if cfg.code.position_aware:
            conditioning = Registry.create(
                "conditioning", "position_aware",
                n_z=cfg.code.n_z,
                g=cfg.code.grid,
                channels=cfg.code.grid_channels
            )
        else:
            # Broadcast-based conditioning (concat or FiLM)
            conditioning = Registry.create(
                "conditioning", cfg.code.conditioning_method,
                n_z=cfg.code.n_z
            )

    # Registry.create: field and integrators (shared across pair/cohort)
    if cfg.field.kind == "stationary":
        field = Registry.create(
            "field", "stationary",
            activation=cfg.field.activation,
            conditioning=conditioning
        )
    elif cfg.field.kind == "equivariant_stationary":
        field = Registry.create(
            "field", "equivariant_stationary",
            hidden_irreps=cfg.field.hidden_irreps,
            gate_hidden_dim=cfg.field.gate_hidden_dim,
            use_tensor_product_self=cfg.field.use_tensor_product_self,
            layers_cfg=cfg.field.layers,
            conditioning=conditioning
        )
    elif cfg.field.kind == "equivariant_contextual":
        field = Registry.create(
            "field", "equivariant_contextual",
            context_irreps=cfg.field.context_irreps if hasattr(cfg.field, 'context_irreps') else "32x0e + 16x1o",
            sh_lmax=cfg.field.sh_lmax if hasattr(cfg.field, 'sh_lmax') else 2,
            layers_cfg=cfg.field.layers,
            conditioning=conditioning
        )
    elif cfg.field.kind == "equivariant_contextual_simple":
        field = Registry.create(
            "field", "equivariant_contextual_simple",
            hidden_dim=getattr(cfg.field, 'hidden_dim', 128),
            n_layers=getattr(cfg.field, 'n_layers', 3),
            conditioning=conditioning
        )
    elif cfg.field.kind == "equivariant_template":
        field = Registry.create(
            "field", "equivariant_template",
            layers_cfg=cfg.field.layers,
            conditioning=conditioning
        )
    else:
        field = Registry.create(
            "field", cfg.field.kind,
            num_blocks=cfg.field.num_steps,
            width=cfg.field.width,
            activation=cfg.field.activation,
            conditioning=conditioning
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

    # Conditionally wrap flow at identity for pose-only training
    if freeze_flow_at_identity:
        flow = IdentityFlowWrapper(flow)

    # Registry.create: data term and isometry loss (shared)
    data_term = Registry.create(
        "data_term", cfg.loss.data_name,
        **cfg.loss.data_kwargs
    )
    iso_loss = Registry.create(
        "iso_loss", "isometry",
        loss_type=cfg.loss.isometry_type,
        sample_points=cfg.loss.isometry_samples
    ) if cfg.loss.isometry_weight > 0 else None

    # Build loss composer (shared). data/kinetic/code_reg/isometry are produced by
    # the mapping error and code source; cfg.loss.terms adds anything else.
    data_weight = 1.0 / (2 * cfg.loss.sigma**2)
    extra_terms, loss_terms = _build_loss_terms(cfg.loss)
    terms = [
        LossTerm("data", weight=data_weight),
        LossTerm("kinetic", weight=cfg.loss.kinetic_weight),
        LossTerm("code_reg", weight=cfg.loss.code_reg_weight),
        LossTerm("isometry", weight=cfg.loss.isometry_weight),
        *extra_terms,
    ]
    composer = LossComposer(terms)

    # Build augmentation (shared across pair/cohort)
    # Only pass seed for so3 and se3; none doesn't accept it
    aug_kwargs = {}
    if cfg.augmentation.kind in ("so3", "se3"):
        aug_kwargs["seed"] = cfg.augmentation.seed
    if cfg.augmentation.kind == "se3":
        aug_kwargs["translation_scale"] = cfg.augmentation.translation_scale

    augmentation = Registry.create(
        "augmentation", cfg.augmentation.kind,
        **aug_kwargs
    )

    # Branch on training mode (mapping_error created after loading shapes)
    if cfg.train.mode == "cohort":
        return _build_cohort(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose, loss_terms)
    else:
        return _build_pair(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose, loss_terms)


def _build_code_source(kind, code_cfg):
    """Build a ShapeCode instance (none, auto_decoder, or encoder).

    Args:
        kind: "none" | "auto_decoder" | "encoder"
        code_cfg: CodeCfg instance with all settings

    Returns:
        ShapeCode instance ready to use in training
    """
    if kind == "encoder":
        return _build_encoder_code_source(
            code_cfg.encoder_config,
            code_cfg.graph_spec,
            code_cfg.n_z
        )
    else:
        # For "none" and "auto_decoder", use Registry with standard args
        return Registry.create("code", kind)


def _build_code_source_cohort(num_shapes, kind, code_cfg):
    """Build a ShapeCode for cohort mode (auto_decoder and encoder need num_shapes).

    Args:
        num_shapes: number of shapes in cohort
        kind: "none" | "auto_decoder" | "encoder"
        code_cfg: CodeCfg instance with all settings

    Returns:
        ShapeCode instance ready to use in training
    """
    if kind == "encoder":
        return _build_encoder_code_source(
            code_cfg.encoder_config,
            code_cfg.graph_spec,
            code_cfg.n_z
        )
    elif kind == "auto_decoder":
        return Registry.create("code", kind, num_shapes=num_shapes, n_z=code_cfg.n_z)
    else:
        # "none"
        return Registry.create("code", kind)


def _build_pair(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose=False, loss_terms=None):
    """Build PairRegistration stepper."""
    # Load and normalize shapes
    source_shape = load_shape(cfg.source)
    target_shape = load_shape(cfg.target)
    normalized, transform = joint_normalize([source_shape, target_shape],
                                           domain=_normalization_domain(cfg))
    source_norm, target_norm = normalized

    # Create mapping error strategy after knowing source size
    subsample_n = cfg.loss.resolve_subsample_count(source_norm.points.shape[1])
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)
    else:  # default to forward
        mapping_error = UnidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)

    # Create loader with normalized shapes
    class SimpleBatch:
        def __init__(self, shape):
            self.points = shape.points  # [1, N, 3]
            self.weights = shape.weights
            self.faces = shape.faces  # [F, 3] for trajectory export

    loader = OneBatchLoader((SimpleBatch(source_norm), SimpleBatch(target_norm)))

    # Build code source (handles none, auto_decoder, encoder)
    code_source = _build_code_source(cfg.code.kind, cfg.code)

    # Create optimizer (single param group)
    optimizer = torch.optim.Adam(flow.parameters(), lr=cfg.train.lr, weight_decay=cfg.loss.weight_decay)

    # Create stepper
    stepper = PairRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss, augmentation=augmentation, use_encoder_pose=use_encoder_pose, loss_terms=loss_terms)

    # Dump config to output dir
    _dump_config(cfg, "pair")

    return stepper, loader, transform


def _build_cohort(cfg, flow, data_term, composer, iso_loss, augmentation, use_encoder_pose=False, loss_terms=None):
    """Build CohortRegistration stepper."""
    # Load all cohort shapes from directory
    cohort_paths = sorted(glob.glob(os.path.join(cfg.source, "*.obj"))) + \
                   sorted(glob.glob(os.path.join(cfg.source, "*.ply"))) + \
                   sorted(glob.glob(os.path.join(cfg.source, "*.vtp")))
    if not cohort_paths:
        raise ValueError(f"No shape files found in {cfg.source} (.obj, .ply, or .vtp)")

    cohort_shapes = [load_shape(p) for p in cohort_paths]
    target_shape = load_shape(cfg.target)

    # Normalize cohort + template together
    normalized, transform = joint_normalize(cohort_shapes + [target_shape],
                                           domain=_normalization_domain(cfg))
    cohort_norm = normalized[:-1]
    target_norm = normalized[-1]

    # Create mapping error strategy after knowing source size (use first shape)
    subsample_n = cfg.loss.resolve_subsample_count(cohort_norm[0].points.shape[1])
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)
    else:  # default to forward
        mapping_error = UnidirectionalMappingError(subsample_n=subsample_n if subsample_n > 0 else None, save_full=cfg.loss.save_full)

    # Create loader for cohort
    loader = CohortBatchLoader(cohort_norm, target_norm, batch_size=cfg.train.batch)

    # Build code source (handles none, auto_decoder, encoder with cohort-specific logic)
    code_source = _build_code_source_cohort(len(cohort_norm), cfg.code.kind, cfg.code)

    # Create optimizer with two param groups: flow with weight_decay, codes without
    optimizer = torch.optim.Adam([
        {"params": flow.parameters(), "weight_decay": cfg.loss.weight_decay},
        {"params": code_source.parameters(), "weight_decay": 0.0},
    ], lr=cfg.train.lr)

    # Create stepper
    stepper = CohortRegistration(flow, code_source, data_term, mapping_error, composer, optimizer, iso_loss=iso_loss, augmentation=augmentation, use_encoder_pose=use_encoder_pose, loss_terms=loss_terms)

    # Dump config to output dir
    _dump_config(cfg, "cohort")

    return stepper, loader, transform


def _dump_config(cfg, mode):
    """Dump config to output dir for provenance."""
    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "mode": mode,
            "centering": getattr(cfg, "centering", "box"),
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
        freeze_flow_at_identity = getattr(cfg, 'freeze_flow_at_identity', False)
        # Seed rng for export strategy if seed is set
        export_rng = None
        if cfg.train.seed is not None:
            export_rng = torch.Generator(device="cpu")
            export_rng.manual_seed(cfg.train.seed)
        callbacks = [
            NetworkStructureInspector(),
            VerboseCallback(log_every=log_every, pose_only_mode=freeze_flow_at_identity),
            GradientLogger(every_n_steps=log_every),
            RequiresGradMonitor(every_n_steps=log_every),
            TrajectoryExporter(every_n_steps=save_every, export_shapes=export_shapes,
                             export_strategy=export_strategy, rng=export_rng),
            DiagnosticsCallback(every_n_steps=save_every),
        ]

    stepper, loader, transform = build(cfg)

    # Pass transform to TrajectoryExporter
    for cb in callbacks:
        if isinstance(cb, TrajectoryExporter):
            cb.transform = transform

    # Add EncoderGraphLogger if using encoder
    if cfg.code.kind == "encoder":
        graph_log_cadence = getattr(cfg.train, 'save_every', 100)
        callbacks.append(EncoderGraphLogger(every_n_steps=graph_log_cadence))

    # Add PoseLogger if using encoder pose
    if getattr(cfg, 'use_encoder_pose', False):
        pose_log_cadence = getattr(cfg.train, 'log_every', 50)
        augmentation_kind = cfg.augmentation.kind if hasattr(cfg, 'augmentation') else 'se3'
        callbacks.append(PoseLogger(every_n_steps=pose_log_cadence, augmentation_kind=augmentation_kind))
        # The pose as GEOMETRY rather than as a matrix: phi(T) before and after the
        # predicted pose, plus the sample it should land on. Shares save_every with the
        # trajectory export, since both write .vtp and are read the same way.
        callbacks.append(PoseShapeExporter(every_n_steps=save_every,
                                           export_shapes=export_shapes or 2))

    orchestrator = TrainingOrchestrator(
        stepper=stepper,
        dataloader=loader,
        callbacks=callbacks,
        log_dir=cfg.output_dir,
    )
    return orchestrator.run(num_steps=cfg.train.steps)
