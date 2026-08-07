from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, List, Any, Dict


# Where joint_normalize places the shapes -- equivalently, where the ORIGIN sits
# relative to them. Not cosmetic: SO3Augmentation and the encoder pose are both
# applied as ``points @ R``, i.e. about the ORIGIN, so this decides whether a
# rotation is a spin or an orbit.
#
#   "box"    [0,1]^3, centroid at (0.5,0.5,0.5), 0.87 from the origin. A rotation
#            ORBITS the shape further than its own diameter, which gives chamfer a
#            strong pose signal (measured: 64x contrast vs 9x). But a rotation-only
#            pose can then only reach targets whose centroid stays on that sphere,
#            so samples that arrive already rotated in place are NOT reachable.
#   "origin" [-0.5,0.5]^3, centroid at the origin. A rotation is a spin in place, so
#            pre-rotated samples are reachable and the pose head's translation (an
#            absolute centroid) composes correctly. Weaker chamfer signal.
CENTERING_DOMAINS = {"box": (0.0, 1.0), "origin": (-0.5, 0.5)}

# Floating-point precision for the whole run. Named here, resolved to a torch dtype in
# the runner, so this module stays importable without torch.
#
# Nothing in the pipeline hardcodes precision -- every tensor derives its dtype from
# the data (``dtype=q0.dtype``, ``dtype=points.dtype``, ...) -- so this one key
# propagates everywhere by itself.
#
#   "float32"  the default, and what every result before this key was produced with.
#   "float64"  ~2x slower on CPU and considerably worse on GPU. Worth it as a
#              DIAGNOSTIC: the pose vectors are the residual of a first moment that
#              nearly cancels (measured ~1% of the per-node norms), and cancellation
#              is exactly where float32 loses relative precision -- 7 significant
#              digits in, ~5 left in the quantity actually used.
DTYPES = ("float32", "float64")


@dataclass
class FieldCfg:
    kind: str = "time_varying"              # time_varying | stationary | equivariant_stationary
    num_steps: int = 10                     # K — configured in ONE place
    width: int = 512                        # block width (ignored by stationary/equivariant_stationary)
    activation: str = "relu"
    fa: tuple = (64, 64, 64)                # stationary: FA-NN layer widths
    df: tuple = (256, 256, 256, 256, 256)   # DF-NN layer widths
    fourier_n_e: int = 3
    # Equivariant stationary field (only if kind == "equivariant_stationary")
    hidden_irreps: str = "4x0e + 2x1o"      # intermediate irreps representation (single-layer fallback)
    gate_hidden_dim: int = 64               # hidden dim for gating network in SelfInteraction
    use_tensor_product_self: bool = True    # use SelfInteraction (V⊗V+gating) vs simple Linear
    layers: Optional[List[dict]] = dataclass_field(default=None)  # layer stack config for depth
    # Equivariant contextual field (only if kind == "equivariant_contextual")
    context_irreps: str = "32x0e + 16x1o"   # features after message passing aggregation
    sh_lmax: int = 2                        # spherical harmonics lmax for bipartite convolution
    # Simple equivariant contextual field (only if kind == "equivariant_contextual_simple")
    hidden_dim: int = 128                   # hidden dimension in MLP layers
    n_layers: int = 3                       # number of MLP layers


@dataclass
class CodeCfg:
    kind: str = "none"                      # none | auto_decoder | encoder
    n_z: int = 256
    position_aware: bool = False            # use grid interpolation (True) or broadcast (False)?
    conditioning_method: str = "concat"     # concat | film — how to apply features to velocity
    grid: int = 2                           # g — grid resolution (only if position_aware=True)
    grid_channels: int = 32                 # C — grid-head output per corner (only if position_aware=True)

    # Encoder-specific config (only used if kind == "encoder")
    encoder_config: Optional[EncoderConfig] = dataclass_field(default=None)
    graph_spec: Optional[GraphSpec] = dataclass_field(default=None)


@dataclass
class LossCfg:
    data_name: str = "chamfer"              # chamfer | l2 | emd
    data_kwargs: dict = dataclass_field(default_factory=dict)
    direction: str = "forward"              # forward | bidirectional
    sigma: float = 0.1                      # data weight = 1/(2σ²), named per D4
    kinetic_weight: float = 1.0
    code_reg_weight: float = 0.0
    weight_decay: float = 0.0               # Θ only — see Phase 6
    isometry_weight: float = 0.0            # disabled by default
    equivariant_deformation_weight: float = 0.0   # ||phi(g.T) - g.phi(T)||^2; 0 = not computed
    equivariant_deformation_kwargs: dict = dataclass_field(default_factory=dict)  # field_only, translation
    pose_supervision_weight: float = 0.0          # ||R_hat - R_aug||_F^2 against the drawn element; 0 = not computed
    pose_supervision_kwargs: dict = dataclass_field(default_factory=dict)        # translation
    flow_rotation_penalty_weight: float = 0.0     # ||R_flow - I||_F^2; fixes the pose/deformation gauge
    flow_rotation_penalty_kwargs: dict = dataclass_field(default_factory=dict)
    isometry_type: str = "strain"           # or "det" / "orthogonal"
    isometry_samples: int = 64              # points per step for isometry (64 = ~5x speedup)
    subsample_M: float = 2000               # subsample source to N points before flow. int (absolute) or 0<x<1 (fraction); 0 = disabled
    save_full: bool = False                 # when subsampling: also compute and export full trajectory
    # Open-ended extra terms, resolved through the "loss_term" Registry category.
    # Each entry: {kind: <registry name>, weight: float, name: <metric key>, kwargs: {}}
    # Adding a loss is an entry here plus a registry line -- never a stepper edit.
    terms: List[dict] = dataclass_field(default_factory=list)

    def resolve_subsample_count(self, n_source: int) -> int:
        """Resolve subsample_M to absolute point count given source size.

        Args:
            n_source: number of source points

        Returns:
            Absolute number of points to subsample to (or 0 if disabled)
        """
        if self.subsample_M == 0:
            return 0
        if self.subsample_M >= 1:
            return int(self.subsample_M)
        # Treat as fraction
        return max(1, int(self.subsample_M * n_source))

@dataclass
class AugmentationCfg:
    """Configuration for source shape augmentation (random group transformations)."""
    kind: str = "none"                      # none | so3 | se3
    seed: Optional[int] = None              # Random seed for reproducibility
    translation_scale: float = 0.2          # Only for SE3: bounds on uniform translation


@dataclass
class TrainCfg:
    mode: str = "pair"                      # pair | cohort
    steps: int = 2000
    batch: int = 8
    lr: float = 1e-4
    seed: Optional[int] = None
    log_every: int = 50
    save_every: int = 500
    val_every: int = 0
    export_shapes: int = 0                  # 0 = all shapes (pair: always 1, cohort: min(batch_size, cohort_size))
    export_strategy: str = "sequential"     # sequential | random


@dataclass
class ExperimentCfg:
    source: str                             # pair: two paths; cohort: folder + template
    target: str
    output_dir: str
    field: FieldCfg = dataclass_field(default_factory=FieldCfg)
    code: CodeCfg = dataclass_field(default_factory=CodeCfg)
    loss: LossCfg = dataclass_field(default_factory=LossCfg)
    augmentation: AugmentationCfg = dataclass_field(default_factory=AugmentationCfg)
    train: TrainCfg = dataclass_field(default_factory=TrainCfg)
    centering: str = "box"                  # box | origin — see CENTERING_DOMAINS
    dtype: str = "float32"                  # float32 | float64 — see DTYPES
    use_encoder_pose: bool = False          # Apply learned encoder pose to flow output (unidirectional only)
    freeze_flow_at_identity: bool = False   # If True, skip flow computation; encoder learns pose only

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentCfg":
        """Create ExperimentCfg from dict, erroring on unknown keys.

        Handles nested config for encoder_config and graph_spec.
        """
        allowed_keys = {"source", "target", "output_dir", "field", "code", "loss", "augmentation", "train", "centering", "dtype", "use_encoder_pose", "freeze_flow_at_identity"}
        unknown = set(d.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

        field_cfg = FieldCfg(**d.get("field", {})) if "field" in d else FieldCfg()
        loss_cfg = LossCfg(**d.get("loss", {})) if "loss" in d else LossCfg()
        augmentation_cfg = AugmentationCfg(**d.get("augmentation", {})) if "augmentation" in d else AugmentationCfg()
        train_cfg = TrainCfg(**d.get("train", {})) if "train" in d else TrainCfg()

        centering = d.get("centering", "box")
        if centering not in CENTERING_DOMAINS:
            raise ValueError(
                f"unknown centering {centering!r}; expected one of "
                f"{sorted(CENTERING_DOMAINS)}"
            )

        dtype = d.get("dtype", "float32")
        if dtype not in DTYPES:
            raise ValueError(
                f"unknown dtype {dtype!r}; expected one of {list(DTYPES)}"
            )

        # Handle code config with special parsing for encoder_config and graph_spec
        code_dict = d.get("code", {})
        code_cfg = _parse_code_config(code_dict)

        return cls(
            source=d["source"],
            target=d["target"],
            output_dir=d["output_dir"],
            field=field_cfg,
            code=code_cfg,
            loss=loss_cfg,
            augmentation=augmentation_cfg,
            train=train_cfg,
            centering=centering,
            dtype=dtype,
            use_encoder_pose=d.get("use_encoder_pose", False),
            freeze_flow_at_identity=d.get("freeze_flow_at_identity", False),
        )


def _parse_code_config(code_dict: dict) -> CodeCfg:
    """Parse code config, handling nested encoder_config and graph_spec.

    Separates encoder/graph config from base code config and creates
    EncoderConfig/GraphSpec objects when kind == "encoder".

    Note: EncoderConfig/GraphSpec are only imported when needed (lazy import),
    so basic config parsing doesn't require e3nn/torch_geometric.
    """
    # Extract nested configs before passing to CodeCfg
    encoder_cfg_dict = code_dict.pop("encoder_config", {})
    graph_cfg_dict = code_dict.pop("graph_spec", {})

    # Parse encoder_config if provided and kind is encoder
    encoder_cfg = None
    if code_dict.get("kind") == "encoder":
        from src.spec import EncoderConfig, EncoderLayerConfig
        if encoder_cfg_dict:
            # Parse layer configs
            layers_data = encoder_cfg_dict.pop("layers", [])
            layers = [EncoderLayerConfig(**layer) for layer in layers_data]
            encoder_cfg = EncoderConfig(layers=layers, **encoder_cfg_dict)
        else:
            # Use default EncoderConfig
            encoder_cfg = EncoderConfig()

        # Validate n_z matches encoder's latent_dim
        config_n_z = code_dict.get("n_z", 256)
        encoder_latent_dim = encoder_cfg.latent_dim
        if config_n_z != encoder_latent_dim:
            print(f"WARNING: code.n_z={config_n_z} does not match encoder_config.latent_dim={encoder_latent_dim}")
            print(f"         Using encoder_config.latent_dim={encoder_latent_dim} (encoder determines code dimension)")
            code_dict["n_z"] = encoder_latent_dim

    # Parse graph_spec if provided and kind is encoder
    graph_spec = None
    if code_dict.get("kind") == "encoder":
        from src.spec import GraphSpec
        graph_spec = GraphSpec(**graph_cfg_dict) if graph_cfg_dict else GraphSpec()

    return CodeCfg(encoder_config=encoder_cfg, graph_spec=graph_spec, **code_dict)
