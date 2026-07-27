from dataclasses import dataclass, field as dataclass_field
from typing import Optional


@dataclass
class FieldCfg:
    kind: str = "time_varying"              # time_varying | stationary
    num_steps: int = 10                     # K — configured in ONE place
    width: int = 512                        # block width (ignored by stationary)
    activation: str = "relu"
    fa: tuple = (64, 64, 64)                # stationary: FA-NN layer widths
    df: tuple = (256, 256, 256, 256, 256)   # DF-NN layer widths
    fourier_n_e: int = 3


@dataclass
class CodeCfg:
    kind: str = "none"                      # none | auto_decoder | encoder
    n_z: int = 256
    conditioning: str = "none"              # none | concat | position_aware
    grid: int = 2                           # g
    grid_channels: int = 32                 # C — grid-head output per corner


@dataclass
class LossCfg:
    data_name: str = "chamfer"              # chamfer | l2 | emd
    data_kwargs: dict = dataclass_field(default_factory=dict)
    direction: str = "forward"              # forward | bidirectional
    sigma: float = 0.1                      # data weight = 1/(2σ²), named per D4
    kinetic_weight: float = 1.0
    code_reg_weight: float = 0.0
    weight_decay: float = 0.0               # Θ only — see Phase 6
    isometry_weight: float = 0.0  # ← disabled by default
    isometry_type: str = "strain"  # or "det" / "orthogonal"
    isometry_samples: int = 64  # points per step for isometry (64 = ~5x speedup)
    iso_kwargs: dict = dataclass_field(default_factory=lambda: {"loss_type": "strain", "sample_points": 64})
    subsample_M: int = 2000                 # adaptive subsample target size (T28); 0 = disabled
    subsample_a: float = 0.15               # fraction of hard examples to keep (T28)

@dataclass
class TrainCfg:
    mode: str = "pair"                      # pair | cohort
    steps: int = 2000
    batch: int = 8
    subsample: int = 0                      # M; 0 = all points
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
    train: TrainCfg = dataclass_field(default_factory=TrainCfg)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentCfg":
        """Create ExperimentCfg from dict, erroring on unknown keys."""
        allowed_keys = {"source", "target", "output_dir", "field", "code", "loss", "train"}
        unknown = set(d.keys()) - allowed_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

        field_cfg = FieldCfg(**d.get("field", {})) if "field" in d else FieldCfg()
        code_cfg = CodeCfg(**d.get("code", {})) if "code" in d else CodeCfg()
        loss_cfg = LossCfg(**d.get("loss", {})) if "loss" in d else LossCfg()
        train_cfg = TrainCfg(**d.get("train", {})) if "train" in d else TrainCfg()

        return cls(
            source=d["source"],
            target=d["target"],
            output_dir=d["output_dir"],
            field=field_cfg,
            code=code_cfg,
            loss=loss_cfg,
            train=train_cfg,
        )
