# ResNet-LDDMM Architecture

A clean, modular implementation of **neural ODE diffeomorphic shape registration**, supporting both single-pair (Milestone A) and cohort-based (Milestone B) training with swappable components and lazy configuration.

---

## 1. Overview

This codebase registers 3D shapes by learning a **velocity field** that deforms the source geometry to match the target. The learned deformation is a **diffeomorphism** (invertible, smooth, structure-preserving) enforced through:

- **Neural ODE integration**: forward Euler for forward integration, modified Euler for inverse
- **Flow regularization**: kinetic energy penalty on velocity magnitude
- **Learnable codes** (optional): per-shape latent vectors enable amortisation across cohorts
- **Pluggable data terms**: Chamfer distance, weighted variants, normal-aware, and optimal transport

### Two Milestones, One Backend

| Axis | **Milestone A (PoC)** | **Milestone B (Amortised)** |
|------|------|---------|
| **Field** | Time-varying (per-step blocks) | Stationary (one net, reused) |
| **Codes** | None (per-pair) | Auto-decoder or encoder |
| **Training** | One pair at a time | Whole cohort, batch training |
| **Inverse** | (n/a) | Modified Euler backward |
| **Loss** | Unidirectional | Bidirectional |

**Design principle**: Both milestones instantiate the same architecture. The orthogonal axes (time-dependence and code-conditioning) are independent — a valid config matrix has **four cells**, three implemented here (fourth is a free ablation).

---

## 2. Architectural Principles

### 2.1 Contracts Over Implementation

Four core **abstract base classes** define the seams where components plug in:

```
VelocityField (ABC)
  → TimeVaryingField, StationaryField
  
ShapeCode (ABC)
  → NoCode (unconditioned)
  → AutoDecoderCodes (per-shape embeddings)
  → EncoderCodes (learned encoder)

DataTerm (ABC)
  → CDData (Chamfer distance)
  → L2Data (point-wise MSE)
  → WeightedCDData, PCDData, NCDData (variants)
  → SinkhornData (optimal transport)

Integrator (ABC)
  → ForwardEuler (q_{k+1} = q_k + dt·v_k)
  → ModifiedEuler (implicit backward step)

Conditioning (ABC)
  → NoConditioning (identity)
  → ConcatConditioning (direct append)
  → FiLMConditioning
  → PositionAware (AD-SVFD: grid interpolation)
```

Each ABC specifies:
- **Signature**: what arguments the method takes
- **Semantics**: what the output means (velocity, not position; per-point, not masked)
- **Invariants**: properties that must hold (dt computed once, kinetic energy properly scaled)

### 2.2 Composition Root Pattern

**Only one place creates components**: `runner.build()` and `runner.run()`.

Leaf modules **never call `Registry.create`**. This enforces:
1. **No hidden dependencies** — every component injected is visible in `build()`
2. **Testability** — tests inject mocks, not relying on registry state
3. **Layering** — config (YAML) never imported by physics code

The composition root:
1. Loads YAML config → `ExperimentCfg` dataclass
2. Branches on `cfg.train.mode` (pair vs. cohort)
3. Creates field, integrators, loss terms via registry
4. Wires them into a stepper (`PairRegistration` or `CohortRegistration`)
5. Hands off to `TrainingOrchestrator` (reused, external)

### 2.3 Immutable Data Flow

Three immutable data structures carry information:

```
YAML config
  ↓
ExperimentCfg (frozen dataclass)
  ↓
Registry.create(...)
  ↓
Flow → Trajectory (frozen dataclass)
  ↓
Loss terms
  ↓
Optimizer step
```

No mutations between stages. This makes:
- **Debugging easy**: stack trace shows which config produced which components
- **Testing deterministic**: same config → same components → same gradients
- **Profiling clear**: no side effects to track

### 2.4 Optional Dependencies (Lazy Loading)

Heavy dependencies load **only when used**:

- **`geomloss`** (optimal transport): imported inside `SinkhornData.forward()`, not at module load time
- **`e3nn`** (equivariant encoding): imported inside `EncoderCodes` only if encoder variant is chosen
- **`graphix`** (pose encoder): same lazy pattern

When a dependency is missing:
- Configs that don't use it still work
- Configs that require it raise `ImportError` at **runtime** (during `forward()`), not import time
- Tests can run with `sys.modules` poisoned to verify lazy-loading guarantee

---

## 3. Core Abstractions and Contracts

### 3.1 VelocityField

```python
class VelocityField(nn.Module, ABC):
    def forward(self, x: Tensor, step: Optional[int] = None, 
                code: Optional[Tensor] = None) -> Tensor:
        """
        x: [B, N, 3] positions
        step: which Euler step is asking (used by time_varying, ignored by stationary)
        code: [B, N_z] per-shape code (None if NoConditioning)
        
        Returns: [B, N, 3] velocity field
        """
```

**Semantics**: Returns **velocity** (d/dt position), never position itself. Never applies activation to a position residual sum (invariant: residual carries positions unperturbed).

**Implementations**:
- **TimeVaryingField**: L blocks (one per Euler step), selected by `step`. Includes optional Conditioning.
- **StationaryField**: One backbone, reused K times. Structure: FA-NN → Fourier PE → DF-NN.

### 3.2 Integrator

```python
class Integrator(ABC):
    def integrate(self, q0: Tensor, field: VelocityField, 
                  code: Optional[Tensor], num_steps: int) -> Trajectory:
        """
        q0: [B, N, 3] initial positions
        Returns: Trajectory with points [K+1,B,N,3] and velocities [K,B,N,3]
        """
```

**Semantics**: Computes `dt = 1/num_steps` **once, here** (sole owner of this invariant). Integration is stateless — no memory of previous steps.

**Implementations**:
- **ForwardEuler**: q_{k+1} = q_k + dt·v_k (direct)
- **ModifiedEuler**: Predictor-corrector backward step (inverse)

Both return a `Trajectory`, which stores:
- `points`: [K+1, B, N, 3] — shapes at steps 0…K
- `velocities`: [K, B, N, 3] — velocities during integration
- `dt`: scalar, used by kinetic energy calculation

### 3.3 ShapeCode

```python
class ShapeCode(nn.Module, ABC):
    def forward(self, batch) -> Optional[Tensor]:
        """batch: batch-like with shape info (e.g., shape_ids)
        Returns: [B, n_z] per-shape codes, or None"""
    
    def penalty(self) -> Optional[Tensor]:
        """Returns: scalar regularization loss, or None"""
```

**Semantics**: Maps batch → codes (or None). The stepper never knows which implementation is used.

**Implementations**:
- **NoCode**: Returns `None, None` (unconditioned, Milestone A)
- **AutoDecoderCodes**: `nn.Embedding(num_shapes, n_z)`, indexed by shape id. `penalty()` returns ‖Z‖²
- **EncoderCodes**: Wraps GroupEncoder. `penalty()` returns None (encoder is pretrained)

### 3.4 DataTerm

```python
class DataTerm(RegistrationLoss, ABC):
    def forward(self, pred: Tensor, target: Tensor,
                pred_w: Optional[Tensor] = None,
                tgt_w: Optional[Tensor] = None,
                normals: Optional[Tensor] = None) -> Tensor:
        """
        pred: [B, N, 3] predicted point cloud
        target: [B, M, 3] target point cloud
        pred_w, tgt_w: [B, N] / [B, M] per-point weights (optional)
        normals: [B, M, 3] target surface normals (optional)
        
        Returns: scalar loss
        """
```

**Semantics**: Maps predicted and target point clouds → scalar loss. Signature is **consistent across all variants** — extra signals (weights, normals) flow through the same interface.

**Implementations**:
- **CDData**: Chamfer distance via reused `chamfer_loss`
- **L2Data**: Per-point MSE
- **WeightedCDData**: Chamfer with target-point weighting (area-weighted)
- **PCDData**: Symmetric point cloud distance
- **NCDData**: Normal-aware: penalizes misaligned normals
- **SinkhornData**: Optimal transport via geomloss (lazily loaded)

### 3.5 Conditioning

```python
class Conditioning(nn.Module, ABC):
    dim: int  # per-point feature dimension
    
    def forward(self, x: Tensor, code: Optional[Tensor]) -> Optional[Tensor]:
        """
        x: [B, N, 3] positions
        code: [B, N_z] per-shape code
        
        Returns: [B, N, dim] per-point features, or None if dim == 0
        """
```

**Semantics**: Broadcasts a global code to per-point features. The field decides how to use them; conditioning decides how to expand them.

**Implementations**:
- **NoConditioning**: `dim=0`, returns None
- **ConcatConditioning**: Broadcasts code, returns `[B, N, n_z]`
- **PositionAware**: Grid-head linear layer → reshape → trilinear interpolation (AD-SVFD)

### 3.6 NeuralODEFlow

Not an ABC, but the facade that wires components together:

```python
class NeuralODEFlow(nn.Module):
    def __init__(self, field: VelocityField, direct: Integrator,
                 inverse: Integrator, num_steps: int):
        self.field = field
        self.direct = direct
        self.inv = inverse
        self.num_steps = num_steps
    
    def forward(self, q0: Tensor, code: Optional[Tensor]) -> Trajectory:
        """Integrate forward"""
        return self.direct.integrate(q0, self.field, code, self.num_steps)
    
    def inverse(self, qT: Tensor, code: Optional[Tensor]) -> Trajectory:
        """Integrate backward"""
        return self.inv.integrate(qT, self.field, code, self.num_steps)
```

---

## 4. Component Organization

### 4.1 Directories and Modules

```
src/resnet_lddmm/
├── cli.py                        # Entry point: YAML → config → run
├── config.py                     # ExperimentCfg and sub-configs
├── runner.py                     # Composition root (build + run)
├── registrations.py              # Registry entries (all components registered here)
│
├── fields/                       # Velocity field implementations
│   ├── base.py                   # VelocityField ABC
│   ├── time_varying.py           # TimeVaryingField (L blocks)
│   ├── blocks.py                 # VelocityBlock, FourierFeatures
│   └── __init__.py
│
├── conditioning/                 # Code → per-point features
│   ├── base.py                   # Conditioning ABC
│   ├── film.py                   # ConcatConditioning, FiLM
│   ├── position_aware.py         # PositionAware (grid interpolation)
│   └── __init__.py
│
├── codes/                        # Shape code implementations
│   ├── base.py                   # ShapeCode ABC
│   ├── none.py                   # NoCode
│   ├── auto_decoder.py           # AutoDecoderCodes (embeddings)
│   ├── encoder.py                # EncoderCodes (lazy-loaded)
│   └── __init__.py
│
├── losses/                       # Loss terms
│   ├── data_terms.py             # DataTerm ABC + implementations
│   ├── mapping_error.py          # Unidirectional/Bidirectional strategies
│   └── __init__.py
│
├── registration/                 # Steppers (training orchestration)
│   ├── pair.py                   # PairRegistration (Milestone A)
│   ├── cohort.py                 # CohortRegistration (Milestone B)
│   ├── subsample.py              # Adaptive subsampling (T28)
│   └── __init__.py
│
├── integrators.py                # Integrator ABC + ForwardEuler, ModifiedEuler
├── flow.py                       # NeuralODEFlow (facade)
├── trajectory.py                 # Trajectory (frozen dataclass)
├── io.py                         # Shape, FrameTransform, load/export
├── callbacks.py                  # TrajectoryExporter, DiagnosticsCallback
├── diagnostics.py                # Jacobian, Lipschitz, triangle flips
├── generate.py                   # Sampling and interpolation (generative)
└── __init__.py
```

### 4.2 External Dependencies (Reused, Not Copied)

Core reused modules:
- **`src.learning.registry`**: Registry system (lazy component loading)
- **`src.learning.losses.composer`**: LossComposer (weighted loss aggregation)
- **`src.learning.trainers.E3_end2end`**: TrainingOrchestrator (training loop, callbacks)
- **`src.learning.callbacks.base`**: Callback interface (logging, checkpointing)
- **`src.learning.loader.loaders`**: OneBatchLoader, CohortBatchLoader
- **`src.vtk`**: VTK I/O (load VTP/OBJ, save trajectories)

These are frozen — their contracts are stable. New features go into `resnet_lddmm`.

---

## 5. Data Flow and Execution

### 5.1 Startup: YAML → Stepper

```
user$ python -m src.resnet_lddmm.cli configs/my_config.yaml

cli.py (entry point)
  ↓ load_config()
ExperimentCfg.from_dict()
  ↓ validation (unknown keys error)
ExperimentCfg (immutable)
  ↓ runner.run(cfg)
runner.build(cfg)
  ├─ Registry.create("field", cfg.field.kind, ...)      → VelocityField
  ├─ ForwardEuler()                                      → Integrator
  ├─ ModifiedEuler()                                     → Integrator
  ├─ NeuralODEFlow(field, direct, inverse, num_steps)    → Flow
  ├─ Registry.create("data_term", cfg.loss.data_name, ...) → DataTerm
  ├─ LossComposer(terms)                                 → Composer
  ├─ Registry.create("code", cfg.code.kind, ...)         → ShapeCode
  ├─ torch.optim.Adam(params, ...)                       → Optimizer
  └─ PairRegistration / CohortRegistration               → Stepper
  ↓
runner.run() continues:
  ├─ build callbacks: VerboseCallback, TrajectoryExporter, DiagnosticsCallback
  ├─ TrainingOrchestrator(stepper, loader, callbacks, ...) → Orchestrator
  └─ orchestrator.run(num_steps=cfg.train.steps)         → training loop
```

### 5.2 Training Loop: One Step

Each step (pair or cohort):

```
Orchestrator.on_step_start()
  ├─ Stepper.train_step(source, target)
  │   ├─ code = code_source(source)          → [B, n_z] or None
  │   ├─ traj_fwd = flow(source.points, code)
  │   │   └─ Integrator.integrate() → loop K steps:
  │   │       └─ v = field(q_k, step=k, code=code) → [B,N,3]
  │   │       └─ q_{k+1} = q_k + (1/K)·v
  │   ├─ traj_bwd = flow.inverse(target.points, code)  [bidirectional only]
  │   ├─ data_loss = data_term(traj_fwd.end, target.points, ...)
  │   ├─ kinetic_loss = traj.kinetic_energy()
  │   ├─ code_reg = code_source.penalty()               [if auto_decoder]
  │   ├─ isometry_loss = iso_loss(traj, field)          [if enabled]
  │   ├─ total_loss = Composer.compute({data, kinetic, code_reg, isometry})
  │   ├─ loss.backward()
  │   └─ optimizer.step()
  │   └─ return (traj_fwd, loss_float, breakdown_dict)
  ├─ Callbacks.on_step_end(...)
  │   ├─ VerboseCallback: print loss
  │   ├─ TrajectoryExporter: export shapes to VTP (every N steps)
  │   └─ DiagnosticsCallback: compute Jacobian, flips, Lipschitz
  └─ update metrics, increment step
```

### 5.3 Data Flow Invariants

Four invariants are enforced structurally:

1. **Positions stay in residual stream**: VelocityField returns velocity, not position. Only integration adds it.
2. **dt = 1/K is computed once** (in Integrator), scaled consistently in kinetic energy.
3. **One frame for all shapes** (joint normalization), enabling position-aware conditioning.
4. **Forward Euler ∘ ModifiedEuler ≈ identity**: property test guards invertibility.

---

## 6. Configuration System

### 6.1 Dataclass Hierarchy

```python
ExperimentCfg
  ├─ source: str           # pair: path; cohort: folder
  ├─ target: str
  ├─ output_dir: str
  ├─ field: FieldCfg
  │   ├─ kind: str              # time_varying | stationary
  │   ├─ num_steps: int         # K (configured once, here)
  │   ├─ width: int             # block width (time_varying)
  │   ├─ activation: str
  │   ├─ fa: tuple              # stationary FA widths
  │   ├─ df: tuple              # stationary DF widths
  │   └─ fourier_n_e: int
  ├─ code: CodeCfg
  │   ├─ kind: str              # none | auto_decoder | encoder
  │   ├─ n_z: int               # code dimension
  │   ├─ conditioning: str       # none | concat | position_aware
  │   └─ grid: int, grid_channels: int
  ├─ loss: LossCfg
  │   ├─ data_name: str         # chamfer | l2 | weighted_chamfer | ...
  │   ├─ data_kwargs: dict
  │   ├─ direction: str         # forward | bidirectional
  │   ├─ sigma: float           # data term weight = 1/(2σ²)
  │   ├─ kinetic_weight: float
  │   ├─ code_reg_weight: float
  │   ├─ weight_decay: float
  │   └─ isometry_weight: float
  └─ train: TrainCfg
      ├─ mode: str             # pair | cohort
      ├─ steps: int
      ├─ batch: int            # cohort only
      ├─ lr: float
      ├─ seed: int | None
      └─ log_every, save_every, export_shapes, ...
```

### 6.2 Config Validation

`ExperimentCfg.from_dict(data)` enforces:
1. All required top-level keys present (source, target, output_dir)
2. **No unknown keys** — typos in YAML caught immediately
3. Nested configs (field, code, loss, train) have defaults

### 6.3 Configuration Examples

**Milestone A (time-varying + none)**:
```yaml
field: { kind: time_varying, num_steps: 10, width: 512, activation: relu }
code: { kind: none }
loss: { data_name: chamfer, direction: forward, sigma: 0.1, kinetic_weight: 1.0 }
train: { mode: pair, steps: 2000, lr: 1e-4 }
```

**Milestone B (stationary + auto_decoder)**:
```yaml
field: { kind: stationary, num_steps: 10, fa: [64,64,64], df: [256,256,256,256,256], fourier_n_e: 3 }
code: { kind: auto_decoder, n_z: 256, conditioning: position_aware }
loss: { data_name: chamfer, direction: bidirectional, kinetic_weight: 1e-4, code_reg_weight: 1e-3 }
train: { mode: cohort, steps: 2000, batch: 8, lr: 1e-3 }
```

**Phase 9 ablation (time-varying + auto_decoder)**:
```yaml
field: { kind: time_varying, num_steps: 10, width: 512 }
code: { kind: auto_decoder, n_z: 256 }  ← orthogonal axes, same interface
loss: { data_name: chamfer, direction: bidirectional }
train: { mode: cohort, steps: 2000 }
```

---

## 7. The Composition Root

`runner.py` is **the only place** that:
1. Calls `Registry.create(...)`
2. Assembles components into a stepper
3. Wires everything together

### 7.1 build(cfg) → Stepper

The `build()` function:

```python
def build(cfg: ExperimentCfg) -> (Stepper, Loader, Transform):
    # 1. Field (branching on kind)
    if cfg.field.kind == "stationary":
        field = Registry.create("field", "stationary", activation=...)
    else:
        field = Registry.create("field", cfg.field.kind, 
                               num_blocks=cfg.field.num_steps, 
                               width=cfg.field.width, ...)
    
    # 2. Integrators
    integrator_direct = ForwardEuler()
    integrator_inverse = ModifiedEuler()
    
    # 3. Flow (wires field + integrators)
    flow = NeuralODEFlow(field, integrator_direct, integrator_inverse, 
                         cfg.field.num_steps)
    
    # 4. Loss terms (via registry)
    data_term = Registry.create("data_term", cfg.loss.data_name, **cfg.loss.data_kwargs)
    iso_loss = Registry.create("iso_loss", "isometry", ...) if cfg.loss.isometry_weight > 0 else None
    
    # 5. Loss composer
    composer = LossComposer([
        LossTerm("data", weight=1/(2*cfg.loss.sigma**2)),
        LossTerm("kinetic", weight=cfg.loss.kinetic_weight),
        LossTerm("code_reg", weight=cfg.loss.code_reg_weight),
        LossTerm("isometry", weight=cfg.loss.isometry_weight),
    ])
    
    # 6. Mapping error strategy
    if cfg.loss.direction == "bidirectional":
        mapping_error = BidirectionalMappingError()
    else:
        mapping_error = UnidirectionalMappingError()
    
    # 7. Branch on training mode
    if cfg.train.mode == "cohort":
        return _build_cohort(cfg, flow, data_term, mapping_error, composer, iso_loss)
    else:
        return _build_pair(cfg, flow, data_term, mapping_error, composer, iso_loss)
```

### 7.2 _build_pair() vs _build_cohort()

**Pair** (Milestone A):
- Loads two shapes, normalizes jointly
- Creates `OneBatchLoader` (returns same batch every step)
- Code source: `Registry.create("code", "none")` → NoCode
- Optimizer: single param group (flow only)
- Stepper: `PairRegistration`

**Cohort** (Milestone B):
- Loads all shapes from directory, normalizes jointly with template
- Creates `CohortBatchLoader` (batches shapes, samples each training step)
- Code source: `Registry.create("code", "auto_decoder", num_shapes=..., n_z=...)`
- Optimizer: **two param groups** (flow with weight_decay, codes without)
- Stepper: `CohortRegistration`

The steppers (`PairRegistration`, `CohortRegistration`) differ only in:
- How they index codes (per-batch vs. per-shape)
- Whether they expose `infer()` (code-only fine-tuning)
- The data loading strategy

---

## 8. Entry Point: CLI

`cli.py` is minimal:

```python
def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    
    cfg = load_config(args.config)          # YAML → ExperimentCfg
    ctx = run(cfg)                          # runner.run()
    print(f"Output saved to: {cfg.output_dir}")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
python -m src.resnet_lddmm.cli configs/rabbit_cohort.yaml
```

The CLI:
1. Validates YAML syntax and config structure (unknown keys catch typos)
2. Passes to `runner.run()` (which calls `runner.build()` and `orchestrator.run()`)
3. Catches exceptions and exits cleanly

---

## 9. Key Design Patterns

### 9.1 Null Object Pattern

**NoCode** and **NoConditioning** represent the absence of a feature:

```python
class NoCode(ShapeCode):
    def forward(self, batch) -> None: return None
    def penalty(self) -> None: return None

class NoConditioning(Conditioning):
    dim = 0
    def forward(self, x, code) -> None: return None
```

The stepper doesn't branch on `if code is None`; it just passes `None` through the entire flow. Loss terms check `if value is None` and skip it.

### 9.2 Facade Pattern

**NeuralODEFlow** wraps field + two integrators:

```python
class NeuralODEFlow:
    def forward(self, q0, code) -> Trajectory:
        return self.direct.integrate(q0, self.field, code, self.num_steps)
    
    def inverse(self, qT, code) -> Trajectory:
        return self.inv.integrate(qT, self.field, code, self.num_steps)
```

Steppers call `flow.forward()` and `flow.inverse()` without knowing about integrators.

### 9.3 Strategy Pattern

**MappingError** encapsulates loss direction (unidirectional vs bidirectional):

```python
class MappingError(ABC):
    def __call__(self, flow, data_term, source, target, code):
        """Compute (data_loss, kinetic_loss)"""

class UnidirectionalMappingError(MappingError):
    def __call__(self, ...):
        traj = flow(source.points, code)
        data = data_term(traj.end, target.points, ...)
        kinetic = traj.kinetic_energy()
        return data, kinetic

class BidirectionalMappingError(MappingError):
    def __call__(self, ...):
        traj_fwd = flow(source.points, code)
        traj_bwd = flow.inverse(target.points, code)
        data = data_term(traj_fwd.end, target.points, ...) + \
               data_term(traj_bwd.end, source.points, ...)
        kinetic = traj_fwd.kinetic_energy() + traj_bwd.kinetic_energy()
        return data, kinetic
```

The stepper just calls `mapping_error(...)` without caring which strategy.

### 9.4 Registry Pattern

**Lazy component loading** via `Registry.create(category, name, **kwargs)`:

```python
# In registrations.py:
Registry.register("field", "time_varying", "src.resnet_lddmm.fields.time_varying:TimeVaryingField")

# In runner.py:
field = Registry.create("field", cfg.field.kind, num_blocks=..., width=..., activation=...)
```

Registry:
1. Looks up the string `"src.resnet_lddmm.fields.time_varying:TimeVaryingField"`
2. Imports the module (if not already imported)
3. Gets the class
4. Calls `TimeVaryingField(**kwargs)`
5. Returns the instance

Benefits:
- No circular imports (modules register themselves at load time)
- Tests can mock `Registry._entries` to inject stubs
- New components register by adding one line (no runner changes needed)

### 9.5 Lazy Loading for Optional Dependencies

**geomloss** (Sinkhorn optimal transport) is optional:

```python
class SinkhornData(DataTerm):
    def _get_loss_fn(self):
        if not hasattr(self, '_loss_fn'):
            import geomloss  # ← Imported here, not at module level
            self._loss_fn = geomloss.SamplesLoss("sinkhorn", ...)
        return self._loss_fn
    
    def forward(self, pred, target, ...):
        loss_fn = self._get_loss_fn()  # ← Raises ImportError if geomloss missing
        return loss_fn(pred, target)
```

Configs without Sinkhorn work even if geomloss is uninstalled. Configs using Sinkhorn fail only at `forward()` time.

---

## 10. Tracing a Request: End-to-End

**User runs**: `python -m src.resnet_lddmm.cli configs/rabbit_cohort.yaml`

### Step 1: Startup
```
cli.py: main()
  → argparse.parse_args()
  → load_config("configs/rabbit_cohort.yaml")
    → yaml.safe_load()
    → ExperimentCfg.from_dict(data)
    → validate (no unknown keys)
    → return ExperimentCfg instance
```

### Step 2: Composition
```
runner.run(cfg)
  → build(cfg)
    → Registry.create("field", "stationary", ...)
      → import src.resnet_lddmm.fields.time_varying
      → StationaryField(...)
    → ForwardEuler()
    → ModifiedEuler()
    → NeuralODEFlow(field, direct, inverse, num_steps=10)
    → Registry.create("data_term", "chamfer")
      → CDData()
    → Registry.create("code", "auto_decoder", num_shapes=3, n_z=256)
      → AutoDecoderCodes(num_shapes, n_z)
    → Registry.create("iso_loss", "isometry", loss_type="strain", sample_points=64)
      → IsometryLoss(...)
    → LossComposer([data, kinetic, code_reg, isometry])
    → torch.optim.Adam([flow params, code params], lr=1e-3)
    → CohortRegistration(flow, codes, data_term, bidirectional, composer, optimizer, iso_loss)
    → return (stepper, loader, transform)
  → build callbacks
    → VerboseCallback(log_every=50)
    → TrajectoryExporter(every_n_steps=500, export_shapes=2)
    → DiagnosticsCallback(every_n_steps=500)
  → TrainingOrchestrator(stepper, loader, callbacks, log_dir=output_dir)
```

### Step 3: Training Loop (per step)
```
orchestrator.run(num_steps=2000)
  for step in range(2000):
    batch = loader.next_batch()  # [B shapes, target template]
    stepper.train_step(batch[0], batch[1])
      → code_source(batch[0])     # AutoDecoderCodes[batch indices]
      → traj_fwd = flow(source_points, code)
        → ForwardEuler.integrate(q0, field, code, 10)
          for k in range(10):
            v = field(q_k, step=k, code=code)  # StationaryField or TimeVaryingField
            q_{k+1} = q_k + 0.1 * v
          return Trajectory(points=[q0, ..., q10], velocities=[v0, ..., v9], dt=0.1)
      → traj_bwd = flow.inverse(target_points, code)
        → ModifiedEuler.integrate(qT, field, code, 10)
          [similar, but backward with reversed steps]
      → data_loss = data_term(traj_fwd.end, target.points, target.weights, normals=...)
        → Chamfer distance (or weighted variant)
      → kinetic_loss = traj_fwd.kinetic_energy() + traj_bwd.kinetic_energy()
      → code_reg_loss = code_source.penalty()
        → ‖Z‖²_F (Frobenius norm of embedding table)
      → isometry_loss = iso_loss(traj_fwd, field)
      → total = composer.compute({data, kinetic, code_reg, isometry})
        → 1/(2σ²)·data + weight_kinetic·kinetic + weight_code_reg·code_reg + weight_isometry·isometry
      → total.backward()
      → optimizer.step()
      → return (traj_fwd, loss_value, breakdown)
    callbacks.on_step_end(step, loss, traj)
      → VerboseCallback: print loss
      → TrajectoryExporter: export VTPs if step % save_every == 0
      → DiagnosticsCallback: compute diagnostics if step % save_every == 0
```

### Step 4: Output
```
outputs/rabbit_cohort/
  config.json                      # Provenance (resolved config)
  metrics.jsonl                    # Per-step losses
  step_0000/
    shape_0.vtp                    # First shape at step 0
    shape_1.vtp
    ...
  step_0500/
    shape_0.vtp                    # After 500 steps
    ...
```

---

## 11. Testing Philosophy

### 11.1 Unit Tests: Component in Isolation

Each ABC and implementation is tested independently:

```python
class TestTimeVaryingField:
    def test_output_shape(self):
        field = TimeVaryingField(num_blocks=10, width=256, ...)
        x = torch.randn(2, 100, 3)
        v = field(x, step=0, code=None)
        assert v.shape == (2, 100, 3)

    def test_distinct_blocks(self):
        # Verify each block has different parameters
        field = TimeVaryingField(num_blocks=10, ...)
        params_0 = list(field.blocks[0].parameters())
        params_1 = list(field.blocks[1].parameters())
        assert not torch.allclose(params_0[0], params_1[0])
```

### 11.2 Integration Tests: Components Together

```python
class TestPairRegistrationIntegration:
    def test_loss_decreases(self):
        # Load real hand shapes, run 20 steps, check loss decreases
        cfg = hand_pair_config()
        stepper, loader, _ = runner.build(cfg)
        
        losses = []
        for step in range(20):
            source, target = loader.next_batch()
            _, loss, _ = stepper.train_step(source, target)
            losses.append(loss)
        
        assert losses[-1] < losses[0]  # Loss should decrease
```

### 11.3 Contract Tests: ABC Compliance

```python
def test_velocity_field_returns_velocity(field: VelocityField):
    # Any VelocityField implementation must satisfy this
    x = torch.randn(2, 100, 3)
    v = field(x, step=0, code=None)
    
    # Velocity should be small magnitude (not positions)
    assert v.abs().max() < 10.0
```

### 11.4 Lazy Loading Tests

```python
def test_auto_decoder_works_without_geomloss(monkeypatch):
    # Poison geomloss from sys.modules
    import sys
    monkeypatch.setitem(sys.modules, "geomloss", None)
    
    # AutoDecoderCodes should still work
    cfg = config_with_auto_decoder_no_sinkhorn()
    stepper, _, _ = runner.build(cfg)  # Should not raise
    
    # But SinkhornData would raise at forward() time
    cfg2 = config_with_sinkhorn()
    stepper2, _, _ = runner.build(cfg2)  # Should not raise (lazy)
    with pytest.raises(ImportError):
        stepper2.train_step(...)  # ← Raises here
```

---

## 12. Extension Points

To add a new component, follow this recipe:

### Adding a New VelocityField

1. **Create** `src/resnet_lddmm/fields/my_field.py`:
   ```python
   from src.resnet_lddmm.fields.base import VelocityField
   
   class MyField(VelocityField):
       def __init__(self, num_blocks, width, activation, conditioning=None):
           super().__init__()
           self.conditioning = conditioning or NoConditioning()
           # your layers
       
       def forward(self, x, step=None, code=None):
           # return velocity
   ```

2. **Register** in `src/resnet_lddmm/registrations.py`:
   ```python
   Registry.register("field", "my_field", "src.resnet_lddmm.fields.my_field:MyField")
   ```

3. **Test** in `tests/resnet_lddmm/test_fields.py`:
   ```python
   def test_my_field_shape():
       field = MyField(num_blocks=10, width=256, activation="relu")
       x = torch.randn(2, 100, 3)
       v = field(x, step=0, code=None)
       assert v.shape == (2, 100, 3)
   ```

4. **Use** in config:
   ```yaml
   field: { kind: my_field, num_blocks: 10, width: 256, activation: relu }
   ```

### Adding a New DataTerm

1. **Create** `src/resnet_lddmm/losses/data_terms.py` (extend existing file):
   ```python
   class MyDataTerm(DataTerm):
       def forward(self, pred, target, pred_w=None, tgt_w=None, normals=None):
           # Compute loss
           return loss  # scalar tensor
   ```

2. **Register**:
   ```python
   Registry.register("data_term", "my_loss", "src.resnet_lddmm.losses.data_terms:MyDataTerm")
   ```

3. **Test**:
   ```python
   def test_my_data_term_identical_clouds():
       term = MyDataTerm()
       pred = torch.randn(2, 100, 3)
       loss = term(pred, pred)
       assert loss.item() < 1e-4  # Should be ~0
   ```

4. **Use** in config:
   ```yaml
   loss: { data_name: my_loss, data_kwargs: {} }
   ```

---

## 13. Common Pitfalls and Invariants

### 13.1 VelocityField Invariants

❌ **Wrong**: Field returns position-like values
```python
def forward(self, x, step, code):
    return self.layer(x)  # ← x already contains position + activation
```

✓ **Right**: Field returns velocity only
```python
def forward(self, x, step, code):
    return self.layer(x)  # ← called on position, returns velocity
```

❌ **Wrong**: Applying activation to residual sum
```python
def forward(self, x, step, code):
    return torch.relu(self.fc1(x) + self.fc2(x))  # ← ReLU on sum
```

✓ **Right**: Activation inside, residual carries positions
```python
def forward(self, x, step, code):
    return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
```

### 13.2 Integrator Invariant

❌ **Wrong**: dt computed per-step
```python
for k in range(num_steps):
    dt = 1.0 / (num_steps - k)  # ← Varies per step!
    q = q + dt * v
```

✓ **Right**: dt = 1/K computed once
```python
dt = 1.0 / num_steps  # ← Single source of truth
for k in range(num_steps):
    q = q + dt * v
```

### 13.3 Kinetic Energy Invariant

❌ **Wrong**: Forgetting dt scaling
```python
kinetic = 0.5 * velocities.pow(2).sum()  # ← Missing dt
```

✓ **Right**: Scaled by dt (integral approximation)
```python
kinetic = 0.5 * velocities.pow(2).sum(-1).mean() * dt
```

### 13.4 Frame Invariant

❌ **Wrong**: Shapes normalized separately
```python
source_norm = normalize(source)
target_norm = normalize(target)  # ← Different frames!
```

✓ **Right**: Joint normalization (one frame)
```python
normalized = joint_normalize([source, target])
source_norm, target_norm = normalized  # ← Same frame
```

---

## 14. Summary

**ResNet-LDDMM** is a modular shape registration system built on:

1. **Four core ABCs** (VelocityField, Integrator, ShapeCode, DataTerm, Conditioning)
   - Define contracts, enable swapping
   - Implementations added without changing existing code

2. **Composition root pattern** (runner.py)
   - Single `build()` and `run()` function
   - No leaf modules call Registry
   - Enables testing and debugging

3. **Configuration-first design**
   - YAML → dataclass → components
   - Unknown keys caught early
   - Supports the full 2×2 design space

4. **Lazy loading for optional dependencies**
   - geomloss, e3nn imported only if used
   - Configs degrade gracefully

5. **Orthogonal axes**
   - Time-dependence (time-varying vs. stationary) ⊥ code-conditioning (none vs. auto-decoder vs. encoder)
   - Four valid config combinations (three implemented, fourth is free ablation)

An engineer reading this document should be able to:
- Trace a request from YAML config to tensor operations
- Add a new component (field, loss, code) in five lines
- Understand why design decisions were made (invariants, contracts)
- Test in isolation and integration
- Debug issues by inspecting the composition root

