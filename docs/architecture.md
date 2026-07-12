# Architecture

System design, module organization, and component interactions.

## Directory Structure

```
src/tmgg/
├── models/                        # Neural network architectures
│   ├── base.py                    # GraphModel base class
│   ├── factory.py                 # Registry-based model creation
│   ├── digress/                   # DiGress GraphTransformer
│   ├── gnn/                       # GNN, NodeVarGNN, GNNSymmetric
│   ├── layers/                    # Shared layers (GCN, MHA, TopKEigenLayer)
│   └── spectral_denoisers/        # Spectral positional encoding models
├── diffusion/                     # Diffusion framework
│   ├── noise_process.py           # NoiseProcess ABC + Categorical/Continuous
│   ├── sampler.py                 # Sampler ABC + CategoricalSampler/ContinuousSampler
│   ├── collectors.py              # StepMetricCollector protocol
│   ├── schedule.py                # NoiseSchedule
│   ├── diffusion_math.py          # KL, log-likelihood, posterior math
│   └── diffusion_sampling.py      # Discrete/continuous sampling utilities
├── data/
│   ├── datasets/                  # GraphData, SyntheticGraphDataset, graph types
│   ├── data_modules/              # BaseGraphDataModule → MultiGraph → GraphDataModule
│   └── noising/                   # NoiseGenerator hierarchy, SizeDistribution
├── experiments/
│   ├── _shared_utils/             # Shared training infrastructure
│   │   ├── lightning_modules/     # BaseGraphModule, DiffusionModule, SingleStepDenoisingModule
│   │   ├── evaluation_metrics/    # GraphEvaluator, MMD metrics, ORCA
│   │   └── orchestration/         # run_experiment, sanity_check, progress bar
│   ├── digress_denoising/         # DiGress denoising runner
│   ├── gnn_denoising/             # GNN denoising runner
│   ├── spectral_arch_denoising/   # Spectral architecture runner
│   ├── discrete_diffusion_generative/  # Discrete diffusion + custom datamodule
│   ├── gaussian_diffusion_generative/  # Continuous diffusion runner
│   ├── eigenstructure_study/      # Spectral analysis CLI + modules
│   ├── embedding_study/           # Embedding dimension study
│   └── exp_configs/               # Hydra YAML configs (see docs/configuration.md)
├── utils/
│   └── spectral/                  # Laplacian, spectral deltas
└── modal/                         # Modal cloud execution
    ├── cli.py                     # CLI: spawn, sweep, aggregate
    └── _lib/                      # Modal function definitions, image config
```

## Core Abstractions

### GraphModel

Base class for all graph models (`models/base.py`). Defines the unified forward signature that all training modules expect: `forward(data: GraphData, t: Tensor | None) -> GraphData`. The `t` parameter carries a normalised diffusion timestep for generative models; denoising models ignore it.

### Lightning Module Hierarchy

`BaseGraphModule` (`_shared_utils/lightning_modules/base_graph_module.py`) provides optimizer/scheduler construction, parameter logging, and batch device transfer. No training logic.

`DiffusionModule` (`_shared_utils/lightning_modules/diffusion_module.py`) implements multi-step diffusion training, composing injected `NoiseProcess`, `Sampler`, `NoiseSchedule`, and `GraphEvaluator` components. Handles VLB estimation during validation.

`SingleStepDenoisingModule` (`_shared_utils/lightning_modules/denoising_module.py`) subclasses `DiffusionModule` with T=1 and no sampler. Overrides validation to evaluate per noise level.

### GraphEvaluator

Stateless metric computer (`_shared_utils/evaluation_metrics/graph_evaluator.py`). Accepts `refs` and `generated` graph lists, computes MMD (degree, clustering, spectral, orbit), SBM accuracy, planarity, uniqueness, and novelty. Both Lightning modules call it at epoch end with graphs pulled from the datamodule.

### Data Modules

`BaseGraphDataModule` defines the contract. Two concrete hierarchies:

`MultiGraphDataModule` → `GraphDataModule` serves denoising experiments with adjacency-tensor batches. `SyntheticCategoricalDataModule` (in `experiments/discrete_diffusion_generative/datamodule.py`) serves categorical diffusion with one-hot encoded features. Both produce `GraphData` batches.

## Execution Flow

```
CLI (e.g., tmgg-spectral-arch)
    │
    ▼
@hydra.main loads composed YAML config
    │
    ▼
run_experiment(config)
    ├── set seed, configure matmul precision
    ├── hydra.utils.instantiate(config.data)   → DataModule
    ├── hydra.utils.instantiate(config.model)  → LightningModule (with nested model)
    ├── create callbacks + loggers from config
    ├── hydra.utils.instantiate(config.trainer) → Trainer
    ├── trainer.fit(model, datamodule)
    ├── trainer.test(model, datamodule)
    └── return results dict
```

Hydra's `_target_` fields in YAML configs specify which classes to instantiate. The model config's outer `_target_` points to the Lightning module; the nested `model._target_` points to the `GraphModel` subclass. See `docs/configuration.md` for the config hierarchy.

## Design Patterns

**Factory + Registry:** `ModelRegistry` in `models/factory.py` maps string identifiers to model constructors. Used by the factory function `create_model()` and for model discovery via `ModelRegistry.list_models()`.

**Composition over inheritance:** `DiffusionModule` composes `NoiseProcess`, `Sampler`, `NoiseSchedule`, and `GraphEvaluator` as injected dependencies. Different experiment types combine different implementations of these components.

**Strategy via config:** Noise generators, optimizers, schedulers, and loggers are interchangeable through YAML configuration. The same training loop works with different strategies without code changes.
