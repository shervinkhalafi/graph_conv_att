# Configuration

TMGG uses Hydra for configuration. All configs live under
`src/tmgg/experiments/exp_configs/`.

## Config hierarchy

Two independent inheritance families cover all experiment types.

```
_base_infra.yaml                         ← shared: optimizer, scheduler, seed, paths, W&B
├── base_config_denoising.yaml           ← adds: task/denoising (loss, noise, data)
│   ├── base_config_spectral_arch.yaml   ← selects: models/spectral/linear_pe
│   ├── base_config_gnn.yaml             ← selects: models/gnn/standard_gnn
│   ├── base_config_gnn_transformer.yaml ← selects: models/hybrid/hybrid_with_transformer
│   ├── base_config_digress.yaml         ← selects: models/digress/digress_transformer
│   └── base_config_baseline.yaml        ← selects: models/baselines/linear
├── base_config_discrete_diffusion_generative.yaml  ← own data + noise (no task/denoising)
└── base_config_gaussian_diffusion.yaml             ← own data + noise (no task/denoising)

_base_structural_study.yaml              ← minimal: wandb_entity, seed, paths (no training)
├── base_config_eigenstructure.yaml
└── base_config_embedding_study.yaml
```

Generative experiments (`discrete_diffusion_generative`, `gaussian_diffusion`) inherit
`_base_infra` directly and skip the denoising task config entirely, because they manage
their own data and noise schedules. Structural study experiments do not inherit
`_base_infra` at all; they carry no trainer, optimizer, or data group.

## Directory layout

`base/` holds trainer, logger, callback, and progress bar defaults composed by
`_base_infra`. `models/` contains architecture configs organized by family: `spectral`,
`gnn`, `digress`, `discrete`, `baselines`, and `hybrid`. `data/` defines datasets —
SBM variants at several sizes, grid, single-graph fixtures, and PyG benchmarks (ENZYMES,
PROTEINS). `task/` holds task-specific config groups; currently only `denoising.yaml`.
`stage/` provides overrides for multi-stage sweeps (learning rate, noise levels, batch
size). `report/` contains analysis templates for eigenstructure, benchmarks, and
diffusion quality.

## Inspecting configs

```bash
uv run tmgg-spectral-arch --cfg job              # Print fully composed config
uv run tmgg-spectral-arch --cfg job --package model  # Print just the model section
uv run tmgg-spectral-arch --info config          # Show which file provided each value
```

## Common CLI overrides

```bash
uv run tmgg-gnn model=models/gnn/symmetric_gnn       # Switch model architecture
uv run tmgg-spectral-arch data=sbm_n200               # Switch dataset
uv run tmgg-digress 'noise_levels=[0.1,0.3,0.5]'     # Override noise levels
uv run tmgg-spectral-arch trainer.max_steps=5000      # Change training length
uv run tmgg-experiment +stage=stage2_validation       # Load stage overrides
```

## Hydra composition quick reference

**`_self_`** must appear last in the `defaults` list for a file's explicit keys to take
precedence over anything composed above it. All TMGG base configs follow this convention.
Placing `_self_` first would reverse the priority, letting composed defaults silently
overwrite values declared in the file.

**`@package _global_`** at the top of a config file merges its keys directly into the
global namespace rather than nesting them under the config group name. Use this when
you want a file's contents to appear at the root of the composed config rather than
under a sub-key.

**`@model` package target** — writing `models/gnn/standard_gnn@model` in a defaults list
mounts that config under the `model:` key in the composed output. Without the `@model`
annotation the keys would land at the root, colliding with other top-level keys.

**`${...}` interpolation** — OmegaConf resolves references lazily at access time, so a
model config can declare `noise_type: ${data.noise_type}` before the `data` group is
composed. Accessing a key whose target does not exist raises `InterpolationKeyError`
immediately; there is no silent fallback.
