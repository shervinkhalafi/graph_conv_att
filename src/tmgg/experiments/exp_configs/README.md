# TMGG Configuration Guide

This directory contains Hydra configuration files for TMGG experiments.

## Quick Start

```bash
# Single experiment (local)
tmgg-experiment +stage=stage1_poc

# Single experiment with model override
tmgg-experiment +stage=stage1_poc model=models/spectral/filter_bank

# Multirun via Hydra (local, built-in launcher)
tmgg-experiment +stage=stage1_poc --multirun \
  model=models/spectral/linear_pe,models/spectral/filter_bank \
  learning_rate=1e-3,1e-2

# Single experiment on Modal
tmgg-modal run tmgg-spectral-arch model.k=16 seed=1

# Single experiment on Modal (dry-run: resolve config, print summary, exit)
tmgg-modal run tmgg-spectral-arch --dry-run

# Single experiment on Modal (detached / fire-and-forget)
tmgg-modal run tmgg-spectral-arch seed=2 --detach --gpu fast

# Multirun on Modal via Hydra launcher
tmgg-experiment +stage=stage1_poc --multirun \
  hydra/launcher=tmgg_modal \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

## Config Precedence

Configuration is composed in the following order (later overrides earlier):

1. `_base_infra.yaml` - shared trainer, logger, callbacks, and path settings
2. `base_config_*.yaml` - experiment-type defaults (for example denoising vs generative)
3. Defaults-listed config groups inside the base config - model, callbacks, logger, and sometimes data
4. `stage/*.yaml` - stage-specific overrides (via `tmgg-experiment +stage=...`)
5. CLI overrides - highest precedence

One important current nuance: some base configs define their `data:` block inline
instead of through a Hydra defaults entry. In those cases, `data=...` will fail
because there is no existing defaults-list slot to override. Appending `+data=...`
adds a packaged config under `data`, which can replace fields including `_target_`.

## Directory Structure

```
exp_configs/
├── _base_infra.yaml             # Shared trainer/logger/callback/path defaults
├── base_config_denoising.yaml   # Shared denoising base
├── base_config_spectral_arch.yaml
├── base_config_gnn.yaml
├── base_config_gnn_transformer.yaml
├── base_config_digress.yaml
├── base_config_gaussian_diffusion.yaml
├── base_config_discrete_diffusion_generative.yaml
├── data/                        # Dataset configurations
│   ├── base_dataloader.yaml     # Shared GraphDataModule defaults
│   ├── single_graph_base.yaml   # Base for single-graph denoising datasets
│   ├── sbm_default.yaml         # Enumerated SBM denoising preset
│   ├── sbm_single_graph.yaml    # Single-graph SBM denoising preset
│   └── sbm_digress.yaml         # Batch-size override for DiGress-style SBM data
├── models/                      # Model architectures
│   ├── spectral/                # Spectral denoising models
│   │   ├── linear_pe.yaml
│   │   ├── filter_bank.yaml
│   │   └── self_attention.yaml
│   ├── digress/                 # DiGress denoising variants
│   └── discrete/                # Discrete diffusion model configs
├── stage/                       # Stage-specific configs
│   ├── stage1_poc.yaml          # Proof of concept
│   ├── stage1_sanity.yaml       # Sanity check
│   └── stage2_validation.yaml   # Cross-dataset validation
└── hydra/                       # Hydra-specific configs
    └── launcher/                # Custom launchers
        └── tmgg_modal.yaml      # Modal cloud execution
```

## Shared Training Defaults

`_base_infra.yaml` currently provides these shared top-level defaults:

```yaml
learning_rate: 1e-3
weight_decay: 1e-4
optimizer_type: adamw
amsgrad: false
```

Experiment-specific model configs can and do override these. For example,
`models/discrete/discrete_sbm_official.yaml` sets the DiGress-style optimizer
values inside the instantiated `DiffusionModule`.

## Available Stages

| Stage | Command | Description |
|-------|---------|-------------|
| stage1_poc | `+stage=stage1_poc` | Initial ER graph training |
| stage1_sanity | `+stage=stage1_sanity` | Constant noise memorization test |
| stage2_validation | `+stage=stage2_validation` | Multi-dataset validation |

For cross-graph validation (generalization test):
```bash
tmgg-experiment +stage=stage2_validation cross_graph=true
```

## Model Architectures

### Spectral Denoising Models

| Config | Description |
|--------|-------------|
| `models/spectral/linear_pe` | Linear positional encoding |
| `models/spectral/filter_bank` | Spectral filter bank |
| `models/spectral/self_attention` | Self-attention baseline |

### DiGress Baselines

| Config | Description |
|--------|-------------|
| `models/digress/digress_sbm_small` | DiGress with official learning rate |
| `models/digress/digress_sbm_small_highlr` | DiGress with high learning rate (ablation) |

## Data Configurations

Most `data/*.yaml` presets are written for the denoising path and inherit from
`data/base_dataloader.yaml`, which sets `_target_: tmgg.data.GraphDataModule`.
The discrete generative base config instead defines its own inline
`SyntheticCategoricalDataModule`.

That means:

- `data=...` works only when the base config already exposes a `data` defaults slot
- `+data=...` appends a packaged `@package data` config and may replace the datamodule target
- appending a denoising preset onto a generative config can therefore swap in the wrong datamodule

Common denoising-oriented presets:

| Config | Description |
|--------|-------------|
| `data/er_single_graph` | Erdos-Renyi random graphs |
| `data/sbm_single_graph` | Stochastic Block Model |
| `data/tree_single_graph` | Random trees |
| `data/pyg_qm9_single_graph` | QM9 molecular graphs |
| `data/pyg_enzymes_single_graph` | ENZYMES protein structures |
| `data/pyg_proteins_single_graph` | PROTEINS dataset |

## Cloud Execution (Modal)

### Single Experiment

`tmgg-modal run` resolves the Hydra config locally via compose API, then dispatches the resolved config to Modal. Output paths are patched to use the Modal volume (`/data/outputs`).

```bash
# Standard GPU (A10G), blocking
tmgg-modal run tmgg-spectral-arch model.k=16 seed=1

# Fast GPU (A100), detached (fire-and-forget)
tmgg-modal run tmgg-spectral-arch seed=2 --detach --gpu fast

# Dry run: resolve config and print summary without dispatching
tmgg-modal run tmgg-spectral-arch --dry-run
```

Available `--gpu` tiers: `debug` (T4), `standard` (A10G, default), `fast` (A100), `multi` (A100x2), `h100` (H100).

### Multirun on Modal

Use Hydra's `--multirun` with the `tmgg_modal` launcher. The launcher patches output paths to the Modal volume (same as `tmgg-modal run`) and dispatches all jobs via `ModalRunner.run_sweep()`.

```bash
tmgg-experiment +stage=stage1_poc --multirun \
  hydra/launcher=tmgg_modal \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

For local sweeps, use Hydra's built-in launcher:
```bash
tmgg-experiment +stage=stage1_poc --multirun hydra/launcher=basic \
  model=models/spectral/linear_pe,models/spectral/filter_bank
```

### W&B Logging

Most training configs default to the W&B logger through `_base_infra.yaml`.
The current shared default is `allow_no_wandb=true`, which degrades missing
credentials to a warning. Set `allow_no_wandb=false` when you want runs to
fail loudly if W&B logging is unavailable:

```bash
tmgg-experiment +stage=stage1_poc allow_no_wandb=false
```

## Common Overrides

```bash
# Change learning rate
tmgg-experiment +stage=stage1_poc learning_rate=5e-3

# Change training steps
tmgg-experiment +stage=stage1_poc trainer.max_steps=10000

# Change batch size
tmgg-experiment +stage=stage1_poc data.batch_size=32

# Allow training without W&B logging
tmgg-experiment +stage=stage1_poc allow_no_wandb=true
```
