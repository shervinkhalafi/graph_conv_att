# Cloud Execution

TMGG runs on Modal through the `tmgg-modal` CLI and the `tmgg.modal.runner.ModalRunner`.
This document describes the current execution surface, the required secrets, and the
paths that still exist in the tree today.

## Current Execution Surface

There are two supported Modal entry points:

- `tmgg-modal run ...` for a single resolved experiment config
- `tmgg-experiment --multirun hydra/launcher=tmgg_modal ...` for Hydra multiruns

The `tmgg-modal` CLI also exposes:

- `tmgg-modal evaluate` to run checkpoint evaluation on Modal GPUs
- `tmgg-modal aggregate` to pull evaluation results back from the Modal volume

The old `spawn_single.py`, `launch_sweep.py`, `generate_configs.py`, and
`stage_definitions/` pipeline no longer exists.

## Architecture

The current Modal path is:

```text
local CLI / Hydra
    |
    +-- tmgg-modal run
    |      |
    |      +-- resolve Hydra config locally
    |      +-- ModalRunner.run_experiment() or .spawn_experiment()
    |
    +-- tmgg-experiment --multirun hydra/launcher=tmgg_modal
           |
           +-- TmggLauncher.launch()
           +-- ModalRunner.run_sweep()

ModalRunner
    |
    +-- modal.Function.from_name("graph-denoising", "modal_run_cli*")
    +-- send fully resolved YAML

Modal container
    |
    +-- tmgg.modal._functions: modal_run_cli / _fast / _debug
    +-- write YAML to a temp config directory
    +-- subprocess.run([cli_cmd, --config-path=..., --config-name=run_config])
```

Important implementation points:

- The deployed Modal app name is `graph-denoising`.
- Runtime lookups go through `modal.Function.from_name(...)`, not direct imports of the decorated functions.
- The deployed functions are `modal_run_cli`, `modal_run_cli_fast`, and `modal_run_cli_debug`.
- The Modal image is built from `tmgg.modal._lib.image.create_tmgg_image()` and includes `graph-tool` and ORCA support so generative evaluation metrics can run in the container.

## Prerequisites

Before using Modal execution, you need:

1. A Modal account.
2. Modal CLI authentication:

```bash
modal token new
```

3. Secrets for Tigris storage and W&B.

## Secrets

TMGG expects two Modal secret groups:

| Secret Group | Keys | Purpose |
|--------------|------|---------|
| `tigris-credentials` | `TMGG_TIGRIS_BUCKET`, `TMGG_TIGRIS_ACCESS_KEY`, `TMGG_TIGRIS_SECRET_KEY` | Checkpoints and metrics storage |
| `wandb-credentials` | `WANDB_API_KEY` | W&B logging |

### Manual setup

```bash
uv run modal secret create tigris-credentials --force \
    TMGG_TIGRIS_BUCKET="your-bucket" \
    TMGG_TIGRIS_ACCESS_KEY="your-key" \
    TMGG_TIGRIS_SECRET_KEY="your-secret"

uv run modal secret create wandb-credentials --force \
    WANDB_API_KEY="your-wandb-key"
```

## Deployment

Deploy the current app before launching runs:

```bash
mise run modal-deploy
```

`modal-deploy` first refreshes Modal secrets from the current environment and then runs:

```bash
uv run modal deploy -m tmgg.modal._functions
```

For development with hot reload:

```bash
uv run modal serve -m tmgg.modal._functions
```

## GPU Tiers

| Tier | Modal GPU |
|------|-----------|
| `debug` | `T4` |
| `standard` | `A10G` |
| `fast` | `A100-40GB` |
| `multi` | `A100-40GB:2` |
| `h100` | `H100` |

The deployed functions all use a 24-hour timeout.

## Running A Single Experiment

Use `tmgg-modal run` when you want one resolved config launched on Modal:

```bash
uv run tmgg-modal run tmgg-spectral-arch model.k=16 seed=1 --gpu standard
```

Detached execution:

```bash
uv run tmgg-modal run tmgg-spectral-arch model.k=16 seed=1 --gpu fast --detach
```

Dry-run config resolution:

```bash
uv run tmgg-modal run tmgg-spectral-arch model.k=16 seed=1 --dry-run
```

The command resolves Hydra locally, computes a config hash, shows the generated
`run_id`, and then dispatches the resolved YAML to Modal.

## Running Multiruns On Modal

Use Hydra multirun with the custom launcher:

```bash
uv run tmgg-experiment +stage=stage1_poc --multirun \
    hydra/launcher=tmgg_modal \
    seed=1,2,3
```

The launcher:

- resolves each Hydra job locally
- patches `paths.output_dir` / `paths.results_dir` onto the Modal volume
- dispatches the batch through `ModalRunner.run_sweep()`

For local multiruns, use Hydra's built-in launcher instead:

```bash
uv run tmgg-experiment +stage=stage1_poc --multirun hydra/launcher=basic seed=1,2,3
```

## W&B Logging

Most training configs in TMGG already default to W&B logging through `_base_infra.yaml`.
In practice, Modal runs need:

- the `wandb-credentials` Modal secret
- `allow_no_wandb=false` if you want the run to fail loudly when logging is unavailable

Example:

```bash
uv run tmgg-modal run tmgg-discrete-gen \
    models/discrete@model=discrete_sbm_official \
    allow_no_wandb=false \
    wandb_entity=your-wandb-entity \
    wandb_project=discrete-diffusion \
    --gpu standard
```

## Upstream-style DiGress SBM Run

The checked-in `discrete_sbm_official.yaml` is a close local baseline, but not a literal
copy of the upstream DiGress SBM training setup. If you want the upstream DiGress SBM
training horizon and feature settings, the currently supported Modal launch looks like:

```bash
uv run tmgg-modal run tmgg-discrete-gen \
    models/discrete@model=discrete_sbm_official \
    learning_rate=0.0002 \
    weight_decay=1e-12 \
    amsgrad=true \
    data.num_graphs=200 \
    data.train_ratio=0.64 \
    data.val_ratio=0.16 \
    trainer.max_steps=550000 \
    trainer.val_check_interval=1100 \
    model.eval_every_n_steps=1100 \
    model.noise_schedule.timesteps=1000 \
    data.batch_size=12 \
    data.graph_config.p_intra=1.0 \
    data.graph_config.p_inter=0.0 \
    +model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures \
    +model.model.extra_features.extra_features_type=all \
    +model.model.extra_features.max_n_nodes=20 \
    +model.evaluator.p_intra=1.0 \
    +model.evaluator.p_inter=0.0 \
    allow_no_wandb=false \
    --gpu standard \
    --detach
```

This reproduces the upstream optimizer, diffusion-step count, and validation cadence in
terms of steps, using a 200-graph synthetic split that approximates the upstream SBM
train/val/test counts. It still uses TMGG's current synthetic categorical SBM datamodule
rather than the original upstream SPECTRE SBM dataset.

## Evaluation and Aggregation

Run checkpoint evaluation on Modal:

```bash
uv run tmgg-modal evaluate -r some_run_id --gpu debug
```

Aggregate evaluation results from the Modal volume into a local parquet file:

```bash
uv run tmgg-modal aggregate --output results/mmd_results.parquet --cache-dir results/mmd_cache
```

## Troubleshooting

### "Modal app is not deployed"

Redeploy:

```bash
mise run modal-deploy
```

### "W&B logger configured but WANDB_API_KEY is not set"

Set the `WANDB_API_KEY` Modal secret:

```bash
uv run modal secret create wandb-credentials --force WANDB_API_KEY="$WANDB_API_KEY"
```

### Missing `sbm_accuracy` or `orbit_mmd`

Local dry-runs may warn if `graph-tool` or ORCA are unavailable on your machine. The Modal
image is built to include those dependencies, so the warning matters primarily for local
execution and local `--dry-run` checks.
