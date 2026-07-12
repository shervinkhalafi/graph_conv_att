# How to run experiments

This guide covers the practical steps for running each type of experiment in TMGG, both locally and on Modal. For parameter reference, see [experiments.md](experiments.md); for config hierarchy details, see [configuration.md](configuration.md). For the per-run performance opt-ins (notably `torch.compile`) and what's automatic vs manual, see [performance-toggles.md](performance-toggles.md).

## Prerequisites

Install the project and its dependencies:

```bash
uv sync              # production dependencies
uv sync --all-extras # includes test + dev tools
```

All experiment CLIs use Hydra for configuration. Override any parameter on the command line with `key=value` syntax. Training is configured in **steps** (not epochs) throughout.

## Run a denoising experiment locally

Denoising experiments train a model to reconstruct a clean adjacency matrix from a noisy input. Five architectures are available, each with its own CLI:

```bash
# Spectral PE architectures (linear PE, filter bank, self-attention)
uv run tmgg-spectral-arch

# DiGress graph transformer
uv run tmgg-digress

# GNN
uv run tmgg-gnn

# Hybrid GNN + transformer
uv run tmgg-gnn-transformer

# Linear or MLP baselines
uv run tmgg-baseline
uv run tmgg-baseline model=baselines/mlp  # switch to MLP variant
```

Common overrides that apply to all denoising experiments:

```bash
uv run tmgg-spectral-arch \
    trainer.max_steps=50000 \
    learning_rate=1e-3 \
    data=sbm_n100 \
    noise_levels=[0.01,0.1,0.2] \
    seed=42
```

To verify that the setup works before a long run:

```bash
# Single-batch forward/backward pass
uv run tmgg-spectral-arch trainer.fast_dev_run=true

# Full stack trace on errors
HYDRA_FULL_ERROR=1 uv run tmgg-spectral-arch
```

**What success looks like:** `train/loss` decreasing over steps, `val/loss` logged at each validation interval. Checkpoints appear under `outputs/`.

## Run a generative experiment locally

Generative experiments train diffusion models that generate new graphs from noise, evaluated by comparing generated graph statistics against a reference distribution via MMD.

### Gaussian diffusion (continuous noise)

```bash
uv run tmgg-gaussian-gen

# Override architecture and graph size
uv run tmgg-gaussian-gen \
    model.model_type=filter_bank \
    model.num_diffusion_steps=200 \
    data.dataset_type=sbm \
    data.num_nodes=50
```

All denoising architectures (`linear_pe`, `filter_bank`, `self_attention`, `gnn`, etc.) are available as `model.model_type`.

### Discrete diffusion (categorical noise, Vignac et al. 2023)

```bash
uv run tmgg-discrete-gen

# Adjust diffusion schedule and graph properties
uv run tmgg-discrete-gen \
    model.diffusion_steps=200 \
    data.num_nodes=30 \
    data.num_graphs=5000 \
    data.dataset_type=sbm
```

The discrete diffusion model uses categorical transitions (identity to uniform interpolation) and trains a `DiscreteGraphTransformer` backbone to predict clean graph structure.

**What success looks like:** `val/epoch_NLL` decreasing over training. MMD metrics (`val/degree_mmd`, `val/clustering_mmd`, `val/spectral_mmd`) logged at each validation epoch — lower is better. The `*_mmd` keys hold squared MMD values (V-statistic, GraphRNN/GRAN convention) — see [`docs/eval/mmd-units-and-protocol.md`](eval/mmd-units-and-protocol.md) before comparing to DiGress / HiGen tables.

> **Loss comparability caveat:** `val/loss` from denoising (BCEWithLogits), Gaussian generative (MSE), and discrete diffusion (VLB/NLL) are on fundamentally different scales. Use MMD metrics to compare across experiment families.

## Evaluate a trained model

### Denoising experiments

Run the test set evaluation on a trained checkpoint:

```python
import pytorch_lightning as pl
from tmgg.experiments.spectral_arch_denoising.lightning_module import (
    SpectralDenoisingLightningModule,
)

model = SpectralDenoisingLightningModule.load_from_checkpoint(
    "outputs/.../checkpoints/last.ckpt"
)
trainer = pl.Trainer(accelerator="auto")
trainer.test(model, datamodule=dm)  # logs test/loss
```

### Discrete diffusion evaluation CLI

Evaluate a discrete diffusion checkpoint by sampling graphs and computing MMD against a reference distribution:

```bash
uv run tmgg-discrete-eval \
    --checkpoint outputs/discrete_diffusion/.../last.ckpt \
    --dataset sbm \
    --num-samples 500 \
    --num-nodes 20 \
    --output results.json
```

The output reports degree MMD, clustering MMD, and spectral MMD. Pass `--device cuda` for GPU-accelerated sampling.

## Launch experiments on Modal

Modal provides cloud GPUs for running experiments without local hardware. This requires one-time setup (see [cloud.md](cloud.md) for secrets configuration).

### Deploy the Modal app

Redeploy after any code change (including config changes):

```bash
mise run modal-deploy
```

### Run a single experiment


```bash
uv run python -m tmgg.modal.cli.spawn_single \
    --config ./configs/discrete_gen/2026-02-16/discrete_gen_discrete_default_T100_lr1e-4_s1.json \
    --gpu debug
```

GPU tiers: `debug` (T4, cheapest), `standard` (A10G), `fast` (A100-40GB), `multi` (A100x2), `h100`.

### Run a parallel sweep

```bash
uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/discrete_gen/2026-02-16/ \
    --gpu standard \
    --skip-existing \
    --wandb-entity your-wandb-entity \
    --wandb-project discrete-diffusion
```

Useful flags:

- `--filter "T500"` — only launch configs whose run_id contains "T500"
- `--skip-existing` — check W&B and skip already-completed runs
- `--limit 5` — cap at 5 experiments
- `--dry-run` — preview without launching
- `--delay 0.5` — pause between spawns (avoid rate limits)

## Create a hyperparameter sweep

### Generate configs from a stage definition

Stage definitions specify the architecture grid, hyperparameter space, and seeds. Available stages are YAML files in `src/tmgg/modal/stage_definitions/`.

```bash
# List available stages
ls src/tmgg/modal/stage_definitions/

# Generate configs for a stage
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage_discrete_gen \
    --output-dir ./configs/discrete_gen/2026-02-16/

# Preview without writing files
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage_discrete_gen \
    --output-dir /tmp/preview \
    --dry-run
```

The generator produces one JSON file per hyperparameter-seed combination, ready for `launch_sweep`.

### Available stages

| Stage | Experiment type | Grid size | Description |
|-------|----------------|-----------|-------------|
| `stage1` | Spectral denoising | 108 | POC: 3 architectures, SBM n=50 |
| `stage1b_shrinkage` | Spectral denoising | — | Shrinkage wrapper variants |
| `stage1c_digress_gnn` | DiGress denoising | 144 | GNN Q/K/V projection comparison |
| `stage1f_digress_spectral` | DiGress denoising | 144 | Spectral filter projection comparison |
| `stage2` | Spectral denoising | — | Cross-dataset validation (4 datasets) |
| `stage_discrete_gen` | Discrete diffusion | 8 | Sanity check (1 arch, 2 lr, 2 T, 2 seeds) |

### Write a custom stage definition

Create a YAML file in `src/tmgg/modal/stage_definitions/`:

```yaml
name: my_sweep
base_config: base_config_spectral_arch   # experiment type

architectures:
  - models/spectral/linear_pe
  - models/spectral/self_attention

datasets:                                 # optional; omit to use base default
  - sbm_default

hyperparameters:
  learning_rate: [1e-4, 1e-3]
  weight_decay: [1e-2, 1e-3]
  "+model.k": [8, 16]                    # + prefix for new Hydra fields

seeds: [1, 2, 3]

run_id_template: "my_sweep_{arch}_{data}_{lr}_{wd}_{k}_s{seed}"
```

Template variables `{lr}`, `{wd}`, `{k}`, `{diffusion_steps}` include a prefix (e.g., `lr1e-4`, `k16`, `T500`). `{arch}`, `{data}`, and `{seed}` are bare values.

### Local multirun (Hydra)

For quick local sweeps without Modal:

```bash
uv run tmgg-spectral-arch --multirun \
    model.k=8,16,32 \
    learning_rate=1e-4,1e-3 \
    seed=1,2,3
```

## Monitor running experiments

### Weights & Biases

Enable W&B logging and check results at [wandb.ai](https://wandb.ai):

```bash
uv run tmgg-spectral-arch logger=wandb wandb_project=my-project
```

### TensorBoard

```bash
# Start TensorBoard on local output directory
tensorboard --logdir outputs/

# Or on S3 (for Modal runs)
tensorboard --logdir s3://your-bucket/tensorboard/
```

### Modal dashboard

Track running/completed Modal jobs at [modal.com/apps](https://modal.com/apps). Each spawned experiment appears as a function call under the `graph-denoising` app.

## Workflow: discrete diffusion sanity check on Modal

This end-to-end example runs the discrete diffusion sanity check sweep:

```bash
# 1. Deploy current code to Modal
mise run modal-deploy

# 2. Generate the 8 sanity-check configs
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage_discrete_gen \
    --output-dir ./configs/discrete_gen/2026-02-16/

# 3. Launch all 8 on debug GPUs (cheapest)
uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/discrete_gen/2026-02-16/ \
    --gpu debug \
    --wandb-entity your-wandb-entity \
    --wandb-project discrete-diffusion

# 4. Monitor progress on W&B and Modal dashboard
# 5. Once complete, evaluate best checkpoint:
uv run tmgg-discrete-eval \
    --checkpoint <path-to-best-checkpoint> \
    --num-samples 500 --num-nodes 20 --output eval_results.json
```

## Workflow: re-run denoising experiments with updated configs

After fixing configs or code on the `cleanup` branch:

```bash
# 1. Redeploy to pick up fixes
mise run modal-deploy

# 2. Regenerate configs (picks up config changes)
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage1 \
    --output-dir ./configs/stage1/2026-02-16/

# 3. Launch, skipping any runs that already completed
uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/stage1/2026-02-16/ \
    --gpu standard \
    --skip-existing \
    --wandb-entity your-wandb-entity \
    --wandb-project spectral_denoising

# For DiGress-specific comparisons:
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage1c_digress_gnn \
    --output-dir ./configs/stage1c/2026-02-16/

uv run python -m tmgg.modal.cli.launch_sweep \
    --config-dir ./configs/stage1c/2026-02-16/ \
    --gpu standard \
    --wandb-entity your-wandb-entity \
    --wandb-project spectral_denoising
```

## Config generation pipeline

Config generation follows a two-phase design because Hydra's `defaults:` system composes one model config per invocation, while batch sweeps need to iterate over multiple architectures. Phase 1 uses Hydra to compose the base config and resolve all interpolations into a plain dict. Phase 2 loads each architecture YAML, strips its `${...}` interpolation placeholders, deep-merges it into the base, applies HP overrides, and syncs coupled parameters -- all as plain dict operations.

### Where each piece of config comes from

The final JSON config that reaches Modal is assembled from four sources:

1. **Base config YAML** (`exp_configs/<base_config>.yaml`) -- defines the experiment type via Hydra `defaults:` (data, model skeleton, trainer settings). Hydra resolves all interpolations during Phase 1.
2. **Architecture YAML** (`exp_configs/models/<family>/<arch>.yaml`) -- contains concrete model parameters (`_target_`, layer counts, hidden dims) plus `${learning_rate}`-style placeholders for local CLI use. `strip_interpolations()` removes those placeholders so Phase 1 values survive.
3. **Hyperparameter overrides** (from the stage definition's `hyperparameters:` grid) -- applied at dotted paths after the architecture merge. Root-level keys (e.g. `learning_rate`) go into the Hydra overrides during Phase 1; `model.`-prefixed keys are applied in Phase 2 by `apply_hyperparameters()`.
4. **Environment** -- W&B entity/project, GPU tier, and output paths are set at launch time by `launch_sweep` or `spawn_single`, not baked into the config JSON.

### Coupled parameters

When Hydra interpolations like `timesteps: ${model.diffusion_steps}` are lost during the Phase 1 to Phase 2 dict conversion, `ExperimentConfigBuilder.COUPLED_PARAMS` explicitly propagates the source to the target. If you add a new cross-section dependency, add a `(source_path, target_path)` tuple there.

### Debugging config generation

Use `--dry-run` to preview generated configs without writing files:

```bash
uv run python -m tmgg.modal.cli.generate_configs \
    --stage stage_discrete_gen \
    --output-dir /tmp/preview \
    --dry-run
```

When a generated config looks wrong, inspect the JSON output directly -- every config is a self-contained dict with no interpolation strings remaining. Common issues to check:

- **Missing model fields** -- the architecture YAML may lack a key that the base config sets via interpolation. Verify that `strip_interpolations()` preserved the expected base value rather than dropping it.
- **Stale coupled params** -- if `noise_schedule.timesteps` does not match `model.diffusion_steps` after an HP override, check that the coupling exists in `COUPLED_PARAMS`.
- **Numeric string coercion** -- YAML loads `1e-4` as a string; `_coerce_numeric()` converts it to float. If a value arrives as the wrong type, the coercion regex may need updating.
