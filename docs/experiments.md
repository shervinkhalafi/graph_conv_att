# Experiments

This document covers running experiments, interpreting results, and using the stage-based experiment system.

## Running Experiments

### Basic Usage

```bash
# Run with default configuration
uv run tmgg-spectral-arch

# Override parameters
uv run tmgg-gnn trainer.max_steps=100000 model.num_layers=4

# Different data configuration
uv run tmgg-spectral-arch data=sbm_default
```

### Available Commands

| Command | Model Type | Base Config |
|---------|------------|-------------|
| `tmgg-spectral-arch` | Spectral PE models | `base_config_spectral_arch.yaml` |
| `tmgg-digress` | DiGress transformer | `base_config_digress.yaml` |
| `tmgg-gnn` | Standard GNN | `base_config_gnn.yaml` |
| `tmgg-gnn-transformer` | GNN + Transformer | `base_config_gnn_transformer.yaml` |
| `tmgg-baseline` | Linear / MLP baselines | `base_config_baseline.yaml` |

The `tmgg-baseline` command runs simple linear or MLP baselines for comparison. Override the model variant with `model=baselines/mlp`.

### Discrete Diffusion

The framework also supports discrete diffusion for generative graph modeling, where a model learns to reverse a categorical noise process. Training corrupts adjacency matrices according to a noise schedule (categorical transition matrices per Vignac et al. 2023) and trains the model to predict clean structure. See the [Generative Graph Modeling](#generative-graph-modeling) section for details and configuration.

### Sanity Checks

Validate setup without full training:

```bash
# Run sanity check (tests data loading, forward pass, loss computation)
uv run tmgg-spectral-arch sanity_check=true

# Fast dev run (one batch only)
uv run tmgg-gnn trainer.fast_dev_run=true
```

## Output Structure

Each run creates a timestamped directory:

```
outputs/YYYY-MM-DD/HH-MM-SS/
├── config.yaml              # Resolved configuration
├── checkpoints/
│   ├── model-step=005000-val_loss=0.1234.ckpt
│   ├── model-step=010000-val_loss=0.0987.ckpt
│   └── last.ckpt
├── tensorboard/             # TensorBoard logs (if enabled)
│   └── events.out.tfevents.*
└── .hydra/
    ├── config.yaml          # Original config
    ├── hydra.yaml           # Hydra settings
    └── overrides.yaml       # Command-line overrides
```

### Checkpoints

The framework saves:
- Top 3 models by validation loss
- Last checkpoint

Checkpoint naming: `model-step={N}-val_loss={X}.ckpt`

### Loading Checkpoints

```python
from tmgg.experiments.spectral_arch_denoising.lightning_module import SpectralDenoisingLightningModule

model = SpectralDenoisingLightningModule.load_from_checkpoint(
    "outputs/.../checkpoints/model-step=010000-val_loss=0.0987.ckpt"
)
```

## Logging Options

### TensorBoard (Default)

```bash
uv run tmgg-spectral-arch logger=tensorboard

# View logs
tensorboard --logdir outputs/
```

### Weights & Biases

```bash
WANDB_API_KEY="your-api-key" uv run tmgg-spectral-arch

# Or with project name
WANDB_API_KEY="your-api-key" uv run tmgg-spectral-arch wandb_project="my-project"
```

For analysis tools and exported data, see internal export tooling (maintained separately).

### CSV Logger

```bash
uv run tmgg-spectral-arch logger=csv
```

### Multiple Loggers

```bash
uv run tmgg-spectral-arch logger=multi  # TensorBoard + CSV
```

## Metrics

The framework tracks these metrics:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss per step |
| `val/loss` | Validation loss per validation check |
| `test/loss` | Test loss (final evaluation) |
| `train/loss_epoch` | Average training loss per epoch |

Additional metrics computed in final evaluation:
- Eigenvalue error
- Subspace distance
- Reconstruction MAE/MSE

> **Caveat:** Loss values are not comparable across experiment types.
> Denoising uses BCEWithLogits, Gaussian generative uses MSE, and
> discrete diffusion uses VLB/cross-entropy. Comparing `val/loss`
> across these experiment families is meaningless — use MMD metrics
> or graph-theoretic statistics for cross-experiment comparison.

### Metric Regimes

Denoising and generative experiments track fundamentally different metric sets, since they answer different questions (reconstruction quality vs. distributional fidelity).

**Denoising** logs reconstruction quality against a known clean target: MSE, Frobenius error, eigenvalue error, subspace distance, and edge accuracy.

**Generative** logs distributional similarity between generated and held-out reference graphs: MMD on degree distribution, clustering coefficient, and spectral statistics.

These metric sets are not directly comparable. To bridge them, one can generate graphs from the generative model and compute reconstruction metrics against a held-out test set, or compute MMD statistics on the output of denoised graphs.

## Hyperparameter Sweeps

### Multirun

Run multiple configurations sequentially:

```bash
# Single parameter sweep
uv run tmgg-spectral-arch --multirun model.num_layers=4,8,16

# Multiple parameters (grid search)
uv run tmgg-spectral-arch --multirun \
  model.num_layers=4,8 \
  model.learning_rate=0.001,0.01 \
  seed=1,2,3
```

### Grid Search Command

```bash
uv run tmgg-grid-search  # Uses grid search configuration
```

## Stage-Based Experiments

The stage system runs systematic experiments across architectures and datasets with a fixed compute budget. This allows methodical validation of spectral PE approaches before scaling.

### Stage 1: Proof of Concept

**Budget**: 4.4 GPU-hours

**Purpose**: Validate that spectral PE architectures can denoise graphs on a single-graph SBM protocol (n=50).

**Architectures**:
- Linear PE
- Filter Bank
- Self-Attention
- DiGress (official LR=0.0002)
- DiGress (high LR=1e-2, matching spectral)

**Dataset**: SBM n=50, single-graph protocol (same graph for train/val/test, only noise varies)

**Optimizer**: Adam, no weight decay, no LR scheduling

**Hyperparameter sweep**:
- learning_rate: [1e-3, 1e-2]
- noise_levels: [0.01, 0.05, 0.1, 0.2, 0.3]
- model.k: 50 (full spectrum)
- 4 trials per configuration, 3 seeds

**Success criteria**: ≥15% improvement over Linear PE baseline on val_loss.

```bash
uv run tmgg-experiment +stage=stage1_poc              # Single run
uv run tmgg-experiment +stage=stage1_poc sweep=true   # Full sweep
```

### Stage 1 Sanity: Constant Noise Memorization

**Budget**: <20 GPU-minutes

**Purpose**: Validate model functionality by testing memorization of a fixed noisy→clean mapping. This debugging stage verifies gradient flow, model capacity, and optimizer behavior.

**Architectures**: Linear PE, Filter Bank, Self-Attention

**Key setting**: `fixed_noise_seed: 42` produces identical noise at every training step.

**Hyperparameters** (fixed, no sweep):
- learning_rate: 1e-2
- weight_decay: 0.0
- model.k: 50

**Success criteria**: `val_accuracy >= 0.99`

If this stage fails, something fundamental is broken in the architecture or optimizer.

```bash
uv run tmgg-experiment +stage=stage1_sanity
```

### Stage 2: Cross-Dataset Validation

**Budget**: ~19 GPU-hours

**Purpose**: Validate denoising across diverse graph families, with two protocols controlled by `cross_graph`:

| Protocol | `cross_graph` | Description |
|----------|---------------|-------------|
| Single-graph | `false` (default) | Train/val/test use the same graph, only noise varies |
| Cross-graph | `true` | Train/val/test use different graphs (generalization test) |

**Architectures**:
- Linear PE, Filter Bank, Self-Attention (high LR=1e-2, AMSGrad, weight_decay=1e-12)
- DiGress variants (official and high LR)

**Datasets** (9 graph types):
- Synthetic: Erdős-Rényi, d-regular, tree, ring of cliques, LFR, SBM
- PyG benchmarks: QM9, ENZYMES, PROTEINS

**Hyperparameter sweep**:
- noise_levels: [0.01, 0.1, 0.2]
- 4 trials per configuration, 3 seeds

**Success criteria**:
- reconstruction_error < 0.05
- generalization_gap < 0.15

```bash
uv run tmgg-experiment +stage=stage2_validation                      # Single-graph (default)
uv run tmgg-experiment +stage=stage2_validation cross_graph=true     # Cross-graph
uv run tmgg-experiment +stage=stage2_validation sweep=true           # Full sweep
```

### Stage 3: Dataset Diversity

**Budget**: 400 GPU-hours (future work)

**Purpose**: Validate across all graph families.

**Trigger condition**: Run only if Stage 2 achieves <3% error on multi-graph and matches DiGress.

**Architectures**: Filter Bank, Self-Attention, DiGress variants

**Datasets**:
- SBM: n=50, n=100, n=200
- Erdős-Rényi, d-regular, trees
- LFR benchmark (planted community)
- Ring of cliques

**Hyperparameter sweep**:
- learning_rate: [1e-3, 1e-2]
- model.k: [50, 100, 200]
- noise_levels: [0.01, 0.05, 0.1, 0.2, 0.3]
- 8 trials, 5 seeds

```bash
uv run tmgg-experiment +stage=stage3_diversity sweep=true
```

### Stage 4: Real-World Benchmarks

**Budget**: 300 GPU-hours (future work)

**Purpose**: Validate on PyTorch Geometric benchmark datasets.

**Trigger condition**: Run if Stage 3 validates cross-family generalization.

**Datasets**: PyG QM9, ENZYMES, PROTEINS

**Hyperparameter sweep**:
- model.k: [16, 32, 64] (variable graph sizes)
- 6 trials, 5 seeds
- 60-minute timeout, fast GPU tier

```bash
uv run tmgg-experiment +stage=stage4_benchmarks sweep=true
```

### Stage 5: Full Validation

**Budget**: 1500 GPU-hours (future work, for publication)

**Purpose**: Comprehensive ablations and robustness analysis.

**Architectures**: All (Linear PE, Filter Bank, Self-Attention, DiGress variants)

**Datasets**: All 11 datasets from previous stages

**Ablation studies**:
- Spectral polynomial depth: [3, 5, 8]
- Eigenvector count: [2, 4, 8, 16, 32, 64]
- Attention key dimension: [32, 64, 128]

**Statistical analysis**:
- Wilcoxon signed-rank test
- Holm-Bonferroni correction
- Significance level: 0.05

```bash
uv run tmgg-experiment +stage=stage5_full sweep=true
```

### Stage Configuration

Stage configs are defined in `exp_configs/stage/`. Example structure:

```yaml
# stage1_poc.yaml
# @package _global_
defaults:
  - override /data: sbm_single_graph

stage: stage1_poc

# Optimizer settings (harmonized across stages)
learning_rate: 1e-2
weight_decay: 1e-12
optimizer_type: adamw
amsgrad: true
scheduler_config:
  type: none

model:
  k: 50

noise_levels: [0.1]
eval_noise_levels: [0.1]

# Sweep metadata (used by coordinator when sweep=true)
_sweep_config:
  architectures:
    - models/spectral/linear_pe
    - models/spectral/filter_bank
    - models/spectral/self_attention
    - models/digress/digress_sbm_small
    - models/digress/digress_sbm_small_highlr
  hyperparameter_space:
    learning_rate: [1e-3, 1e-2]
    noise_levels:
      - [0.01]
      - [0.1]
  num_trials: 4
  seeds: [1, 2, 3]
  timeout_seconds: 600
  success_criteria:
    metric: val_loss
    improvement_threshold: 0.15
    baseline: linear_pe
```

## Generative Graph Modeling

TMGG currently has two generative training paths:

- `tmgg-gaussian-gen` for continuous diffusion on dense adjacency matrices
- `tmgg-discrete-gen` for DiGress-style categorical diffusion on node and edge states

These are separate from the denoising experiments such as `tmgg-spectral-arch`
and `tmgg-digress`, which reconstruct a corrupted input graph.

### How It Works

Both generative paths sample a timestep, apply a forward corruption process, and
train a model to predict the clean graph state. Validation periodically samples
new graphs and compares them against held-out references with `GraphEvaluator`.

The discrete path additionally logs diffusion likelihood terms such as
`val/epoch_NLL`, `val/kl_prior`, `val/kl_diffusion`, and `val/gen/full_chain_vlb`.

### Running Generative Experiments

The continuous and categorical runners have separate base configs and entry points:

```bash
# Continuous diffusion
uv run tmgg-gaussian-gen

# Discrete DiGress-style diffusion
uv run tmgg-discrete-gen
```

Common overrides:

```bash
# Continuous: change diffusion horizon
uv run tmgg-gaussian-gen model.noise_schedule.timesteps=200

# Discrete: use the local upstream-style SBM baseline
uv run tmgg-discrete-gen models/discrete@model=discrete_sbm_official

# Generative data overrides use graph_type / graph_config, not dataset_type / dataset_config
uv run tmgg-discrete-gen data.graph_type=erdos_renyi data.num_nodes=100
```

`tmgg-gaussian-gen` writes under `outputs/generative/...`. `tmgg-discrete-gen`
uses the standard run-based output path inherited from `_base_infra.yaml`.

### Continuous Diffusion Defaults

The continuous base config (`base_config_gaussian_diffusion.yaml`) defaults to:

```yaml
model:
  noise_schedule:
    schedule_type: cosine_iddpm
    timesteps: 100
  loss_type: mse
  evaluator:
    kernel: gaussian
    sigma: 1.0

data:
  _target_: tmgg.data.data_modules.multigraph_data_module.MultiGraphDataModule
  graph_type: sbm
  num_nodes: 50
  num_graphs: 1000
```

### Discrete Diffusion Defaults

The discrete base config (`base_config_discrete_diffusion_generative.yaml`) defaults to:

```yaml
model:
  noise_schedule:
    schedule_type: cosine_iddpm
    timesteps: 500
  evaluator:
    eval_num_samples: 128
  loss_type: cross_entropy
  lambda_E: 5.0

data:
  _target_: tmgg.data.data_modules.synthetic_categorical.SyntheticCategoricalDataModule
  graph_type: sbm
  num_nodes: 20
  num_graphs: 2000
  graph_config:
    num_blocks: 2
    p_intra: 0.7
    p_inter: 0.1
```

For the local DiGress-style SBM baseline, the checked-in model preset is:

```bash
uv run tmgg-discrete-gen models/discrete@model=discrete_sbm_official
```

That preset gives you the local "official" architecture and optimizer values, but
it is not a literal copy of the upstream DiGress training setup on its own.

### Evaluation Metrics

Both generative paths evaluate sampled graphs with `GraphEvaluator`. The discrete
training loop logs every non-`None` evaluator metric under `val/gen/*`.

| Metric | Logged as | Description |
|--------|-----------|-------------|
| Degree MMD | `val/gen/degree_mmd` | Divergence between degree distributions |
| Clustering MMD | `val/gen/clustering_mmd` | Divergence between clustering coefficient distributions |
| Spectral MMD | `val/gen/spectral_mmd` | Divergence between normalized Laplacian spectra |
| Orbit MMD | `val/gen/orbit_mmd` | Graphlet-orbit divergence when ORCA is available |
| SBM Accuracy | `val/gen/sbm_accuracy` | SBM consistency when `graph-tool` is available |
| Planarity | `val/gen/planarity_accuracy` | Fraction of connected generated graphs that are planar |
| Uniqueness | `val/gen/uniqueness` | Fraction of non-isomorphic generated graphs |
| Novelty | `val/gen/novelty` | Only available when the evaluator is constructed with training graphs |

Lower MMD values are better. Higher accuracy, uniqueness, and novelty values are better.

### Example: SBM Generation with Multirun

```bash
# Sweep over discrete diffusion horizons and seeds
uv run tmgg-discrete-gen --multirun \
  model.noise_schedule.timesteps=500,1000 \
  seed=1,2,3
```

## Debugging

### Verbose Logging

```bash
# Show full stack traces
HYDRA_FULL_ERROR=1 uv run tmgg-spectral-arch

# Debug mode
uv run tmgg-spectral-arch trainer.fast_dev_run=true
```

### Common Issues

**CUDA out of memory**: Reduce batch size
```bash
uv run tmgg-spectral-arch data.batch_size=32
```

**NaN gradients**: Enable eigenvalue regularization
```bash
uv run tmgg-gnn model.eigenvalue_reg=0.001
```

**Slow training**: Check GPU availability
```bash
uv run tmgg-spectral-arch trainer.accelerator=gpu
```

## Reproducibility

For reproducible experiments:

```bash
# Set seed
uv run tmgg-spectral-arch seed=42

# Deterministic mode (slower but reproducible)
uv run tmgg-spectral-arch trainer.deterministic=true
```

The full configuration is saved in `outputs/.../config.yaml` for reproduction.

## Analysis Tools

Beyond training experiments, the framework provides CLI tools for spectral analysis.

### Eigenstructure Study

Analyze graph eigenstructure and how it changes under noise. CLI: `tmgg-eigenstructure`

**Commands:**

| Command | Description |
|---------|-------------|
| `collect` | Compute eigendecompositions for a dataset |
| `analyze` | Run spectral analysis on collected data |
| `noised` | Collect decompositions for noised graphs |
| `compare` | Compare original vs noised eigenstructure |

**Default output:** `results/eigenstructure_study/`

```bash
# Collect eigendecompositions for SBM graphs
uv run tmgg-eigenstructure collect -d sbm \
    -c '{"num_nodes": 50, "p_intra": 0.8, "p_inter": 0.1, "num_partitions": 100}'

# Analyze collected data
uv run tmgg-eigenstructure analyze -i results/eigenstructure_study

# Add noise and compare
uv run tmgg-eigenstructure noised -i results/eigenstructure_study \
    -t gaussian -n 0.01,0.05,0.1

uv run tmgg-eigenstructure compare \
    -i results/eigenstructure_study \
    -n results/eigenstructure_study/noised
```

### Embedding Dimension Study

Find minimal embedding dimensions for exact graph reconstruction. CLI: `tmgg-embedding-study`

**Commands:**

| Command | Description |
|---------|-------------|
| `run` | Run dimension search on datasets |
| `analyze` | Analyze existing study results |

**Default output:** `results/embedding_study/`

```bash
# Run study on SBM graphs
uv run tmgg-embedding-study run --datasets sbm --output results/embedding_study

# Analyze results
uv run tmgg-embedding-study analyze --input results/embedding_study/embedding_study.json
```

Results are stored as JSON (statistics) and safetensors (embeddings).
