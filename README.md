# Graph Convolutional Attention: A Spectral Perspective on Graph Denoising and Diffusion

Code accompanying the paper. We analyze attention-based graph denoising from a
spectral perspective, show that linear attention learns an average spectral
filter, and introduce **Graph Convolutional Attention (GCA)**, a
permutation-equivariant realization that provably improves on linear attention
and transfers to DiGress-style discrete diffusion. The Python package is `tmgg`.
Experiments are driven by Hydra configs and run locally or on Modal, with
optional Weights & Biases logging.

## Setup

Three steps get you to a runnable checkout:

```bash
# 1. Clone
git clone <repository-url>
cd graph-denoising

# 2. Configure environment (defaults to WANDB_MODE=offline — no account needed)
cp .env.example .env
#    Optional: fill WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT to enable
#    logging. Get a key at https://wandb.ai/authorize

# 3. Install and prepare data
bash setup.sh
```

`setup.sh` checks prerequisites, runs `uv sync`, pre-compiles the ORCA
orbit-counter (which needs a C++ compiler, `g++`), and restores the bundled
dataset caches via `setup_data.sh`. Prerequisites: [uv](https://docs.astral.sh/uv/),
`g++`, and `zstd`.

The `.env` file loads automatically if you use either
[direnv](https://direnv.net) (run `direnv allow` once — the shipped `.envrc`
sources `.env`) or `mise`, which loads it via its env-file setting. Without
either, run `set -a; source .env; set +a` before launching an experiment.

> **Datasets.** IMDB-BINARY and DEEZER-EGO-NETS auto-download via PyG on first
> use. SPECTRE-SBM ships in the data bundle (restored by `setup_data.sh`) and
> also auto-downloads from its source URL if the bundle is absent. See
> [docs/data.md](docs/data.md) for the full dataset catalog.

## Quick Start

Run your first experiment:

```bash
# Spectral / GCA denoising (main experiment type)
uv run tmgg-spectral-arch

# GNN-based denoising with custom training steps
uv run tmgg-gnn trainer.max_steps=50000

# Spectral denoising with a specific eigenvector count
uv run tmgg-spectral-arch model.k=50
```

Logging is off by default (`WANDB_MODE=offline`). To stream metrics to W&B, set
`WANDB_MODE=online` and fill `WANDB_API_KEY` in `.env`.

Note: Training is configured in **steps**, not epochs (see
[Configuration](docs/configuration.md)).

## Environment Variables

All environment variables are **optional for local runs**. They configure cloud
execution, storage backends, and logging. The canonical, annotated list lives in
[`.env.example`](.env.example).

### Path Discovery (Modal)

| Variable | Required | Description |
|----------|----------|-------------|
| `TMGG_PATH` | No | Path to the `tmgg` package root (directory containing `src/tmgg/`). Auto-discovered when `modal/` and `tmgg/` are siblings; set only for non-standard layouts. |

### Storage (Tigris / S3-compatible, Modal)

Used for checkpoint and metrics persistence when running on Modal. Configure
these as Modal secrets; they are read by the `tmgg.modal` package.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TMGG_TIGRIS_BUCKET` | Yes* | — | Storage bucket name |
| `TMGG_TIGRIS_ENDPOINT` | No | `https://fly.storage.tigris.dev` | S3-compatible endpoint URL |
| `TMGG_TIGRIS_ACCESS_KEY` | Yes* | — | Access key |
| `TMGG_TIGRIS_SECRET_KEY` | Yes* | — | Secret key |

*Required only for the Modal storage backend; unused for local runs.

```bash
modal secret create tigris-credentials \
  TMGG_TIGRIS_BUCKET=my-bucket \
  TMGG_TIGRIS_ACCESS_KEY=... \
  TMGG_TIGRIS_SECRET_KEY=...
```

### Logging

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `WANDB_MODE` | No | `offline` | `offline` disables logging; set `online` to stream to W&B. |
| `WANDB_API_KEY` | No | — | W&B API key ([wandb.ai/authorize](https://wandb.ai/authorize)). Needed when `WANDB_MODE=online`. |
| `WANDB_ENTITY` | No | — | W&B entity (your username or team). |
| `WANDB_PROJECT` | No | — | W&B project name. |

## CLI Commands

| Command | Description |
|---------|-------------|
| `tmgg-gnn` | GNN-based denoising |
| `tmgg-gnn-transformer` | GNN + Transformer hybrid denoising |
| `tmgg-digress` | DiGress transformer model |
| `tmgg-mod-attention` | Graph Convolutional Attention (GCA / GCAT) — the paper's proposed denoiser; `model.filter_qk` toggles GT (off) ↔ GCAT (on), `model.filter_num_terms` sets the polynomial order |
| `tmgg-spectral-arch` | Spectral-attention analysis denoisers (linear-PE / filter-bank / self-attention; §2–3 theory family, not GCA) |
| `tmgg-gaussian-gen` | Gaussian diffusion generative |
| `tmgg-discrete-gen` | Discrete diffusion generative (DiGress) |
| `tmgg-discrete-eval` | Discrete diffusion evaluation |
| `tmgg-baseline` | Linear/MLP baseline denoising |
| `tmgg-experiment` | Unified stage runner (e.g., `+stage=stage1_poc`) |
| `tmgg-grid-search` | Hyperparameter grid search |
| `tmgg-eigenstructure` | Eigenstructure study (collect, analyze, noised, compare) |
| `tmgg-embedding-study` | Embedding dimension study (run, analyze) |

All commands support Hydra overrides:

```bash
# Override model parameters
uv run tmgg-spectral-arch model.k=50 model.d_k=128

# Override training steps and learning rate
uv run tmgg-gnn trainer.max_steps=50000 model.learning_rate=0.001

# Hyperparameter sweep
uv run tmgg-spectral-arch --multirun model.k=8,16,32
```

### W&B Project Naming

Each CLI command logs to a specific W&B project when logging is enabled. Most
training configs inherit the default W&B logger from `_base_infra.yaml`; the
project name is set in the corresponding base config and can be overridden with
`wandb_project=...`.

| CLI Commands | W&B Project | Base Config |
|-------------|-------------|-------------|
| `tmgg-gnn`, `tmgg-gnn-transformer`, `tmgg-spectral-arch`, `tmgg-digress`, `tmgg-baseline` | `architecture-study` | `base_config_{gnn,gnn_transformer,spectral_arch,digress,baseline}.yaml` |
| `tmgg-gaussian-gen` | `gaussian-diffusion` | `base_config_gaussian_diffusion.yaml` |
| `tmgg-discrete-gen`, `tmgg-discrete-eval` | `discrete-diffusion` | `base_config_discrete_diffusion_generative.yaml` |
| `tmgg-grid-search` | `graph-denoising` | `grid_search_base.yaml` |

The shared `_base_infra.yaml` composes the common trainer, logger, callbacks,
and path settings that experiment-specific base configs build on. The config
tree lives under `src/tmgg/experiments/exp_configs/`.

## Project Structure

```
graph-denoising/
├── src/tmgg/
│   ├── models/              # Neural network architectures
│   │   ├── gnn/             # Graph neural networks
│   │   ├── layers/          # Shared layers (GCN, MHA, Eigen)
│   │   ├── attention/       # Attention denoisers
│   │   ├── modified_attention/  # Graph Convolutional Attention (GCA)
│   │   ├── spectral_denoisers/
│   │   ├── digress/         # DiGress transformer baseline
│   │   ├── baselines/       # Linear/MLP baselines
│   │   └── hybrid/          # GNN + transformer hybrids
│   ├── experiments/         # Experiment runners + Hydra configs
│   │   ├── spectral_arch_denoising/
│   │   ├── digress_denoising/
│   │   ├── discrete_diffusion_generative/
│   │   ├── gnn_denoising/
│   │   ├── gnn_transformer_denoising/
│   │   ├── gaussian_diffusion_generative/
│   │   ├── lin_mlp_baseline_denoising/
│   │   ├── eigenstructure_study/
│   │   ├── embedding_study/
│   │   ├── stages/          # Multi-stage experiments
│   │   └── exp_configs/     # Hydra base/model/data/stage configs
│   ├── data/                # Data loading, datasets, noise processes
│   ├── diffusion/           # Diffusion processes
│   ├── evaluation/          # Metrics (incl. ORCA orbit counts)
│   ├── training/            # Training loop and callbacks
│   ├── modal/               # Cloud execution (Modal)
│   └── utils/               # Shared utilities
├── scripts/                 # Standalone analysis/plotting scripts
├── tests/                   # Test suite
└── docs/                    # Detailed documentation
```

## Documentation

For detailed documentation, see the [docs/](docs/) folder:

- [Architecture](docs/architecture.md) — System design and module organization
- [Configuration](docs/configuration.md) — Hydra config system and common overrides
- [Models](docs/models.md) — Model architectures and parameters
- [Data](docs/data.md) — Data pipeline, datasets, and noise types
- [Experiments](docs/experiments.md) — Running experiments and interpreting results
- [Cloud](docs/cloud.md) — Cloud execution with Modal
- [Extending](docs/extending.md) — Adding new models, datasets, and backends

## Model Architectures

**Spectral Denoisers**: The main focus of the paper. Three architectures
operating in the spectral domain:
- Linear PE: Â = V W V^T + bias
- Filter Bank: Polynomial spectral filters
- Self-Attention: Query-key attention on eigenvectors

**Graph Convolutional Attention (GCA)**: The paper's proposed
permutation-equivariant attention denoiser.

**DiGress**: Diffusion-based transformer baseline for comparison.

**Attention Models**: Multi-layer transformer attention processing adjacency
matrices directly.

**GNN Models**: Spectral graph neural networks using eigendecomposition
embeddings. Variants include standard GNN, symmetric GNN (shared embeddings),
and node-variant GNN.

**Hybrid Models**: Combine GNN embeddings with transformer-based denoising.

## Noise Types

The framework supports multiple noise models for training and evaluation:

- **Gaussian**: Additive Gaussian noise on adjacency matrices
- **Rotation**: Eigenspace rotation via skew-symmetric matrices
- **Digress**: Categorical transition matrices (Vignac et al. 2023),
  interpolating between identity and the uniform distribution
- **Edge Flip**: Bernoulli edge flipping
- **Logit**: Gaussian noise in logit space, producing soft adjacency values

## Testing

```bash
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow" -v
```

## Code Quality

Pre-commit hooks enforce code quality; `.pre-commit-config.yaml` sits at the
repository root. Install and run the checks with:

```bash
uv run pre-commit install         # one-time hook setup
uv run pre-commit run --all-files # run every hook manually

# Or run individual tools:
uv run ruff check --fix src/      # linting
uv run ruff format src/           # formatting
uv run basedpyright               # type checking
uv run tach check                 # module boundary enforcement
```

## Reproducing paper results

See [REPRODUCE.md](REPRODUCE.md) for the exact commands behind each paper table
and figure.

## Citation

If you use this software or build on this work, please cite the paper; machine-
readable metadata is in [CITATION.cff](CITATION.cff).

## License

See the [LICENSE](LICENSE) file for details.
