# Get started

This guide walks you through installation and running your first experiment.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installed

## Install

From the repository root:

```bash
uv sync
```

For development and tests:

```bash
uv sync --all-extras
```

## Run your first experiment

```bash
# Spectral denoising (main experiment type)
uv run tmgg-spectral-arch
```

Other entry points:

```bash
# DiGress transformer
uv run tmgg-digress

# GNN-based denoising
uv run tmgg-gnn

# Hybrid GNN + Transformer
uv run tmgg-gnn-transformer
```

## Common overrides

TMGG uses Hydra overrides. A few examples:

```bash
# Increase training steps
uv run tmgg-spectral-arch trainer.max_steps=50000

# Change model depth
uv run tmgg-gnn model.num_layers=8

# Switch dataset configuration
uv run tmgg-spectral-arch data=sbm_default
```

Note: training is configured in **steps**, not epochs.

## Next steps

- Read [Configuration](configuration.md) to understand the Hydra hierarchy.
- See [Experiments](experiments.md) for stage-based workflows and outputs.
- Check [Data](data.md) and [Models](models.md) for dataset and architecture details.
