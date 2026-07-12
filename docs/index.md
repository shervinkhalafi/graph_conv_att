# TMGG Documentation

TMGG is a research framework for graph denoising with attention, GNN, and hybrid architectures. It uses Hydra for reproducible experiments, supports multiple noise models, and can run locally or on the cloud.

## Quick start

```bash
uv sync
uv run tmgg-spectral-arch
```

Need a different model or a custom configuration? Start with the **Get started** guide and the **Configuration** reference.

## What’s in this documentation

- [Get started](get-started.md) — install, run a first experiment, and learn the basics
- [Configuration](configuration.md) — Hydra config hierarchy and override patterns
- [Experiments](experiments.md) — stage-based workflows, outputs, and logging
- [Data](data.md) — datasets, noise types, and data modules
- [Models](models.md) — available architectures and parameters
- [Cloud execution](cloud.md) — current Modal CLI, launcher, secrets, and deployment flow
- [Architecture](architecture.md) — system design and module layout
- [Extending](extending.md) — add new models, datasets, or backends

## When to use which entry point

- `tmgg-spectral-arch` — spectral positional encoding models (main focus)
- `tmgg-digress` — DiGress transformer
- `tmgg-gnn` — GNN-based denoising
- `tmgg-gnn-transformer` — hybrid GNN + transformer model
- `tmgg-gaussian-gen` / `tmgg-discrete-gen` — generative diffusion experiments

The first four entry points run *denoising* experiments that reconstruct corrupted graphs. The generative runner trains a full diffusion model to *generate* novel graphs from noise, evaluated via MMD metrics. If you're unsure, start with `tmgg-spectral-arch` and explore overrides in the Configuration guide.
