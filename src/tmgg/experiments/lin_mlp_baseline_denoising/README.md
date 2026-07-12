# Baselines

Simple denoising models (linear projection, MLP) that serve as sanity checks
for the training pipeline. If baselines cannot learn to denoise, the issue is
in the pipeline infrastructure rather than the model architecture; if they learn
but spectral or GNN models do not, the issue is architectural.

## Paradigm

**Adjacency denoising** — receives a noisy adjacency matrix, predicts the clean one.

## Models

- `linear`: learned `W @ A @ W^T + b` projection
- `mlp`: flatten the adjacency, pass through an MLP, reshape back

Both inherit from `GraphModel` and are created through `models/factory.py`.

## Usage

No standalone CLI entry point. Baselines run through the unified stage system:

```bash
tmgg-experiment +stage=stage1_poc model=models/baselines/linear
```
