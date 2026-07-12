# Generative (Continuous Diffusion)

**Continuous diffusion for graph generation** using any denoising model from
`models/factory.py` (GNN, spectral, DiGress wrapper, etc.). The forward process
adds Gaussian noise to adjacency matrices; the reverse process iteratively
denoises, producing new graphs.

This differs from `discrete_diffusion/` in that noise operates on continuous
adjacency values rather than categorical one-hot features.

## Paradigm

**Continuous graph generation** — sample timestep, add Gaussian noise scaled by
the noise schedule, train the model to predict the clean adjacency. Generation
proceeds by iterative denoising from pure noise. Validation computes MMD metrics
(degree, clustering, spectral) against a reference set.

## Models

Any model registered in `ModelRegistry` can be used here via `models/factory.py`.

## Usage

No standalone CLI entry point. Invoked through the stage system or directly:

```bash
tmgg-experiment +stage=stage2_validation    # via stages
python -m tmgg.experiments.gaussian_diffusion_generative.runner  # direct
```
