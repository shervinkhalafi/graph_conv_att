# DiGress Denoising

Applies the DiGress GraphTransformer architecture (Vignac et al., 2023) to
**adjacency denoising**: the model receives a noisy adjacency matrix and
predicts the clean one, using the same loss and training loop as all other
`*_denoising` experiments.

This is distinct from `discrete_diffusion/`, which uses the same transformer
architecture for **categorical graph generation** via a full diffusion pipeline.

## Paradigm

**Adjacency denoising** — same as spectral, GNN, and hybrid experiments.
Optionally extracts eigenvectors internally when `use_eigenvectors=True`,
matching the spectral denoisers' conditioning on graph structure.

## Models

`GraphTransformer` from `models/digress/transformer_model.py`, wrapped with
`assume_adjacency_input=True` so it accepts raw adjacency matrices rather than
one-hot categorical features.

## CLI

```bash
tmgg-digress            # Hydra config: base_config_digress.yaml
```
