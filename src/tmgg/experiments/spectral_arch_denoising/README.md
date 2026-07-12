# Spectral Denoising

Spectral architectures for adjacency denoising — models that operate in the
eigenspace of the graph Laplacian or use learned spectral filters. This is the
primary experiment family for the research question of whether spectral methods
offer advantages over spatial (GNN) approaches for graph denoising.

## Paradigm

**Adjacency denoising** — receives noisy adjacency, predicts clean adjacency.

## Models

- `linear_pe` — linear projection with positional (eigenvector) encoding
- `filter_bank` — learned graph filter bank operating on Laplacian eigenvalues
- `self_attention` — self-attention over spectral features
- `topk_eigen` — top-k eigenvector projection
- `shrinkage_wrapper` — wraps any spectral model with learned shrinkage thresholding

All from `models/spectral_denoisers/`, created through `models/factory.py`.

## CLI

```bash
tmgg-spectral-arch      # Hydra config: base_config_spectral_arch.yaml
```
