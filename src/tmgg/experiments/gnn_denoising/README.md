# GNN Denoising

Graph neural network based adjacency denoising. Uses message-passing GNN
architectures that operate directly on graph structure without requiring
eigendecomposition.

## Paradigm

**Adjacency denoising** — receives noisy adjacency, predicts clean adjacency.

## Models

- `GNN` — standard GNN with configurable layers and polynomial terms
- `GNN_sym` — symmetry-preserving variant
- `NVGNN` — node-varying GNN

All from `models/gnn/`, created through `models/factory.py`.

## CLI

```bash
tmgg-gnn                # Hydra config: base_config_gnn.yaml
```
