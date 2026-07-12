# Hybrid Denoising

Combines GNN message-passing layers with Transformer self-attention layers for
adjacency denoising. The GNN layers extract local structural features which the
Transformer layers then refine with global attention.

## Paradigm

**Adjacency denoising** — receives noisy adjacency, predicts clean adjacency.

## Models

`Hybrid` model from `models/hybrid/hybrid.py` — configurable GNN stage
(num_layers, polynomial terms, feature dimensions) followed by a Transformer
stage (num_layers, num_heads, optional spectral attention).

## CLI

```bash
tmgg-gnn-transformer    # Hydra config: base_config_gnn_transformer.yaml
```
