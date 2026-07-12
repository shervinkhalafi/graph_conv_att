# Discrete Diffusion

DiGress-style **categorical diffusion for graph generation**. Unlike the
`*_denoising` experiments (which corrupt then recover adjacency matrices), this
pipeline operates on one-hot node and edge features through a forward corruption
process and a learned reverse denoising process, generating entirely new graphs.

This is the generative counterpart to `digress_denoising/`, which uses the same
transformer architecture but for adjacency denoising.

## Paradigm

**Categorical graph generation** — forward process: apply categorical noise to
one-hot features via transition matrices; reverse process: iteratively predict
clean features from noisy input using cross-entropy loss (training) or ancestral
sampling (generation). Validation uses the variational lower bound (VLB).

## Components

- `GraphTransformer` — denoising backbone (`models/digress/`)
- `NoiseSchedule` — cosine/custom/linear noise schedules (`tmgg.diffusion.schedule`)
- `ExtraFeatures` — optional structural augmentation (cycles, eigenvalues)
- `SyntheticCategoricalDataModule` — SBM graph generation in one-hot format
- `evaluate_discrete_samples` — MMD evaluation against reference graphs

## CLI

```bash
tmgg-discrete-gen       # Train (Hydra config: base_config_discrete_diffusion_generative.yaml)
tmgg-discrete-eval      # Evaluate a saved checkpoint
```
