# Stages (Orchestration)

Unified CLI entry point that coordinates multi-stage experiment sweeps across
model families. Rather than running individual experiment CLIs, researchers
define stage configurations (YAML groups) that specify which models, datasets,
and hyperparameter ranges to sweep over.

## Paradigm

**Orchestration** — not a model or training paradigm itself, but the layer that
dispatches work to the other experiment modules.

## Features

- Hydra config group overrides (`+stage=stage1_poc`, `+stage=stage2_validation`)
- `--multirun` support for hyperparameter sweeps via `TmggLauncher` Hydra plugin

## CLI

```bash
tmgg-experiment +stage=stage1_poc
tmgg-experiment +stage=stage2_validation --multirun model=models/spectral/linear_pe,models/gnn/gnn
```
