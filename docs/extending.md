# Extending TMGG

How-to guides for adding models, experiments, datasets, and noise types. Each section solves one task; they can be read independently.

## Adding a new model to an existing experiment

Use this when you have a new graph neural network architecture and want to evaluate it within the existing denoising or diffusion framework. No new Lightning module or runner needed.

**Prerequisites:** A working TMGG installation (`uv pip install -e .`). Familiarity with PyTorch `nn.Module` and YAML syntax.

### Create the model class

Add a file under `src/tmgg/models/` that subclasses `GraphModel`. The constructor receives architecture hyperparameters; `forward` takes a `GraphData` batch and an optional normalised timestep.

```python
# src/tmgg/models/mymodels/my_model.py
from typing import Any

import torch
import torch.nn as nn
from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.base import GraphModel


class MyModel(GraphModel):
    """Short description of what this architecture does."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Define layers...

    def forward(
        self, data: GraphData, t: torch.Tensor | None = None
    ) -> GraphData:
        # Return predicted clean graph as logits (not probabilities).
        ...

    def get_config(self) -> dict[str, Any]:
        return {"hidden_dim": self.hidden_dim, "num_layers": self.num_layers}
```

`forward` must return a `GraphData` with the same shape as the input. For denoising models, the output represents logits of the clean adjacency; for categorical diffusion models, it represents class probabilities after softmax is applied externally.

### Create the model config YAML

The YAML file tells Hydra how to instantiate the Lightning module and your model. The outer `_target_` points to the Lightning module class; the nested `model._target_` points to your model class.

```yaml
# src/tmgg/experiments/exp_configs/models/mymodels/my_model.yaml
_target_: tmgg.experiments._shared_utils.lightning_modules.denoising_module.SingleStepDenoisingModule

model_name: my_model
model:
  _target_: tmgg.models.mymodels.my_model.MyModel
  hidden_dim: 64
  num_layers: 4

# These interpolate from the base config and task config — don't hardcode them.
learning_rate: ${learning_rate}
weight_decay: ${weight_decay}
optimizer_type: ${optimizer_type}
amsgrad: ${amsgrad}
scheduler_config: ${scheduler_config}
noise_type: ${noise_type}
noise_levels: ${noise_levels}
eval_noise_levels: ${eval_noise_levels}
loss_type: ${loss_type}
seed: ${seed}
spectral_k: ${spectral_k}
```

For diffusion (generative) models, change the outer `_target_` to `DiffusionModule` and add diffusion-specific parameters (`noise_process`, `sampler`, `noise_schedule`). See `models/discrete/discrete_default.yaml` for a working example.

### Create a base config that selects your model

```yaml
# src/tmgg/experiments/exp_configs/base_config_my_model.yaml
defaults:
  - base_config_denoising
  - models/mymodels/my_model@model
  - _self_

experiment_name: my_model_denoising
wandb_project: tmgg-my-model
```

`base_config_denoising` provides training infrastructure (`_base_infra`), the denoising task config (loss type, noise levels, data source), and trainer/logger/callback defaults. Your config only needs to select the model and set the experiment name.

`_self_` must appear last in `defaults:` — this ensures keys you set explicitly (like `experiment_name`) override anything from the composed defaults.

### Create the runner and CLI entry point

```python
# src/tmgg/experiments/my_experiment/runner.py
from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.experiments._shared_utils.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_my_model",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Run my model denoising experiment."""
    return run_experiment(cfg)


if __name__ == "__main__":
    main()
```

Register the entry point in `pyproject.toml`:

```toml
[project.scripts]
tmgg-my-model = "tmgg.experiments.my_experiment.runner:main"
```

Reinstall with `uv pip install -e .`, then verify:

```bash
uv run tmgg-my-model --help       # Should print Hydra help
uv run tmgg-my-model --cfg job    # Should print the composed config
```

### Verification checklist

- `uv run tmgg-my-model --cfg job` prints a valid composed config with your model's parameters.
- `uv run tmgg-my-model trainer.max_steps=10` runs a short training loop without errors.
- basedpyright reports no errors on your model file.

---

## Adding a new experiment type

Use this when you need fundamentally different training logic — a VAE, a GAN, a contrastive method, or anything that doesn't fit the existing denoising or diffusion pipelines.

**Prerequisites:** Completing "Adding a new model" above. Understanding of PyTorch Lightning's `training_step` / `validation_step` / `on_validation_epoch_end` lifecycle.

### Decide which base class to subclass

| If your experiment... | Subclass | Why |
|---|---|---|
| Corrupts graphs and predicts the clean version in one step | `SingleStepDenoisingModule` | Handles noise level iteration, per-level metrics, spectral deltas |
| Runs iterative forward/reverse diffusion (categorical or continuous) | `DiffusionModule` | Handles noise schedule, sampler, VLB estimation, generative evaluation |
| Has custom training logic that doesn't fit either | `BaseGraphModule` | Provides optimizer/scheduler config, parameter logging, and nothing else |

`BaseGraphModule` is the minimal base. It handles optimizer construction, LR scheduling, and hyperparameter saving. You implement `training_step`, `validation_step`, and `on_validation_epoch_end` yourself.

### Create the Lightning module

```python
# src/tmgg/experiments/vae_graph/lightning_module.py
from typing import Any

import torch
from tmgg.experiments._shared_utils.lightning_modules.base_graph_module import (
    BaseGraphModule,
)
from tmgg.models.base import GraphModel


class GraphVAEModule(BaseGraphModule):
    """Graph VAE training module with ELBO loss."""

    def __init__(
        self,
        *,
        model: GraphModel,
        model_name: str = "",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,
        # VAE-specific
        kl_weight: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            model_name=model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            scheduler_config=scheduler_config,
        )
        self.save_hyperparameters(ignore=["model"])
        self.kl_weight = kl_weight

    def training_step(self, batch, batch_idx):
        # Your ELBO computation here
        ...

    def validation_step(self, batch, batch_idx):
        ...
```

Call `self.save_hyperparameters(ignore=["model"])` in `__init__` — Lightning uses this for checkpoint reconstruction, and the model object itself is not serialisable.

### Understand the config composition hierarchy

TMGG configs compose in layers. Understanding this hierarchy prevents surprises:

```
_base_infra.yaml                  ← optimizer, scheduler, seed, paths, W&B, Hydra output
  │
  ├─ task/denoising.yaml          ← loss_type, noise config, data group (denoising only)
  │     └─ /data: sbm_default    ← graph generation params
  │
  ├─ base/trainer/default.yaml    ← Lightning Trainer settings
  ├─ base/logger/default.yaml     ← W&B/CSV logger config
  ├─ base/callbacks/default.yaml  ← checkpointing, early stopping
  └─ base/progress_bar/default.yaml
```

**For denoising experiments:** `base_config_denoising.yaml` composes `_base_infra` + `task/denoising` + trainer/logger/callback defaults. Your experiment config inherits from `base_config_denoising` and adds a model.

**For new experiment types:** Inherit from `_base_infra` directly (skip `task/denoising` since your experiment doesn't use noise levels or BCE loss). Define your own data and model sections.

```yaml
# src/tmgg/experiments/exp_configs/base_config_vae_graph.yaml
defaults:
  - _base_infra
  - models/vae/vae_default@model
  - _self_

experiment_name: vae_graph
wandb_project: tmgg-vae

# VAE-specific config (not from _base_infra or any task config)
kl_weight: 1.0

data:
  _target_: tmgg.data.data_modules.data_module.GraphDataModule
  graph_type: sbm
  num_graphs: 5000
  num_nodes: 20
  batch_size: ${batch_size:32}
  num_workers: ${num_workers}
  seed: ${seed}
```

Composing from `_base_infra` gives you optimizer settings, scheduler, seed, paths, and W&B config. Everything else (data, model, experiment-specific params) you define explicitly.

### Create model config, runner, and CLI entry point

Follow the same steps as "Adding a new model" above — create a model config YAML pointing at your Lightning module via `_target_`, a runner script, and a `pyproject.toml` entry.

### Config debugging

When the composed config doesn't look right:

```bash
# Print the fully composed config without running anything
uv run tmgg-vae-graph --cfg job

# Print just the model section
uv run tmgg-vae-graph --cfg job --package model

# Show which config file provided each value
uv run tmgg-vae-graph --info config
```

Common issues:

| Symptom | Cause | Fix |
|---|---|---|
| Your explicit keys are ignored | `_self_` not last in `defaults:` | Move `_self_` to the end |
| `InterpolationKeyError` on startup | A `${...}` references a key that doesn't exist in the composed config | Check that the referenced key is defined in `_base_infra` or your config |
| Model gets wrong parameters | `@model` suffix missing in defaults | Use `models/vae/vae_default@model` (the `@model` tells Hydra to mount the config under the `model:` key) |
| Hydra errors about unknown keys | An old `model.eval_num_samples` or similar removed parameter | Remove the key from your YAML |

### Verification checklist

1. `uv run tmgg-<name> --cfg job` prints a valid config.
2. `uv run tmgg-<name> trainer.max_steps=10` completes a short run.
3. Lightning module calls `save_hyperparameters(ignore=["model"])`.
4. basedpyright reports no errors on your module.
5. Runner function returns `run_experiment(cfg)` (Hydra sweeps depend on this).

---

## Adding a new synthetic graph generator

Use this when you want to train on a new graph family (e.g., Watts-Strogatz, Barabási-Albert) using the existing data pipeline.

**Prerequisites:** The graph generator must produce adjacency matrices as NumPy arrays of shape `(num_graphs, n, n)`.

### Implement the generator

Add a function to `src/tmgg/data/datasets/synthetic_graphs.py`:

```python
def generate_my_graphs(
    n: int,
    num_graphs: int,
    my_param: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate my custom graph type.

    Parameters
    ----------
    n
        Number of nodes per graph.
    num_graphs
        Number of graphs to generate.
    my_param
        Controls some property of the generated graphs.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, n, n)``.
    """
    rng = np.random.default_rng(seed)
    graphs = np.zeros((num_graphs, n, n))
    for i in range(num_graphs):
        # Your generation logic here
        ...
    return graphs
```

### Register the graph type

In `SyntheticGraphDataset` (same file), add your type to `VALID_TYPES` and the dispatch in `_generate()`:

```python
VALID_TYPES = {..., "my_type"}

def _generate(self):
    if self.graph_type == "my_type":
        return generate_my_graphs(
            self.n, self.num_graphs, seed=self.seed,
            my_param=self.kwargs.get("my_param", 0.5),
        )
```

### Create a data config

```yaml
# src/tmgg/experiments/exp_configs/data/my_graphs.yaml
graph_type: my_type
num_graphs: 5000
num_nodes: 20
graph_config:
  my_param: 0.5
```

Use it by overriding the data group: `uv run tmgg-spectral-arch data=my_graphs`.

---

## Adding a new noise type

Use this when you need a new corruption process for denoising experiments (e.g., node dropout, spectral perturbation).

**Prerequisites:** Understanding of how noise generators work — see `src/tmgg/data/noising/noise.py` for existing implementations.

### Implement the noise function

Add a new `NoiseGenerator` subclass in `src/tmgg/data/noising/noise.py`:

```python
class MyNoiseGenerator(NoiseGenerator):
    """Applies my custom noise to adjacency matrices."""

    def add_noise(self, A: torch.Tensor, eps: float) -> torch.Tensor:
        """Apply noise at level eps to adjacency batch A.

        Parameters
        ----------
        A
            Clean adjacency matrices, shape ``(batch, n, n)``.
        eps
            Noise level in [0, 1].

        Returns
        -------
        torch.Tensor
            Noisy adjacency matrices, same shape.
        """
        # Your noise implementation
        ...
```

### Register the noise type

In `src/tmgg/data/noising/noise.py`, add your type to `create_noise_generator`:

```python
def create_noise_generator(noise_type: str, **kwargs) -> NoiseGenerator:
    if noise_type == "my_noise":
        return MyNoiseGenerator(**kwargs)
    ...
```

Then use it in configs: `noise_type: my_noise`.
