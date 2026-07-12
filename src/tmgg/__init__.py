"""TMGG: Graph denoising experiments with Hydra and PyTorch Lightning."""

import math

from omegaconf import OmegaConf

from . import models

__version__ = "0.1.0"
__all__ = ["models"]

# Register custom OmegaConf resolvers
OmegaConf.register_new_resolver("tmgg_version", lambda: __version__, replace=True)
OmegaConf.register_new_resolver(
    "if_",
    lambda cond, true_val, false_val: true_val if cond else false_val,
    replace=True,
)
OmegaConf.register_new_resolver("not_", lambda x: not x, replace=True)
OmegaConf.register_new_resolver(
    "epochs_to_steps",
    lambda num_samples, batch_size, epochs: (
        math.ceil(int(num_samples) / int(batch_size)) * int(epochs)
    ),
    replace=True,
)
