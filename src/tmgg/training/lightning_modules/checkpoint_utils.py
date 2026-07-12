"""Utilities for loading checkpoints with mismatched hyperparameters."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import TypeVar

import pytorch_lightning as pl
import torch

from tmgg.training.exceptions import CheckpointMismatchError

T = TypeVar("T", bound=pl.LightningModule)


def load_checkpoint_with_fallback[T: pl.LightningModule](
    module_class: type[T],
    checkpoint_path: str | Path,
    strict: bool = True,
    map_location: str | torch.device | None = None,
    allow_unknown_hparams: bool = False,
    **override_kwargs,
) -> T:
    """
    Load a LightningModule from checkpoint with fallback for mismatched hyperparameters.

    When checkpoint contains hyperparameters that module's __init__ no longer accepts,
    raises an error by default. Set allow_unknown_hparams=True to filter them with a
    warning instead.

    Parameters
    ----------
    module_class
        The LightningModule subclass to instantiate.
    checkpoint_path
        Path to the checkpoint file.
    strict
        Whether to strictly enforce state_dict key matching.
    map_location
        Device to map tensors to during loading.
    allow_unknown_hparams
        If True, filter unknown hyperparameters with a warning instead of raising.
        Defaults to False for strict behavior.
    **override_kwargs
        Additional kwargs to pass to __init__, overriding checkpoint values.

    Returns
    -------
    T
        The loaded module instance.

    Raises
    ------
    CheckpointMismatchError
        If checkpoint contains unknown hyperparameters and allow_unknown_hparams=False.
    TypeError
        If loading fails for reasons other than unexpected keyword arguments.
    """
    checkpoint_path = Path(checkpoint_path)

    # Attempt normal loading
    saved_type_error: TypeError | None = None
    try:
        return module_class.load_from_checkpoint(
            str(checkpoint_path),
            strict=strict,
            map_location=map_location,
            **override_kwargs,
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        saved_type_error = e

    # Introspect __init__ to find accepted parameters
    sig = inspect.signature(module_class.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        raise saved_type_error  # **kwargs accepts anything, filtering won't help

    # Load checkpoint and filter hyperparameters
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    stored_hparams = checkpoint.get("hyper_parameters", {})

    filtered_hparams = {}
    removed_keys = []
    for key, value in stored_hparams.items():
        if key in accepted_params:
            filtered_hparams[key] = value
        else:
            removed_keys.append(key)

    if removed_keys:
        message = (
            f"Checkpoint '{checkpoint_path}' contains hyperparameters not accepted by "
            f"{module_class.__name__}.__init__: {removed_keys}. "
            f"Set allow_unknown_hparams=True to filter them."
        )
        if not allow_unknown_hparams:
            raise CheckpointMismatchError(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    # Instantiate with filtered hparams and load state dict
    final_hparams = {**filtered_hparams, **override_kwargs}
    model = module_class(**final_hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)

    return model
