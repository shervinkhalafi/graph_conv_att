"""Checkpoint format detection and remapping for DiGress models.

This module handles loading pretrained DiGress checkpoints regardless of their
format and remaps state dict keys to match the target model architecture.

Three checkpoint formats are supported:
- ORIGINAL_DIGRESS: Keys like "model.mlp_in_X.0.weight"
- TMGG_LIGHTNING: Keys like "model.transformer.mlp_in_X.0.weight" (Lightning wrapper)
- TMGG_RAW: Keys like "transformer.mlp_in_X.0.weight" (raw model state dict)
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import torch


class _DigressUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules from original DiGress.

    Original DiGress checkpoints may contain pickled references to modules like
    `datasets.guacamol_dataset` that don't exist in tmgg. This unpickler
    substitutes placeholder classes for such missing modules.
    """

    # Modules from original DiGress that may be referenced in checkpoints
    DIGRESS_MODULES = {
        "datasets",
        "datasets.guacamol_dataset",
        "datasets.spectre_dataset",
        "datasets.qm9_dataset",
        "datasets.moses_dataset",
        "diffusion",
        "diffusion.noise_schedule",
        "models",
        "models.transformer_model",
    }

    def find_class(self, module: str, name: str) -> Any:
        """Override to handle missing DiGress modules."""
        # Check if this is a DiGress module that we should stub
        if module in self.DIGRESS_MODULES or module.startswith(
            ("datasets.", "diffusion.", "models.")
        ):
            # Return a placeholder that can be pickled but ignored
            # We create a simple namespace object that won't break unpickling
            return _create_placeholder_class(module, name)

        # Fall back to default behavior
        return super().find_class(module, name)


def _create_placeholder_class(module: str, name: str) -> type:
    """Create a placeholder class for missing DiGress modules."""
    # Create a simple class that can be instantiated without args
    # and has a __reduce__ that returns a simple tuple
    placeholder = type(
        name,
        (),
        {
            "__module__": module,
            "__reduce__": lambda self: (type(self), ()),
            "__repr__": lambda self: f"<Placeholder for {module}.{name}>",
        },
    )
    return placeholder


def _load_checkpoint_with_module_remapping(
    path: Path,
    map_location: str | torch.device | None = None,
) -> Any:
    """Load a checkpoint file, handling missing modules from original DiGress.

    Parameters
    ----------
    path
        Path to checkpoint file.
    map_location
        Device to map tensors to.

    Returns
    -------
    Any
        Loaded checkpoint data.
    """
    with open(path, "rb") as f:
        # Read the file content
        file_content = f.read()

    # Try normal loading first
    try:
        return torch.load(
            io.BytesIO(file_content), map_location=map_location, weights_only=False
        )
    except ModuleNotFoundError:
        pass  # Fall through to custom loading

    # Load with stub modules for missing DiGress modules
    buffer = io.BytesIO(file_content)
    return _load_pytorch_zip_checkpoint(buffer, map_location)


class _StubModule(ModuleType):
    """A stub module that dynamically creates placeholder classes for any attribute.

    Also acts as a package, allowing submodule imports.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__file__ = f"<stub:{name}>"
        self.__path__ = [f"<stub:{name}>"]  # Make it a package
        self.__package__ = name
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        # Let standard dunder attributes raise AttributeError
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = _create_placeholder_class(self.__name__, name)
        return self._cache[name]


def _install_stub_module(name: str, installed: dict[str, Any]) -> _StubModule:
    """Install a stub module and its parents into sys.modules."""
    import sys

    if name in sys.modules:
        return cast(_StubModule, sys.modules[name])

    # Install parent first if needed
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = _install_stub_module(parent_name, installed)
        stub = _StubModule(name)
        sys.modules[name] = stub
        installed[name] = None
        setattr(parent, child_name, stub)
        return stub
    else:
        stub = _StubModule(name)
        sys.modules[name] = stub
        installed[name] = None
        return stub


def _load_pytorch_zip_checkpoint(
    buffer: io.BytesIO,
    map_location: str | torch.device | None = None,
) -> Any:
    """Load PyTorch ZIP checkpoint with module remapping.

    Pre-populates sys.modules with stub modules for DiGress-specific
    imports that don't exist in the tmgg environment.
    """
    import sys

    # All DiGress modules referenced in checkpoints (found by inspecting pickle data)
    # We install them all upfront to handle any pickle references
    stub_modules = [
        # Analysis
        "analysis",
        "analysis.visualization",
        # Datasets
        "datasets",
        "datasets.guacamol_dataset",
        "datasets.spectre_dataset",
        "datasets.qm9_dataset",
        "datasets.moses_dataset",
        "datasets.abstract_dataset",
        # Diffusion
        "diffusion",
        "diffusion.distributions",
        "diffusion.extra_features",
        "diffusion.extra_features_molecular",
        "diffusion.noise_schedule",
        "diffusion.diffusion_utils",
        # Top-level modules
        "diffusion_model",
        "diffusion_model_discrete",
        "utils",
        # Models
        "models",
        "models.transformer_model",
        # Metrics (many variants found in guacamol checkpoint)
        "metrics",
        "metrics.abstract_metrics",
        "metrics.molecular_metrics",
        "metrics.sampling_metrics",
        "metrics.edge_dist_maer",
        "metrics.edge_target_distq",
        "metrics.generated_edge_distr",
        "metrics.generated_n_distr",
        "metrics.generated_node_distr",
        "metrics.generated_valency_distr",
        "metrics.n_dist_maer",
        "metrics.n_target_distq",
        "metrics.node_dist_maer",
        "metrics.node_target_distq",
        "metrics.train_atom_metrics",
        "metrics.train_atom_metricsr",
        "metrics.train_bond_metrics",
        "metrics.train_bond_metricsr",
        "metrics.valency_dist_maer",
        "metrics.valency_target_distq",
        # src-prefixed variants (some checkpoints use this)
        "src",
        "src.analysis",
        "src.analysis.visualization",
        "src.datasets",
        "src.datasets.guacamol_dataset",
        "src.datasets.spectre_dataset",
        "src.datasets.qm9_dataset",
        "src.datasets.moses_dataset",
        "src.datasets.abstract_dataset",
        "src.diffusion",
        "src.diffusion.distributions",
        "src.diffusion.extra_features",
        "src.diffusion.extra_features_molecular",
        "src.diffusion.noise_schedule",
        "src.diffusion.diffusion_utils",
        "src.diffusion_model",
        "src.diffusion_model_discrete",
        "src.models",
        "src.models.transformer_model",
        "src.metrics",
        "src.metrics.abstract_metrics",
        "src.metrics.molecular_metrics",
        "src.metrics.sampling_metrics",
        "src.utils",
    ]

    # Track which modules we've installed so we can clean up
    installed: dict[str, Any] = {}

    # Save originals and install stubs
    for mod_name in stub_modules:
        if mod_name in sys.modules:
            installed[mod_name] = sys.modules[mod_name]
        else:
            _install_stub_module(mod_name, installed)

    try:
        buffer.seek(0)
        return torch.load(buffer, map_location=map_location, weights_only=False)
    finally:
        # Restore original sys.modules state
        for mod_name, original in installed.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original


class CheckpointFormat(Enum):
    """Checkpoint format variants for DiGress models."""

    ORIGINAL_DIGRESS = auto()  # model.mlp_in_X, model.tf_layers
    TMGG_LIGHTNING = auto()  # model.transformer.mlp_in_X, model.transformer.tf_layers
    TMGG_RAW = auto()  # transformer.mlp_in_X, transformer.tf_layers
    UNKNOWN = auto()


def detect_checkpoint_format(state_dict: dict[str, Any]) -> CheckpointFormat:
    """Detect checkpoint format from state dict key patterns.

    Parameters
    ----------
    state_dict
        Model state dictionary.

    Returns
    -------
    CheckpointFormat
        Detected format.
    """
    keys = set(state_dict.keys())

    # Check for tmgg Lightning format (model.transformer.*)
    if any(k.startswith("model.transformer.") for k in keys):
        return CheckpointFormat.TMGG_LIGHTNING

    # Check for tmgg raw format (transformer.*)
    if any(k.startswith("transformer.mlp_in") for k in keys):
        return CheckpointFormat.TMGG_RAW

    # Check for original DiGress format (model.mlp_in*, model.tf_layers*)
    if any(
        k.startswith("model.mlp_in") or k.startswith("model.tf_layers") for k in keys
    ):
        return CheckpointFormat.ORIGINAL_DIGRESS

    return CheckpointFormat.UNKNOWN


def _get_key_prefixes(fmt: CheckpointFormat) -> tuple[str, str]:
    """Get transformer and model prefixes for a format.

    Returns (transformer_prefix, model_prefix) where transformer_prefix is the
    prefix for transformer layers and model_prefix is the prefix for the whole model.
    """
    if fmt == CheckpointFormat.ORIGINAL_DIGRESS:
        return ("model.", "model.")
    elif fmt == CheckpointFormat.TMGG_LIGHTNING:
        return ("model.transformer.", "model.")
    elif fmt == CheckpointFormat.TMGG_RAW:
        return ("transformer.", "")
    else:
        raise ValueError(f"Cannot determine prefixes for format: {fmt}")


def remap_state_dict(
    state_dict: dict[str, Any],
    source_format: CheckpointFormat,
    target_format: CheckpointFormat,
) -> dict[str, Any]:
    """Remap state dict keys between checkpoint formats.

    Parameters
    ----------
    state_dict
        Source state dictionary.
    source_format
        Format of the source state dict.
    target_format
        Desired format for the output state dict.

    Returns
    -------
    dict
        State dictionary with remapped keys.
    """
    if source_format == target_format:
        return state_dict

    if source_format == CheckpointFormat.UNKNOWN:
        raise ValueError("Cannot remap from UNKNOWN format")

    source_prefix, _ = _get_key_prefixes(source_format)
    target_prefix, _ = _get_key_prefixes(target_format)

    remapped: dict[str, Any] = {}

    # DiGress transformer component keys to remap
    # These are the key patterns that differ between formats:
    # - mlp_in_X, mlp_in_E, mlp_in_y (input MLPs)
    # - mlp_out_X, mlp_out_E, mlp_out_y (output MLPs)
    # - tf_layers (transformer layers)
    # - lin_X, lin_E, lin_y (final linear layers)
    # - y_norm (final layer norm)
    transformer_components = (
        "mlp_in_X",
        "mlp_in_E",
        "mlp_in_y",
        "mlp_out_X",
        "mlp_out_E",
        "mlp_out_y",
        "tf_layers",
        "lin_X",
        "lin_E",
        "lin_y",
        "y_norm",
    )

    for key, value in state_dict.items():
        new_key = key

        # Check if this key belongs to a transformer component
        for comp in transformer_components:
            full_source = f"{source_prefix}{comp}"
            if key.startswith(full_source):
                # Replace the source prefix with target prefix
                suffix = key[len(full_source) :]
                new_key = f"{target_prefix}{comp}{suffix}"
                break

        remapped[new_key] = value

    return remapped


@dataclass
class LoadedCheckpoint:
    """Result of loading a DiGress checkpoint.

    Attributes
    ----------
    state_dict
        Model state dictionary (remapped to target format).
    hyper_parameters
        Hyperparameters from checkpoint, if available.
    original_format
        Detected format of the original checkpoint.
    target_format
        Format the state dict was remapped to.
    checkpoint_path
        Path to the loaded checkpoint.
    """

    state_dict: dict[str, Any]
    hyper_parameters: dict[str, Any]
    original_format: CheckpointFormat
    target_format: CheckpointFormat
    checkpoint_path: Path


def load_digress_checkpoint(
    path: str | Path,
    target_format: CheckpointFormat = CheckpointFormat.TMGG_LIGHTNING,
    map_location: str | torch.device | None = None,
) -> LoadedCheckpoint:
    """Load a DiGress checkpoint with automatic format detection and remapping.

    Parameters
    ----------
    path
        Path to checkpoint file.
    target_format
        Target format for the state dict. Defaults to TMGG_LIGHTNING which
        is compatible with the tmgg Lightning checkpoint layout.
    map_location
        Device to map tensors to during loading.

    Returns
    -------
    LoadedCheckpoint
        Loaded and remapped checkpoint data.
    """
    path = Path(path)

    # Load checkpoint with module remapping for original DiGress checkpoints
    checkpoint = _load_checkpoint_with_module_remapping(path, map_location)

    # Extract state dict - handle both Lightning checkpoints and raw state dicts
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        hparams = checkpoint.get("hyper_parameters", {})
    else:
        # Assume it's a raw state dict
        state_dict = checkpoint
        hparams = {}

    # Detect format
    original_format = detect_checkpoint_format(state_dict)

    if original_format == CheckpointFormat.UNKNOWN:
        raise ValueError(
            f"Could not detect checkpoint format. Keys sample: {list(state_dict.keys())[:5]}"
        )

    # Remap state dict
    remapped_state_dict = remap_state_dict(state_dict, original_format, target_format)

    return LoadedCheckpoint(
        state_dict=remapped_state_dict,
        hyper_parameters=hparams,
        original_format=original_format,
        target_format=target_format,
        checkpoint_path=path,
    )


def load_compatible_state_dict(
    path: str | Path,
    model_format: CheckpointFormat = CheckpointFormat.TMGG_LIGHTNING,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Convenience function to get a state dict compatible with a target model format.

    Parameters
    ----------
    path
        Path to checkpoint file.
    model_format
        Format expected by the target model.
    map_location
        Device to map tensors to.

    Returns
    -------
    dict
        State dictionary remapped to the target format.
    """
    loaded = load_digress_checkpoint(path, model_format, map_location)
    return loaded.state_dict
