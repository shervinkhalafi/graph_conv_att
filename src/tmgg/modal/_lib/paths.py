"""Path discovery utilities for Modal experiments.

Since modal is now part of the tmgg package, path discovery is straightforward.
The module lives at src/tmgg/modal/, so paths are resolved relative to that.
"""

from __future__ import annotations

import os
from pathlib import Path


def _is_source_checkout_root(path: Path) -> bool:
    """Return True when *path* looks like the repo root checkout."""
    return (path / "src" / "tmgg").exists() and (path / "pyproject.toml").exists()


def _is_runtime_package_root(path: Path) -> bool:
    """Return True when *path* looks like a package root inside a container."""
    tmgg_pkg = path / "tmgg"
    return (tmgg_pkg / "modal").exists() and (tmgg_pkg / "__init__.py").exists()


def _discover_tmgg_path_from_module_file(module_file: Path) -> Path:
    """Infer the TMGG root from the current module file path.

    Supports both source checkout layout and installed package layout inside
    Modal runtime containers.
    """
    resolved = module_file.resolve()
    candidates = [resolved.parent, *resolved.parents]
    for candidate in candidates:
        if _is_source_checkout_root(candidate):
            return candidate
    for candidate in candidates:
        if _is_runtime_package_root(candidate):
            return candidate
    raise RuntimeError(
        f"Could not infer TMGG root from module path {module_file}. "
        "Expected either a source checkout with src/tmgg or a runtime package "
        "layout with tmgg/__init__.py."
    )


def discover_tmgg_path() -> Path:
    """Get the tmgg package root directory.

    Since modal is now inside tmgg, this is straightforward.
    The package structure is:
        tmgg/           <- returned path
          src/
            tmgg/
              modal/
                paths.py  <- this file

    Returns
    -------
    Path
        Path to tmgg package root (the directory containing src/).

    Notes
    -----
    TMGG_PATH environment variable can override for special cases
    (e.g., development with editable installs in non-standard locations).
    """
    env_path = os.environ.get("TMGG_PATH")
    if env_path:
        path = Path(env_path)
        if _is_source_checkout_root(path) or _is_runtime_package_root(path):
            return path
        raise ValueError(
            f"TMGG_PATH={env_path} does not contain valid tmgg package "
            "(expected source checkout or runtime package layout)"
        )

    return _discover_tmgg_path_from_module_file(Path(__file__))


def _is_valid_tmgg_path(path: Path) -> bool:
    """Check if path contains a valid tmgg package."""
    return _is_source_checkout_root(path) or _is_runtime_package_root(path)


def discover_source_checkout_path(module_file: Path | None = None) -> Path | None:
    """Return the repo root only when running from a source checkout.

    Modal runtime containers import the package from a flattened installed
    layout, not from ``<repo>/src/tmgg``. Image construction should use the
    source checkout only when that layout is actually present.
    """
    root = (
        _discover_tmgg_path_from_module_file(module_file)
        if module_file is not None
        else discover_tmgg_path()
    )
    if _is_source_checkout_root(root):
        return root
    return None


def require_tmgg_path() -> Path:
    """Get tmgg path (always succeeds since we're inside the package).

    Returns
    -------
    Path
        Path to tmgg package root.
    """
    return discover_tmgg_path()
