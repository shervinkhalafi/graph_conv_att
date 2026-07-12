"""Tests for Modal path discovery helpers.

Test rationale
--------------
The same helper module runs in two layouts:

- source checkout: ``<repo>/src/tmgg/modal/_lib/paths.py``
- Modal runtime package: ``/root/tmgg/modal/_lib/paths.py``

The image-build path must use the source checkout only when that layout
actually exists. Runtime containers must not be misclassified as checkouts.
"""

from __future__ import annotations

from pathlib import Path

from tmgg.modal._lib.paths import (
    _discover_tmgg_path_from_module_file,
    discover_source_checkout_path,
)


def test_discover_tmgg_path_from_source_checkout_layout(tmp_path: Path) -> None:
    """Source-tree layout should resolve to the repo root."""
    repo_root = tmp_path / "repo"
    module_file = repo_root / "src" / "tmgg" / "modal" / "_lib" / "paths.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("")
    (repo_root / "src" / "tmgg" / "__init__.py").write_text("")
    (repo_root / "pyproject.toml").write_text("[project]\nname='tmgg'\n")

    assert _discover_tmgg_path_from_module_file(module_file) == repo_root


def test_discover_tmgg_path_from_runtime_package_layout(tmp_path: Path) -> None:
    """Installed-package layout should resolve to the package root parent."""
    runtime_root = tmp_path / "root"
    module_file = runtime_root / "tmgg" / "modal" / "_lib" / "paths.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("")
    (runtime_root / "tmgg" / "__init__.py").write_text("")

    assert _discover_tmgg_path_from_module_file(module_file) == runtime_root


def test_discover_source_checkout_path_returns_none_for_runtime_layout(
    tmp_path: Path,
) -> None:
    """Runtime package layout is not a source checkout and should return None."""
    runtime_root = tmp_path / "root"
    module_file = runtime_root / "tmgg" / "modal" / "_lib" / "paths.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("")
    (runtime_root / "tmgg" / "__init__.py").write_text("")

    assert discover_source_checkout_path(module_file=module_file) is None
