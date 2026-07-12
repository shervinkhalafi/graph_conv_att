"""Tests for Modal image build helpers.

Test rationale
--------------
The Modal image must compile ORCA at the package runtime location inside the
container. If the build points at a stale source directory or compiles into an
unused side directory, the container can end up executing a host-built binary
with incompatible glibc/libstdc++ versions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tmgg.modal._lib.image import (
    MODAL_IGNORE_PATTERNS,
    _compile_orca_in_image,
    _project_pip_runtime_deps,
    _resolve_orca_source_dir,
    _runtime_env,
    create_tmgg_image,
)


def test_resolve_orca_source_dir_points_at_current_package_location() -> None:
    """ORCA build helper should target the current in-package source directory."""
    repo_root = Path(__file__).resolve().parents[2]

    src_dir = _resolve_orca_source_dir(repo_root)

    assert src_dir == repo_root / "src" / "tmgg" / "evaluation" / "orca"
    assert src_dir.exists()


def test_resolve_orca_source_dir_fails_for_missing_source(tmp_path: Path) -> None:
    """Missing ORCA source should fail loudly during image construction."""
    with pytest.raises(FileNotFoundError, match="ORCA source not found"):
        _resolve_orca_source_dir(tmp_path)


def test_compile_orca_in_image_targets_runtime_module_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image build step should compile ORCA at the package runtime path."""
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        commands.append(cmd)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    _compile_orca_in_image()

    assert len(commands) == 1
    cmd = commands[0]
    assert cmd[:4] == ["g++", "-O2", "-std=c++11", "-o"]
    assert cmd[4].endswith("/src/tmgg/evaluation/orca/orca")
    assert cmd[5].endswith("/src/tmgg/evaluation/orca/orca.cpp")


def test_compile_orca_in_image_removes_preexisting_binary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Build step should delete any stale binary before compiling ORCA again.

    Regression rationale
    --------------------
    A checked-in host-built ORCA executable can be copied into Modal's runtime
    source tree and shadow the image-built package. The image build helper
    should remove any preexisting binary before invoking ``g++`` so the rebuilt
    container artifact is deterministic.
    """
    orca_dir = tmp_path / "orca"
    orca_dir.mkdir()
    (orca_dir / "orca.cpp").write_text("// fake source\n")
    stale_binary = orca_dir / "orca"
    stale_binary.write_text("stale")

    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        assert stale_binary.exists() is False
        commands.append(cmd)

    monkeypatch.setattr("tmgg.modal._lib.image._runtime_orca_dir", lambda: orca_dir)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    _compile_orca_in_image()

    assert len(commands) == 1
    assert stale_binary.exists() is False


def test_runtime_env_stays_minimal_for_single_conda_runtime() -> None:
    """Unified micromamba runtime should not need OpenMP preload shims."""
    env = _runtime_env()

    assert env["WANDB_MODE"] == "online"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "LD_PRELOAD" not in env


def test_project_pip_runtime_deps_cover_declared_runtime_requirements() -> None:
    """Local-source image builds should derive pip deps from pyproject.

    Regression rationale
    --------------------
    The Modal image installs ``tmgg`` with ``--no-deps`` to avoid re-solving
    the native Torch stack via pip. That makes the explicit pip dependency list
    critical. It should be derived from ``pyproject.toml`` so newly added
    runtime packages such as ``pot`` or ``safetensors`` are not silently
    omitted from the image.
    """
    repo_root = Path(__file__).resolve().parents[2]

    deps = _project_pip_runtime_deps(repo_root)

    assert "pot>=0.9.0" in deps
    assert "torch-geometric>=2.5.0" in deps
    assert "rich>=13.0.0" in deps
    assert "click>=8.0.0" in deps
    assert "pandas>=2.0.0" in deps
    assert "seaborn>=0.13.2" in deps
    assert "safetensors>=0.4.0" in deps
    assert "polars>=1.0.0" in deps
    assert all(not dep.startswith("torch>=") for dep in deps)
    assert all(not dep.startswith("modal>=") for dep in deps)


def test_create_tmgg_image_keeps_native_stack_in_micromamba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary-heavy deps should keep one native runtime.

    Regression rationale
    --------------------
    Installing ``graph-tool`` from conda-forge and PyTorch from pip creates two
    independent binary stacks. The image should install ``graph-tool``,
    ``pytorch``, ``torchvision``, and the matching CUDA runtime together via
    micromamba so they share one solver and one native runtime. ``PyG`` is the
    exception: its docs no longer provide conda packages for torch ``>2.5.0``,
    so the pure-Python ``torch-geometric`` package should be installed via pip.
    """

    class FakeImage:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        def apt_install(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("apt_install", args, kwargs))
            return self

        def micromamba_install(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("micromamba_install", args, kwargs))
            return self

        def uv_pip_install(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("uv_pip_install", args, kwargs))
            return self

        def run_commands(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("run_commands", args, kwargs))
            return self

        def env(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("env", args, kwargs))
            return self

    fake_image = FakeImage()

    class FakeImageFactory:
        @staticmethod
        def micromamba(*args: object, **kwargs: object) -> FakeImage:
            fake_image.calls.append(("micromamba", args, kwargs))
            return fake_image

    monkeypatch.setattr("tmgg.modal._lib.image.modal.Image", FakeImageFactory)

    image = create_tmgg_image(None)

    assert image is fake_image

    micromamba_calls = [
        call for call in fake_image.calls if call[0] == "micromamba_install"
    ]
    assert len(micromamba_calls) == 1
    packages = micromamba_calls[0][1]
    channels = micromamba_calls[0][2]["channels"]
    assert "graph-tool" in packages
    assert "pytorch==2.5.1" in packages
    assert "torchvision==0.20.1" in packages
    assert "pytorch-cuda=12.1" in packages
    assert channels == ["pytorch", "nvidia", "conda-forge"]

    pip_calls = [call for call in fake_image.calls if call[0] == "uv_pip_install"]
    pip_args = [str(arg) for _, args, _ in pip_calls for arg in args]
    assert "torch-geometric>=2.5.0" in pip_args
    assert "pot>=0.9.0" in pip_args
    assert "rich>=13.0.0" in pip_args
    assert "click>=8.0.0" in pip_args
    assert "pandas>=2.0.0" in pip_args
    assert "seaborn>=0.13.2" in pip_args
    assert "safetensors>=0.4.0" in pip_args
    assert "polars>=1.0.0" in pip_args
    assert "torch>=2.0.0" not in pip_args
    assert "torchvision" not in pip_args
    assert "tmgg[cloud]" not in pip_args

    run_command_calls = [call for call in fake_image.calls if call[0] == "run_commands"]
    run_commands = [str(arg) for _, args, _ in run_command_calls for arg in args]
    assert any("--no-deps tmgg[cloud]" in cmd for cmd in run_commands)


def test_create_tmgg_image_local_source_build_only_uploads_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-source image builds should ignore root-level helper churn.

    Regression rationale
    --------------------
    Modal rebuilds should depend only on the files that actually enter the
    image. Adding root-level helper scripts such as ``DEBUG-run-*.zsh`` should
    not invalidate the experiment image, because the local-source build path
    uploads only ``src/`` plus packaging metadata.
    """

    class FakeImage:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        def apt_install(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("apt_install", args, kwargs))
            return self

        def micromamba_install(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("micromamba_install", args, kwargs))
            return self

        def add_local_dir(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("add_local_dir", args, kwargs))
            return self

        def add_local_file(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("add_local_file", args, kwargs))
            return self

        def uv_pip_install(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("uv_pip_install", args, kwargs))
            return self

        def run_commands(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("run_commands", args, kwargs))
            return self

        def run_function(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("run_function", args, kwargs))
            return self

        def env(self, *args: object, **kwargs: object) -> FakeImage:
            self.calls.append(("env", args, kwargs))
            return self

    fake_image = FakeImage()

    class FakeImageFactory:
        @staticmethod
        def micromamba(*args: object, **kwargs: object) -> FakeImage:
            fake_image.calls.append(("micromamba", args, kwargs))
            return fake_image

    repo_root = Path(__file__).resolve().parents[2]

    monkeypatch.setattr("tmgg.modal._lib.image.modal.Image", FakeImageFactory)
    monkeypatch.setattr(
        "tmgg.modal._lib.image._project_pip_runtime_deps",
        lambda tmgg_path: ["dummy-dep>=1.0"],
    )

    image = create_tmgg_image(repo_root)

    assert image is fake_image

    add_local_dir_calls = [
        call for call in fake_image.calls if call[0] == "add_local_dir"
    ]
    assert add_local_dir_calls == [
        (
            "add_local_dir",
            (str(repo_root / "src"), "/app/tmgg/src"),
            {"ignore": MODAL_IGNORE_PATTERNS, "copy": True},
        )
    ]

    add_local_file_calls = [
        call for call in fake_image.calls if call[0] == "add_local_file"
    ]
    uploaded_files = {args[0] for _, args, _ in add_local_file_calls}
    assert uploaded_files == {
        str(repo_root / "pyproject.toml"),
        str(repo_root / "README.md"),
    }
