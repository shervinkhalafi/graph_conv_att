"""Tests for graph structural evaluation metrics (via graph_evaluator).

Verifies planarity, uniqueness, novelty, and orbit MMD against known
graphs with deterministic structure. SBM accuracy tests are skipped
when graph-tool is unavailable; ORCA tests are skipped when the binary
cannot be compiled.

Test rationale
--------------
Each test uses small graphs with known properties (K4 is planar, K5 is
not; identical graphs have uniqueness < 1; novel graphs are not in the
training set) to validate the metric functions in isolation. The orbit
MMD test uses the ``run_orca`` auto-compilation feature, so it is skipped
on systems without g++.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from tmgg.evaluation.graph_evaluator import (
    compute_novelty,
    compute_orbit_mmd,
    compute_planarity_accuracy,
    compute_uniqueness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _can_compile_orca() -> bool:
    """Check if ORCA binary can be compiled (g++ available)."""
    import shutil

    return shutil.which("g++") is not None


# ---------------------------------------------------------------------------
# Planarity
# ---------------------------------------------------------------------------


class TestPlanarity:
    """Planarity accuracy on graphs with known planarity status."""

    def test_compute_planarity_accuracy(self) -> None:
        """Mixed set: 2 planar + 1 non-planar -> 2/3 accuracy."""
        graphs = [nx.complete_graph(4), nx.cycle_graph(6), nx.complete_graph(5)]
        acc = compute_planarity_accuracy(graphs)
        assert abs(acc - 2.0 / 3.0) < 1e-6

    def test_eval_planarity_empty(self) -> None:
        """Empty list returns 0.0."""
        assert compute_planarity_accuracy([]) == 0.0


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


class TestUniqueness:
    """Uniqueness metric on graphs with known isomorphism structure.

    The graph_evaluator version of compute_uniqueness always uses exact
    isomorphism testing (faster_could_be_isomorphic + is_isomorphic),
    with no ``precise`` parameter.
    """

    def test_all_identical(self) -> None:
        """N copies of the same graph: only 1 unique isomorphism class."""
        G = nx.cycle_graph(5)
        graphs = [G.copy() for _ in range(10)]
        u = compute_uniqueness(graphs)
        assert u == pytest.approx(1.0 / 10.0)

    def test_all_different(self) -> None:
        """Graphs with different sizes are trivially non-isomorphic."""
        graphs = [nx.cycle_graph(n) for n in range(3, 13)]
        u = compute_uniqueness(graphs)
        assert u == 1.0

    def test_empty_input(self) -> None:
        """Edge case: empty list returns 0.0 (no graphs means no unique ones)."""
        assert compute_uniqueness([]) == 0.0

    def test_skips_empty_graphs(self) -> None:
        """Empty graphs (0 nodes) are skipped in the isomorphism check.

        Since empty graphs are never compared, they never count as
        non-unique -- so all 3 graphs are "unique" (uniqueness = 1.0).
        This matches upstream DiGress behavior.
        """
        graphs = [nx.Graph(), nx.Graph(), nx.cycle_graph(4)]
        u = compute_uniqueness(graphs)
        assert u == 1.0


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


class TestNovelty:
    """Novelty metric checking against a training set."""

    def test_all_novel(self) -> None:
        """No generated graph is isomorphic to any training graph."""
        train = [nx.cycle_graph(5)]
        gen = [nx.cycle_graph(6), nx.cycle_graph(7)]
        assert compute_novelty(gen, train) == 1.0

    def test_all_copied(self) -> None:
        """All generated graphs are copies of training graphs."""
        train = [nx.cycle_graph(5)]
        gen = [nx.cycle_graph(5), nx.cycle_graph(5)]
        assert compute_novelty(gen, train) == 0.0

    def test_partial_overlap(self) -> None:
        """One of two generated graphs is a copy."""
        train = [nx.cycle_graph(5)]
        gen = [nx.cycle_graph(5), nx.cycle_graph(6)]
        assert compute_novelty(gen, train) == pytest.approx(0.5)

    def test_empty_gen(self) -> None:
        """Empty generated set returns 0.0 (no graphs to be novel)."""
        assert compute_novelty([], [nx.cycle_graph(5)]) == 0.0


# ---------------------------------------------------------------------------
# Orbit MMD (requires ORCA)
# ---------------------------------------------------------------------------


class TestOrbitMMD:
    """Orbit MMD metric using ORCA orbit counts."""

    def test_get_binary_path_rebuilds_preexisting_binary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Wrapper should rebuild ORCA instead of trusting a preexisting ELF.

        Regression rationale
        --------------------
        Modal can expose the source tree under ``/root/tmgg``. If that tree
        contains a host-built ORCA executable, blindly reusing it can fail with
        glibc/libstdc++ mismatches inside the container. The wrapper should
        rebuild the binary on first use in a process without making the target
        path temporarily disappear for concurrent readers.
        """
        import tmgg.evaluation.orca as orca_mod

        orca_dir = tmp_path / "orca"
        orca_dir.mkdir()
        (orca_dir / "orca.cpp").write_text("// fake source\n")
        stale_binary = orca_dir / "orca"
        stale_binary.write_text("stale")

        commands: list[list[str]] = []

        def _fake_run(
            cmd: list[str],
            check: bool,
            capture_output: bool,
            text: bool,
        ) -> None:
            assert check is True
            assert capture_output is True
            assert text is True
            assert cmd[0] == "g++"
            assert "-O2" in cmd
            assert "-std=c++11" in cmd
            o_idx = cmd.index("-o")
            assert Path(cmd[o_idx + 1]) != stale_binary
            Path(cmd[o_idx + 1]).write_text("rebuilt")
            commands.append(cmd)

        monkeypatch.setattr(orca_mod, "_ORCA_DIR", orca_dir)
        monkeypatch.setattr(subprocess, "run", _fake_run)

        binary = orca_mod._get_binary_path()

        assert binary == stale_binary
        assert len(commands) == 1
        assert stale_binary.read_text() == "rebuilt"

    def test_run_orca_forwards_subprocess_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wrapper should surface ORCA stderr/stdout when the subprocess fails.

        Regression rationale
        --------------------
        Modal failures currently surface only as ``CalledProcessError`` with an
        exit status, which hides the actual ORCA diagnostic (for example
        duplicate edges or self-loops in the serialized graph). The wrapper
        should forward that diagnostic in a ``RuntimeError`` message.
        """
        import tmgg.evaluation.orca as orca_mod

        G = nx.path_graph(3)

        monkeypatch.setattr(orca_mod, "_get_binary_path", lambda: Path("/fake/orca"))

        def _raise_called_process_error(*args: object, **kwargs: object) -> bytes:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["/fake/orca", "node", "4", "/tmp/orca_input_fake.txt", "std"],
                output=b"Input file contains duplicate undirected edges.\n",
            )

        monkeypatch.setattr(
            orca_mod.subprocess, "check_output", _raise_called_process_error
        )

        with pytest.raises(RuntimeError, match="duplicate undirected edges"):
            orca_mod.run_orca(G)

    @pytest.mark.skipif(
        not _can_compile_orca(),
        reason="g++ not available for ORCA compilation",
    )
    def test_identical_distributions(self) -> None:
        """MMD between identical graph sets should be near zero."""
        graphs = [nx.cycle_graph(8) for _ in range(5)]
        mmd = compute_orbit_mmd(graphs, graphs)
        assert mmd < 0.01

    @pytest.mark.skipif(
        not _can_compile_orca(),
        reason="g++ not available for ORCA compilation",
    )
    def test_different_distributions(self) -> None:
        """MMD between structurally different graph sets should be positive.

        Uses sigma=1.0 (instead of the default 30.0) for a sharper kernel
        that resolves differences on small test graphs.
        """
        ref = [nx.cycle_graph(8) for _ in range(5)]
        gen = [nx.complete_graph(8) for _ in range(5)]
        mmd = compute_orbit_mmd(ref, gen, sigma=1.0)
        assert mmd > 0.01

    @pytest.mark.skipif(
        not _can_compile_orca(),
        reason="g++ not available for ORCA compilation",
    )
    def test_orbit_count_shape(self) -> None:
        """Verify ORCA produces correct shape for individual graphs."""
        from tmgg.evaluation.orca import run_orca

        G = nx.petersen_graph()  # 10 nodes
        counts = run_orca(G)
        assert counts.shape == (10, 15)


# ---------------------------------------------------------------------------
# SBM accuracy (requires graph-tool -- soft skip)
# ---------------------------------------------------------------------------


def _has_graph_tool() -> bool:
    """Check if graph-tool is importable."""
    try:
        import graph_tool  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


class TestSBMAccuracy:
    """SBM accuracy metric. Skipped when graph-tool is not installed."""

    @pytest.mark.skipif(not _has_graph_tool(), reason="graph-tool not installed")
    def test_sbm_accuracy_on_sbm_graphs(self) -> None:
        """Graphs generated from an SBM should pass the SBM test."""
        from tmgg.evaluation.graph_evaluator import (
            compute_sbm_accuracy,
        )

        rng = np.random.default_rng(42)
        graphs = []
        for _ in range(5):
            sizes = [30, 30]
            p = [[0.3, 0.005], [0.005, 0.3]]
            G = nx.stochastic_block_model(sizes, p, seed=int(rng.integers(10000)))
            graphs.append(G)
        acc = compute_sbm_accuracy(graphs, p_intra=0.3, p_inter=0.005)
        # Most SBM-generated graphs should pass
        assert acc >= 0.4

    @pytest.mark.skipif(not _has_graph_tool(), reason="graph-tool not installed")
    def test_sbm_accuracy_on_random_graphs(self) -> None:
        """Random ER graphs should mostly fail the SBM test."""
        from tmgg.evaluation.graph_evaluator import (
            compute_sbm_accuracy,
        )

        graphs = [nx.erdos_renyi_graph(60, 0.3, seed=i) for i in range(5)]
        acc = compute_sbm_accuracy(graphs, p_intra=0.3, p_inter=0.005)
        # ER graphs lack block structure
        assert acc <= 0.6

    def test_parallel_executor_is_process_pool_not_thread_pool(self) -> None:
        """Regression: concurrent graph-tool calls from a Python
        ``ThreadPoolExecutor`` abort the process with ``vector::_M_fill_insert``
        or ``malloc()`` heap-corruption aborts. ``compute_sbm_accuracy`` must
        dispatch its parallel work through a ``ProcessPoolExecutor`` with the
        ``spawn`` start method so each graph-tool call is isolated in a fresh
        interpreter.

        See ``internal notes (not included)``.

        We substitute fake executor classes for
        ``concurrent.futures.ProcessPoolExecutor`` and
        ``concurrent.futures.ThreadPoolExecutor`` inside the graph_evaluator
        module and assert the process one is instantiated, the thread one is
        not. The fake returns precomputed scores so the call succeeds without
        ever launching a real worker or touching graph-tool.
        """
        from collections.abc import Callable
        from typing import Any
        from unittest.mock import patch

        from tmgg.evaluation import graph_evaluator as ge_mod

        dummy_graphs = [nx.empty_graph(5) for _ in range(3)]

        class _FakeExecutor:
            instantiations: int = 0

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                type(self).instantiations += 1

            def __enter__(self) -> _FakeExecutor:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def map(
                self,
                _fn: Callable[..., Any],
                *iterables: Any,
            ) -> list[float]:
                n = len(list(iterables[0]))
                return [0.0] * n

        class FakeProcessPool(_FakeExecutor):
            instantiations = 0

        class FakeThreadPool(_FakeExecutor):
            instantiations = 0

        with (
            patch.object(
                ge_mod.concurrent.futures, "ProcessPoolExecutor", FakeProcessPool
            ),
            patch.object(
                ge_mod.concurrent.futures, "ThreadPoolExecutor", FakeThreadPool
            ),
        ):
            ge_mod.compute_sbm_accuracy(
                dummy_graphs,
                p_intra=0.3,
                p_inter=0.005,
                refinement_steps=0,
                is_parallel=True,
            )

        assert FakeProcessPool.instantiations == 1, (
            "compute_sbm_accuracy must use ProcessPoolExecutor to avoid "
            "graph-tool thread-unsafety (see 2026-04-15-bug-modal-sigabrt.md)"
        )
        assert FakeThreadPool.instantiations == 0, (
            "compute_sbm_accuracy regressed to ThreadPoolExecutor — this "
            "will abort the process on Modal when graph-tool runs. "
            "See internal notes (not included)"
        )
