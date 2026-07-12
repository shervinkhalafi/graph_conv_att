"""Tests for GraphEvaluator -- unified evaluator merging MMD + structural metrics.

Test Rationale
--------------
GraphEvaluator is a stateless evaluator: callers pass refs and generated graphs
directly to evaluate(), and optionally train_graphs in the constructor for novelty.
These tests verify that evaluate(refs, generated) returns all expected fields,
that edge cases (too few references, missing train_graphs) are handled correctly,
and that internal truncation to eval_num_samples works.

The compute_orbit_mmd and compute_sbm_accuracy functions depend on external tools
(orca binary, graph-tool) that may not be available in all environments. Tests
for those metrics are skipped when the dependencies are missing.

All other metrics (degree/clustering/spectral MMD, planarity, uniqueness, novelty)
use only networkx and are always testable.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

import tmgg.evaluation.graph_evaluator as graph_evaluator_module
from tmgg.evaluation.graph_evaluator import (
    EvaluationResults,
    GraphEvaluator,
)

# ---------------------------------------------------------------------------
# Dependency availability checks
# ---------------------------------------------------------------------------
_graph_tool_available = graph_evaluator_module._GRAPH_TOOL_AVAILABLE

_orca_available: bool
try:
    from tmgg.evaluation.orca import (
        is_available as _orca_is_available,
    )

    _orca_available = _orca_is_available()
except Exception:
    _orca_available = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def er_graphs() -> list[nx.Graph[Any]]:
    """Ten Erdos-Renyi graphs with 12 nodes and ~30% edge density."""
    return [nx.erdos_renyi_graph(12, 0.3, seed=i) for i in range(10)]


@pytest.fixture
def different_er_graphs() -> list[nx.Graph[Any]]:
    """Ten Erdos-Renyi graphs with different parameters (higher density)."""
    return [nx.erdos_renyi_graph(12, 0.7, seed=i + 100) for i in range(10)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Verify GraphEvaluator accepts constructor parameters."""

    def test_default_construction(self) -> None:
        ev = GraphEvaluator(eval_num_samples=20)
        assert ev.eval_num_samples == 20

    def test_custom_kernel_and_sigma(self) -> None:
        ev = GraphEvaluator(eval_num_samples=5, kernel="gaussian", sigma=2.0)
        assert ev.kernel == "gaussian"
        assert ev.sigma == 2.0

    def test_default_kernel_is_gaussian_tv(self) -> None:
        ev = GraphEvaluator(eval_num_samples=5)
        assert ev.kernel == "gaussian_tv"

    def test_sbm_params_stored(self) -> None:
        ev = GraphEvaluator(eval_num_samples=5, p_intra=0.4, p_inter=0.01)
        assert ev.p_intra == 0.4
        assert ev.p_inter == 0.01

    def test_per_metric_sigmas_default_to_global(self) -> None:
        """Without overrides, per-metric sigma fields should be ``None`` so
        ``compute_mmd_metrics`` falls back to the global ``sigma``. This is
        the behaviour callers without per-metric awareness depend on."""
        ev = GraphEvaluator(eval_num_samples=5, sigma=2.0)
        assert ev.degree_sigma is None
        assert ev.clustering_sigma is None
        assert ev.spectral_sigma is None

    def test_per_metric_sigmas_stored_when_supplied(self) -> None:
        """Upstream DiGress parity uses ``clustering_sigma=0.1``; locking
        in that the constructor stores per-metric overrides verbatim
        catches any future plumbing-level regression that silently
        collapses the values back to the global default."""
        ev = GraphEvaluator(
            eval_num_samples=5,
            sigma=1.0,
            degree_sigma=0.5,
            clustering_sigma=0.1,
            spectral_sigma=0.7,
        )
        assert ev.degree_sigma == 0.5
        assert ev.clustering_sigma == 0.1
        assert ev.spectral_sigma == 0.7

    def test_is_sbm_graph_has_no_lazy_graph_tool_import(self) -> None:
        """SBM helper should resolve graph-tool at module import time.

        Regression rationale
        --------------------
        Importing ``graph_tool`` inside ``_is_sbm_graph`` delays binary/runtime
        failures until validation-time worker threads. We want those issues to
        surface when the module imports, not deep inside metric execution.
        """
        tree = ast.parse(Path(graph_evaluator_module.__file__).read_text())
        target = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_is_sbm_graph"
        )

        lazy_imports = [
            node
            for node in ast.walk(target)
            if isinstance(node, ast.Import | ast.ImportFrom)
        ]

        assert lazy_imports == []


# ---------------------------------------------------------------------------
# Evaluate -- core MMD metrics (always available)
# ---------------------------------------------------------------------------
class TestEvaluateCore:
    """Verify evaluate() returns EvaluationResults with all fields."""

    def test_evaluate_returns_evaluation_results(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert isinstance(result, EvaluationResults)

    def test_evaluate_has_mmd_fields(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert isinstance(result.degree_mmd, float)
        assert isinstance(result.clustering_mmd, float)
        assert isinstance(result.spectral_mmd, float)

    def test_evaluate_has_planarity(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert isinstance(result.planarity_accuracy, float)
        assert 0.0 <= result.planarity_accuracy <= 1.0

    def test_evaluate_has_uniqueness(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert isinstance(result.uniqueness, float)
        assert 0.0 <= result.uniqueness <= 1.0

    def test_evaluate_has_novelty(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert isinstance(result.novelty, float)
        assert 0.0 <= result.novelty <= 1.0

    def test_evaluate_returns_none_when_too_few_refs(
        self,
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        """evaluate() requires at least 2 reference graphs for MMD."""
        one_graph = [nx.erdos_renyi_graph(10, 0.3, seed=0)]
        result = GraphEvaluator(eval_num_samples=10).evaluate(
            refs=one_graph, generated=different_er_graphs
        )
        assert result is None

    def test_evaluate_returns_none_when_no_refs(
        self,
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        result = GraphEvaluator(eval_num_samples=10).evaluate(
            refs=[], generated=different_er_graphs
        )
        assert result is None

    def test_novelty_is_none_without_train_graphs(
        self, er_graphs: list[nx.Graph[Any]], different_er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """When train_graphs are not provided, novelty should be None."""
        ev = GraphEvaluator(eval_num_samples=10)
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert result.novelty is None

    def test_mmd_near_zero_for_identical_distributions(
        self, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """MMD between a distribution and itself should be near zero."""
        ev = GraphEvaluator(eval_num_samples=10)
        result = ev.evaluate(refs=er_graphs, generated=er_graphs)
        assert result is not None
        assert result.degree_mmd == pytest.approx(0.0, abs=0.01)

    def test_mmd_positive_for_different_distributions(
        self, er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """MMD between different graph families should be clearly positive."""
        ev = GraphEvaluator(eval_num_samples=10)
        # Regular graphs (very different degree distribution from ER)
        regular = [nx.random_regular_graph(3, 12, seed=i) for i in range(10)]
        result = ev.evaluate(refs=er_graphs, generated=regular)
        assert result is not None
        assert result.degree_mmd > 0.01


# ---------------------------------------------------------------------------
# Evaluate -- orbit and SBM (require external tools)
# ---------------------------------------------------------------------------
class TestEvaluateOrbitSBM:
    """Tests for metrics that need graph-tool or orca."""

    @pytest.mark.skipif(
        not _orca_available, reason="orca binary not compiled/available"
    )
    def test_orbit_mmd_is_float_when_orca_available(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert isinstance(result.orbit_mmd, float)

    @pytest.mark.skipif(not _graph_tool_available, reason="graph-tool not installed")
    def test_sbm_accuracy_is_float_when_graph_tool_available(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert isinstance(result.sbm_accuracy, float)

    def test_orbit_mmd_none_when_orca_unavailable(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        """When orca is not available, orbit_mmd should be None."""
        if _orca_available:
            pytest.skip("orca is available; this test checks the unavailable case")
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert result.orbit_mmd is None

    def test_sbm_accuracy_none_when_graph_tool_unavailable(
        self,
        er_graphs: list[nx.Graph[Any]],
        different_er_graphs: list[nx.Graph[Any]],
    ) -> None:
        """When graph-tool is not available, sbm_accuracy should be None."""
        if _graph_tool_available:
            pytest.skip(
                "graph-tool is available; this test checks the unavailable case"
            )
        ev = GraphEvaluator(eval_num_samples=10, train_graphs=er_graphs[:3])
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None
        assert result.sbm_accuracy is None


# ---------------------------------------------------------------------------
# EvaluationResults.to_dict
# ---------------------------------------------------------------------------
class TestEvaluationResultsToDict:
    """Verify to_dict() returns a flat dict suitable for logging."""

    def test_to_dict_keys(self) -> None:
        res = EvaluationResults(
            degree_mmd=0.1,
            clustering_mmd=0.2,
            spectral_mmd=0.3,
            orbit_mmd=0.4,
            sbm_accuracy=0.5,
            planarity_accuracy=0.6,
            uniqueness=0.7,
            novelty=0.8,
        )
        d = res.to_dict()
        expected_keys = {
            "degree_mmd",
            "clustering_mmd",
            "spectral_mmd",
            "orbit_mmd",
            "sbm_accuracy",
            "planarity_accuracy",
            "uniqueness",
            "novelty",
            # Stage 3 telemetry: dataset-agnostic block-structure metrics.
            "modularity_q",
            "spectral_gap_l2",
            "empirical_p_in",
            "empirical_p_out",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self) -> None:
        res = EvaluationResults(
            degree_mmd=0.1,
            clustering_mmd=0.2,
            spectral_mmd=0.3,
            orbit_mmd=None,
            sbm_accuracy=None,
            planarity_accuracy=0.6,
            uniqueness=0.7,
            novelty=None,
        )
        d = res.to_dict()
        assert d["degree_mmd"] == 0.1
        assert d["orbit_mmd"] is None
        assert d["novelty"] is None

    def test_to_dict_all_values_are_float_or_none(self) -> None:
        res = EvaluationResults(
            degree_mmd=0.1,
            clustering_mmd=0.2,
            spectral_mmd=0.3,
            orbit_mmd=None,
            sbm_accuracy=None,
            planarity_accuracy=0.6,
            uniqueness=0.7,
            novelty=0.8,
        )
        d = res.to_dict()
        for v in d.values():
            assert v is None or isinstance(v, float)


# ---------------------------------------------------------------------------
# Stateless evaluate — truncation and multi-call behaviour
# ---------------------------------------------------------------------------
class TestStatelessEvaluate:
    """Verify stateless evaluate(): internal truncation and multi-call correctness."""

    def test_truncation_to_eval_num_samples(
        self, er_graphs: list[nx.Graph[Any]], different_er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """Passing more graphs than eval_num_samples should still yield results.

        The evaluator must truncate internally; callers should not need to
        pre-filter their graph lists.
        """
        # eval_num_samples=5 but we pass 10 refs and 10 generated
        ev = GraphEvaluator(eval_num_samples=5)
        result = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result is not None

    def test_multiple_calls_are_independent(
        self, er_graphs: list[nx.Graph[Any]], different_er_graphs: list[nx.Graph[Any]]
    ) -> None:
        """Calling evaluate() twice with different inputs should yield independent results.

        The evaluator carries no state between calls; the second call uses
        structurally different graph families so the MMD must change.
        """
        ev = GraphEvaluator(eval_num_samples=10)
        result1 = ev.evaluate(refs=er_graphs, generated=different_er_graphs)
        assert result1 is not None

        # Regular graphs as the second reference set — very different degree
        # distribution from either ER family, so degree_mmd must change.
        regular = [nx.random_regular_graph(3, 12, seed=i + 200) for i in range(10)]
        result2 = ev.evaluate(refs=regular, generated=different_er_graphs)
        assert result2 is not None

        assert (
            result1.degree_mmd != result2.degree_mmd
            or result1.clustering_mmd != result2.clustering_mmd
        )
