"""Tests for validation-figure construction from reference/generated graphs.

Test Rationale
--------------
These tests lock down the default visualization payload used during
generative validation. The helper should build both supported figure types
from the same graph sets already used for metric evaluation, and it should
validate configuration loudly rather than guessing.

Key invariants
--------------
- The helper returns both default figure tags.
- The helper truncates to the available number of reference/generated pairs.
- Invalid sample counts fail fast.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pytest

from tmgg.evaluation.visualization import build_validation_visualizations


def _make_graphs() -> tuple[list[nx.Graph[Any]], list[nx.Graph[Any]]]:
    """Return small reference/generated graph sets with varied structure."""
    refs: list[nx.Graph[Any]] = [
        nx.path_graph(5),
        nx.cycle_graph(5),
        nx.star_graph(4),
        nx.complete_graph(5),
    ]
    generated: list[nx.Graph[Any]] = [
        nx.wheel_graph(5),
        nx.path_graph(6),
        nx.cycle_graph(6),
        nx.complete_graph(4),
    ]
    return refs, generated


class TestBuildValidationVisualizations:
    """Validation figures should be deterministic in shape and tagging."""

    def test_returns_both_default_figure_tags(self) -> None:
        """The helper should emit both node-link and adjacency views."""
        refs, generated = _make_graphs()

        figures = build_validation_visualizations(
            refs=refs,
            generated=generated,
            num_samples=8,
        )

        assert set(figures) == {
            "gen-val/graph_samples",
            "gen-val/adjacency_samples",
        }
        assert len(figures["gen-val/graph_samples"].axes) == 8
        assert len(figures["gen-val/adjacency_samples"].axes) == 8

    def test_truncates_to_available_reference_generated_pairs(self) -> None:
        """Requested sample count should cap at available paired graphs."""
        refs, generated = _make_graphs()

        figures = build_validation_visualizations(
            refs=refs[:2],
            generated=generated[:3],
            num_samples=8,
        )

        assert len(figures["gen-val/graph_samples"].axes) == 4
        assert len(figures["gen-val/adjacency_samples"].axes) == 4

    @pytest.mark.parametrize("num_samples", [0, -2, 3, 5])
    def test_rejects_non_positive_or_odd_sample_counts(
        self,
        num_samples: int,
    ) -> None:
        """Visualization sample count must be a positive even integer."""
        refs, generated = _make_graphs()

        with pytest.raises(ValueError, match="positive even"):
            build_validation_visualizations(
                refs=refs,
                generated=generated,
                num_samples=num_samples,
            )
