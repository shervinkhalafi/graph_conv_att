# pyright: reportUnknownMemberType=false
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false

"""Validation-figure helpers for graph generation experiments.

The default generative validation path logs scalar metrics and a compact set
of figures built from the same reference/generated graph sets. This module
keeps the plotting logic separate from the Lightning training loop so the
module remains responsible for orchestration, not figure construction.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_GRAPH_TAG = "gen-val/graph_samples"
_ADJACENCY_TAG = "gen-val/adjacency_samples"
_LAYOUT_SEED = 0


def _validate_num_samples(num_samples: int) -> None:
    """Require a positive even visualization sample count."""
    if num_samples <= 0 or num_samples % 2 != 0:
        raise ValueError("Visualization num_samples must be a positive even integer.")


def _plot_graph_row(
    axes: np.ndarray,
    graphs: list[nx.Graph[Any]],
    *,
    row_label: str,
) -> None:
    """Render one row of node-link graph plots."""
    for axis, graph in zip(axes, graphs, strict=True):
        axis.set_axis_off()
        axis.set_title(
            f"{row_label} | n={graph.number_of_nodes()} e={graph.number_of_edges()}"
        )
        if graph.number_of_nodes() == 0:
            continue
        pos = nx.spring_layout(graph, seed=_LAYOUT_SEED)
        nx.draw_networkx(
            graph,
            pos=pos,
            ax=axis,
            with_labels=False,
            node_size=120,
            width=1.0,
        )


def _plot_adjacency_row(
    axes: np.ndarray,
    graphs: list[nx.Graph[Any]],
    *,
    row_label: str,
) -> None:
    """Render one row of adjacency-matrix heatmaps."""
    for axis, graph in zip(axes, graphs, strict=True):
        axis.set_title(
            f"{row_label} | n={graph.number_of_nodes()} e={graph.number_of_edges()}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        if graph.number_of_nodes() == 0:
            axis.imshow(np.zeros((1, 1)), cmap="Greys", vmin=0.0, vmax=1.0)
            continue
        nodes = sorted(graph.nodes())
        adjacency = nx.to_numpy_array(
            graph,
            nodelist=nodes,
            dtype=np.dtype(np.float64),
        )
        axis.imshow(adjacency, cmap="Greys", vmin=0.0, vmax=1.0)


def _make_axes_grid(
    cols: int,
    *,
    width_per_col: float,
    height: float,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Create a 2-row subplot grid with stable shape."""
    fig, axes = plt.subplots(2, cols, figsize=(width_per_col * cols, height))
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(2, 1)
    return fig, axes_array


def build_validation_visualizations(
    *,
    refs: list[nx.Graph[Any]],
    generated: list[nx.Graph[Any]],
    num_samples: int,
) -> dict[str, matplotlib.figure.Figure]:
    """Build the default validation figure set.

    Parameters
    ----------
    refs
        Reference graphs used for validation metrics.
    generated
        Generated graphs sampled for validation metrics.
    num_samples
        Total number of plotted graphs across both distributions.
        Must be a positive even integer.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Figures keyed by their final logger tags.
    """
    _validate_num_samples(num_samples)

    per_distribution = min(num_samples // 2, len(refs), len(generated))
    refs_to_plot = refs[:per_distribution]
    generated_to_plot = generated[:per_distribution]

    graph_fig, graph_axes = _make_axes_grid(
        per_distribution, width_per_col=3.0, height=6.0
    )
    _plot_graph_row(graph_axes[0], refs_to_plot, row_label="Reference")
    _plot_graph_row(graph_axes[1], generated_to_plot, row_label="Generated")
    graph_fig.tight_layout()

    adjacency_fig, adjacency_axes = _make_axes_grid(
        per_distribution, width_per_col=2.5, height=5.0
    )
    _plot_adjacency_row(adjacency_axes[0], refs_to_plot, row_label="Reference")
    _plot_adjacency_row(adjacency_axes[1], generated_to_plot, row_label="Generated")
    adjacency_fig.tight_layout()

    return {
        _GRAPH_TAG: graph_fig,
        _ADJACENCY_TAG: adjacency_fig,
    }
