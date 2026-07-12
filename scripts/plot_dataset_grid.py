"""Plot up to 64 graphs from a PyG TU dataset on an 8x8 grid.

Two views per dataset, written to a single PDF (one page per view):
  * adjacency-matrix spy plots
  * networkx ``spring_layout`` drawings

Reuses ``PyGDatasetWrapper`` — the same dataset adapter that
``GraphDataModule`` uses under the PyG path — so loading and ordering
match the rest of the project. Pads are stripped per graph using
``num_nodes`` so layouts and spy plots reflect the actual graph.

Run with::

    uv run scripts/plot_dataset_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

GRID = 8
MAX_GRAPHS = GRID * GRID
OUT_DIR = Path("docs/dataset-previews")
DATASETS = ("enzymes", "proteins")


def _trim(adj_padded: np.ndarray, n: int) -> np.ndarray:
    return adj_padded[:n, :n]


def _plot_matrix_grid(adjs: list[np.ndarray], title: str) -> plt.Figure:
    fig, axes = plt.subplots(GRID, GRID, figsize=(GRID * 1.4, GRID * 1.4))
    for ax, adj in zip(axes.flat, adjs, strict=False):
        ax.imshow(adj, cmap="Greys", interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"n={adj.shape[0]}", fontsize=6, pad=1)
    for ax in list(axes.flat)[len(adjs) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _plot_networkx_grid(adjs: list[np.ndarray], title: str) -> plt.Figure:
    fig, axes = plt.subplots(GRID, GRID, figsize=(GRID * 1.4, GRID * 1.4))
    for ax, adj in zip(axes.flat, adjs, strict=False):
        G = nx.from_numpy_array(adj)
        pos = nx.spring_layout(G, seed=0, iterations=50)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.4, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=8, node_color="#1f77b4")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"n={adj.shape[0]}", fontsize=6, pad=1)
        ax.set_aspect("equal")
    for ax in list(axes.flat)[len(adjs) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_dataset(dataset_name: str, out_dir: Path) -> Path:
    wrapper = PyGDatasetWrapper(dataset_name=dataset_name, max_graphs=MAX_GRAPHS)
    n_taken = min(MAX_GRAPHS, len(wrapper))
    adjs = [
        _trim(wrapper.adjacencies[i], int(wrapper.num_nodes[i])) for i in range(n_taken)
    ]

    out_path = out_dir / f"{dataset_name}_grid.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_path) as pdf:
        fig_mat = _plot_matrix_grid(
            adjs, f"{dataset_name.upper()} — first {n_taken} graphs (adjacency)"
        )
        pdf.savefig(fig_mat)
        plt.close(fig_mat)

        fig_nx = _plot_networkx_grid(
            adjs,
            f"{dataset_name.upper()} — first {n_taken} graphs (spring layout)",
        )
        pdf.savefig(fig_nx)
        plt.close(fig_nx)
    return out_path


def main() -> None:
    for name in DATASETS:
        path = plot_dataset(name, OUT_DIR)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
