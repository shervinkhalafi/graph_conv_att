# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.24",
# ]
# ///
"""
Compare spectral properties across all datasets in the eigenstructure study.

Generates bar charts showing spectral gap, algebraic connectivity, entropy,
coherence, and effective rank for each dataset type.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["sbm", "er", "tree", "regular", "enzymes"]
RESULTS_DIR = Path("eigenstructure_results")
OUTPUT_DIR = Path("docs/eigenstructure_study")


def load_analysis(dataset: str) -> dict:
    """Load analysis.json for a dataset."""
    path = RESULTS_DIR / dataset / "analysis" / "analysis.json"
    with open(path) as f:
        return json.load(f)


def create_spectral_comparison():
    """Create multi-panel figure comparing spectral properties."""
    data = {ds: load_analysis(ds) for ds in DATASETS}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        "Spectral Properties Across Graph Types", fontsize=14, fontweight="bold"
    )

    x = np.arange(len(DATASETS))
    width = 0.6
    colors = plt.cm.Set2(np.linspace(0, 1, len(DATASETS)))

    # 1. Spectral Gap
    ax = axes[0, 0]
    means = [data[ds]["spectral_gap_mean"] for ds in DATASETS]
    stds = [data[ds]["spectral_gap_std"] for ds in DATASETS]
    ax.bar(
        x,
        means,
        width,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Spectral Gap (λ₁ - λ₂)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_title("Spectral Gap")
    ax.grid(axis="y", alpha=0.3)

    # 2. Algebraic Connectivity
    ax = axes[0, 1]
    means = [data[ds]["algebraic_connectivity_mean"] for ds in DATASETS]
    stds = [data[ds]["algebraic_connectivity_std"] for ds in DATASETS]
    ax.bar(
        x,
        means,
        width,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Algebraic Connectivity (λ₂)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_title("Algebraic Connectivity")
    ax.grid(axis="y", alpha=0.3)

    # 3. Eigenvalue Entropy (Adjacency)
    ax = axes[0, 2]
    vals = [data[ds]["eigenvalue_entropy_adj"] for ds in DATASETS]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Entropy (bits)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_title("Eigenvalue Entropy (Adjacency)")
    ax.grid(axis="y", alpha=0.3)

    # 4. Eigenvector Coherence
    ax = axes[1, 0]
    means = [data[ds]["coherence_mean"] for ds in DATASETS]
    stds = [data[ds]["coherence_std"] for ds in DATASETS]
    ax.bar(
        x,
        means,
        width,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Coherence")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_title("Eigenvector Coherence")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # 5. Effective Rank (Adjacency)
    ax = axes[1, 1]
    vals = [data[ds]["effective_rank_adj_mean"] for ds in DATASETS]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Effective Rank")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_title("Effective Rank (Adjacency)")
    ax.grid(axis="y", alpha=0.3)

    # 6. Effective Rank (Laplacian)
    ax = axes[1, 2]
    vals = [data[ds]["effective_rank_lap_mean"] for ds in DATASETS]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Effective Rank")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right")
    ax.set_title("Effective Rank (Laplacian)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "spectral_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_spectral_comparison()
