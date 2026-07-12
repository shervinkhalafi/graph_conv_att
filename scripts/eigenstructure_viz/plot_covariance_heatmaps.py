# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.24",
# ]
# ///
"""
Visualize eigenvalue covariance matrices and their evolution under noise.

Creates heatmaps showing the original covariance structure and how it changes
at different noise levels for each dataset.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["sbm", "er", "tree", "regular", "enzymes"]
RESULTS_DIR = Path("eigenstructure_results")
OUTPUT_DIR = Path("docs/eigenstructure_study")


def load_covariance_evolution(dataset: str, noise_type: str) -> dict:
    """Load covariance_evolution.json for a dataset and noise type."""
    path = (
        RESULTS_DIR / dataset / f"covariance_{noise_type}" / "covariance_evolution.json"
    )
    with open(path) as f:
        return json.load(f)


def create_covariance_heatmaps():
    """Create heatmap grid showing covariance evolution across datasets."""
    # Focus on Gaussian noise for visualization
    noise_type = "gaussian"

    fig, axes = plt.subplots(len(DATASETS), 5, figsize=(16, 14))
    fig.suptitle(
        "Eigenvalue Covariance Evolution Under Gaussian Noise",
        fontsize=14,
        fontweight="bold",
    )

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    col_titles = ["Original", "ε=0.01", "ε=0.05", "ε=0.1", "ε=0.2"]

    for row_idx, ds in enumerate(DATASETS):
        try:
            data = load_covariance_evolution(ds, noise_type)
        except FileNotFoundError:
            print(f"Warning: Missing covariance data for {ds}/{noise_type}")
            continue

        # Get original covariance
        orig_cov = np.array(data["original"]["covariance_matrix"])
        n = min(10, orig_cov.shape[0])  # Limit to first 10 eigenvalues
        orig_cov = orig_cov[:n, :n]

        # Normalize for visualization
        max_abs = max(np.abs(orig_cov).max(), 1e-10)

        # Plot original
        ax = axes[row_idx, 0]
        im = ax.imshow(
            orig_cov, cmap="RdBu_r", aspect="equal", vmin=-max_abs, vmax=max_abs
        )
        ax.set_ylabel(ds, fontsize=11, fontweight="bold")
        if row_idx == 0:
            ax.set_title(col_titles[0])
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot noised versions
        for col_idx, noise_level in enumerate(noise_levels[1:], start=1):
            ax = axes[row_idx, col_idx]

            # Find the matching noise entry
            noise_entry = None
            for entry in data["per_noise_level"]:
                if abs(entry["noise_level"] - noise_level) < 1e-6:
                    noise_entry = entry
                    break

            if noise_entry is None:
                ax.text(
                    0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx])
                continue

            noised_cov = np.array(noise_entry["covariance"]["covariance_matrix"])[
                :n, :n
            ]

            # Use same scale as original for comparison
            im = ax.imshow(
                noised_cov, cmap="RdBu_r", aspect="equal", vmin=-max_abs, vmax=max_abs
            )
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Covariance")

    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "covariance_heatmaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def create_frobenius_evolution_plot():
    """Create line plot showing Frobenius norm evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    noise_levels = [0.01, 0.05, 0.1, 0.2]
    colors = plt.cm.Set2(np.linspace(0, 1, len(DATASETS)))

    for noise_idx, noise_type in enumerate(["gaussian", "digress"]):
        ax = axes[noise_idx]
        ax.set_title(f"Covariance Change Under {noise_type.title()} Noise")

        for ds_idx, ds in enumerate(DATASETS):
            try:
                data = load_covariance_evolution(ds, noise_type)
            except FileNotFoundError:
                continue

            # Extract Frobenius norm deltas
            frob_deltas = []
            for noise_level in noise_levels:
                for entry in data["per_noise_level"]:
                    if abs(entry["noise_level"] - noise_level) < 1e-6:
                        orig_cov = np.array(data["original"]["covariance_matrix"])
                        noised_cov = np.array(entry["covariance"]["covariance_matrix"])
                        delta = np.linalg.norm(noised_cov - orig_cov, "fro")
                        frob_deltas.append(delta)
                        break
                else:
                    frob_deltas.append(np.nan)

            ax.plot(
                noise_levels,
                frob_deltas,
                marker="o",
                label=ds,
                color=colors[ds_idx],
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Noise Level (ε)")
        ax.set_ylabel("Frobenius Norm of ΔΣ")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "covariance_frobenius.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_covariance_heatmaps()
    create_frobenius_evolution_plot()
