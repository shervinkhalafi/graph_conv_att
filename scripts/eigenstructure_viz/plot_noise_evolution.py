# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.24",
# ]
# ///
"""
Visualize how eigenstructure metrics evolve with increasing noise.

Generates line plots showing eigenvalue drift, subspace distance, and Procrustes
residuals as functions of noise level, for both Gaussian and DiGress noise types.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

DATASETS = ["sbm", "er", "tree", "regular", "enzymes"]
NOISE_TYPES = ["gaussian", "digress"]
RESULTS_DIR = Path("eigenstructure_results")
OUTPUT_DIR = Path("docs/eigenstructure_study")

DATASET_COLORS = {
    "sbm": "#1f77b4",
    "er": "#ff7f0e",
    "tree": "#2ca02c",
    "regular": "#d62728",
    "enzymes": "#9467bd",
}

DATASET_MARKERS = {
    "sbm": "o",
    "er": "s",
    "tree": "^",
    "regular": "D",
    "enzymes": "v",
}


def load_comparison(dataset: str, noise_type: str) -> list[dict]:
    """Load comparison.json for a dataset and noise type."""
    path = RESULTS_DIR / dataset / f"comparison_{noise_type}" / "comparison.json"
    with open(path) as f:
        return json.load(f)


def create_noise_evolution_plots():
    """Create plots for both noise types."""
    for noise_type in NOISE_TYPES:
        create_single_noise_plot(noise_type)


def create_single_noise_plot(noise_type: str):
    """Create multi-panel plot for a single noise type."""
    # Load all data
    all_data = {}
    for ds in DATASETS:
        try:
            all_data[ds] = load_comparison(ds, noise_type)
        except FileNotFoundError:
            print(f"Warning: Missing comparison data for {ds}/{noise_type}")
            continue

    if not all_data:
        print(f"No data found for {noise_type}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    title_suffix = (
        "Gaussian Noise" if noise_type == "gaussian" else "DiGress Edge Flips"
    )
    fig.suptitle(
        f"Eigenstructure Degradation Under {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Extract noise levels from first dataset
    first_ds = list(all_data.keys())[0]
    noise_levels = [entry["noise_level"] for entry in all_data[first_ds]]

    # 1. Eigenvalue Drift (Adjacency)
    ax = axes[0, 0]
    for ds, data in all_data.items():
        means = [entry["eigenvalue_drift_adj_mean"] for entry in data]
        stds = [entry["eigenvalue_drift_adj_std"] for entry in data]
        ax.errorbar(
            noise_levels,
            means,
            yerr=stds,
            label=ds,
            color=DATASET_COLORS[ds],
            marker=DATASET_MARKERS[ds],
            capsize=3,
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel("Eigenvalue Drift")
    ax.set_title("Eigenvalue Drift (Adjacency)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # 2. Eigenvalue Drift (Laplacian)
    ax = axes[0, 1]
    for ds, data in all_data.items():
        means = [entry["eigenvalue_drift_lap_mean"] for entry in data]
        stds = [entry["eigenvalue_drift_lap_std"] for entry in data]
        ax.errorbar(
            noise_levels,
            means,
            yerr=stds,
            label=ds,
            color=DATASET_COLORS[ds],
            marker=DATASET_MARKERS[ds],
            capsize=3,
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel("Eigenvalue Drift")
    ax.set_title("Eigenvalue Drift (Laplacian)")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # 3. Subspace Distance
    ax = axes[0, 2]
    for ds, data in all_data.items():
        means = [entry["subspace_distance_mean"] for entry in data]
        stds = [entry["subspace_distance_std"] for entry in data]
        ax.errorbar(
            noise_levels,
            means,
            yerr=stds,
            label=ds,
            color=DATASET_COLORS[ds],
            marker=DATASET_MARKERS[ds],
            capsize=3,
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel("Subspace Distance")
    ax.set_title("Subspace Distance (k=10)")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # 4. Procrustes Residual k=1
    ax = axes[1, 0]
    for ds, data in all_data.items():
        means = [entry["procrustes_residual_k1_mean"] for entry in data]
        stds = [entry["procrustes_residual_k1_std"] for entry in data]
        ax.errorbar(
            noise_levels,
            means,
            yerr=stds,
            label=ds,
            color=DATASET_COLORS[ds],
            marker=DATASET_MARKERS[ds],
            capsize=3,
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel("Procrustes Residual")
    ax.set_title("Procrustes Residual (k=1)")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # 5. Procrustes Residual k=4
    ax = axes[1, 1]
    for ds, data in all_data.items():
        means = [entry["procrustes_residual_k4_mean"] for entry in data]
        stds = [entry["procrustes_residual_k4_std"] for entry in data]
        ax.errorbar(
            noise_levels,
            means,
            yerr=stds,
            label=ds,
            color=DATASET_COLORS[ds],
            marker=DATASET_MARKERS[ds],
            capsize=3,
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel("Procrustes Residual")
    ax.set_title("Procrustes Residual (k=4)")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # 6. Procrustes Residual k=8
    ax = axes[1, 2]
    for ds, data in all_data.items():
        means = [entry["procrustes_residual_k8_mean"] for entry in data]
        stds = [entry["procrustes_residual_k8_std"] for entry in data]
        ax.errorbar(
            noise_levels,
            means,
            yerr=stds,
            label=ds,
            color=DATASET_COLORS[ds],
            marker=DATASET_MARKERS[ds],
            capsize=3,
            linewidth=1.5,
            markersize=6,
        )
    ax.set_xlabel("Noise Level (ε)")
    ax.set_ylabel("Procrustes Residual")
    ax.set_title("Procrustes Residual (k=8)")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"noise_evolution_{noise_type}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_noise_evolution_plots()
