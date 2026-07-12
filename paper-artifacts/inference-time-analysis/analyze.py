# /// script
# dependencies = [
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "matplotlib>=3.8",
# ]
# ///
"""Render LaTeX tables + figures from ``data/perf.csv``.

Reads the committed CSV (no W&B access needed) and writes, under ``tables/``
and ``figures/``:

* ``tables/inference_time_main.tex`` — the headline ``booktabs`` table:
  training-step and inference-cycle wall time, absolute and relative to the
  vanilla-DiGress baseline, per dataset and variant.
* ``tables/inference_time_compact.tex`` — a tighter ratio-only variant for a
  narrow column, with parameter counts.
* ``figures/train_vs_val_ratio.{pdf,png}`` — two-panel grouped bars: wall time
  relative to vanilla DiGress, training panel vs inference panel.
* ``figures/inference_absolute.{pdf,png}`` — absolute inference-cycle seconds
  per variant, grouped by dataset.

Run it from anywhere::

    uv run analyze.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "perf.csv"
FIG_DIR = HERE / "figures"
TAB_DIR = HERE / "tables"

VARIANT_ORDER = ["vignac", "pearl", "pearl-spectral", "pearl-gnnconv-norm"]
VARIANT_LABEL_FIG = {
    "vignac": "DiGress\n(eigh)",
    "pearl": "PEARL",
    "pearl-spectral": "PEARL +\nspectral Q/K/V",
    "pearl-gnnconv-norm": "PEARL +\nGNN Q/K/V",
}
VARIANT_LABEL_TEX = {
    "vignac": "DiGress (eigh)",
    "pearl": "PEARL",
    "pearl-spectral": "PEARL + spectral Q/K/V",
    "pearl-gnnconv-norm": "PEARL + GNN Q/K/V",
}
DATASET_ORDER = ["sbm", "enzymes"]
DATASET_LABEL = {
    "sbm": r"SBM ($n_{\max}{=}200$)",
    "enzymes": r"ENZYMES ($n_{\max}{=}126$)",
}
DATASET_COLOR = {"sbm": "#0072B2", "enzymes": "#D55E00"}  # Wong, colourblind-safe

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _pct(ratio: float) -> float:
    """Signed % change vs baseline; negative = faster than vanilla."""
    return (ratio - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def render_ratio_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    panels = [
        ("train_ratio_vs_vignac", "Training step", axes[0]),
        ("val_ratio_vs_vignac", "Inference cycle (sampling)", axes[1]),
    ]
    variants = [v for v in VARIANT_ORDER if v != "vignac"]
    x = np.arange(len(variants))
    width = 0.38
    for col, title, ax in panels:
        for i, ds in enumerate(DATASET_ORDER):
            sub = df[df.dataset == ds].set_index("variant")
            ratios = [sub.loc[v, col] for v in variants]
            bars = ax.bar(
                x + (i - 0.5) * width,
                ratios,
                width,
                label=DATASET_LABEL[ds],
                color=DATASET_COLOR[ds],
                edgecolor="black",
                linewidth=0.4,
            )
            for rect, r in zip(bars, ratios, strict=True):
                ax.annotate(
                    f"{_pct(r):+.0f}%",
                    (rect.get_x() + rect.get_width() / 2, r),
                    ha="center",
                    va="bottom" if r >= 1 else "top",
                    fontsize=6.5,
                    xytext=(0, 2 if r >= 1 else -2),
                    textcoords="offset points",
                )
        ax.axhline(1.0, color="black", lw=0.9, ls="--", zorder=0)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABEL_FIG[v] for v in variants])
        ax.set_ylim(0.7, 1.85)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("wall time / vanilla DiGress")
    axes[1].legend(loc="upper right", frameon=False)
    axes[0].text(
        0.02,
        0.96,
        "below 1.0 = faster than vanilla",
        transform=axes[0].transAxes,
        fontsize=6.5,
        va="top",
        style="italic",
        color="0.3",
    )
    fig.suptitle(
        "Per-step wall time relative to vanilla DiGress (lower = better)",
        fontsize=10,
        y=1.02,
    )
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"train_vs_val_ratio.{ext}")
    plt.close(fig)
    print(f"figure  -> {FIG_DIR}/train_vs_val_ratio.{{pdf,png}}")


def render_absolute_figure(df: pd.DataFrame) -> None:
    """Absolute inference-cycle seconds per variant, grouped by dataset (log y)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(VARIANT_ORDER))
    width = 0.38
    for i, ds in enumerate(DATASET_ORDER):
        sub = df[df.dataset == ds].set_index("variant")
        vals = [sub.loc[v, "val_per_cycle_s"] for v in VARIANT_ORDER]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=DATASET_LABEL[ds],
            color=DATASET_COLOR[ds],
            edgecolor="black",
            linewidth=0.4,
        )
        for rect, v in zip(bars, vals, strict=True):
            ax.annotate(
                f"{v:.0f}s",
                (rect.get_x() + rect.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=6.5,
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.set_yscale("log")
    ax.set_ylim(top=3500)  # headroom above the tallest ENZYMES bar (~1890 s)
    ax.set_ylabel("inference cycle wall time (s, log)")
    ax.set_title("Inference (sampling) cost per validation cycle")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL_FIG[v] for v in VARIANT_ORDER])
    # Legend outside the axes (right) so it never collides with the tall bars.
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(axis="y", lw=0.3, alpha=0.5, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"inference_absolute.{ext}")
    plt.close(fig)
    print(f"figure  -> {FIG_DIR}/inference_absolute.{{pdf,png}}")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def render_main_table(df: pd.DataFrame) -> None:
    lines = [
        r"% Auto-generated by analyze.py from data/perf.csv -- do not edit by hand.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Training-step and inference-cycle wall time of the"
        r" feature/projection variants relative to the vanilla DiGress baseline"
        r" (eigh-based \texttt{ExtraFeatures(all)}). Training-step time is a robust"
        r" median over the full run history (forward + backward + optimizer)."
        r" Inference-cycle time is wall-clock-derived"
        r" (run time $-$ median train step $\times$ step count, per"
        r" validation cycle) and bundles diffusion sampling, MMD scoring and I/O;"
        r" read it as an upper bound on pure model inference. Ratios $<1$"
        r" (negative \%) are faster than vanilla. Single seed; A100; runs preempted"
        r" at ${\sim}17.7$\,h. \texttt{compile\_model=false} on all runs.}",
        r"  \label{tab:inference-time}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{Training step (s)} & \multicolumn{2}{c}{Inference cycle (s)} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"    Dataset & Variant & abs & vs base & abs & vs base \\",
        r"    \midrule",
    ]
    for ds in DATASET_ORDER:
        sub = df[df.dataset == ds].set_index("variant")
        lines.append(f"    \\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}")
        for v in VARIANT_ORDER:
            row = sub.loc[v]
            if v == "vignac":
                tr_rel = va_rel = "---"
            else:
                tr_rel = f"{_pct(row['train_ratio_vs_vignac']):+.0f}\\%"
                va_rel = f"{_pct(row['val_ratio_vs_vignac']):+.0f}\\%"
            lines.append(
                f"      & {VARIANT_LABEL_TEX[v]} & {row['train_step_med']:.3f} & {tr_rel}"
                f" & {row['val_per_cycle_s']:.0f} & {va_rel} \\\\"
            )
        lines.append(r"    \midrule" if ds != DATASET_ORDER[-1] else r"    \bottomrule")
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    out = TAB_DIR / "inference_time_main.tex"
    out.write_text("\n".join(lines))
    print(f"table   -> {out}")


def render_compact_table(df: pd.DataFrame) -> None:
    """Ratio-only, with params -- for a narrow column."""
    lines = [
        r"% Auto-generated by analyze.py from data/perf.csv -- do not edit by hand.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Inference-cycle wall time relative to vanilla DiGress"
        r" (negative \% = faster). Inference-cycle time is wall-clock-derived per"
        r" validation cycle (sampling + scoring + I/O). Params are total model"
        r" parameters. Single seed; A100.}",
        r"  \label{tab:inference-time-compact}",
        r"  \begin{tabular}{llrr}",
        r"    \toprule",
        r"    Dataset & Variant & Params & Inference vs base \\",
        r"    \midrule",
    ]
    for ds in DATASET_ORDER:
        sub = df[df.dataset == ds].set_index("variant")
        lines.append(f"    \\multirow{{4}}{{*}}{{{DATASET_LABEL[ds]}}}")
        for v in VARIANT_ORDER:
            row = sub.loc[v]
            params = f"{row['total_parameters'] / 1e6:.1f}M"
            rel = (
                "---"
                if v == "vignac"
                else f"{_pct(row['val_ratio_vs_vignac']):+.0f}\\%"
            )
            lines.append(f"      & {VARIANT_LABEL_TEX[v]} & {params} & {rel} \\\\")
        lines.append(r"    \midrule" if ds != DATASET_ORDER[-1] else r"    \bottomrule")
    lines += [r"  \end{tabular}", r"\end{table}", ""]
    out = TAB_DIR / "inference_time_compact.tex"
    out.write_text("\n".join(lines))
    print(f"table   -> {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    render_ratio_figure(df)
    render_absolute_figure(df)
    render_main_table(df)
    render_compact_table(df)
    print("\n% headline (negative = faster than vanilla DiGress):")
    for ds in DATASET_ORDER:
        sub = df[df.dataset == ds].set_index("variant")
        print(f"  {ds}:")
        for v in VARIANT_ORDER:
            if v == "vignac":
                continue
            tr = _pct(sub.loc[v, "train_ratio_vs_vignac"])
            va = _pct(sub.loc[v, "val_ratio_vs_vignac"])
            print(f"    {v:<20} train {tr:+5.0f}%   inference {va:+5.0f}%")


if __name__ == "__main__":
    main()
