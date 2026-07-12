# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

"""Multi-format artifact dump for one ``evaluate_checkpoint`` invocation.

Owns the *per-checkpoint subfolder* layout written by the eval-all
worker (and by the standalone CLI when ``--output-dir`` is supplied).
The intent is durability and post-hoc analysability: every generated
and reference graph is serialised as a JSON edge-list (re-loadable
without pickle), per-batch validation diagnostics land in CSV, and a
fixed-count viz batch lands as PNGs so a glance at the folder shows
whether a checkpoint produced anything sensible.

Why a separate module rather than inlining in ``evaluate_cli``: the
CLI proper stays focused on argument parsing + the
sample/score sequence; this file owns the (chunkier) write surface
plus the val-pass-with-per-batch-capture trick, both of which are
reused unchanged from ``eval_all_checkpoints_impl``.

The val-pass works around Lightning's requirement that ``self.log``
runs inside a ``Trainer.fit/validate`` context: we monkey-patch
``module.log`` to a no-op for the duration of the loop, call
``module.on_validation_epoch_start`` (which clears the ``_vlb_*``
accumulator lists), then iterate the val dataloader by hand and
snapshot the last appended values from each list per batch. This is
narrower than spinning up a real ``Trainer.validate`` call, which
would also re-fire ``on_validation_epoch_end`` and trigger a second
gen-val sample-and-score pipeline (~2-5 min wasted per ckpt).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import torch

logger = logging.getLogger(__name__)

#: Field order for ``val_per_batch.csv``. Stable header so cross-ckpt
#: pandas concat doesn't drift; missing fields land as empty cells.
_VAL_BATCH_FIELDS: tuple[str, ...] = (
    "batch_idx",
    "batch_size",
    "nll",
    "kl_prior",
    "kl_diffusion",
    "reconstruction",
    "log_pN",
)


@dataclass
class EvalTimings:
    """Per-stage wall-clock timings for one ``evaluate_checkpoint`` call.

    All values are seconds of wall time captured with
    :func:`time.perf_counter`. Stages that did not run (e.g.
    ``val_pass_s`` when ``--output-dir`` was not set) are ``None``
    rather than ``0`` so a downstream consumer can distinguish "not
    measured" from "instantaneous".
    """

    load_s: float | None = None
    val_pass_s: float | None = None
    sample_s: float | None = None
    eval_s: float | None = None
    total_s: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "load_s": self.load_s,
            "val_pass_s": self.val_pass_s,
            "sample_s": self.sample_s,
            "eval_s": self.eval_s,
            "total_s": self.total_s,
        }


@dataclass
class ValBatchRows:
    """Container for per-batch val-pass diagnostics.

    Each entry corresponds to one validation batch and carries the
    snapshotted scalar values from the diffusion module's ``_vlb_*``
    accumulators after that batch's ``validation_step`` returned.
    """

    rows: list[dict[str, float | int | None]] = field(default_factory=list)


def graph_to_edge_list_dict(graph: nx.Graph[Any]) -> dict[str, Any]:
    """Serialise a NetworkX graph to a JSON-friendly edge-list dict.

    The output is intentionally minimal but lossless under
    :func:`edge_list_dict_to_graph`: nodes are listed in insertion
    order, edges as ``[u, v]`` pairs, and any non-empty node/edge
    attribute dicts ride along. Pickle-free by design — the user
    flagged ``.pkl`` ckpt loads as a security tripwire.
    """
    nodes = list(graph.nodes())
    edges = [[u, v] for u, v in graph.edges()]
    node_attrs = {str(n): dict(graph.nodes[n]) for n in nodes if dict(graph.nodes[n])}
    edge_attrs = {
        f"{u},{v}": dict(graph.edges[u, v])
        for u, v in graph.edges()
        if dict(graph.edges[u, v])
    }
    return {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "nodes": nodes,
        "edges": edges,
        "node_attrs": node_attrs,
        "edge_attrs": edge_attrs,
    }


def edge_list_dict_to_graph(payload: dict[str, Any]) -> nx.Graph[Any]:
    """Reverse of :func:`graph_to_edge_list_dict`.

    Used by tests and by the analysis-dump scripts that load saved
    graphs without going through Modal. Round-trips
    ``graph_to_edge_list_dict(g)`` exactly for graph topology and
    node/edge attribute payloads.
    """
    g: nx.Graph[Any] = nx.Graph()
    g.add_nodes_from(payload.get("nodes", []))
    g.add_edges_from(tuple(e) for e in payload.get("edges", []))
    for n_str, attrs in payload.get("node_attrs", {}).items():
        # JSON keys are always strings; recover ints when applicable.
        node_key: Any = int(n_str) if n_str.isdigit() else n_str
        g.nodes[node_key].update(attrs)
    for uv_str, attrs in payload.get("edge_attrs", {}).items():
        u_str, v_str = uv_str.split(",", 1)
        u: Any = int(u_str) if u_str.isdigit() else u_str
        v: Any = int(v_str) if v_str.isdigit() else v_str
        g.edges[u, v].update(attrs)
    return g


def graphs_to_edge_list_payload(graphs: list[nx.Graph[Any]]) -> dict[str, Any]:
    """Wrap a graph list in a top-level payload dict.

    Top-level metadata (``count``, ``serialization``) makes the file
    self-describing for the analysis side without forcing every loader
    to re-derive it from ``len(payload['graphs'])``.
    """
    return {
        "count": len(graphs),
        "serialization": "edge_list_v1",
        "graphs": [graph_to_edge_list_dict(g) for g in graphs],
    }


@contextmanager
def _silenced_lightning_log(module: Any) -> Generator[None, None, None]:
    """Monkey-patch ``module.log`` to a no-op for the with-block.

    Lightning's ``LightningModule.log`` asserts a Trainer is attached.
    Calling ``validation_step`` outside ``Trainer.validate()`` would
    raise on every ``self.log(...)``. Per-instance attribute shadowing
    works because Python looks up instance attrs before class attrs.
    """
    original_log: Callable[..., Any] = module.log

    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    module.log = _noop
    try:
        yield
    finally:
        module.log = original_log


def _safe_pop_last(buf: list[Any]) -> float | None:
    """Snapshot the last appended scalar tensor from an accumulator list.

    Returns ``None`` when the list is empty (e.g. continuous diffusion
    branch ran instead of categorical and that batch's ``_vlb_*`` was
    never appended to). Coerces 0-dim torch tensors to Python floats
    so the row writes cleanly to CSV.
    """
    if not buf:
        return None
    val = buf[-1]
    if isinstance(val, torch.Tensor):
        return float(val.item())
    if isinstance(val, int | float):
        return float(val)
    return None


def run_val_pass_with_per_batch_capture(
    module: Any,
    datamodule: Any,
    *,
    max_batches: int | None = None,
) -> ValBatchRows:
    """Walk the val dataloader and capture per-batch diagnostics.

    Hits the ``categorical`` branch of ``validation_step`` (the only
    branch in the discrete repro panel) and snapshots the values that
    branch appends to ``module._vlb_*`` after each batch returns.
    Continuous-diffusion runs would land in the other branch and the
    ``_vlb_*`` lists would stay empty; the snapshot tolerates that
    and writes ``None`` for those fields.

    Parameters
    ----------
    module
        The instantiated DiffusionModule (must already be ``.eval()``
        and on the right device).
    datamodule
        The Hydra-instantiated training-time datamodule, already
        ``setup("fit")``-ed.
    max_batches
        Cap on validation batches walked. ``None`` walks the full val
        split; pass an integer (e.g. 50) when sampling is the
        bottleneck and a representative slice will do.

    Returns
    -------
    ValBatchRows
        One entry per walked batch. CSV-friendly via
        :func:`write_val_per_batch_csv`.
    """
    rows = ValBatchRows()
    # Populate the module's size distribution from the datamodule. In
    # the trainer flow Lightning's ``setup()`` does this; the
    # checkpoint-loading CLI flow skips ``setup()`` (no Trainer
    # attached), so ``validation_step``'s categorical branch crashes
    # with ``_size_distribution is None``. Mirror what the trainer
    # would have done — both ExactDensity and Categorical (a
    # subclass) need it for the log_pN term in the VLB.
    if getattr(module, "_size_distribution", None) is None:
        module._size_distribution = datamodule.get_size_distribution("train")
    val_loader = datamodule.val_dataloader()
    module.on_validation_epoch_start()
    with _silenced_lightning_log(module), torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(module.device)
            module.validation_step(batch, batch_idx)
            rows.rows.append(
                {
                    "batch_idx": batch_idx,
                    "batch_size": int(batch.node_mask.shape[0]),
                    "nll": _safe_pop_last(module._vlb_nll),  # pyright: ignore[reportPrivateUsage]
                    "kl_prior": _safe_pop_last(module._vlb_kl_prior),  # pyright: ignore[reportPrivateUsage]
                    "kl_diffusion": _safe_pop_last(module._vlb_kl_diffusion),  # pyright: ignore[reportPrivateUsage]
                    "reconstruction": _safe_pop_last(module._vlb_reconstruction),  # pyright: ignore[reportPrivateUsage]
                    "log_pN": _safe_pop_last(module._vlb_log_pn),  # pyright: ignore[reportPrivateUsage]
                }
            )
    return rows


def _render_graph_png(graph: nx.Graph[Any], path: Path, *, label: str) -> None:
    """Render one graph to a single PNG file at ``path``."""
    fig: matplotlib.figure.Figure = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.set_title(f"{label} | n={graph.number_of_nodes()} e={graph.number_of_edges()}")
    if graph.number_of_nodes() > 0:
        pos = nx.spring_layout(graph, seed=0)
        nx.draw_networkx(
            graph,
            pos=pos,
            ax=ax,
            with_labels=False,
            node_size=80,
            width=1.0,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def render_individual_graph_pngs(
    graphs: list[nx.Graph[Any]],
    out_dir: Path,
    *,
    prefix: str,
    count: int,
) -> list[Path]:
    """Render up to ``count`` graphs to ``out_dir/{prefix}_NN.png``.

    Returns the list of paths actually written so the caller can
    surface them in ``summary.md`` without re-walking the directory.
    Tolerates ``count > len(graphs)`` (writes whatever's available).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    n_to_render = min(count, len(graphs))
    width = max(2, len(str(max(n_to_render - 1, 0))))
    for i in range(n_to_render):
        path = out_dir / f"{prefix}_{i:0{width}d}.png"
        _render_graph_png(graphs[i], path, label=f"{prefix}[{i}]")
        written.append(path)
    return written


def write_val_per_batch_csv(rows: ValBatchRows, path: Path) -> None:
    """Write ``ValBatchRows`` to a CSV with the canonical header."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_VAL_BATCH_FIELDS))
        writer.writeheader()
        for row in rows.rows:
            writer.writerow({k: row.get(k) for k in _VAL_BATCH_FIELDS})


def _flatten_metric_dict(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    """Recursively flatten a nested metric dict into ``a.b.c`` keys."""
    if isinstance(value, dict):
        for sub_key, sub_val in value.items():
            yield from _flatten_metric_dict(
                f"{prefix}.{sub_key}" if prefix else str(sub_key), sub_val
            )
    else:
        yield prefix, value


def write_metrics_csv(metrics: dict[str, Any], path: Path) -> None:
    """Write a single-row CSV from a (possibly nested) metrics dict.

    Cross-ckpt downstream analysis often wants
    ``pd.concat([pd.read_csv(p) for p in glob('*/metrics.csv')])``;
    a single-row layout makes that one-liner trivial.
    """
    flat = dict(_flatten_metric_dict("", metrics))
    fieldnames = sorted(flat.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(flat)


def _format_metric_value(value: Any) -> str:
    """Render a metric value for the human summary."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return str(value)


def _summary_markdown(
    *,
    checkpoint_name: str,
    results: dict[str, Any],
    timings: EvalTimings,
    val_rows: ValBatchRows | None,
    n_generated: int,
    n_reference: int,
    viz_files: list[Path],
    out_dir: Path,
) -> str:
    """Build the human-glance ``summary.md`` for one ckpt subfolder."""
    lines: list[str] = []
    lines.append(f"# eval-all summary: {checkpoint_name}")
    lines.append("")
    lines.append("## counts")
    lines.append(f"- generated: {n_generated}")
    lines.append(f"- reference: {n_reference}")
    if val_rows is not None:
        lines.append(f"- val_batches_walked: {len(val_rows.rows)}")
    lines.append("")
    lines.append("## timings (s)")
    for k, v in timings.to_dict().items():
        lines.append(f"- {k}: {_format_metric_value(v)}")
    lines.append("")
    lines.append("## metrics")
    mmd_results = results.get("mmd_results", {})
    if isinstance(mmd_results, dict):
        for k in sorted(mmd_results.keys()):
            lines.append(f"- {k}: {_format_metric_value(mmd_results[k])}")
    lines.append("")
    if viz_files:
        lines.append("## viz")
        for path in viz_files:
            rel = path.relative_to(out_dir)
            lines.append(f"![{rel}]({rel})")
        lines.append("")
    return "\n".join(lines)


def dump_eval_artifacts(
    out_dir: Path,
    *,
    checkpoint_name: str,
    results: dict[str, Any],
    generated_graphs: list[nx.Graph[Any]],
    reference_graphs: list[nx.Graph[Any]],
    val_rows: ValBatchRows | None,
    timings: EvalTimings,
    viz_count: int,
) -> dict[str, str]:
    """Write the full per-checkpoint artifact set into ``out_dir``.

    Returns
    -------
    dict[str, str]
        Mapping ``artifact_name -> str(path)`` so the caller can
        surface the manifest in ``index.jsonl`` or in W&B run config.

    Notes
    -----
    Order is deliberate: small files (json/csv) first so a partial
    write is still useful, viz last because it's the slowest. Every
    file is written under the same ``out_dir`` so a single
    ``modal Volume.commit`` after this returns flushes the entire
    ckpt.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, default=str))
    paths["metrics_json"] = str(metrics_path)

    metrics_csv_path = out_dir / "metrics.csv"
    write_metrics_csv(results.get("mmd_results", {}), metrics_csv_path)
    paths["metrics_csv"] = str(metrics_csv_path)

    timings_path = out_dir / "timings.json"
    timings_path.write_text(json.dumps(timings.to_dict(), indent=2))
    paths["timings_json"] = str(timings_path)

    if val_rows is not None:
        val_csv_path = out_dir / "val_per_batch.csv"
        write_val_per_batch_csv(val_rows, val_csv_path)
        paths["val_per_batch_csv"] = str(val_csv_path)

    gen_payload = graphs_to_edge_list_payload(generated_graphs)
    gen_path = out_dir / "generated_graphs.json"
    gen_path.write_text(json.dumps(gen_payload))
    paths["generated_graphs_json"] = str(gen_path)

    ref_payload = graphs_to_edge_list_payload(reference_graphs)
    ref_path = out_dir / "reference_graphs.json"
    ref_path.write_text(json.dumps(ref_payload))
    paths["reference_graphs_json"] = str(ref_path)

    viz_dir = out_dir / "viz"
    gen_viz = render_individual_graph_pngs(
        generated_graphs, viz_dir, prefix="generated", count=viz_count
    )
    ref_viz = render_individual_graph_pngs(
        reference_graphs, viz_dir, prefix="reference", count=viz_count
    )
    viz_files = gen_viz + ref_viz
    paths["viz_dir"] = str(viz_dir)

    summary = _summary_markdown(
        checkpoint_name=checkpoint_name,
        results=results,
        timings=timings,
        val_rows=val_rows,
        n_generated=len(generated_graphs),
        n_reference=len(reference_graphs),
        viz_files=viz_files,
        out_dir=out_dir,
    )
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary)
    paths["summary_md"] = str(summary_path)

    return paths


def stage_timer() -> Callable[[], float]:
    """Tiny ``perf_counter`` factory: ``t0 = stage_timer(); ...; t0()``."""
    start = time.perf_counter()
    return lambda: time.perf_counter() - start
