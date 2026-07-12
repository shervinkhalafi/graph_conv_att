"""Unit tests for the eval-all persistence layer.

Covers:
* ``filter_checkpoint_entries`` (ckpt-name + step intersect filter).
* ``_append_index_row`` writes one JSONL row per call without
  clobbering previous rows.
* ``graph_to_edge_list_dict`` round-trips through
  ``edge_list_dict_to_graph`` for both empty and labelled graphs.
* ``write_metrics_csv`` flattens nested keys with ``.`` separator.
* ``write_val_per_batch_csv`` produces the canonical header and one
  row per ``ValBatchRows`` entry.

The end-to-end ``eval_all_checkpoints_impl`` orchestration (W&B init,
subprocess dispatch, volume commits) is exercised by the existing
async-eval smoke run; the intent here is to lock the
locally-testable surface so future refactors don't silently break the
filter / dump / index machinery.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import pytest

from tmgg.experiments.discrete_diffusion_generative._eval_dump import (
    ValBatchRows,
    edge_list_dict_to_graph,
    graph_to_edge_list_dict,
    write_metrics_csv,
    write_val_per_batch_csv,
)
from tmgg.modal._lib.eval_all import (
    CheckpointEntry,
    _append_index_row,
    filter_checkpoint_entries,
)


def _entries() -> list[CheckpointEntry]:
    """Tiny fixture: three stepped ckpts + one with no parsed step."""
    return [
        CheckpointEntry(
            path=Path("/data/run/checkpoints/model-step=05000.ckpt"), step=5000
        ),
        CheckpointEntry(
            path=Path("/data/run/checkpoints/model-step=10000.ckpt"), step=10000
        ),
        CheckpointEntry(
            path=Path("/data/run/checkpoints/model-step=25000.ckpt"), step=25000
        ),
        CheckpointEntry(path=Path("/data/run/checkpoints/last.ckpt"), step=None),
    ]


def test_filter_returns_full_list_when_no_filters_passed() -> None:
    """No filters → identity (caller distinguishes 'all' from 'empty')."""
    assert filter_checkpoint_entries(_entries()) == _entries()


def test_filter_by_step_drops_unsteppable_entries() -> None:
    """Step filter is incompatible with last.ckpt (step=None) so it drops."""
    out = filter_checkpoint_entries(_entries(), step_filter=[5000, 25000])
    assert [e.step for e in out] == [5000, 25000]


def test_filter_by_ckpt_matches_either_path_or_basename() -> None:
    """Match against str(path) OR path.name so users can paste either form."""
    by_basename = filter_checkpoint_entries(
        _entries(), ckpt_filter=["model-step=10000.ckpt"]
    )
    assert [e.step for e in by_basename] == [10000]
    by_full = filter_checkpoint_entries(
        _entries(), ckpt_filter=["/data/run/checkpoints/model-step=10000.ckpt"]
    )
    assert [e.step for e in by_full] == [10000]


def test_filter_intersects_when_both_supplied() -> None:
    """Both filters → intersection (entry must satisfy both)."""
    out = filter_checkpoint_entries(
        _entries(),
        ckpt_filter=["model-step=10000.ckpt", "model-step=25000.ckpt"],
        step_filter=[10000],
    )
    assert [e.step for e in out] == [10000]


def test_append_index_row_writes_one_line_per_call(tmp_path: Path) -> None:
    """index.jsonl must be append-only — successive calls accumulate."""
    index_path = tmp_path / "eval_all" / "run" / "index.jsonl"
    _append_index_row(index_path, {"step": 5000, "status": "completed"})
    _append_index_row(index_path, {"step": 10000, "status": "failed"})
    lines = index_path.read_text().strip().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["step"] for r in rows] == [5000, 10000]
    assert [r["status"] for r in rows] == ["completed", "failed"]


def test_graph_round_trip_empty_graph() -> None:
    """An n=0 graph survives the JSON round-trip (count + edge list both empty)."""
    g: nx.Graph = nx.Graph()
    payload = graph_to_edge_list_dict(g)
    assert payload["n_nodes"] == 0
    assert payload["n_edges"] == 0
    g2 = edge_list_dict_to_graph(payload)
    assert g2.number_of_nodes() == 0
    assert g2.number_of_edges() == 0


def test_graph_round_trip_with_node_and_edge_attrs() -> None:
    """Topology + node/edge attribute dicts must survive the round trip."""
    g: nx.Graph = nx.Graph()
    g.add_node(0, label="atom_C")
    g.add_node(1, label="atom_N")
    g.add_edge(0, 1, bond_type="single")
    payload = graph_to_edge_list_dict(g)
    g2 = edge_list_dict_to_graph(payload)
    assert g2.nodes[0]["label"] == "atom_C"
    assert g2.nodes[1]["label"] == "atom_N"
    assert g2.edges[0, 1]["bond_type"] == "single"


def test_metrics_csv_flattens_nested_dict(tmp_path: Path) -> None:
    """A nested ``{eval: {degree_mmd: 0.01}}`` becomes one ``eval.degree_mmd`` cell."""
    metrics = {"eval": {"degree_mmd": 0.01, "clustering_mmd": 0.02}, "novelty": None}
    out = tmp_path / "metrics.csv"
    write_metrics_csv(metrics, out)
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    row = rows[0]
    assert row["eval.degree_mmd"] == "0.01"
    assert row["eval.clustering_mmd"] == "0.02"
    # JSON ``None`` flattens to empty cell via DictWriter coercion.
    assert row["novelty"] == ""


def test_val_per_batch_csv_writes_canonical_header(tmp_path: Path) -> None:
    """Header is fixed so cross-ckpt pandas concat is straightforward."""
    rows = ValBatchRows(
        rows=[
            {
                "batch_idx": 0,
                "batch_size": 32,
                "nll": 1.234,
                "kl_prior": 0.001,
                "kl_diffusion": 1.0,
                "reconstruction": -0.05,
                "log_pN": -2.0,
            },
            {
                "batch_idx": 1,
                "batch_size": 32,
                "nll": 1.111,
                "kl_prior": None,
                "kl_diffusion": 0.9,
                "reconstruction": -0.04,
                "log_pN": -2.0,
            },
        ]
    )
    out = tmp_path / "val_per_batch.csv"
    write_val_per_batch_csv(rows, out)
    parsed = list(csv.DictReader(out.open()))
    assert len(parsed) == 2
    assert parsed[0]["batch_idx"] == "0"
    assert parsed[0]["nll"] == "1.234"
    # ``None`` value lands as empty cell rather than the literal "None".
    assert parsed[1]["kl_prior"] == ""


def test_dump_eval_artifacts_writes_full_layout(tmp_path: Path) -> None:
    """End-to-end shape check on the dump module's file layout."""
    from tmgg.experiments.discrete_diffusion_generative._eval_dump import (
        EvalTimings,
        dump_eval_artifacts,
    )

    g_a: nx.Graph = nx.cycle_graph(5)
    g_b: nx.Graph = nx.path_graph(4)
    paths = dump_eval_artifacts(
        tmp_path / "ckpt-step=5000",
        checkpoint_name="model-step=5000.ckpt",
        results={
            "checkpoint_name": "model-step=5000.ckpt",
            "mmd_results": {"degree_mmd": 0.01, "spectral_mmd": 0.02},
        },
        generated_graphs=[g_a, g_b],
        reference_graphs=[g_a, g_b],
        val_rows=ValBatchRows(
            rows=[
                {
                    "batch_idx": 0,
                    "batch_size": 8,
                    "nll": 1.0,
                    "kl_prior": 0.0,
                    "kl_diffusion": 1.0,
                    "reconstruction": 0.0,
                    "log_pN": -1.0,
                }
            ]
        ),
        timings=EvalTimings(
            load_s=1.0, val_pass_s=2.0, sample_s=3.0, eval_s=4.0, total_s=10.0
        ),
        viz_count=2,
    )
    out_dir = tmp_path / "ckpt-step=5000"
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "timings.json").exists()
    assert (out_dir / "val_per_batch.csv").exists()
    assert (out_dir / "generated_graphs.json").exists()
    assert (out_dir / "reference_graphs.json").exists()
    assert (out_dir / "summary.md").exists()
    # Two viz PNGs per side (capped at viz_count=2). Width is the max
    # of 2 and len(str(viz_count-1)) so single-digit viz_count still
    # produces zero-padded ``generated_00.png`` filenames for stable
    # lexicographic sort.
    viz_dir = out_dir / "viz"
    assert sorted(p.name for p in viz_dir.glob("generated_*.png")) == [
        "generated_00.png",
        "generated_01.png",
    ]
    assert sorted(p.name for p in viz_dir.glob("reference_*.png")) == [
        "reference_00.png",
        "reference_01.png",
    ]
    # Manifest must surface every artifact.
    for key in (
        "metrics_json",
        "metrics_csv",
        "timings_json",
        "val_per_batch_csv",
        "generated_graphs_json",
        "reference_graphs_json",
        "summary_md",
        "viz_dir",
    ):
        assert key in paths


def test_val_pass_populates_size_distribution_from_datamodule() -> None:
    """Regression: val-pass must hydrate ``_size_distribution`` from the
    datamodule before iterating, because the CLI flow skips Lightning's
    ``setup()`` (no Trainer attached). Pre-fix, the categorical branch
    of ``validation_step`` raised ``_size_distribution is None`` on the
    very first batch, killing every per-ckpt eval-all dump.
    """
    from tmgg.experiments.discrete_diffusion_generative._eval_dump import (
        run_val_pass_with_per_batch_capture,
    )

    class _StubSize:
        def log_prob(self, _node_counts: object) -> object:
            raise AssertionError("not needed for this regression test")

    class _StubModule:
        def __init__(self) -> None:
            self._size_distribution: object | None = None
            self.device: str = "cpu"
            self.log = lambda *a, **kw: None

        def on_validation_epoch_start(self) -> None:
            return None

    class _StubDM:
        def __init__(self) -> None:
            self.size = _StubSize()

        def get_size_distribution(self, split: str) -> _StubSize:
            assert split == "train"
            return self.size

        def val_dataloader(self) -> list[object]:
            return []  # zero batches → loop body never executes

    module = _StubModule()
    dm = _StubDM()
    rows = run_val_pass_with_per_batch_capture(module, dm)
    assert module._size_distribution is dm.size
    assert rows.rows == []


@pytest.mark.parametrize(
    "edges",
    [
        [(0, 1), (1, 2), (2, 0)],  # triangle
        [],  # disconnected
    ],
)
def test_generated_graphs_json_loads_back_to_same_topology(
    tmp_path: Path, edges: list[tuple[int, int]]
) -> None:
    """``json.dump → json.load → edge_list_dict_to_graph`` is identity for topology."""
    g: nx.Graph = nx.Graph()
    g.add_nodes_from([0, 1, 2])
    g.add_edges_from(edges)
    payload = graph_to_edge_list_dict(g)
    blob = tmp_path / "g.json"
    blob.write_text(json.dumps(payload))
    parsed = json.loads(blob.read_text())
    g2 = edge_list_dict_to_graph(parsed)
    assert sorted(g2.nodes()) == sorted(g.nodes())
    assert sorted(map(tuple, sorted(map(sorted, g2.edges())))) == sorted(
        map(tuple, sorted(map(sorted, g.edges())))
    )
