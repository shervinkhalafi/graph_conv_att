"""Stage 3 telemetry: dataset-agnostic block-structure metrics.

These tests pin the qualitative signal of
:func:`compute_block_structure_metrics`: a 2-block SBM with strong
contrast should yield high modularity Q and recover (p_in, p_out)
near the ground truth, while an Erdős–Rényi graph at matched
density should give Q near 0 and roughly equal p̂_in / p̂_out.

We use NetworkX's stochastic_block_model and erdos_renyi_graph
fixtures to generate graphs without pulling in the heavier
``GraphEvaluator.evaluate()`` path.
"""

from __future__ import annotations

import networkx as nx
import torch

from tmgg.evaluation.graph_evaluator import compute_block_structure_metrics


def _make_sbm(p_in: float, p_out: float, sizes: tuple[int, int], seed: int) -> nx.Graph:
    """Two-block SBM with the given intra/inter probabilities."""
    pmat = [[p_in, p_out], [p_out, p_in]]
    return nx.stochastic_block_model(sizes, pmat, seed=seed)  # pyright: ignore[reportReturnType]


def test_modularity_q_high_on_clear_sbm() -> None:
    """Strong block contrast ⇒ Q > 0.3.

    With ``p_in = 0.9``, ``p_out = 0.05`` the modularity of any
    sensible 2-cluster partition is well above 0.3 (Newman's
    "communities clearly identifiable" threshold).
    """
    graphs = [_make_sbm(0.9, 0.05, (15, 15), seed=s) for s in range(8)]
    res = compute_block_structure_metrics(graphs)
    assert res["modularity_q"] is not None
    assert res["modularity_q"] > 0.3, res


def test_modularity_q_low_on_er() -> None:
    """ER graphs have no block structure ⇒ Q stays well below the SBM Q.

    The Fiedler-vector cut on an ER graph still produces *some*
    partition; finite-sample noise gives Q ≈ 0.15-0.2 at n=30. The
    qualitative claim we want to pin is that ER is **far below** the
    clear-SBM Q value — at least 2× lower. We compare against the
    matched-density SBM rather than asserting an absolute threshold so
    this test is robust to changes in the partition algorithm.
    """
    er_graphs = [nx.erdos_renyi_graph(30, 0.3, seed=s) for s in range(8)]
    sbm_graphs = [_make_sbm(0.9, 0.05, (15, 15), seed=s) for s in range(8)]
    er_res = compute_block_structure_metrics(er_graphs)
    sbm_res = compute_block_structure_metrics(sbm_graphs)
    assert er_res["modularity_q"] is not None
    assert sbm_res["modularity_q"] is not None
    assert er_res["modularity_q"] < sbm_res["modularity_q"] / 2, (
        er_res,
        sbm_res,
    )


def test_p_in_p_out_recovery_on_clean_sbm() -> None:
    """Recovered (p̂_in, p̂_out) within ~0.15 of the ground truth.

    The Fiedler-vector partition is not optimal (it's a 2-eigenvector
    relaxation of min-cut), so we allow generous slack — the goal of
    this test is to confirm the *qualitative* recovery, not match a
    likelihood-based estimator.
    """
    graphs = [_make_sbm(0.85, 0.05, (20, 20), seed=s) for s in range(16)]
    res = compute_block_structure_metrics(graphs)
    assert res["empirical_p_in"] is not None
    assert res["empirical_p_out"] is not None
    assert abs(res["empirical_p_in"] - 0.85) < 0.15, res
    assert abs(res["empirical_p_out"] - 0.05) < 0.15, res
    # Ordering invariant: p̂_in > p̂_out by a wide margin.
    assert res["empirical_p_in"] - res["empirical_p_out"] > 0.5


def test_spectral_lambda2_separates_sbm_from_er() -> None:
    """λ₂ of the SBM adjacency is far above λ₂ of ER at matched density.

    For a 2-block SBM with sizes (n,n), λ_max ≈ (p_in + p_out)*n/2
    (the all-ones direction) and λ₂ ≈ (p_in - p_out)*n/2 (block
    contrast). For ER at matched density there is no block-contrast
    direction, so λ₂ stays near the bulk (≈ √(n*p*(1-p))). The metric
    we report ('spectral_gap_l2') is λ₂ itself.
    """
    sbms = [_make_sbm(0.9, 0.05, (15, 15), seed=s) for s in range(8)]
    sbm_avg_edges = sum(g.number_of_edges() for g in sbms) / len(sbms)
    p_match = sbm_avg_edges / (30 * 29 / 2)
    ers = [nx.erdos_renyi_graph(30, p_match, seed=s) for s in range(8)]
    sbm_res = compute_block_structure_metrics(sbms)
    er_res = compute_block_structure_metrics(ers)
    assert sbm_res["spectral_gap_l2"] is not None
    assert er_res["spectral_gap_l2"] is not None
    assert sbm_res["spectral_gap_l2"] > er_res["spectral_gap_l2"], (
        sbm_res,
        er_res,
    )


def test_empty_input_returns_all_none() -> None:
    """No graphs ⇒ all four block-structure fields are ``None``."""
    res = compute_block_structure_metrics([])
    assert res == {
        "modularity_q": None,
        "spectral_gap_l2": None,
        "empirical_p_in": None,
        "empirical_p_out": None,
    }


def test_skips_too_small_graphs() -> None:
    """Singletons / empty graphs are filtered before the eigh path.

    A pathological input where every graph has < 2 nodes should
    return ``None``s rather than crashing inside ``torch.linalg.eigh``.
    """
    g0 = nx.Graph()  # empty
    g1 = nx.Graph()
    g1.add_node(0)  # single node
    res = compute_block_structure_metrics([g0, g1])
    assert res["modularity_q"] is None


def test_runs_on_cpu_device_explicit() -> None:
    """``device='cpu'`` explicit path returns finite values.

    Smoke test for the ``device`` keyword; we don't assert on a CUDA
    path here since CI doesn't have a GPU.
    """
    graphs = [_make_sbm(0.8, 0.1, (8, 8), seed=s) for s in range(4)]
    res = compute_block_structure_metrics(graphs, device="cpu")
    for value in res.values():
        assert value is not None
        assert torch.isfinite(torch.tensor(value))
