"""Unified graph generation evaluator combining MMD and structural metrics.

This module provides a single ``GraphEvaluator`` that computes all graph
generation evaluation metrics in one pass:

- **Distributional MMD** (degree, clustering, spectral) via the existing
  ``compute_mmd_metrics`` infrastructure.
- **Orbit MMD** via ORCA graphlet counting (requires compiled ``orca`` binary).
- **SBM accuracy** via graph-tool community detection (requires ``graph-tool``).
- **Planarity accuracy** via networkx planarity check.
- **Uniqueness** as the fraction of distinct isomorphism classes among
  generated graphs.
- **Novelty** as the fraction of generated graphs not isomorphic to any
  training graph (requires training graphs via the constructor).

External dependencies (graph-tool, orca) are checked at import time.
When unavailable, the corresponding metric fields are set to ``None``; a
warning is emitted once at import for each missing dependency. When available,
failures propagate without suppression.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import multiprocessing
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import torch
from scipy.stats import chi2

from tmgg.evaluation.mmd_metrics import (
    compute_mmd,
    compute_mmd_metrics,
)
from tmgg.evaluation.orca import (
    is_available as _orca_is_available,
)
from tmgg.evaluation.orca import run_orca

if TYPE_CHECKING:
    from tmgg.data.datasets.graph_types import (
        DenseGraphState,
        GraphState,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency availability
# ---------------------------------------------------------------------------
_GRAPH_TOOL_IMPORT_ERROR: ImportError | None = None
gt: Any | None = None
if importlib.util.find_spec("graph_tool") is not None:
    try:
        import graph_tool.all as gt  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        _GRAPH_TOOL_IMPORT_ERROR = exc
        _GRAPH_TOOL_AVAILABLE = False
        warnings.warn(
            "graph-tool import failed during evaluator import; "
            "sbm_accuracy will be None in GraphEvaluator results.\n"
            f"Import error: {exc}",
            stacklevel=2,
        )
    else:
        _GRAPH_TOOL_AVAILABLE = True
else:
    _GRAPH_TOOL_AVAILABLE = False
_ORCA_AVAILABLE = _orca_is_available()

if not _GRAPH_TOOL_AVAILABLE and _GRAPH_TOOL_IMPORT_ERROR is None:
    warnings.warn(
        "graph-tool not installed; sbm_accuracy will be None in GraphEvaluator results.",
        stacklevel=2,
    )
if not _ORCA_AVAILABLE:
    warnings.warn(
        "orca binary not found; orbit_mmd will be None in GraphEvaluator results. "
        "Compile orca.cpp and set ORCA_PATH or place the binary "
        "in the expected location.",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# EvaluationResults
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResults:
    """Flat container for all graph generation evaluation metrics.

    All ``*_mmd`` fields hold **squared MMD values** (V-statistic biased
    estimator), not square-root MMD distances — this is the
    GraphRNN/GRAN convention that DiGress and HiGen also use. See
    ``docs/eval/mmd-units-and-protocol.md`` for unit conventions and
    cross-paper comparison rules.

    Attributes
    ----------
    degree_mmd
        MMD² on degree histograms (always computed).
    clustering_mmd
        MMD² on clustering coefficient histograms (always computed).
    spectral_mmd
        MMD² on normalized Laplacian eigenvalue histograms (always computed).
    orbit_mmd
        MMD² on ORCA orbit counts. None if orca binary is unavailable.
    sbm_accuracy
        Fraction of generated graphs consistent with an SBM. None if
        graph-tool is unavailable.
    planarity_accuracy
        Fraction of generated graphs that are connected and planar.
        None if skipped via ``skip_metrics``.
    uniqueness
        Fraction of generated graphs in distinct isomorphism classes.
        None if skipped via ``skip_metrics``.
    novelty
        Fraction of generated graphs not isomorphic to any training graph.
        None if train_graphs were not provided in the constructor or skipped.
    """

    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float
    orbit_mmd: float | None
    sbm_accuracy: float | None
    planarity_accuracy: float | None
    uniqueness: float | None
    novelty: float | None
    # Block-structure metrics (Stage 3 telemetry). All four are mean
    # values over generated graphs; ``modularity_q`` and the
    # ``empirical_p_*`` come from a 2-block spectral partition (sign of
    # the Fiedler vector of the symmetric normalised Laplacian).
    modularity_q: float | None = None
    spectral_gap_l2: float | None = None
    empirical_p_in: float | None = None
    empirical_p_out: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Return a flat dictionary of all metrics, suitable for logging."""
        return {
            "degree_mmd": self.degree_mmd,
            "clustering_mmd": self.clustering_mmd,
            "spectral_mmd": self.spectral_mmd,
            "orbit_mmd": self.orbit_mmd,
            "sbm_accuracy": self.sbm_accuracy,
            "planarity_accuracy": self.planarity_accuracy,
            "uniqueness": self.uniqueness,
            "novelty": self.novelty,
            "modularity_q": self.modularity_q,
            "spectral_gap_l2": self.spectral_gap_l2,
            "empirical_p_in": self.empirical_p_in,
            "empirical_p_out": self.empirical_p_out,
        }


# ---------------------------------------------------------------------------
# Structural metric helpers (ported from DiGress spectre_utils)
# ---------------------------------------------------------------------------


def _orca_normalized_counts(graph: nx.Graph[Any]) -> np.ndarray:
    """Compute normalized ORCA orbit counts for a graph.

    Removes self-loops, runs ORCA, and returns the mean orbit count vector
    (averaged over nodes). Graphs with fewer than 2 nodes are skipped by
    callers before reaching this function.
    """
    g_clean = graph.copy()
    g_clean.remove_edges_from(nx.selfloop_edges(g_clean))
    counts = run_orca(g_clean)
    return np.sum(counts, axis=0) / g_clean.number_of_nodes()


def compute_orbit_mmd(
    ref_graphs: list[nx.Graph[Any]],
    gen_graphs: list[nx.Graph[Any]],
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 30.0,
) -> float:
    """Compute MMD² on normalized ORCA orbit count vectors.

    Returns the V-statistic biased squared-MMD value (the "orbit MMD"
    column in the GraphRNN/GRAN/DiGress/HiGen literature is this same
    quantity, not its square root). See
    ``docs/eval/mmd-units-and-protocol.md`` for the unit convention.

    Parameters
    ----------
    ref_graphs
        Reference graphs.
    gen_graphs
        Generated graphs.
    kernel
        Kernel for MMD computation.
    sigma
        Kernel bandwidth (default 30.0, matching DiGress).

    Returns
    -------
    float
        Orbit MMD² value (despite the literature calling it "orbit MMD").
    """
    ref_counts: list[np.ndarray] = [
        _orca_normalized_counts(g) for g in ref_graphs if g.number_of_nodes() >= 2
    ]
    gen_counts: list[np.ndarray] = [
        _orca_normalized_counts(g) for g in gen_graphs if g.number_of_nodes() >= 2
    ]

    if not ref_counts or not gen_counts:
        return 0.0

    # Orbit counts are raw graphlet counts, not histograms — magnitude is the
    # signal. Upstream DiGress's ``orbit_stats_all`` calls compute_mmd with
    # is_hist=False, sigma=30.0 (digress-upstream-readonly/src/analysis/
    # spectre_utils.py:490). Matching that here.
    return compute_mmd(
        ref_counts, gen_counts, kernel=kernel, sigma=sigma, is_hist=False
    )


# Ported from DiGress (Vignac et al., ICLR 2023) / SPECTRE (Martinkus et al.,
# NeurIPS 2022).  Upstream: cvignac/DiGress, src/analysis/spectre_utils.py,
# functions ``eval_acc_sbm_graph`` and ``is_sbm_graph``.


def _is_sbm_graph(
    g_nx: nx.Graph[Any],
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    strict: bool = True,
    refinement_steps: int = 1000,
) -> float:
    """Test whether a single graph is consistent with an SBM.

    Fits a stochastic block model via graph-tool's ``minimize_blockmodel_dl``,
    refines via merge-split MCMC, then applies a Wald test comparing recovered
    intra-/inter-block edge probabilities against expected values.

    Parameters
    ----------
    g_nx
        Graph to evaluate.
    p_intra
        Expected within-community edge probability.
    p_inter
        Expected between-community edge probability.
    strict
        If True, reject graphs with extreme block counts or sizes and
        return a binary pass/fail. If False, return the raw mean p-value.
    refinement_steps
        Number of MCMC refinement sweeps.

    Returns
    -------
    float
        ``1.0`` / ``0.0`` when *strict*, raw mean p-value otherwise.
    """
    if gt is None:
        raise RuntimeError("graph-tool is unavailable; cannot compute SBM accuracy.")

    adj = nx.adjacency_matrix(g_nx).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))

    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        # Graph too degenerate for blockmodel inference; scores 0.
        return 0.0 if not strict else 0.0

    for _ in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]

    if strict and (
        (node_counts > 40).sum() > 0
        or (node_counts < 20).sum() > 0
        or n_blocks > 5
        or n_blocks < 2
    ):
        return 0.0

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p_val = 1 - chi2.cdf(abs(np.asarray(W)), 1)
    mean_p = float(p_val.mean())

    if strict:
        return float(mean_p > 0.9)
    return mean_p


def compute_sbm_accuracy(
    graphs: list[nx.Graph[Any]],
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    strict: bool = True,
    refinement_steps: int = 1000,
    is_parallel: bool = True,
) -> float:
    """Fraction of graphs consistent with a stochastic block model.

    Evaluates each graph via ``_is_sbm_graph`` (blockmodel inference +
    Wald test), optionally in parallel. When ``is_parallel=True`` the
    per-graph calls run inside a ``ProcessPoolExecutor`` that uses the
    ``spawn`` start method. A thread-pool backend was tried initially
    but graph-tool's blockmodel routines are not safe to call
    concurrently from Python threads: the workers share graph-tool's
    internal OpenMP/C++ state and abort the process with heap-
    corruption signals (``vector::_M_fill_insert``, ``malloc()
    unaligned tcache chunk``). Process isolation avoids the hazard at
    the cost of interpreter-spawn overhead. See
    ``internal notes (not included)``.

    Parameters
    ----------
    graphs
        Generated graphs to evaluate.
    p_intra
        Expected within-community edge probability.
    p_inter
        Expected between-community edge probability.
    strict
        If True, reject graphs with extreme block counts or sizes.
    refinement_steps
        Number of MCMC refinement sweeps.
    is_parallel
        If True, evaluate graphs in a spawn-based
        ``ProcessPoolExecutor``. If False, evaluate sequentially in
        the caller's process.

    Returns
    -------
    float
        Fraction of graphs passing the SBM test (strict) or mean p-value.

    Raises
    ------
    ImportError
        If graph-tool is not installed.
    """
    if not graphs:
        return 0.0

    if not is_parallel:
        count = sum(
            _is_sbm_graph(g, p_intra, p_inter, strict, refinement_steps) for g in graphs
        )
        return count / len(graphs)

    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(mp_context=mp_context) as executor:
        scores = list(
            executor.map(
                _is_sbm_graph,
                graphs,
                [p_intra] * len(graphs),
                [p_inter] * len(graphs),
                [strict] * len(graphs),
                [refinement_steps] * len(graphs),
            )
        )
    count = sum(scores)

    return count / len(graphs)


def compute_planarity_accuracy(graphs: list[nx.Graph[Any]]) -> float:
    """Fraction of generated graphs that are both connected and planar.

    Parameters
    ----------
    graphs
        Generated graphs.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if not graphs:
        return 0.0
    count = sum(
        1
        for g in graphs
        if g.number_of_nodes() > 0 and nx.is_connected(g) and nx.check_planarity(g)[0]
    )
    return count / len(graphs)


def compute_uniqueness(graphs: list[nx.Graph[Any]]) -> float:
    """Fraction of generated graphs in distinct isomorphism classes.

    Uses ``nx.faster_could_be_isomorphic`` as a fast filter followed by
    ``nx.is_isomorphic`` for confirmation, matching DiGress's approach.

    Parameters
    ----------
    graphs
        Generated graphs.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if not graphs:
        return 0.0

    count_non_unique = 0
    evaluated: list[nx.Graph[Any]] = []

    for g in graphs:
        if g.number_of_nodes() == 0:
            continue
        is_unique = True
        for g_old in evaluated:
            if nx.faster_could_be_isomorphic(g, g_old) and nx.is_isomorphic(g, g_old):
                count_non_unique += 1
                is_unique = False
                break
        if is_unique:
            evaluated.append(g)

    return (len(graphs) - count_non_unique) / len(graphs)


def compute_block_structure_metrics(
    graphs: list[nx.Graph[Any]],
    *,
    device: torch.device | str | None = None,
) -> dict[str, float | None]:
    """Mean modularity Q, λ₂ spectral gap, and (p̂_in, p̂_out) over a sample.

    Cheap dataset-agnostic block-structure proxies. The 2-block partition
    is recovered from the sign of the Fiedler vector (eigenvector
    associated with the second-smallest eigenvalue of the symmetric
    normalised Laplacian). Modularity Q and (p̂_in, p̂_out) are
    computed against that partition. The spectral gap reuses
    :func:`tmgg.experiments.eigenstructure_study.analyzer.compute_spectral_gap`
    on the *adjacency* eigenvalues.

    All four metrics are means across the input batch. Graphs with
    fewer than 2 nodes contribute nothing (skipped). Empty inputs
    return ``None`` for every key.

    Parameters
    ----------
    graphs
        Generated NetworkX graphs (already binarised by
        :meth:`GraphEvaluator.to_networkx_graphs`).
    device
        Optional torch device. Defaults to CPU; passing ``"cuda"``
        runs the batched eigendecompositions on GPU. The conversion
        from NetworkX still happens on CPU regardless — the eigh
        call dominates cost only for ``n`` ≳ 100.

    Returns
    -------
    dict[str, float | None]
        ``modularity_q``, ``spectral_gap_l2``, ``empirical_p_in``,
        ``empirical_p_out``. Each value is a Python float or ``None``
        when no graph contributed (e.g. all inputs had < 2 nodes).
    """
    if not graphs:
        return {
            "modularity_q": None,
            "spectral_gap_l2": None,
            "empirical_p_in": None,
            "empirical_p_out": None,
        }

    target_device = torch.device(device) if device is not None else torch.device("cpu")

    # Pad each graph to a common ``n_max`` so the batched eigh / Q
    # formula run as a single tensor op. ``node_mask`` carries which
    # rows/cols are real for that graph.
    valid_graphs: list[nx.Graph[Any]] = [g for g in graphs if g.number_of_nodes() >= 2]
    if not valid_graphs:
        return {
            "modularity_q": None,
            "spectral_gap_l2": None,
            "empirical_p_in": None,
            "empirical_p_out": None,
        }
    n_max = max(g.number_of_nodes() for g in valid_graphs)
    bs = len(valid_graphs)

    A = torch.zeros((bs, n_max, n_max), dtype=torch.float32)
    node_mask = torch.zeros((bs, n_max), dtype=torch.bool)
    for i, g in enumerate(valid_graphs):
        a = nx.to_numpy_array(g, dtype=np.dtype(np.float32))
        n_i = a.shape[0]
        # Symmetrise + zero diagonal defensively (the binarised
        # adjacency from ``GraphEvaluator.to_networkx_graphs`` is
        # already simple-undirected, but cheap to enforce).
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        A[i, :n_i, :n_i] = torch.from_numpy(a)
        node_mask[i, :n_i] = True

    A = A.to(target_device)
    node_mask = node_mask.to(target_device)
    mask_2d = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    # Second-largest adjacency eigenvalue λ₂. For an SBM with two
    # roughly equal blocks, λ_max ≈ (p_in + p_out) * n / 2 (the
    # all-ones direction) and λ₂ ≈ (p_in - p_out) * n / 2 (the
    # block-contrast direction); ER graphs have λ₂ near the bulk
    # so the metric separates them cleanly. The eigh call's
    # ascending order means index ``-2`` is the second-largest.
    # Fully padded rows produce trailing zero eigenvalues, so the
    # top-2 picks the real-graph signal regardless of padding.
    eigvals_A = torch.linalg.eigvalsh(A)  # ascending, (B, n_max)
    spectral_gap_mean = eigvals_A[:, -2].mean().item()

    # Symmetric normalised Laplacian L_norm = I - D^{-1/2} A D^{-1/2}.
    # Mask padding rows/cols by zeroing both A and the corresponding
    # I rows so they yield trivial zero eigenvalues that we discard.
    deg = (A * mask_2d).sum(dim=-1)  # (B, n_max)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    eye = torch.eye(n_max, device=target_device).unsqueeze(0).expand(bs, -1, -1)
    norm_factor = deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)
    L_sym = (eye - A * norm_factor) * mask_2d.to(A.dtype)

    # Padding rows produce extra zero eigenvalues that would confuse
    # the Fiedler-vector pick if we ran a single batched eigh. Cheaper
    # and clearer to run eigh per graph on the live submatrix:
    # O(n³) per graph but ``n`` is small in our regime (≤ 100), and
    # the call is dwarfed by NetworkX conversion.
    fiedler = torch.zeros((bs, n_max), device=target_device)
    for i, g in enumerate(valid_graphs):
        n_i = g.number_of_nodes()
        sub = L_sym[i, :n_i, :n_i]
        _, vec_sub = torch.linalg.eigh(sub)
        # eigvals[0] ≈ 0 for a connected block; pick index 1
        # (Fiedler). For a 2-node graph we still pick index 1.
        idx = 1 if n_i >= 2 else 0
        fiedler[i, :n_i] = vec_sub[:, idx]

    # 2-block partition from sign of Fiedler vector. Pad rows
    # (already 0) get cluster 0; that's harmless because their
    # contribution is masked out everywhere downstream.
    cluster = (fiedler >= 0).to(A.dtype)  # (B, n_max), 0/1

    # Modularity Q = (1/2m) Σ_ij (A_ij - d_i d_j / 2m) δ(c_i, c_j),
    # restricted to real positions. ``2m`` is the sum of A over the
    # masked region (each undirected edge counted twice).
    same_cluster = (cluster.unsqueeze(-1) == cluster.unsqueeze(-2)).to(A.dtype)
    same_cluster = same_cluster * mask_2d.to(A.dtype)
    deg_outer = deg.unsqueeze(-1) * deg.unsqueeze(-2)  # (B, n_max, n_max)
    two_m = deg.sum(dim=-1).clamp(min=1e-12)  # (B,)
    q_terms = (A - deg_outer / two_m.view(-1, 1, 1)) * same_cluster
    q = q_terms.sum(dim=(-1, -2)) / two_m
    modularity_q = float(q.mean().item())

    # Empirical (p̂_in, p̂_out): block-pair edge densities. ``intra``
    # already incorporates ``mask_2d`` via ``same_cluster``; ``inter``
    # adds the mask explicitly. Diagonal is zero in A so intra-pair
    # counts naturally exclude self-pairs from the edge sum, but we
    # subtract n_i from the pair count (one diagonal per node) so the
    # density denominator matches the off-diagonal numerator.
    intra = same_cluster
    inter = mask_2d.to(A.dtype) * (cluster.unsqueeze(-1) != cluster.unsqueeze(-2)).to(
        A.dtype
    )
    intra_edges = (A * intra).sum(dim=(-1, -2))
    inter_edges = (A * inter).sum(dim=(-1, -2))
    n_real = node_mask.sum(dim=-1).to(A.dtype)
    intra_pairs = (intra.sum(dim=(-1, -2)) - n_real).clamp(min=1.0)
    inter_pairs = inter.sum(dim=(-1, -2)).clamp(min=1.0)
    p_in = (intra_edges / intra_pairs).mean().item()
    p_out = (inter_edges / inter_pairs).mean().item()

    return {
        "modularity_q": modularity_q,
        "spectral_gap_l2": spectral_gap_mean,
        "empirical_p_in": p_in,
        "empirical_p_out": p_out,
    }


def compute_novelty(
    generated: list[nx.Graph[Any]], train_graphs: list[nx.Graph[Any]]
) -> float:
    """Fraction of generated graphs not isomorphic to any training graph.

    Parameters
    ----------
    generated
        Generated graphs.
    train_graphs
        Training set graphs.

    Returns
    -------
    float
        Fraction in [0, 1]. 1.0 means all generated graphs are novel.
    """
    if not generated:
        return 0.0

    count_isomorphic = 0
    for gen_g in generated:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(gen_g, train_g) and nx.is_isomorphic(
                gen_g, train_g
            ):
                count_isomorphic += 1
                break

    return 1.0 - count_isomorphic / len(generated)


# ---------------------------------------------------------------------------
# GraphEvaluator
# ---------------------------------------------------------------------------
class GraphEvaluator:
    """Unified evaluator for graph generation quality.

    Accepts reference and generated graphs directly in ``evaluate()``, and
    optionally training graphs in the constructor for novelty computation.
    Computes all standard metrics: degree/clustering/spectral MMD, orbit
    MMD, SBM accuracy, planarity, uniqueness, and novelty.

    Parameters
    ----------
    eval_num_samples
        Maximum number of reference and generated graphs to evaluate.
        Inputs are truncated to this length inside ``evaluate()``.
    kernel
        Kernel for MMD computation (``"gaussian_tv"`` or ``"gaussian"``).
    sigma
        Fallback bandwidth for the MMD kernel when no per-metric override
        is supplied.
    p_intra
        Expected within-community edge probability for SBM accuracy.
    p_inter
        Expected between-community edge probability for SBM accuracy.
    skip_metrics
        Metric names to skip (``"orbit"``, ``"sbm"``, ``"planarity"``,
        ``"uniqueness"``, ``"novelty"``). Skipped metrics return ``None``
        in :class:`EvaluationResults`.
    train_graphs
        Training set graphs used for novelty evaluation. If None, novelty
        is not computed.
    degree_sigma, clustering_sigma, spectral_sigma
        Per-metric kernel bandwidths. Upstream DiGress ships
        ``clustering_sigma=0.1`` with the others at 1.0 for
        ``gaussian_tv``; parity runs should use those values. Each
        defaults to ``None``, which falls back to ``sigma``.
    sbm_refinement_steps
        Number of multiflip MCMC sweeps used to refine the block-model
        fit inside :func:`compute_sbm_accuracy`. Default 100 matches
        upstream DiGress's live ``SpectreSamplingMetrics`` value
        (``src/analysis/spectre_utils.py:830``). The underlying function
        defaults to 1000, but upstream never invokes it at that value —
        they always override to 100. Setting higher gives tighter
        block-model fits but is roughly 10× slower.

    Examples
    --------
    >>> evaluator = GraphEvaluator(eval_num_samples=100, train_graphs=train_set)
    >>> results = evaluator.evaluate(refs=validation_graphs, generated=generated_graphs)
    >>> print(results.to_dict())
    """

    def __init__(
        self,
        eval_num_samples: int,
        kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
        sigma: float = 1.0,
        p_intra: float = 0.3,
        p_inter: float = 0.005,
        skip_metrics: set[str] | frozenset[str] | list[str] | None = None,
        train_graphs: list[nx.Graph[Any]] | None = None,
        degree_sigma: float | None = None,
        clustering_sigma: float | None = None,
        spectral_sigma: float | None = None,
        sbm_refinement_steps: int = 100,
    ) -> None:
        self.eval_num_samples = eval_num_samples
        self.kernel: Literal["gaussian", "gaussian_tv"] = kernel
        self.sigma = sigma
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.skip_metrics: frozenset[str] = frozenset(skip_metrics or ())
        self.train_graphs = list(train_graphs) if train_graphs is not None else []
        self._train_graphs_set = train_graphs is not None
        self.degree_sigma: float | None = degree_sigma
        self.clustering_sigma: float | None = clustering_sigma
        self.spectral_sigma: float | None = spectral_sigma
        self.sbm_refinement_steps: int = sbm_refinement_steps

    # ------------------------------------------------------------------
    # GraphState / DenseGraphState → NetworkX
    # ------------------------------------------------------------------
    @staticmethod
    def _to_networkx_list(
        samples: GraphState | DenseGraphState | Sequence[Any],
    ) -> list[nx.Graph[Any]]:
        """Decode a sparse / dense graph carrier (or a sequence of them)
        into a flat list of NetworkX graphs.

        Both :class:`tmgg.data.datasets.graph_types.GraphState` and
        :class:`tmgg.data.datasets.graph_types.DenseGraphState` expose a
        ``to_networkx_list()`` method; the sparse path (PyG-flat) is
        cheap and avoids a dense detour, the dense path uses the same
        argmax-class adjacency rule that the type's
        :meth:`DenseGraphState.dense_adjacency` enforces. Sequences
        (lists of single-graph batches, the format produced by
        :meth:`tmgg.data.data_modules.base_data_module.BaseGraphDataModule.get_reference_graphs`)
        are flattened by concatenating each element's
        ``to_networkx_list()``. Plain :class:`networkx.Graph` inputs
        pass through unchanged so the legacy nx-only test paths and
        out-of-tree consumers keep working.

        Parameters
        ----------
        samples
            One of:

            - a single batched ``GraphState`` / ``DenseGraphState``
              (batch dim ≥ 1);
            - a sequence of per-graph ``GraphState`` / ``DenseGraphState``
              carriers (each typically has a leading batch dim of 1);
            - a sequence of pre-decoded ``networkx.Graph``.

        Returns
        -------
        list[networkx.Graph]
            One simple, undirected graph per input row.
        """
        # Single batched carrier (sparse or dense): defer to its own
        # to_networkx_list() — both implementations honour the
        # canonical class-0 = no-edge convention via dense_adjacency.
        # The isinstance against Sequence-likes (list / tuple) keeps the
        # branch order unambiguous: a list of carriers also has
        # ``to_networkx_list`` on its elements but not on itself.
        to_networkx_list_fn = getattr(samples, "to_networkx_list", None)
        if to_networkx_list_fn is not None and not isinstance(samples, list | tuple):
            return to_networkx_list_fn()

        # Sequence form. Three shapes coexist post-Wave 4:
        # - per-graph ``DenseGraphState`` from
        #   ``BaseGraphDataModule.get_reference_graphs`` (batch dim 1);
        # - per-graph ``GraphState`` from sampler chunking;
        # - legacy ``networkx.Graph`` from pre-Wave-4 tests / CLI paths.
        out: list[nx.Graph[Any]] = []
        for item in samples:  # pyright: ignore[reportGeneralTypeIssues]
            item_fn = getattr(item, "to_networkx_list", None)
            if item_fn is not None:
                out.extend(item_fn())
            elif hasattr(item, "number_of_nodes"):
                # Already an nx.Graph — pass through.
                out.append(item)
            else:
                raise TypeError(
                    "GraphEvaluator._to_networkx_list received an item with "
                    "neither ``to_networkx_list`` nor ``number_of_nodes``: "
                    f"{type(item).__name__}. Expected GraphState, "
                    "DenseGraphState, or networkx.Graph."
                )
        return out

    def evaluate(
        self,
        refs: GraphState | DenseGraphState | Sequence[Any],
        generated: GraphState | DenseGraphState | Sequence[Any],
    ) -> EvaluationResults | None:
        """Compute all evaluation metrics.

        Returns ``None`` if fewer than 2 reference graphs or fewer than 2
        generated graphs remain after truncation (MMD needs at least 2
        samples per distribution).

        Both ``refs`` and ``generated`` are truncated to ``eval_num_samples``
        before metric computation, so callers may pass all available graphs
        without pre-filtering.

        Parameters
        ----------
        refs
            Reference graphs as either a batched
            :class:`tmgg.data.datasets.graph_types.GraphState` /
            :class:`tmgg.data.datasets.graph_types.DenseGraphState`,
            a sequence of per-graph carriers (batch dim 1; the format
            emitted by
            :meth:`tmgg.data.data_modules.base_data_module.BaseGraphDataModule.get_reference_graphs`),
            or a sequence of pre-decoded :class:`networkx.Graph`. The
            evaluator decodes all carriers via
            :meth:`_to_networkx_list` for the MMD / structural metric
            pipeline; categorical-class indices ride along on the
            resulting graphs as ``x_class`` / ``e_class`` attributes
            (see :meth:`GraphState.to_networkx`) for downstream
            consumers that want them.
        generated
            Generated graphs in the same format.

        Returns
        -------
        EvaluationResults or None
            All metrics, or None if insufficient graphs after truncation.
        """
        nx_refs_full = self._to_networkx_list(refs)
        nx_generated_full = self._to_networkx_list(generated)

        nx_refs = nx_refs_full[: self.eval_num_samples]
        nx_generated = nx_generated_full[: self.eval_num_samples]

        if len(nx_refs) < 2:
            return None

        if len(nx_generated) < 2:
            return None

        # --- Core MMD metrics (always available) ---
        mmd_results = compute_mmd_metrics(
            nx_refs,
            nx_generated,
            kernel=self.kernel,
            sigma=self.sigma,
            degree_sigma=self.degree_sigma,
            clustering_sigma=self.clustering_sigma,
            spectral_sigma=self.spectral_sigma,
        )

        # --- Orbit MMD (requires orca, skippable) ---
        orbit_mmd: float | None = None
        if "orbit" not in self.skip_metrics and _ORCA_AVAILABLE:
            orbit_mmd = compute_orbit_mmd(nx_refs, nx_generated, kernel=self.kernel)

        # --- SBM accuracy (requires graph-tool, skippable) ---
        sbm_accuracy: float | None = None
        if "sbm" not in self.skip_metrics and _GRAPH_TOOL_AVAILABLE:
            sbm_accuracy = compute_sbm_accuracy(
                nx_generated,
                p_intra=self.p_intra,
                p_inter=self.p_inter,
                refinement_steps=self.sbm_refinement_steps,
            )

        # --- Planarity (skippable) ---
        planarity_accuracy: float | None = None
        if "planarity" not in self.skip_metrics:
            planarity_accuracy = compute_planarity_accuracy(nx_generated)

        # --- Uniqueness (skippable) ---
        uniqueness: float | None = None
        if "uniqueness" not in self.skip_metrics:
            uniqueness = compute_uniqueness(nx_generated)

        # --- Novelty (needs train_graphs from constructor, skippable) ---
        novelty: float | None = None
        if "novelty" not in self.skip_metrics and self._train_graphs_set:
            novelty = compute_novelty(nx_generated, self.train_graphs)

        # --- Block-structure metrics (Stage 3 telemetry, skippable) ---
        # ``modularity_q``, ``spectral_gap_l2``, and the empirical
        # (p_in, p_out) recovered from a 2-block spectral partition.
        # All four are dataset-agnostic; they smooth out the
        # binary-saturated ``sbm_accuracy`` signal.
        block_metrics: dict[str, float | None] = {
            "modularity_q": None,
            "spectral_gap_l2": None,
            "empirical_p_in": None,
            "empirical_p_out": None,
        }
        if "block_structure" not in self.skip_metrics:
            block_metrics = compute_block_structure_metrics(nx_generated)

        return EvaluationResults(
            degree_mmd=mmd_results.degree_mmd,
            clustering_mmd=mmd_results.clustering_mmd,
            spectral_mmd=mmd_results.spectral_mmd,
            orbit_mmd=orbit_mmd,
            sbm_accuracy=sbm_accuracy,
            planarity_accuracy=planarity_accuracy,
            uniqueness=uniqueness,
            novelty=novelty,
            modularity_q=block_metrics["modularity_q"],
            spectral_gap_l2=block_metrics["spectral_gap_l2"],
            empirical_p_in=block_metrics["empirical_p_in"],
            empirical_p_out=block_metrics["empirical_p_out"],
        )
