"""Maximum Mean Discrepancy (MMD) metrics for graph generation evaluation.

This module computes distributional distance between reference and generated
graphs using MMD on graph-theoretic properties. Implementation is aligned
with DiGress (Vignac et al., 2023) for direct comparability:

- Spectral statistic uses normalized Laplacian eigenvalues (200 bins, fixed range)
- Clustering histogram uses 100 bins
- Default kernel is gaussian_tv: k(x,y) = exp(-TV(x,y)² / 2σ²)
- MMD uses unbiased estimator (unique pairs, no self-pairs)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, cast

import networkx as nx
import numpy as np
import ot
import torch
from scipy.linalg import toeplitz


@dataclass
class MMDResults:
    """Results from MMD² computation (DiGress-compatible).

    All fields hold **squared MMD values** (V-statistic biased estimator),
    not square-root MMD distances. This matches the GraphRNN/GRAN
    convention that DiGress and HiGen also follow — see
    ``docs/eval/mmd-units-and-protocol.md`` for why this matters when
    comparing to published anchors.

    Attributes
    ----------
    degree_mmd
        MMD² on degree histograms.
    clustering_mmd
        MMD² on clustering coefficient histograms.
    spectral_mmd
        MMD² on normalized Laplacian eigenvalue histograms (DiGress "spectre").
    """

    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "degree_mmd": self.degree_mmd,
            "clustering_mmd": self.clustering_mmd,
            "spectral_mmd": self.spectral_mmd,
        }


def adjacency_to_networkx(
    adjacency: torch.Tensor | np.ndarray,
    num_nodes: int | None = None,
) -> nx.Graph[Any]:
    """Convert adjacency matrix to NetworkX graph.

    Parameters
    ----------
    adjacency
        Adjacency matrix of shape ``(n, n)``.
    num_nodes
        If provided, extract the top-left ``(num_nodes, num_nodes)`` submatrix
        before conversion. Use this to strip zero-padding from batched PyG
        graphs that were padded to a common size.

    Returns
    -------
    nx.Graph
        Undirected graph.
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.detach().cpu().numpy()

    if num_nodes is not None:
        adjacency = adjacency[:num_nodes, :num_nodes]

    # Threshold to binary for probabilistic adjacency
    A = (adjacency > 0.5).astype(float)
    return nx.from_numpy_array(A)


def _histogram_pmf(
    values: np.ndarray | list[int] | list[float],
    bins: np.ndarray | int,
    range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Bin ``values`` and normalise to a PMF (sum-to-one histogram).

    Parameters
    ----------
    values
        Observed values to histogram.
    bins
        Bin edges (array) or bin count (int, requires ``range``).
    range
        ``(low, high)`` passed to ``np.histogram``. Required when
        ``bins`` is an int.

    Returns
    -------
    np.ndarray
        Float64 PMF. Zero vector when ``values`` is empty.
    """
    hist, _ = np.histogram(values, bins=bins, range=range, density=False)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist.astype(np.float64)


def compute_degree_histogram(
    G: nx.Graph[Any], max_degree: int | None = None
) -> np.ndarray:
    """Compute normalised degree histogram.

    Parameters
    ----------
    G
        NetworkX graph.
    max_degree
        Maximum degree to include. If ``None``, uses the maximum
        observed degree.

    Returns
    -------
    np.ndarray
        PMF over degree bins. Zero vector for an empty graph.
    """
    # Cast required: networkx stubs expose DegreeView incompletely
    degree_view = G.degree()
    degrees: list[int] = [int(d) for _, d in cast(Any, degree_view)]
    if len(degrees) == 0:
        return np.zeros(1, dtype=np.float64)

    max_d = max_degree if max_degree is not None else max(degrees)
    # Integer bins [0, 1, ..., max_d+1): one bin per degree value.
    # Range is data-dependent because degree support varies across graph
    # families (e.g. max_d=2 for paths, max_d=n-1 for stars).
    degree_bins = np.arange(0, max_d + 2)
    return _histogram_pmf(degrees, bins=degree_bins)


def compute_clustering_histogram(G: nx.Graph[Any], num_bins: int = 100) -> np.ndarray:
    """Compute histogram of clustering coefficients (DiGress style).

    Parameters
    ----------
    G
        NetworkX graph.
    num_bins
        Number of bins. DiGress uses 100.

    Returns
    -------
    np.ndarray
        PMF over clustering-coefficient bins. Zero vector for an empty graph.
    """
    if G.number_of_nodes() == 0:
        return np.zeros(num_bins, dtype=np.float64)

    # Cast required: networkx stubs type nx.clustering incompletely
    clustering_dict = cast(dict[Any, float], nx.clustering(G))
    coefficients = list(clustering_dict.values())
    # Clustering coefficients are bounded to [0, 1] by definition,
    # so a fixed range with uniform bins is appropriate.
    bin_edges = np.linspace(0, 1, num_bins + 1)
    return _histogram_pmf(coefficients, bins=bin_edges)


def compute_spectral_histogram(G: nx.Graph[Any], num_bins: int = 200) -> np.ndarray:
    """Compute histogram of normalised Laplacian eigenvalues (DiGress spectre style).

    Parameters
    ----------
    G
        NetworkX graph.
    num_bins
        Number of bins. DiGress uses 200.

    Returns
    -------
    np.ndarray
        PMF over eigenvalue bins. Zero vector for an empty graph.
    """
    if G.number_of_nodes() == 0:
        return np.zeros(num_bins, dtype=np.float64)

    laplacian = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    # Normalised Laplacian eigenvalues lie in [0, 2] by theory.  The lower
    # bound -1e-5 accommodates floating-point eigenvalues that land slightly
    # negative.  Range and bin count follow upstream DiGress (spectre metric).
    spectral_range = (-1e-5, 2.0)
    return _histogram_pmf(eigenvalues, bins=num_bins, range=spectral_range)


@dataclass
class GraphStatistics:
    """Statistics computed from a collection of graphs (DiGress-compatible).

    Attributes
    ----------
    degree
        List of degree histograms.
    clustering
        List of clustering coefficient histograms.
    spectral
        List of normalized Laplacian eigenvalue histograms (DiGress "spectre").
    """

    degree: list[np.ndarray]
    clustering: list[np.ndarray]
    spectral: list[np.ndarray]


def compute_graph_statistics(
    graphs: list[nx.Graph[Any]],
    max_workers: int | None = None,
) -> GraphStatistics:
    """Compute histograms of graph properties for a collection of graphs.

    Parameters
    ----------
    graphs
        List of NetworkX graphs.
    max_workers
        Maximum number of parallel workers.

    Returns
    -------
    GraphStatistics
        Statistics for all graphs.
    """

    def compute_single(
        G: nx.Graph[Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            compute_degree_histogram(G),
            compute_clustering_histogram(G),
            compute_spectral_histogram(G),
        )

    if max_workers == 1 or len(graphs) < 4:
        # Single-threaded for small batches
        results = [compute_single(g) for g in graphs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compute_single, graphs))

    degree = [r[0] for r in results]
    clustering = [r[1] for r in results]
    spectral = [r[2] for r in results]

    return GraphStatistics(
        degree=degree,
        clustering=clustering,
        spectral=spectral,
    )


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute Gaussian kernel between two histograms.

    Parameters
    ----------
    x
        First histogram.
    y
        Second histogram.
    sigma
        Kernel bandwidth.

    Returns
    -------
    float
        Kernel value.
    """
    # Pad to same length
    max_len = max(len(x), len(y))
    x_padded = np.zeros(max_len)
    y_padded = np.zeros(max_len)
    x_padded[: len(x)] = x
    y_padded[: len(y)] = y

    # Normalize to valid probability distributions
    x_sum = x_padded.sum()
    y_sum = y_padded.sum()
    if x_sum > 0:
        x_padded = x_padded / x_sum
    if y_sum > 0:
        y_padded = y_padded / y_sum

    # Compute L2 distance and Gaussian kernel
    dist = np.linalg.norm(x_padded - y_padded)
    return float(np.exp(-(dist**2) / (2 * sigma**2)))


def gaussian_tv_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel with Total Variation distance (DiGress style).

    Computes k(x,y) = exp(-TV(x,y)² / 2σ²), matching DiGress's gaussian_tv kernel.

    Parameters
    ----------
    x
        First histogram.
    y
        Second histogram.
    sigma
        Kernel bandwidth.

    Returns
    -------
    float
        Kernel value in (0, 1].
    """
    # Pad to same length. No PMF normalisation here: matches upstream
    # ``gaussian_tv`` (digress-upstream-readonly/src/analysis/dist_helper.py).
    # Normalisation is the caller's job via ``compute_mmd(is_hist=True)``.
    max_len = max(len(x), len(y))
    x_padded = np.zeros(max_len)
    y_padded = np.zeros(max_len)
    x_padded[: len(x)] = x
    y_padded[: len(y)] = y

    tv = 0.5 * np.sum(np.abs(x_padded - y_padded))
    return float(np.exp(-(tv**2) / (2 * sigma**2)))


def gaussian_emd_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
    distance_scaling: float = 1.0,
) -> float:
    """Gaussian kernel with Earth Mover's Distance (DiGress style).

    Uses optimal transport distance instead of TV or L2, providing
    sensitivity to distributional shape that TV ignores. Matches the
    ``gaussian_emd`` kernel from ``analysis/dist_helper.py`` in the
    upstream DiGress codebase.

    Parameters
    ----------
    x
        First histogram.
    y
        Second histogram.
    sigma
        Kernel bandwidth.
    distance_scaling
        Scale factor for the Toeplitz distance matrix. Upstream uses
        ``bins`` (e.g. 100 for clustering) to normalise bin distances.

    Returns
    -------
    float
        Kernel value in (0, 1].
    """
    # Pad to same length
    support_size = max(len(x), len(y))
    x_padded = np.zeros(support_size)
    y_padded = np.zeros(support_size)
    x_padded[: len(x)] = x
    y_padded[: len(y)] = y

    # Normalize to PMF
    x_sum = x_padded.sum()
    y_sum = y_padded.sum()
    if x_sum > 0:
        x_padded = x_padded / x_sum
    if y_sum > 0:
        y_padded = y_padded / y_sum

    # Toeplitz distance matrix (DiGress convention)
    d_mat = toeplitz(range(support_size)).astype(float) / distance_scaling

    emd_val = cast(float, ot.emd2(x_padded, y_padded, d_mat))  # pyright: ignore[reportUnknownMemberType]  # pot library has no type stubs
    return float(np.exp(-(emd_val * emd_val) / (2 * sigma * sigma)))


def compute_mmd(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel: Literal["gaussian", "gaussian_tv", "gaussian_emd"] = "gaussian_tv",
    sigma: float = 1.0,
    distance_scaling: float = 1.0,
    is_hist: bool = True,
) -> float:
    """Compute the **biased V-statistic squared MMD** between two sample sets.

    Returns the V-statistic estimator of MMD² (squared MMD), not the
    square-root MMD distance:

    ``MMD² = (1/n²) Σ_{i,j} k(x_i,x_j) + (1/m²) Σ_{i,j} k(y_i,y_j)
            - (2/(n·m)) Σ_{i,j} k(x_i,y_j)``

    where ``i==j`` terms are *included* in the within-set sums (this is
    the biased estimator; the unbiased U-statistic excludes them and
    would shift reported numbers by O(1/n)). Matches upstream DiGress's
    MMD implementation (``src/analysis/dist_helper.py::compute_mmd``)
    bit-for-bit, which is why the function name is unsuffixed "mmd"
    rather than "mmd_squared" — the GraphRNN/GRAN/SPECTRE/DiGress/HiGen
    literature uniformly calls this quantity "MMD" without the square,
    and we follow that convention so values are directly comparable to
    published anchors.

    See ``docs/eval/mmd-units-and-protocol.md`` for the unit convention
    and how it interacts with DiGress's MMD-ratio reporting and HiGen's
    raw-MMD anchors.

    Parameters
    ----------
    samples1
        First set of histogram samples.
    samples2
        Second set of histogram samples.
    kernel
        Kernel type: ``"gaussian"`` (L2-based), ``"gaussian_tv"``
        (TV-based, DiGress default), or ``"gaussian_emd"`` (Earth
        Mover's Distance via POT).
    sigma
        Bandwidth for kernel.
    distance_scaling
        Scale factor for the EMD distance matrix. Only used with
        ``"gaussian_emd"``; upstream DiGress uses ``bins`` (e.g. 100
        for clustering).
    is_hist
        If ``True`` (default), PMF-normalise each sample (``s / sum(s)``)
        before computing kernel values. Matches upstream's contract:
        histogram metrics (degree/clustering/spectral) use ``is_hist=True``;
        raw-count metrics (orbit, motif graphlet counts) must pass
        ``is_hist=False`` so magnitude information is preserved.

    Returns
    -------
    float
        Biased squared-MMD value, non-negative. Despite being labelled
        "MMD" in the literature and in the W&B logging keys, this is
        MMD², not the square-root MMD distance.
    """
    if len(samples1) < 2 or len(samples2) < 2:
        return float("inf")

    if is_hist:
        samples1 = [s / (np.sum(s) + 1e-6) for s in samples1]
        samples2 = [s / (np.sum(s) + 1e-6) for s in samples2]

    def kernel_fn(x: np.ndarray, y: np.ndarray) -> float:
        if kernel == "gaussian":
            return gaussian_kernel(x, y, sigma)
        if kernel == "gaussian_emd":
            return gaussian_emd_kernel(x, y, sigma, distance_scaling)
        return gaussian_tv_kernel(x, y, sigma)

    # V-statistic: enumerate all ``n^2`` ordered pairs including ``i==j``.
    # ``k(x_i, x_i) = 1`` for all stationary kernels we use, so the
    # diagonal contributes ``n / n^2 = 1/n`` per within-set sum — the
    # exact bias absorbed by the V- vs. U-statistic distinction.
    k11_values = [kernel_fn(x, y) for x in samples1 for y in samples1]
    k11 = float(np.mean(k11_values)) if k11_values else 0.0

    k22_values = [kernel_fn(x, y) for x in samples2 for y in samples2]
    k22 = float(np.mean(k22_values)) if k22_values else 0.0

    # Cross-set sum is identical between U- and V-statistics; left as-is.
    k12_values = [kernel_fn(x, y) for x in samples1 for y in samples2]
    k12 = float(np.mean(k12_values)) if k12_values else 0.0

    return max(0.0, k11 + k22 - 2 * k12)


def compute_mmd_metrics(
    ref_graphs: list[nx.Graph[Any]],
    gen_graphs: list[nx.Graph[Any]],
    kernel: Literal["gaussian", "gaussian_tv", "gaussian_emd"] = "gaussian_tv",
    sigma: float = 1.0,
    max_workers: int | None = None,
    *,
    degree_sigma: float | None = None,
    clustering_sigma: float | None = None,
    spectral_sigma: float | None = None,
) -> MMDResults:
    """Compute MMD² metrics between reference and generated graph distributions.

    Each returned value is a biased V-statistic squared MMD — see
    :func:`compute_mmd` and ``docs/eval/mmd-units-and-protocol.md``.

    Parameters
    ----------
    ref_graphs
        List of reference NetworkX graphs.
    gen_graphs
        List of generated NetworkX graphs.
    kernel
        Kernel type: ``"gaussian"`` (L2-based), ``"gaussian_tv"``
        (TV-based, DiGress default), or ``"gaussian_emd"`` (Earth
        Mover's Distance via POT).
    sigma
        Fallback bandwidth applied when a per-metric override is ``None``.
    max_workers
        Maximum number of parallel workers for statistics computation.
    degree_sigma, clustering_sigma, spectral_sigma
        Per-metric kernel bandwidths. When ``None`` each falls back to
        ``sigma``. Upstream DiGress uses ``clustering_sigma=0.1`` while
        leaving the others at 1.0; this matches the upstream
        ``dist_helper.gaussian_tv`` call sites.

    Returns
    -------
    MMDResults
        Squared-MMD values (V-statistic) for each graph statistic.
    """
    ref_stats = compute_graph_statistics(ref_graphs, max_workers)
    gen_stats = compute_graph_statistics(gen_graphs, max_workers)

    degree_s = sigma if degree_sigma is None else degree_sigma
    clustering_s = sigma if clustering_sigma is None else clustering_sigma
    spectral_s = sigma if spectral_sigma is None else spectral_sigma

    # ``gaussian_emd`` requires distance_scaling proportional to the
    # number of bins for the clustering histogram; keep that here so the
    # EMD path is still correct when a caller opts into it. TV/L2 do not
    # use ``distance_scaling``.
    degree_mmd = compute_mmd(ref_stats.degree, gen_stats.degree, kernel, degree_s)
    clustering_mmd = compute_mmd(
        ref_stats.clustering,
        gen_stats.clustering,
        kernel,
        sigma=clustering_s,
        distance_scaling=100.0 if kernel == "gaussian_emd" else 1.0,
    )
    spectral_mmd = compute_mmd(
        ref_stats.spectral, gen_stats.spectral, kernel, spectral_s
    )

    return MMDResults(
        degree_mmd=degree_mmd,
        clustering_mmd=clustering_mmd,
        spectral_mmd=spectral_mmd,
    )
