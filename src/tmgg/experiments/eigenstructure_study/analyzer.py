"""Spectral analysis of collected eigenstructure data.

Provides analysis functions for studying band gaps, eigenvalue distributions,
eigenvector coherence, and subspace properties of graph collections.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from tmgg.utils.spectral.spectral_deltas import (
    compute_alg_connectivity_delta,
)
from tmgg.utils.spectral.spectral_deltas import (
    compute_eigengap_delta as _eigengap_delta_primitive,
)

from .storage import iter_batches, load_decomposition_batch, load_manifest


@dataclass
class SpectralAnalysisResult:
    """Container for spectral analysis results."""

    dataset_name: str
    num_graphs: int

    # Spectral gap statistics (lambda_max - lambda_{max-1} for adjacency)
    spectral_gap_mean: float
    spectral_gap_std: float
    spectral_gap_min: float
    spectral_gap_max: float

    # Algebraic connectivity / band gap (lambda_2 for Laplacian)
    algebraic_connectivity_mean: float
    algebraic_connectivity_std: float
    algebraic_connectivity_min: float
    algebraic_connectivity_max: float

    # Eigenvalue distribution statistics
    eigenvalue_entropy_adj: float
    eigenvalue_entropy_lap: float

    # Eigenvector coherence (localization measure)
    coherence_mean: float
    coherence_std: float

    # Effective rank (participation ratio of eigenvalues)
    effective_rank_adj_mean: float
    effective_rank_lap_mean: float


def compute_spectral_gap(eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Compute spectral gap: difference between largest two eigenvalues.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Sorted eigenvalues of shape (batch, n), ascending order.

    Returns
    -------
    torch.Tensor
        Spectral gaps of shape (batch,).

    Notes
    -----
    For adjacency matrices, the spectral gap relates to mixing time
    and expansion properties of the graph.
    """
    return eigenvalues[:, -1] - eigenvalues[:, -2]


def compute_algebraic_connectivity(lap_eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Compute algebraic connectivity (Fiedler value): second smallest Laplacian eigenvalue.

    Parameters
    ----------
    lap_eigenvalues : torch.Tensor
        Sorted Laplacian eigenvalues of shape (batch, n), ascending order.

    Returns
    -------
    torch.Tensor
        Algebraic connectivity of shape (batch,).

    Notes
    -----
    For connected graphs, lambda_1 = 0 and lambda_2 > 0. The algebraic
    connectivity measures robustness to edge removal and bounds
    isoperimetric properties.
    """
    return lap_eigenvalues[:, 1]


def compute_eigenvector_coherence(eigenvectors: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvector coherence: max squared component magnitude.

    Parameters
    ----------
    eigenvectors : torch.Tensor
        Eigenvectors of shape (batch, n, n).

    Returns
    -------
    torch.Tensor
        Coherence values of shape (batch,).

    Notes
    -----
    Low coherence indicates delocalized eigenvectors spread across all nodes.
    High coherence indicates localized eigenvectors concentrated on few nodes.
    For random graphs, coherence scales as O(log(n)/n).
    """
    squared = eigenvectors**2
    max_per_vec = squared.max(dim=1).values  # (batch, n)
    return max_per_vec.max(dim=1).values  # (batch,)


def compute_eigenvalue_entropy(eigenvalues: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute entropy of normalized eigenvalue distribution.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Eigenvalues of shape (batch, n).
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    float
        Mean entropy across all graphs.

    Notes
    -----
    High entropy indicates spread-out eigenvalues; low entropy indicates
    clustering. Normalized by log(n) to be scale-invariant.
    """
    abs_eigs = eigenvalues.abs()
    # Normalize to probability distribution per graph
    probs = abs_eigs / (abs_eigs.sum(dim=-1, keepdim=True) + eps)
    entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    return entropy.mean().item()


def compute_effective_rank(eigenvalues: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute effective rank via participation ratio of eigenvalues.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Eigenvalues of shape (batch, n).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    float
        Mean effective rank across all graphs.

    Notes
    -----
    The participation ratio is (sum |lambda_i|)^2 / sum |lambda_i|^2,
    which equals n for uniform eigenvalues and 1 for a single nonzero eigenvalue.
    """
    abs_eigs = eigenvalues.abs()
    sum_squared = (abs_eigs**2).sum(dim=-1)
    sum_abs_squared = (abs_eigs.sum(dim=-1)) ** 2
    pr = sum_abs_squared / (sum_squared + eps)
    return pr.mean().item()


def compute_eigengap_delta(
    eigenvalues_clean: torch.Tensor,
    eigenvalues_other: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute change in spectral gap between clean and other (noisy/denoised).

    Thin wrapper around :func:`tmgg.utils.spectral.spectral_deltas.compute_eigengap_delta`
    that also returns the absolute difference for callers that need it.

    Parameters
    ----------
    eigenvalues_clean : torch.Tensor
        Clean graph eigenvalues, shape (batch, n), ascending order.
    eigenvalues_other : torch.Tensor
        Noisy or denoised eigenvalues, same shape.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with ``absolute`` and ``relative`` delta tensors,
        shape ``(batch,)``. Positive relative delta means the gap increased.
    """
    delta_absolute = compute_spectral_gap(eigenvalues_other) - compute_spectral_gap(
        eigenvalues_clean
    )
    delta_relative = _eigengap_delta_primitive(eigenvalues_clean, eigenvalues_other)
    return {"absolute": delta_absolute, "relative": delta_relative}


def compute_algebraic_connectivity_delta(
    lap_eigenvalues_clean: torch.Tensor,
    lap_eigenvalues_other: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute change in Fiedler value (lambda_2 of Laplacian).

    Thin wrapper around :func:`tmgg.utils.spectral.spectral_deltas.compute_alg_connectivity_delta`
    that also returns the absolute difference for callers that need it.

    Parameters
    ----------
    lap_eigenvalues_clean : torch.Tensor
        Clean graph Laplacian eigenvalues, shape (batch, n), ascending order.
    lap_eigenvalues_other : torch.Tensor
        Noisy or denoised Laplacian eigenvalues, same shape.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with ``absolute`` and ``relative`` delta tensors,
        shape ``(batch,)``. Positive relative delta means connectivity increased.
    """
    delta_absolute = compute_algebraic_connectivity(
        lap_eigenvalues_other
    ) - compute_algebraic_connectivity(lap_eigenvalues_clean)
    delta_relative = compute_alg_connectivity_delta(
        lap_eigenvalues_clean, lap_eigenvalues_other
    )
    return {"absolute": delta_absolute, "relative": delta_relative}


def compute_principal_angles(
    V1: torch.Tensor, V2: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Compute principal angles between subspaces spanned by top-k eigenvectors.

    Parameters
    ----------
    V1, V2 : torch.Tensor
        Eigenvector matrices of shape (n, n).
    k : int
        Number of top eigenvectors to consider.

    Returns
    -------
    torch.Tensor
        Principal angles in radians, shape (k,).

    Notes
    -----
    Principal angles measure the alignment between two subspaces. The first
    angle is 0 if the subspaces share a common direction, and pi/2 if they
    are orthogonal.
    """
    # Top-k eigenvectors by largest eigenvalue (last k columns, ascending order)
    U1 = V1[:, -k:]
    U2 = V2[:, -k:]

    # SVD of cross-correlation gives cosines of principal angles
    _, sigmas, _ = torch.linalg.svd(U1.T @ U2)
    sigmas = torch.clamp(sigmas, -1.0, 1.0)
    return torch.acos(sigmas)


def compute_eigenvalue_covariance(eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance matrix of eigenvalues across graphs.

    Given N graphs each with k eigenvalues, this computes the k×k sample
    covariance matrix showing how eigenvalue positions covary across the
    population. High off-diagonal values indicate eigenvalue positions
    that tend to move together.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Eigenvalues of shape (N, k) where N = num_graphs, k = num_eigenvalues.

    Returns
    -------
    torch.Tensor
        Sample covariance matrix of shape (k, k).

    Notes
    -----
    Uses Bessel correction (N-1 denominator) for unbiased sample covariance.
    For random graphs from the same generative model, off-diagonal terms
    reveal structural dependencies in the spectrum.
    """
    if eigenvalues.shape[0] < 2:
        raise ValueError("Need at least 2 graphs to compute covariance")

    # Center the data
    centered = eigenvalues - eigenvalues.mean(dim=0, keepdim=True)
    # Compute sample covariance: (X^T X) / (N - 1)
    return (centered.T @ centered) / (eigenvalues.shape[0] - 1)


def compute_covariance_summary(cov: torch.Tensor) -> dict[str, float]:
    """
    Compute summary statistics for an eigenvalue covariance matrix.

    Parameters
    ----------
    cov : torch.Tensor
        Covariance matrix of shape (k, k).

    Returns
    -------
    dict[str, float]
        Summary statistics including:
        - frobenius_norm: Overall magnitude of covariance
        - trace: Total variance (sum of eigenvalue variances)
        - condition_number: Numerical conditioning of the covariance
        - off_diagonal_sum: Sum of off-diagonal terms (total covariation)
        - off_diagonal_ratio: Fraction of total variance in cross-terms
        - max_eigenvalue: Largest principal component of the covariance
        - min_eigenvalue: Smallest principal component
    """
    trace = torch.trace(cov).item()
    off_diag = cov.sum().item() - trace

    # Eigenvalues of covariance matrix (principal component variances)
    cov_eigenvalues = torch.linalg.eigvalsh(cov)

    return {
        "frobenius_norm": torch.norm(cov, p="fro").item(),
        "trace": trace,
        "condition_number": torch.linalg.cond(cov).item(),
        "off_diagonal_sum": off_diag,
        "off_diagonal_ratio": off_diag / (trace + 1e-10) if trace > 0 else 0.0,
        "max_eigenvalue": cov_eigenvalues.max().item(),
        "min_eigenvalue": cov_eigenvalues.min().item(),
    }


@dataclass
class CovarianceResult:
    """Container for eigenvalue covariance analysis results."""

    matrix_type: str  # "adjacency" or "laplacian"
    num_graphs: int
    num_eigenvalues: int

    # Covariance matrix stored as nested list for JSON serialization
    covariance_matrix: list[list[float]]

    # Summary statistics
    frobenius_norm: float
    trace: float
    condition_number: float
    off_diagonal_sum: float
    off_diagonal_ratio: float
    max_eigenvalue: float
    min_eigenvalue: float


def compute_B_invariants(B: torch.Tensor) -> torch.Tensor:
    """Frame-invariant summaries of the clean-signal projection ``B``.

    Returns a stack of per-graph invariants with shape ``(N, k + 2)``:

    * column 0 — ``trace(B_i)`` (sum of top-*k* Ritz values of ``A_i`` in
      the noisy basis ``V̂_{k,i}``);
    * column 1 — ``||B_i||_F²`` (Frobenius-squared mass);
    * columns 2..k+1 — sorted eigenvalues of ``B_i`` in ascending order.

    All three are invariant under orthogonal conjugation
    ``B → R^T B R``, so these summaries are well-defined regardless of
    which frame ``V̂_{k,i}`` is expressed in. The improvement-gap
    surrogate computed on these invariants is a frame-free cross-check
    for the matrix-valued surrogate on eq. (18): if both agree
    directionally, the claim survives the frame-convention ambiguity
    flagged in the reviewer-2 audit.

    Parameters
    ----------
    B : torch.Tensor
        Per-graph projection tensors of shape ``(N, k, k)``.

    Returns
    -------
    torch.Tensor
        Invariant summary tensor of shape ``(N, k + 2)``.
    """
    if B.ndim != 3 or B.shape[1] != B.shape[2]:
        raise ValueError(f"B must have shape (N, k, k); got {tuple(B.shape)}")
    B = B.float()
    trace = torch.diagonal(B, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)  # (N, 1)
    frob_sq = B.pow(2).sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)  # (N, 1)
    # Symmetrise in case of numerical asymmetry, then eigvalsh.
    B_sym = 0.5 * (B + B.transpose(-2, -1))
    eigvals = torch.linalg.eigvalsh(B_sym)  # (N, k), ascending
    return torch.cat([trace, frob_sq, eigvals], dim=-1)


@dataclass(frozen=True)
class ImprovementGapResult:
    """Improvement-gap surrogate for one ``(noise_level, k)`` pair.

    Quantifies the between-graph variance of the conditional mean
    ``E[B | Λ̃_k]`` where ``B = V̂_k^T A V̂_k`` projects the clean adjacency
    into the noisy top-*k* eigenbasis. See
    ``_NeurIPS_2026__Understanding_Graph_Denoising.pdf`` eq. (18):

    ``ℓ_lin − ℓ_f = tr(Cov(E[vec(B) | Λ̃_k])) = E‖E[B|Λ̃_k] − E[B]‖²_F``

    The quantity ``fve = g_hat / trace_cov_B`` is the **fraction of
    variance explained** (FVE) of ``B`` by the noisy top-*k* eigenvalues —
    the population R² of the Bayes-optimal predictor. Close to 1 means
    the eigenvalue spectrum captures almost all of ``B``'s variability;
    close to 0 means the gap is negligible and a linear denoiser is
    already near-optimal.

    Attributes
    ----------
    noise_level : float
        Noise magnitude ``ε`` at which the surrogate was computed.
    k : int
        Top-*k* eigenspace dimension.
    estimator : str
        Estimator name, ``"knn"`` or ``"bin"``.
    num_graphs : int
        Number of graphs used in the estimate.
    g_hat : float
        Surrogate estimate ``(1/N) Σ ‖B̂_i − μ_B‖²_F`` of the improvement
        gap, where ``B̂_i`` is the conditional-mean estimate.
    trace_cov_B : float
        Total trace of ``Cov(vec(B))`` = ``(1/N) Σ ‖B_i − μ_B‖²_F``.
        Provides the denominator of the FVE.
    fve : float
        Fraction of variance explained, ``g_hat / trace_cov_B`` (set to
        0.0 if ``trace_cov_B == 0``).
    knn_neighbours : int | None
        Number of neighbours used for the kNN estimator (``None`` for
        ``"bin"``).
    num_bins : int | None
        Number of quantile bins used for the binning estimator (``None``
        for ``"knn"``).
    """

    noise_level: float
    k: int
    estimator: str
    num_graphs: int
    g_hat: float
    trace_cov_B: float
    fve: float
    knn_neighbours: int | None = None
    num_bins: int | None = None
    # Which per-graph target was used: the matrix ``B`` (default,
    # literally eq. (18)) or its frame-invariant summaries (sanity check
    # that survives the frame-convention ambiguity).
    target: str = "matrix"
    # Which conditioning feature was used: the full top-*k* noisy
    # eigenvalues (default, matches kNN's original feature space) or the
    # 1-D spectral gap (so kNN and binning see the same information).
    conditioning: str = "top_k_eigenvalues"
    # Whether features were permuted — the permutation-null ĝ should
    # drop near zero for a well-calibrated estimator.
    permuted: bool = False

    def to_json_dict(self) -> dict[str, object]:
        """Serialize to the flat JSON layout used by the eigenstructure report."""
        payload: dict[str, object] = {
            "noise_level": self.noise_level,
            "k": self.k,
            "estimator": self.estimator,
            "target": self.target,
            "conditioning": self.conditioning,
            "permuted": self.permuted,
            "num_graphs": self.num_graphs,
            "g_hat": self.g_hat,
            "trace_cov_B": self.trace_cov_B,
            "fve": self.fve,
        }
        if self.knn_neighbours is not None:
            payload["knn_neighbours"] = self.knn_neighbours
        if self.num_bins is not None:
            payload["num_bins"] = self.num_bins
        return payload


def estimate_improvement_gap(
    B: torch.Tensor,
    conditioning_features: torch.Tensor,
    *,
    estimator: str,
    knn_neighbours: int = 10,
    num_bins: int = 4,
    permute_features: bool = False,
    permutation_seed: int = 0,
) -> tuple[float, float, float]:
    """Estimate the improvement-gap surrogate from stacked projections.

    Parameters
    ----------
    B : torch.Tensor
        Per-graph targets. Either a matrix-valued stack of shape
        ``(N, k, k)`` (the canonical ``B`` matrices in eq. (18)) or a
        vector-valued stack of shape ``(N, d)`` (e.g. frame-invariant
        summaries produced by :func:`compute_B_invariants`). The total
        variance is measured as the mean squared Frobenius (resp.
        Euclidean) deviation from the sample mean.
    conditioning_features : torch.Tensor
        Per-graph features used to condition ``E[B | ·]``, shape ``(N, d)``.
        For the canonical kNN estimator this is the sorted top-*k* noisy
        eigenvalues; for ``"bin"`` it is typically a single-column spectral
        gap ``λ_1 − λ_2``.
    estimator : str
        ``"knn"`` — k-nearest-neighbour average in feature space.
        ``"bin"`` — quantile-binned between-group variance.
    knn_neighbours : int
        Neighbourhood size for the kNN estimator (ignored for ``"bin"``).
    num_bins : int
        Bin count for the binning estimator (ignored for ``"knn"``).
    permute_features : bool
        If ``True``, randomly shuffle ``conditioning_features`` along axis 0
        before estimating. Used to compute the permutation-null
        surrogate: a correctly-calibrated estimator should return
        ``g_hat ≈ 0`` (FVE ≈ 0) when the conditioning is decorrelated
        from the targets. A large null ĝ signals finite-sample bias in
        the estimator (usually kNN at small ``N`` / small ``m``).
    permutation_seed : int
        Seed for the permutation RNG when ``permute_features=True``. Kept
        explicit so null runs are reproducible.

    Returns
    -------
    tuple[float, float, float]
        ``(g_hat, trace_cov_B, fve)`` where ``fve`` is the fraction of
        variance explained ``g_hat / trace_cov_B``.

    Notes
    -----
    By the law of total variance the surrogate satisfies
    ``g_hat ≤ trace_cov_B`` up to estimation error. The returned FVE is
    clamped to ``0.0`` when ``trace_cov_B == 0`` to keep the JSON output
    numerically clean.
    """
    if B.ndim not in (2, 3):
        raise ValueError(f"B must have shape (N, k, k) or (N, d); got {tuple(B.shape)}")
    if B.ndim == 3 and B.shape[1] != B.shape[2]:
        raise ValueError(f"Matrix B must be square; got {tuple(B.shape)}")
    num_graphs = B.shape[0]
    if num_graphs < 2:
        raise ValueError("Need at least 2 graphs to estimate the surrogate")

    B = B.float()
    # mean across graphs, keeping shape broadcast-compatible
    mu_B = B.mean(dim=0, keepdim=True)
    deviations = B - mu_B
    if B.ndim == 3:
        # Frobenius-squared deviation per graph, averaged over graphs.
        trace_cov_B = (deviations.pow(2).sum(dim=(-2, -1))).mean().item()
    else:
        trace_cov_B = (deviations.pow(2).sum(dim=-1)).mean().item()

    if permute_features:
        perm_rng = torch.Generator().manual_seed(permutation_seed)
        perm = torch.randperm(num_graphs, generator=perm_rng)
        conditioning_features = conditioning_features[perm]

    # Sum axes for per-graph squared-deviation norms — differs between the
    # matrix (Frobenius) and vector (Euclidean) target shapes.
    sum_dims: tuple[int, ...] = (-2, -1) if B.ndim == 3 else (-1,)

    if estimator == "knn":
        features = conditioning_features.float()
        # cdist needs at least 2-D; promote a 1-D conditioning vector to
        # a single-column matrix so kNN on scalar features (e.g. the 1-D
        # spectral gap) doesn't trip the underlying torch kernel.
        if features.ndim == 1:
            features = features.unsqueeze(1)
        # Pairwise Euclidean distances in feature space, (N, N).
        dists = torch.cdist(features, features)
        # Pick the m+1 smallest (self + m neighbours); drop self below.
        topk = min(knn_neighbours + 1, num_graphs)
        _, neighbour_idx = dists.topk(topk, largest=False)
        # Drop the self column (distance 0).
        neighbour_idx = neighbour_idx[:, 1:]
        # B_hat[i] = mean of B[neighbour_idx[i]]; shape (N, m, *B.shape[1:])
        gathered = B[neighbour_idx]
        B_hat = gathered.mean(dim=1)
        gap_deviations = B_hat - mu_B.squeeze(0)
        g_hat = gap_deviations.pow(2).sum(dim=sum_dims).mean().item()
    elif estimator == "bin":
        if conditioning_features.ndim != 1:
            raise ValueError(
                "Binning estimator expects a 1-D conditioning feature "
                f"(e.g. spectral gap); got shape {tuple(conditioning_features.shape)}"
            )
        feat = conditioning_features.float()
        # Quantile edges; include 0 and 1 to cover the full range.
        quantiles = torch.linspace(0.0, 1.0, num_bins + 1)
        edges = torch.quantile(feat, quantiles)
        # bucketize returns 0 for feat < edges[1]; shift so bins start at 0.
        bin_idx = torch.bucketize(feat, edges[1:-1])
        # Between-bin variance: Σ_b (|S_b| / N) ||μ_b − μ_B||²
        weighted_sum = 0.0
        for b in range(num_bins):
            mask = bin_idx == b
            count = int(mask.sum().item())
            if count == 0:
                continue
            mu_b = B[mask].mean(dim=0)
            weighted_sum += (count / num_graphs) * (
                (mu_b - mu_B.squeeze(0)).pow(2).sum().item()
            )
        g_hat = weighted_sum
    else:
        raise ValueError(f"Unknown estimator '{estimator}'")

    fve = g_hat / trace_cov_B if trace_cov_B > 0 else 0.0
    return g_hat, trace_cov_B, fve


class SpectralAnalyzer:
    """
    Analyze collected eigenstructure data.

    Parameters
    ----------
    input_dir : Path
        Directory containing batch_*.safetensors files and manifest.json.
    """

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.manifest = load_manifest(self.input_dir)

    def analyze(self) -> SpectralAnalysisResult:
        """
        Run full spectral analysis on collected data.

        Returns
        -------
        SpectralAnalysisResult
            Dataclass with all computed statistics.
        """
        all_spectral_gaps: list[torch.Tensor] = []
        all_algebraic_conn: list[torch.Tensor] = []
        all_coherences: list[torch.Tensor] = []
        all_adj_eigenvalues: list[torch.Tensor] = []
        all_lap_eigenvalues: list[torch.Tensor] = []

        batch_paths = iter_batches(self.input_dir)
        logger.info(f"Analyzing {len(batch_paths)} batches")

        for batch_path in batch_paths:
            tensors, _ = load_decomposition_batch(batch_path)

            eig_adj = tensors["eigenvalues_adj"]
            eig_lap = tensors["eigenvalues_lap"]
            vec_adj = tensors["eigenvectors_adj"]

            all_spectral_gaps.append(compute_spectral_gap(eig_adj))
            all_algebraic_conn.append(compute_algebraic_connectivity(eig_lap))
            all_coherences.append(compute_eigenvector_coherence(vec_adj))
            all_adj_eigenvalues.append(eig_adj)
            all_lap_eigenvalues.append(eig_lap)

        # Concatenate across batches
        spectral_gaps = torch.cat(all_spectral_gaps)
        algebraic_conn = torch.cat(all_algebraic_conn)
        coherences = torch.cat(all_coherences)
        adj_eigenvalues = torch.cat(all_adj_eigenvalues)
        lap_eigenvalues = torch.cat(all_lap_eigenvalues)

        return SpectralAnalysisResult(
            dataset_name=self.manifest["dataset_name"],
            num_graphs=self.manifest["num_graphs"],
            spectral_gap_mean=spectral_gaps.mean().item(),
            spectral_gap_std=spectral_gaps.std().item(),
            spectral_gap_min=spectral_gaps.min().item(),
            spectral_gap_max=spectral_gaps.max().item(),
            algebraic_connectivity_mean=algebraic_conn.mean().item(),
            algebraic_connectivity_std=algebraic_conn.std().item(),
            algebraic_connectivity_min=algebraic_conn.min().item(),
            algebraic_connectivity_max=algebraic_conn.max().item(),
            eigenvalue_entropy_adj=compute_eigenvalue_entropy(adj_eigenvalues),
            eigenvalue_entropy_lap=compute_eigenvalue_entropy(lap_eigenvalues),
            coherence_mean=coherences.mean().item(),
            coherence_std=coherences.std().item(),
            effective_rank_adj_mean=compute_effective_rank(adj_eigenvalues),
            effective_rank_lap_mean=compute_effective_rank(lap_eigenvalues),
        )

    def save_results(self, result: SpectralAnalysisResult, output_dir: Path) -> Path:
        """
        Save analysis results to JSON.

        Parameters
        ----------
        result : SpectralAnalysisResult
            Analysis results to save.
        output_dir : Path
            Directory to write analysis.json.

        Returns
        -------
        Path
            Path to saved JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "analysis.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        return output_path

    def compute_eigenvalue_histogram(
        self, num_bins: int = 50, matrix_type: str = "adjacency"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute histogram of eigenvalues across all graphs.

        Parameters
        ----------
        num_bins : int
            Number of histogram bins.
        matrix_type : str
            Either "adjacency" or "laplacian".

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (bin_edges, counts) for the histogram.
        """
        all_eigenvalues: list[torch.Tensor] = []
        key = "eigenvalues_adj" if matrix_type == "adjacency" else "eigenvalues_lap"

        for batch_path in iter_batches(self.input_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            all_eigenvalues.append(tensors[key].flatten())

        eigenvalues = torch.cat(all_eigenvalues)
        return torch.histogram(eigenvalues, bins=num_bins)

    def compute_subspace_distances(
        self, k: int = 10, sample_size: int = 100
    ) -> dict[str, Any]:
        """
        Compute pairwise subspace distances between random pairs of graphs.

        Parameters
        ----------
        k : int
            Number of top eigenvectors to use for subspace comparison.
        sample_size : int
            Number of random pairs to sample.

        Returns
        -------
        dict
            Statistics on principal angles between graph subspaces.
        """
        all_eigenvectors: list[torch.Tensor] = []

        for batch_path in iter_batches(self.input_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            all_eigenvectors.append(tensors["eigenvectors_adj"])

        eigenvectors = torch.cat(all_eigenvectors, dim=0)
        num_graphs = eigenvectors.shape[0]

        if num_graphs < 2:
            return {"error": "Need at least 2 graphs for subspace comparison"}

        # Sample random pairs
        rng = torch.Generator().manual_seed(42)
        idx1 = torch.randint(0, num_graphs, (sample_size,), generator=rng)
        idx2 = torch.randint(0, num_graphs, (sample_size,), generator=rng)

        first_angles: list[float] = []
        mean_angles: list[float] = []

        for i, j in zip(idx1.tolist(), idx2.tolist(), strict=False):
            if i == j:
                continue
            V1, V2 = eigenvectors[i], eigenvectors[j]
            angles = compute_principal_angles(V1, V2, k)
            first_angles.append(angles[0].item())
            mean_angles.append(angles.mean().item())

        first_angles_t = torch.tensor(first_angles)
        mean_angles_t = torch.tensor(mean_angles)

        return {
            "k": k,
            "num_pairs": len(first_angles),
            "first_principal_angle_mean": first_angles_t.mean().item(),
            "first_principal_angle_std": first_angles_t.std().item(),
            "mean_principal_angle_mean": mean_angles_t.mean().item(),
            "mean_principal_angle_std": mean_angles_t.std().item(),
        }

    def compute_eigenvalue_covariance(
        self, matrix_type: str = "adjacency"
    ) -> CovarianceResult:
        """
        Compute eigenvalue covariance matrix across all graphs.

        Parameters
        ----------
        matrix_type : str
            Either "adjacency" or "laplacian".

        Returns
        -------
        CovarianceResult
            Dataclass with covariance matrix and summary statistics.
        """
        all_eigenvalues: list[torch.Tensor] = []
        key = "eigenvalues_adj" if matrix_type == "adjacency" else "eigenvalues_lap"

        for batch_path in iter_batches(self.input_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            all_eigenvalues.append(tensors[key])

        eigenvalues = torch.cat(all_eigenvalues, dim=0)  # (N, k)
        logger.info(
            f"Computing {matrix_type} covariance for {eigenvalues.shape[0]} graphs "
            f"with {eigenvalues.shape[1]} eigenvalues each"
        )

        cov = compute_eigenvalue_covariance(eigenvalues)
        summary = compute_covariance_summary(cov)

        return CovarianceResult(
            matrix_type=matrix_type,
            num_graphs=eigenvalues.shape[0],
            num_eigenvalues=eigenvalues.shape[1],
            covariance_matrix=cov.tolist(),
            **summary,
        )
