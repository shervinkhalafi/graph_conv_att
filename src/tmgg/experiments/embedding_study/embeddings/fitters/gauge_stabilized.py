"""Gauge-stabilized fitting for LPCA embeddings.

Implements techniques from the LPCA gauge stabilization approach:
1. Canonicalized eigenvector handling for deterministic results
2. Hadamard-based Θ computation for flat spectrum stabilization
3. Θ-space interpolation for initialization
4. Hadamard anchor regularization during optimization

The key insight is that LPCA factorizations have GL(r) gauge freedom—
(X, Y) and (X@R, Y@R^{-T}) yield identical Θ = X@Y^T. This causes
discontinuous embedding jumps under small graph perturbations.
The stabilization techniques anchor embeddings toward a graph-agnostic
Hadamard reference, providing smoother behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.optim as optim
from scipy.linalg import hadamard as scipy_hadamard

from ..base import EmbeddingResult, GraphEmbedding
from ..lpca import LPCAAsymmetric, LPCASymmetric


def canonicalize_eigenvectors(
    V: torch.Tensor,
    eigenvalues: torch.Tensor,
    tol: float = 1e-8,
    round_decimals: int = 6,
) -> torch.Tensor:
    """
    Canonicalize eigenvectors for deterministic results across runs.

    Applies two rules to resolve eigenvector sign and basis ambiguity:
    1. Sign rule: first nonzero entry of each eigenvector is positive
    2. Degenerate subspace handling: lexicographic sort on rounded |v|

    Parameters
    ----------
    V : torch.Tensor
        Eigenvector matrix of shape (n, n), columns are eigenvectors.
    eigenvalues : torch.Tensor
        Corresponding eigenvalues of shape (n,).
    tol : float
        Tolerance for detecting degenerate eigenvalues.
    round_decimals : int
        Decimal places for rounding in lexicographic comparison.

    Returns
    -------
    torch.Tensor
        Canonicalized eigenvector matrix of same shape.
    """
    n = V.shape[1]
    V_canon = V.clone()

    # Apply sign rule: first nonzero entry positive
    for j in range(n):
        v = V_canon[:, j]
        # Find first nonzero entry
        nonzero_mask = v.abs() > 1e-10
        if nonzero_mask.any():
            first_nonzero_idx = int(nonzero_mask.nonzero()[0, 0].item())
            if v[first_nonzero_idx] < 0:
                V_canon[:, j] = -v

    # Handle degenerate subspaces via lexicographic sort
    # Group eigenvectors by eigenvalue (within tolerance)
    sorted_idx = torch.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[sorted_idx]

    i = 0
    while i < n:
        # Find end of degenerate group
        j = i + 1
        while j < n and abs(eigenvalues_sorted[j] - eigenvalues_sorted[i]) < tol:
            j += 1

        # If more than one eigenvector in group, sort lexicographically
        if j - i > 1:
            group_indices = sorted_idx[i:j]
            group_vecs = V_canon[:, group_indices]

            # Round and sort by absolute value (lexicographic)
            rounded = (group_vecs.abs() * (10**round_decimals)).round()
            # Convert to tuple for sorting
            sort_keys = [tuple(rounded[:, k].tolist()) for k in range(j - i)]
            perm = sorted(range(j - i), key=lambda k: sort_keys[k])

            V_canon[:, group_indices] = group_vecs[:, perm]

        i = j

    return V_canon


def compute_hadamard_matrix(n: int) -> torch.Tensor:
    """
    Compute Hadamard matrix of appropriate size.

    Parameters
    ----------
    n : int
        Target size. Will be padded to next power of 2 if needed.

    Returns
    -------
    torch.Tensor
        Hadamard matrix of shape (n_pad, n_pad) where n_pad >= n is a power of 2.
    """
    # Hadamard matrices exist for powers of 2
    n_pad = 1 << (n - 1).bit_length()
    H = torch.from_numpy(scipy_hadamard(n_pad)).float()
    return H


def compute_hadamard_theta(
    n: int,
    rank: int,
    scale: float = 2.0,
) -> torch.Tensor:
    """
    Compute Θ matrix derived from Hadamard embeddings.

    The Hadamard matrix provides a graph-agnostic reference with flat spectrum
    (all singular values equal sqrt(n)), preventing collapse to low effective rank.

    Parameters
    ----------
    n : int
        Number of nodes.
    rank : int
        Embedding dimension.
    scale : float
        Scaling factor for Hadamard embeddings.

    Returns
    -------
    torch.Tensor
        Hadamard-derived Θ matrix of shape (n, n).
    """
    H = compute_hadamard_matrix(n)
    # Truncate to n rows and rank columns, normalize
    X_had = H[:n, :rank] / (n**0.5) * scale
    # Θ = X @ X^T for symmetric case
    theta_had = X_had @ X_had.T
    return theta_had


def compute_hadamard_anchors(
    n: int,
    rank: int,
    scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hadamard anchor embeddings for regularization.

    Parameters
    ----------
    n : int
        Number of nodes.
    rank : int
        Embedding dimension.
    scale : float
        Scaling factor for Hadamard embeddings.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (X_had, Y_had) anchor embeddings, both of shape (n, rank).
        For symmetric case, X_had == Y_had.
    """
    H = compute_hadamard_matrix(n)
    X_had = H[:n, :rank] / (n**0.5) * scale
    return X_had, X_had.clone()


def interpolated_theta_init(
    adjacency: torch.Tensor,
    rank: int,
    alpha: float = 0.1,
    logit_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize embeddings via Θ-space interpolation.

    Blends graph-specific Θ_svd with graph-agnostic Θ_had before factorizing,
    which preserves the flat spectrum properties of Hadamard while incorporating
    graph signal for faster convergence.

    Parameters
    ----------
    adjacency : torch.Tensor
        Binary adjacency matrix of shape (n, n).
    rank : int
        Target embedding dimension.
    alpha : float
        Interpolation weight for Hadamard component (default 0.1 = 10% Hadamard).
    logit_scale : float
        Scale factor for logit transformation of adjacency.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (X, Y) initial embeddings, each of shape (n, rank).
    """
    n = adjacency.shape[0]

    # Graph-specific Θ: logit-transformed adjacency centered at 0.5
    theta_svd = (adjacency - 0.5) * logit_scale

    # Graph-agnostic Θ from Hadamard
    theta_had = compute_hadamard_theta(n, rank)

    # Normalize scales before blending to avoid one dominating
    std_svd = theta_svd.std()
    std_had = theta_had.std()
    if std_svd > 1e-8 and std_had > 1e-8:
        theta_svd_norm = theta_svd / std_svd
        theta_had_norm = theta_had / std_had
        theta_init = (1 - alpha) * theta_svd_norm + alpha * theta_had_norm
        # Rescale to original magnitude
        theta_init = theta_init * std_svd
    else:
        theta_init = (1 - alpha) * theta_svd + alpha * theta_had

    # Ensure symmetric (for numerical stability)
    theta_init = (theta_init + theta_init.T) / 2

    # Eigendecompose to get X, Y
    eigenvalues, eigenvecs = torch.linalg.eigh(theta_init)

    # Sort by absolute eigenvalue magnitude (descending)
    idx = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvecs = eigenvecs[:, idx]

    # Canonicalize for determinism
    eigenvecs = canonicalize_eigenvectors(eigenvecs, eigenvalues)

    # Take top-rank eigenvectors
    eigenvecs_k = eigenvecs[:, :rank]
    lam_k = eigenvalues[:rank]

    # Factorize: X = eigenvecs_k * sqrt(|λ|), Y = eigenvecs_k * sign(λ) * sqrt(|λ|)
    sqrt_abs = torch.sqrt(lam_k.abs())
    sign_lam = torch.sign(lam_k)
    sign_lam[sign_lam == 0] = 1

    X = eigenvecs_k * sqrt_abs.unsqueeze(0)
    Y = eigenvecs_k * (sign_lam * sqrt_abs).unsqueeze(0)

    return X, Y


def interpolated_svd_init(
    adjacency: torch.Tensor,
    rank: int,
    alpha: float = 0.1,
    logit_scale: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize embeddings via SVD-space interpolation.

    Unlike Θ-space interpolation (interpolated_theta_init) which blends logit
    matrices before decomposition, this decomposes each matrix first, aligns
    eigenvectors via Procrustes rotation, then interpolates spectral components.

    This approach preserves spectral structure from both sources more directly,
    at the cost of requiring eigenvector alignment.

    Parameters
    ----------
    adjacency : torch.Tensor
        Binary adjacency matrix of shape (n, n).
    rank : int
        Target embedding dimension.
    alpha : float
        Interpolation weight for Hadamard component (default 0.1 = 10% Hadamard).
    logit_scale : float
        Scale factor for logit transformation of adjacency.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (X, Y) initial embeddings, each of shape (n, rank).
    """
    n = adjacency.shape[0]

    # Graph-specific Θ: logit-transformed adjacency
    theta_svd = (adjacency - 0.5) * logit_scale
    theta_svd = (theta_svd + theta_svd.T) / 2  # Ensure symmetric

    # Graph-agnostic Θ from Hadamard
    theta_had = compute_hadamard_theta(n, rank)

    # Eigendecompose both
    lam_s, V_s = torch.linalg.eigh(theta_svd)
    lam_h, V_h = torch.linalg.eigh(theta_had)

    # Sort by absolute eigenvalue magnitude (descending)
    idx_s = torch.argsort(lam_s.abs(), descending=True)
    idx_h = torch.argsort(lam_h.abs(), descending=True)
    lam_s, V_s = lam_s[idx_s], V_s[:, idx_s]
    lam_h, V_h = lam_h[idx_h], V_h[:, idx_h]

    # Canonicalize for determinism
    V_s = canonicalize_eigenvectors(V_s, lam_s)
    V_h = canonicalize_eigenvectors(V_h, lam_h)

    # Take top-rank components
    lam_s_k, V_s_k = lam_s[:rank], V_s[:, :rank]
    lam_h_k, V_h_k = lam_h[:rank], V_h[:, :rank]

    # Align V_h to V_s via Procrustes rotation
    # Find optimal orthogonal R such that V_h_k @ R ≈ V_s_k
    M = V_s_k.T @ V_h_k
    U, _, Vt = torch.linalg.svd(M)
    R = Vt.T @ U.T  # Optimal rotation
    V_h_aligned = V_h_k @ R

    # Interpolate eigenvalues and eigenvectors
    lam_init = (1 - alpha) * lam_s_k + alpha * lam_h_k
    V_init = (1 - alpha) * V_s_k + alpha * V_h_aligned

    # Re-orthogonalize via QR (interpolation breaks orthogonality)
    V_init, _ = torch.linalg.qr(V_init)

    # Factorize: X = V * sqrt(|λ|), Y = V * sign(λ) * sqrt(|λ|)
    sqrt_abs = torch.sqrt(lam_init.abs())
    sign_lam = torch.sign(lam_init)
    sign_lam[sign_lam == 0] = 1

    X = V_init * sqrt_abs.unsqueeze(0)
    Y = V_init * (sign_lam * sqrt_abs).unsqueeze(0)

    return X, Y


def interpolate_embeddings_naive(
    X1: torch.Tensor,
    Y1: torch.Tensor,
    X2: torch.Tensor,
    Y2: torch.Tensor,
    t: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive linear interpolation of embeddings (baseline, expected to fail).

    This demonstrates why direct embedding interpolation is problematic:
    embeddings from different graphs live in different gauges, so linear
    interpolation produces nonsensical results.

    Parameters
    ----------
    X1, Y1 : torch.Tensor
        Source embeddings from first graph.
    X2, Y2 : torch.Tensor
        Source embeddings from second graph.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Interpolated (X, Y) embeddings.
    """
    X = (1 - t) * X1 + t * X2
    Y = (1 - t) * Y1 + t * Y2
    return X, Y


@dataclass
class GaugeStabilizedConfig:
    """Configuration for gauge-stabilized LPCA fitting.

    Attributes
    ----------
    init_mode
        Initialization mode: "theta" blends Θ matrices before eigendecomposition,
        "svd" decomposes first then aligns and interpolates eigenvectors.
    alpha
        Interpolation weight: (1-α)*graph-specific + α*Hadamard. Default 0.1.
    lambda_had
        Hadamard anchor regularization weight. Default 0.01.
    use_anchor
        Whether to use Hadamard anchor regularization during optimization.
    lr
        Learning rate for Adam optimizer.
    max_steps
        Maximum optimization steps.
    tol_fnorm
        Frobenius norm tolerance for early stopping.
    tol_accuracy
        Edge accuracy tolerance for early stopping.
    patience
        Steps without improvement before early stopping.
    loss_type
        Loss function: "bce" or "mse".
    log_interval
        Steps between progress logging (0 to disable).
    """

    init_mode: Literal["theta", "svd"] = "theta"
    alpha: float = 0.1
    lambda_had: float = 0.01
    use_anchor: bool = True
    lr: float = 0.01
    max_steps: int = 10000
    tol_fnorm: float = 0.01
    tol_accuracy: float = 0.99
    patience: int = 500
    loss_type: Literal["bce", "mse"] = "bce"
    log_interval: int = 0


class GaugeStabilizedFitter:
    """Fits LPCA embeddings with Hadamard-based gauge stabilization.

    Combines three stabilization techniques:
    1. Θ-space interpolation for initialization
    2. Canonicalized eigenvectors for determinism
    3. Hadamard anchor regularization during optimization

    This addresses the GL(r) gauge freedom in LPCA factorizations, producing
    embeddings that vary smoothly under graph perturbations.
    """

    def __init__(self, config: GaugeStabilizedConfig | None = None) -> None:
        """Initialize gauge-stabilized fitter.

        Parameters
        ----------
        config
            Fitting configuration. Uses defaults if None.
        """
        self.config = config or GaugeStabilizedConfig()

    def initialize(
        self,
        embedding: LPCASymmetric | LPCAAsymmetric,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize embedding using configured interpolation mode.

        Parameters
        ----------
        embedding
            LPCA embedding model to initialize.
        target
            Target adjacency matrix.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (X_had, Y_had) Hadamard anchors for regularization.
        """
        if self.config.init_mode == "theta":
            X_init, Y_init = interpolated_theta_init(
                target,
                embedding.dimension,
                alpha=self.config.alpha,
            )
        else:  # "svd"
            X_init, Y_init = interpolated_svd_init(
                target,
                embedding.dimension,
                alpha=self.config.alpha,
            )

        with torch.no_grad():
            embedding.X.copy_(X_init)
            if isinstance(embedding, LPCAAsymmetric):
                embedding.Y.copy_(Y_init)

        # Compute Hadamard anchors
        X_had, Y_had = compute_hadamard_anchors(
            embedding.num_nodes,
            embedding.dimension,
        )

        return X_had.to(target.device), Y_had.to(target.device)

    def fit(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Fit embedding with gauge stabilization.

        Parameters
        ----------
        embedding
            The embedding model to optimize (should be LPCA).
        target
            Target adjacency matrix of shape (n, n).

        Returns
        -------
        EmbeddingResult
            Result containing fitted embeddings and metrics.
        """
        if not isinstance(embedding, LPCASymmetric | LPCAAsymmetric):
            raise TypeError(
                f"GaugeStabilizedFitter requires LPCA embedding, got {type(embedding)}"
            )

        cfg = self.config

        # Initialize with Θ-space interpolation
        X_had, Y_had = self.initialize(embedding, target)

        # Set up optimizer
        optimizer = optim.Adam(embedding.parameters(), lr=cfg.lr)

        best_fnorm = float("inf")
        steps_without_improvement = 0
        converged = False

        for step in range(cfg.max_steps):
            optimizer.zero_grad()

            # Reconstruction loss
            loss = embedding.compute_loss(target, loss_type=cfg.loss_type)

            # Hadamard anchor regularization
            if cfg.use_anchor and cfg.lambda_had > 0:
                X, Y = embedding.get_embeddings()
                anchor_loss = cfg.lambda_had * (
                    torch.norm(X - X_had) ** 2
                    + (torch.norm(Y - Y_had) ** 2 if Y is not None else 0)
                )
                loss = loss + anchor_loss

            loss.backward()
            optimizer.step()

            # Evaluate periodically
            if step % 100 == 0 or step == cfg.max_steps - 1:
                fnorm, accuracy = embedding.evaluate(target)

                if cfg.log_interval > 0 and step % cfg.log_interval == 0:
                    print(
                        f"Step {step}: loss={loss.item():.6f}, "
                        f"fnorm={fnorm:.4f}, accuracy={accuracy:.4f}"
                    )

                # Check early stopping criteria
                if fnorm < cfg.tol_fnorm and accuracy >= cfg.tol_accuracy:
                    converged = True
                    break

                # Track improvement for patience
                if fnorm < best_fnorm - 1e-6:
                    best_fnorm = fnorm
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 100

                if steps_without_improvement >= cfg.patience:
                    break

        result = embedding.to_result(target)
        result.converged = converged
        return result

    def fit_init_only(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Initialize embedding without optimization (for testing initialization quality).

        Parameters
        ----------
        embedding
            The embedding model to initialize.
        target
            Target adjacency matrix.

        Returns
        -------
        EmbeddingResult
            Result with initial embeddings and metrics.
        """
        if not isinstance(embedding, LPCASymmetric | LPCAAsymmetric):
            raise TypeError(
                f"GaugeStabilizedFitter requires LPCA embedding, got {type(embedding)}"
            )

        _ = self.initialize(embedding, target)  # Anchors not needed for init-only

        fnorm, accuracy = embedding.evaluate(target)
        converged = (
            fnorm < self.config.tol_fnorm and accuracy >= self.config.tol_accuracy
        )

        result = embedding.to_result(target)
        result.converged = converged
        return result
