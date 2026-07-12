"""Noise definitions for graph denoising and diffusion experiments.

Stateless noise functions (``add_gaussian_noise``, ``add_edge_flip_noise``, etc.)
operate directly on adjacency matrices. Definition classes (``NoiseDefinition``
and subclasses) wrap these functions with optional state management and two
interfaces:

- ``add_noise(A, eps)`` operates on raw adjacency tensors (legacy, still
  supported).
- ``apply_noise(data, noise_level)`` operates on a ``DenseGraphState`` (which
  exposes ``to_edge_scalar`` / ``from_structure_only``) and is the canonical
  interface going forward.
"""

# pyright: reportConstantRedefinition=false
# Uppercase single-letter variables (A, V, R) follow standard math notation
# and are intentionally reassigned within functions.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor


def random_skew_symmetric_matrix(
    n: int, rng: np.random.Generator | None = None
) -> Tensor:
    """Create a random n x n skew-symmetric matrix.

    Parameters
    ----------
    n
        Matrix dimension.
    rng
        NumPy random generator for reproducibility. When ``None``, a fresh
        ``np.random.default_rng()`` is created (independent of the global
        numpy RNG state).

    Returns
    -------
    Tensor
        Skew-symmetric matrix of shape ``(n, n)``, dtype ``float32``.
    """
    if rng is None:
        rng = np.random.default_rng()
    A = rng.random((n, n))
    skew = (A - A.T) / 2
    return torch.tensor(skew, dtype=torch.float32)


def add_rotation_noise(
    A: torch.Tensor | np.ndarray, eps: float | Tensor, skew: Tensor
) -> torch.Tensor:
    """Add rotation noise to an adjacency matrix by rotating eigenvectors.

    Computes ``V_rot @ diag(λ) @ V_rot^T`` where ``V_rot = V @ expm(eps * skew)``
    and ``(λ, V)`` are the eigenvalues/eigenvectors of ``A``.

    Parameters
    ----------
    A
        Input adjacency matrix, single ``(n, n)`` or batched ``(bs, n, n)``.
    eps
        Rotation angle scaling factor. Scalar applies uniform rotation;
        a ``(bs,)`` tensor applies per-sample rotation angles.
    skew
        Skew-symmetric matrix of shape ``(n, n)``.

    Returns
    -------
    torch.Tensor
        Rotated adjacency matrix with the same shape as *A*.
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()
    skew = skew.to(device=A.device, dtype=A.dtype)

    # Reshape eps for broadcasting: scalar -> unchanged, (bs,) -> (bs, 1, 1)
    if isinstance(eps, Tensor):
        eps = eps.to(device=A.device, dtype=A.dtype).reshape(-1, 1, 1)

    # Batched matrix exponential: eps * skew -> (bs, n, n) or (n, n)
    R = torch.linalg.matrix_exp(eps * skew)

    eigenvalues, V = torch.linalg.eigh(A)
    V_rot = V @ R
    eig_diag = torch.diag_embed(eigenvalues)
    A_noisy = V_rot @ eig_diag @ V_rot.transpose(-2, -1)

    return torch.real(A_noisy)


def add_gaussian_noise(
    A: torch.Tensor | np.ndarray,
    eps: float | Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add symmetric Gaussian noise to adjacency matrix.

    Generates noise for the upper triangle and mirrors it to the lower
    triangle, matching the symmetrization convention of all other noise
    types (edge_flip, logit, digress). Diagonal entries receive no noise.

    Parameters
    ----------
    A
        Input adjacency matrix, single ``(n, n)`` or batched ``(bs, n, n)``.
    eps
        Noise standard deviation. Scalar applies uniformly; a ``(bs,)``
        tensor applies per-sample noise levels.
    generator
        Optional torch RNG for reproducible noise. When ``None``, uses the
        global RNG.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix with symmetric noise.
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    if isinstance(eps, Tensor):
        eps = eps.to(device=A.device, dtype=A.dtype).reshape(-1, 1, 1)

    noise = eps * torch.randn(
        A.shape, dtype=A.dtype, device=A.device, generator=generator
    )

    if A.dim() == 2:
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.T
    else:
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.transpose(-2, -1)

    return A + noise


def add_logit_noise(
    A: torch.Tensor | np.ndarray,
    sigma: float | Tensor,
    clamp_eps: float = 1e-6,
) -> torch.Tensor:
    """Add Gaussian noise in logit space, returning soft adjacency in (0, 1).

    This noise process operates in the logit (log-odds) space, which provides
    natural bounds for the resulting noisy adjacency values. The transformation
    pipeline is:

        A in {0,1}  ->  clamp to [eps, 1-eps]  ->  logit: L = log(A/(1-A))
            ->  L_noisy = L + N(0, sigma^2)  ->  sigmoid: P = 1/(1+e^{-L_noisy})

    Parameters
    ----------
    A
        Input adjacency matrix (binary or soft), single ``(n, n)`` or
        batched ``(bs, n, n)``.
    sigma
        Standard deviation of Gaussian noise in logit space. Scalar applies
        uniformly; a ``(bs,)`` tensor applies per-sample noise levels.
        Typical range is 0.5 to 5.0.
    clamp_eps
        Small epsilon for clamping to avoid log(0) or log(inf). Default 1e-6.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix with values in (0, 1). The output is symmetric
        if the input was symmetric, since noise is applied symmetrically.

    Notes
    -----
    Unlike flip-based noise (DiGress), this produces soft/continuous outputs.
    The sigma parameter controls noise severity in log-odds space:
    sigma=0.5 is mild, sigma=2.0 moderate, sigma=5.0 aggressive.

    Examples
    --------
    >>> A = torch.eye(5)
    >>> A_noisy = add_logit_noise(A, sigma=1.0)
    >>> assert A_noisy.min() > 0 and A_noisy.max() < 1
    >>> assert torch.allclose(A_noisy, A_noisy.T)  # Symmetric
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    if isinstance(sigma, Tensor):
        sigma = sigma.to(device=A.device, dtype=A.dtype).reshape(-1, 1, 1)

    # Clamp for numerical stability (avoid log(0) and log(inf))
    A_clamped = A.clamp(clamp_eps, 1 - clamp_eps)

    # Transform to logit space: L = log(A / (1 - A))
    L = torch.logit(A_clamped)

    # Generate symmetric noise (upper triangle mirrored to lower)
    noise = torch.randn_like(L) * sigma

    if L.dim() == 2:
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.T
    else:
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.transpose(-2, -1)

    # Add noise in logit space and transform back via sigmoid
    L_noisy = L + noise
    return torch.sigmoid(L_noisy)


def add_digress_noise(
    A: torch.Tensor | np.ndarray,
    alpha_bar: float | Tensor,
) -> torch.Tensor:
    """Apply DiGress-style categorical noise to a binary adjacency matrix.

    Implements the forward process from Vignac et al. (2023) for binary edges
    (K=2 classes). The transition matrix is:

        Q_bar_t = alpha_bar * I + (1 - alpha_bar) * U

    where U is the 2x2 uniform matrix. For binary edges, each entry flips
    with probability (1 - alpha_bar) / 2.

    Parameters
    ----------
    A
        Binary adjacency matrix (0s and 1s), single ``(n, n)`` or batched
        ``(bs, n, n)``.
    alpha_bar
        Cumulative noise schedule parameter in [0, 1]. Scalar applies
        uniformly; a ``(bs,)`` tensor applies per-sample levels.

    Returns
    -------
    torch.Tensor
        Noisy binary adjacency matrix, same shape as input. Symmetric
        if input was symmetric.
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    # Flip probability from DiGress transition: (1 - alpha_bar) / 2
    flip_prob = (1.0 - alpha_bar) / 2.0

    # Delegate to edge-flip noise with the derived probability
    return add_edge_flip_noise(A, p=flip_prob)


def add_edge_flip_noise(
    A: torch.Tensor | np.ndarray,
    p: float | Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Flip edges with probability p (symmetric Bernoulli noise).

    Each upper-triangular entry is independently flipped with probability
    ``p``, then mirrored to the lower triangle to preserve symmetry.

    Parameters
    ----------
    A
        Input adjacency matrix (0s and 1s), single ``(n, n)`` or
        batched ``(bs, n, n)``.
    p
        Probability of flipping each edge. Scalar applies uniformly;
        a ``(bs,)`` tensor applies per-sample flip probabilities.
    generator
        Optional torch RNG for reproducible noise. When ``None``, uses the
        global RNG.

    Returns
    -------
    torch.Tensor
        Noisy adjacency matrix with same shape as input.
    """
    A = torch.tensor(A, dtype=torch.float32) if isinstance(A, np.ndarray) else A.float()

    # Reshape p for broadcasting: (bs,) -> (bs, 1, 1)
    if isinstance(p, Tensor):
        p = p.to(device=A.device, dtype=A.dtype).reshape(-1, 1, 1)

    # Generate random values for each element
    random_values = torch.rand(
        A.shape, dtype=A.dtype, device=A.device, generator=generator
    )

    # Create masks for upper triangular part (excluding diagonal)
    if A.dim() == 2:
        upper_mask = torch.triu(torch.ones_like(A, dtype=torch.bool), diagonal=1)
    else:
        num_nodes = A.shape[1]
        upper_mask = (
            torch.triu(
                torch.ones(num_nodes, num_nodes, dtype=torch.bool, device=A.device),
                diagonal=1,
            )
            .unsqueeze(0)
            .expand(A.shape[0], -1, -1)
        )

    # Create flip mask only for upper triangle; p broadcasts as (bs,1,1)
    flip_mask_upper = (random_values < p) & upper_mask

    # Make symmetric flip mask by mirroring upper triangle to lower
    if A.dim() == 2:
        flip_mask = flip_mask_upper | flip_mask_upper.T
    else:
        flip_mask = flip_mask_upper | flip_mask_upper.transpose(-2, -1)

    # Flip the elements where the mask is True
    A_noisy = torch.where(flip_mask, 1 - A, A)

    return A_noisy


# ---------------------------------------------------------------------------
# Noise definition classes
# ---------------------------------------------------------------------------


class NoiseDefinition(ABC):
    """Abstract base class for noise definitions.

    A ``NoiseDefinition`` describes the mathematical noise transformation
    applied to graphs. Subclasses implement two interfaces:

    - ``add_noise(A, eps)`` operates on raw adjacency tensors.
    - ``apply_noise(data, noise_level)`` operates on a ``DenseGraphState`` and
      is the canonical interface for new code.

    For single-step denoising, ``SingleStepDenoisingModule`` uses the noise
    definition directly. For multi-step diffusion,
    ``ContinuousNoiseProcess`` (in ``tmgg.diffusion.noise_process``) wraps a
    ``NoiseDefinition`` with a ``NoiseSchedule`` to handle
    timestep-to-noise-level conversion and posterior computation.
    ``CategoricalNoiseProcess`` owns its categorical stationary distribution
    and reverse-process math directly.
    """

    @abstractmethod
    def add_noise(self, A: torch.Tensor, eps: float | Tensor) -> torch.Tensor:
        """Add noise to an adjacency matrix.

        Parameters
        ----------
        A : Tensor
            Input adjacency matrix, shape ``(n, n)`` or ``(bs, n, n)``.
        eps : float or Tensor
            Noise intensity whose interpretation varies by subclass.
            Scalar applies uniform noise; a ``(bs,)`` tensor applies
            per-sample noise levels.

            ============== ========================= ================
            Subclass       Interpretation            Typical range
            ============== ========================= ================
            Gaussian       Additive std dev          (0, inf), e.g. 0.1-1.0
            EdgeFlip       Per-edge flip probability [0, 1]
            Digress        ``1 - alpha_bar``         [0, 1]
            Logit          Logit-space std dev       (0, inf), e.g. 0.5-5.0
            Rotation       Rotation angle scaling    (0, inf), e.g. 0.1-1.0
            ============== ========================= ================

            All subclasses treat ``eps = 0`` as no noise (identity).

        Returns
        -------
        Tensor
            Noisy adjacency matrix with the same shape as *A*.
        """
        pass

    def apply_noise(self, data: Any, noise_level: float | Tensor) -> Any:
        """Apply noise to a ``DenseGraphState`` instance.

        Extracts the dense edge state from *data* via ``to_edge_scalar``,
        applies noise via ``add_noise``, and wraps the result back into a
        new ``DenseGraphState`` via ``from_structure_only``. This is the
        canonical interface for new code; ``add_noise`` remains for
        callers that work with raw tensors.

        Parameters
        ----------
        data : DenseGraphState
            Input graph data. Typed as ``Any`` to avoid a circular tach
            dependency (``tmgg.utils.noising`` cannot depend on ``tmgg.data``),
            but must be a ``DenseGraphState`` instance at runtime (it must
            expose ``to_edge_scalar`` and ``from_structure_only``).
        noise_level
            Noise intensity (same semantics as *eps* in ``add_noise``).

        Returns
        -------
        DenseGraphState
            Noised graph data.
        """
        edge_state = (
            data.to_edge_scalar(source="feat")
            if data.E_feat is not None
            else data.to_edge_scalar(source="class")
        )
        noisy = self.add_noise(edge_state, noise_level)
        return type(data).from_structure_only(data.node_mask, noisy)

    @property
    @abstractmethod
    def requires_state(self) -> bool:
        """Whether this noise definition maintains internal state."""
        pass


class GaussianNoise(NoiseDefinition):
    """Gaussian noise definition (stateless).

    Adds symmetric Gaussian noise to adjacency matrices. Each
    upper-triangular entry receives independent N(0, eps^2) noise,
    mirrored to the lower triangle.
    """

    @staticmethod
    def add_gaussian_noise(
        A: torch.Tensor | np.ndarray,
        eps: float | Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Add symmetric Gaussian noise to adjacency matrix.

        See module-level ``add_gaussian_noise`` for full documentation.
        """
        return add_gaussian_noise(A, eps, generator=generator)

    def add_noise(self, A: torch.Tensor, eps: float | Tensor) -> torch.Tensor:
        """Add symmetric Gaussian noise with std dev ``eps``."""
        return self.add_gaussian_noise(A, eps)

    @property
    def requires_state(self) -> bool:
        return False


class EdgeFlipNoise(NoiseDefinition):
    """Symmetric Bernoulli edge-flip noise definition (stateless)."""

    @staticmethod
    def add_edge_flip_noise(
        A: torch.Tensor | np.ndarray,
        p: float | Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Flip edges with probability p (symmetric Bernoulli noise).

        See module-level ``add_edge_flip_noise`` for full documentation.
        """
        return add_edge_flip_noise(A, p, generator=generator)

    def add_noise(self, A: torch.Tensor, eps: float | Tensor) -> torch.Tensor:
        """Flip edges with probability eps."""
        return self.add_edge_flip_noise(A, eps)

    @property
    def requires_state(self) -> bool:
        return False


class DigressNoise(NoiseDefinition):
    """DiGress categorical noise definition (Vignac et al. 2023).

    Implements the forward diffusion process from DiGress for binary
    adjacency matrices. The eps parameter from the NoiseDefinition interface
    is converted to alpha_bar = 1 - eps before applying noise:
    - eps=0 -> alpha_bar=1 -> no noise (clean)
    - eps=1 -> alpha_bar=0 -> fully random (uniform)

    At a given eps, this produces half the flip rate of EdgeFlipNoise,
    because DiGress models noise as a categorical transition where the maximum
    flip probability (at alpha_bar=0) is 0.5, not 1.0.
    """

    @staticmethod
    def add_digress_noise(
        A: torch.Tensor | np.ndarray,
        alpha_bar: float | Tensor,
    ) -> torch.Tensor:
        """Apply DiGress-style categorical noise to a binary adjacency matrix.

        See module-level ``add_digress_noise`` for full documentation.
        """
        return add_digress_noise(A, alpha_bar=alpha_bar)

    def add_noise(self, A: torch.Tensor, eps: float | Tensor) -> torch.Tensor:
        """Apply DiGress categorical noise with ``alpha_bar = 1 - eps``."""
        alpha_bar = 1.0 - eps
        return self.add_digress_noise(A, alpha_bar=alpha_bar)

    @property
    def requires_state(self) -> bool:
        return False


class RotationNoise(NoiseDefinition):
    """Rotation noise definition with skew matrix management.

    Maintains a skew-symmetric matrix used to rotate eigenvectors,
    providing a consistent rotation throughout the experiment.

    Notes
    -----
    The skew matrix dimension is fixed at construction and must match
    the input graph size exactly. Variable-size graphs or padded batches
    will raise a dimension error. Create separate definitions for each
    graph size, or use ``GaussianNoise`` for variable-size inputs.

    Unlike other noise types (Gaussian, edge_flip, DiGress) which produce
    valid binary adjacency matrices, rotation noise produces
    continuous-valued matrices via ``V_rot @ diag(lambda) @ V_rot^T``.
    Downstream thresholding or rounding is the caller's responsibility.
    """

    def __init__(self, k: int, seed: int | None = None):
        self.k = k
        self.seed = seed
        self.skew = self._generate_skew_matrix(k, seed)

    def _generate_skew_matrix(self, k: int, seed: int | None) -> Tensor:
        """Generate a random skew-symmetric matrix as a torch Tensor."""
        rng = np.random.default_rng(seed)
        A = rng.random((k, k))
        skew = (A - A.T) / 2
        return torch.tensor(skew, dtype=torch.float32)

    @staticmethod
    def add_rotation_noise(
        A: torch.Tensor | np.ndarray, eps: float | Tensor, skew: Tensor
    ) -> torch.Tensor:
        """Add rotation noise to an adjacency matrix by rotating eigenvectors.

        See module-level ``add_rotation_noise`` for full documentation.
        """
        return add_rotation_noise(A, eps, skew)

    def add_noise(self, A: torch.Tensor, eps: float | Tensor) -> torch.Tensor:
        """Rotate eigenvectors by ``expm(eps * skew)``; ``eps`` scales the angle."""
        if A.shape[-1] != self.skew.shape[0]:
            raise ValueError(
                f"Graph dimension {A.shape[-1]} != skew matrix dimension "
                f"{self.skew.shape[0]}. RotationNoise was created "
                f"with k={self.skew.shape[0]}."
            )
        return self.add_rotation_noise(A, eps, self.skew)

    @property
    def requires_state(self) -> bool:
        return True

    def get_config(self) -> dict[str, int | None]:
        """Get configuration for this noise definition."""
        return {
            "k": self.k,
            "seed": self.seed,
        }


class LogitNoise(NoiseDefinition):
    """Logit-space noise definition.

    Operates in the log-odds space, providing naturally bounded outputs.
    Noise is added in logit space and transformed back via sigmoid,
    ensuring outputs remain in (0, 1).

    Note: This noise type is experimental and not yet used in standard sweeps.
    The eps parameter represents the standard deviation in logit space, not a
    flip probability. Typical values range from 0.5 to 5.0.

    Parameters
    ----------
    clamp_eps
        Small epsilon for clamping input values to avoid numerical issues
        with log(0) or log(inf). Default 1e-6.

    See Also
    --------
    add_logit_noise : The underlying noise function with full documentation.
    """

    def __init__(self, clamp_eps: float = 1e-6):
        self.clamp_eps = clamp_eps

    @staticmethod
    def add_logit_noise(
        A: torch.Tensor | np.ndarray,
        sigma: float | Tensor,
        clamp_eps: float = 1e-6,
    ) -> torch.Tensor:
        """Add Gaussian noise in logit space, returning soft adjacency in (0, 1).

        See module-level ``add_logit_noise`` for full documentation.
        """
        return add_logit_noise(A, sigma=sigma, clamp_eps=clamp_eps)

    def add_noise(self, A: torch.Tensor, eps: float | Tensor) -> torch.Tensor:
        """Add noise in logit space.

        Parameters
        ----------
        A
            Input adjacency matrix (binary or soft).
        eps
            Standard deviation of Gaussian noise in logit space.
            Scalar or ``(bs,)`` tensor for per-sample levels.
            Typical range: 0.5 to 5.0.

        Returns
        -------
        torch.Tensor
            Noisy adjacency with values in (0, 1).
        """
        return self.add_logit_noise(A, sigma=eps, clamp_eps=self.clamp_eps)

    @property
    def requires_state(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_noise_definition(
    noise_type: str,
    rotation_k: int | None = None,
    seed: int | None = None,
    **kwargs: float,
) -> NoiseDefinition:
    """Factory function to create noise definitions.

    Parameters
    ----------
    noise_type
        Type of noise: ``"gaussian"``, ``"edge_flip"``, ``"digress"``,
        ``"rotation"``, or ``"logit"``.
    rotation_k
        Dimension for rotation noise skew matrix (required when
        ``noise_type="rotation"``).
    seed
        Random seed for reproducible noise generation.
    **kwargs
        Additional keyword arguments.

        - ``clamp_eps``: For logit noise, epsilon for numerical stability
          (default 1e-6).

    Returns
    -------
    NoiseDefinition

    Raises
    ------
    ValueError
        If *noise_type* is unknown or required parameters are missing.
    """
    if noise_type == "gaussian":
        return GaussianNoise()
    elif noise_type == "edge_flip":
        return EdgeFlipNoise()
    elif noise_type == "digress":
        return DigressNoise()
    elif noise_type == "rotation":
        if rotation_k is None:
            raise ValueError("rotation_k parameter is required for rotation noise")
        return RotationNoise(k=rotation_k, seed=seed)
    elif noise_type == "logit":
        clamp_eps = kwargs.get("clamp_eps", 1e-6)
        return LogitNoise(clamp_eps=clamp_eps)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
