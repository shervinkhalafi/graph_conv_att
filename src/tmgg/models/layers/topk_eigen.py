"""Top-k eigenvector extraction with diagnostic error handling.

Provides ``TopKEigenLayer`` for extracting the top-k eigenvectors (by
eigenvalue magnitude) from symmetric adjacency matrices, and
``EigenDecompositionError`` for rich diagnostic context when
eigendecomposition fails on ill-conditioned matrices.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false
# torch._C._LinAlgError is a private C-extension type not declared in the
# PyTorch stubs; accessing it is valid at runtime for catching LAPACK errors.

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class EigenDecompositionError(Exception):
    """Eigendecomposition failure with diagnostic context.

    Raised when ``torch.linalg.eigh`` fails on a specific matrix in a batch.
    Captures condition number, norms, NaN/Inf checks, and diagonal dominance
    to help diagnose ill-conditioned inputs.
    """

    def __init__(
        self, matrix_idx: int, matrix: torch.Tensor, original_error: Exception
    ):
        self.matrix_idx = matrix_idx
        self.matrix = matrix
        self.original_error = original_error

        self.debugging_context = self._compute_debugging_metrics(matrix)

        super().__init__(self._format_message())

    def _compute_debugging_metrics(self, A: torch.Tensor) -> dict[str, Any]:
        """Compute key metrics for debugging ill-conditioned matrices."""
        with torch.no_grad():
            A_np = A.cpu().numpy()

            try:
                singular_values = np.linalg.svd(A_np, compute_uv=False)
                condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
            # Broad catch is intentional: this runs inside a diagnostic for an
            # already-failed eigendecomposition. A secondary SVD failure here
            # should not mask the original error.
            except Exception:
                condition_number = float("inf")

            frobenius_norm = torch.norm(A, p="fro").item()
            trace = torch.trace(A).item()
            has_nan = torch.isnan(A).any().item()
            has_inf = torch.isinf(A).any().item()

            diagonal = torch.diag(A)
            off_diagonal_sum = torch.sum(torch.abs(A), dim=1) - torch.abs(diagonal)
            min_diagonal_dominance = torch.min(
                torch.abs(diagonal) - off_diagonal_sum
            ).item()

            return {
                "condition_number": condition_number,
                "frobenius_norm": frobenius_norm,
                "trace": trace,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "min_diagonal_dominance": min_diagonal_dominance,
                "matrix_shape": list(A.shape),
            }

    def _format_message(self) -> str:
        """Format error message with debugging context."""
        ctx = self.debugging_context
        return (
            f"Eigendecomposition failed for matrix {self.matrix_idx}. "
            f"Debugging context: "
            f"condition_number={ctx['condition_number']:.2e}, "
            f"frobenius_norm={ctx['frobenius_norm']:.2e}, "
            f"trace={ctx['trace']:.2e}, "
            f"has_nan={ctx['has_nan']}, "
            f"has_inf={ctx['has_inf']}, "
            f"min_diagonal_dominance={ctx['min_diagonal_dominance']:.2e}, "
            f"shape={ctx['matrix_shape']}. "
            f"Original error: {str(self.original_error)}"
        )


class TopKEigenLayer(nn.Module):
    """Extract top-k eigenvectors from symmetric adjacency matrices.

    Given a batch of symmetric adjacency matrices, computes eigendecomposition
    and returns the top-k eigenvectors ordered by eigenvalue magnitude. Sign
    ambiguity is resolved by enforcing the first nonzero entry of each
    eigenvector to be positive (controllable via ``sign_normalize``).

    When k >= n (requesting all eigenvectors), returns them in ascending
    eigenvalue order from ``eigh`` without reordering.

    Selects eigenvectors corresponding to eigenvalues with largest absolute
    value (|λ|), analogous to PCA's variance-maximizing selection. This
    differs from spectral clustering, which typically uses the smallest
    Laplacian eigenvalues.

    Parameters
    ----------
    k : int
        Number of eigenvectors to extract. If k exceeds the matrix dimension n,
        all n eigenvectors are returned in ascending eigenvalue order.
    eigenvalue_reg : float
        Diagonal regularization added before eigendecomposition. Spreads
        eigenvalues apart, improving gradient stability. Default 0.0.
    sign_normalize : bool
        Whether to normalize eigenvector signs so the first nonzero entry
        is positive. Default True.

    Examples
    --------
    >>> layer = TopKEigenLayer(k=4)
    >>> A = torch.randn(8, 20, 20)
    >>> A = (A + A.transpose(-1, -2)) / 2  # symmetrize
    >>> V, Lambda = layer(A)
    >>> V.shape
    torch.Size([8, 20, 4])
    >>> Lambda.shape
    torch.Size([8, 4])
    """

    def __init__(
        self,
        k: int,
        eigenvalue_reg: float = 0.0,
        sign_normalize: bool = True,
    ):
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.eigenvalue_reg = eigenvalue_reg
        self.sign_normalize = sign_normalize

    def forward(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract top-k eigenvectors from adjacency matrices.

        Parameters
        ----------
        A : torch.Tensor
            Batch of symmetric adjacency matrices of shape (batch, n, n).
            Can also accept unbatched input of shape (n, n).

        Returns
        -------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k) if unbatched.
            When k < n, ordered by decreasing eigenvalue magnitude.
            When k >= n, ordered by ascending eigenvalue (eigh order).
        Lambda : torch.Tensor
            Corresponding eigenvalues of shape (batch, k) or (k,) if unbatched.
        """
        unbatched = A.ndim == 2
        if unbatched:
            A = A.unsqueeze(0)

        batch_size, n, _ = A.shape
        k = min(self.k, n)

        # Optional diagonal regularization
        if self.eigenvalue_reg > 0:
            eye = torch.eye(n, device=A.device, dtype=A.dtype)
            A = A + self.eigenvalue_reg * eye

        # Enforce symmetry before eigh: floating-point operations can
        # introduce small asymmetries, and eigh silently uses only the
        # lower triangle, which can cause inconsistent results
        A_sym = (A + A.transpose(-1, -2)) / 2

        # Eigendecomposition with diagnostic error handling
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
        except torch._C._LinAlgError as e:
            for i in range(batch_size):
                try:
                    torch.linalg.eigh(A_sym[i])
                except torch._C._LinAlgError:
                    raise EigenDecompositionError(i, A[i], e) from e
            raise
        # eigenvalues: (batch, n), eigenvectors: (batch, n, n)

        if k < n:
            # Select top-k by magnitude (descending order)
            abs_eigenvalues = eigenvalues.abs()
            _, topk_indices = torch.topk(abs_eigenvalues, k, dim=-1)
            # topk_indices: (batch, k)

            # Gather eigenvalues
            Lambda = torch.gather(eigenvalues, dim=-1, index=topk_indices)

            # Gather eigenvectors: expand indices for the n dimension
            expanded_indices = topk_indices.unsqueeze(1).expand(-1, n, -1)
            V = torch.gather(eigenvectors, dim=-1, index=expanded_indices)
        else:
            # k >= n: return all eigenvectors in ascending eigenvalue order
            V = eigenvectors
            Lambda = eigenvalues

        if self.sign_normalize:
            V = self._normalize_signs(V)

        if unbatched:
            V = V.squeeze(0)
            Lambda = Lambda.squeeze(0)

        return V, Lambda

    def _normalize_signs(self, V: torch.Tensor) -> torch.Tensor:
        """Normalize eigenvector signs so first nonzero entry is positive.

        Parameters
        ----------
        V : torch.Tensor
            Eigenvectors of shape (batch, n, k).

        Returns
        -------
        torch.Tensor
            Sign-normalized eigenvectors of same shape.
        """
        # For each eigenvector column, find the first nonzero entry
        # and flip sign if it's negative
        batch, n, k = V.shape

        # Find first nonzero entry per eigenvector
        # Use a small threshold to handle numerical zeros
        eps = 1e-10
        abs_V = V.abs()

        # Create mask of nonzero entries
        nonzero_mask = abs_V > eps  # (batch, n, k)

        # For each column, find the index of first nonzero entry
        # We do this by finding argmax of cumsum of the mask (first True becomes 1)
        cumsum_mask = nonzero_mask.cumsum(dim=1)  # (batch, n, k)
        first_nonzero_mask = (cumsum_mask == 1) & nonzero_mask

        # Get the sign of the first nonzero entry
        # Where mask is True, take the sign; elsewhere 0
        first_signs = torch.where(
            first_nonzero_mask, torch.sign(V), torch.zeros_like(V)
        )
        # Sum over n dimension to get the sign for each eigenvector
        sign_multipliers = first_signs.sum(dim=1)  # (batch, k)

        # Handle case where eigenvector is all zeros (shouldn't happen in practice)
        # This can indicate k > rank(A) or numerical issues. The actual
        # sign correction below uses ``torch.where`` (branch-free), so it
        # composes with ``torch.compile`` without graph breaks. The warning
        # emission, by contrast, requires a Python-level bool branch on
        # ``mask.any()`` — that triggers a torch._dynamo Dynamic-control-flow
        # error under compile. Gating the warning under ``if __debug__:``
        # makes it disappear entirely under ``PYTHONOPTIMIZE=1`` (production
        # default in the Modal image env, see
        # ``tmgg/modal/_lib/image._runtime_env``), so the compile trace
        # never sees the data-dependent branch.
        zero_eigenvector_mask = sign_multipliers == 0
        # Skip the warning emission under ``-O`` AND under ``torch.compile``.
        # ``zero_eigenvector_mask.any()`` is a data-dependent bool branch
        # that dynamo cannot trace; ``torch.compiler.is_compiling()`` and
        # ``__debug__`` both constant-fold so the dev path keeps the warning
        # while production-fast (-O or compiled) skips the GPU sync.
        if (
            __debug__
            and not torch.compiler.is_compiling()
            and zero_eigenvector_mask.any()
        ):
            import warnings

            warnings.warn(
                "Zero eigenvector detected in sign normalization; may indicate k > rank(A) "
                "or numerical issues in eigendecomposition.",
                RuntimeWarning,
                stacklevel=2,
            )
        sign_multipliers = torch.where(
            zero_eigenvector_mask, torch.ones_like(sign_multipliers), sign_multipliers
        )

        # Apply sign correction
        return V * sign_multipliers.unsqueeze(1)

    def extra_repr(self) -> str:
        parts = [f"k={self.k}"]
        if self.eigenvalue_reg > 0:
            parts.append(f"eigenvalue_reg={self.eigenvalue_reg}")
        if not self.sign_normalize:
            parts.append("sign_normalize=False")
        return ", ".join(parts)
