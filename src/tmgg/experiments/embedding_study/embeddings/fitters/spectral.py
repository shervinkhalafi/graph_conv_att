"""Spectral/SVD-based fitting for graph embeddings.

Provides closed-form solutions and spectral initialization for embedding
models based on eigendecomposition or SVD of the adjacency matrix.
"""

from __future__ import annotations

import torch

from ..base import EmbeddingResult, GraphEmbedding
from ..dot_product import (
    DotProductAsymmetric,
    DotProductSymmetric,
)
from ..lpca import LPCAAsymmetric, LPCASymmetric


class SpectralFitter:
    """Fits embeddings using spectral decomposition.

    For dot product embeddings, provides closed-form SVD solution.
    For other embeddings, provides spectral initialization that can
    be refined via gradient descent.
    """

    def __init__(
        self,
        tol_fnorm: float = 0.01,
        tol_accuracy: float = 0.99,
    ) -> None:
        """Initialize spectral fitter.

        Parameters
        ----------
        tol_fnorm
            Frobenius norm tolerance for success check.
        tol_accuracy
            Edge accuracy tolerance for success check.
        """
        self.tol_fnorm = tol_fnorm
        self.tol_accuracy = tol_accuracy

    def fit(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Fit embedding using spectral methods.

        For DotProduct: uses truncated SVD (closed-form optimal).
        For LPCA: uses logit-transformed SVD initialization.
        For others: uses standard SVD initialization.

        Parameters
        ----------
        embedding
            The embedding model to initialize/fit.
        target
            Target adjacency matrix of shape (n, n).

        Returns
        -------
        EmbeddingResult
            Result with spectrally-fitted embeddings.
        """
        if isinstance(embedding, DotProductSymmetric | DotProductAsymmetric):
            return self._fit_dot_product(embedding, target)
        elif isinstance(embedding, LPCASymmetric | LPCAAsymmetric):
            return self._fit_lpca(embedding, target)
        else:
            return self._fit_generic(embedding, target)

    def _fit_dot_product(
        self,
        embedding: DotProductSymmetric | DotProductAsymmetric,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Closed-form SVD solution for dot product embeddings.

        For symmetric: A ≈ X·Xᵀ via eigendecomposition
        For asymmetric: A ≈ X·Yᵀ via SVD
        """
        d = embedding.dimension

        if isinstance(embedding, DotProductSymmetric):
            # Eigendecomposition for symmetric case
            eigenvalues, eigenvectors = torch.linalg.eigh(target)
            # Sort descending by absolute value
            idx = torch.argsort(eigenvalues.abs(), descending=True)
            eigenvalues = eigenvalues[idx[:d]]
            eigenvectors = eigenvectors[:, idx[:d]]

            # X = V * sqrt(|λ|) * sign(λ)
            # Handle negative eigenvalues by putting sign into embeddings
            signs = torch.sign(eigenvalues)
            signs[signs == 0] = 1
            scale = torch.sqrt(eigenvalues.abs())
            X = eigenvectors * scale.unsqueeze(0) * signs.unsqueeze(0)

            with torch.no_grad():
                embedding.X.copy_(X)
        else:
            # SVD for asymmetric case
            U, S, Vh = torch.linalg.svd(target, full_matrices=False)
            U = U[:, :d]
            S = S[:d]
            V = Vh[:d, :].T

            # X = U * sqrt(S), Y = V * sqrt(S)
            sqrt_S = torch.sqrt(S)
            X = U * sqrt_S.unsqueeze(0)
            Y = V * sqrt_S.unsqueeze(0)

            with torch.no_grad():
                embedding.X.copy_(X)
                embedding.Y.copy_(Y)

        fnorm, accuracy = embedding.evaluate(target)
        converged = fnorm < self.tol_fnorm and accuracy >= self.tol_accuracy

        result = embedding.to_result(target)
        result.converged = converged
        return result

    def _fit_lpca(
        self,
        embedding: LPCASymmetric | LPCAAsymmetric,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Initialize LPCA via logit-transformed SVD.

        Apply logit transform to adjacency, then use SVD. This is not
        optimal but provides a good starting point for gradient refinement.
        """
        d = embedding.dimension

        # Clamp target to avoid infinities in logit
        clamped = target.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(clamped / (1 - clamped))  # logit transform

        if isinstance(embedding, LPCASymmetric):
            eigenvalues, eigenvectors = torch.linalg.eigh(logits)
            idx = torch.argsort(eigenvalues.abs(), descending=True)
            eigenvalues = eigenvalues[idx[:d]]
            eigenvectors = eigenvectors[:, idx[:d]]

            signs = torch.sign(eigenvalues)
            signs[signs == 0] = 1
            scale = torch.sqrt(eigenvalues.abs())
            X = eigenvectors * scale.unsqueeze(0) * signs.unsqueeze(0)

            with torch.no_grad():
                embedding.X.copy_(X)
        else:
            U, S, Vh = torch.linalg.svd(logits, full_matrices=False)
            U = U[:, :d]
            S = S[:d]
            V = Vh[:d, :].T

            sqrt_S = torch.sqrt(S)
            X = U * sqrt_S.unsqueeze(0)
            Y = V * sqrt_S.unsqueeze(0)

            with torch.no_grad():
                embedding.X.copy_(X)
                embedding.Y.copy_(Y)

        fnorm, accuracy = embedding.evaluate(target)
        converged = fnorm < self.tol_fnorm and accuracy >= self.tol_accuracy

        result = embedding.to_result(target)
        result.converged = converged
        return result

    def _fit_generic(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
    ) -> EmbeddingResult:
        """Generic SVD-based initialization for other embedding types.

        Uses top-d singular vectors scaled appropriately.
        """
        d = embedding.dimension

        U, S, Vh = torch.linalg.svd(target, full_matrices=False)
        U = U[:, :d]
        S = S[:d]

        # Scale by sqrt of singular values
        sqrt_S = torch.sqrt(S)
        X = U * sqrt_S.unsqueeze(0)

        with torch.no_grad():
            X_param = getattr(embedding, "X", None)
            if X_param is not None:
                X_param.copy_(X)

            # For asymmetric embeddings, also set Y
            Y_param = getattr(embedding, "Y", None)
            if Y_param is not None:
                V = Vh[:d, :].T
                Y = V * sqrt_S.unsqueeze(0)
                Y_param.copy_(Y)

        fnorm, accuracy = embedding.evaluate(target)
        converged = fnorm < self.tol_fnorm and accuracy >= self.tol_accuracy

        result = embedding.to_result(target)
        result.converged = converged
        return result

    def initialize(
        self,
        embedding: GraphEmbedding,
        target: torch.Tensor,
    ) -> None:
        """Initialize embedding parameters spectrally without creating result.

        Useful as a pre-step before gradient-based refinement.

        Parameters
        ----------
        embedding
            Embedding to initialize in-place.
        target
            Target adjacency matrix.
        """
        self.fit(embedding, target)
