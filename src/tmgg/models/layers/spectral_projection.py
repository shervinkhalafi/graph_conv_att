"""Spectral filter bank projection for attention Q/K/V.

Implements eigenvalue-polynomial filtering in the spectral domain, intended
as a drop-in replacement for spatial GNN projections in attention mechanisms.
"""

import torch
import torch.nn as nn


class SpectralProjectionLayer(nn.Module):
    """Spectral filter bank projection for attention Q/K/V.

    Computes Y = V @ W where W is a polynomial in eigenvalues:
        W = Σ_{ℓ=0}^{K-1} Λ^ℓ ⊙ H^{(ℓ)}

    This scales eigenmodes by learned frequency-dependent weights, providing
    a spectral analog to the spatial polynomial filter in BareGraphConvolutionLayer.

    Parameters
    ----------
    k : int
        Number of eigenvectors (embedding dimension from eigendecomposition).
    out_dim : int
        Output dimension for the projection.
    num_terms : int
        Number of polynomial terms (degree K-1). Default is 3.
    normalize_eigenvalues : bool
        If True (default), rescale eigenvalues to ``[-1, 1]`` per-graph
        via ``Lambda / max|Lambda|`` before raising them to powers
        (clamped at ``1e-6`` from below to avoid division by zero on
        all-zero spectra). If False, feed the input eigenvalues
        directly to the polynomial. Defaulting to True preserves
        historical behaviour. Setting False is appropriate when the
        caller has already rescaled the eigenvalues upstream
        (e.g. ``L`` already scaled to ``2L/λ_max - I``); the caller
        is then responsible for keeping ``|λ| ≤ 1`` so ``Λ^ℓ`` does
        not overflow.

    Notes
    -----
    Unlike BareGraphConvolutionLayer which uses spatial message passing:
        Y = Σ A_norm^i @ X @ H[i]

    This layer uses spectral filtering:
        W = Σ Λ^ℓ ⊙ H[ℓ],  Y = V @ W @ H_out

    where V are eigenvectors, Λ are eigenvalues. The spectral approach operates
    directly in the eigenspace without message passing.

    For use in DiGress attention, replace BareGraphConvolutionLayer with this
    layer for Q/K/V projections.

    Examples
    --------
    >>> layer = SpectralProjectionLayer(k=16, out_dim=64, num_terms=3)
    >>> V = torch.randn(4, 50, 16)  # batch, nodes, k eigenvectors
    >>> Lambda = torch.randn(4, 16)  # batch, k eigenvalues
    >>> Y = layer(V, Lambda)
    >>> Y.shape
    torch.Size([4, 50, 64])
    """

    def __init__(
        self,
        k: int,
        out_dim: int,
        num_terms: int = 3,
        normalize_eigenvalues: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.out_dim = out_dim
        self.num_terms = num_terms
        self.normalize_eigenvalues = normalize_eigenvalues

        # Polynomial coefficient matrices H^{(ℓ)} ∈ R^{k×k} for ℓ = 0, ..., K-1
        self.H = nn.ParameterList(
            [nn.Parameter(torch.empty(k, k)) for _ in range(num_terms)]
        )
        for h in self.H:
            nn.init.xavier_uniform_(h)

        # Output projection: k -> out_dim
        self.out_proj = nn.Linear(k, out_dim, bias=False)

    def forward(self, V: torch.Tensor, Lambda: torch.Tensor) -> torch.Tensor:
        """Apply spectral polynomial filter and project.

        Parameters
        ----------
        V : torch.Tensor
            Eigenvectors of shape (batch, n, k).
        Lambda : torch.Tensor
            Eigenvalues of shape (batch, k). When
            ``self.normalize_eigenvalues`` is False, the caller is
            responsible for keeping ``|Lambda| ≤ 1``.

        Returns
        -------
        torch.Tensor
            Projected features of shape (batch, n, out_dim).
        """
        if self.normalize_eigenvalues:
            # Rescale eigenvalues to [-1, 1] for numerical stability
            Lambda_max = Lambda.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            Lambda_used = Lambda / Lambda_max
        else:
            Lambda_used = Lambda

        # Compute W = Σ Λ^ℓ ⊙ H^{(ℓ)}
        from tmgg.models.layers.graph_ops import spectral_polynomial

        # Iterate ``ParameterList`` directly into a fresh Python list
        # rather than calling ``list(self.H)`` — dynamo cannot trace the
        # ``list()`` builtin on a ``ParameterList`` (graph-break:
        # "Failed to trace builtin operator list with argument types
        # ['ParameterList']"). The list-comprehension form unrolls at
        # trace time into a fixed-length Python list of Parameter refs,
        # which the spectral_polynomial helper iterates via a Python
        # ``for`` loop — both compile cleanly.
        W = spectral_polynomial(Lambda_used, [h for h in self.H])

        # Apply spectral filter: Y = V @ W
        Y = torch.matmul(V, W)  # (batch, n, k)

        # Project to output dimension
        Y = self.out_proj(Y)  # (batch, n, out_dim)

        return Y
