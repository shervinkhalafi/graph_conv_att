"""Multi-head attention denoiser on spectral embeddings.

``ModifiedAttentionDenoiser`` extends :class:`SpectralDenoiser` with
multi-head attention that optionally uses graph-convolutional filters for
Q/K/V projections. The adjacency readout combines per-head attention scores
via a learned linear combination.

Pipeline::

    1. Extract top-k eigenvectors V ∈ R^{n×k} (inherited from SpectralDenoiser)
    2. Project V → (n, d_model) via input_proj
    3. Apply L layers of MultiHeadAttention (with optional graph-conv Q/K/V)
    4. Combine per-head scores via learned weights → adjacency logits (n×n)

The helper modules ``GraphConvolutionFilter`` and ``MultiHeadAttention``
are defined in this file and can be reused independently.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmgg.models.spectral_denoisers.base_spectral import (
    EmbeddingSource,
    SpectralDenoiser,
)


class GraphConvolutionFilter(nn.Module):
    """K-tap graph convolution filter: Z = Σ_{i=0}^{K-1} A^i X H_i.

    Parameters
    ----------
    K : int
        Number of filter taps (polynomial order).
    F : int
        Number of input channels.
    G : int
        Number of output channels.
    layer_norm : bool
        Apply LayerNorm to the output.
    """

    def __init__(self, K: int, F: int, G: int, layer_norm: bool = False):
        super().__init__()
        self.K = K
        self.F = F
        self.G = G

        self.H = nn.Parameter(torch.empty(K, F, G))
        nn.init.xavier_uniform_(self.H)

        self.layer_norm: nn.LayerNorm | None = nn.LayerNorm(G) if layer_norm else None

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply graph convolution.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix, shape ``(B, N, N)``.
        X : torch.Tensor
            Node features, shape ``(B, N, F)``.

        Returns
        -------
        torch.Tensor
            Filtered features, shape ``(B, N, G)``.
        """
        Z = X @ self.H[0]
        A_power_i = A.clone()
        for i in range(1, self.K):
            Z = Z + torch.bmm(A_power_i, X @ self.H[i])
            A_power_i = torch.bmm(A_power_i, A)
        if self.layer_norm is not None:
            Z = self.layer_norm(Z)
        return Z


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional graph-convolutional Q/K/V projections.

    When ``filter_qk`` or ``filter_v`` is True the corresponding projections
    use :class:`GraphConvolutionFilter` instead of a plain ``nn.Linear``,
    allowing the attention to incorporate multi-hop neighbourhood information.

    The module also produces a scalar *combined score* per edge by linearly
    mixing the per-head raw attention scores, which can be used directly as
    adjacency logits.

    Parameters
    ----------
    d_model : int
        Model dimension (input and output).
    num_heads : int
        Number of attention heads.
    d_k : int | None
        Key/query dimension per head. Defaults to ``d_model // num_heads``.
    d_v : int | None
        Value dimension per head. Defaults to ``d_model // num_heads``.
    dropout : float
        Dropout probability.
    bias : bool
        Use bias in linear projections.
    filter_qk : bool
        Use graph-conv filters for Q and K projections.
    filter_v : bool
        Use graph-conv filters for V projection.
    filter_num_terms : int
        Polynomial order for graph-conv filters.
    apply_softmax : bool
        Apply softmax normalisation to attention scores.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        filter_qk: bool = False,
        filter_v: bool = False,
        filter_num_terms: int = 2,
        apply_softmax: bool = True,
        compute_edge_scores: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        self.filter_qk = filter_qk
        self.filter_v = filter_v
        self.apply_softmax = apply_softmax
        self.compute_edge_scores = compute_edge_scores
        self.scale = self.d_k**-0.5

        if filter_qk:
            self.W_q: nn.Module = GraphConvolutionFilter(
                K=filter_num_terms, F=d_model, G=num_heads * self.d_k, layer_norm=True
            )
            self.W_k: nn.Module = GraphConvolutionFilter(
                K=filter_num_terms, F=d_model, G=num_heads * self.d_k, layer_norm=True
            )
        else:
            self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=bias)
            self.W_k = nn.Linear(d_model, num_heads * self.d_k, bias=bias)

        if compute_edge_scores:
            # Score-only head: Q and K produce attention scores that are
            # combined across heads into per-edge logits.  No V/O/norm needed
            # because the node-feature output is not used downstream.
            self.score_combination = nn.Linear(num_heads, 1, bias=False)
            self.W_v: nn.Module | None = None
            self.W_o: nn.Module | None = None
            self.layer_norm: nn.Module | None = None
        else:
            self.score_combination: nn.Linear | None = None
            if filter_v:
                self.W_v = GraphConvolutionFilter(
                    K=filter_num_terms, F=d_model, G=num_heads * self.d_v
                )
            else:
                self.W_v = nn.Linear(d_model, num_heads * self.d_v, bias=bias)
            self.W_o = nn.Linear(num_heads * self.d_v, d_model, bias=bias)
            self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, A: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix, shape ``(B, N, N)``. Required when graph-conv
            filters are enabled; ignored (but still accepted) otherwise.
        x : torch.Tensor
            Node features, shape ``(B, N, d_model)``.

        Returns
        -------
        output : torch.Tensor
            Refined node features, shape ``(B, N, d_model)``.
        combined_scores : torch.Tensor | None
            Learned per-edge score combining all heads, shape ``(B, N, N)``.
            Only produced when ``compute_edge_scores=True``.
        """
        batch_size = x.size(0)

        q = self.W_q(A, x) if self.filter_qk else self.W_q(x)
        k = self.W_k(A, x) if self.filter_qk else self.W_k(x)

        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.compute_edge_scores:
            assert self.score_combination is not None
            combined_scores = self.score_combination(
                scores.permute(0, 2, 3, 1)
            ).squeeze(-1)
            return x, combined_scores

        assert self.W_v is not None
        assert self.W_o is not None
        assert self.layer_norm is not None

        v = self.W_v(A, x) if self.filter_v else self.W_v(x)
        v = v.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        attn_weights = F.softmax(scores, dim=-1) if self.apply_softmax else scores

        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_v)
        )
        output = self.W_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output, None


class ModifiedAttentionDenoiser(SpectralDenoiser):
    """Multi-head attention denoiser on spectral eigenvector embeddings.

    Stacks ``num_layers`` :class:`MultiHeadAttention` blocks on top-k
    eigenvectors, optionally using graph-convolutional filters for the Q/K/V
    projections. Adjacency logits are produced by combining per-head attention
    scores from the final layer via a learned linear mix.

    Parameters
    ----------
    k : int
        Number of eigenvectors.
    d_model : int
        Hidden dimension for the attention blocks.
    num_heads : int
        Number of attention heads. Must divide ``d_model``.
    num_layers : int
        Number of stacked attention layers.
    dropout : float
        Dropout probability.
    filter_qk : bool
        Use graph-conv filters for Q/K projections.
    filter_v : bool
        Use graph-conv filters for V projection.
    filter_num_terms : int
        Polynomial order for graph-conv filters.
    apply_softmax : bool
        Apply softmax in the attention mechanism.
    """

    def __init__(
        self,
        k: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        filter_qk: bool = False,
        filter_v: bool = False,
        filter_num_terms: int = 2,
        apply_softmax: bool = True,
        embedding_source: EmbeddingSource = "eigenvector",
        pearl_num_layers: int = 3,
        pearl_hidden_dim: int = 64,
        pearl_input_samples: int = 32,
        pearl_max_nodes: int = 200,
    ):
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        super().__init__(
            k=k,
            embedding_source=embedding_source,
            pearl_num_layers=pearl_num_layers,
            pearl_hidden_dim=pearl_hidden_dim,
            pearl_input_samples=pearl_input_samples,
            pearl_max_nodes=pearl_max_nodes,
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self._dropout = dropout
        self.filter_qk = filter_qk
        self.filter_v = filter_v
        self.filter_num_terms = filter_num_terms
        self.apply_softmax = apply_softmax

        self.input_proj = nn.Linear(k, d_model)

        self.layers = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    filter_qk=filter_qk,
                    filter_v=filter_v,
                    filter_num_terms=filter_num_terms,
                    apply_softmax=apply_softmax,
                    compute_edge_scores=(i == num_layers - 1),
                )
                for i in range(num_layers)
            ]
        )

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Transform eigenvectors through attention layers to adjacency logits.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors, shape ``(batch, n, k)`` or ``(n, k)``.
        Lambda : torch.Tensor
            Eigenvalues (unused).
        A : torch.Tensor
            Input adjacency matrix, passed to graph-conv filter layers.

        Returns
        -------
        torch.Tensor
            Adjacency logits, shape ``(batch, n, n)`` or ``(n, n)``.
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)
            A = A.unsqueeze(0)

        h = self.input_proj(V)

        edge_scores: torch.Tensor | None = None
        for layer in self.layers:
            h, edge_scores = layer(A, h)

        assert edge_scores is not None
        A_reconstructed = edge_scores

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self._dropout,
                "filter_qk": self.filter_qk,
                "filter_v": self.filter_v,
                "filter_num_terms": self.filter_num_terms,
                "apply_softmax": self.apply_softmax,
            }
        )
        return config
