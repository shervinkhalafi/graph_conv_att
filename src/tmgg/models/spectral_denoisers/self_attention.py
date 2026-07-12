"""Self-attention denoiser using proper Vaswani et al. (2017) attention.

Implements single-layer self-attention on eigenvector embeddings:
    1. Project V to Q, K, Val via three learned projection matrices
    2. Compute attention weights: softmax(Q K^T / sqrt(d_k))
    3. Aggregate values: H = attn_weights @ Val
    4. Readout: A_hat = (H W_Qout) (H W_Kout)^T / sqrt(d_out)

The attention stage refines node representations by allowing each node to
attend to all others with learned, softmax-normalized weights. The readout
stage converts refined embeddings to adjacency logits via a bilinear form.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmgg.data.datasets.graph_types import GraphData

from ..base import EdgeSource
from .base_spectral import EmbeddingSource, SpectralDenoiser


class SelfAttentionDenoiser(SpectralDenoiser):
    """Self-attention denoiser following Vaswani et al. (2017).

    Applies single-head self-attention to eigenvector embeddings, producing
    refined node representations, then predicts adjacency via bilinear readout.

    The attention mechanism computes::

        Q = V W_Q,  K = V W_K,  Val = V W_V
        attn = softmax(Q K^T / sqrt(d_k))    (rows sum to 1)
        H = attn @ Val                         (attended node embeddings)
        A_hat = (H W_Qout) (H W_Kout)^T / sqrt(d_out)  (adjacency logits)

    Parameters
    ----------
    k : int
        Number of eigenvectors (input dimension).
    d_k : int
        Query/key/value dimension. Default 64.
    d_out : int or None
        Readout projection dimension. Defaults to ``d_k`` if None.

    Notes
    -----
    Unlike BilinearDenoiser which computes raw QK^T without softmax,
    this model applies proper softmax normalization over the key dimension,
    ensuring attention weights form a valid probability distribution. The
    value projection allows the model to learn what information to propagate,
    rather than using eigenvectors directly for edge prediction.
    """

    def __init__(
        self,
        k: int,
        d_k: int = 64,
        d_out: int | None = None,
        embedding_source: EmbeddingSource = "eigenvector",
        pearl_num_layers: int = 3,
        pearl_hidden_dim: int = 64,
        pearl_input_samples: int = 32,
        pearl_max_nodes: int = 200,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__(
            k=k,
            embedding_source=embedding_source,
            pearl_num_layers=pearl_num_layers,
            pearl_hidden_dim=pearl_hidden_dim,
            pearl_input_samples=pearl_input_samples,
            pearl_max_nodes=pearl_max_nodes,
            edge_source=edge_source,
            output_dims_x_class=output_dims_x_class,
            output_dims_x_feat=output_dims_x_feat,
            output_dims_e_class=output_dims_e_class,
            output_dims_e_feat=output_dims_e_feat,
        )
        self.d_k = d_k
        self.d_out = d_out if d_out is not None else d_k
        self.attn_scale = d_k**-0.5
        self.out_scale = self.d_out**-0.5

        # Attention projections (Q, K, V)
        self.W_Q = nn.Parameter(torch.empty(k, d_k))
        self.W_K = nn.Parameter(torch.empty(k, d_k))
        self.W_V = nn.Parameter(torch.empty(k, d_k))

        # Readout projections (attended embeddings -> adjacency)
        self.W_Q_out = nn.Parameter(torch.empty(d_k, self.d_out))
        self.W_K_out = nn.Parameter(torch.empty(d_k, self.d_out))

        for p in [self.W_Q, self.W_K, self.W_V, self.W_Q_out, self.W_K_out]:
            nn.init.xavier_uniform_(p)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Attention on eigenvectors, then bilinear readout to adjacency.

        Parameters
        ----------
        V : torch.Tensor
            Eigenvector embeddings, shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused).
        A : torch.Tensor
            Input adjacency (unused).

        Returns
        -------
        torch.Tensor
            Adjacency logits, shape (batch, n, n) or (n, n).
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        # 1. Project to Q, K, Val
        Q = torch.matmul(V, self.W_Q)  # (batch, n, d_k)
        K = torch.matmul(V, self.W_K)  # (batch, n, d_k)
        Val = torch.matmul(V, self.W_V)  # (batch, n, d_k)

        # 2. Softmax-normalized attention weights
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.attn_scale
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n, n), rows sum to 1

        # 3. Aggregate values
        H = torch.matmul(attn_weights, Val)  # (batch, n, d_k)

        # 4. Bilinear readout to adjacency logits
        Q_out = torch.matmul(H, self.W_Q_out)  # (batch, n, d_out)
        K_out = torch.matmul(H, self.W_K_out)  # (batch, n, d_out)
        A_reconstructed = torch.matmul(Q_out, K_out.transpose(-1, -2)) * self.out_scale

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_attention_weights(self, data: GraphData) -> torch.Tensor:
        """Compute and return the softmax attention weights for interpretability.

        Parameters
        ----------
        data
            Graph features. The adjacency is extracted internally.

        Returns
        -------
        torch.Tensor
            Attention weights, shape (batch, n, n), rows sum to 1.
        """
        V = super().get_features(data)

        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        Q = torch.matmul(V, self.W_Q)
        K = torch.matmul(V, self.W_K)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.attn_scale
        attn = F.softmax(scores, dim=-1)

        if unbatched:
            attn = attn.squeeze(0)
        return attn

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["d_k"] = self.d_k
        config["d_out"] = self.d_out
        return config
