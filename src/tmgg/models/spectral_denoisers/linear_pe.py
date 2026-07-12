"""Linear Positional Encoding denoiser for graph denoising.

Implements the simplest spectral denoiser: a linear transformation in
eigenspace with optional node-specific bias correction. Supports both
symmetric (VWV^T) and asymmetric (XY^T) reconstruction modes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..base import EdgeSource
from .base_spectral import EmbeddingSource, SpectralDenoiser


class LinearPE(SpectralDenoiser):
    """Linear Positional Encoding denoiser.

    Reconstructs the adjacency matrix using eigenvector projections with
    optional asymmetric mode:

    Symmetric mode (default):
        Â = V W V^T + 1 b^T + b 1^T

    Asymmetric mode:
        X = V W_X,  Y = V W_Y
        Â = X Y^T + 1 b^T + b 1^T

    where V ∈ R^{n×k} are the top-k eigenvectors of the noisy adjacency,
    W (or W_X, W_Y) ∈ R^{k×k} are learnable weight matrices, and
    b ∈ R^{max_n} is a learnable bias vector.

    Outputs raw logits. Use model.predict(logits) for [0,1] probabilities.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use.
    max_nodes : int, optional
        Maximum number of nodes in any graph. Required if use_bias is True.
        Default is 200.
    use_bias : bool, optional
        Whether to use the node-specific bias term. For multi-graph settings
        with varying sizes, consider setting this to False. Default is True.
    asymmetric : bool, optional
        If True, use separate W_X and W_Y weight matrices for asymmetric
        reconstruction X @ Y.T. Default is False.

    Notes
    -----
    The bias term adds a rank-2 correction to the low-rank reconstruction.
    Entry (i, j) receives bias b_i + b_j, which captures node degree effects
    not captured by the top-k eigenspace.

    Asymmetric mode doubles the parameter count but allows the model to learn
    different row and column embeddings, which can capture asymmetric
    structure in directed graphs or improve expressiveness.

    The ``max_nodes`` parameter sizes the learnable bias term b ∈ R^{max_nodes}.
    When ``use_bias`` is true, a graph with more than ``max_nodes`` nodes raises
    a ``ValueError`` (the bias cannot cover it); raise ``max_nodes`` to the
    dataset's largest graph. When ``use_bias`` is false the parameter is unused.
    The default of 200 covers SBM and Erdos-Renyi benchmarks; override
    for larger graphs.

    Examples
    --------
    >>> model = LinearPE(k=8, max_nodes=50)
    >>> A_noisy = torch.randn(4, 50, 50)
    >>> A_noisy = (A_noisy + A_noisy.transpose(-1, -2)) / 2
    >>> logits = model(A_noisy)
    >>> predictions = model.predict(logits)  # apply sigmoid
    >>> predictions.shape
    torch.Size([4, 50, 50])

    >>> # Asymmetric mode
    >>> model_asym = LinearPE(k=8, max_nodes=50, asymmetric=True)
    >>> logits_asym = model_asym(A_noisy)
    """

    def __init__(
        self,
        k: int,
        max_nodes: int = 200,
        use_bias: bool = True,
        asymmetric: bool = False,
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
        self.max_nodes = max_nodes
        self.use_bias = use_bias
        self.asymmetric = asymmetric

        if asymmetric:
            # Separate weight matrices for X and Y embeddings
            self.W_X = nn.Parameter(torch.empty(k, k))
            self.W_Y = nn.Parameter(torch.empty(k, k))
            nn.init.xavier_uniform_(self.W_X)
            nn.init.xavier_uniform_(self.W_Y)
            self.register_parameter("W", None)
        else:
            # Single weight matrix W ∈ R^{k×k}
            self.W = nn.Parameter(torch.empty(k, k))
            nn.init.xavier_uniform_(self.W)
            self.register_parameter("W_X", None)
            self.register_parameter("W_Y", None)

        # Learnable bias vector b ∈ R^{max_nodes}
        if use_bias:
            self.b = nn.Parameter(torch.zeros(max_nodes))
        else:
            self.register_parameter("b", None)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VWV^T (symmetric) or XY^T (asymmetric) + bias.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused in this model).
        A : torch.Tensor
            Input adjacency (unused, available for other architectures).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency logits.
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        batch_size, n, k = V.shape

        if self.asymmetric:
            # Asymmetric reconstruction: X Y^T where X = V W_X, Y = V W_Y
            assert self.W_X is not None and self.W_Y is not None
            X = torch.matmul(V, self.W_X)  # (batch, n, k)
            Y = torch.matmul(V, self.W_Y)  # (batch, n, k)
            A_reconstructed = torch.matmul(X, Y.transpose(-1, -2))  # (batch, n, n)
        else:
            # Symmetric reconstruction: V W_sym V^T
            assert self.W is not None
            W_sym = (self.W + self.W.T) / 2  # Enforce symmetry of W
            VW = torch.matmul(V, W_sym)  # (batch, n, k)
            A_reconstructed = torch.matmul(VW, V.transpose(-1, -2))  # (batch, n, n)

        # Add bias term if enabled
        if self.use_bias and self.b is not None:
            if n > self.max_nodes:
                raise ValueError(
                    f"LinearPE received a graph with n={n} nodes but was built "
                    f"with max_nodes={self.max_nodes}; the learnable bias term "
                    f"b ∈ R^max_nodes cannot cover it. Set model.max_nodes to at "
                    f"least the dataset's largest graph (e.g. PROTEINS=620, "
                    f"COLLAB=492, DEEZER=363)."
                )
            # Get bias for actual graph size
            b = self.b[:n]  # (n,)
            # Compute 1 b^T + b 1^T
            # This adds b_i + b_j to entry (i, j)
            ones = torch.ones(n, device=V.device, dtype=V.dtype)
            # outer products: (n, 1) @ (1, n) -> (n, n)
            bias_term = b.unsqueeze(1) @ ones.unsqueeze(0) + ones.unsqueeze(
                1
            ) @ b.unsqueeze(0)
            # Broadcast over batch
            A_reconstructed = A_reconstructed + bias_term.unsqueeze(0)

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "max_nodes": self.max_nodes,
                "use_bias": self.use_bias,
                "asymmetric": self.asymmetric,
            }
        )
        return config
