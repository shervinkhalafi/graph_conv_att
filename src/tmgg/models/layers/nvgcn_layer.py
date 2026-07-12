"""Node-variant graph convolution layer.

REDESIGNED: The original implementation used node-specific parameters with shape
(num_terms+1, num_channels, num_nodes), which caused:
- Dynamic layer recreation breaking optimizer state
- Parameter truncation/padding losing learned weights
- Incompatibility with variable graph sizes

The new design uses node-agnostic parameters with shape
(num_terms+1, num_channels_in, num_channels_out), supporting any graph size.
"""

import torch
import torch.nn as nn

from .graph_ops import poly_graph_conv, sym_normalize_adjacency


class NodeVarGraphConvolutionLayer(nn.Module):
    """Graph convolution layer with polynomial filters.

    This layer computes a polynomial graph convolution:
        Y = sum_{i=0}^{num_terms} A^i @ X @ H[i]

    where A is the (normalized) adjacency matrix and H[i] are learnable weight
    matrices. Unlike the original "node-variant" design, parameters are shared
    across all nodes, enabling support for variable graph sizes.

    Parameters
    ----------
    num_terms : int
        Number of polynomial terms (excluding the identity term).
    num_channels_in : int
        Input feature dimension.
    num_channels_out : int
        Output feature dimension.
    """

    def __init__(
        self, num_terms: int, num_channels_in: int, num_channels_out: int
    ) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.num_channels_in = num_channels_in
        self.num_channels_out = num_channels_out

        # Node-agnostic parameters: (num_terms+1, in_channels, out_channels)
        self.H = nn.Parameter(
            torch.randn(num_terms + 1, self.num_channels_in, self.num_channels_out)
        )
        nn.init.xavier_uniform_(self.H)

        # GELU provides better nonlinearity than Tanh after LayerNorm
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.num_channels_out)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of graph convolution.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, num_nodes, num_nodes).
        X : torch.Tensor
            Input features of shape (batch, num_nodes, num_channels_in).

        Returns
        -------
        torch.Tensor
            Output features of shape (batch, num_nodes, num_channels_out).
        """
        X = X.to(self.H.dtype)
        A = A.to(self.H.dtype)

        A_norm = sym_normalize_adjacency(A)
        Y = poly_graph_conv(A_norm, X, self.H)
        Y = self.layer_norm(Y)
        Y = self.activation(Y)
        return Y
