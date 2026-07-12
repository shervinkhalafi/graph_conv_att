import torch
import torch.nn as nn

from .graph_ops import poly_graph_conv, sym_normalize_adjacency


class GraphConvolutionLayer(nn.Module):
    """Graph convolution layer using polynomial filters."""

    def __init__(self, num_terms: int, num_channels: int):
        """
        Initialize graph convolution layer.

        Args:
            num_terms: Number of terms in polynomial filter
            num_channels: Number of input/output channels
        """
        super().__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels

        self.H = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_channels))
        nn.init.xavier_uniform_(self.H)

        # GELU provides better nonlinearity than Tanh after LayerNorm,
        # since LayerNorm already normalizes to ~N(0,1) where Tanh is nearly linear
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.

        Args:
            A: Adjacency matrix
            X: Input features

        Returns:
            Convolved features
        """
        X = X.to(self.H.dtype)
        A = A.to(self.H.dtype)

        A_norm = sym_normalize_adjacency(A)
        Y_hat = poly_graph_conv(A_norm, X, self.H)
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat


class BareGraphConvolutionLayer(nn.Module):
    """Graph convolution layer for use as attention projection.

    This variant omits LayerNorm and activation, intended for use inside
    attention blocks that have their own normalization. Computes:

        Y = Σ_{i=0}^{num_terms} S^i @ X @ H[i]

    where the graph shift operator ``S`` is either the symmetrically
    normalized adjacency ``D^{-1/2} A D^{-1/2}`` (default,
    ``normalize_adjacency=True``) or the raw input ``A`` itself
    (``normalize_adjacency=False``). The latter is the right choice
    when the caller has already normalized ``A`` upstream or wants to
    feed a different graph shift operator (e.g. a Laplacian-form
    matrix) directly.

    Parameters
    ----------
    num_terms
        Number of polynomial terms (excluding identity). Total terms is
        num_terms + 1 (including the identity term i=0).
    num_channels
        Input and output feature dimension (same for Q/K/V compatibility).
    normalize_adjacency
        If True (default), apply ``sym_normalize_adjacency`` to ``A``
        inside ``forward`` before raising it to powers. If False, use
        the input ``A`` directly as the shift operator. Defaulting to
        True preserves historical behaviour. Setting False makes the
        caller responsible for keeping the spectral radius of ``A``
        bounded so the polynomial filter does not overflow on dense
        graphs.
    """

    def __init__(
        self,
        num_terms: int,
        num_channels: int,
        normalize_adjacency: bool = True,
    ) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels
        self.normalize_adjacency = normalize_adjacency

        self.H = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_channels))
        nn.init.xavier_uniform_(self.H)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Apply polynomial graph convolution.

        Parameters
        ----------
        A
            Adjacency matrix (or pre-normalized graph shift operator
            when ``self.normalize_adjacency`` is False), shape
            ``(batch, n, n)``.
        X
            Node features, shape ``(batch, n, num_channels)``.

        Returns
        -------
        torch.Tensor
            Convolved features, shape ``(batch, n, num_channels)``.
        """
        X = X.to(self.H.dtype)
        A = A.to(self.H.dtype)

        S = sym_normalize_adjacency(A) if self.normalize_adjacency else A
        return poly_graph_conv(S, X, self.H)
