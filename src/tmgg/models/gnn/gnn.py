"""Graph Neural Network models for graph denoising."""

from typing import Any, ClassVar, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    GraphData,
    GraphDistribution,
)
from tmgg.models.layers.eigen_embedding import TruncatedEigenEmbedding
from tmgg.models.layers.gcn import GraphConvolutionLayer

from ..base import (
    EdgeSource,
    GraphModel,
    _coerce_input_to,
    _coerce_output_to,
    read_edge_scalar,
    write_edge_scalar,
)


class GNN(GraphModel):
    """Graph Neural Network for adjacency matrix reconstruction.

    Computes separate X and Y node embeddings and reconstructs the
    adjacency via ``X @ Y.T``. By default the output is symmetrized
    with ``(A + A.T) / 2``; set ``symmetrized_output=False`` to keep
    the raw asymmetric product.
    """

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim_in: int = 10,
        feature_dim_out: int = 10,
        eigenvalue_reg: float = 0.0,
        symmetrized_output: bool = True,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        """Initialize GNN.

        Parameters
        ----------
        num_layers
            Number of graph convolution layers.
        num_terms
            Number of terms in polynomial filters.
        feature_dim_in
            Input feature dimension (truncated eigenvector dimension).
        feature_dim_out
            Output embedding dimension for adjacency reconstruction.
        eigenvalue_reg
            Diagonal regularization for eigendecomposition stability.
        symmetrized_output
            If True (default), symmetrize the reconstructed adjacency
            via ``(A + A.T) / 2`` after the dot product.
        edge_source
            Per-spec input read selector. ``"feat"`` (default) reads from
            ``E_feat`` to match the historical denoising path; ``"class"``
            reads from ``E_class`` for the DiGress-architecture comparison
            panel.
        output_dims_x_class, output_dims_x_feat, output_dims_e_class, output_dims_e_feat
            Per-field output widths required by the Wave 7 architecture
            contract. The GNN is a scalar adjacency denoiser and predicts
            only one edge field; the default ``output_dims_e_feat=1`` puts
            the prediction in ``E_feat``. Set ``output_dims_e_class=2`` to
            instead emit a two-channel ``E_class`` (one-hot ``[1-adj, adj]``).
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim_in = feature_dim_in
        self.feature_dim_out = feature_dim_out
        self.eigenvalue_reg = eigenvalue_reg
        self.symmetrized_output = symmetrized_output
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        self.embedding_layer = TruncatedEigenEmbedding(
            target_dim=feature_dim_in, eigenvalue_reg=eigenvalue_reg
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolutionLayer(num_terms, feature_dim_in))

        self.out_x = nn.Linear(feature_dim_in, feature_dim_out)
        self.out_y = nn.Linear(feature_dim_in, feature_dim_out)

    def _embed(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (X, Y) node embeddings from adjacency.

        Shared between ``forward()`` (which reconstructs the adjacency)
        and ``embeddings()`` (used by hybrid models).
        """
        z = self.embedding_layer(A)

        for layer in self.layers:
            z = layer(A, z)

        return self.out_x(z), self.out_y(z)

    @override
    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> "GraphDistribution | DenseGraphDistribution":
        """Compute denoised graph from input graph data.

        Parameters
        ----------
        data
            Graph features in any of the four concrete carriers.
            Coerced internally to :class:`DenseGraphState`; the dense
            scalar adjacency is then read from the split field selected
            by ``self.edge_source`` (``"feat"`` → ``E_feat``, ``"class"``
            → ``E_class``).
        t
            Diffusion timestep tensor, or None. Concatenated to ``y``
            via the standard two-line pattern; the GNN body itself does
            not currently consume ``y``.
        output_dense
            If True, return a :class:`DenseGraphDistribution`. Default
            False returns the sparse :class:`GraphDistribution` carrier.

        Returns
        -------
        GraphDistribution | DenseGraphDistribution
            Denoised graph distribution with the prediction in the
            configured edge field (default ``E_feat``).
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)
        emb_x, emb_y = self._embed(A)
        result_adj = torch.bmm(emb_x, emb_y.transpose(1, 2))
        if self.symmetrized_output:
            result_adj = (result_adj + result_adj.transpose(1, 2)) / 2
        out_dense_state = write_edge_scalar(
            d, edge_scalar=result_adj, target=self._output_target
        )
        if t is not None:
            new_y = torch.cat([out_dense_state.y, t.unsqueeze(-1)], dim=-1)
            out_dense_state = out_dense_state.replace(y=new_y)
        # Convert dense state-content output to dense distribution-content.
        out_dist = out_dense_state.to_distribution()
        target = DenseGraphDistribution if output_dense else GraphDistribution
        coerced = _coerce_output_to(out_dist, target=target)
        assert isinstance(coerced, (GraphDistribution, DenseGraphDistribution))
        return coerced

    def embeddings(self, data: GraphData) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute node embeddings without adjacency reconstruction.

        Used by hybrid models that need embeddings for further processing.

        Parameters
        ----------
        data
            Graph features in any of the four concrete carriers; coerced
            internally to :class:`DenseGraphState`. The dense scalar
            adjacency is read via the ``edge_source`` selector (matching
            the selector used by ``forward``).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (X, Y) embeddings, each of shape
            ``(batch, n, feature_dim_out)``.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)
        return self._embed(A)

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
            "eigenvalue_reg": self.eigenvalue_reg,
            "symmetrized_output": self.symmetrized_output,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
