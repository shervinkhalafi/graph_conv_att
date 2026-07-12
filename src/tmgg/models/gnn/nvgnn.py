from typing import Any, ClassVar, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    GraphData,
    GraphDistribution,
)
from tmgg.models.layers.eigen_embedding import TruncatedEigenEmbedding
from tmgg.models.layers.nvgcn_layer import NodeVarGraphConvolutionLayer

from ..base import (
    EdgeSource,
    GraphModel,
    _coerce_input_to,
    _coerce_output_to,
    read_edge_scalar,
    write_edge_scalar,
)


class NodeVarGNN(GraphModel):
    """Node-variant Graph Neural Network."""

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        num_layers: int,
        num_terms: int = 3,
        feature_dim: int = 10,
        eigenvalue_reg: float = 0.0,
        symmetrized_output: bool = True,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        """Initialize Node-variant GNN.

        Parameters
        ----------
        num_layers
            Number of layers.
        num_terms
            Number of polynomial terms.
        feature_dim
            Feature dimension.
        eigenvalue_reg
            Diagonal regularization for eigendecomposition stability.
        symmetrized_output
            If True (default), symmetrize the reconstructed adjacency
            via ``(A + A.T) / 2`` after the dot product.
        edge_source
            Per-spec input read selector (``"feat"`` reads ``E_feat``,
            ``"class"`` reads ``E_class``). Default ``"feat"`` matches
            the historical denoising path.
        output_dims_x_class, output_dims_x_feat, output_dims_e_class, output_dims_e_feat
            Per-field output widths required by the Wave 7 architecture
            contract. Default ``output_dims_e_feat=1`` puts the prediction
            in ``E_feat``; set ``output_dims_e_class=2`` to instead emit a
            two-channel ``E_class``.
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim = feature_dim
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
            target_dim=feature_dim, eigenvalue_reg=eigenvalue_reg
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                NodeVarGraphConvolutionLayer(num_terms, feature_dim, feature_dim)
            )

        self.out_x = nn.Linear(feature_dim, feature_dim)
        self.out_y = nn.Linear(feature_dim, feature_dim)

    @override
    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> "GraphDistribution | DenseGraphDistribution":
        """Compute denoised graph from input graph data.

        Coerces ``data`` to a :class:`DenseGraphState`, reads the dense
        scalar adjacency through ``read_edge_scalar`` (respecting
        ``self.edge_source``), and writes the prediction to the configured
        split edge field via ``write_edge_scalar``. Output is returned as
        a sparse :class:`GraphDistribution` by default, or a
        :class:`DenseGraphDistribution` when ``output_dense=True``.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        x = read_edge_scalar(d, self.edge_source)
        z = self.embedding_layer(x)

        for layer in self.layers:
            z = layer(x, z)
        emb_x = self.out_x(z)
        emb_y = self.out_y(z)
        result_adj = torch.bmm(emb_x, emb_y.transpose(1, 2))
        if self.symmetrized_output:
            result_adj = (result_adj + result_adj.transpose(1, 2)) / 2
        out_dense_state = write_edge_scalar(
            d, edge_scalar=result_adj, target=self._output_target
        )
        if t is not None:
            new_y = torch.cat([out_dense_state.y, t.unsqueeze(-1)], dim=-1)
            out_dense_state = out_dense_state.replace(y=new_y)
        out_dist = out_dense_state.to_distribution()
        target = DenseGraphDistribution if output_dense else GraphDistribution
        coerced = _coerce_output_to(out_dist, target=target)
        assert isinstance(coerced, (GraphDistribution, DenseGraphDistribution))
        return coerced

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim": self.feature_dim,
            "eigenvalue_reg": self.eigenvalue_reg,
            "symmetrized_output": self.symmetrized_output,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
