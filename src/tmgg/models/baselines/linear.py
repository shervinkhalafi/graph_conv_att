"""Linear baseline model for graph denoising.

A minimal model that learns a direct linear transformation of the adjacency matrix.
Initialized at identity to enable learning from a reasonable starting point.
"""

from typing import Any, ClassVar

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
)

from ..base import (
    EdgeSource,
    GraphModel,
    _coerce_input_to,
    _coerce_output_to,
    read_edge_scalar,
)


class LinearBaseline(GraphModel):
    """Linear transformation baseline: A_pred = W @ A @ W.T + b.

    This model applies a learnable linear transformation to the input adjacency
    matrix. Initialized with W=I (identity) so the initial output equals the
    input, providing a stable starting point for gradient descent.

    Use this model to verify the training pipeline works correctly. If this
    simple model cannot learn, the issue lies in the training loop or data,
    not in model architecture complexity.

    Parameters
    ----------
    max_nodes
        Maximum number of nodes in the graph. Determines parameter dimensions.

    Attributes
    ----------
    W : nn.Parameter
        Learnable weight matrix of shape (max_nodes, max_nodes).
    b : nn.Parameter
        Learnable bias matrix of shape (max_nodes, max_nodes).

    Examples
    --------
    >>> model = LinearBaseline(max_nodes=32)
    >>> A = torch.eye(32).unsqueeze(0)  # batch of 1
    >>> logits = model(A)  # Returns raw logits
    >>> probs = model.predict(logits)  # Apply sigmoid for [0, 1]
    """

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        max_nodes: int,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        # Initialize W at identity for stable starting point
        self.W = nn.Parameter(torch.eye(max_nodes))
        self.b = nn.Parameter(torch.zeros(max_nodes, max_nodes))

    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> "GraphDistribution | DenseGraphDistribution":
        """Apply learnable linear transformation to the dense adjacency.

        Coerces the input to a :class:`DenseGraphState`, reads the dense
        scalar adjacency through :func:`read_edge_scalar` (which respects
        ``self.edge_source``), applies ``W A W^T + b``, and writes the
        prediction to the configured split edge field. When ``t`` is
        provided it is appended to the global ``y`` tensor following the
        spec's two-line pattern. The state-typed output is then converted
        to a distribution and emitted in the requested layout via
        :func:`_coerce_output_to`.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)
        B, N, _ = A.shape

        W = self.W[:N, :N]
        b = self.b[:N, :N]

        W_expanded = W.unsqueeze(0).expand(B, -1, -1)
        out_adj = torch.bmm(torch.bmm(W_expanded, A), W_expanded.transpose(-2, -1))
        out_adj = out_adj + b.unsqueeze(0)

        if self._output_target == "feat":
            out_dense = DenseGraphState.from_structure_only(d.node_mask, out_adj)
        else:  # "class"
            out_dense = DenseGraphState.from_edge_scalar(
                out_adj, node_mask=d.node_mask, target="E_class"
            )
        out_dense = out_dense.replace(y=d.y)
        if t is not None:
            new_y = torch.cat([out_dense.y, t.unsqueeze(-1)], dim=-1)
            out_dense = out_dense.replace(y=new_y)
        out_dist = out_dense.to_distribution()
        target = DenseGraphDistribution if output_dense else GraphDistribution
        return _coerce_output_to(out_dist, target=target)  # type: ignore[return-value]

    def get_config(self) -> dict[str, Any]:
        """Return model configuration."""
        return {
            "model_type": "LinearBaseline",
            "max_nodes": self.max_nodes,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
