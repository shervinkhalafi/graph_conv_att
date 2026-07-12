"""MLP baseline model for graph denoising.

A simple multi-layer perceptron that flattens the adjacency matrix, processes
it through hidden layers, and reshapes back. Tests whether the training
pipeline can train any neural network at all.
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


class MLPBaseline(GraphModel):
    """MLP baseline: flatten -> MLP -> reshape.

    This model treats the adjacency matrix as a flat vector, processes it
    through a standard MLP, and reshapes back to a matrix. It ignores graph
    structure entirely but should be able to learn if the training loop works.

    Use this model as a sanity check: if an MLP cannot learn to denoise,
    the training pipeline itself has issues.

    Parameters
    ----------
    max_nodes
        Maximum number of nodes in the graph.
    hidden_dim
        Size of hidden layers in the MLP. Default 256.
    num_layers
        Number of hidden layers. Default 2.

    Attributes
    ----------
    mlp : nn.Sequential
        The MLP architecture with ReLU activations.

    Examples
    --------
    >>> model = MLPBaseline(max_nodes=32, hidden_dim=256)
    >>> A = torch.rand(4, 32, 32)  # batch of 4
    >>> logits = model(A)  # Returns raw logits
    >>> probs = model.predict(logits)  # Apply sigmoid for [0, 1]
    """

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.flatten_dim = max_nodes * max_nodes
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        # Build MLP
        layers = []
        in_dim = self.flatten_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, self.flatten_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> "GraphDistribution | DenseGraphDistribution":
        """Apply MLP to the flattened adjacency.

        Coerces the input to a :class:`DenseGraphState`, reads the dense
        scalar adjacency through :func:`read_edge_scalar`, runs the MLP on
        the flattened representation, and writes the prediction to the
        configured split edge field. The ``t`` tensor is appended to the
        global ``y`` when supplied. The state-typed output is then
        converted to a distribution and emitted in the requested layout
        via :func:`_coerce_output_to`.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)
        B, N, _ = A.shape

        if self.max_nodes > N:
            A_padded = torch.zeros(B, self.max_nodes, self.max_nodes, device=A.device)
            A_padded[:, :N, :N] = A
        else:
            A_padded = A

        flat = A_padded.view(B, -1)
        out_flat = self.mlp(flat)
        out_adj = out_flat.view(B, self.max_nodes, self.max_nodes)

        if self.max_nodes > N:
            out_adj = out_adj[:, :N, :N]

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
            "model_type": "MLPBaseline",
            "max_nodes": self.max_nodes,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "flatten_dim": self.flatten_dim,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
