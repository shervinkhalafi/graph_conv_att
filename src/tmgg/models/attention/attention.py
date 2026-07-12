"""Attention-based models for graph denoising."""

from typing import Any, ClassVar, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
)
from tmgg.models.layers.mha_layer import MultiHeadSelfAttention

from ..base import (
    EdgeSource,
    GraphModel,
    _coerce_input_to,
    _coerce_output_to,
    read_edge_scalar,
)


class MultiLayerAttention(GraphModel):
    """Multi-layer attention model for graph denoising."""

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_residual: bool = True,
        edge_source: EdgeSource = "feat",
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ):
        """
        Initialize the Multi-Layer Attention module.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            d_k: Dimension of keys (default: d_model // num_heads)
            d_v: Dimension of values (default: d_model // num_heads)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
            use_residual: Whether to apply residual connections in attention layers
            edge_source: Per-spec input read selector (``"feat"`` / ``"class"``).
            output_dims_x_class, output_dims_x_feat, output_dims_e_class, output_dims_e_feat:
                Per-field output widths required by the Wave 7 architecture
                contract. Default ``output_dims_e_feat=1`` preserves the
                historical denoising path.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        self.dropout = dropout
        self.bias = bias
        self.use_residual = use_residual
        self.edge_source: EdgeSource = edge_source
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

        # Create stack of attention layers
        self.layers = nn.ModuleList(
            [
                MultiHeadSelfAttention(
                    d_model,
                    num_heads=num_heads,
                    d_k=self.d_k,
                    d_v=self.d_v,
                    dropout=dropout,
                    bias=bias,
                    use_residual=use_residual,
                )
                for _ in range(num_layers)
            ]
        )

    def apply_attention(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply attention layers to a raw tensor.

        Exposed for hybrid models that need raw feature-tensor processing
        without the GraphData wrapping/unwrapping of ``forward()``.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, seq, d_model)``.
        mask
            Optional attention mask.

        Returns
        -------
        torch.Tensor
            Processed tensor, same shape as input.
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    @override
    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> "GraphDistribution | DenseGraphDistribution":
        """Forward pass through all attention layers.

        Coerces the input to a :class:`DenseGraphState`, reads the dense
        scalar adjacency, runs the stacked self-attention layers as
        before, and writes the prediction to the configured split edge
        field. The state-typed output is then converted to a distribution
        and emitted in the requested layout via :func:`_coerce_output_to`.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)
        A = read_edge_scalar(d, self.edge_source)
        out_adj = self.apply_attention(A)
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

    @override
    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_k": self.d_k,
            "d_v": self.d_v,
            "dropout": self.dropout,
            "bias": self.bias,
            "use_residual": self.use_residual,
            "edge_source": self.edge_source,
            "output_dims_x_class": self.output_dims_x_class,
            "output_dims_x_feat": self.output_dims_x_feat,
            "output_dims_e_class": self.output_dims_e_class,
            "output_dims_e_feat": self.output_dims_e_feat,
        }
