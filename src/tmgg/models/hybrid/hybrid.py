"""Hybrid models combining GNN embeddings with transformer denoising."""

from typing import Any, ClassVar, override

import torch
import torch.nn as nn

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
)

from ..attention import MultiLayerAttention
from ..base import (
    BaseModel,
    EdgeSource,
    EmbeddingProvider,
    GraphModel,
    _coerce_input_to,
    _coerce_output_to,
)
from ..gnn import GNN


class SequentialDenoisingModel(GraphModel):
    """Sequential model combining GNN embeddings with attention-based denoising.

    First generates embeddings using a GNN, then applies a transformer
    to denoise these embeddings, and finally reconstructs the adjacency matrix.
    """

    embedding_model: EmbeddingProvider
    denoising_model: nn.Module | None

    _internal_in: ClassVar[type] = DenseGraphDistribution
    _internal_out: ClassVar[type] = DenseGraphDistribution

    def __init__(
        self,
        embedding_model: EmbeddingProvider,
        denoising_model: nn.Module | None = None,
        output_dims_x_class: int | None = None,
        output_dims_x_feat: int | None = None,
        output_dims_e_class: int | None = None,
        output_dims_e_feat: int | None = 1,
    ) -> None:
        """Initialize the sequential denoising model.

        Parameters
        ----------
        embedding_model
            Model satisfying ``EmbeddingProvider``: must expose
            ``embeddings(GraphData) -> (X, Y)`` and ``get_config()``.
        denoising_model
            Optional model for denoising concatenated embeddings. If it is a
            ``MultiLayerAttention``, the raw ``apply_attention()`` path is
            used; otherwise it is called directly on the feature tensor.
        output_dims_x_class, output_dims_x_feat, output_dims_e_class, output_dims_e_feat
            Per-field output widths required by the Wave 7 architecture
            contract. The hybrid model delegates input reads to its
            embedding provider (whose ``edge_source`` drives the scalar
            adjacency read) and writes the final prediction to the
            configured split edge field. Default ``output_dims_e_feat=1``
            preserves the historical denoising path.
        """
        super().__init__()
        if not isinstance(embedding_model, EmbeddingProvider):
            raise TypeError(
                f"embedding_model must satisfy EmbeddingProvider protocol "
                f"(requires embeddings() and get_config() methods), "
                f"got {type(embedding_model).__name__}"
            )
        self.embedding_model = embedding_model  # pyright: ignore[reportIncompatibleVariableOverride]
        self.denoising_model = denoising_model
        self.output_dims_x_class = output_dims_x_class
        self.output_dims_x_feat = output_dims_x_feat
        self.output_dims_e_class = output_dims_e_class
        self.output_dims_e_feat = output_dims_e_feat
        self._output_target: EdgeSource = (
            "class" if output_dims_e_class is not None else "feat"
        )

    @override
    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> "GraphDistribution | DenseGraphDistribution":
        """Combine GNN embedding and transformer denoising.

        Coerces the input to a :class:`DenseGraphState` and delegates
        edge-tensor reading to the embedding provider (whose own
        ``edge_source`` drives that choice). The hybrid body stacks the
        provider's embeddings, optionally denoises them, and reconstructs
        an adjacency scalar; the scalar is written to the configured
        split edge field. The state-typed output is then converted to a
        distribution and emitted in the requested layout via
        :func:`_coerce_output_to`.
        """
        d = _coerce_input_to(data, target=DenseGraphDistribution)
        assert isinstance(d, DenseGraphDistribution)

        # Generate embeddings via the EmbeddingProvider protocol
        X, Y = self.embedding_model.embeddings(d)

        # Concatenate embeddings
        Z = torch.cat([X, Y], dim=2)  # Shape: (batch_size, num_nodes, 2*feature_dim)

        # Apply denoising if available
        if self.denoising_model is not None:
            # Use raw tensor path for MultiLayerAttention; fall back to direct call
            if isinstance(self.denoising_model, MultiLayerAttention):
                Z_denoised = self.denoising_model.apply_attention(Z)
            else:
                Z_denoised = self.denoising_model(Z)
            # Validate shape match for residual connection
            assert Z_denoised.shape == Z.shape, (
                f"Denoising model output shape {Z_denoised.shape} != input shape {Z.shape}. "
                "Check d_model matches 2 * feature_dim_out."
            )
            Z_pred = Z + Z_denoised
        else:
            Z_pred = Z

        # Split back into X and Y embeddings
        feature_dim = X.shape[2]
        X_pred = Z_pred[:, :, :feature_dim]
        Y_pred = Z_pred[:, :, feature_dim:]

        # Reconstruct adjacency matrix
        A_recon = torch.bmm(X_pred, Y_pred.transpose(1, 2))
        if self._output_target == "feat":
            out_dense = DenseGraphState.from_structure_only(d.node_mask, A_recon)
        else:  # "class"
            out_dense = DenseGraphState.from_edge_scalar(
                A_recon, node_mask=d.node_mask, target="E_class"
            )
        out_dense = out_dense.replace(y=d.y)
        if t is not None:
            new_y = torch.cat([out_dense.y, t.unsqueeze(-1)], dim=-1)
            out_dense = out_dense.replace(y=new_y)
        out_dist = out_dense.to_distribution()
        target = DenseGraphDistribution if output_dense else GraphDistribution
        return _coerce_output_to(out_dist, target=target)  # type: ignore[return-value]

    @override
    def get_config(self) -> dict[str, object]:
        """Get configuration for both embedding and denoising components."""
        embedding_config = self.embedding_model.get_config()

        config: dict[str, object] = {
            "model_type": "SequentialDenoisingModel",
            "embedding_model": embedding_config,
            "has_denoising": self.denoising_model is not None,
        }
        if self.denoising_model is not None and isinstance(
            self.denoising_model, BaseModel
        ):
            config["denoising_model"] = self.denoising_model.get_config()
        return config


def create_sequential_model(
    gnn_config: dict[str, Any],
    transformer_config: dict[str, Any] | None = None,
    *,
    edge_source: EdgeSource = "feat",
    output_dims_x_class: int | None = None,
    output_dims_x_feat: int | None = None,
    output_dims_e_class: int | None = None,
    output_dims_e_feat: int | None = 1,
) -> SequentialDenoisingModel:
    """Create a sequential denoising model from GNN + optional attention config.

    Parameters
    ----------
    gnn_config
        Configuration for the GNN embedding model.
    transformer_config
        Optional configuration for the attention denoising model.
    edge_source
        Per-spec input read selector applied to the inner GNN embedding
        provider. ``"feat"`` (default) reads from ``E_feat``; ``"class"``
        reads from ``E_class`` for the DiGress-architecture comparison
        panel.
    output_dims_x_class, output_dims_x_feat, output_dims_e_class, output_dims_e_feat
        Per-field output widths forwarded to
        :class:`SequentialDenoisingModel`. The hybrid writes its final
        adjacency scalar to the split edge field chosen by the class/feat
        split; defaults reproduce the historical denoising path
        (``output_dims_e_feat=1``).

    Returns
    -------
    SequentialDenoisingModel
        Configured model combining GNN embeddings with optional attention.
    """
    # Create GNN embedding model
    embedding_model = GNN(
        num_layers=gnn_config.get("num_layers", 2),
        num_terms=gnn_config.get("num_terms", 2),
        feature_dim_in=gnn_config.get("feature_dim_in", 20),
        feature_dim_out=gnn_config.get("feature_dim_out", 5),
        eigenvalue_reg=gnn_config.get("eigenvalue_reg", 0.0),
        edge_source=edge_source,
    )

    # Create transformer denoising model if config provided
    denoising_model = None
    if transformer_config is not None:
        # The input dimension for the transformer is 2 * feature_dim_out (concatenated X, Y)
        d_model = 2 * gnn_config.get("feature_dim_out", 5)

        denoising_model = MultiLayerAttention(
            d_model=d_model,
            num_heads=transformer_config.get("num_heads", 4),
            num_layers=transformer_config.get("num_layers", 4),
            d_k=transformer_config.get("d_k", None),
            d_v=transformer_config.get("d_v", None),
            dropout=transformer_config.get("dropout", 0.0),
            bias=transformer_config.get("bias", True),
        )

    return SequentialDenoisingModel(
        embedding_model,
        denoising_model,
        output_dims_x_class=output_dims_x_class,
        output_dims_x_feat=output_dims_x_feat,
        output_dims_e_class=output_dims_e_class,
        output_dims_e_feat=output_dims_e_feat,
    )
