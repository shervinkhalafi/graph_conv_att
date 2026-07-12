"""Transformer-internal subclass for DiGress's spectral attention context.

``DenseGraphTransformerData`` extends ``DenseGraphDistribution`` with frozen
spectral attention context (eigvec / eigval) and a frozen binary adjacency.
The transformer's body is distribution-internal (every (i, j) slot has a
learned hidden value), so the base type is ``DenseGraphDistribution``.
``eigvec`` / ``eigval`` / ``binary_adj`` are read-only context that
survives layer iteration unchanged.

``binary_adj`` carries the input-time topology (computed once at the
transformer entry from the categorical input) so GNN projections in
:class:`NodeEdgeBlock` see the original graph rather than the hidden
edge-feature argmax. Without this field the per-layer
``h.dense_adjacency()`` would derive an adjacency from the (high-d,
unstructured) hidden ``E_class`` whose channel-0 value is meaningless,
giving an effectively all-1 mask off-diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from tmgg.data.datasets.graph_types import DenseGraphDistribution


@dataclass(frozen=True)
class DenseGraphTransformerData(DenseGraphDistribution):
    """DenseGraphDistribution + frozen spectral and adjacency context."""

    eigvec: Tensor | None = None  # (B, n_max, k)
    eigval: Tensor | None = None  # (B, k)
    binary_adj: Tensor | None = None  # (B, n_max, n_max), 0/1

    @classmethod
    def from_base(
        cls,
        base: DenseGraphDistribution,
        *,
        eigvec: Tensor | None = None,
        eigval: Tensor | None = None,
        binary_adj: Tensor | None = None,
    ) -> DenseGraphTransformerData:
        return cls(
            num_nodes_per_graph=base.num_nodes_per_graph,
            y=base.y,
            X_class=base.X_class,
            X_feat=base.X_feat,
            E_class=base.E_class,
            E_feat=base.E_feat,
            eigvec=eigvec,
            eigval=eigval,
            binary_adj=binary_adj,
        )

    def to_base(self) -> DenseGraphDistribution:
        """Drop transformer-specific fields. Used at the forward output."""
        return DenseGraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=self.X_class,
            X_feat=self.X_feat,
            E_class=self.E_class,
            E_feat=self.E_feat,
        )
