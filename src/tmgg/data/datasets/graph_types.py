# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.
"""Typed containers for categorical graph data.

``GraphData`` is the abstract base of a 2x2 type grid: sparse vs dense
carrier, state vs distribution content. The four concrete leaves
(``GraphState``, ``GraphDistribution``, ``DenseGraphState``,
``DenseGraphDistribution``) all carry the universal fields and split feature
fields (``X_class`` / ``X_feat`` / ``E_class`` / ``E_feat``); at least one
of the edge fields MUST be populated.

The DiGress transformer's spectral attention context (eigenvectors,
eigenvalues) lives on a transformer-internal subclass
``tmgg.models.digress.data_types.DenseGraphTransformerData`` that extends
``DenseGraphDistribution``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    import networkx as nx
    from torch_geometric.data import Batch, Data


@dataclass(frozen=True)
class GraphData:
    """Abstract base for every concrete graph carrier (sparse or dense).

    Carries only the universal fields (per-graph node count, graph-level y).
    Concrete subtypes add carrier-specific fields. This class is not
    instantiated directly; use one of `GraphState`, `GraphDistribution`,
    `DenseGraphState`, `DenseGraphDistribution`.
    """

    num_nodes_per_graph: Tensor  # (B,)        per-graph node count, int64
    y: Tensor  # (B, dy)     graph-level features

    def __post_init__(self) -> None:
        if self.num_nodes_per_graph.dtype != torch.long:
            raise ValueError(
                "GraphData.num_nodes_per_graph must be int64 (torch.long); "
                f"got {self.num_nodes_per_graph.dtype}."
            )
        if self.num_nodes_per_graph.dim() != 1:
            raise ValueError(
                "GraphData.num_nodes_per_graph must be 1D (B,); "
                f"got shape {tuple(self.num_nodes_per_graph.shape)}."
            )

    def replace(self, **kwargs: object) -> Self:
        return _dc_replace(self, **kwargs)

    def type_as(self, x: Tensor) -> Self:
        """Cast tensors to match `x`'s dtype/device. Override per-subtype as needed."""
        raise NotImplementedError("Concrete subclasses must override type_as.")

    def to(self, device: torch.device | str) -> Self:
        """Move all tensors to `device`. Override per-subtype as needed."""
        raise NotImplementedError("Concrete subclasses must override to.")

    def dense_adjacency(self) -> Tensor:
        """Return the binary `(B, n_max, n_max)` adjacency. Override per-subtype."""
        raise NotImplementedError("Concrete subclasses must override dense_adjacency.")


@dataclass(frozen=True)
class _StateGraph(GraphData):
    """Marker: state-content semantics.

    Per-element data on a chosen edge_index (sparse) or per-position content
    over `(B, n_max, n_max, K)` (dense). Examples: DiGress sample (z_t),
    classification logits over input topology, continuous edge weights.
    """

    def to_distribution(self) -> _DistributionGraph:
        raise NotImplementedError


@dataclass(frozen=True)
class _DistributionGraph(GraphData):
    """Marker: distribution-content semantics.

    Per-position values on a complete edge_index (sparse) or
    `(B, n_max, n_max, K)` (dense), typically a probability distribution
    or continuous noise output.
    """

    def argmax(self) -> _StateGraph:
        raise NotImplementedError

    def sample(self, *, generator: torch.Generator | None = None) -> _StateGraph:
        raise NotImplementedError


# ---- Helpers for sparse invariant checks --------------------------------


def _check_sparse_invariants(
    *,
    batch: Tensor,
    num_nodes_per_graph: Tensor,
    edge_index: Tensor,
    edge_class: Tensor | None,
    edge_feat: Tensor | None,
) -> None:
    """Cheap, unconditional invariants shared by GraphState and GraphDistribution.

    Symmetry of edge_index is checked under __debug__ separately
    (expensive: O(sum_E log sum_E)).
    """
    if batch.dtype != torch.long:
        raise ValueError(f"batch must be int64; got {batch.dtype}.")
    if edge_index.dtype != torch.long or edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must be int64 of shape (2, sum_E); got dtype "
            f"{edge_index.dtype} shape {tuple(edge_index.shape)}."
        )
    if edge_class is not None and edge_class.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_class.shape[0] ({edge_class.shape[0]}) must match "
            f"edge_index.shape[1] ({edge_index.shape[1]})."
        )
    if edge_feat is not None and edge_feat.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_feat.shape[0] ({edge_feat.shape[0]}) must match "
            f"edge_index.shape[1] ({edge_index.shape[1]})."
        )
    if edge_class is None and edge_feat is None:
        raise ValueError("At least one of edge_class or edge_feat must be populated.")
    # Endpoint integrity.
    bs = int(num_nodes_per_graph.shape[0])
    counts = torch.bincount(batch, minlength=bs)
    if not torch.equal(counts, num_nodes_per_graph):
        raise ValueError(
            f"batch.bincount() {counts.tolist()} disagrees with "
            f"num_nodes_per_graph {num_nodes_per_graph.tolist()}."
        )
    if edge_index.shape[1] > 0:
        if (edge_index[0] == edge_index[1]).any():
            raise ValueError("edge_index contains self-loops.")
        if not torch.equal(batch[edge_index[0]], batch[edge_index[1]]):
            raise ValueError("edge_index endpoints span different graphs.")
    if __debug__:
        _assert_edge_index_symmetric(edge_index)


def _assert_edge_index_symmetric(edge_index: Tensor) -> None:
    """Verify both directions present for every undirected edge.

    PyG convention: an undirected edge (u, v) appears as (u, v) and (v, u)
    in `edge_index`. The symmetric closure of the column set must equal
    the column set itself.
    """
    if edge_index.shape[1] == 0:
        return
    fwd0 = edge_index[0].to(torch.int64)
    fwd1 = edge_index[1].to(torch.int64)
    # Membership test via row-encoded ints (assumes each node id < 2**31).
    enc_fwd = fwd0 * (1 << 32) + fwd1
    enc_rev = fwd1 * (1 << 32) + fwd0
    if not torch.equal(torch.sort(enc_fwd).values, torch.sort(enc_rev).values):
        raise AssertionError(
            "edge_index is not symmetric (PyG convention requires both "
            "directions for undirected graphs)."
        )


def _no_edge_one_hot_fill(edge_class: Tensor | None) -> Tensor | None:
    """Build a `(d_ec,)` no-edge one-hot fill vector, or None if edge_class is None."""
    if edge_class is None:
        return None
    d_ec = int(edge_class.shape[-1])
    fill = torch.zeros(d_ec, dtype=edge_class.dtype, device=edge_class.device)
    if d_ec >= 1:
        fill[0] = 1.0
    return fill


# ---- Sparse leaf types --------------------------------------------------


@dataclass(frozen=True)
class GraphState(_StateGraph):
    """Sparse + state. PyG-flat layout. edge_index covers a chosen subset of
    off-diagonal pairs; per-element content is unconstrained by type.

    Invariant: sum_E ≤ sum_i n_i (n_i - 1) (no self-loops, both directions
    for undirected).
    """

    batch: Tensor  # (sum_n,)         node → graph index, int64
    x_class: Tensor | None  # (sum_n, d_xc)
    x_feat: Tensor | None  # (sum_n, d_xf)
    edge_index: Tensor  # (2, sum_E)       directed entries (PyG)
    edge_class: Tensor | None  # (sum_E, d_ec)
    edge_feat: Tensor | None  # (sum_E, d_ef)

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_sparse_invariants(
            batch=self.batch,
            num_nodes_per_graph=self.num_nodes_per_graph,
            edge_index=self.edge_index,
            edge_class=self.edge_class,
            edge_feat=self.edge_feat,
        )

    def to(self, device: torch.device | str) -> GraphState:
        return GraphState(
            num_nodes_per_graph=self.num_nodes_per_graph.to(device),
            y=self.y.to(device),
            batch=self.batch.to(device),
            x_class=None if self.x_class is None else self.x_class.to(device),
            x_feat=None if self.x_feat is None else self.x_feat.to(device),
            edge_index=self.edge_index.to(device),
            edge_class=None if self.edge_class is None else self.edge_class.to(device),
            edge_feat=None if self.edge_feat is None else self.edge_feat.to(device),
        )

    def type_as(self, x: Tensor) -> GraphState:
        """Cast floating-point feature tensors to match `x`'s dtype.

        `batch`, `num_nodes_per_graph`, and `edge_index` stay int64.
        """
        return GraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y.type_as(x),
            batch=self.batch,
            x_class=None if self.x_class is None else self.x_class.type_as(x),
            x_feat=None if self.x_feat is None else self.x_feat.type_as(x),
            edge_index=self.edge_index,
            edge_class=None if self.edge_class is None else self.edge_class.type_as(x),
            edge_feat=None if self.edge_feat is None else self.edge_feat.type_as(x),
        )

    def dense_adjacency(self) -> Tensor:
        """Convert to dense and return the binary adjacency."""
        return self.to_dense(
            edge_class_fill=_no_edge_one_hot_fill(self.edge_class),
        ).dense_adjacency()

    def to_dense(
        self,
        *,
        edge_class_fill: Tensor | None = None,
        edge_feat_fill: Tensor | float = 0.0,
        x_class_fill: Tensor | None = None,
        x_feat_fill: Tensor | float = 0.0,
        n_max: int | None = None,
    ) -> DenseGraphState:
        bs = int(self.num_nodes_per_graph.shape[0])
        n_max_actual = (
            int(self.num_nodes_per_graph.max().item()) if n_max is None else int(n_max)
        )
        device = self.batch.device

        # Per-row position within graph.
        cum = torch.zeros(bs + 1, dtype=torch.long, device=device)
        cum[1:] = torch.cumsum(self.num_nodes_per_graph, dim=0)
        pos_in_graph = (
            torch.arange(self.batch.shape[0], device=device) - cum[self.batch]
        )

        X_class_dense: Tensor | None = None
        if self.x_class is not None:
            d_xc = int(self.x_class.shape[-1])
            fill_xc = (
                x_class_fill
                if x_class_fill is not None
                else torch.zeros(d_xc, dtype=self.x_class.dtype, device=device)
            )
            X_class_dense = fill_xc.expand(bs, n_max_actual, d_xc).clone()
            X_class_dense[self.batch, pos_in_graph] = self.x_class

        X_feat_dense: Tensor | None = None
        if self.x_feat is not None:
            d_xf = int(self.x_feat.shape[-1])
            if isinstance(x_feat_fill, Tensor):
                base_xf = x_feat_fill.to(self.x_feat.dtype).to(device)
            else:
                base_xf = torch.full(
                    (d_xf,),
                    float(x_feat_fill),
                    dtype=self.x_feat.dtype,
                    device=device,
                )
            X_feat_dense = base_xf.expand(bs, n_max_actual, d_xf).clone()
            X_feat_dense[self.batch, pos_in_graph] = self.x_feat

        # Edge tensors: scatter from edge_index.
        src_in_graph = self.edge_index[0] - cum[self.batch[self.edge_index[0]]]
        dst_in_graph = self.edge_index[1] - cum[self.batch[self.edge_index[1]]]
        graph_ids = self.batch[self.edge_index[0]]

        E_class_dense: Tensor | None = None
        if self.edge_class is not None:
            d_ec = int(self.edge_class.shape[-1])
            if edge_class_fill is None:
                raise ValueError(
                    "GraphState.to_dense(): edge_class_fill is required when "
                    "edge_class is populated. Pass a (d_ec,) vector — e.g. "
                    "use the `state_to_dense_sample` convenience for "
                    "no-edge-one-hot fill, or `state_to_dense_logits` for zeros."
                )
            fill_ec = edge_class_fill.to(self.edge_class.dtype).to(device)
            if fill_ec.shape != (d_ec,):
                raise ValueError(
                    f"edge_class_fill must have shape ({d_ec},); got "
                    f"{tuple(fill_ec.shape)}."
                )
            E_class_dense = fill_ec.expand(bs, n_max_actual, n_max_actual, d_ec).clone()
            E_class_dense[graph_ids, src_in_graph, dst_in_graph] = self.edge_class
            # Diagonal zeroed.
            diag = torch.arange(n_max_actual, device=device)
            E_class_dense[:, diag, diag, :] = 0.0

        E_feat_dense: Tensor | None = None
        if self.edge_feat is not None:
            d_ef = int(self.edge_feat.shape[-1])
            if isinstance(edge_feat_fill, Tensor):
                base_ef = edge_feat_fill.to(self.edge_feat.dtype).to(device)
            else:
                base_ef = torch.full(
                    (d_ef,),
                    float(edge_feat_fill),
                    dtype=self.edge_feat.dtype,
                    device=device,
                )
            E_feat_dense = base_ef.expand(bs, n_max_actual, n_max_actual, d_ef).clone()
            E_feat_dense[graph_ids, src_in_graph, dst_in_graph] = self.edge_feat
            diag = torch.arange(n_max_actual, device=device)
            E_feat_dense[:, diag, diag, :] = 0.0

        # Zero out padding positions so the dense form matches the
        # complete-pair semantics used by GraphDistribution.to_sparse +
        # DenseGraphDistribution.to_sparse: only valid (b, i) × (b, j) pairs
        # carry content; padding positions are zeroed regardless of fill.
        # This makes the lift-then-carrier and carrier-then-lift paths
        # commute on the entire (B, n_max, n_max, d_ec) tensor.
        arange_n = torch.arange(n_max_actual, device=device)
        valid_per_graph = arange_n.unsqueeze(0) < self.num_nodes_per_graph.unsqueeze(1)
        pair_mask = valid_per_graph.unsqueeze(-1) & valid_per_graph.unsqueeze(-2)
        if E_class_dense is not None:
            E_class_dense = E_class_dense * pair_mask.unsqueeze(-1).to(
                E_class_dense.dtype
            )
        if E_feat_dense is not None:
            E_feat_dense = E_feat_dense * pair_mask.unsqueeze(-1).to(E_feat_dense.dtype)
        if X_class_dense is not None:
            X_class_dense = X_class_dense * valid_per_graph.unsqueeze(-1).to(
                X_class_dense.dtype
            )
        if X_feat_dense is not None:
            X_feat_dense = X_feat_dense * valid_per_graph.unsqueeze(-1).to(
                X_feat_dense.dtype
            )

        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=X_class_dense,
            X_feat=X_feat_dense,
            E_class=E_class_dense,
            E_feat=E_feat_dense,
        )

    def to_distribution(self) -> GraphDistribution:
        # Strategy: scatter to dense (with no-edge fill for categorical, zero
        # for continuous), then call DenseGraphDistribution.to_sparse() —
        # which produces complete edge_index. This preserves active edges'
        # content unchanged while filling inactive positions per the
        # canonical convention.
        fill = _no_edge_one_hot_fill(self.edge_class)
        dense_state = self.to_dense(edge_class_fill=fill)
        dense_dist = DenseGraphDistribution(
            num_nodes_per_graph=dense_state.num_nodes_per_graph,
            y=dense_state.y,
            X_class=dense_state.X_class,
            X_feat=dense_state.X_feat,
            E_class=dense_state.E_class,
            E_feat=dense_state.E_feat,
        )
        return dense_dist.to_sparse()

    @classmethod
    def from_pyg_batch(
        cls,
        batch: Batch,
        *,
        num_atom_types_x: int | None = None,
        num_bond_types_e: int | None = None,
    ) -> GraphState:
        """Construct a GraphState directly from a PyG Batch.

        Replaces the legacy dense-eager `DenseGraphState.from_pyg_batch` for
        the production pipeline (which now starts sparse and densifies on
        demand inside dense-internal models). The sparse output keeps active
        edges in `edge_index` with `edge_class` one-hot per edge (DiGress
        canonical encoding).

        Parameters
        ----------
        batch
            PyG Batch from ``Batch.from_data_list()``.
        num_atom_types_x
            Optional explicit width for the per-node ``x_class`` one-hot
            when the input ``batch`` carries a per-graph ``x`` attribute
            of integer atom-class indices. When ``None`` and ``x`` is
            present, the width is inferred from ``int(x.max()) + 1`` —
            adequate for tests but underspecified for production datasets
            that may not see every class in a given batch.
        num_bond_types_e
            Optional explicit width for the per-edge ``edge_class`` one-hot
            when the input carries integer bond-class indices on
            ``edge_attr``. When absent, the legacy 2-class
            ``[no-edge, edge]`` encoding is used.
        """
        from torch_geometric.utils import remove_self_loops

        bs = int(batch.num_graphs)
        edge_index_in: Tensor = getattr(batch, "edge_index")  # noqa: B009
        batch_vec: Tensor = getattr(batch, "batch")  # noqa: B009
        edge_attr_in: Tensor | None = getattr(batch, "edge_attr", None)
        edge_index_clean, edge_attr_clean = remove_self_loops(
            edge_index_in, edge_attr_in
        )

        node_counts = torch.bincount(batch_vec, minlength=bs)

        # Edge classes (one-hot per edge).
        if edge_attr_clean is None:
            d_ec = 2
            edge_class = torch.zeros(
                edge_index_clean.shape[1],
                d_ec,
                dtype=torch.float32,
                device=edge_index_clean.device,
            )
            edge_class[:, 1] = 1.0  # all retained edges are "edge"
        else:
            if edge_attr_clean.dim() != 1:
                raise ValueError(
                    "from_pyg_batch: edge_attr must be 1D integer indices."
                )
            edge_attr_long = edge_attr_clean.long()
            if num_bond_types_e is None:
                d_ec = int(edge_attr_long.max().item()) + 1
            else:
                d_ec = int(num_bond_types_e)
            edge_class = F.one_hot(edge_attr_long, num_classes=d_ec).to(torch.float32)

        # Node classes (optional, from batch.x).
        x_attr: Tensor | None = getattr(batch, "x", None)
        x_class: Tensor | None = None
        if x_attr is not None:
            if x_attr.dim() != 1:
                raise ValueError(
                    "from_pyg_batch: batch.x must be 1D integer atom-class indices."
                )
            if torch.is_floating_point(x_attr):
                raise ValueError(
                    "from_pyg_batch: batch.x must be integer dtype "
                    "(atom-class indices)."
                )
            idx_long = x_attr.long()
            if num_atom_types_x is None:
                d_xc = int(idx_long.max().item()) + 1
            else:
                d_xc = int(num_atom_types_x)
            x_class = F.one_hot(idx_long, num_classes=d_xc).to(torch.float32)

        y = torch.zeros(bs, 0, device=batch_vec.device)

        return cls(
            num_nodes_per_graph=node_counts.long(),
            y=y,
            batch=batch_vec.long(),
            x_class=x_class,
            x_feat=None,
            edge_index=edge_index_clean.long(),
            edge_class=edge_class,
            edge_feat=None,
        )

    def to_networkx_list(self) -> list[nx.Graph[Any]]:
        """Expand the batched GraphState into a per-graph nx Graph list.

        PyG-flat layout is decoded directly; the dense detour is avoided.
        Each graph is a simple, undirected ``nx.Graph`` with ``x_class``
        per-node and ``e_class`` per-edge attributes when those fields
        are populated.
        """
        import networkx as nx

        bs = int(self.num_nodes_per_graph.shape[0])
        result: list[nx.Graph[Any]] = []
        device = self.batch.device

        cum = torch.zeros(bs + 1, dtype=torch.long, device=device)
        cum[1:] = torch.cumsum(self.num_nodes_per_graph, dim=0)

        for b in range(bs):
            n_b = int(self.num_nodes_per_graph[b].item())
            edge_mask = self.batch[self.edge_index[0]] == b
            ei = self.edge_index[:, edge_mask] - cum[b]
            g = nx.Graph()
            g.add_nodes_from(range(n_b))

            # Per-node x_class index (argmax over x_class[b]).
            if self.x_class is not None:
                node_mask = self.batch == b
                xc_b = self.x_class[node_mask].argmax(dim=-1).detach().cpu().tolist()
                for node, idx in enumerate(xc_b):
                    g.nodes[node]["x_class"] = int(idx)

            # Per-edge e_class: argmax over edge_class for this graph's edges.
            if self.edge_class is not None:
                ec_b = self.edge_class[edge_mask].argmax(dim=-1).detach().cpu().tolist()
            else:
                ec_b = None

            # Take upper-triangle to avoid duplicate edges from PyG bidirectionality.
            ei_t = ei.t().tolist()
            for k, (u, v) in enumerate(ei_t):
                if u < v:
                    if ec_b is not None:
                        g.add_edge(u, v, e_class=int(ec_b[k]))
                    else:
                        g.add_edge(u, v)
            result.append(g)
        return result

    def to_networkx(self, batch_index: int = 0) -> nx.Graph[Any]:
        """Single-graph ``nx.Graph`` from the batch's row ``batch_index``."""
        return self.to_networkx_list()[batch_index]

    @classmethod
    def collate(cls, graphs: list[GraphState]) -> GraphState:
        """Collate a list of single-graph GraphState into a batch.

        Each input GraphState should represent ONE graph (B=1). Output is
        a multi-graph batch with concatenated node/edge tensors and
        ``batch`` / ``num_nodes_per_graph`` updated.
        """
        if not graphs:
            raise ValueError("GraphState.collate(): empty list.")
        device = graphs[0].batch.device

        num_nodes = torch.tensor(
            [int(g.num_nodes_per_graph.sum().item()) for g in graphs],
            dtype=torch.long,
            device=device,
        )

        # Offsets per graph for re-indexing edge_index.
        cum = torch.zeros(len(graphs) + 1, dtype=torch.long, device=device)
        cum[1:] = torch.cumsum(num_nodes, dim=0)

        # Concatenate batch (each input has batch=0; we offset by graph index).
        batch = torch.cat(
            [
                torch.full(
                    (int(num_nodes[i].item()),),
                    i,
                    dtype=torch.long,
                    device=device,
                )
                for i in range(len(graphs))
            ]
        )

        edge_index = torch.cat(
            [g.edge_index + cum[i] for i, g in enumerate(graphs)],
            dim=1,
        )

        # Concatenate features (None if all None; raise if mixed).
        def _cat_or_none(name: str, dim: int) -> Tensor | None:
            firsts = [getattr(g, name) for g in graphs]
            if all(f is None for f in firsts):
                return None
            if any(f is None for f in firsts):
                raise ValueError(
                    f"GraphState.collate: {name} populated inconsistently across graphs."
                )
            return torch.cat(firsts, dim=dim)

        x_class = _cat_or_none("x_class", 0)
        x_feat = _cat_or_none("x_feat", 0)
        edge_class = _cat_or_none("edge_class", 0)
        edge_feat = _cat_or_none("edge_feat", 0)

        # y: stack per-graph y rows (each input has y of shape (1, dy)).
        y = torch.cat([g.y for g in graphs], dim=0)

        return cls(
            num_nodes_per_graph=num_nodes,
            y=y,
            batch=batch,
            x_class=x_class,
            x_feat=x_feat,
            edge_index=edge_index,
            edge_class=edge_class,
            edge_feat=edge_feat,
        )


@dataclass(frozen=True)
class GraphDistribution(_DistributionGraph):
    """Sparse + distribution. edge_index covers EVERY off-diagonal pair within
    each graph; edge_class / edge_feat carry per-pair content (logits/probs
    for diffusion outputs, continuous values for Gaussian noise).

    Invariant: sum_E == sum_i n_i (n_i - 1).
    """

    batch: Tensor
    x_class: Tensor | None
    x_feat: Tensor | None
    edge_index: Tensor  # (2, sum_E_complete)
    edge_class: Tensor | None
    edge_feat: Tensor | None

    def __post_init__(self) -> None:
        super().__post_init__()
        # Completeness check FIRST: an incomplete edge_index is also
        # asymmetric, but the more informative error is "incomplete" not
        # "asymmetric". Only after a complete-pair edge_index do the
        # general sparse invariants (dtype, endpoint integrity, symmetry
        # under __debug__) become meaningful.
        n = self.num_nodes_per_graph
        expected_E = int((n * (n - 1)).sum().item())
        actual_E = int(self.edge_index.shape[1])
        if actual_E != expected_E:
            raise ValueError(
                f"GraphDistribution requires complete off-diagonal edge_index: "
                f"expected sum n_i(n_i-1) = {expected_E}, got {actual_E}."
            )
        _check_sparse_invariants(
            batch=self.batch,
            num_nodes_per_graph=self.num_nodes_per_graph,
            edge_index=self.edge_index,
            edge_class=self.edge_class,
            edge_feat=self.edge_feat,
        )

    def to(self, device: torch.device | str) -> GraphDistribution:
        return GraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph.to(device),
            y=self.y.to(device),
            batch=self.batch.to(device),
            x_class=None if self.x_class is None else self.x_class.to(device),
            x_feat=None if self.x_feat is None else self.x_feat.to(device),
            edge_index=self.edge_index.to(device),
            edge_class=None if self.edge_class is None else self.edge_class.to(device),
            edge_feat=None if self.edge_feat is None else self.edge_feat.to(device),
        )

    def type_as(self, x: Tensor) -> GraphDistribution:
        return GraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y.type_as(x),
            batch=self.batch,
            x_class=None if self.x_class is None else self.x_class.type_as(x),
            x_feat=None if self.x_feat is None else self.x_feat.type_as(x),
            edge_index=self.edge_index,
            edge_class=None if self.edge_class is None else self.edge_class.type_as(x),
            edge_feat=None if self.edge_feat is None else self.edge_feat.type_as(x),
        )

    def dense_adjacency(self) -> Tensor:
        return self.to_dense().dense_adjacency()

    def to_dense(self) -> DenseGraphDistribution:
        bs = int(self.num_nodes_per_graph.shape[0])
        n_max = int(self.num_nodes_per_graph.max().item())
        device = self.batch.device

        cum = torch.zeros(bs + 1, dtype=torch.long, device=device)
        cum[1:] = torch.cumsum(self.num_nodes_per_graph, dim=0)
        pos_in_graph = (
            torch.arange(self.batch.shape[0], device=device) - cum[self.batch]
        )

        X_class_d: Tensor | None = None
        if self.x_class is not None:
            d_xc = int(self.x_class.shape[-1])
            X_class_d = torch.zeros(
                bs, n_max, d_xc, dtype=self.x_class.dtype, device=device
            )
            X_class_d[self.batch, pos_in_graph] = self.x_class

        X_feat_d: Tensor | None = None
        if self.x_feat is not None:
            d_xf = int(self.x_feat.shape[-1])
            X_feat_d = torch.zeros(
                bs, n_max, d_xf, dtype=self.x_feat.dtype, device=device
            )
            X_feat_d[self.batch, pos_in_graph] = self.x_feat

        src = self.edge_index[0] - cum[self.batch[self.edge_index[0]]]
        dst = self.edge_index[1] - cum[self.batch[self.edge_index[1]]]
        graphs = self.batch[self.edge_index[0]]

        E_class_d: Tensor | None = None
        if self.edge_class is not None:
            d_ec = int(self.edge_class.shape[-1])
            E_class_d = torch.zeros(
                bs,
                n_max,
                n_max,
                d_ec,
                dtype=self.edge_class.dtype,
                device=device,
            )
            E_class_d[graphs, src, dst] = self.edge_class
            diag = torch.arange(n_max, device=device)
            E_class_d[:, diag, diag, :] = 0.0

        E_feat_d: Tensor | None = None
        if self.edge_feat is not None:
            d_ef = int(self.edge_feat.shape[-1])
            E_feat_d = torch.zeros(
                bs,
                n_max,
                n_max,
                d_ef,
                dtype=self.edge_feat.dtype,
                device=device,
            )
            E_feat_d[graphs, src, dst] = self.edge_feat
            diag = torch.arange(n_max, device=device)
            E_feat_d[:, diag, diag, :] = 0.0

        return DenseGraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=X_class_d,
            X_feat=X_feat_d,
            E_class=E_class_d,
            E_feat=E_feat_d,
        )

    def argmax(self) -> GraphState:
        return self.to_dense().argmax().to_sparse()

    def sample(self, *, generator: torch.Generator | None = None) -> GraphState:
        return self.to_dense().sample(generator=generator).to_sparse()


@dataclass(frozen=True)
class DenseGraphState(_StateGraph):
    """Dense + state. (B, n_max, ...) padded carrier with state-typed content.

    Content invariant is convention-bound, not enforced (e.g., one-hot for
    DiGress samples, logits for classification predictions). The TYPE only
    guarantees you are *interpreting* this data as state.

    Parameters
    ----------
    X_class : Tensor, optional
        Categorical node features (one-hot / PMF), shape
        ``(bs, n, dx_class)``. ``None`` for structure-only graphs.
    X_feat : Tensor, optional
        Continuous node features, shape ``(bs, n, dx_feat)``.
    E_class : Tensor, optional
        Categorical edge features (one-hot / PMF), shape
        ``(bs, n, n, de_class)``. Channel 0 conventionally encodes "no edge".
    E_feat : Tensor, optional
        Continuous edge features, shape ``(bs, n, n, de_feat)``. Single-channel
        adjacency weights use ``de_feat == 1``.

    Notes
    -----
    The legacy ``node_mask`` field is now derived as a :func:`functools.cached_property`
    from ``num_nodes_per_graph`` and the populated split tensors' ``n_max``.
    """

    X_class: Tensor | None = None
    X_feat: Tensor | None = None
    E_class: Tensor | None = None
    E_feat: Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        spec_ref = "internal notes (not included)"

        if self.E_class is None and self.E_feat is None:
            raise ValueError(
                f"DenseGraphState requires at least one of E_class or E_feat "
                f"populated (see {spec_ref})."
            )

        bs = int(self.num_nodes_per_graph.shape[0])

        def _leading(name: str, t: Tensor, expected: tuple[int, ...]) -> None:
            got = tuple(int(s) for s in t.shape[: len(expected)])
            if got != expected:
                raise ValueError(
                    f"DenseGraphState.{name} leading dims {got} must match "
                    f"({expected[0]}, n_max...)."
                )

        if self.X_class is not None:
            n_max = int(self.X_class.shape[1])
            _leading("X_class", self.X_class, (bs, n_max))
        elif self.X_feat is not None:
            n_max = int(self.X_feat.shape[1])
            _leading("X_feat", self.X_feat, (bs, n_max))
        elif self.E_class is not None:
            n_max = int(self.E_class.shape[1])
        else:
            assert self.E_feat is not None  # invariant above
            n_max = int(self.E_feat.shape[1])

        if self.E_class is not None:
            _leading("E_class", self.E_class, (bs, n_max, n_max))
        if self.E_feat is not None:
            _leading("E_feat", self.E_feat, (bs, n_max, n_max))
        if self.X_class is not None:
            _leading("X_class", self.X_class, (bs, n_max))
        if self.X_feat is not None:
            _leading("X_feat", self.X_feat, (bs, n_max))

    @cached_property
    def node_mask(self) -> Tensor:
        bs = int(self.num_nodes_per_graph.shape[0])
        if self.E_class is not None:
            n_max = int(self.E_class.shape[1])
        elif self.E_feat is not None:
            n_max = int(self.E_feat.shape[1])
        elif self.X_class is not None:
            n_max = int(self.X_class.shape[1])
        else:
            assert self.X_feat is not None
            n_max = int(self.X_feat.shape[1])
        arange = torch.arange(n_max, device=self.num_nodes_per_graph.device)
        return arange.unsqueeze(0).expand(bs, -1) < self.num_nodes_per_graph.unsqueeze(
            1
        )

    def dense_adjacency(self) -> Tensor:
        if self.E_class is not None and self.E_class.shape[-1] > 1:
            adj = (self.E_class.argmax(dim=-1) > 0).float()
        elif self.E_class is not None:
            adj = self.E_class[..., 0].clone()
        elif self.E_feat is not None:
            e = self.E_feat
            scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
            adj = (scalar > 0.5).float()
        else:
            raise ValueError("dense_adjacency() requires E_class or E_feat.")
        mask_2d = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        return adj * mask_2d.float()

    def to_sparse(
        self,
        *,
        edge_active: Callable[[Tensor], Tensor] | None = None,
    ) -> GraphState:
        bs = int(self.num_nodes_per_graph.shape[0])
        device = self.num_nodes_per_graph.device

        # Build active mask per (b, i, j) over off-diagonal positions.
        if self.E_class is not None:
            ref = self.E_class
            active = ref.argmax(dim=-1) > 0 if edge_active is None else edge_active(ref)
        elif self.E_feat is not None:
            ref = self.E_feat
            active = (
                ref.abs().sum(dim=-1) > 0 if edge_active is None else edge_active(ref)
            )
        else:
            raise ValueError("to_sparse() requires E_class or E_feat populated.")

        # Mask out positions outside node_mask × node_mask and the diagonal.
        nm = self.node_mask
        pair_mask = nm.unsqueeze(-1) & nm.unsqueeze(-2)  # (B, n_max, n_max)
        n_max = int(nm.shape[1])
        eye = torch.eye(n_max, dtype=torch.bool, device=device).unsqueeze(0)
        active = active.bool() & pair_mask & ~eye

        # Build sparse edge_index.
        b_idx, i_idx, j_idx = active.nonzero(as_tuple=True)
        # Convert (b, i) → global node id.
        cum = torch.zeros(bs + 1, dtype=torch.long, device=device)
        cum[1:] = torch.cumsum(self.num_nodes_per_graph, dim=0)
        src_global = cum[b_idx] + i_idx
        dst_global = cum[b_idx] + j_idx
        edge_index = torch.stack([src_global, dst_global], dim=0)

        # Gather edge tensors.
        edge_class = (
            self.E_class[b_idx, i_idx, j_idx] if self.E_class is not None else None
        )
        edge_feat = (
            self.E_feat[b_idx, i_idx, j_idx] if self.E_feat is not None else None
        )

        # Node features: gather valid positions.
        valid_b, valid_i = nm.nonzero(as_tuple=True)
        x_class = self.X_class[valid_b, valid_i] if self.X_class is not None else None
        x_feat = self.X_feat[valid_b, valid_i] if self.X_feat is not None else None
        batch = valid_b

        return GraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            batch=batch,
            x_class=x_class,
            x_feat=x_feat,
            edge_index=edge_index,
            edge_class=edge_class,
            edge_feat=edge_feat,
        )

    def to_distribution(self) -> DenseGraphDistribution:
        # Same fields, different type tag (no tensor reshape).
        return DenseGraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=self.X_class,
            X_feat=self.X_feat,
            E_class=self.E_class,
            E_feat=self.E_feat,
        )

    def replace(self, **kwargs: object) -> DenseGraphState:
        """Return a copy with selected fields overridden.

        Thin typed wrapper over :func:`dataclasses.replace`. Accepts any
        subset of the dataclass fields; the returned instance re-runs
        ``__post_init__`` and therefore the same validation invariants.
        """
        return _dc_replace(self, **kwargs)

    def mask(self, collapse: bool = False) -> DenseGraphState:
        """Zero out features at masked positions, optionally collapsing to indices.

        Operates on every populated split field; skips ``None`` fields.
        Mirrors upstream DiGress's ``PlaceHolder.mask(node_mask, collapse)``
        in ``digress-upstream-readonly/src/utils.py:103-131`` so that
        callers can use the upstream form directly. When ``collapse=False``
        (the default) the behaviour is unchanged: featured tensors are
        zeroed at padded positions and the categorical edge tensors are
        asserted symmetric. When ``collapse=True`` the categorical fields
        are argmaxed to integer indices with ``-1`` as a sentinel at
        masked positions (continuous ``X_feat`` / ``E_feat`` are still
        zeroed); the symmetry assertion is skipped because integer indices
        do not satisfy the same float-equality predicate.

        Parameters
        ----------
        collapse
            If ``False`` (default), zero features at masked positions and
            keep the one-hot / continuous representation. If ``True``,
            argmax categorical fields to integer class indices and use
            ``-1`` as the sentinel at masked positions.

        Returns
        -------
        DenseGraphState
            New instance with masked values zeroed (or, when
            ``collapse=True``, with categorical fields argmaxed to
            integer indices).

        Raises
        ------
        AssertionError
            With ``collapse=False``: if any populated edge-feature tensor
            is not symmetric after masking.
        """
        x_mask = self.node_mask.unsqueeze(-1)  # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)  # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)  # (bs, 1, n, 1)

        X_class_masked: Tensor | None = None
        E_class_masked: Tensor | None = None

        if collapse:
            edge_pair_mask = (e_mask1 * e_mask2).squeeze(-1)  # (bs, n, n)
            if self.X_class is not None:
                X_class_masked = torch.argmax(self.X_class, dim=-1)
                X_class_masked[self.node_mask == 0] = -1
            if self.E_class is not None:
                E_class_masked = torch.argmax(self.E_class, dim=-1)
                E_class_masked[edge_pair_mask == 0] = -1
        else:
            if self.X_class is not None:
                X_class_masked = self.X_class * x_mask
            if self.E_class is not None:
                E_class_masked = self.E_class * e_mask1 * e_mask2
                # Symmetry sanity check. ``mask()`` is hit ≥3× per training
                # step (forward noise, transformer output, loss target);
                # ``bool(allclose(...))`` syncs the GPU stream every call.
                # Guarded by ``__debug__`` so production runs (Python -O /
                # PYTHONOPTIMIZE=1) skip it. Symmetry is a math identity
                # of the producer (multiplying a symmetric input by
                # ``e_mask1 * e_mask2`` preserves symmetry).
                if __debug__:
                    assert torch.allclose(
                        E_class_masked, torch.transpose(E_class_masked, -3, -2)
                    ), (
                        "Edge features E_class must be symmetric after masking. "
                        f"Max asymmetry: {(E_class_masked - torch.transpose(E_class_masked, -3, -2)).abs().max().item():.2e}"
                    )

        # Continuous fields are always zeroed at masked positions; collapse
        # only affects the categorical (one-hot) split.
        X_feat_masked = self.X_feat * x_mask if self.X_feat is not None else None
        E_feat_masked: Tensor | None = None
        if self.E_feat is not None:
            E_feat_masked = self.E_feat * e_mask1 * e_mask2
            if not collapse:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
                if __debug__:
                    assert torch.allclose(
                        E_feat_masked, torch.transpose(E_feat_masked, -3, -2)
                    ), (
                        "Edge features E_feat must be symmetric after masking. "
                        f"Max asymmetry: {(E_feat_masked - torch.transpose(E_feat_masked, -3, -2)).abs().max().item():.2e}"
                    )

        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=X_class_masked,
            X_feat=X_feat_masked,
            E_class=E_class_masked,
            E_feat=E_feat_masked,
        )

    def mask_zero_diag(self) -> DenseGraphState:
        """Zero masked positions and the diagonal of every edge tensor.

        Used inside the transformer where self-loops are excluded from
        the attention mechanism. Operates on every populated split
        field.
        """
        n = self.node_mask.size(-1)
        mask_diag = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        eye = torch.eye(n, device=self.node_mask.device, dtype=torch.bool)
        # Broadcast the eye across leading dims
        while eye.dim() < mask_diag.dim():
            eye = eye.unsqueeze(0)
        mask_diag = mask_diag * (~eye)

        x_mask = self.node_mask.unsqueeze(-1)
        e_mask = mask_diag.unsqueeze(-1)

        X_class_masked = self.X_class * x_mask if self.X_class is not None else None
        X_feat_masked = self.X_feat * x_mask if self.X_feat is not None else None
        E_class_masked = self.E_class * e_mask if self.E_class is not None else None
        E_feat_masked = self.E_feat * e_mask if self.E_feat is not None else None

        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=X_class_masked,
            X_feat=X_feat_masked,
            E_class=E_class_masked,
            E_feat=E_feat_masked,
        )

    def type_as(self, x: Tensor) -> DenseGraphState:
        """Return a new instance with feature tensors cast to match ``x``.

        Split ``X_class`` / ``X_feat`` / ``E_class`` / ``E_feat`` fields
        are cast when present. ``num_nodes_per_graph`` stays int64.
        """
        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y.type_as(x),
            X_class=self.X_class.type_as(x) if self.X_class is not None else None,
            X_feat=self.X_feat.type_as(x) if self.X_feat is not None else None,
            E_class=self.E_class.type_as(x) if self.E_class is not None else None,
            E_feat=self.E_feat.type_as(x) if self.E_feat is not None else None,
        )

    def to(self, device: torch.device | str) -> DenseGraphState:
        """Move all tensors to ``device``, returning a new instance.

        Needed for Lightning's ``transfer_batch_to_device`` since frozen
        dataclasses are not supported by ``apply_to_collection``.
        """
        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph.to(device),
            y=self.y.to(device),
            X_class=self.X_class.to(device) if self.X_class is not None else None,
            X_feat=self.X_feat.to(device) if self.X_feat is not None else None,
            E_class=self.E_class.to(device) if self.E_class is not None else None,
            E_feat=self.E_feat.to(device) if self.E_feat is not None else None,
        )

    # ---- Conversion classmethods / methods ----------------------------

    @classmethod
    def from_structure_only(
        cls, node_mask: Tensor, edge_scalar: Tensor
    ) -> DenseGraphState:
        """Construct a structure-only graph with only ``E_feat`` populated.

        Wraps a dense scalar adjacency as a single-channel ``E_feat`` tensor
        of shape ``(bs, n, n, 1)`` (or ``(n, n, 1)`` for single graphs),
        leaving every other split field empty.

        Parameters
        ----------
        node_mask
            Boolean or float mask, ``(n,)`` or ``(bs, n)``.
        edge_scalar
            Dense scalar edge tensor, ``(n, n)`` or ``(bs, n, n)``.

        Returns
        -------
        DenseGraphState
            Instance with ``E_feat`` set and every other feature field
            ``None``.
        """
        single = edge_scalar.dim() == 2
        if single:
            edge_scalar = edge_scalar.unsqueeze(0)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)

        if edge_scalar.dim() != 3:
            raise ValueError(
                "from_structure_only() expects a 2D or 3D edge_scalar tensor, "
                f"got shape {tuple(edge_scalar.shape)}"
            )
        if node_mask.dim() != 2:
            raise ValueError(
                "from_structure_only() expects a 1D or 2D node_mask, "
                f"got shape {tuple(node_mask.shape)}"
            )

        bs, n, _ = edge_scalar.shape
        if tuple(node_mask.shape) != (bs, n):
            raise ValueError(
                "from_structure_only(): node_mask shape "
                f"{tuple(node_mask.shape)} incompatible with edge_scalar "
                f"shape {tuple(edge_scalar.shape)}"
            )

        e_feat = edge_scalar.float().unsqueeze(-1)
        y = torch.zeros(bs, 0, device=edge_scalar.device, dtype=e_feat.dtype)
        num_nodes_per_graph = node_mask.sum(dim=-1).long()
        return cls(
            num_nodes_per_graph=num_nodes_per_graph,
            y=y,
            E_feat=e_feat,
        )

    @classmethod
    def from_edge_scalar(
        cls,
        edge_scalar: Tensor,
        *,
        node_mask: Tensor,
        target: Literal["E_class", "E_feat"],
    ) -> DenseGraphState:
        """Construct a graph populating exactly one split edge field from a scalar.

        Parameters
        ----------
        edge_scalar
            Dense scalar edges, ``(n, n)`` or ``(bs, n, n)``. For
            ``target="E_class"`` values are treated as 0/1 indicators of
            edge presence (no hard threshold applied; callers should pass
            already-binary tensors).
        node_mask
            Node-validity mask, ``(n,)`` or ``(bs, n)``.
        target
            Which split field to populate:
            ``"E_feat"`` (single-channel continuous adjacency) or
            ``"E_class"`` (two-channel one-hot, channel 0 = no-edge).

        Returns
        -------
        DenseGraphState
            Instance with the selected split edge field populated and
            the other split fields ``None``.
        """
        if target == "E_feat":
            return cls.from_structure_only(node_mask, edge_scalar)

        single = edge_scalar.dim() == 2
        if single:
            edge_scalar = edge_scalar.unsqueeze(0)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)

        if edge_scalar.dim() != 3:
            raise ValueError(
                "from_edge_scalar() expects a 2D or 3D edge_scalar tensor, "
                f"got shape {tuple(edge_scalar.shape)}"
            )
        if node_mask.dim() != 2:
            raise ValueError(
                "from_edge_scalar() expects a 1D or 2D node_mask, "
                f"got shape {tuple(node_mask.shape)}"
            )

        bs, n, _ = edge_scalar.shape
        if tuple(node_mask.shape) != (bs, n):
            raise ValueError(
                "from_edge_scalar(): node_mask shape "
                f"{tuple(node_mask.shape)} incompatible with edge_scalar "
                f"shape {tuple(edge_scalar.shape)}"
            )

        adj = edge_scalar.float()
        e_class = torch.zeros(bs, n, n, 2, device=adj.device, dtype=adj.dtype)
        e_class[..., 0] = 1.0 - adj
        e_class[..., 1] = adj

        y = torch.zeros(bs, 0, device=adj.device, dtype=adj.dtype)
        num_nodes_per_graph = node_mask.sum(dim=-1).long()
        return cls(
            num_nodes_per_graph=num_nodes_per_graph,
            y=y,
            E_class=e_class,
        )

    def to_edge_scalar(self, *, source: Literal["class", "feat"]) -> Tensor:
        """Return a dense scalar adjacency from the requested split edge field.

        Both paths mask the returned tensor by the outer product of
        ``node_mask`` so padded positions are zero.

        Parameters
        ----------
        source
            ``"feat"`` reads ``E_feat`` directly (squeezing a trailing
            single-channel axis if present). ``"class"`` returns the
            edge-probability ``1 − E_class[..., 0]`` for multi-channel
            categorical edges.

        Returns
        -------
        Tensor
            Dense scalar adjacency, shape ``(n, n)`` or ``(bs, n, n)``.

        Raises
        ------
        ValueError
            If the requested source field is ``None``.
        """
        if source == "feat":
            if self.E_feat is None:
                raise ValueError(
                    "to_edge_scalar(source='feat') requires E_feat to be "
                    "populated; got None."
                )
            e = self.E_feat
            edge_scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
        else:  # source == "class"
            if self.E_class is None:
                raise ValueError(
                    "to_edge_scalar(source='class') requires E_class to be "
                    "populated; got None."
                )
            if self.E_class.shape[-1] > 1:
                edge_scalar = 1.0 - self.E_class[..., 0]
            else:
                edge_scalar = self.E_class[..., 0]

        mask2d = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        return edge_scalar * mask2d.to(edge_scalar.dtype)

    def to_networkx(self, batch_index: int = 0) -> nx.Graph[Any]:
        """Convert one graph slice to a NetworkX ``Graph``.

        ``DenseGraphState`` is always batched under the post-refactor
        invariants (``num_nodes_per_graph`` is required to be 1-D, so
        the cached ``node_mask`` is always 2-D ``(B, n_max)``). The
        default ``batch_index=0`` matches the single-graph carrier
        convention; pass an explicit index to extract any other row.

        Honours ``node_mask`` (padding rows/cols dropped) and
        :meth:`dense_adjacency` (channel 0 = "no edge"). When
        ``X_class`` is populated, the per-node argmax index lands as
        the ``x_class`` node attribute. When ``E_class`` is populated,
        the per-edge argmax index lands as the ``e_class`` edge
        attribute. The class indices let downstream consumers recover
        atom / bond types for molecular runs without re-loading the
        codec.

        Raises
        ------
        IndexError
            ``batch_index`` is out of range for the batched leading dimension.
        """
        import networkx as nx

        node_mask_row = self.node_mask[batch_index]
        adj_row = self.dense_adjacency()[batch_index]

        n_valid = int(node_mask_row.sum().item())
        adj_valid = adj_row[:n_valid, :n_valid].detach().cpu().numpy()
        graph = nx.from_numpy_array(adj_valid)

        if self.X_class is not None:
            x_class_idx = (
                self.X_class[batch_index][:n_valid]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            for node, idx in enumerate(x_class_idx):
                graph.nodes[node]["x_class"] = int(idx)

        if self.E_class is not None:
            e_class_idx = (
                self.E_class[batch_index][:n_valid, :n_valid]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            for u, v in graph.edges():
                graph.edges[u, v]["e_class"] = int(e_class_idx[u][v])

        return graph

    def to_networkx_list(self) -> list[nx.Graph[Any]]:
        """Expand the batched ``DenseGraphState`` into a per-graph nx list.

        Equivalent to ``[self.to_networkx(i) for i in range(bs)]``.
        Subsumes the legacy :meth:`GraphEvaluator.to_networkx_graphs`
        for the single-batched-instance case.
        """
        bs = int(self.num_nodes_per_graph.shape[0])
        return [self.to_networkx(i) for i in range(bs)]

    @classmethod
    def synth_structure_only_x_class(cls, node_mask: Tensor, c_x: int) -> Tensor:
        """Derive a structure-only X_class tensor from node_mask alone.

        Synthesis convention (spec 2026-04-27-x-class-synth-unification):
        - C_x = 1: ones at valid nodes, zeros at padding. Canonical
          structure-only encoding; noise process is identity on X.
        - C_x = 2: [1 - node_ind, node_ind] one-hot (legacy [no-node,
          node]). Kept for backward compat with existing C_x=2 model
          presets.
        - C_x >= 3: raise — real categorical X must be populated by
          the dataset.

        E has no symmetric helper: edges are an adjacency property
        orthogonal to node_mask, so synthesis from node_mask alone is
        undefined for any C_e. ``_read_categorical_e`` raises when
        E_class is None.

        Parameters
        ----------
        node_mask
            Boolean mask of valid nodes, shape ``(B, N)``.
        c_x
            Categorical class width including any structural-filler
            slot. See spec §3 for the regime table.

        Returns
        -------
        Tensor
            Shape ``(B, N, c_x)``, dtype ``float32``, on the same
            device as ``node_mask``.
        """
        node_ind = node_mask.float()
        if c_x == 1:
            return node_ind.unsqueeze(-1)
        if c_x == 2:
            synth = torch.stack([1.0 - node_ind, node_ind], dim=-1)
            # Zero padding rows so loss predicates exclude them
            return synth * node_ind.unsqueeze(-1)
        raise ValueError(
            "DenseGraphState.synth_structure_only_x_class: synthesis is only "
            f"defined for C_x in {{1, 2}}; got C_x={c_x}. C_x>=3 implies "
            "real categorical content; the dataset MUST populate "
            "X_class. See 2026-04-27-x-class-synth-unification-spec §3."
        )

    @classmethod
    def from_pyg_batch(
        cls,
        batch: Batch,
        *,
        n_max_static: int | None = None,
        num_atom_types_x: int | None = None,
        num_bond_types_e: int | None = None,
    ) -> DenseGraphState:
        """Dense-direct PyG → DenseGraphState (legacy fast path).

        Equivalent to ``GraphState.from_pyg_batch(batch).to_dense(...)`` but
        retained for tests / callers that exercise the dense path directly.
        """
        sparse = GraphState.from_pyg_batch(
            batch,
            num_atom_types_x=num_atom_types_x,
            num_bond_types_e=num_bond_types_e,
        )
        return sparse.to_dense(
            edge_class_fill=_no_edge_one_hot_fill(sparse.edge_class),
            n_max=n_max_static,
        )

    def to_pyg(self) -> Data:
        """Convert this (unbatched) DenseGraphState to a PyG Data object.

        Returns
        -------
        torch_geometric.data.Data
            Sparse COO representation with ``edge_index`` and ``num_nodes``.

        Raises
        ------
        ValueError
            If the DenseGraphState has batch size > 1.
        """
        from torch_geometric.data import Data
        from torch_geometric.utils import dense_to_sparse

        adj = self.dense_adjacency()
        if adj.ndim == 3:
            if adj.shape[0] != 1:
                raise ValueError(
                    f"to_pyg() requires unbatched or batch-size-1 DenseGraphState, "
                    f"got batch size {adj.shape[0]}"
                )
            adj = adj.squeeze(0)

        n = (
            int(self.node_mask.sum().item())
            if self.node_mask.ndim == 1
            else adj.shape[0]
        )
        adj_trimmed = adj[:n, :n]
        edge_index, _ = dense_to_sparse(adj_trimmed)
        return Data(edge_index=edge_index, num_nodes=n)

    @staticmethod
    def collate(graphs: list[DenseGraphState]) -> DenseGraphState:
        """Collate variable-size graphs into a padded batch.

        Pads every populated split field to the maximum node count.
        Padded node positions receive "no-node" (one-hot index 0) in
        categorical fields, padded edge positions receive "no-edge"
        (one-hot index 0), and the derived ``node_mask`` cached_property
        is ``False`` for padded slots.

        Parameters
        ----------
        graphs
            Individual (B=1) ``DenseGraphState`` instances. All graphs
            MUST share the same set of populated split fields.

        Returns
        -------
        DenseGraphState
            Batched instance with shapes ``(bs, n_max, ...)``.
        """
        if not graphs:
            raise ValueError("DenseGraphState.collate() requires a non-empty list.")

        bs = len(graphs)
        # Each input is B=1; n_i is num_nodes_per_graph[0] for that input.
        ns = [int(g.num_nodes_per_graph[0].item()) for g in graphs]
        n_max = max(ns)
        dy = int(graphs[0].y.shape[-1])

        num_nodes_per_graph = torch.tensor(ns, dtype=torch.long)
        y_batch = torch.zeros(bs, dy, dtype=graphs[0].y.dtype)

        def _pad_field(
            field_name: Literal["X_class", "X_feat", "E_class", "E_feat"],
        ) -> Tensor | None:
            """Pad a single split field from every graph, or return None."""
            first = getattr(graphs[0], field_name)
            if first is None:
                for g in graphs[1:]:
                    if getattr(g, field_name) is not None:
                        raise ValueError(
                            f"DenseGraphState.collate(): {field_name} populated "
                            "inconsistently across graphs."
                        )
                return None

            is_edge = field_name.startswith("E_")
            d = int(first.shape[-1])
            if is_edge:
                batch = torch.zeros(bs, n_max, n_max, d, dtype=first.dtype)
            else:
                batch = torch.zeros(bs, n_max, d, dtype=first.dtype)

            for i, g in enumerate(graphs):
                tensor = getattr(g, field_name)
                if tensor is None:
                    raise ValueError(
                        f"DenseGraphState.collate(): {field_name} missing on graph {i} "
                        "but present on graph 0."
                    )
                ni = ns[i]
                # Each input is B=1, so tensor has shape (1, ni, ...) for nodes
                # or (1, ni, ni, ...) for edges; squeeze the leading batch dim.
                t_unb = tensor[0] if tensor.dim() in (3, 4) else tensor
                if is_edge:
                    batch[i, :ni, :ni] = t_unb
                    # Padded edge positions -> "no edge" (class 0) when
                    # categorical; continuous fields stay at zero.
                    if field_name == "E_class":
                        batch[i, :ni, ni:, 0] = 1.0
                        batch[i, ni:, :, 0] = 1.0
                else:
                    batch[i, :ni] = t_unb
                    if field_name == "X_class":
                        batch[i, ni:, 0] = 1.0
            return batch

        for i, g in enumerate(graphs):
            if dy > 0:
                y_batch[i] = g.y[0] if g.y.dim() == 2 else g.y

        X_class = _pad_field("X_class")
        X_feat = _pad_field("X_feat")
        E_class = _pad_field("E_class")
        E_feat = _pad_field("E_feat")

        return DenseGraphState(
            num_nodes_per_graph=num_nodes_per_graph,
            y=y_batch,
            X_class=X_class,
            X_feat=X_feat,
            E_class=E_class,
            E_feat=E_feat,
        )


def collapse_to_indices(data: DenseGraphState) -> tuple[Tensor, Tensor | None]:
    """Argmax ``E_class`` (and optionally ``X_class``) to class indices.

    Thin tuple-returning alias for ``data.mask(collapse=True)``, kept so
    external callers that pre-date the merge keep working. The canonical
    form is ``DenseGraphState.mask(collapse=True)`` (mirrors upstream's
    ``PlaceHolder.mask(node_mask, collapse=True)``).

    Parameters
    ----------
    data
        One-hot encoded graph features with ``node_mask``. ``E_class``
        MUST be populated.

    Returns
    -------
    (E_idx, X_idx)
        ``E_idx`` has shape ``(bs, n, n)``; ``X_idx`` has shape
        ``(bs, n)`` or is ``None`` when ``data.X_class`` is ``None``.

    Raises
    ------
    ValueError
        If ``data.E_class`` is ``None``.
    """
    if data.E_class is None:
        raise ValueError("collapse_to_indices() requires data.E_class to be populated.")

    collapsed = data.mask(collapse=True)
    assert collapsed.E_class is not None  # E_class non-None already guarded above
    return collapsed.E_class, collapsed.X_class


@dataclass(frozen=True)
class DenseGraphDistribution(_DistributionGraph):
    """Dense + distribution. (B, n_max, ...) padded carrier with
    distribution-typed content.
    """

    X_class: Tensor | None = None
    X_feat: Tensor | None = None
    E_class: Tensor | None = None
    E_feat: Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # Reuse the same shape checks as DenseGraphState.
        if self.E_class is None and self.E_feat is None:
            raise ValueError(
                "DenseGraphDistribution requires at least one of E_class or E_feat."
            )
        bs = int(self.num_nodes_per_graph.shape[0])
        if self.E_class is not None:
            n_max = int(self.E_class.shape[1])
            if tuple(self.E_class.shape[:3]) != (bs, n_max, n_max):
                raise ValueError(
                    f"E_class shape mismatch: {tuple(self.E_class.shape)} vs "
                    f"({bs}, {n_max}, {n_max}, *)."
                )
        if self.E_feat is not None:
            n_max = int(self.E_feat.shape[1])
            if tuple(self.E_feat.shape[:3]) != (bs, n_max, n_max):
                raise ValueError(
                    f"E_feat shape mismatch: {tuple(self.E_feat.shape)} vs "
                    f"({bs}, {n_max}, {n_max}, *)."
                )

    def to(self, device: torch.device | str) -> DenseGraphDistribution:
        return DenseGraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph.to(device),
            y=self.y.to(device),
            X_class=None if self.X_class is None else self.X_class.to(device),
            X_feat=None if self.X_feat is None else self.X_feat.to(device),
            E_class=None if self.E_class is None else self.E_class.to(device),
            E_feat=None if self.E_feat is None else self.E_feat.to(device),
        )

    def type_as(self, x: Tensor) -> DenseGraphDistribution:
        return DenseGraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y.type_as(x),
            X_class=None if self.X_class is None else self.X_class.type_as(x),
            X_feat=None if self.X_feat is None else self.X_feat.type_as(x),
            E_class=None if self.E_class is None else self.E_class.type_as(x),
            E_feat=None if self.E_feat is None else self.E_feat.type_as(x),
        )

    def to_sparse(self) -> GraphDistribution:
        bs = int(self.num_nodes_per_graph.shape[0])
        device = self.num_nodes_per_graph.device
        nm = self.node_mask
        n_max = int(nm.shape[1])

        # Complete edge_index = all (i, j) where i ≠ j and both in node_mask.
        pair_mask = nm.unsqueeze(-1) & nm.unsqueeze(-2)
        eye = torch.eye(n_max, dtype=torch.bool, device=device).unsqueeze(0)
        select = pair_mask & ~eye
        b_idx, i_idx, j_idx = select.nonzero(as_tuple=True)

        cum = torch.zeros(bs + 1, dtype=torch.long, device=device)
        cum[1:] = torch.cumsum(self.num_nodes_per_graph, dim=0)
        src_global = cum[b_idx] + i_idx
        dst_global = cum[b_idx] + j_idx
        edge_index = torch.stack([src_global, dst_global], dim=0)

        edge_class = (
            self.E_class[b_idx, i_idx, j_idx] if self.E_class is not None else None
        )
        edge_feat = (
            self.E_feat[b_idx, i_idx, j_idx] if self.E_feat is not None else None
        )

        valid_b, valid_i = nm.nonzero(as_tuple=True)
        x_class = self.X_class[valid_b, valid_i] if self.X_class is not None else None
        x_feat = self.X_feat[valid_b, valid_i] if self.X_feat is not None else None
        batch = valid_b

        return GraphDistribution(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            batch=batch,
            x_class=x_class,
            x_feat=x_feat,
            edge_index=edge_index,
            edge_class=edge_class,
            edge_feat=edge_feat,
        )

    def argmax(self) -> DenseGraphState:
        # Categorical: argmax along channel and re-encode as one-hot.
        # Padding (b, i) outside node_mask and diagonal pairs are zeroed
        # to match the complete-pair semantics of the lift→argmax round
        # trip (see test_dense_distribution_argmax_recovers_state).
        nm = self.node_mask  # (B, n_max)
        pair_mask = nm.unsqueeze(-1) & nm.unsqueeze(-2)  # (B, n_max, n_max)
        n_max = int(nm.shape[1])
        eye = torch.eye(
            n_max, dtype=torch.bool, device=self.num_nodes_per_graph.device
        ).unsqueeze(0)
        e_pair_mask = pair_mask & ~eye
        X_class_oh: Tensor | None = None
        E_class_oh: Tensor | None = None
        if self.X_class is not None:
            idx = self.X_class.argmax(dim=-1)
            xc_oh = F.one_hot(idx, num_classes=self.X_class.shape[-1]).to(
                self.X_class.dtype
            )
            X_class_oh = xc_oh * nm.unsqueeze(-1).to(xc_oh.dtype)
        if self.E_class is not None:
            idx = self.E_class.argmax(dim=-1)
            ec_oh = F.one_hot(idx, num_classes=self.E_class.shape[-1]).to(
                self.E_class.dtype
            )
            E_class_oh = ec_oh * e_pair_mask.unsqueeze(-1).to(ec_oh.dtype)
        # Continuous: threshold |.|>0.5 to {0,1}.
        X_feat_b: Tensor | None = None
        E_feat_b: Tensor | None = None
        if self.X_feat is not None:
            xf_b = (self.X_feat.abs() > 0.5).to(self.X_feat.dtype)
            X_feat_b = xf_b * nm.unsqueeze(-1).to(xf_b.dtype)
        if self.E_feat is not None:
            ef_b = (self.E_feat.abs() > 0.5).to(self.E_feat.dtype)
            E_feat_b = ef_b * e_pair_mask.unsqueeze(-1).to(ef_b.dtype)
        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=X_class_oh,
            X_feat=X_feat_b,
            E_class=E_class_oh,
            E_feat=E_feat_b,
        )

    def sample(self, *, generator: torch.Generator | None = None) -> DenseGraphState:
        X_class_oh: Tensor | None = None
        E_class_oh: Tensor | None = None
        if self.X_class is not None:
            probs = F.softmax(self.X_class, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            idx = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(
                -1
            )
            idx = idx.reshape(probs.shape[:-1])
            X_class_oh = F.one_hot(idx, num_classes=self.X_class.shape[-1]).to(
                self.X_class.dtype
            )
        if self.E_class is not None:
            probs = F.softmax(self.E_class, dim=-1)
            flat = probs.reshape(-1, probs.shape[-1])
            idx = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(
                -1
            )
            idx = idx.reshape(probs.shape[:-1])
            E_class_oh = F.one_hot(idx, num_classes=self.E_class.shape[-1]).to(
                self.E_class.dtype
            )
        # Continuous: pass through (Gaussian "sampling" upstream in the noise process).
        return DenseGraphState(
            num_nodes_per_graph=self.num_nodes_per_graph,
            y=self.y,
            X_class=X_class_oh,
            X_feat=self.X_feat,
            E_class=E_class_oh,
            E_feat=self.E_feat,
        )

    @cached_property
    def node_mask(self) -> Tensor:
        bs = int(self.num_nodes_per_graph.shape[0])
        if self.E_class is not None:
            n_max = int(self.E_class.shape[1])
        elif self.E_feat is not None:
            n_max = int(self.E_feat.shape[1])
        elif self.X_class is not None:
            n_max = int(self.X_class.shape[1])
        else:
            assert self.X_feat is not None
            n_max = int(self.X_feat.shape[1])
        arange = torch.arange(n_max, device=self.num_nodes_per_graph.device)
        return arange.unsqueeze(0).expand(bs, -1) < self.num_nodes_per_graph.unsqueeze(
            1
        )

    def dense_adjacency(self) -> Tensor:
        if self.E_class is not None and self.E_class.shape[-1] > 1:
            adj = (self.E_class.argmax(dim=-1) > 0).float()
        elif self.E_class is not None:
            adj = self.E_class[..., 0].clone()
        elif self.E_feat is not None:
            e = self.E_feat
            scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
            adj = (scalar > 0.5).float()
        else:
            raise ValueError("dense_adjacency() requires E_class or E_feat.")
        mask_2d = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        return adj * mask_2d.float()


# ---- Convenience constructors for common to_dense fill patterns ---------


def state_to_dense_sample(state: GraphState) -> DenseGraphState:
    """Convenience: dense conversion with no-edge one-hot fill (DiGress samples)."""
    fill = _no_edge_one_hot_fill(state.edge_class)
    return state.to_dense(edge_class_fill=fill)


def state_to_dense_logits(state: GraphState) -> DenseGraphState:
    """Convenience: dense conversion with zero fill (classification logits)."""
    if state.edge_class is None:
        return state.to_dense()
    d_ec = int(state.edge_class.shape[-1])
    fill = torch.zeros(
        d_ec, dtype=state.edge_class.dtype, device=state.edge_class.device
    )
    return state.to_dense(edge_class_fill=fill)
