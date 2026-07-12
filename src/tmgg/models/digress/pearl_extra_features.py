"""PEARL-based positional encoding as an :class:`ExtraFeatures` swap-in.

Replaces the eigendecomposition-based ``"eigenvalues"`` / ``"all"`` paths
of :class:`tmgg.models.digress.extra_features.ExtraFeatures` with PEARL
(Position Encoding with Adaptable Random Layers, Hejin et al., ICLR
2025; ``ehejin/Pearl-PE``) positional encodings produced by GNN message
passing. Eliminates the per-step spectral ``torch.linalg.eigh`` call
that dominates eval CUDA on the SBM Greedy profile (~57 % of CUDA per
step at ``num_samples=32``, T=500).

Cycle features (`NodeCycleFeatures`) are preserved unchanged because
they are cheap (no eigendecomposition; just powers of the adjacency)
and provide complementary structural signal.

Contract: matches :class:`ExtraFeatures` exactly so callers can swap
``_target_`` in Hydra without touching the model. The two methods
``__call__`` and ``adjust_dims`` carry the same signatures and return
shapes that the model's input projection can consume.

Output widths: with ``pearl_dim=k`` (the per-node embedding dim from
PEARL):

- ``extra_X``: ``(bs, n, 3 + k)`` — 3 cycle features + k PEARL channels
- ``extra_E``: ``(bs, n, n, 0)`` — empty (PEARL is a node-only PE)
- ``extra_y``: ``(bs, 5)`` — 1 normalised n_nodes + 4 cycle globals

Compared to upstream ``"all"`` (6+0+11): no ``n_components`` /
``batched_eigenvalues`` / ``nonlcc_indicator`` / ``k_lowest_eigvec``
spectral channels (those required eigh). PEARL provides the
positional information that the spectral channels were carrying.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from tmgg.models.digress.extra_features import NodeCycleFeatures
from tmgg.models.layers.pearl_embedding import PEARLEmbedding


def pearl_extra_features_dims(
    pearl_dim: int, use_cycles: bool = True
) -> tuple[int, int, int]:
    """Return the extra feature dimensions ``(extra_X, extra_E, extra_y)``.

    Sibling of :func:`extra_features.extra_features_dims`; used by the
    transformer to size the input projection.

    Parameters
    ----------
    pearl_dim
        Per-node PEARL embedding width.
    use_cycles
        If True (default), include the 3 per-node cycle channels and 4
        per-graph cycle-global channels alongside PEARL. If False, return
        a PEARL-only contract: ``(pearl_dim, 0, 1)`` (only the normalised
        node count survives on the per-graph side).
    """
    cycle_x = 3 if use_cycles else 0
    cycle_y = 4 if use_cycles else 0
    # cycle_x channels (k3, k4, k5 per-node) + PEARL k channels.
    extra_x = cycle_x + pearl_dim
    # PEARL is a node-only PE; no edge features added.
    extra_e = 0
    # 1 normalised node count + cycle_y per-graph cycle globals (k4..k7).
    extra_y = 1 + cycle_y
    return extra_x, extra_e, extra_y


class PEARLExtraFeatures(nn.Module):
    """Cycle features + PEARL per-node positional encoding.

    Drop-in replacement for :class:`ExtraFeatures` that omits the
    eigendecomposition path. Implements the same ``__call__`` and
    ``adjust_dims`` contract.

    Parameters
    ----------
    max_n_nodes
        Maximum number of nodes across the dataset, used to normalise
        the node-count feature appended to ``y``. Same semantic as
        :class:`ExtraFeatures.max_n_nodes`.
    pearl_output_dim
        Per-node embedding dimension produced by :class:`PEARLEmbedding`
        (the ``output_dim`` argument). Drives the width of ``extra_X``.
    pearl_num_layers
        Number of GNN layers in :class:`PEARLEmbedding`. PEARL paper
        defaults to 3.
    pearl_mode
        ``"random"`` for R-PEARL (recommended; handles variable graph
        sizes naturally) or ``"basis"`` for B-PEARL.
    pearl_hidden_dim
        Hidden dimension of the PEARL message-passing MLPs.
    pearl_input_samples
        Number of random input samples for R-PEARL (ignored for B-PEARL).
    pearl_max_nodes
        Required for B-PEARL basis-vector dimension; ignored for
        R-PEARL. Default matches typical SPECTRE SBM ceiling.

    Notes
    -----
    The single learnable :class:`PEARLEmbedding` instance is registered
    as a child module so its parameters participate in
    ``.parameters()``, gradient flow, and ``state_dict`` serialisation.
    Upstream :class:`ExtraFeatures` is stateless; this class is not.
    """

    def __init__(
        self,
        max_n_nodes: int,
        pearl_output_dim: int = 16,
        pearl_num_layers: int = 3,
        pearl_mode: str = "random",
        pearl_hidden_dim: int = 64,
        pearl_input_samples: int = 32,
        pearl_max_nodes: int = 200,
        use_cycles: bool = True,
        **_extra: object,
    ) -> None:
        # ``**_extra`` absorbs leftover kwargs that survive Hydra's
        # deep-merge of the PEARL overlay onto the upstream ``ExtraFeatures``
        # block (notably ``extra_features_type=all`` from the parent
        # ``discrete_sbm_vignac_repro`` yaml). Hydra cannot delete keys
        # during a defaults-list merge — only override them — so
        # absorbing them here is the cleanest swap-in path. The
        # ``ExtraFeatures``-specific fields are semantically irrelevant
        # to PEARL (we're not running eigh anyway).
        super().__init__()
        self.max_n_nodes = max_n_nodes
        self.pearl_output_dim = pearl_output_dim
        self.use_cycles = use_cycles
        self.ncycles: NodeCycleFeatures | None = (
            NodeCycleFeatures() if use_cycles else None
        )
        self.pearl = PEARLEmbedding(
            output_dim=pearl_output_dim,
            num_layers=pearl_num_layers,
            mode=pearl_mode,
            hidden_dim=pearl_hidden_dim,
            input_samples=pearl_input_samples,
            max_nodes=pearl_max_nodes,
        )

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute extra features. Mirrors :class:`ExtraFeatures.__call__`."""
        # Normalised node count: n_valid / max_n_nodes, shape (bs, 1)
        n = node_mask.sum(dim=1).unsqueeze(1).float() / self.max_n_nodes

        # PEARL positional encoding from the binary adjacency. ``E[..., 1:]``
        # collapses the per-class one-hot into "any non-no-edge" presence;
        # ``mask_2d`` zeros padded positions before the GNN propagation so
        # padded nodes don't leak into real-node embeddings.
        adj = E[..., 1:].sum(dim=-1).float()  # (bs, n, n)
        adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        pearl_emb = self.pearl(adj)  # (bs, n, pearl_output_dim)
        # Re-mask the embedding to zero on padded positions so downstream
        # consumers see the canonical "padded → 0" invariant.
        pearl_emb = pearl_emb * node_mask.unsqueeze(-1).float()
        pearl_emb = pearl_emb.type_as(E)

        if self.ncycles is not None:
            # Cycle features (cheap; powers of A, no eigendecomposition).
            x_cycles, y_cycles = self.ncycles(E, node_mask)
            extra_x = torch.cat([x_cycles, pearl_emb], dim=-1)
            extra_y = torch.hstack([n, y_cycles])
        else:
            extra_x = pearl_emb
            # ``n`` is already float (see ``.float()`` above); kept as float
            # to match the model's ``torch.cat([y, extra_y], dim=-1)`` site,
            # which asserts ``y.is_floating_point()`` and expects the canonical
            # Float-cast convention from ``diffusion_sampling.py``.
            extra_y = n

        extra_edge_attr = torch.zeros((*E.shape[:-1], 0), device=E.device).type_as(E)
        return extra_x, extra_edge_attr, extra_y

    def adjust_dims(self, input_dims: dict[str, int]) -> dict[str, int]:
        """Return *input_dims* with PEARL extra-feature widths added."""
        dx, de, dy = pearl_extra_features_dims(
            self.pearl_output_dim, use_cycles=self.use_cycles
        )
        return {
            "X": input_dims["X"] + dx,
            "E": input_dims["E"] + de,
            "y": input_dims["y"] + dy,
        }
