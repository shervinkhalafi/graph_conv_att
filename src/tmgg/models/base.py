"""Abstract base classes and structural protocols for graph models."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
    GraphState,
    _DistributionGraph,
    _StateGraph,
    state_to_dense_sample,
)

EdgeSource = Literal["class", "feat"]
"""Discriminator selecting which split edge field a denoising-style
architecture reads from. Defined here so every architecture family can
import a single canonical literal type."""


def read_edge_scalar(
    data: DenseGraphState | DenseGraphDistribution, source: EdgeSource
) -> Tensor:
    """Return a dense scalar adjacency from the requested split edge field.

    Reads the same channel-zero / squeeze reduction applied by
    :meth:`DenseGraphState.to_edge_scalar`, generalised to also accept
    :class:`DenseGraphDistribution` (whose ``E_class`` / ``E_feat`` tensors
    are RAM-for-RAM identical to the state-content carrier per spec §2.1
    but whose type contract does not expose ``to_edge_scalar``). The
    continuous-adjacency model families (GNN / spectral / baselines /
    attention / hybrid) declare ``_internal_in=DenseGraphDistribution`` so
    that ``ContinuousNoiseProcess`` outputs (which are distribution-typed)
    flow through ``_coerce_input_to`` losslessly; this helper bridges the
    method-availability gap.

    Parameters
    ----------
    data
        Input dense graph data, either state-content or distribution-content.
        Callers coerce upstream input via :func:`_coerce_input_to` before
        invoking this helper.
    source
        Which split edge field to prefer (``"class"`` reads ``E_class``;
        ``"feat"`` reads ``E_feat``).

    Returns
    -------
    Tensor
        Dense scalar adjacency, ``(bs, n, n)`` or ``(n, n)``.
    """
    if isinstance(data, DenseGraphState):
        return data.to_edge_scalar(source=source)
    # DenseGraphDistribution: mirror the to_edge_scalar body using direct
    # field access. The two dense types share E_class / E_feat layouts
    # exactly, so the reduction is identical.
    if source == "feat":
        if data.E_feat is None:
            raise ValueError(
                "read_edge_scalar(source='feat') requires E_feat to be "
                "populated; got None."
            )
        e = data.E_feat
        edge_scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
    else:  # source == "class"
        if data.E_class is None:
            raise ValueError(
                "read_edge_scalar(source='class') requires E_class to be "
                "populated; got None."
            )
        if data.E_class.shape[-1] > 1:
            edge_scalar = 1.0 - data.E_class[..., 0]
        else:
            edge_scalar = data.E_class[..., 0]
    mask2d = data.node_mask.unsqueeze(-1) * data.node_mask.unsqueeze(-2)
    return edge_scalar * mask2d.to(edge_scalar.dtype)


def write_edge_scalar(
    data: DenseGraphState | DenseGraphDistribution,
    *,
    edge_scalar: Tensor,
    target: EdgeSource,
) -> DenseGraphState:
    """Return a copy of ``data`` with a scalar prediction written to ``target``.

    Wave 7 output-side helper. The single scalar adjacency is packed into
    the requested split edge field of a fresh :class:`DenseGraphState` that
    shares ``data.node_mask`` and carries ``data.y`` through unchanged.
    Unrelated split fields are left ``None``; downstream code reads
    ``edge_source``-selected tensors via :func:`read_edge_scalar`. Accepts
    either dense carrier on the input side: ``node_mask`` is a
    :class:`functools.cached_property` on both, and ``y`` lives on
    :class:`GraphData` itself, so the helper treats them identically.

    Parameters
    ----------
    data
        Input dense graph; ``node_mask`` and ``y`` propagate unchanged.
        Callers coerce upstream input via :func:`_coerce_input_to` before
        invoking this helper.
    edge_scalar
        Dense scalar adjacency, ``(bs, n, n)`` or ``(n, n)``.
    target
        Which split edge field to populate (``"feat"`` writes ``E_feat``
        with shape ``(..., 1)``; ``"class"`` writes a two-channel
        ``[1 - adj, adj]`` ``E_class``).

    Returns
    -------
    DenseGraphState
        New instance carrying the prediction in the selected split-edge
        field.
    """
    if target == "feat":
        decoded = DenseGraphState.from_structure_only(data.node_mask, edge_scalar)
    else:  # target == "class"
        decoded = DenseGraphState.from_edge_scalar(
            edge_scalar, node_mask=data.node_mask, target="E_class"
        )
    return decoded.replace(y=data.y)


def append_timestep_to_y(y: Tensor, t: Tensor | None) -> Tensor:
    """Append a normalised timestep scalar to a global feature tensor.

    Mirrors the two-line pattern used by :class:`GraphTransformer`'s
    ``use_timestep`` path: when ``t`` is provided, concatenate it as a
    final ``y`` channel; otherwise return ``y`` unchanged.
    """
    if t is None:
        return y
    return torch.cat([y, t.unsqueeze(-1)], dim=-1)


type ParameterCountTree = dict[str, int | ParameterCountTree]
"""Recursive parameter-count structure returned by ``BaseModel.parameter_count``."""


def get_parameter_count_int(counts: ParameterCountTree, key: str) -> int:
    """Return a required integer leaf from a parameter-count tree.

    Parameters
    ----------
    counts
        Hierarchical parameter-count mapping.
    key
        Name of the leaf entry to extract.

    Returns
    -------
    int
        Integer parameter count stored at ``counts[key]``.

    Raises
    ------
    TypeError
        If ``counts[key]`` is itself a nested subtree instead of an integer
        leaf. Internal callers use this to fail fast when the expected count
        shape drifts.
    """
    value = counts[key]
    if not isinstance(value, int):
        raise TypeError(f"Expected integer parameter count at key '{key}'")
    return value


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Model that produces node embeddings for hybrid composition.

    Satisfied by any model with an ``embeddings`` method (e.g. ``GNN``)
    and the standard ``get_config`` introspection hook.
    """

    def embeddings(self, data: GraphData) -> tuple[Tensor, Tensor]:
        """Compute (X, Y) node embeddings from graph data."""
        ...

    def get_config(self) -> dict[str, Any]:
        """Return model configuration dictionary."""
        ...


class BaseModel(nn.Module, ABC):
    """Base class for all models with configuration support."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return model hyperparameters for logging and serialization."""
        pass

    def parameter_count(self) -> ParameterCountTree:
        """Count trainable parameters in this module and its children.

        Returns
        -------
        ParameterCountTree
            Hierarchical counts: ``"total"`` for the full module,
            ``"self"`` for parameters not in children, and one entry
            per named child module.
        """
        self_total = 0
        total = 0
        child_breakdown: ParameterCountTree = {}

        # Count parameters directly owned by this module (not in children)
        for _, param in self.named_parameters(recurse=False):
            if param.requires_grad:
                self_total += param.numel()

        # Recursively count child modules
        for name, module in self.named_children():
            if isinstance(module, BaseModel):
                # Child has parameter_count method - use it for recursive counting
                counts = module.parameter_count()
                child_breakdown[name] = counts
                total += get_parameter_count_int(counts, "total")
            else:
                # Standard PyTorch module - count its parameters
                child_total = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                if child_total > 0:
                    child_breakdown[name] = {"total": child_total}
                    total += child_total

        return {
            "total": total + self_total,
            "self": self_total,
            **child_breakdown,
        }


class GraphModel(BaseModel, ABC):
    """Unified model interface: all models accept and return GraphData.

    This is the standard base class for graph models in the training loop.
    """

    @abstractmethod
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Forward pass operating on typed graph data.

        Parameters
        ----------
        data
            Batched graph features (X, E, y, node_mask).
        t
            Normalised diffusion timestep, or ``None`` for unconditional
            models.

        Returns
        -------
        GraphData
            Predicted graph features.
        """
        ...


# --- Coercion helpers for the 4-type GraphData grid ----------------------
#
# The 4 concrete types form a 2x2 matrix on (carrier x content):
# carrier in {sparse, dense}, content in {state, distribution}. Carrier
# conversion (to_dense / to_sparse) is structural and lossless on
# distributions (and lossless on states up to the no-edge-fill convention).
# Content lift (state -> distribution) is lossless. Content collapse
# (distribution -> state, via argmax / sample) is information-lossy and
# must be requested explicitly at the call site, not silently injected by
# a coercion helper.


def _coerce_input_to(data: GraphData, *, target: type) -> GraphData:
    """Coerce ``data`` to one of the 4 concrete GraphData types.

    Total over the 16-cell ``(input_type, target)`` matrix EXCEPT the four
    distribution -> state cells, which are rejected with :class:`TypeError`
    so that callers cannot silently lose information; collapse must be
    chosen explicitly via ``.argmax()`` or ``.sample()``.

    Parameters
    ----------
    data
        Source graph data, an instance of one of :class:`GraphState`,
        :class:`GraphDistribution`, :class:`DenseGraphState`, or
        :class:`DenseGraphDistribution`.
    target
        One of the four concrete classes above.

    Returns
    -------
    GraphData
        Instance of ``target``.

    Raises
    ------
    TypeError
        If ``target`` is a state-content type and ``data`` is a
        distribution-content type, or if ``target`` is not one of the
        four concrete classes.
    """
    if isinstance(data, target):
        return data
    # Reject lossy collapse: distribution -> state must be caller-explicit.
    if issubclass(target, _StateGraph) and isinstance(data, _DistributionGraph):
        raise TypeError(
            f"{type(data).__name__} -> {target.__name__} requires explicit "
            ".argmax() or .sample() at the call site (lossy collapse)."
        )
    # Below: target is one of the four concrete types and ``data`` is one
    # of the same four (with the collapse cells already rejected).
    if target is DenseGraphState:
        if isinstance(data, GraphState):
            return state_to_dense_sample(data)
        # All other paths to a state target either match ``isinstance``
        # above or were rejected as a distribution -> state collapse.
        raise AssertionError(
            f"unreachable: target=DenseGraphState, data={type(data).__name__}"
        )
    if target is DenseGraphDistribution:
        if isinstance(data, GraphDistribution):
            return data.to_dense()
        if isinstance(data, GraphState):
            return data.to_distribution().to_dense()
        if isinstance(data, DenseGraphState):
            return data.to_distribution()
        raise AssertionError(
            f"unreachable: target=DenseGraphDistribution, data={type(data).__name__}"
        )
    if target is GraphState:
        if isinstance(data, DenseGraphState):
            return data.to_sparse()
        raise AssertionError(
            f"unreachable: target=GraphState, data={type(data).__name__}"
        )
    if target is GraphDistribution:
        if isinstance(data, DenseGraphDistribution):
            return data.to_sparse()
        if isinstance(data, GraphState):
            return data.to_distribution()
        if isinstance(data, DenseGraphState):
            return data.to_distribution().to_sparse()
        raise AssertionError(
            f"unreachable: target=GraphDistribution, data={type(data).__name__}"
        )
    raise TypeError(f"_coerce_input_to: unknown target {target}.")


def _coerce_output_to(out: GraphData, *, target: type) -> GraphData:
    """Coerce a model output to the requested return type.

    Diffusion outputs are always distribution-content, so the only flip
    required at the boundary is dense <-> sparse on the same content axis.
    For symmetry (a few callers return state-content predictions) we also
    cover the dense <-> sparse flip on the state axis.

    Parameters
    ----------
    out
        Model output, an instance of one of the four concrete types.
    target
        Requested return type.

    Returns
    -------
    GraphData
        Instance of ``target``.

    Raises
    ------
    TypeError
        If the requested ``(out_type, target)`` flip is not one of the
        supported same-content carrier conversions.
    """
    if isinstance(out, target):
        return out
    if isinstance(out, DenseGraphDistribution) and target is GraphDistribution:
        return out.to_sparse()
    if isinstance(out, GraphDistribution) and target is DenseGraphDistribution:
        return out.to_dense()
    if isinstance(out, DenseGraphState) and target is GraphState:
        return out.to_sparse()
    if isinstance(out, GraphState) and target is DenseGraphState:
        return state_to_dense_sample(out)
    raise TypeError(
        f"_coerce_output_to: unsupported {type(out).__name__} -> {target.__name__}"
    )


__all__ = [
    "BaseModel",
    "EdgeSource",
    "EmbeddingProvider",
    "GraphModel",
    "ParameterCountTree",
    "_coerce_input_to",
    "_coerce_output_to",
    "append_timestep_to_y",
    "get_parameter_count_int",
    "read_edge_scalar",
    "write_edge_scalar",
]
