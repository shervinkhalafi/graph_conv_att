"""Unit checks: PEARL + spectral + tiny DiGress compose under torch.compile.

Each test wraps the unit under ``torch.compile``, runs a forward, and
asserts that ``torch._dynamo.utils.counters`` reports **zero
graph-breaks** and **zero compile errors**. The threshold is strict:
any data-dependent control-flow regression in PEARLEmbedding,
PEARLExtraFeatures, TopKEigenLayer, SpectralProjectionLayer, or the
GraphTransformer's overall forward will fail loudly here instead of
surfacing as a silent dynamo recompile or a crash on Modal.

Rationale
---------
``torch.dynamo`` graph-breaks on data-dependent ifs / dynamic shapes /
unsupported ops. Recompiles cost ~30-80s each in the inductor pipeline;
many small graphs blow past Modal's 900s heartbeat. These tests are
the regression net for the 2026-05-04 round of compile-readiness fixes
(branch-free ``masked_softmax``, vestigial ``uy.type_as`` removal,
``__debug__``-gated assertions in ``noise_process``, sign-normalize
warning gating in ``topk_eigen``, residual-flag hoisting in
``pearl_embedding``).

Note: ``__debug__`` is True under pytest (no ``-O`` flag), so any
``__debug__``-gated branch IS visible to dynamo here. We accept this
because the gated paths emit warnings only — they don't compute new
tensors — so dynamo elides them via guard-folding rather than
graph-break. Tests verify the *production* (PYTHONOPTIMIZE=1) path
indirectly: if the dev path doesn't break, the production path
certainly won't either.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo

from tmgg.models.digress.pearl_extra_features import PEARLExtraFeatures
from tmgg.models.layers.pearl_embedding import PEARLEmbedding
from tmgg.models.layers.spectral_projection import SpectralProjectionLayer
from tmgg.models.layers.topk_eigen import TopKEigenLayer


def _device() -> torch.device:
    """Return CUDA when available; CPU otherwise.

    Compile + GPU is the production path; CPU works for laptop dev.
    Tests must pass on both.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reset_dynamo() -> None:
    """Wipe dynamo's compile cache + counters for a clean test."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


def _assert_no_graph_breaks(name: str) -> None:
    """Fail if dynamo reports any graph-breaks or compile errors for this trace."""
    counters = dict(torch._dynamo.utils.counters)
    graph_break_count = sum(counters.get("graph_break", {}).values())
    inline_call_errors = sum(counters.get("inline_call", {}).values())
    unimplemented = sum(counters.get("unimplemented", {}).values())
    assert graph_break_count == 0, (
        f"{name}: torch.compile produced {graph_break_count} graph-break(s):\n"
        f"  {dict(counters.get('graph_break', {}))}"
    )
    assert inline_call_errors == 0, (
        f"{name}: torch.compile produced {inline_call_errors} inline_call error(s):\n"
        f"  {dict(counters.get('inline_call', {}))}"
    )
    assert unimplemented == 0, (
        f"{name}: torch.compile produced {unimplemented} unimplemented op(s):\n"
        f"  {dict(counters.get('unimplemented', {}))}"
    )


# ---------------------------------------------------------------------------
# PEARLEmbedding
# ---------------------------------------------------------------------------


def test_pearl_embedding_random_compiles_clean() -> None:
    """R-PEARL forward through a small adjacency must compile graph-break-free.

    Default mode for every PEARL repro variant.
    """
    _reset_dynamo()
    device = _device()
    layer = PEARLEmbedding(
        output_dim=8,
        num_layers=3,
        mode="random",
        hidden_dim=16,
        input_samples=8,
    ).to(device)
    layer.train()
    compiled = torch.compile(layer)

    bs, n = 2, 12
    A = torch.rand(bs, n, n, device=device)
    A = (A + A.transpose(-1, -2)) / 2

    out = compiled(A)
    assert out.shape == (bs, n, 8)
    _assert_no_graph_breaks("PEARLEmbedding(random)")


def test_pearl_embedding_random_eval_compiles_clean() -> None:
    """R-PEARL EVAL forward must compile graph-break-free.

    Regression test for the 2026-05-04 fix that replaced
    ``torch.Generator(device=device).manual_seed(42)`` (which dynamo
    cannot trace — "Unsupported: UserDefinedObjectVariable(Generator)")
    with a fixed-seed ``_eval_random_features`` buffer pre-allocated
    at ``__init__``. The earlier ``train()``-mode test missed this
    because the eval branch was untaken.
    """
    _reset_dynamo()
    device = _device()
    layer = PEARLEmbedding(
        output_dim=8,
        num_layers=3,
        mode="random",
        hidden_dim=16,
        input_samples=8,
        max_nodes=20,  # buffer ceiling
    ).to(device)
    layer.eval()  # ← this is the key — exercises the buffer-slice branch
    compiled = torch.compile(layer)

    bs, n = 2, 12
    A = torch.rand(bs, n, n, device=device)
    A = (A + A.transpose(-1, -2)) / 2

    out = compiled(A)
    assert out.shape == (bs, n, 8)
    _assert_no_graph_breaks("PEARLEmbedding(random, eval)")


def test_pearl_embedding_basis_compiles_clean() -> None:
    """B-PEARL forward must also compile without graph-breaks.

    Less common in our configs but exposed via ``mode="basis"``.
    """
    _reset_dynamo()
    device = _device()
    layer = PEARLEmbedding(
        output_dim=8,
        num_layers=3,
        mode="basis",
        hidden_dim=16,
        max_nodes=20,
    ).to(device)
    layer.eval()  # B-PEARL doesn't take the random branch
    compiled = torch.compile(layer)

    bs, n = 2, 12
    A = torch.rand(bs, n, n, device=device)
    A = (A + A.transpose(-1, -2)) / 2

    out = compiled(A)
    assert out.shape == (bs, n, 8)
    _assert_no_graph_breaks("PEARLEmbedding(basis)")


# ---------------------------------------------------------------------------
# PEARLExtraFeatures
# ---------------------------------------------------------------------------


def test_pearl_extra_features_compiles_clean() -> None:
    """End-to-end PEARLExtraFeatures must compile without graph-breaks.

    Wraps ``NodeCycleFeatures`` (cycle counts via adjacency powers) +
    ``PEARLEmbedding`` (R-PEARL on the binary adjacency). Both child
    modules tested individually above; this verifies the wrapper's
    own ``__call__`` (the masking, concatenation, and dtype handling)
    composes cleanly.
    """
    _reset_dynamo()
    device = _device()
    extra = PEARLExtraFeatures(
        max_n_nodes=20,
        pearl_output_dim=8,
        pearl_num_layers=2,
        pearl_mode="random",
        pearl_hidden_dim=16,
        pearl_input_samples=8,
    ).to(device)
    extra.train()
    # Wrap as a callable Module subclass so torch.compile sees a forward.
    # PEARLExtraFeatures uses ``__call__`` directly (not ``forward``);
    # we adapt by wrapping in nn.Module that delegates.
    import torch.nn as nn

    class _Wrap(nn.Module):
        def __init__(self, e: PEARLExtraFeatures) -> None:
            super().__init__()
            self.e = e

        def forward(
            self,
            X: torch.Tensor,
            E: torch.Tensor,
            y: torch.Tensor,
            node_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.e(X, E, y, node_mask)

    compiled = torch.compile(_Wrap(extra))

    bs, n = 2, 12
    X = torch.zeros(bs, n, 1, device=device)
    E = torch.zeros(bs, n, n, 2, device=device)
    E[..., 1] = (torch.rand(bs, n, n, device=device) > 0.5).float()
    E[..., 1] = (E[..., 1] + E[..., 1].transpose(-1, -2)).clamp_(max=1.0)
    E[..., 0] = 1.0 - E[..., 1]
    y = torch.zeros(bs, 0, device=device)
    node_mask = torch.ones(bs, n, dtype=torch.bool, device=device)

    extra_x, extra_e, extra_y = compiled(X, E, y, node_mask)
    # 3 cycle features + 8 PEARL channels
    assert extra_x.shape == (bs, n, 3 + 8)
    assert extra_e.shape == (bs, n, n, 0)
    # 1 normalised node count + 4 cycle globals
    assert extra_y.shape == (bs, 5)
    _assert_no_graph_breaks("PEARLExtraFeatures")


# ---------------------------------------------------------------------------
# TopKEigenLayer (the topk_eigen.py:261 fix)
# ---------------------------------------------------------------------------


def test_topk_eigen_compiles_clean() -> None:
    """TopKEigenLayer must compile graph-break-free.

    Specifically guards the 2026-05-04 fix to ``topk_eigen.py:261``
    where ``if zero_eigenvector_mask.any():`` (a data-dependent Python
    branch on a tensor reduction, used only to emit ``warnings.warn``)
    triggered ``Dynamic control flow not supported`` in the spec
    variant of the SBM repro. Now gated under ``if __debug__:`` so
    production strips it; the tensor compute path stays branch-free.
    """
    _reset_dynamo()
    device = _device()
    layer = TopKEigenLayer(k=4).to(device)
    layer.eval()
    compiled = torch.compile(layer)

    bs, n = 2, 12
    A = torch.rand(bs, n, n, device=device)
    A = (A + A.transpose(-1, -2)) / 2

    V, Lambda = compiled(A)
    assert V.shape == (bs, n, 4)
    assert Lambda.shape == (bs, 4)
    _assert_no_graph_breaks("TopKEigenLayer")


# ---------------------------------------------------------------------------
# SpectralProjectionLayer (used by ``+model.compile_model=true`` spec variant)
# ---------------------------------------------------------------------------


def test_spectral_projection_compiles_clean() -> None:
    """SpectralProjectionLayer (Q/K/V eigh-projection) must compile clean.

    The layer itself has no data-dependent control flow — its only
    ``if`` is on ``self.normalize_eigenvalues`` (a constant Python bool).
    This test pins that property so a future regression fails loudly.
    """
    _reset_dynamo()
    device = _device()
    layer = SpectralProjectionLayer(
        k=4, out_dim=16, num_terms=3, normalize_eigenvalues=True
    ).to(device)
    layer.eval()
    compiled = torch.compile(layer)

    bs, n, k = 2, 12, 4
    V = torch.randn(bs, n, k, device=device)
    Lambda = torch.randn(bs, k, device=device)

    out = compiled(V, Lambda)
    assert out.shape == (bs, n, 16)
    _assert_no_graph_breaks("SpectralProjectionLayer")


# ---------------------------------------------------------------------------
# Tiny end-to-end DiGress GraphTransformer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DiGress GraphTransformer compile path is CUDA-targeted; CPU also works "
    "but is too slow for an interactive unit test (>60 s on a small graph).",
)
def test_tiny_digress_with_pearl_compiles_clean() -> None:
    """A 2-layer DiGress transformer + PEARL extras must compile clean.

    Smallest possible end-to-end model exercising the full forward
    chain: cycle features + PEARL embedding + transformer layers +
    final projection. If this passes, the SBM repro launch with
    ``+model.compile_model=true +experiment=discrete_sbm_pearl_repro``
    will not graph-break on entry.
    """
    from tmgg.models.digress.transformer_model import GraphTransformer

    _reset_dynamo()
    device = _device()

    extra = PEARLExtraFeatures(
        max_n_nodes=12,
        pearl_output_dim=4,
        pearl_num_layers=2,
        pearl_mode="random",
        pearl_hidden_dim=8,
        pearl_input_samples=4,
    )
    # PEARLExtraFeatures.adjust_dims maps base dims → input dims.
    base_input_dims = {"X": 1, "E": 2, "y": 0}
    base_output_dims = {"X": 1, "E": 2, "y": 0}

    # Use a tiny config that exercises every layer.
    model = GraphTransformer(
        n_layers=2,
        input_dims=base_input_dims,
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={
            "dx": 8,
            "de": 4,
            "dy": 4,
            "n_head": 2,
            "dim_ffX": 16,
            "dim_ffE": 8,
            "dim_ffy": 16,
        },
        output_dims=base_output_dims,
        output_dims_x_class=1,
        output_dims_e_class=2,
        extra_features=extra,
    ).to(device)
    model.train()
    compiled = torch.compile(model)

    from tmgg.data.datasets.graph_types import (
        DenseGraphDistribution,
        DenseGraphState,
    )

    bs, n = 2, 12
    num_nodes_per_graph = torch.full((bs,), n, dtype=torch.long, device=device)
    X_class = torch.zeros(bs, n, 1, device=device)
    E_class = torch.zeros(bs, n, n, 2, device=device)
    E_class[..., 1] = (torch.rand(bs, n, n, device=device) > 0.5).float()
    E_class[..., 1] = (E_class[..., 1] + E_class[..., 1].transpose(-1, -2)).clamp_(
        max=1.0
    )
    E_class[..., 0] = 1.0 - E_class[..., 1]
    y = torch.zeros(bs, 0, device=device)
    data = DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y,
        X_class=X_class,
        E_class=E_class,
    )
    t = torch.zeros(bs, device=device)

    out = compiled(data, t=t, output_dense=True)
    # Output is a DenseGraphDistribution; confirm shapes are sensible
    assert isinstance(out, DenseGraphDistribution)
    assert out.X_class is not None and out.X_class.shape == (bs, n, 1)
    assert out.E_class is not None and out.E_class.shape == (bs, n, n, 2)
    _assert_no_graph_breaks("GraphTransformer + PEARLExtraFeatures")
