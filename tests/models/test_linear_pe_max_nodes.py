"""Regression: LinearPE must fail loudly when a graph exceeds ``max_nodes``.

Rationale (REPRODUCE.md flag #3, node-count caps):
    ``max_nodes`` sizes the learnable bias b ∈ R^{max_nodes} (linear_pe.py). For a
    graph with n > max_nodes, the slice ``self.b[:n]`` silently yields only
    ``max_nodes`` entries and the downstream ``1 bᵀ + b 1ᵀ`` outer product raises an
    opaque shape/broadcast error. Per CLAUDE.md "fail loud and informatively", we
    require an explicit ValueError naming n and max_nodes instead.

Invariants:
    - n <= max_nodes: forward works, output shape (batch, n, n).
    - n  > max_nodes (use_bias=True): ValueError mentioning "max_nodes".
    - use_bias=False: no cap applies (bias unused), any n works.
"""

import pytest
import torch

from tmgg.models.spectral_denoisers import LinearPE


def test_linear_pe_rejects_graph_exceeding_max_nodes():
    model = LinearPE(k=4, max_nodes=8, use_bias=True)
    n = 16  # > max_nodes
    V = torch.randn(1, n, 4)
    Lambda = torch.zeros(1, n)
    A = torch.zeros(1, n, n)
    with pytest.raises(ValueError, match="max_nodes"):
        model._spectral_forward(V, Lambda, A)


def test_linear_pe_accepts_graph_within_max_nodes():
    model = LinearPE(k=4, max_nodes=8, use_bias=True)
    n = 5  # <= max_nodes
    V = torch.randn(1, n, 4)
    out = model._spectral_forward(V, torch.zeros(1, n), torch.zeros(1, n, n))
    assert out.shape == (1, n, n)


def test_linear_pe_no_cap_without_bias():
    model = LinearPE(k=4, max_nodes=8, use_bias=False)
    n = 16  # > max_nodes but bias disabled → allowed
    V = torch.randn(1, n, 4)
    out = model._spectral_forward(V, torch.zeros(1, n), torch.zeros(1, n, n))
    assert out.shape == (1, n, n)
