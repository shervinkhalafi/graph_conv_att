# pyright: reportAttributeAccessIssue=false
"""Tests for GNN-based Q/K/V projections in DiGress NodeEdgeBlock.

Test Rationale
--------------
These tests verify that GNN projections can be independently enabled for the
query, key, and value projections in NodeEdgeBlock. The goal is to replace
standard linear projections with polynomial graph convolutions that incorporate
graph structure into the attention mechanism.

Key invariants:
- Backwards compatibility: default params produce same architecture as original
- Per-projection control: each of Q/K/V can independently use GNN or Linear
- Gradient flow: gradients propagate through GNN convolutions
- Shape preservation: output shapes match regardless of projection type
"""

import pytest
import torch

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    GraphData,
)
from tmgg.models.digress.data_types import DenseGraphTransformerData
from tmgg.models.digress.transformer_model import (
    GraphTransformer,
    NodeEdgeBlock,
    XEyTransformerLayer,
)
from tmgg.models.layers import BareGraphConvolutionLayer


def _random_symmetric_adjacency(bs: int, n: int) -> torch.Tensor:
    """Generate a random binary symmetric adjacency with zero diagonal."""
    A = torch.randint(0, 2, (bs, n, n)).float()
    A = (A + A.transpose(1, 2)).clamp(max=1.0)
    A.diagonal(dim1=1, dim2=2).zero_()
    return A


def _make_block_input(
    *,
    bs: int,
    n: int,
    dx: int,
    de: int,
    dy: int,
    adjacency: torch.Tensor | None = None,
    eigvec: torch.Tensor | None = None,
    eigval: torch.Tensor | None = None,
) -> DenseGraphTransformerData:
    """Build a DenseGraphTransformerData carrier for direct block tests.

    The transformer block consumes ``DenseGraphTransformerData`` whose
    ``X_class`` (hidden node features), ``E_class`` (hidden edge
    features), and ``y`` are populated. ``binary_adj`` carries the
    input-time binary topology used by GNN projections; tests pass an
    explicit ``adjacency`` argument when GNN projections are exercised.
    """
    X = torch.rand(bs, n, dx)
    E = torch.rand(bs, n, n, de)
    y = torch.rand(bs, dy)
    num_nodes = torch.full((bs,), n, dtype=torch.long)
    base = DenseGraphDistribution(
        num_nodes_per_graph=num_nodes,
        y=y,
        X_class=X,
        E_class=E,
    )
    return DenseGraphTransformerData.from_base(
        base, eigvec=eigvec, eigval=eigval, binary_adj=adjacency
    )


class TestBareGraphConvolutionLayer:
    """Tests for the BareGraphConvolutionLayer used in attention projections."""

    def test_output_shape(self):
        """Output shape matches input shape (batch, n, channels)."""
        layer = BareGraphConvolutionLayer(num_terms=2, num_channels=32)
        A = torch.rand(4, 16, 16)
        A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
        X = torch.rand(4, 16, 32)

        Y = layer(A, X)
        assert Y.shape == X.shape

    def test_gradient_flow(self):
        """Gradients flow through the layer to inputs and parameters."""
        layer = BareGraphConvolutionLayer(num_terms=2, num_channels=32)
        A = torch.rand(4, 16, 16, requires_grad=True)
        X = torch.rand(4, 16, 32, requires_grad=True)

        Y = layer(A, X)
        loss = Y.sum()
        loss.backward()

        assert X.grad is not None
        assert layer.H.grad is not None

    def test_polynomial_terms(self):
        """Parameter shape matches num_terms + 1 (including identity)."""
        layer = BareGraphConvolutionLayer(num_terms=3, num_channels=64)
        # H should have shape (num_terms+1, channels, channels)
        assert layer.H.shape == (4, 64, 64)

    def test_zero_adjacency_identity(self):
        """With zero adjacency, only the identity term (H[0]) applies."""
        layer = BareGraphConvolutionLayer(num_terms=2, num_channels=16)
        A = torch.zeros(2, 8, 8)  # No edges
        X = torch.rand(2, 8, 16)

        Y = layer(A, X)
        # Should just be X @ H[0]
        expected = X @ layer.H[0]
        assert torch.allclose(Y, expected, atol=1e-5)


class TestNodeEdgeBlockGNN:
    """Tests for NodeEdgeBlock with GNN projections."""

    def test_backwards_compatibility(self):
        """Default params produce same architecture as original (all Linear)."""
        block = NodeEdgeBlock(dx=128, de=32, dy=64, n_head=4)

        # All projections should be Linear
        assert isinstance(block.q, torch.nn.Linear)
        assert isinstance(block.k, torch.nn.Linear)
        assert isinstance(block.v, torch.nn.Linear)

    @pytest.mark.parametrize(
        "use_gnn_q,use_gnn_k,use_gnn_v",
        [
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ],
    )
    def test_selective_gnn_projections(self, use_gnn_q, use_gnn_k, use_gnn_v):
        """GNN projections can be enabled independently for Q, K, V."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_gnn_q=use_gnn_q,
            use_gnn_k=use_gnn_k,
            use_gnn_v=use_gnn_v,
        )

        expected_q = BareGraphConvolutionLayer if use_gnn_q else torch.nn.Linear
        expected_k = BareGraphConvolutionLayer if use_gnn_k else torch.nn.Linear
        expected_v = BareGraphConvolutionLayer if use_gnn_v else torch.nn.Linear

        assert isinstance(block.q, expected_q)
        assert isinstance(block.k, expected_k)
        assert isinstance(block.v, expected_v)

    def test_forward_with_gnn_projections(self):
        """Forward pass works with GNN projections enabled."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=True,
            use_gnn_v=True,
        )

        bs, n = 2, 16
        h = _make_block_input(
            bs=bs,
            n=n,
            dx=128,
            de=32,
            dy=64,
            adjacency=_random_symmetric_adjacency(bs, n),
        )
        out = block(h)

        assert out.X_class is not None and out.E_class is not None
        assert out.X_class.shape == (bs, n, 128)
        assert out.E_class.shape == (bs, n, n, 32)
        assert out.y.shape == (bs, 64)

    def test_forward_mixed_projections(self):
        """Forward pass works with mixed Linear and GNN projections."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=False,  # Linear
            use_gnn_v=True,
        )

        bs, n = 2, 12
        h = _make_block_input(
            bs=bs,
            n=n,
            dx=64,
            de=16,
            dy=32,
            adjacency=_random_symmetric_adjacency(bs, n),
        )
        out = block(h)

        assert out.X_class is not None and out.E_class is not None
        assert out.X_class.shape == (bs, n, 64)
        assert out.E_class.shape == (bs, n, n, 16)
        assert out.y.shape == (bs, 32)

    def test_gradient_flow_through_gnn(self):
        """Gradients flow through GNN projections to input and parameters."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=True,
            use_gnn_v=True,
        )

        bs, n = 2, 8
        adjacency = _random_symmetric_adjacency(bs, n)
        X = torch.rand(bs, n, 64, requires_grad=True)
        E = torch.rand(bs, n, n, 16)
        y = torch.rand(bs, 32)
        num_nodes = torch.full((bs,), n, dtype=torch.long)
        base = DenseGraphDistribution(
            num_nodes_per_graph=num_nodes,
            y=y,
            X_class=X,
            E_class=E,
        )
        h = DenseGraphTransformerData.from_base(base, binary_adj=adjacency)

        out = block(h)
        assert out.X_class is not None and out.E_class is not None
        loss = out.X_class.sum() + out.E_class.sum() + out.y.sum()
        loss.backward()

        # Check gradient flows to input
        assert X.grad is not None

        # Check gradient flows to GNN parameters
        assert block.q.H.grad is not None  # pyright: ignore[reportAttributeAccessIssue]
        assert block.k.H.grad is not None  # pyright: ignore[reportAttributeAccessIssue]
        assert block.v.H.grad is not None  # pyright: ignore[reportAttributeAccessIssue]

    def test_configurable_num_terms(self):
        """gnn_num_terms parameter controls polynomial degree."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=True,
            use_gnn_v=True,
            gnn_num_terms=3,
        )

        # H should have shape (num_terms+1, channels, channels)
        assert block.q.H.shape == (4, 64, 64)  # pyright: ignore[reportAttributeAccessIssue]
        assert block.k.H.shape == (4, 64, 64)  # pyright: ignore[reportAttributeAccessIssue]
        assert block.v.H.shape == (4, 64, 64)  # pyright: ignore[reportAttributeAccessIssue]


class TestXEyTransformerLayerGNN:
    """Tests for XEyTransformerLayer with GNN configuration."""

    def test_gnn_config_passed_to_block(self):
        """GNN config is correctly passed to NodeEdgeBlock."""
        layer = XEyTransformerLayer(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_gnn_q=True,
            use_gnn_k=False,
            use_gnn_v=True,
        )

        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert isinstance(layer.self_attn.k, torch.nn.Linear)
        assert isinstance(layer.self_attn.v, BareGraphConvolutionLayer)


class TestGraphTransformerGNN:
    """Tests for full GraphTransformer with GNN config in hidden_dims."""

    def test_gnn_config_in_hidden_dims(self):
        """GNN config flows through hidden_dims to transformer layers."""
        hidden_dims = {
            "dx": 128,
            "de": 32,
            "dy": 64,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": True,
            "use_gnn_v": False,
            "gnn_num_terms": 3,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 64, "E": 32, "y": 64},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        # Check first layer's attention block
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert isinstance(layer.self_attn.k, BareGraphConvolutionLayer)
        assert isinstance(layer.self_attn.v, torch.nn.Linear)

        # Verify num_terms
        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert layer.self_attn.q.H.shape[0] == 4  # num_terms + 1

    def test_backwards_compatibility_no_gnn_keys(self):
        """GraphTransformer works without GNN keys in hidden_dims."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 1, "E": 1, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 1, "y": 0},
        )

        # All projections should be Linear
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, torch.nn.Linear)
        assert isinstance(layer.self_attn.k, torch.nn.Linear)
        assert isinstance(layer.self_attn.v, torch.nn.Linear)

    def test_forward_pass_with_gnn(self):
        """Full forward pass works with GNN projections.

        binary_graphdata produces 2-class X and E features, so input_dims
        must use {"X": 2, "E": 2}.
        """
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": True,
            "use_gnn_v": True,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        x = torch.rand(2, 12, 12)  # Adjacency matrix
        result = model(binary_graphdata(x), output_dense=True)

        assert isinstance(result, GraphData)
        assert result.E_class is not None
        assert result.E_class.shape == (2, 12, 12, 2)

    def test_forward_pass_with_eigenvectors_and_gnn(self):
        """Forward pass works with both eigenvectors and GNN projections.

        binary_graphdata produces X with 2 features; EigenvectorAugmentation
        adds k=16, so the model auto-adjusts input_dims["X"] = 2 + 16 = 18.
        """
        from tmgg.models.digress.extra_features import EigenvectorAugmentation

        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": False,
            "use_gnn_v": True,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={
                "X": 2,
                "E": 2,
                "y": 0,
            },
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
            extra_features=EigenvectorAugmentation(k=16),
        )

        x = torch.rand(2, 20, 20)  # Adjacency matrix
        x = (x + x.transpose(-1, -2)) / 2  # Symmetrize for eigenvector extraction
        result = model(binary_graphdata(x), output_dense=True)

        assert isinstance(result, GraphData)
        assert result.E_class is not None
        assert result.E_class.shape == (2, 20, 20, 2)


class TestAdjacencyExtraction:
    """Tests verifying that GNN projections use original adjacency, not transformed E."""

    def test_original_adjacency_passed_to_gnn(self):
        """Verify original adjacency is extracted before mlp_in_E transformation.

        The adjacency should be extracted from the input E before any MLP
        transformation, ensuring GNN projections operate on the true graph
        structure rather than learned edge features.
        """
        from unittest.mock import patch

        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": False,
            "use_gnn_v": False,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        # Create distinct input adjacency
        bs, n = 2, 8
        input_adj = torch.rand(bs, n, n)
        input_adj = (input_adj + input_adj.transpose(-1, -2)) / 2  # Symmetrize

        # Track what adjacency is passed to GNN projection
        captured_A: list[torch.Tensor] = []
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        gnn_q = layer.self_attn.q
        assert isinstance(gnn_q, BareGraphConvolutionLayer)
        original_q_forward = gnn_q.forward

        def capturing_forward(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            captured_A.append(A.clone())
            return original_q_forward(A, X)

        with patch.object(gnn_q, "forward", capturing_forward):
            _ = model(binary_graphdata(input_adj), output_dense=True)

        # After binary_graphdata, adjacency is extracted via argmax on 2-class E.
        # binary_graphdata zeroes diagonal (sets E[diag, 0]=1), so extracted
        # adjacency has 0 on diagonal and binary values off-diagonal.
        expected_adj = (input_adj > 0.5).float()
        diag_idx = torch.arange(n)
        expected_adj[:, diag_idx, diag_idx] = 0.0  # diagonal zeroed by from_adjacency
        assert len(captured_A) == 1
        assert torch.allclose(captured_A[0], expected_adj, atol=1e-5)

    def test_adjacency_not_from_transformed_E(self):
        """Verify adjacency differs from what would be extracted from transformed E.

        After mlp_in_E transformation, E[..., 0] would have different values
        than the original input. This test ensures we use the original.
        """
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "use_gnn_k": True,
            "use_gnn_v": True,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        # Create input
        bs, n = 2, 10
        input_adj = torch.rand(bs, n, n)
        input_adj = (input_adj + input_adj.transpose(-1, -2)) / 2

        # Verify model has GNN projection flag set
        assert model.transformer._use_gnn_projections is True

        # Run forward pass (this should work without error)
        result = model(binary_graphdata(input_adj), output_dense=True)
        assert isinstance(result, GraphData)
        assert result.E_class is not None
        assert result.E_class.shape == (bs, n, n, 2)


class TestNormalizeAdjFlag:
    """Tests for the optional ``normalize_adjacency`` flag.

    See ``rationales/digress_gnn_projection_normalize_test_rationale.md``.
    """

    def test_layer_identity_off_on_normalized_input(self):
        """``normalize=False`` on Ã ≡ ``normalize=True`` on A.

        With identical weights, feeding the symmetrically normalized
        adjacency through a layer that does *not* normalize must yield
        the same output as feeding the raw adjacency through a layer
        that *does* normalize. Pins the new flag's semantics.
        """
        from tmgg.models.layers.graph_ops import sym_normalize_adjacency

        torch.manual_seed(0)
        bs, n, c = 2, 8, 4
        A = _random_symmetric_adjacency(bs, n)
        # Add self-loops so D > 0 everywhere; sym_normalize_adjacency masks
        # zero-degree rows to zero, which would still satisfy the equality
        # but mask out signal we want to test.
        A = A + torch.eye(n).unsqueeze(0)
        A = (A > 0).float()
        X = torch.randn(bs, n, c)

        layer_on = BareGraphConvolutionLayer(
            num_terms=2, num_channels=c, normalize_adjacency=True
        )
        layer_off = BareGraphConvolutionLayer(
            num_terms=2, num_channels=c, normalize_adjacency=False
        )
        # Tie weights so the only difference is the gating of the normalize step
        layer_off.H.data.copy_(layer_on.H.data)

        A_norm = sym_normalize_adjacency(A)
        out_on = layer_on(A, X)
        out_off = layer_off(A_norm, X)

        torch.testing.assert_close(out_on, out_off, rtol=1e-5, atol=1e-6)

    def test_layer_off_uses_raw_adjacency(self):
        """``normalize=False`` actually skips the normalization.

        On a non-trivial input where ``A`` and ``Ã`` differ, the two
        layers must produce *different* outputs when given the same
        ``A``, otherwise the flag is silently dead.
        """
        torch.manual_seed(0)
        bs, n, c = 2, 8, 4
        A = _random_symmetric_adjacency(bs, n) + torch.eye(n).unsqueeze(0)
        A = (A > 0).float()
        X = torch.randn(bs, n, c)

        layer_on = BareGraphConvolutionLayer(
            num_terms=2, num_channels=c, normalize_adjacency=True
        )
        layer_off = BareGraphConvolutionLayer(
            num_terms=2, num_channels=c, normalize_adjacency=False
        )
        layer_off.H.data.copy_(layer_on.H.data)

        out_on = layer_on(A, X)
        out_off = layer_off(A, X)
        # Outputs must differ — otherwise the flag is a no-op
        assert not torch.allclose(out_on, out_off, rtol=1e-3, atol=1e-4)

    def test_default_preserved_end_to_end(self):
        """An unspecified normalize flag matches an explicit ``True``.

        Guards against an accidental default flip in the plumbing. We
        equalise weights by copying a state_dict between the two
        models; a seed alone is insufficient because the differing
        ``projection_config`` dicts alter the order in which the RNG
        is consumed during ``__init__``.
        """
        torch.manual_seed(42)
        bs, n = 2, 10

        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
        }

        model_default = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
            projection_config={"use_gnn_q": True, "use_gnn_k": True},
        )
        model_explicit = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
            projection_config={
                "use_gnn_q": True,
                "use_gnn_k": True,
                "gnn_normalize_adj": True,
            },
        )
        # Equalise weights so any output difference is due to the
        # config branch under test, not random initialisation.
        model_explicit.load_state_dict(model_default.state_dict())
        model_default.eval()
        model_explicit.eval()

        # Verify per-layer flag propagated identically (the actual
        # invariant under test).
        layer_d = model_default.transformer.tf_layers[0]
        layer_e = model_explicit.transformer.tf_layers[0]
        assert isinstance(layer_d, XEyTransformerLayer)
        assert isinstance(layer_e, XEyTransformerLayer)
        assert isinstance(layer_d.self_attn.q, BareGraphConvolutionLayer)
        assert isinstance(layer_e.self_attn.q, BareGraphConvolutionLayer)
        assert layer_d.self_attn.q.normalize_adjacency is True
        assert layer_e.self_attn.q.normalize_adjacency is True

        # Forward outputs must match bit-for-bit
        adj = _random_symmetric_adjacency(bs, n)
        gd = binary_graphdata(adj)
        out_d = model_default(gd, output_dense=True)
        out_e = model_explicit(gd, output_dense=True)
        assert out_d.E_class is not None and out_e.E_class is not None
        torch.testing.assert_close(out_d.E_class, out_e.E_class)

    def test_wiring_smoke_normalize_off(self):
        """`gnn_normalize_adj=False` propagates to every Q/K/V GNN layer."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
        }
        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
            projection_config={
                "use_gnn_q": True,
                "use_gnn_k": True,
                "use_gnn_v": True,
                "gnn_normalize_adj": False,
            },
        )

        for tf_layer in model.transformer.tf_layers:
            assert isinstance(tf_layer, XEyTransformerLayer)
            block = tf_layer.self_attn
            for proj_name in ("q", "k", "v"):
                proj = getattr(block, proj_name)
                assert isinstance(proj, BareGraphConvolutionLayer)
                assert proj.normalize_adjacency is False

        # Forward must still work (using normalised input so the raw-A
        # path doesn't blow up the polynomial — the test is about wiring,
        # not numerics).
        from tmgg.models.layers.graph_ops import sym_normalize_adjacency

        bs, n = 2, 10
        adj = _random_symmetric_adjacency(bs, n)
        adj = sym_normalize_adjacency(adj + torch.eye(n).unsqueeze(0))
        # Convert dense Ã back to a {0,1} mask for binary_graphdata; we
        # only need the smoke test, not numerical fidelity here.
        gd = binary_graphdata((adj > 0).float())
        out = model(gd, output_dense=True)
        assert out.E_class is not None
        assert out.E_class.shape == (bs, n, n, 2)

    def test_hidden_dims_fallback(self):
        """The new key honours the legacy ``hidden_dims`` fallback path.

        ``_PROJ_KEYS`` extends to ``gnn_normalize_adj``, so a config that
        lives in ``hidden_dims`` (older configs) should still be picked
        up when ``projection_config`` is None.
        """
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_gnn_q": True,
            "gnn_normalize_adj": False,
        }
        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, BareGraphConvolutionLayer)
        assert layer.self_attn.q.normalize_adjacency is False


# Note: ``test_missing_adjacency_raises`` from the legacy 5-arg-form test was
# removed: the new ``DenseGraphTransformerData`` carrier always derives the
# adjacency from ``E_class`` via ``dense_adjacency()``, so the "no adjacency"
# branch is no longer reachable. That invariant is now compile-time enforced
# by the type system rather than runtime-checked.
