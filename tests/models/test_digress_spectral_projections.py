# pyright: reportAttributeAccessIssue=false
"""Tests for spectral Q/K/V projections in DiGress NodeEdgeBlock.

Test Rationale
--------------
These tests verify that spectral projections can be independently enabled for
query, key, and value projections in NodeEdgeBlock. Spectral projections use
eigenvalue polynomial filters (V @ W where W = sum_l Lambda^l * H[l]) instead
of linear or GNN-based projections.

Key invariants:
- Backwards compatibility: default params produce linear architecture
- Per-projection control: Q/K/V can independently use spectral projections
- Mutual exclusivity: GNN and spectral cannot both be enabled for same projection
- Gradient flow: gradients propagate through spectral polynomial filters
- Shape preservation: output shapes match regardless of projection type
"""

import pytest
import torch

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.datasets.graph_types import DenseGraphDistribution, GraphData
from tmgg.models.digress.data_types import DenseGraphTransformerData
from tmgg.models.digress.transformer_model import (
    GraphTransformer,
    NodeEdgeBlock,
    XEyTransformerLayer,
)
from tmgg.models.layers import SpectralProjectionLayer


def _make_block_input(
    *,
    bs: int,
    n: int,
    dx: int,
    de: int,
    dy: int,
    eigvec: torch.Tensor | None = None,
    eigval: torch.Tensor | None = None,
) -> DenseGraphTransformerData:
    """Build a DenseGraphTransformerData carrier for direct block tests."""
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
    return DenseGraphTransformerData.from_base(base, eigvec=eigvec, eigval=eigval)


class TestNodeEdgeBlockSpectral:
    """Tests for NodeEdgeBlock with spectral projections."""

    def test_backwards_compatibility(self):
        """Default params produce same architecture as original (all Linear)."""
        block = NodeEdgeBlock(dx=128, de=32, dy=64, n_head=4)

        # All projections should be Linear by default
        assert isinstance(block.q, torch.nn.Linear)
        assert isinstance(block.k, torch.nn.Linear)
        assert isinstance(block.v, torch.nn.Linear)

    def test_mutual_exclusivity(self):
        """GNN and spectral cannot both be enabled for the same projection."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeEdgeBlock(
                dx=128,
                de=32,
                dy=64,
                n_head=4,
                use_gnn_q=True,
                use_spectral_q=True,
            )

        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeEdgeBlock(
                dx=128,
                de=32,
                dy=64,
                n_head=4,
                use_gnn_k=True,
                use_spectral_k=True,
            )

        with pytest.raises(ValueError, match="mutually exclusive"):
            NodeEdgeBlock(
                dx=128,
                de=32,
                dy=64,
                n_head=4,
                use_gnn_v=True,
                use_spectral_v=True,
            )

    @pytest.mark.parametrize(
        "use_spectral_q,use_spectral_k,use_spectral_v",
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
    def test_selective_spectral_projections(
        self, use_spectral_q, use_spectral_k, use_spectral_v
    ):
        """Spectral projections can be enabled independently for Q, K, V."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_spectral_q=use_spectral_q,
            use_spectral_k=use_spectral_k,
            use_spectral_v=use_spectral_v,
            spectral_k=16,
        )

        expected_q = SpectralProjectionLayer if use_spectral_q else torch.nn.Linear
        expected_k = SpectralProjectionLayer if use_spectral_k else torch.nn.Linear
        expected_v = SpectralProjectionLayer if use_spectral_v else torch.nn.Linear

        assert isinstance(block.q, expected_q)
        assert isinstance(block.k, expected_k)
        assert isinstance(block.v, expected_v)

    def test_forward_with_spectral_projections(self):
        """Forward pass works with spectral projections enabled."""
        block = NodeEdgeBlock(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=True,
            use_spectral_v=True,
            spectral_k=16,
        )

        bs, n = 2, 20
        h = _make_block_input(
            bs=bs,
            n=n,
            dx=128,
            de=32,
            dy=64,
            eigvec=torch.rand(bs, n, 16),
            eigval=torch.rand(bs, 16),
        )
        out = block(h)

        assert out.X_class is not None and out.E_class is not None
        assert out.X_class.shape == (bs, n, 128)
        assert out.E_class.shape == (bs, n, n, 32)
        assert out.y.shape == (bs, 64)

    def test_forward_mixed_projections(self):
        """Forward pass works with mixed Linear and Spectral projections."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=False,  # Linear
            use_spectral_v=True,
            spectral_k=16,
        )

        bs, n = 2, 16
        h = _make_block_input(
            bs=bs,
            n=n,
            dx=64,
            de=16,
            dy=32,
            eigvec=torch.rand(bs, n, 16),
            eigval=torch.rand(bs, 16),
        )
        out = block(h)

        assert out.X_class is not None and out.E_class is not None
        assert out.X_class.shape == (bs, n, 64)
        assert out.E_class.shape == (bs, n, n, 16)
        assert out.y.shape == (bs, 32)

    def test_gradient_flow_through_spectral(self):
        """Gradients flow through spectral projections to input and parameters."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=True,
            use_spectral_v=True,
            spectral_k=8,
        )

        bs, n = 2, 12
        X = torch.rand(bs, n, 64, requires_grad=True)
        E = torch.rand(bs, n, n, 16)
        y = torch.rand(bs, 32)
        num_nodes = torch.full((bs,), n, dtype=torch.long)

        V = torch.rand(bs, n, 8, requires_grad=True)
        Lambda = torch.rand(bs, 8, requires_grad=True)

        base = DenseGraphDistribution(
            num_nodes_per_graph=num_nodes,
            y=y,
            X_class=X,
            E_class=E,
        )
        h = DenseGraphTransformerData.from_base(base, eigvec=V, eigval=Lambda)

        out = block(h)
        assert out.X_class is not None and out.E_class is not None
        loss = out.X_class.sum() + out.E_class.sum() + out.y.sum()
        loss.backward()

        # Check gradient flows to input
        assert X.grad is not None

        # Check gradient flows to spectral parameters
        assert isinstance(block.q, SpectralProjectionLayer)
        assert isinstance(block.k, SpectralProjectionLayer)
        assert isinstance(block.v, SpectralProjectionLayer)
        assert any(p.grad is not None for p in block.q.parameters())
        assert any(p.grad is not None for p in block.k.parameters())
        assert any(p.grad is not None for p in block.v.parameters())

    def test_configurable_spectral_params(self):
        """spectral_k and spectral_num_terms parameters control layer config."""
        block = NodeEdgeBlock(
            dx=64,
            de=16,
            dy=32,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=True,
            use_spectral_v=True,
            spectral_k=24,
            spectral_num_terms=5,
        )

        assert isinstance(block.q, SpectralProjectionLayer)
        assert block.q.k == 24
        assert block.q.num_terms == 5
        assert len(block.q.H) == 5


class TestXEyTransformerLayerSpectral:
    """Tests for XEyTransformerLayer with spectral configuration."""

    def test_spectral_config_passed_to_block(self):
        """Spectral config is correctly passed to NodeEdgeBlock."""
        layer = XEyTransformerLayer(
            dx=128,
            de=32,
            dy=64,
            n_head=4,
            use_spectral_q=True,
            use_spectral_k=False,
            use_spectral_v=True,
            spectral_k=16,
        )

        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert isinstance(layer.self_attn.k, torch.nn.Linear)
        assert isinstance(layer.self_attn.v, SpectralProjectionLayer)


class TestGraphTransformerSpectral:
    """Tests for full GraphTransformer with spectral projections."""

    def test_spectral_config_in_hidden_dims(self):
        """Spectral config flows through hidden_dims to transformer layers."""
        hidden_dims = {
            "dx": 128,
            "de": 32,
            "dy": 64,
            "n_head": 4,
            "use_spectral_q": True,
            "use_spectral_k": True,
            "use_spectral_v": False,
            "spectral_k": 16,
            "spectral_num_terms": 3,
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
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert isinstance(layer.self_attn.k, SpectralProjectionLayer)
        assert isinstance(layer.self_attn.v, torch.nn.Linear)

        # Verify spectral params
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert layer.self_attn.q.k == 16
        assert layer.self_attn.q.num_terms == 3

    def test_backwards_compatibility_no_spectral_keys(self):
        """GraphTransformer works without spectral keys in hidden_dims."""
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

    def test_forward_pass_with_spectral(self):
        """Full forward pass works with spectral projections."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_spectral_q": True,
            "use_spectral_k": True,
            "use_spectral_v": True,
            "spectral_k": 16,
        }

        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        # Forward pass with adjacency matrix
        x = torch.rand(2, 20, 20)
        x = (x + x.transpose(-1, -2)) / 2  # Symmetrize for eigendecomposition
        result = model(binary_graphdata(x), output_dense=True)

        assert isinstance(result, GraphData)
        assert result.E_class is not None
        assert result.E_class.shape == (2, 20, 20, 2)

    def test_eigenvector_extraction_for_spectral(self):
        """Eigenvectors are computed when spectral projections are used."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_spectral_q": True,
            "use_spectral_k": False,
            "use_spectral_v": False,
            "spectral_k": 16,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        # Verify model has spectral projection flag set
        assert model.transformer._use_spectral_projections is True

        # Forward pass should work
        x = torch.rand(2, 20, 20)
        x = (x + x.transpose(-1, -2)) / 2
        result = model(binary_graphdata(x), output_dense=True)

        assert isinstance(result, GraphData)
        assert result.E_class is not None
        assert result.E_class.shape == (2, 20, 20, 2)

    def test_gradient_flow_full_model(self):
        """Gradients flow through spectral projections in full model."""
        hidden_dims = {
            "dx": 32,
            "de": 8,
            "dy": 16,
            "n_head": 2,
            "use_spectral_q": True,
            "use_spectral_k": True,
            "use_spectral_v": True,
            "spectral_k": 8,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 8, "y": 16},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        x = torch.rand(2, 12, 12)
        x = (x + x.transpose(-1, -2)) / 2

        result = model(binary_graphdata(x), output_dense=True)
        assert result.E_class is not None
        # Use only edge-probability channel: both channels of 2-class encoding
        # sum to 1.0 per position, so E.sum() is constant with zero gradient.
        loss = result.E_class[..., 1].sum()
        loss.backward()

        # Verify gradients flow to spectral projection parameters
        layer = model.transformer.tf_layers[0]
        assert isinstance(layer, XEyTransformerLayer)
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.self_attn.q.parameters()
        )

    def test_spectral_uses_correct_adjacency_channel(self):
        """Spectral eigendecomposition uses actual edges, not the no-edge channel.

        Regression test for a bug where E[..., 0] (no-edge probability) was
        passed to the eigen layer instead of the binary adjacency derived from
        argmax. With two-class encoding, E[..., 0] = 1 - adj and E[..., 1] = adj,
        so using channel 0 computes eigenvectors of the complement graph.
        """
        hidden_dims = {
            "dx": 32,
            "de": 8,
            "dy": 16,
            "n_head": 2,
            "use_spectral_q": True,
            "use_spectral_k": False,
            "use_spectral_v": False,
            "spectral_k": 4,
        }

        model = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 8, "y": 16},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
        )

        # Build a sparse path graph: 0-1-2-3 (4 edges out of 12 possible)
        n = 4
        adj = torch.zeros(1, n, n)
        for i in range(n - 1):
            adj[0, i, i + 1] = 1.0
            adj[0, i + 1, i] = 1.0

        gd = binary_graphdata(adj)

        # Verify encoding: channel 0 is no-edge, channel 1 is edge
        assert gd.E_class is not None
        assert gd.E_class[0, 0, 1, 1] == 1.0, "Edge (0,1) should have class 1"
        assert gd.E_class[0, 0, 1, 0] == 0.0, "Edge (0,1) should NOT have class 0"
        assert gd.E_class[0, 0, 2, 0] == 1.0, "Non-edge (0,2) should have class 0"

        # Hook into the eigen_layer to capture the adjacency it receives
        captured = {}
        assert model.transformer.eigen_layer is not None
        original_forward = model.transformer.eigen_layer.forward

        def capture_forward(adj_input):
            captured["adj"] = adj_input.detach().clone()
            return original_forward(adj_input)

        model.transformer.eigen_layer.forward = capture_forward  # type: ignore[assignment]

        with torch.no_grad():
            model(gd, output_dense=True)

        assert "adj" in captured, "Eigen layer should have been called"
        received_adj = captured["adj"][0]

        # The received adjacency should match the actual graph, not its complement
        assert received_adj[0, 1] == 1.0, "Edge (0,1) should be 1 in eigen input"
        assert received_adj[0, 2] == 0.0, "Non-edge (0,2) should be 0 in eigen input"


class TestNormalizeEigenvaluesFlag:
    """Tests for the optional ``normalize_eigenvalues`` flag.

    See ``rationales/digress_spectral_projection_normalize_test_rationale.md``.
    """

    @staticmethod
    def _nondegenerate_lambda(bs: int, k: int) -> torch.Tensor:
        """Random Lambda with min-magnitude bounded away from zero.

        The default normalization is ``Lambda / max|Lambda|`` clamped at
        ``1e-6``; using non-trivial magnitudes ensures the rescale is a
        non-degenerate division so the off≡on identity actually
        exercises both branches.
        """
        Lam = torch.randn(bs, k)
        # Floor magnitudes at 0.1 so max|.| is well above the 1e-6 clamp
        Lam = Lam.sign() * Lam.abs().clamp(min=0.1)
        return Lam

    def test_layer_identity_off_on_normalized_input(self):
        """``normalize=False`` on Λ̄ ≡ ``normalize=True`` on Λ.

        With identical weights, feeding the per-graph rescaled
        eigenvalues through a layer that does *not* normalize must
        match feeding the raw eigenvalues through a layer that *does*
        normalize. Pins the new flag's semantics.
        """
        torch.manual_seed(0)
        bs, n, k, out_dim = 2, 8, 6, 12
        V = torch.randn(bs, n, k)
        Lam = self._nondegenerate_lambda(bs, k)

        layer_on = SpectralProjectionLayer(
            k=k, out_dim=out_dim, num_terms=3, normalize_eigenvalues=True
        )
        layer_off = SpectralProjectionLayer(
            k=k, out_dim=out_dim, num_terms=3, normalize_eigenvalues=False
        )
        # Tie weights so the only difference is the gating of the rescale
        for h_on, h_off in zip(layer_on.H, layer_off.H, strict=True):
            h_off.data.copy_(h_on.data)
        layer_off.out_proj.weight.data.copy_(layer_on.out_proj.weight.data)

        Lam_max = Lam.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        Lam_norm = Lam / Lam_max

        out_on = layer_on(V, Lam)
        out_off = layer_off(V, Lam_norm)
        torch.testing.assert_close(out_on, out_off, rtol=1e-5, atol=1e-6)

    def test_layer_off_uses_raw_eigenvalues(self):
        """``normalize=False`` actually skips the rescale.

        On Lambda whose max magnitude is not 1, the two layers must
        produce different outputs when given the same input — otherwise
        the flag is silently dead.
        """
        torch.manual_seed(0)
        bs, n, k, out_dim = 2, 8, 6, 12
        V = torch.randn(bs, n, k)
        Lam = self._nondegenerate_lambda(bs, k) * 3.0  # max|Lam| ≠ 1

        layer_on = SpectralProjectionLayer(
            k=k, out_dim=out_dim, num_terms=3, normalize_eigenvalues=True
        )
        layer_off = SpectralProjectionLayer(
            k=k, out_dim=out_dim, num_terms=3, normalize_eigenvalues=False
        )
        for h_on, h_off in zip(layer_on.H, layer_off.H, strict=True):
            h_off.data.copy_(h_on.data)
        layer_off.out_proj.weight.data.copy_(layer_on.out_proj.weight.data)

        out_on = layer_on(V, Lam)
        out_off = layer_off(V, Lam)
        assert not torch.allclose(out_on, out_off, rtol=1e-3, atol=1e-4)

    def test_default_preserved_end_to_end(self):
        """An unspecified normalize flag matches an explicit ``True``.

        Guards against an accidental default flip in the plumbing. We
        equalise weights by copying a state_dict between the two
        models; a seed alone is insufficient because the differing
        ``projection_config`` dicts alter the order in which the RNG
        is consumed during ``__init__``.
        """
        bs, n = 2, 12

        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
        }

        torch.manual_seed(42)
        model_default = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
            projection_config={"use_spectral_q": True, "spectral_k": 8},
        )
        model_explicit = GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
            hidden_dims=hidden_dims,
            output_dims={"X": 0, "E": 2, "y": 0},
            projection_config={
                "use_spectral_q": True,
                "spectral_k": 8,
                "spectral_normalize_eigenvalues": True,
            },
        )
        # Equalise weights so any output difference is due to the
        # config branch under test, not random initialisation.
        model_explicit.load_state_dict(model_default.state_dict())
        model_default.eval()
        model_explicit.eval()

        layer_d = model_default.transformer.tf_layers[0]
        layer_e = model_explicit.transformer.tf_layers[0]
        assert isinstance(layer_d, XEyTransformerLayer)
        assert isinstance(layer_e, XEyTransformerLayer)
        assert isinstance(layer_d.self_attn.q, SpectralProjectionLayer)
        assert isinstance(layer_e.self_attn.q, SpectralProjectionLayer)
        assert layer_d.self_attn.q.normalize_eigenvalues is True
        assert layer_e.self_attn.q.normalize_eigenvalues is True

        adj = torch.rand(bs, n, n)
        adj = (adj > 0.5).float()
        adj = (adj + adj.transpose(-1, -2)).clamp(max=1.0)
        adj.diagonal(dim1=1, dim2=2).zero_()
        gd = binary_graphdata(adj)
        out_d = model_default(gd, output_dense=True)
        out_e = model_explicit(gd, output_dense=True)
        assert out_d.E_class is not None and out_e.E_class is not None
        torch.testing.assert_close(out_d.E_class, out_e.E_class)

    def test_wiring_smoke_normalize_off(self):
        """``spectral_normalize_eigenvalues=False`` propagates to every Q/K/V."""
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
                "use_spectral_q": True,
                "use_spectral_k": True,
                "use_spectral_v": True,
                "spectral_k": 8,
                "spectral_normalize_eigenvalues": False,
            },
        )

        for tf_layer in model.transformer.tf_layers:
            assert isinstance(tf_layer, XEyTransformerLayer)
            block = tf_layer.self_attn
            for proj_name in ("q", "k", "v"):
                proj = getattr(block, proj_name)
                assert isinstance(proj, SpectralProjectionLayer)
                assert proj.normalize_eigenvalues is False

        # Forward must still work end-to-end. The eigen_layer feeds
        # eigenvalues from a real adjacency, so |λ| stays bounded.
        bs, n = 2, 12
        adj = torch.rand(bs, n, n)
        adj = (adj > 0.5).float()
        adj = (adj + adj.transpose(-1, -2)).clamp(max=1.0)
        adj.diagonal(dim1=1, dim2=2).zero_()
        out = model(binary_graphdata(adj), output_dense=True)
        assert out.E_class is not None
        assert out.E_class.shape == (bs, n, n, 2)

    def test_hidden_dims_fallback(self):
        """The new key honours the legacy ``hidden_dims`` fallback path."""
        hidden_dims = {
            "dx": 64,
            "de": 16,
            "dy": 32,
            "n_head": 4,
            "use_spectral_q": True,
            "spectral_k": 8,
            "spectral_normalize_eigenvalues": False,
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
        assert isinstance(layer.self_attn.q, SpectralProjectionLayer)
        assert layer.self_attn.q.normalize_eigenvalues is False
