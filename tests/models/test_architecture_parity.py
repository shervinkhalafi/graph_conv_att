"""Parity tests for Wave 7 architecture refactor.

Test rationale
--------------
Spec goal G2 (``internal notes (not included)``) asks
each architecture family to populate the same ``_class`` / ``_feat`` fields
that are populated in the input. These tests construct dual-populated
``GraphData`` (legacy ``X`` / ``E`` plus the new split fields) and assert
that the forward pass returns a ``GraphData`` with both legacy and split
fields populated and shapes preserved.

One representative is exercised per family:

- DiGress : ``GraphTransformer`` (Wave 7.1)
- GNN     : ``GNNDenoiser`` (Wave 7.2) — TODO commit 7.2
- Spectral: ``LinearPEDenoiser`` (Wave 7.3) — TODO commit 7.3
- Baseline: ``LinearBaseline`` (Wave 7.4) — TODO commit 7.4
- Hybrid  : ``SequentialDenoiser`` (Wave 7.4) — TODO commit 7.4
- Attention: ``MultiLayerAttentionDenoiser`` (Wave 7.4) — TODO commit 7.4

Tests for the not-yet-migrated families are added in their respective
commits.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
)
from tmgg.models.attention.attention import MultiLayerAttention
from tmgg.models.baselines.linear import LinearBaseline
from tmgg.models.baselines.mlp import MLPBaseline
from tmgg.models.digress.transformer_model import GraphTransformer
from tmgg.models.gnn.gnn import GNN
from tmgg.models.gnn.gnn_sym import GNNSymmetric
from tmgg.models.gnn.nvgnn import NodeVarGNN
from tmgg.models.hybrid.hybrid import SequentialDenoisingModel
from tmgg.models.spectral_denoisers.bilinear import (
    BilinearDenoiser,
    BilinearDenoiserWithMLP,
    MultiLayerBilinearDenoiser,
)
from tmgg.models.spectral_denoisers.filter_bank import GraphFilterBank
from tmgg.models.spectral_denoisers.linear_pe import LinearPE
from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser

BS = 2
N = 6
DX = 2
DE = 2
DY = 1


def _make_categorical_graphdata() -> DenseGraphState:
    """Construct a DenseGraphState with X_class / E_class populated."""
    X = torch.randn(BS, N, DX)
    E = torch.randn(BS, N, N, DE)
    # Symmetrise edges to satisfy the GraphData symmetry contract.
    E = 0.5 * (E + E.transpose(1, 2))
    y = torch.randn(BS, DY)
    num_nodes_per_graph = torch.full((BS,), N, dtype=torch.long)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=y,
        X_class=X,
        E_class=E,
    )


class TestDigressGraphTransformerParity:
    """GraphTransformer (Wave 7.1) — categorical X_class/E_class path."""

    def _build(self) -> GraphTransformer:
        return GraphTransformer(
            n_layers=2,
            input_dims={"X": DX, "E": DE, "y": DY},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX, "E": DE, "y": DY},
        )

    def test_categorical_input_yields_class_output(self) -> None:
        model = self._build()
        data = _make_categorical_graphdata()
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.E_class is not None
        assert out.X_class.shape == (BS, N, DX)
        assert out.E_class.shape == (BS, N, N, DE)
        # Categorical architecture leaves _feat fields None.
        assert out.X_feat is None
        assert out.E_feat is None

    @pytest.mark.parametrize("c_x", [1, 2])
    def test_structure_only_input_synthesises_node_features(self, c_x: int) -> None:
        """Structure-only input (X_class=None) still runs: the architecture
        synthesises a per-node feature from node_mask internally.

        Parametrized over ``c_x ∈ {1, 2}`` per 2026-04-27 spec §3:
        - C_x = 1 is the canonical structure-only encoding (one class).
        - C_x = 2 is the legacy ``[no-node, node]`` encoding.
        Both paths must produce a valid model output of width ``c_x``.
        """
        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": c_x, "E": DE, "y": DY},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": c_x, "E": DE, "y": DY},
        )
        E = torch.randn(BS, N, N, DE)
        E = 0.5 * (E + E.transpose(1, 2))
        # GraphTransformer's extra_features is None here; synth yields
        # an X tensor of width ``c_x`` per the canonical helper.
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((BS,), N, dtype=torch.long),
            E_class=E,
            y=torch.randn(BS, DY),
        )
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.E_class is not None
        assert out.X_class.shape == (BS, N, c_x)
        assert out.E_class.shape == (BS, N, N, DE)

    def test_structure_only_input_raises_for_c_x_geq_3(self) -> None:
        """C_x>=3 with X_class=None must raise — real categorical X must
        come from the dataset (2026-04-27 spec §3).
        """
        c_x = 3
        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": c_x, "E": DE, "y": DY},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": c_x, "E": DE, "y": DY},
        )
        E = torch.randn(BS, N, N, DE)
        E = 0.5 * (E + E.transpose(1, 2))
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((BS,), N, dtype=torch.long),
            E_class=E,
            y=torch.randn(BS, DY),
        )
        with pytest.raises(ValueError, match=r"C_x"):
            model(data)

    def test_output_dims_constructor_params_accepted(self) -> None:
        """Wave 7 contract: output_dims_{x,e}_{class,feat} are constructor params."""
        model = GraphTransformer(
            n_layers=2,
            input_dims={"X": DX, "E": DE, "y": DY},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX, "E": DE, "y": DY},
            output_dims_x_class=DX,
            output_dims_x_feat=None,
            output_dims_e_class=DE,
            output_dims_e_feat=None,
        )
        assert model.output_dims_x_class == DX
        assert model.output_dims_x_feat is None
        assert model.output_dims_e_class == DE
        assert model.output_dims_e_feat is None


def _make_feat_populated_graphdata() -> DenseGraphState:
    """Construct a DenseGraphState with ``E_feat`` populated (denoising-path input)."""
    edge = torch.rand(BS, N, N)
    edge = 0.5 * (edge + edge.transpose(1, 2))  # symmetrise
    diag = torch.arange(N)
    edge[:, diag, diag] = 0.0
    e_feat = edge.unsqueeze(-1)
    return DenseGraphState(
        num_nodes_per_graph=torch.full((BS,), N, dtype=torch.long),
        y=torch.zeros(BS, 0),
        E_feat=e_feat,
    )


class TestGNNFamilyParity:
    """GNN family (Wave 7.2) — scalar denoising path on E_feat."""

    def test_gnn_feat_input_yields_efeat_output(self) -> None:
        model = GNN(num_layers=2, num_terms=2, feature_dim_in=4, feature_dim_out=4)
        data = _make_feat_populated_graphdata()
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        # Default edge_source="feat" + output_dims_e_feat=1 → writes to E_feat.
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)
        # Categorical field untouched (GNN is not configured to predict it).
        assert out.E_class is None

    def test_gnn_symmetric_runs_on_e_feat_input(self) -> None:
        model = GNNSymmetric(
            num_layers=2, num_terms=2, feature_dim_in=4, feature_dim_out=4
        )
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)
        assert out.E_class is None

    def test_node_var_gnn_runs_on_e_feat_input(self) -> None:
        model = NodeVarGNN(num_layers=2, num_terms=2, feature_dim=4)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)
        assert out.E_class is None

    def test_gnn_configurable_for_class_output(self) -> None:
        """Switching ``output_dims_e_class=2`` writes a one-hot E_class."""
        model = GNN(
            num_layers=2,
            num_terms=2,
            feature_dim_in=4,
            feature_dim_out=4,
            edge_source="class",
            output_dims_e_class=2,
            output_dims_e_feat=None,
        )
        # Build a class-populated input.
        bs, n = 2, 5
        adj = torch.zeros(bs, n, n)
        adj[:, 0, 1] = adj[:, 1, 0] = 1.0
        e_class = torch.stack([1.0 - adj, adj], dim=-1)
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((bs,), n, dtype=torch.long),
            y=torch.zeros(bs, 0),
            E_class=e_class,
        )
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_class is not None
        assert out.E_class.shape == (bs, n, n, 2)
        assert out.E_feat is None

    def test_gnn_accepts_timestep_concatenation(self) -> None:
        """Spec contract: architectures concat ``t`` onto ``data.y``."""
        model = GNN(num_layers=2, num_terms=2, feature_dim_in=4, feature_dim_out=4)
        data = _make_feat_populated_graphdata()
        t = torch.rand(BS)
        out = model(data, t=t)
        # y started at width 0; +1 from t.
        assert out.y.shape == (BS, 1)


class TestSpectralFamilyParity:
    """Spectral family (Wave 7.3) — edge_source-configurable scalar denoisers."""

    def test_linear_pe_feat_input_yields_efeat_output(self) -> None:
        model = LinearPE(k=4, max_nodes=N, use_bias=True)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)
        assert out.E_class is None

    def test_graph_filter_bank_runs(self) -> None:
        model = GraphFilterBank(k=4, polynomial_degree=3)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_self_attention_runs(self) -> None:
        model = SelfAttentionDenoiser(k=4, d_k=8)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_bilinear_runs(self) -> None:
        model = BilinearDenoiser(k=4, d_k=8)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_bilinear_with_mlp_runs(self) -> None:
        model = BilinearDenoiserWithMLP(k=4, d_k=8, mlp_hidden_dim=16, mlp_num_layers=2)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_multilayer_bilinear_runs(self) -> None:
        model = MultiLayerBilinearDenoiser(
            k=4, d_model=8, num_heads=2, num_layers=2, use_mlp=False
        )
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_linear_pe_accepts_timestep(self) -> None:
        model = LinearPE(k=4, max_nodes=N, use_bias=True)
        t = torch.rand(BS)
        out = model(_make_feat_populated_graphdata(), t=t)
        assert out.y.shape == (BS, 1)

    def test_linear_pe_class_output_mode(self) -> None:
        """Config-only flip: writes two-channel E_class instead of E_feat."""
        model = LinearPE(
            k=4,
            max_nodes=N,
            use_bias=True,
            edge_source="class",
            output_dims_e_class=2,
            output_dims_e_feat=None,
        )
        bs, n = 2, 5
        adj = torch.zeros(bs, n, n)
        adj[:, 0, 1] = adj[:, 1, 0] = 1.0
        e_class = torch.stack([1.0 - adj, adj], dim=-1)
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((bs,), n, dtype=torch.long),
            y=torch.zeros(bs, 0),
            E_class=e_class,
        )
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_class is not None
        assert out.E_class.shape == (bs, n, n, 2)
        assert out.E_feat is None


class TestBaselinesHybridAttentionParity:
    """Baselines (linear, mlp), hybrid sequential denoiser, and attention (Wave 7.4)."""

    def test_linear_baseline_runs(self) -> None:
        model = LinearBaseline(max_nodes=N)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)
        assert out.E_class is None

    def test_mlp_baseline_runs(self) -> None:
        model = MLPBaseline(max_nodes=N, hidden_dim=16, num_layers=2)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_multilayer_attention_runs(self) -> None:
        model = MultiLayerAttention(d_model=N, num_heads=1, num_layers=2)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_sequential_denoiser_runs(self) -> None:
        embedding = GNN(num_layers=2, num_terms=2, feature_dim_in=4, feature_dim_out=4)
        model = SequentialDenoisingModel(embedding_model=embedding)
        out = model(_make_feat_populated_graphdata(), output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_feat is not None
        assert out.E_feat.shape == (BS, N, N, 1)

    def test_linear_baseline_accepts_timestep(self) -> None:
        model = LinearBaseline(max_nodes=N)
        t = torch.rand(BS)
        out = model(_make_feat_populated_graphdata(), t=t)
        assert out.y.shape == (BS, 1)

    def test_mlp_baseline_class_output_mode(self) -> None:
        """Config-only flip on the MLP baseline emits E_class."""
        model = MLPBaseline(
            max_nodes=N,
            hidden_dim=16,
            num_layers=2,
            edge_source="class",
            output_dims_e_class=2,
            output_dims_e_feat=None,
        )
        bs, n = 2, 5
        adj = torch.zeros(bs, n, n)
        adj[:, 0, 1] = adj[:, 1, 0] = 1.0
        e_class = torch.stack([1.0 - adj, adj], dim=-1)
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((bs,), n, dtype=torch.long),
            y=torch.zeros(bs, 0),
            E_class=e_class,
        )
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.E_class is not None
        assert out.E_class.shape == (bs, n, n, 2)
        assert out.E_feat is None
