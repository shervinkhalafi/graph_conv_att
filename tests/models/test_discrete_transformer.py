"""Tests for GraphTransformer model.

Test rationale: The graph transformer is the core neural network in the
diffusion pipeline. Shape mismatches between its output and the loss/sampling
modules cause silent dimension errors that are hard to trace. These tests
verify shape correctness, output finiteness, and config serialization.
"""

import pytest
import torch
from torch import nn

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
)
from tmgg.models.digress.data_types import DenseGraphTransformerData
from tmgg.models.digress.extra_features import EigenvectorAugmentation
from tmgg.models.digress.transformer_model import GraphTransformer, _GraphTransformer

BS = 4
N = 10
DX_IN = 2
DE_IN = 2
DY_IN = 1
# Output dims must be <= input dims due to the skip connection in
# _GraphTransformer (X_to_out = X[..., :out_dim_X]).
DX_OUT = 2
DE_OUT = 2
DY_OUT = 1


@pytest.fixture()
def model() -> GraphTransformer:
    """Small GraphTransformer for testing."""
    return GraphTransformer(
        n_layers=2,
        input_dims={"X": DX_IN, "E": DE_IN, "y": DY_IN},
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
        output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
    )


@pytest.fixture()
def graph_data() -> DenseGraphState:
    """Random categorical input as DenseGraphState."""
    X = torch.randn(BS, N, DX_IN)
    E = torch.randn(BS, N, N, DE_IN)
    return DenseGraphState(
        num_nodes_per_graph=torch.full((BS,), N, dtype=torch.long),
        y=torch.randn(BS, DY_IN),
        X_class=X,
        E_class=E,
    )


class TestForwardPass:
    """Verify forward pass produces correctly shaped, finite outputs."""

    def test_output_is_graph_data(
        self,
        model: GraphTransformer,
        graph_data: DenseGraphState,
    ) -> None:
        """Forward should return a DenseGraphDistribution under output_dense=True."""
        out = model(graph_data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)

    def test_output_shapes(
        self,
        model: GraphTransformer,
        graph_data: DenseGraphState,
    ) -> None:
        """X, E, y output shapes must match the configured output_dims."""
        out = model(graph_data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.E_class is not None
        assert out.X_class.shape == (BS, N, DX_OUT)
        assert out.E_class.shape == (BS, N, N, DE_OUT)
        assert out.y.shape == (BS, DY_OUT)

    def test_outputs_finite(
        self,
        model: GraphTransformer,
        graph_data: DenseGraphState,
    ) -> None:
        """All output tensors must contain finite values (no NaN or Inf)."""
        out = model(graph_data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.E_class is not None
        assert torch.isfinite(out.X_class).all(), "X contains NaN or Inf"
        assert torch.isfinite(out.E_class).all(), "E contains NaN or Inf"
        assert torch.isfinite(out.y).all(), "y contains NaN or Inf"

    def test_partial_node_mask(
        self,
        model: GraphTransformer,
    ) -> None:
        """Forward pass works when some nodes are masked out."""
        # 7 real nodes per graph, last 3 padded.
        num_nodes_per_graph = torch.full((BS,), 7, dtype=torch.long)
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        data = DenseGraphState(
            num_nodes_per_graph=num_nodes_per_graph,
            y=torch.randn(BS, DY_IN),
            X_class=X,
            E_class=E,
        )
        out = model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.X_class.shape == (BS, N, DX_OUT)
        assert torch.isfinite(out.X_class).all()


class TestGetConfig:
    """Verify get_config returns a complete, correct configuration dict."""

    def test_expected_keys(self, model: GraphTransformer) -> None:
        config = model.get_config()
        expected_keys = {
            "n_layers",
            "input_dims",
            "hidden_mlp_dims",
            "hidden_dims",
            "output_dims",
            "extra_features",
            "use_timestep",
            "use_upstream_hidden_edge_diagonal",
        }
        assert set(config.keys()) == expected_keys

    def test_n_layers_value(self, model: GraphTransformer) -> None:
        config = model.get_config()
        assert config["n_layers"] == 2

    def test_augmentation_defaults(self, model: GraphTransformer) -> None:
        """Vanilla model defaults: no extras, no timestep, upstream-parity edge diagonal."""
        config = model.get_config()
        assert config["extra_features"] is None
        assert config["use_timestep"] is False
        assert config["use_upstream_hidden_edge_diagonal"] is True


class TestEigenvectorMode:
    """Verify the eigenvector augmentation path through GraphTransformer.

    When an EigenvectorAugmentation is passed as extra_features, the
    transformer extracts top-k eigenvectors from the noisy adjacency and
    concatenates them with X before the inner transformer. The augmentation
    handles dim adjustment via adjust_dims, so input_dims uses base dims only.
    """

    K = 5

    @pytest.fixture()
    def eigen_model(self) -> GraphTransformer:
        return GraphTransformer(
            n_layers=2,
            input_dims={"X": DX_IN, "E": DE_IN, "y": DY_IN},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX_OUT, "E": DE_OUT, "y": DY_OUT},
            extra_features=EigenvectorAugmentation(k=self.K),
        )

    def test_forward_shape(self, eigen_model: GraphTransformer) -> None:
        """Eigenvector path produces correct output shapes."""
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((BS,), N, dtype=torch.long),
            y=torch.randn(BS, DY_IN),
            X_class=X,
            E_class=E,
        )
        out = eigen_model(data, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.E_class is not None
        assert out.X_class.shape == (BS, N, DX_OUT)
        assert out.E_class.shape == (BS, N, N, DE_OUT)

    def test_config_reports_augmentation(self, eigen_model: GraphTransformer) -> None:
        config = eigen_model.get_config()
        assert config["extra_features"] == "EigenvectorAugmentation"
        assert config["use_timestep"] is False


class TestTimestepMode:
    """Verify the timestep injection path through GraphTransformer.

    When use_timestep=True, the transformer appends the normalised diffusion
    timestep to y before the inner transformer, adding one dimension to y.
    """

    @pytest.fixture()
    def timestep_model(self) -> GraphTransformer:
        return GraphTransformer(
            n_layers=2,
            input_dims={"X": DX_IN, "E": DE_IN, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 8, "n_head": 2},
            output_dims={"X": DX_OUT, "E": DE_OUT, "y": 0},
            use_timestep=True,
        )

    def test_forward_with_timestep(self, timestep_model: GraphTransformer) -> None:
        """Forward pass with a timestep tensor produces correct output shapes."""
        X = torch.randn(BS, N, DX_IN)
        E = torch.randn(BS, N, N, DE_IN)
        data = DenseGraphState(
            num_nodes_per_graph=torch.full((BS,), N, dtype=torch.long),
            y=torch.zeros(BS, 0),
            X_class=X,
            E_class=E,
        )
        t = torch.rand(BS)
        out = timestep_model(data, t=t, output_dense=True)
        assert isinstance(out, DenseGraphDistribution)
        assert out.X_class is not None
        assert out.X_class.shape == (BS, N, DX_OUT)

    def test_config_reports_timestep(self, timestep_model: GraphTransformer) -> None:
        config = timestep_model.get_config()
        assert config["use_timestep"] is True


class _CaptureTransformerLayer(nn.Module):
    """Record the hidden edge tensor passed into the transformer stack.

    The post-refactor layer signature is ``forward(h: DenseGraphTransformerData)
    -> DenseGraphTransformerData``; this stub captures ``h.E_class`` and
    returns the input unchanged.
    """

    captured_E: torch.Tensor | None

    def __init__(self) -> None:
        super().__init__()
        self.captured_E = None

    def forward(
        self, h: DenseGraphTransformerData
    ) -> DenseGraphTransformerData:
        assert h.E_class is not None
        self.captured_E = h.E_class.detach().clone()
        return h


class TestHiddenEdgeDiagonalParity:
    """Verify the DiGress hidden-edge diagonal compatibility toggle.

    Test rationale: upstream DiGress padding-masks the hidden edge tensor
    after ``mlp_in_E`` but does not zero its diagonal until the final
    residual output. TMGG currently calls ``mask_zero_diag()`` at that
    pre-layer boundary. The new flag must keep the current default while
    allowing ``repro_exact`` to opt into live upstream behavior.
    """

    @staticmethod
    def _model(*, use_upstream_hidden_edge_diagonal: bool) -> _GraphTransformer:
        model = _GraphTransformer(
            n_layers=1,
            input_dims={"X": 2, "E": 2, "y": 1},
            hidden_mlp_dims={"X": 8, "E": 8, "y": 8},
            hidden_dims={"dx": 8, "de": 4, "dy": 4, "n_head": 2},
            output_dims={"X": 2, "E": 2, "y": 1},
            use_upstream_hidden_edge_diagonal=use_upstream_hidden_edge_diagonal,
        )
        capture = _CaptureTransformerLayer()
        model.tf_layers = nn.ModuleList([capture])
        return model

    @staticmethod
    def _graph_data() -> DenseGraphState:
        bs, n = 2, 4
        # First graph keeps all 4 nodes; second has 3 real + 1 padded.
        num_nodes_per_graph = torch.tensor([n, n - 1], dtype=torch.long)
        E = torch.zeros(bs, n, n, 2)
        E[..., 0] = 1.0
        E[:, 0, 1, :] = torch.tensor([0.0, 1.0])
        E[:, 1, 0, :] = torch.tensor([0.0, 1.0])
        diag = torch.arange(n)
        E[:, diag, diag, :] = 0.0
        return DenseGraphState(
            num_nodes_per_graph=num_nodes_per_graph,
            y=torch.zeros(bs, 1),
            X_class=torch.ones(bs, n, 2),
            E_class=E,
        )

    def test_default_zeros_hidden_edge_diagonal(self) -> None:
        """Default behavior remains the existing pre-layer zero diagonal."""
        torch.manual_seed(0)
        model = self._model(use_upstream_hidden_edge_diagonal=False)
        capture = model.tf_layers[0]
        data = self._graph_data()
        _ = model(data)
        assert isinstance(capture, _CaptureTransformerLayer)
        assert capture.captured_E is not None
        assert data.E_class is not None
        n = int(data.E_class.shape[1])
        diag = torch.arange(n)
        assert torch.count_nonzero(capture.captured_E[:, diag, diag, :]) == 0

    def test_upstream_toggle_preserves_hidden_edge_diagonal(self) -> None:
        """Enabled flag keeps the live upstream padding-only pre-layer mask."""
        torch.manual_seed(0)
        model = self._model(use_upstream_hidden_edge_diagonal=True)
        capture = model.tf_layers[0]
        data = self._graph_data()
        _ = model(data)
        assert isinstance(capture, _CaptureTransformerLayer)
        assert capture.captured_E is not None
        assert data.E_class is not None
        n = int(data.E_class.shape[1])
        diag = torch.arange(n)
        real_node_diag = capture.captured_E[:, diag, diag, :][data.node_mask]
        assert torch.count_nonzero(real_node_diag) > 0
