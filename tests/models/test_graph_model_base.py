# pyright: reportAttributeAccessIssue=false
"""Tests for the GraphModel abstract base class.

Testing Strategy
----------------
GraphModel is the unified model interface introduced by the training loop
unification design. It extends BaseModel and requires a single abstract
method: ``forward(data: GraphData, t: Tensor | None) -> GraphData``.

These tests verify:
1. GraphModel cannot be instantiated directly (it is abstract).
2. A concrete subclass that implements ``forward`` and ``get_config`` works.
3. ``parameter_count()`` is inherited from BaseModel and functions correctly.
4. ``forward`` accepts both with-timestep and without-timestep calls.
5. ``binary_graphdata()`` round-trips correctly through a trivial model.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.base import BaseModel, GraphModel, get_parameter_count_int

# -- Concrete test fixture ---------------------------------------------------


class IdentityGraphModel(GraphModel):
    """Minimal concrete GraphModel that returns input unchanged.

    Includes a dummy linear layer so ``parameter_count()`` has something
    to report.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden, hidden)

    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData:
        """Return data unchanged (identity model)."""
        return data

    def get_config(self) -> dict[str, Any]:
        return {"hidden": self.linear.in_features}


class MissingGetConfigModel(GraphModel):
    """Subclass that implements forward but omits get_config.

    Should still be abstract because get_config is required by BaseModel.
    """

    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData:
        return data


class MissingForwardModel(GraphModel):
    """Subclass that implements get_config but omits forward.

    Should still be abstract because forward is required by GraphModel.
    """

    def get_config(self) -> dict[str, Any]:
        return {}


# -- Abstractness tests ------------------------------------------------------


class TestGraphModelAbstract:
    """GraphModel and incomplete subclasses cannot be instantiated."""

    def test_graphmodel_is_abstract(self) -> None:
        """GraphModel itself cannot be instantiated because forward and
        get_config are abstract."""
        with pytest.raises(TypeError, match="abstract method"):
            GraphModel()  # type: ignore[abstract]

    def test_missing_get_config_is_abstract(self) -> None:
        """A subclass that only implements forward (not get_config) remains
        abstract."""
        with pytest.raises(TypeError, match="abstract method"):
            MissingGetConfigModel()  # type: ignore[abstract]

    def test_missing_forward_is_abstract(self) -> None:
        """A subclass that only implements get_config (not forward) remains
        abstract."""
        with pytest.raises(TypeError, match="abstract method"):
            MissingForwardModel()  # type: ignore[abstract]


# -- Inheritance tests -------------------------------------------------------


class TestGraphModelInheritance:
    """GraphModel is a proper subclass of BaseModel."""

    def test_subclass_of_base_model(self) -> None:
        assert issubclass(GraphModel, BaseModel)

    def test_concrete_instance_is_base_model(self) -> None:
        model = IdentityGraphModel()
        assert isinstance(model, BaseModel)
        assert isinstance(model, GraphModel)


# -- Concrete subclass tests ------------------------------------------------


class TestConcreteGraphModel:
    """Verify that a complete concrete subclass works end-to-end."""

    @pytest.fixture
    def model(self) -> IdentityGraphModel:
        return IdentityGraphModel(hidden=8)

    @pytest.fixture
    def batched_data(self) -> GraphData:
        """A small batched GraphData from adjacency matrices."""
        adj = torch.zeros(2, 4, 4)
        adj[0, 0, 1] = 1.0
        adj[0, 1, 0] = 1.0
        adj[1, 0, 2] = 1.0
        adj[1, 2, 0] = 1.0
        return binary_graphdata(adj)

    def test_forward_without_timestep(
        self, model: IdentityGraphModel, batched_data: GraphData
    ) -> None:
        """forward(data) works when t is None (unconditional)."""
        result = model(batched_data)
        assert isinstance(result, GraphData)
        assert result.X_class is not None and batched_data.X_class is not None
        assert result.E_class is not None and batched_data.E_class is not None
        assert torch.equal(result.X_class, batched_data.X_class)
        assert torch.equal(result.E_class, batched_data.E_class)

    def test_forward_with_timestep(
        self, model: IdentityGraphModel, batched_data: GraphData
    ) -> None:
        """forward(data, t) works when t is a normalised timestep tensor."""
        t = torch.tensor([0.5, 0.8])
        result = model(batched_data, t=t)
        assert isinstance(result, GraphData)
        assert result.X_class is not None and batched_data.X_class is not None
        assert torch.equal(result.X_class, batched_data.X_class)

    def test_forward_with_none_timestep_explicit(
        self, model: IdentityGraphModel, batched_data: GraphData
    ) -> None:
        """Explicitly passing t=None produces the same result as omitting it."""
        result_omitted = model(batched_data)
        result_explicit = model(batched_data, t=None)
        assert (
            result_omitted.X_class is not None and result_explicit.X_class is not None
        )
        assert (
            result_omitted.E_class is not None and result_explicit.E_class is not None
        )
        assert torch.equal(result_omitted.X_class, result_explicit.X_class)
        assert torch.equal(result_omitted.E_class, result_explicit.E_class)

    def test_get_config(self, model: IdentityGraphModel) -> None:
        """get_config returns the expected hyperparameter dict."""
        config = model.get_config()
        assert config == {"hidden": 8}

    def test_parameter_count(self, model: IdentityGraphModel) -> None:
        """parameter_count() inherited from BaseModel reports the linear layer."""
        counts = model.parameter_count()
        assert "total" in counts
        assert get_parameter_count_int(counts, "total") > 0
        assert "linear" in counts
        # Linear(8, 8) has 8*8 weight + 8 bias = 72 parameters
        linear_counts = counts["linear"]
        assert isinstance(linear_counts, dict)
        assert get_parameter_count_int(linear_counts, "total") == 72

    def test_single_graph_forward(self, model: IdentityGraphModel) -> None:
        """Forward works with a single (unbatched) graph created from a
        2D adjacency matrix."""
        adj = torch.zeros(3, 3)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0
        data = binary_graphdata(adj)
        result = model(data)
        assert isinstance(result, GraphData)
        assert result.E_class is not None and data.E_class is not None
        assert torch.equal(result.E_class, data.E_class)
