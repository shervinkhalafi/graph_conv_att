"""Smoke test: every model class can be instantiated with minimal config."""

import pytest

from tmgg.models.base import BaseModel
from tmgg.models.baselines import LinearBaseline, MLPBaseline
from tmgg.models.gnn import GNN, GNNSymmetric, NodeVarGNN
from tmgg.models.spectral_denoisers import GraphFilterBank, LinearPE
from tmgg.models.spectral_denoisers.bilinear import (
    BilinearDenoiser,
    BilinearDenoiserWithMLP,
    MultiLayerBilinearDenoiser,
)
from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (LinearPE, {"k": 4}),
        (GraphFilterBank, {"k": 4, "polynomial_degree": 3}),
        (SelfAttentionDenoiser, {"k": 4, "d_k": 8}),
        (BilinearDenoiser, {"k": 4, "d_k": 8}),
        (
            BilinearDenoiserWithMLP,
            {"k": 4, "d_k": 8, "mlp_hidden_dim": 16, "mlp_num_layers": 1},
        ),
        (
            MultiLayerBilinearDenoiser,
            {"k": 4, "d_model": 8, "num_heads": 2, "num_layers": 1},
        ),
        (
            GNN,
            {
                "num_layers": 1,
                "num_terms": 2,
                "feature_dim_in": 8,
                "feature_dim_out": 4,
            },
        ),
        (
            GNNSymmetric,
            {
                "num_layers": 1,
                "num_terms": 2,
                "feature_dim_in": 8,
                "feature_dim_out": 4,
            },
        ),
        (NodeVarGNN, {"num_layers": 1, "num_terms": 2, "feature_dim": 8}),
        (LinearBaseline, {"max_nodes": 20}),
        (MLPBaseline, {"max_nodes": 20, "hidden_dim": 16, "num_layers": 2}),
    ],
    ids=lambda x: x.__name__ if isinstance(x, type) else "",
)
def test_model_constructible(model_cls: type, kwargs: dict) -> None:
    """Each model class should instantiate and satisfy the BaseModel contract.

    Test rationale: smoke test ensuring all model constructors work with
    minimal valid parameters and expose the required interface (forward,
    get_config).
    """
    model = model_cls(**kwargs)
    assert isinstance(model, BaseModel)
    assert hasattr(model, "forward")
    config = model.get_config()
    assert isinstance(config, dict)
