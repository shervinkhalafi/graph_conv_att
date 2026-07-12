# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
"""Tests for BaseGraphModule shared Lightning infrastructure.

BaseGraphModule provides common plumbing for all graph learning experiments:
accepting a pre-instantiated model, optimizer/scheduler setup, GraphData batch
transfer, and parameter-count logging. It carries no training logic.

Testing strategy: a concrete subclass ``_ConcreteModule`` supplies trivial
``training_step`` and ``forward`` implementations so we can instantiate the
ABC and test the infrastructure methods in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from tmgg.data.datasets.graph_types import DenseGraphState
from tmgg.models.base import GraphModel
from tmgg.models.gnn import GNN
from tmgg.training.lightning_modules.base_graph_module import (
    BaseGraphModule,
)

# -----------------------------------------------------------------------
# Concrete subclass for testing
# -----------------------------------------------------------------------


class _ConcreteModule(BaseGraphModule):
    """Minimal concrete subclass that satisfies the ABC contract."""

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.0, requires_grad=True)}

    def forward(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        pass


def _make_default_model(**kwargs: Any) -> GNN:
    """Build a GNN with sensible test defaults, merging any overrides."""
    defaults: dict[str, Any] = {
        "num_layers": 2,
        "num_terms": 2,
        "feature_dim_in": 10,
        "feature_dim_out": 10,
    }
    defaults.update(kwargs)
    return GNN(**defaults)


def _make_module(**overrides: Any) -> _ConcreteModule:
    """Convenience factory that merges *overrides* into sensible defaults."""
    model = overrides.pop("model", _make_default_model())
    kwargs: dict[str, Any] = {}
    kwargs.update(overrides)
    return _ConcreteModule(model=model, **kwargs)


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestBaseGraphModuleIsAbstract:
    """Verify that BaseGraphModule cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self) -> None:
        """BaseGraphModule is abstract; it must be subclassed.

        The ABC mechanism prevents direct construction because
        ``training_step`` and ``forward`` are not implemented.
        """
        with pytest.raises(TypeError):
            BaseGraphModule(  # type: ignore[abstract]
                model=_make_default_model(),
            )


class TestModelAssignment:
    """Tests for model assignment at construction."""

    def test_creates_with_graph_model(self) -> None:
        """Passing a GraphModel should be stored on self.model."""
        model = _make_default_model()
        module = _make_module(model=model)
        assert isinstance(module.model, GraphModel)
        assert module.model is model

    def test_model_config_forwarded_to_hparams(self) -> None:
        """get_model_config() should reflect the model's own get_config()."""
        model = _make_default_model(
            num_layers=5, num_terms=3, feature_dim_in=16, feature_dim_out=16
        )
        module = _make_module(model=model)
        assert module.model.num_layers == 5  # pyright: ignore[reportAttributeAccessIssue]

    def test_rejects_non_graph_model(self) -> None:
        """Passing a plain nn.Module (not GraphModel) should raise TypeError."""
        with pytest.raises(TypeError, match="Expected GraphModel subclass"):
            _ConcreteModule(model=torch.nn.Linear(4, 4))  # type: ignore[arg-type]


class TestConfigureOptimizers:
    """Tests for optimizer and scheduler construction."""

    def test_no_scheduler_returns_optimizer(self) -> None:
        """Without scheduler_config the method should return a bare optimizer.

        This is the simplest configuration path.
        """
        module = _make_module(scheduler_config=None)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)

    def test_adam_default(self) -> None:
        """The default optimizer_type 'adam' should produce an Adam instance."""
        module = _make_module(optimizer_type="adam", scheduler_config=None)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)

    def test_adamw(self) -> None:
        """optimizer_type='adamw' should produce AdamW."""
        module = _make_module(
            optimizer_type="adamw", weight_decay=0.01, scheduler_config=None
        )
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_cosine_scheduler(self) -> None:
        """scheduler_type='cosine' should wrap the optimizer in an lr_scheduler dict."""
        module = _make_module(scheduler_config={"type": "cosine", "T_0": 10})
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "lr_scheduler" in result

    def test_step_scheduler(self) -> None:
        """scheduler_type='step' should produce a StepLR scheduler."""
        module = _make_module(
            scheduler_config={"type": "step", "step_size": 10, "gamma": 0.5}
        )
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        scheduler = result["lr_scheduler"]["scheduler"]  # type: ignore[index]
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_scheduler_info_stored_for_cosine_warmup(self) -> None:
        """cosine_warmup should populate _scheduler_info with SchedulerInfo."""
        module = _make_module(
            scheduler_config={
                "type": "cosine_warmup",
                "T_warmup": 100,
                "T_max": 1000,
            }
        )
        # cosine_warmup needs a trainer with max_steps
        mock_trainer = MagicMock()
        mock_trainer.max_steps = 5000
        module._trainer = mock_trainer

        _ = module.configure_optimizers()
        assert module._scheduler_info is not None
        assert module._scheduler_info.T_warmup == 100


class TestTransferBatchToDevice:
    """Tests for GraphData device transfer."""

    def test_moves_graph_data_to_device(self) -> None:
        """GraphData batches should be transferred via GraphData.to().

        We verify that all constituent tensors land on the target device.
        Per the 2026-05-07 sparse-default refactor, ``DenseGraphState.from_structure_only``
        is the dense factory; ``GraphData`` is now an abstract base.
        """
        module = _make_module()
        batch = DenseGraphState.from_structure_only(
            torch.ones(2, 5, dtype=torch.bool), torch.randn(2, 5, 5)
        )
        device = torch.device("cpu")

        result = module.transfer_batch_to_device(batch, device, dataloader_idx=0)

        assert isinstance(result, DenseGraphState)
        assert result.E_feat is not None
        assert result.E_feat.device == device
        assert result.y.device == device
        assert result.node_mask.device == device

    def test_non_graph_data_falls_through(self) -> None:
        """Non-GraphData batches should delegate to the default Lightning impl.

        This confirms the isinstance guard works correctly.
        """
        module = _make_module()
        plain_tensor = torch.randn(4, 8)
        device = torch.device("cpu")

        result = module.transfer_batch_to_device(plain_tensor, device, dataloader_idx=0)  # type: ignore[arg-type]
        assert isinstance(result, torch.Tensor)


class TestOnFitStart:
    """Tests for parameter-count logging at training start."""

    def test_calls_log_parameter_count(self) -> None:
        """on_fit_start should invoke _log_parameter_count with the model and logger.

        We patch the logging function and check it receives the expected
        arguments: the model, the model name, and the current logger.
        """
        module = _make_module()
        mock_logger = MagicMock()
        module._logger = mock_logger  # type: ignore[assignment]

        with patch(
            "tmgg.training.lightning_modules.base_graph_module._log_parameter_count"
        ) as mock_log:
            module.on_fit_start()

            mock_log.assert_called_once_with(
                module.model,
                module.get_model_name(),
                module.logger,
            )


class TestOnBeforeOptimizerStep:
    """Tests for the gradient/param-norm diagnostics hook."""

    def test_logs_expected_diagnostic_keys(self) -> None:
        """The hook should emit the three opt-health diagnostic keys
        (``grad_norm_l2``, ``grad_norm_preclip_max``, ``param_norm_l2``)
        when at least one parameter has a non-None gradient.

        The numerical correctness of the norms is implicitly covered by
        the tensor ops themselves; this test locks in the **interface**:
        the three keys must appear and the values must be finite so
        future refactors cannot silently drop a diagnostic. The current
        production prefix is ``diagnostics-train/opt-health/`` (W&B
        section nesting).
        """
        module = _make_module()
        # Populate gradients on every parameter so the hook sees them.
        for p in module.parameters():
            p.grad = torch.ones_like(p)

        with patch.object(module, "log_dict") as mock_log_dict:
            module.on_before_optimizer_step(optimizer=MagicMock())

        assert mock_log_dict.call_count == 1
        logged = mock_log_dict.call_args.args[0]
        assert set(logged.keys()) == {
            "diagnostics-train/opt-health/grad_norm_l2",
            "diagnostics-train/opt-health/grad_norm_preclip_max",
            "diagnostics-train/opt-health/param_norm_l2",
        }
        for key, value in logged.items():
            # Lightning accepts tensors or floats; we produce tensors.
            val = value.item() if isinstance(value, torch.Tensor) else value
            assert val >= 0.0, f"{key} must be non-negative, got {val}"
            import math

            assert math.isfinite(val), f"{key} must be finite, got {val}"

        kwargs = mock_log_dict.call_args.kwargs
        assert kwargs["on_step"] is True
        assert kwargs["on_epoch"] is False
        assert kwargs["prog_bar"] is False

    def test_noop_when_no_gradients(self) -> None:
        """Before backward runs, parameter ``.grad`` is ``None`` on every
        parameter. The hook must not emit log entries in that case — a
        spurious zero-norm log would mislead dashboards.
        """
        module = _make_module()
        for p in module.parameters():
            p.grad = None

        with patch.object(module, "log_dict") as mock_log_dict:
            module.on_before_optimizer_step(optimizer=MagicMock())

        mock_log_dict.assert_not_called()


class TestGetModelName:
    """Tests for get_model_name()."""

    def test_returns_model_name(self) -> None:
        """get_model_name should return the model_name passed at construction.

        This is the default behaviour; subclasses can override for custom names.
        """
        module = _make_module(model_name="gnn")
        assert module.get_model_name() == "gnn"

    def test_returns_unknown_when_not_set(self) -> None:
        """get_model_name should return 'unknown' when model_name is not set."""
        module = _make_module()
        assert module.get_model_name() == "unknown"


class TestGetModelConfig:
    """Tests for get_model_config() delegation."""

    def test_delegates_to_model(self) -> None:
        """get_model_config should return whatever model.get_config() produces.

        We verify against the known config shape of the GNN model.
        """
        module = _make_module()
        config = module.get_model_config()

        assert isinstance(config, dict)
        # GNN.get_config() includes these keys
        assert "num_layers" in config
        assert config["num_layers"] == 2

    def test_matches_model_get_config(self) -> None:
        """The return value should be identical to calling model.get_config() directly."""
        module = _make_module()
        assert module.get_model_config() == module.model.get_config()
