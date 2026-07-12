"""Tests for SingleStepDenoisingModule.

SingleStepDenoisingModule extends BaseGraphModule for T=1 denoising: sample
a noise level, corrupt, predict clean in one pass. These tests verify
instantiation, training_step, forward bridge, per-noise-level evaluation,
the explicit noise-level contract, and both loss types.

The tests use a GNN model instantiated directly, matching production usage.
Batches are built with the ``binary_graphdata`` helper on random symmetric
binary matrices and then converted to the sparse ``GraphState`` carrier
via ``DenseGraphState.to_sparse()``, which is what
``SingleStepDenoisingModule.training_step`` consumes after the 2026-05-07
sparse-default refactor.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from tests._helpers.graph_builders import binary_graphdata

from tmgg.data.datasets.graph_types import GraphState

# -----------------------------------------------------------------------
# Shared constants and helpers
# -----------------------------------------------------------------------
from tmgg.models.gnn import GNN as _GNN
from tmgg.training.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

_MODEL_CONFIG: dict[str, Any] = {
    "num_layers": 2,
    "num_terms": 2,
    "feature_dim_in": 10,
    "feature_dim_out": 10,
}


def _make_default_model() -> _GNN:
    """Instantiate the default test GNN model."""
    return _GNN(**_MODEL_CONFIG)


_NOISE_LEVELS = [0.1, 0.3, 0.5]
_NUM_NODES = 10
_BATCH_SIZE = 2


def _make_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> GraphState:
    """Create a synthetic sparse ``GraphState`` batch.

    Builds the dense companion via ``binary_graphdata`` then converts to
    the sparse layout that production datamodules now emit.
    """
    adj = torch.zeros(bs, n, n)
    for i in range(bs):
        upper = (torch.rand(n, n) > 0.5).float()
        sym = upper.triu(diagonal=1)
        adj[i] = sym + sym.t()
    return binary_graphdata(adj).to_sparse()


def _make_module(
    loss_type: str = "bce_logits",
    noise_levels: list[float] | None = None,
    eval_noise_levels: list[float] | None = None,
    **overrides: Any,
) -> SingleStepDenoisingModule:
    """Build a SingleStepDenoisingModule with sensible test defaults."""
    model = overrides.pop("model", _make_default_model())
    kwargs: dict[str, Any] = {
        "model": model,
        "noise_type": "digress",
        "noise_levels": noise_levels if noise_levels is not None else _NOISE_LEVELS,
        "eval_noise_levels": eval_noise_levels,
        "loss_type": loss_type,
        "seed": 42,
    }
    kwargs.update(overrides)
    return SingleStepDenoisingModule(**kwargs)


# -----------------------------------------------------------------------
# 1. Instantiation
# -----------------------------------------------------------------------


class TestInstantiation:
    """Verify the module constructs correctly with all parameter variants."""

    def test_default_params(self) -> None:
        """The module should initialise with the specified defaults and
        set the criterion to BCEWithLogitsLoss (via loss_type='bce_logits').
        """
        m = _make_module()
        assert isinstance(m.criterion, nn.BCEWithLogitsLoss)
        assert m.noise_type == "digress"
        assert m.learning_rate == 0.001

    def test_is_diffusion_module_subclass(self) -> None:
        """SingleStepDenoisingModule must be a DiffusionModule subclass,
        per the design doc's requirement that it semantically hardcodes
        T=1 and sampler=None within the DiffusionModule hierarchy.
        """
        from tmgg.training.lightning_modules.diffusion_module import (
            DiffusionModule,
        )

        m = _make_module()
        assert isinstance(m, DiffusionModule)

    def test_mse_criterion(self) -> None:
        """When loss_type='MSE', the criterion should be nn.MSELoss."""
        m = _make_module(loss_type="mse")
        assert isinstance(m.criterion, nn.MSELoss)

    def test_bce_criterion(self) -> None:
        """loss_type='bce_logits' installs BCEWithLogitsLoss."""
        m = _make_module(loss_type="bce_logits")
        assert isinstance(m.criterion, nn.BCEWithLogitsLoss)

    def test_unknown_loss_type_raises(self) -> None:
        """An unrecognised loss_type should raise ValueError immediately."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            _make_module(loss_type="huber")

    def test_drops_unused_diffusion_hparams(self) -> None:
        """The denoising module should not retain generative sampler hparams.

        Test rationale
        --------------
        Real stage-runner smokes enable a local logger. Lightning then merges
        module and datamodule hparams for logging metadata. ``num_nodes`` and
        ``eval_every_n_steps`` belong to multi-step graph generation, not to
        single-step denoising, so keeping them on the module creates false
        conflicts with datamodule-owned values.
        """
        m = _make_module()
        assert "num_nodes" not in m.hparams
        assert "eval_every_n_steps" not in m.hparams
        assert "num_nodes" not in m.hparams_initial
        assert "eval_every_n_steps" not in m.hparams_initial


# -----------------------------------------------------------------------
# 2. training_step
# -----------------------------------------------------------------------


class TestTrainingStep:
    """Verify that training_step produces finite loss with gradients."""

    def test_finite_loss_with_gradient(self) -> None:
        """training_step should return a scalar tensor with a grad_fn
        attached, and its value should be finite.
        """
        m = _make_module()
        batch = _make_batch()
        loss = m.training_step(batch, batch_idx=0)
        assert loss.grad_fn is not None, "loss has no grad_fn"
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    def test_logs_metrics(self) -> None:
        """training_step should call self.log for train/loss,
        train/accuracy, and train/noise_level.

        We patch self.log and inspect the call args to confirm the
        expected metric keys are present.
        """
        m = _make_module()
        batch = _make_batch()

        logged: dict[str, Any] = {}

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = value

        m.log = capture_log  # type: ignore[assignment]
        _ = m.training_step(batch, batch_idx=0)

        assert (
            "train/loss" in logged
        ), f"Missing train/loss. Logged keys: {list(logged)}"
        assert "train/accuracy" in logged
        assert "train/noise_level" in logged

    def test_mse_loss_type_works(self) -> None:
        """training_step should work correctly with MSE loss, producing
        a finite loss with gradient.
        """
        m = _make_module(loss_type="mse")
        batch = _make_batch()
        loss = m.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.grad_fn is not None

    def test_bce_loss_type_works(self) -> None:
        """training_step should work correctly with bce_logits loss,
        producing a finite loss with gradient.
        """
        m = _make_module(loss_type="bce_logits")
        batch = _make_batch()
        loss = m.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss)
        assert loss.grad_fn is not None


# -----------------------------------------------------------------------
# 3. forward
# -----------------------------------------------------------------------


class TestForward:
    """Verify that forward() bridges raw tensors to/from GraphData."""

    def test_tensor_in_tensor_out(self) -> None:
        """forward() should accept a raw (B, N, N) tensor and return a
        raw (B, N, N) tensor of the same batch and spatial dimensions.
        """
        m = _make_module()
        x = torch.rand(_BATCH_SIZE, _NUM_NODES, _NUM_NODES)
        out = m.forward(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (_BATCH_SIZE, _NUM_NODES, _NUM_NODES)

    def test_forward_with_timestep(self) -> None:
        """forward() should accept an optional timestep tensor without
        error, even though single-step models typically ignore it.
        """
        m = _make_module()
        x = torch.rand(_BATCH_SIZE, _NUM_NODES, _NUM_NODES)
        t = torch.zeros(_BATCH_SIZE)
        out = m.forward(x, t=t)
        assert out.shape == (_BATCH_SIZE, _NUM_NODES, _NUM_NODES)


# -----------------------------------------------------------------------
# 4. _val_or_test
# -----------------------------------------------------------------------


class TestValOrTest:
    """Verify per-noise-level evaluation and averaged metric logging."""

    def test_evaluates_all_noise_levels(self) -> None:
        """_val_or_test should log per-noise-level metrics for every
        level in eval_noise_levels (3 levels => 3 per-level loss keys).
        """
        levels = [0.1, 0.3, 0.5]
        m = _make_module(noise_levels=levels, eval_noise_levels=levels)

        logged: dict[str, Any] = {}

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = value

        m.log = capture_log  # type: ignore[assignment]
        batch = _make_batch()
        _ = m._val_or_test("val", batch)

        for eps in levels:
            key = f"val_{eps}/loss"
            assert key in logged, (
                f"Expected per-level key '{key}' not found. "
                f"Logged keys: {sorted(logged)}"
            )

    def test_logs_averaged_metrics(self) -> None:
        """_val_or_test should log noise-level-averaged val/loss and
        val/accuracy alongside the per-level keys.
        """
        m = _make_module()
        logged: dict[str, Any] = {}

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:
            logged[name] = value

        m.log = capture_log  # type: ignore[assignment]
        batch = _make_batch()
        _ = m._val_or_test("val", batch)

        assert "val/loss" in logged
        assert "val/accuracy" in logged

    def test_validation_step_delegates(self) -> None:
        """validation_step should delegate to _val_or_test('val', batch)
        and return a dict.
        """
        m = _make_module()
        m.log = lambda *a, **kw: None  # type: ignore[assignment]
        batch = _make_batch()
        result = m.validation_step(batch, batch_idx=0)
        assert isinstance(result, dict)
        assert "val_loss" in result

    def test_test_step_delegates(self) -> None:
        """test_step should delegate to _val_or_test('test', batch)
        and return a dict with 'test_loss'.
        """
        m = _make_module()
        m.log = lambda *a, **kw: None  # type: ignore[assignment]
        batch = _make_batch()
        result = m.test_step(batch, batch_idx=0)
        assert isinstance(result, dict)
        assert "test_loss" in result


# -----------------------------------------------------------------------
# 5. noise_levels property
# -----------------------------------------------------------------------


class TestNoiseLevelsProperty:
    """Verify the explicit noise-level contract on construction."""

    def test_explicit_noise_levels(self) -> None:
        """When noise_levels are provided at construction, the property
        should return them directly without needing a datamodule.
        """
        levels = [0.2, 0.4]
        m = _make_module(noise_levels=levels)
        assert m.noise_levels == levels

    def test_missing_noise_levels_argument_raises(self) -> None:
        """Omitting noise_levels should fail at construction time."""
        kwargs = cast(
            Any,
            {
                "model": _make_default_model(),
                "seed": 42,
            },
        )
        with pytest.raises(TypeError, match="noise_levels"):
            SingleStepDenoisingModule(**kwargs)

    def test_none_noise_levels_raises(self) -> None:
        """Passing ``None`` should fail loudly instead of falling back."""
        with pytest.raises(ValueError, match="noise_levels"):
            SingleStepDenoisingModule(
                model=_make_default_model(),
                noise_levels=cast(Any, None),
                seed=42,
            )


# -----------------------------------------------------------------------
# 6. eval_noise_levels fallback
# -----------------------------------------------------------------------


class TestEvalNoiseLevelsFallback:
    """Verify eval_noise_levels falls back to noise_levels when unset."""

    def test_explicit_eval_levels(self) -> None:
        """When eval_noise_levels is provided, the property should return
        those levels, independent of noise_levels.
        """
        m = _make_module(
            noise_levels=[0.1, 0.3, 0.5],
            eval_noise_levels=[0.2, 0.4],
        )
        assert m.eval_noise_levels == [0.2, 0.4]

    def test_falls_back_to_noise_levels(self) -> None:
        """When eval_noise_levels is None, the property should return
        the same list as noise_levels.
        """
        m = _make_module(
            noise_levels=[0.1, 0.3, 0.5],
            eval_noise_levels=None,
        )
        assert m.eval_noise_levels == [0.1, 0.3, 0.5]


# -----------------------------------------------------------------------
# 7. Wave 5.2 per-field loss iteration
# -----------------------------------------------------------------------


class TestPerFieldLossIteration:
    """Verify the Wave 5.2 rewrite threads the loss through ``noise_process.fields``.

    Test rationale
    --------------
    The pre-Wave-5.2 loss called ``self.criterion(output, adj)`` directly.
    Wave 5.2 inserts a ``_per_field_edge_loss`` helper that iterates
    ``self.noise_process.fields`` and multiplies by
    ``self.lambda_per_field[field]`` so downstream composition with the
    unified GraphData refactor stays explicit. Invariant: for a default
    denoising module (``E_feat`` Gaussian noise, ``lambda_per_field[E_feat]
    = 1.0``), the per-field loss equals the raw criterion call. Scaling
    the ``E_feat`` weight must then scale the training loss linearly.
    """

    def test_default_fields_is_single_E_feat(self) -> None:
        """The denoising module pins ``noise_process.fields`` to ``{'E_feat'}``.

        Supporting additional fields requires a per-field forward
        bridge that does not yet exist on this module; the constructor
        raises when a caller tries to wire a non-conforming field set.
        """
        m = _make_module()
        assert m.noise_process.fields == frozenset({"E_feat"})

    def test_denoising_default_lambda_per_field_is_unit(self) -> None:
        """Denoising overrides the DiGress ``lambda_E = 5.0`` default.

        Single-step denoising has no multi-step VLB structure; the
        ``lambda_E = 5.0`` convention belongs to generative diffusion.
        The denoising module overrides ``lambda_per_field`` in
        ``__init__`` so the historical loss magnitude (unit weight on
        every field) stays unchanged.
        """
        m = _make_module()
        assert m.lambda_per_field["E_feat"] == 1.0
        assert m.lambda_per_field["E_class"] == 1.0

    def test_per_field_loss_equals_raw_criterion_when_unit_weight(self) -> None:
        """Per-field loss reduces to ``self.criterion(...)`` under unit weight.

        The default denoising configuration keeps every field weight at
        1.0. The per-field loop collapses to a single criterion call so
        the returned scalar must match the legacy direct call byte-for-byte.
        """
        m = _make_module()
        batch = _make_batch()
        adj = batch.dense_adjacency()

        # Fake prediction with finite logits; use the same tensor on
        # both sides so the call is deterministic.
        output = torch.randn_like(adj)
        direct = m.criterion(output, adj)
        looped = m._per_field_edge_loss(output, adj)  # pyright: ignore[reportPrivateUsage]
        assert torch.allclose(direct, looped, atol=1e-6)

    def test_per_field_loss_scales_with_lambda_per_field(self) -> None:
        """Doubling ``lambda_per_field['E_feat']`` doubles the per-field loss.

        The Wave 5.2 helper multiplies each per-field term by its
        weight. With a single declared field, scaling the weight by 2
        must scale the scalar loss by 2 exactly.
        """
        m = _make_module()
        batch = _make_batch()
        adj = batch.dense_adjacency()
        output = torch.randn_like(adj)

        base = m._per_field_edge_loss(output, adj)  # pyright: ignore[reportPrivateUsage]
        m.lambda_per_field["E_feat"] = 2.0
        doubled = m._per_field_edge_loss(output, adj)  # pyright: ignore[reportPrivateUsage]
        assert torch.allclose(doubled, 2.0 * base, atol=1e-6)

    def test_construction_rejects_unsupported_fields(self) -> None:
        """Feeding a non-E_feat GaussianNoiseProcess must fail at construction.

        The module builds its own ``GaussianNoiseProcess`` internally,
        so the invariant check is the only guardrail. Here we patch the
        module constructor to inject a process with an unexpected
        field set and assert the error fires loudly.
        """
        from unittest.mock import patch

        from tmgg.diffusion.noise_process import GaussianNoiseProcess
        from tmgg.diffusion.schedule import NoiseSchedule
        from tmgg.utils.noising.noise import GaussianNoise

        bad_process = GaussianNoiseProcess(
            GaussianNoise(),
            schedule=NoiseSchedule("linear_ddpm", timesteps=1),
            fields=frozenset({"X_feat", "E_feat"}),
        )

        with (
            patch(
                "tmgg.training.lightning_modules.denoising_module.GaussianNoiseProcess",
                return_value=bad_process,
            ),
            pytest.raises(ValueError, match="E_feat"),
        ):
            _make_module()
