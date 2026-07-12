# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false
"""Tests for DiffusionModule multi-step diffusion training loop.

DiffusionModule extends BaseGraphModule with forward/reverse diffusion
by composing NoiseProcess, Sampler, NoiseSchedule, and GraphEvaluator.
These tests verify that the composition wires correctly and that each
lifecycle hook (training_step, validation_step, on_validation_epoch_end)
behaves as specified.

The tests use a GNN model instantiated directly, paired with a
ContinuousNoiseProcess wrapping GaussianNoise and a
ContinuousSampler, since those require no external dependencies and
work with simple binary-topology graphs lifted into edge-state space.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock, patch

import pytest
import torch
from tests._helpers.graph_builders import (
    binary_graphdata,
    edge_scalar_graphdata,
    legacy_edge_scalar,
)

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphState,
)
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    ExactDensityNoiseProcess,
)
from tmgg.diffusion.sampler import CategoricalSampler, ContinuousSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.models.base import GraphModel

# -----------------------------------------------------------------------
# Shared fixtures and helpers
# -----------------------------------------------------------------------
from tmgg.models.gnn import GNN as _GNN
from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
    _categorical_kl_per_graph,
    _categorical_reconstruction_log_prob,
)
from tmgg.utils.noising.noise import GaussianNoise
from tmgg.utils.noising.size_distribution import SizeDistribution

_MODEL_CONFIG: dict[str, Any] = {
    "num_layers": 2,
    "num_terms": 2,
    "feature_dim_in": 10,
    "feature_dim_out": 10,
}


def _make_default_gnn() -> _GNN:
    """Instantiate the default test GNN model."""
    return _GNN(**_MODEL_CONFIG)


_TIMESTEPS = 10
_NUM_NODES = 6
_BATCH_SIZE = 3


def _make_schedule() -> NoiseSchedule:
    return NoiseSchedule("linear_ddpm", timesteps=_TIMESTEPS)


def _make_noise_process() -> ContinuousNoiseProcess:
    return ContinuousNoiseProcess(GaussianNoise(), schedule=_make_schedule())


def _make_sampler() -> ContinuousSampler:
    return ContinuousSampler()


def _make_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> GraphState:
    """Create a continuous edge-state sparse batch from a binary topology.

    Production datamodules emit sparse ``GraphState`` post the
    2026-05-07 sparse-default refactor; tests that drive
    ``training_step`` / ``validation_step`` directly therefore must
    feed sparse batches.
    """
    adj = torch.zeros(bs, n, n)
    for i in range(bs):
        # Random symmetric adjacency with no self-loops
        upper = (torch.rand(n, n) > 0.5).float()
        sym = upper.triu(diagonal=1)
        adj[i] = sym + sym.t()
    return edge_scalar_graphdata(adj).to_sparse()


def _make_dense_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> DenseGraphState:
    """Dense companion of ``_make_batch`` for tests that operate on dense state."""
    return _make_batch(bs=bs, n=n).to_dense()


def _make_categorical_batch(bs: int = _BATCH_SIZE, n: int = _NUM_NODES) -> GraphState:
    """Create a categorical binary-topology sparse batch for discrete tests."""
    dense = _make_dense_batch(bs=bs, n=n)
    adj = legacy_edge_scalar(dense)
    return binary_graphdata(adj).to_sparse()


def _make_dense_categorical_batch(
    bs: int = _BATCH_SIZE, n: int = _NUM_NODES
) -> DenseGraphState:
    """Dense companion of ``_make_categorical_batch``."""
    dense = _make_dense_batch(bs=bs, n=n)
    adj = legacy_edge_scalar(dense)
    return binary_graphdata(adj)


def _path_graph_data(n: int) -> DenseGraphState:
    """Build a single-graph ``DenseGraphState`` for a path graph on ``n`` nodes.

    Used by on_validation_epoch_end tests as the per-graph reference /
    generated-sample shape per the 2026-05-01 universal-transport
    refactor (was ``nx.path_graph(n)``; the refactor flipped the
    public contract from ``list[nx.Graph]`` to per-graph
    ``DenseGraphState`` carriers with leading batch dim 1).
    """
    e_class = torch.zeros(1, n, n, 2)
    e_class[..., 0] = 1.0  # default: no edge
    for i in range(n - 1):
        e_class[0, i, i + 1, 0] = 0.0
        e_class[0, i, i + 1, 1] = 1.0
        e_class[0, i + 1, i, 0] = 0.0
        e_class[0, i + 1, i, 1] = 1.0
    return DenseGraphState(
        num_nodes_per_graph=torch.tensor([n], dtype=torch.long),
        E_class=e_class,
        y=torch.zeros(1, 0),
    )


def _make_module(
    evaluator: GraphEvaluator | None = None,
    loss_type: str = "mse",
    **overrides: Any,
) -> DiffusionModule:
    """Build a DiffusionModule with sensible test defaults."""
    schedule = _make_schedule()
    noise_process = _make_noise_process()
    sampler = _make_sampler()

    model = overrides.pop("model", _make_default_gnn())
    kwargs: dict[str, Any] = {
        "model": model,
        "noise_process": noise_process,
        "sampler": sampler,
        "noise_schedule": schedule,
        "evaluator": evaluator,
        "loss_type": loss_type,
        "num_nodes": _NUM_NODES,
        "eval_every_n_steps": 1,
    }
    kwargs.update(overrides)
    module = DiffusionModule(**kwargs)
    # Inject a stub size distribution so validation_step doesn't trip the
    # parity #32 fail-loud guard. Production code populates this in
    # DiffusionModule.setup() from the datamodule; unit tests bypass setup.
    module._size_distribution = SizeDistribution(  # pyright: ignore[reportPrivateUsage]
        sizes=(_NUM_NODES,), counts=(1,), max_size=_NUM_NODES
    )
    return module


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestInstantiation:
    """DiffusionModule can be created with all required components."""

    def test_creates_with_all_components(self) -> None:
        """Instantiation succeeds and stores every injected component.

        The module must hold references to the noise process, sampler,
        schedule, and evaluator that were passed at construction time.
        """
        evaluator = GraphEvaluator(eval_num_samples=10)
        module = _make_module(evaluator=evaluator)

        assert isinstance(module.noise_process, ContinuousNoiseProcess)
        assert isinstance(module.sampler, ContinuousSampler)
        assert isinstance(module.noise_schedule, NoiseSchedule)
        assert module.evaluator is evaluator

    def test_creates_without_evaluator(self) -> None:
        """Instantiation works when no evaluator is provided.

        The evaluator is optional; generative evaluation is simply
        skipped when it is None.
        """
        module = _make_module(evaluator=None)
        assert module.evaluator is None

    def test_T_property(self) -> None:
        """The ``T`` property reflects the schedule's timestep count."""
        module = _make_module()
        assert module.T == _TIMESTEPS

    def test_unknown_loss_type_raises(self) -> None:
        """An unrecognised loss_type must raise ValueError immediately.

        We fail fast rather than deferring the error to the first
        training step.
        """
        with pytest.raises(ValueError, match="Unknown loss_type"):
            _make_module(loss_type="huber")

    def test_cross_entropy_criterion(self) -> None:
        """loss_type='cross_entropy' installs CrossEntropyLoss."""
        module = _make_module(loss_type="cross_entropy")
        assert isinstance(module.criterion, torch.nn.CrossEntropyLoss)

    def test_mse_criterion(self) -> None:
        """loss_type='mse' installs MSELoss."""
        module = _make_module(loss_type="mse")
        assert isinstance(module.criterion, torch.nn.MSELoss)

    def test_bce_logits_criterion(self) -> None:
        """loss_type='bce_logits' installs BCEWithLogitsLoss."""
        module = _make_module(loss_type="bce_logits")
        assert isinstance(module.criterion, torch.nn.BCEWithLogitsLoss)


class TestForward:
    """Verify forward() delegates to the underlying model."""

    def test_forward_returns_graph_data(self) -> None:
        """forward() must return GraphData with the same batch shape.

        It is a thin wrapper around ``self.model(data, t=t)`` and the
        GNN model now returns the sparse :class:`GraphDistribution`
        carrier by default; we read the populated sparse edge field
        (``edge_feat`` or ``edge_class``).
        """
        module = _make_module()
        batch = _make_batch(bs=2, n=_NUM_NODES)
        result = module.forward(batch)
        assert isinstance(result, GraphData)
        # Continuous GNN models write into edge_feat (or edge_class
        # depending on edge-source config). Accept whichever sparse
        # field is populated.
        out_edge = (
            result.edge_feat  # pyright: ignore[reportAttributeAccessIssue]
            if getattr(result, "edge_feat", None) is not None
            else result.edge_class  # pyright: ignore[reportAttributeAccessIssue]
        )
        assert out_edge is not None
        # Sparse layout: leading dim is sum_E (per-edge entries).
        assert out_edge.dim() == 2
        # Per-graph node count from the universal base.
        assert int(result.num_nodes_per_graph.shape[0]) == 2

    def test_forward_passes_timestep(self) -> None:
        """forward() relays the ``t`` argument to the model.

        We patch the model's forward to capture the call and confirm
        the timestep tensor arrives unmodified.
        """
        module = _make_module()
        batch = _make_batch(bs=2, n=_NUM_NODES)
        t = torch.tensor([0.5, 0.8])

        original_forward = module.model.forward

        calls: list[tuple[GraphData, torch.Tensor | None]] = []

        def spy(
            data: GraphData,
            t: torch.Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> GraphData:
            calls.append((data, t))
            return original_forward(data, t=t, output_dense=output_dense)

        module.model.forward = spy  # type: ignore[assignment]
        module.forward(batch, t=t)

        assert len(calls) == 1
        passed_t = calls[0][1]
        assert passed_t is not None
        assert torch.allclose(passed_t, t)


class TestTrainingStep:
    """training_step must produce a finite scalar loss with gradient."""

    def test_uses_noise_process_condition_vector_for_model_input(self) -> None:
        """training_step should get model conditioning from the process.

        Test rationale: Step 1 removes module-owned conditioning semantics.
        The module should ask the noise process for the condition vector and
        pass that exact tensor to the model, rather than reconstructing
        ``t / T`` locally.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch()
        fixed_t = torch.tensor([2, 5, 7], dtype=torch.long)
        expected_condition = torch.tensor([0.13, 0.57, 0.91], dtype=torch.float32)

        observed_t: list[torch.Tensor] = []
        observed_condition: list[torch.Tensor | None] = []

        def condition_vector(t: torch.Tensor) -> torch.Tensor:
            observed_t.append(t.detach().clone())
            return expected_condition.to(device=t.device)

        original_forward = module.model.forward

        def spy_forward(
            data: GraphData,
            t: torch.Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> GraphData:
            observed_condition.append(None if t is None else t.detach().clone())
            return original_forward(data, t=t, output_dense=output_dense)

        module.noise_process.process_state_condition_vector = condition_vector  # type: ignore[method-assign]
        module.model.forward = spy_forward  # type: ignore[assignment]

        with patch(
            "tmgg.training.lightning_modules.diffusion_module.torch.randint",
            return_value=fixed_t.clone(),
        ):
            module.training_step(batch, batch_idx=0)

        assert len(observed_t) == 1
        torch.testing.assert_close(observed_t[0], fixed_t)
        assert len(observed_condition) == 1
        assert observed_condition[0] is not None
        torch.testing.assert_close(observed_condition[0], expected_condition)

    def test_loss_is_finite_and_has_grad(self) -> None:
        """A single training step on a synthetic batch must return a
        finite scalar tensor that retains its computation graph (i.e.
        ``requires_grad`` is True). This confirms the forward noise,
        model prediction, and loss computation all compose correctly.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0, "loss must be a scalar"
        assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
        assert loss.requires_grad, "loss must retain grad for backprop"

    def test_mse_loss_is_finite(self) -> None:
        """MSE loss variant also produces a finite scalar.

        MSE compares edge features directly rather than via argmax,
        so this exercises a different code path in _compute_loss.
        """
        module = _make_module(loss_type="mse")
        batch = _make_batch()
        loss = module.training_step(batch, batch_idx=0)

        assert torch.isfinite(loss)
        assert loss.requires_grad


class TestComputeLoss:
    """_compute_loss handles both categorical and continuous modes."""

    def test_cross_entropy_with_one_hot_target(self) -> None:
        """CrossEntropyLoss path: converts one-hot targets to class
        indices and computes edge + node loss. Both must contribute
        to a finite scalar.

        ``_compute_loss`` operates on the dense carrier; we use the
        dense companion fixtures for both prediction and target.
        """
        module = _make_module(loss_type="cross_entropy")
        batch = _make_dense_categorical_batch(bs=2, n=4)
        assert batch.X_class is not None
        assert batch.E_class is not None
        # Simulate model prediction: same shape but with logits
        pred_X = torch.randn_like(batch.X_class)
        pred_E = torch.randn_like(batch.E_class)
        pred = DenseGraphDistribution(
            num_nodes_per_graph=batch.num_nodes_per_graph,
            y=batch.y,
            X_class=pred_X,
            E_class=pred_E,
        )
        loss = module._compute_loss(pred, batch)
        assert torch.isfinite(loss)

    def test_mse_loss_finite(self) -> None:
        """MSE path: compares edge tensors element-wise. The result
        must be finite and non-negative.
        """
        module = _make_module(loss_type="mse")
        batch = _make_dense_batch(bs=2, n=4)
        assert batch.E_feat is not None
        pred_E = torch.randn_like(batch.E_feat)
        # Symmetrise to keep downstream invariants happy.
        pred_E = 0.5 * (pred_E + pred_E.transpose(-3, -2))
        pred = DenseGraphDistribution(
            num_nodes_per_graph=batch.num_nodes_per_graph,
            y=batch.y,
            E_feat=pred_E,
        )
        loss = module._compute_loss(pred, batch)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0


class TestComputeLossYWiring:
    """y_class / y_feat per-field CE wiring (parity #27 / #44 / D-13).

    Verifies that the new graph-level field hooks added in D-13 plug
    into the per-field loss loop without changing SBM behaviour
    (``lambda_y = 0`` ⇒ y-term contributes nothing) and that opting in
    with ``lambda_y > 0`` makes the y-CE term participate. The default
    keeps upstream-parity SBM training bit-for-bit identical.
    """

    def test_lambda_y_zero_preserves_sbm_loss(self) -> None:
        """SBM regression invariant: λ_y = 0 leaves total loss unchanged.

        Default ``lambda_y`` is ``0.0``. Even if we artificially declare
        ``y_class`` as a noise-process field on a structure-only
        categorical SBM batch, the y-term must contribute zero to the
        total, because the per-field weight is zero. This guards
        against an accidental bias toward y-CE on SBM runs.
        """
        module = _make_module(loss_type="cross_entropy")
        batch = _make_dense_categorical_batch(bs=2, n=4)
        assert batch.X_class is not None and batch.E_class is not None

        pred = DenseGraphDistribution(
            num_nodes_per_graph=batch.num_nodes_per_graph,
            y=batch.y,
            X_class=torch.randn_like(batch.X_class),
            E_class=torch.randn_like(batch.E_class),
        )
        # Baseline: SBM-style fields only.
        loss_baseline = module._compute_loss(pred, batch)

        # Pretend the noise process now also targets y_class. With the
        # default ``lambda_y = 0.0`` this MUST not change the total.
        module.noise_process.fields = frozenset(  # pyright: ignore[reportAttributeAccessIssue]
            module.noise_process.fields | {"y_class"}
        )
        loss_with_y_field = module._compute_loss(pred, batch)

        assert torch.allclose(loss_baseline, loss_with_y_field), (
            f"lambda_y=0 must not change loss when y_class is added to "
            f"the field set; got baseline={loss_baseline.item()}, "
            f"with_y={loss_with_y_field.item()}"
        )

    def test_lambda_y_positive_adds_y_ce_term(self) -> None:
        """Opting in: λ_y > 0 makes y-CE participate in total loss.

        Build two modules differing only in ``lambda_y``. With a
        non-degenerate ``y`` carrying a real classification target on
        the batch, declaring ``y_class`` as a noise-process field must
        change the total loss when ``lambda_y > 0``.
        """
        bs, n, dy = 4, 4, 3
        adj = torch.zeros(bs, n, n)
        for i in range(bs):
            upper = (torch.rand(n, n) > 0.5).float()
            sym = upper.triu(diagonal=1)
            adj[i] = sym + sym.t()
        from tests._helpers.graph_builders import binary_graphdata

        batch_no_y = binary_graphdata(adj)
        # Build a real y target: one-hot per graph.
        y_targets = torch.zeros(bs, dy)
        y_targets[torch.arange(bs), torch.tensor([0, 1, 2, 0])] = 1.0
        batch = batch_no_y.replace(y=y_targets)

        # Predicted y logits: kept fixed across the two modules.
        pred_X = torch.randn_like(batch.X_class)  # pyright: ignore[reportArgumentType]
        pred_E = torch.randn_like(batch.E_class)  # pyright: ignore[reportArgumentType]
        pred_y_logits = torch.randn(bs, dy)
        pred = DenseGraphDistribution(
            num_nodes_per_graph=batch.num_nodes_per_graph,
            y=pred_y_logits,
            X_class=pred_X,
            E_class=pred_E,
        )

        module_off = _make_module(loss_type="cross_entropy", lambda_y=0.0)
        module_on = _make_module(loss_type="cross_entropy", lambda_y=2.0)
        for m in (module_off, module_on):
            m.noise_process.fields = frozenset(  # pyright: ignore[reportAttributeAccessIssue]
                m.noise_process.fields | {"y_class"}
            )

        loss_off = module_off._compute_loss(pred, batch)
        loss_on = module_on._compute_loss(pred, batch)

        assert torch.isfinite(loss_off)
        assert torch.isfinite(loss_on)
        # The on-module must include a strictly larger contribution
        # from the y-CE term (random logits ⇒ positive expected CE).
        assert loss_on.item() > loss_off.item(), (
            f"lambda_y > 0 must increase total loss when y-CE > 0; got "
            f"off={loss_off.item()}, on={loss_on.item()}"
        )

    def test_lambda_y_recorded_on_module(self) -> None:
        """Constructor stores ``lambda_y`` on both the scalar and the
        per-field map so downstream callers can introspect either.
        """
        module = _make_module(loss_type="cross_entropy", lambda_y=0.7)
        assert module.lambda_y == 0.7
        assert module.lambda_per_field["y_class"] == 0.7
        assert module.lambda_per_field["y_feat"] == 0.7


class TestValidationStep:
    """validation_step computes loss and VLB but does not accumulate graphs."""

    def test_runs_without_evaluator(self) -> None:
        """When no evaluator is attached, validation_step should
        still run without error.
        """
        module = _make_module(evaluator=None)
        batch = _make_batch()
        # Should not raise
        module.validation_step(batch, batch_idx=0)

    def test_runs_with_evaluator(self) -> None:
        """With an evaluator attached, validation_step should still
        run without error and not accumulate any per-batch state.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator)
        batch = _make_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)
        # No _ref_graphs buffer should exist
        assert not hasattr(module, "_ref_graphs")


class TestOnValidationEpochEnd:
    """on_validation_epoch_end pulls refs from datamodule and evaluates."""

    def test_skips_when_no_evaluator(self) -> None:
        """on_validation_epoch_end is a no-op when evaluator is None.

        Must not raise even if called repeatedly.
        """
        module = _make_module(evaluator=None)
        # Should not raise
        module.on_validation_epoch_end()

    def test_skips_when_not_eval_step(self) -> None:
        """If global_step is not a multiple of eval_every_n_steps,
        the method returns without running generation.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=5)

        mock_trainer = MagicMock()
        mock_trainer.global_step = 2  # not a multiple of 5
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        # Should not raise — skips generation because global_step % 5 != 0
        module.on_validation_epoch_end()

    def test_skips_at_global_step_zero(self) -> None:
        """Regression: the generative evaluator must not run on an
        untrained model. ``on_validation_epoch_end`` fires during
        Lightning's initial sanity check with ``global_step == 0``, and
        any modulo gate like ``0 % N == 0`` lets the untrained-model
        eval path run by default. Metrics computed on random-init
        samples are meaningless, and the per-metric C++ paths (graph-
        tool SBM, ORCA subprocess) are unnecessary work at that point.

        See ``internal notes (not included)``.
        """
        evaluator = GraphEvaluator(eval_num_samples=4)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=1)

        mock_dm = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.global_step = 0
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with patch.object(module, "generate_graphs") as mock_gen:
            module.on_validation_epoch_end()

        mock_dm.get_reference_graphs.assert_not_called()
        mock_gen.assert_not_called()

    def test_calls_get_reference_graphs_on_datamodule(self) -> None:
        """At epoch end past step 0, on_validation_epoch_end should
        call get_reference_graphs on the datamodule to obtain refs.
        """
        evaluator = GraphEvaluator(eval_num_samples=4)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=1)

        mock_dm = MagicMock()
        # Per the 2026-05-01 universal-transport refactor refs are
        # GraphData (was nx.Graph). Visualization is enabled by default
        # in DiffusionModule, so mock returns must support .to_networkx().
        mock_dm.get_reference_graphs.return_value = [
            _path_graph_data(5) for _ in range(4)
        ]

        mock_trainer = MagicMock()
        mock_trainer.global_step = 1  # past the step-0 untrained-model guard
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with patch.object(
            module,
            "generate_graphs",
            return_value=[_path_graph_data(5) for _ in range(4)],
        ):
            module.on_validation_epoch_end()

        mock_dm.get_reference_graphs.assert_called_once_with("val", 4)

    def test_skips_when_too_few_ref_graphs(self) -> None:
        """Generative evaluation needs at least 2 reference graphs for
        MMD. If the datamodule returns fewer, the method returns
        without attempting to generate or evaluate.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator, eval_every_n_steps=1)

        mock_dm = MagicMock()
        mock_dm.get_reference_graphs.return_value = []  # no refs

        mock_trainer = MagicMock()
        mock_trainer.global_step = 1  # past the step-0 untrained-model guard
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        # Should not raise and should not call generate_graphs
        with patch.object(module, "generate_graphs") as mock_gen:
            module.on_validation_epoch_end()
        mock_gen.assert_not_called()

    def test_logs_validation_figures_when_generative_eval_runs(self) -> None:
        """Validation should log figures from the same refs/generated graphs.

        Regression rationale
        --------------------
        The default generative path should emit visual diagnostics whenever it
        already emits generative evaluation metrics. Figure logging must reuse
        the already-sampled graphs instead of triggering a third generation
        pass just for visualization.
        """
        evaluator = GraphEvaluator(eval_num_samples=4)
        module = _make_module(
            evaluator=evaluator,
            eval_every_n_steps=1,
            visualization={"enabled": True, "num_samples": 8},
        )

        # Per the 2026-05-01 universal-transport refactor refs and
        # generated are GraphData; the trainer converts to nx at the
        # leaf via to_networkx() before passing to
        # build_validation_visualizations. The mock_build assert below
        # therefore matches against the converted-nx call args, not
        # the original GraphData.
        refs = [_path_graph_data(5) for _ in range(4)]
        generated = [_path_graph_data(5) for _ in range(4)]
        figures = {
            "gen-val/graph_samples": MagicMock(),
            "gen-val/adjacency_samples": MagicMock(),
        }

        mock_dm = MagicMock()
        mock_dm.get_reference_graphs.return_value = refs

        mock_trainer = MagicMock()
        mock_trainer.global_step = 1  # past the step-0 untrained-model guard
        mock_trainer.datamodule = mock_dm
        mock_trainer.loggers = []
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with (
            patch.object(
                module,
                "generate_graphs",
                side_effect=[generated, [_path_graph_data(5) for _ in range(4)]],
            ) as mock_generate,
            patch.object(
                module.evaluator,
                "evaluate",
                return_value=MagicMock(
                    to_dict=MagicMock(
                        return_value={
                            "degree_mmd": 0.1,
                            "clustering_mmd": 0.2,
                            "spectral_mmd": 0.3,
                            "orbit_mmd": None,
                            "sbm_accuracy": None,
                            "planarity_accuracy": 1.0,
                            "uniqueness": 1.0,
                            "novelty": None,
                        }
                    )
                ),
            ),
            patch(
                "tmgg.training.lightning_modules.diffusion_module.build_validation_visualizations",
                return_value=figures,
            ) as mock_build,
            patch(
                "tmgg.training.lightning_modules.diffusion_module.log_figures"
            ) as mock_log_figures,
        ):
            module.on_validation_epoch_end()

        # The trainer converts GraphData → nx at the leaf right before
        # the build_validation_visualizations call; assert call shape +
        # node count rather than identity, since the converted nx
        # objects are fresh.
        assert mock_build.call_count == 1
        kwargs = mock_build.call_args.kwargs
        assert kwargs["num_samples"] == 8
        assert all(g.number_of_nodes() == 5 for g in kwargs["refs"])
        assert all(g.number_of_nodes() == 5 for g in kwargs["generated"])
        mock_log_figures.assert_called_once_with(
            mock_trainer.loggers,
            figures,
            global_step=1,
        )
        assert mock_generate.call_count == 2

    def test_skips_validation_figure_logging_when_not_eval_step(self) -> None:
        """Figure logging should respect the same eval_every_n_steps gate."""
        evaluator = GraphEvaluator(eval_num_samples=4)
        module = _make_module(
            evaluator=evaluator,
            eval_every_n_steps=5,
            visualization={"enabled": True, "num_samples": 8},
        )

        mock_trainer = MagicMock()
        mock_trainer.global_step = 2
        mock_trainer.datamodule = MagicMock()
        mock_trainer.loggers = []
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with (
            patch(
                "tmgg.training.lightning_modules.diffusion_module.build_validation_visualizations"
            ) as mock_build,
            patch(
                "tmgg.training.lightning_modules.diffusion_module.log_figures"
            ) as mock_log_figures,
        ):
            module.on_validation_epoch_end()

        mock_build.assert_not_called()
        mock_log_figures.assert_not_called()


class TestTestStep:
    """test_step delegates to validation_step."""

    def test_test_step_runs_without_error(self) -> None:
        """test_step delegates to validation_step and should run
        without error, computing loss but not accumulating graphs.
        """
        evaluator = GraphEvaluator(eval_num_samples=100)
        module = _make_module(evaluator=evaluator)
        batch = _make_batch(bs=2)

        # Should not raise
        module.test_step(batch, batch_idx=0)


class TestGenerateGraphs:
    """generate_graphs should use the module-owned sampler inputs."""

    def test_passes_module_noise_process_to_sampler(self) -> None:
        """Generation must pass the module's process instance through.

        Test rationale: Step 2 removes sampler-owned process state. The
        sampler should receive the exact ``module.noise_process`` object so
        setup-time initialization cannot diverge between training and
        generation. Wave 6.1 requires an evaluator so ``generate_graphs``
        can delegate GraphData→nx conversion to ``to_networkx_graphs``.
        """
        module = _make_module(evaluator=GraphEvaluator(eval_num_samples=4))
        observed: dict[str, Any] = {}

        def fake_sample(
            *,
            model: GraphModel,
            noise_process: ContinuousNoiseProcess,
            num_graphs: int,
            num_nodes: int | torch.Tensor,
            device: torch.device,
            collector: Any = None,
            chain_recorder: Any = None,
        ) -> list[GraphData]:
            observed["model"] = model
            observed["noise_process"] = noise_process
            observed["num_graphs"] = num_graphs
            observed["num_nodes"] = num_nodes
            observed["device"] = device
            observed["collector"] = collector
            observed["chain_recorder"] = chain_recorder
            graph = binary_graphdata(torch.zeros(_NUM_NODES, _NUM_NODES))
            return [graph for _ in range(num_graphs)]

        assert module.sampler is not None
        module.sampler.sample = fake_sample  # type: ignore[assignment]

        generated = module.generate_graphs(2)

        assert len(generated) == 2
        assert observed["model"] is module.model
        assert observed["noise_process"] is module.noise_process
        assert observed["num_graphs"] == 2


# -----------------------------------------------------------------------
# VLB wiring tests
# -----------------------------------------------------------------------

_DX = 2  # node classes (matches from_binary_adjacency output)
_DE = 2  # edge classes


class _CategoricalLogitModel(GraphModel):
    """Minimal categorical model stub for DiffusionModule tests.

    Accepts any concrete GraphData carrier and returns a dense
    distribution carrier (``DenseGraphDistribution``). Tests that drive
    this stub through a sparse pipeline get the dense companion
    materialised at the entry point. Honors ``output_dense`` per the
    2026-05-07 sparse-default refactor; production calls into us with
    ``output_dense=True`` from ``DiffusionModule._compute_loss_breakdown``.
    """

    def forward(
        self,
        data: GraphData,
        t: torch.Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> GraphData:
        del t, output_dense
        # Densify if necessary so this stub can deal with sparse inputs
        # produced by the post-2026-05-07 datamodule path.
        if isinstance(data, DenseGraphState | DenseGraphDistribution):
            dense: DenseGraphState | DenseGraphDistribution = data
        else:
            assert isinstance(data, GraphState)
            dense = data.to_dense(
                edge_class_fill=torch.tensor([1.0, 0.0])
                if data.edge_class is not None
                else None,
            )
        assert dense.X_class is not None
        assert dense.E_class is not None
        node_logits = torch.where(
            dense.X_class > 0,
            torch.full_like(dense.X_class, 2.0),
            torch.full_like(dense.X_class, -2.0),
        )
        edge_logits = torch.where(
            dense.E_class > 0,
            torch.full_like(dense.E_class, 2.0),
            torch.full_like(dense.E_class, -2.0),
        )
        return DenseGraphDistribution(
            num_nodes_per_graph=dense.num_nodes_per_graph,
            y=dense.y,
            X_class=node_logits,
            E_class=edge_logits,
        )

    def get_config(self) -> dict[str, Any]:
        return {"type": "categorical_logit_stub"}


def _make_categorical_module(
    limit_distribution: Literal["uniform", "empirical_marginal"] = "uniform",
    *,
    use_marginalised_vlb_kl: bool = False,
    use_upstream_reconstruction: bool = True,
) -> DiffusionModule:
    """Build a DiffusionModule backed by a CategoricalNoiseProcess.

    Uses a categorical-logit stub model so the discrete loss path does not
    depend on a continuous edge-state model.
    """
    schedule = NoiseSchedule("cosine_iddpm", timesteps=_TIMESTEPS)
    cat_np = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=_DX,
        e_classes=_DE,
        limit_distribution=limit_distribution,
    )
    cat_sampler = CategoricalSampler()
    module = DiffusionModule(
        model=_CategoricalLogitModel(),
        noise_process=cat_np,
        sampler=cat_sampler,
        noise_schedule=schedule,
        loss_type="cross_entropy",
        num_nodes=_NUM_NODES,
        eval_every_n_steps=1,
        use_marginalised_vlb_kl=use_marginalised_vlb_kl,
        use_upstream_reconstruction=use_upstream_reconstruction,
    )
    # Inject a stub size distribution so validation_step doesn't trip the
    # parity #32 fail-loud guard. Production code populates this in
    # DiffusionModule.setup() from the datamodule; unit tests bypass setup.
    module._size_distribution = SizeDistribution(  # pyright: ignore[reportPrivateUsage]
        sizes=(_NUM_NODES,), counts=(1,), max_size=_NUM_NODES
    )
    return module


class TestVLBWiring:
    """DiffusionModule computes VLB when noise_process is CategoricalNoiseProcess."""

    def test_categorical_module_uses_exact_density_process(self) -> None:
        """Categorical modules should expose exact-density VLB semantics."""
        module = _make_categorical_module()
        assert isinstance(module.noise_process, ExactDensityNoiseProcess)

    def test_vlb_accumulators_start_empty(self) -> None:
        """All VLB accumulator lists should be empty at construction."""
        module = _make_categorical_module()
        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0
        assert len(module._vlb_kl_diffusion) == 0
        assert len(module._vlb_reconstruction) == 0


class TestCategoricalKLPerGraph:
    """Analytic categorical KL helper used by the Phase D VLB rewrite."""

    def _make_pmf(
        self, prob_x: torch.Tensor, prob_e: torch.Tensor, mask: torch.Tensor
    ) -> DenseGraphDistribution:
        """Build a ``DenseGraphDistribution`` carrying per-position categorical PMFs."""
        return DenseGraphDistribution(
            num_nodes_per_graph=mask.long().sum(dim=-1),
            y=torch.zeros(prob_x.shape[0], 0),
            X_class=prob_x,
            E_class=prob_e,
        )

    def test_kl_zero_for_identical_pmfs(self) -> None:
        """``KL(p || p) == 0`` per graph regardless of mask coverage."""
        bs, n, dx, de = 2, 3, 4, 2
        prob_x = torch.full((bs, n, dx), 1.0 / dx)
        prob_e = torch.full((bs, n, n, de), 1.0 / de)
        mask = torch.ones(bs, n, dtype=torch.bool)

        p = self._make_pmf(prob_x, prob_e, mask)
        q = self._make_pmf(prob_x.clone(), prob_e.clone(), mask)

        kl = _categorical_kl_per_graph(p, q)
        assert torch.allclose(kl, torch.zeros(bs), atol=1e-6)

    def test_kl_strictly_positive_for_disjoint_supports(self) -> None:
        """Putting all mass on different classes drives KL up sharply."""
        bs, n, dx, de = 1, 2, 3, 2
        p_x = torch.zeros(bs, n, dx)
        p_x[..., 0] = 1.0
        q_x = torch.zeros(bs, n, dx)
        q_x[..., 1] = 1.0
        prob_e = torch.full((bs, n, n, de), 1.0 / de)
        mask = torch.ones(bs, n, dtype=torch.bool)

        kl = _categorical_kl_per_graph(
            self._make_pmf(p_x, prob_e, mask),
            self._make_pmf(q_x, prob_e.clone(), mask),
        )
        assert kl.item() > 1.0

    def test_masked_positions_do_not_contribute(self) -> None:
        """Positions outside the node mask must have zero KL contribution."""
        bs, n, dx, de = 1, 4, 2, 2
        # Two graphs that disagree everywhere; only the first two
        # positions are "real" so only those contribute.
        p_x = torch.zeros(bs, n, dx)
        p_x[..., 0] = 1.0
        q_x = torch.zeros(bs, n, dx)
        q_x[..., 1] = 1.0
        prob_e = torch.full((bs, n, n, de), 1.0 / de)
        full_mask = torch.ones(bs, n, dtype=torch.bool)
        partial_mask = torch.tensor([[True, True, False, False]])

        kl_full = _categorical_kl_per_graph(
            self._make_pmf(p_x, prob_e, full_mask),
            self._make_pmf(q_x, prob_e.clone(), full_mask),
        )
        kl_partial = _categorical_kl_per_graph(
            self._make_pmf(p_x, prob_e, partial_mask),
            self._make_pmf(q_x, prob_e.clone(), partial_mask),
        )
        # Half the node positions active → roughly half the X-side KL
        # plus the mask-shrunk edge contribution. Lower-bound is enough.
        assert kl_partial.item() < kl_full.item()
        assert kl_partial.item() > 0.0

    def test_validation_step_populates_vlb_accumulators(self) -> None:
        """After a validation step on a categorical module, each VLB
        accumulator should contain exactly one entry (one per batch).
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        assert len(module._vlb_nll) == 1
        assert len(module._vlb_kl_prior) == 1
        assert len(module._vlb_kl_diffusion) == 1
        assert len(module._vlb_reconstruction) == 1

    def test_vlb_values_finite_with_soft_predictions(self) -> None:
        """All VLB components must be finite when the model outputs proper
        probability distributions (non-zero everywhere). The GNN model
        produces hard one-hot via ``from_adjacency``, which causes NaN in
        log-space. To test finiteness of the VLB pipeline we patch the
        model to return uniform probabilities.
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)

        original_forward = module.model.forward

        def soft_forward(
            data: GraphData,
            t: torch.Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> GraphData:
            del output_dense
            result = original_forward(data, t=t)
            assert result.X_class is not None
            assert result.E_class is not None
            # Convert one-hot output to soft probabilities (uniform + epsilon)
            soft_X = torch.ones_like(result.X_class) / _DX
            soft_E = torch.ones_like(result.E_class) / _DE
            return DenseGraphDistribution(
                num_nodes_per_graph=result.num_nodes_per_graph,
                y=result.y,
                X_class=soft_X,
                E_class=soft_E,
            )

        module.model.forward = soft_forward  # type: ignore[assignment]
        module.validation_step(batch, batch_idx=0)

        assert torch.isfinite(module._vlb_nll[0]), f"NLL is {module._vlb_nll[0]}"
        assert torch.isfinite(module._vlb_kl_prior[0])
        assert torch.isfinite(module._vlb_kl_diffusion[0])
        assert torch.isfinite(module._vlb_reconstruction[0])

    @pytest.mark.skip(
        reason=(
            "Wave 5 per internal notes (not included) "
            "rewires per-field VLB for Gaussian; the 'skip VLB when continuous' "
            "contract this test encodes no longer holds."
        )
    )
    def test_continuous_validation_step_skips_vlb(self) -> None:
        """A continuous-noise module's validation step must not populate
        the VLB accumulators; they should remain empty.
        """
        module = _make_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0

    def test_on_validation_epoch_start_clears_accumulators(self) -> None:
        """``on_validation_epoch_start`` must reset all VLB accumulators."""
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        assert len(module._vlb_nll) == 1
        module.on_validation_epoch_start()
        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0
        assert len(module._vlb_kl_diffusion) == 0
        assert len(module._vlb_reconstruction) == 0

    def test_epoch_end_logs_vlb_and_clears(self) -> None:
        """``on_validation_epoch_end`` should log VLB metrics and clear
        accumulators for a categorical module with accumulated data.
        """
        module = _make_categorical_module()
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        # Mock trainer for current_epoch and log
        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        logged: dict[str, float] = {}
        original_log = module.log

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
            logged[name] = float(value)
            original_log(name, value, **kwargs)

        module.log = capture_log  # type: ignore[assignment]
        module.on_validation_epoch_end()

        assert "val/epoch_NLL" in logged
        assert "val/kl_prior" in logged
        assert "val/kl_diffusion" in logged
        assert "val/reconstruction_logp" in logged

        # Accumulators must be cleared after logging
        assert len(module._vlb_nll) == 0
        assert len(module._vlb_kl_prior) == 0

    @pytest.mark.skip(
        reason=(
            "Wave 5 per internal notes (not included) "
            "rewires per-field VLB for Gaussian; the 'skip VLB when continuous' "
            "contract this test encodes no longer holds."
        )
    )
    def test_epoch_end_no_vlb_when_continuous(self) -> None:
        """A continuous-noise module should not log VLB metrics at all."""
        module = _make_module(evaluator=None)
        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module.validation_step(batch, batch_idx=0)

        logged: dict[str, float] = {}
        original_log = module.log

        def capture_log(name: str, value: Any, **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
            logged[name] = float(value)
            original_log(name, value, **kwargs)

        module.log = capture_log  # type: ignore[assignment]
        module.on_validation_epoch_end()

        assert "val/epoch_NLL" not in logged


class TestCategoricalReconstructionLogProb:
    """Module-local categorical reconstruction helper stays finite and masked."""

    def test_output_shape(self) -> None:
        batch = _make_dense_categorical_batch(bs=_BATCH_SIZE)
        assert batch.X_class is not None
        assert batch.E_class is not None
        X_soft = torch.ones_like(batch.X_class) / _DX
        E_soft = torch.ones_like(batch.E_class) / _DE
        pred_probs = DenseGraphDistribution(
            num_nodes_per_graph=batch.num_nodes_per_graph,
            y=batch.y,
            X_class=X_soft,
            E_class=E_soft,
        )
        result = _categorical_reconstruction_log_prob(
            batch, pred_probs, x_classes=_DX, e_classes=_DE
        )
        assert result.shape == (_BATCH_SIZE,)

    def test_perfect_prediction_near_zero(self) -> None:
        batch = _make_dense_categorical_batch(bs=_BATCH_SIZE)
        assert batch.X_class is not None
        assert batch.E_class is not None
        eps = 1e-7
        x_safe = batch.X_class + eps
        x_safe = x_safe / x_safe.sum(dim=-1, keepdim=True)
        e_safe = batch.E_class + eps
        e_safe = e_safe / e_safe.sum(dim=-1, keepdim=True)
        pred_probs = DenseGraphDistribution(
            num_nodes_per_graph=batch.num_nodes_per_graph,
            y=batch.y,
            X_class=x_safe,
            E_class=e_safe,
        )
        result = _categorical_reconstruction_log_prob(
            batch, pred_probs, x_classes=_DX, e_classes=_DE
        )
        assert torch.isfinite(result).all()
        assert (result.abs() < 0.1).all()


class TestReconstructionMasking:
    """M-6: reconstruction log-prob must mask invalid positions."""

    def test_reconstruction_at_t1_variable_size_batch(self) -> None:
        """Reconstruction log-prob must be finite even with masked-out nodes.

        Test rationale: without masking, invalid positions can produce NaN
        via 0 * log(small) = 0 * -inf = NaN in reconstruction_logp. The fix
        sets pred[invalid] = 1 (so log(1) = 0) and clean[invalid] = 0.
        """
        module = _make_categorical_module()
        # Build a dense batch directly so we can mutate per-position
        # garbage without needing to round-trip through the sparse
        # carrier; ``_compute_reconstruction`` densifies internally
        # anyway. We then rebuild the dense batch with one fewer real
        # node per graph to simulate variable sizes.
        original_dense = _make_dense_categorical_batch(bs=_BATCH_SIZE)
        assert original_dense.X_class is not None
        assert original_dense.E_class is not None
        new_X = original_dense.X_class.clone()
        new_E = original_dense.E_class.clone()
        new_X[:, -1] = torch.rand_like(new_X[:, -1])
        new_E[:, -1, :] = torch.rand_like(new_E[:, -1, :])
        new_E[:, :, -1] = torch.rand_like(new_E[:, :, -1])
        new_counts = (original_dense.num_nodes_per_graph - 1).clamp(min=0).long()
        dense_batch = DenseGraphState(
            num_nodes_per_graph=new_counts,
            y=original_dense.y,
            X_class=new_X,
            E_class=new_E,
        )
        batch = dense_batch.to_sparse()

        # Patch model to return soft probabilities (uniform) so log-space
        # is well-defined at valid positions.
        original_forward = module.model.forward

        def soft_forward(
            data: GraphData,
            t: torch.Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> GraphData:
            del output_dense
            result = original_forward(data, t=t)
            assert result.X_class is not None
            assert result.E_class is not None
            soft_X = torch.ones_like(result.X_class) / _DX
            soft_E = torch.ones_like(result.E_class) / _DE
            return DenseGraphDistribution(
                num_nodes_per_graph=result.num_nodes_per_graph,
                y=result.y,
                X_class=soft_X,
                E_class=soft_E,
            )

        module.model.forward = soft_forward  # type: ignore[assignment]

        result = module._compute_reconstruction(batch)  # pyright: ignore[reportPrivateUsage]
        assert result.shape == (_BATCH_SIZE,)
        assert torch.isfinite(
            result
        ).all(), f"Non-finite reconstruction log-prob: {result}"


class TestUseMarginalisedVlbKlToggle:
    """Cover the D-6 ``use_marginalised_vlb_kl`` toggle on validation_step.

    The toggle selects between the upstream-parity plug-in form (default,
    ``False``) and the lower-variance marginalised form (``True``) for the
    KL_t term in the VLB. Both estimators are unbiased; with one-hot
    ``x_0`` predictions they coincide pointwise. A diffuse soft prediction
    in general yields different per-batch values.
    """

    def _patch_soft_uniform(self, module: DiffusionModule) -> None:
        """Patch the model to emit a uniform soft PMF so the two posterior
        forms are well-defined and (in general) numerically distinct.
        """
        original_forward = module.model.forward

        def soft_forward(
            data: GraphData,
            t: torch.Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> GraphData:
            del output_dense
            result = original_forward(data, t=t)
            assert result.X_class is not None
            assert result.E_class is not None
            soft_X = torch.ones_like(result.X_class) / _DX
            soft_E = torch.ones_like(result.E_class) / _DE
            return DenseGraphDistribution(
                num_nodes_per_graph=result.num_nodes_per_graph,
                y=result.y,
                X_class=soft_X,
                E_class=soft_E,
            )

        module.model.forward = soft_forward  # type: ignore[assignment]

    def test_default_is_upstream_plug_in(self) -> None:
        """The default value of the toggle must be ``False`` (upstream parity)."""
        module = _make_categorical_module()
        assert module.use_marginalised_vlb_kl is False

    def test_both_branches_finite(self) -> None:
        """Both toggle settings produce finite VLB components on a soft batch."""
        torch.manual_seed(0)
        module_off = _make_categorical_module(use_marginalised_vlb_kl=False)
        module_on = _make_categorical_module(use_marginalised_vlb_kl=True)
        self._patch_soft_uniform(module_off)
        self._patch_soft_uniform(module_on)

        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        module_off.validation_step(batch, batch_idx=0)
        module_on.validation_step(batch, batch_idx=0)

        for accumulator in (
            module_off._vlb_kl_diffusion,  # pyright: ignore[reportPrivateUsage]
            module_on._vlb_kl_diffusion,  # pyright: ignore[reportPrivateUsage]
            module_off._vlb_nll,  # pyright: ignore[reportPrivateUsage]
            module_on._vlb_nll,  # pyright: ignore[reportPrivateUsage]
        ):
            assert len(accumulator) == 1
            assert torch.isfinite(accumulator[0]), accumulator[0]

    def test_branches_differ_under_diffuse_prediction(self) -> None:
        """A diffuse soft x_0 prediction yields *different* KL_t values under
        the two forms. Plug-in mixes the soft x_0 once through the Bayes
        formula; marginalised computes a class-by-class mixture of
        posterior PMFs. They agree only at one-hot predictions.
        """
        torch.manual_seed(1)
        module_off = _make_categorical_module(use_marginalised_vlb_kl=False)
        module_on = _make_categorical_module(use_marginalised_vlb_kl=True)
        self._patch_soft_uniform(module_off)
        self._patch_soft_uniform(module_on)

        batch = _make_categorical_batch(bs=_BATCH_SIZE)
        # Use the *same* random t_int draw across both modules so the
        # comparison isolates the toggle behaviour.
        torch.manual_seed(123)
        module_off.validation_step(batch, batch_idx=0)
        torch.manual_seed(123)
        module_on.validation_step(batch, batch_idx=0)

        kl_off = module_off._vlb_kl_diffusion[0]  # pyright: ignore[reportPrivateUsage]
        kl_on = module_on._vlb_kl_diffusion[0]  # pyright: ignore[reportPrivateUsage]
        assert not torch.allclose(kl_off, kl_on, atol=1e-6), (
            f"Plug-in and marginalised KL_t coincided ({kl_off} vs {kl_on}) "
            "even though x_0 prediction is diffuse — the toggle has no effect."
        )


class TestUseUpstreamReconstructionToggle:
    """Cover the D-7 ``use_upstream_reconstruction`` toggle.

    The toggle selects between the upstream-parity z_0 + raw-softmax
    scoring (default ``True``) and the project's previous z_1 +
    marginalised reverse-posterior form (``False``). Both are valid
    estimators of ``log p(x_0)``; the upstream form is the published
    DiGress reconstruction term. They differ numerically by ~1e-3 at
    T=1000.
    """

    def _patch_soft_uniform(self, module: DiffusionModule) -> None:
        """Patch the model to emit a uniform soft PMF (well-defined in
        log-space, distinct from the marginalised reverse-posterior).
        """
        original_forward = module.model.forward

        def soft_forward(
            data: GraphData,
            t: torch.Tensor | None = None,
            *,
            output_dense: bool = False,
        ) -> GraphData:
            del output_dense
            result = original_forward(data, t=t)
            assert result.X_class is not None
            assert result.E_class is not None
            soft_X = torch.ones_like(result.X_class) / _DX
            soft_E = torch.ones_like(result.E_class) / _DE
            return DenseGraphDistribution(
                num_nodes_per_graph=result.num_nodes_per_graph,
                y=result.y,
                X_class=soft_X,
                E_class=soft_E,
            )

        module.model.forward = soft_forward  # type: ignore[assignment]

    def test_default_is_upstream(self) -> None:
        """The toggle defaults to ``True`` (upstream parity)."""
        module = _make_categorical_module()
        assert module.use_upstream_reconstruction is True

    def test_both_branches_finite(self) -> None:
        """Both branches return finite per-graph reconstruction tensors."""
        torch.manual_seed(0)
        module_upstream = _make_categorical_module(use_upstream_reconstruction=True)
        module_marginalised = _make_categorical_module(
            use_upstream_reconstruction=False
        )
        self._patch_soft_uniform(module_upstream)
        self._patch_soft_uniform(module_marginalised)
        batch = _make_categorical_batch(bs=_BATCH_SIZE)

        recon_upstream = module_upstream._compute_reconstruction_upstream_style(  # pyright: ignore[reportPrivateUsage]
            batch
        )
        recon_marginalised = module_marginalised._compute_reconstruction(batch)  # pyright: ignore[reportPrivateUsage]

        assert recon_upstream.shape == (_BATCH_SIZE,)
        assert recon_marginalised.shape == (_BATCH_SIZE,)
        assert torch.isfinite(recon_upstream).all()
        assert torch.isfinite(recon_marginalised).all()

    def test_validation_step_uses_selected_branch(self) -> None:
        """``validation_step`` must dispatch to the upstream-style helper
        when the toggle is True, and to the legacy helper when False.
        """
        torch.manual_seed(1)
        module_upstream = _make_categorical_module(use_upstream_reconstruction=True)
        module_marginalised = _make_categorical_module(
            use_upstream_reconstruction=False
        )
        self._patch_soft_uniform(module_upstream)
        self._patch_soft_uniform(module_marginalised)
        batch = _make_categorical_batch(bs=_BATCH_SIZE)

        upstream_calls: list[str] = []
        marginalised_calls: list[str] = []
        original_upstream = module_upstream._compute_reconstruction_upstream_style  # pyright: ignore[reportPrivateUsage]
        original_legacy = module_marginalised._compute_reconstruction  # pyright: ignore[reportPrivateUsage]

        def trace_upstream(b: GraphData) -> torch.Tensor:
            upstream_calls.append("called")
            return original_upstream(b)

        def trace_legacy(b: GraphData) -> torch.Tensor:
            marginalised_calls.append("called")
            return original_legacy(b)

        module_upstream._compute_reconstruction_upstream_style = trace_upstream  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
        module_marginalised._compute_reconstruction = trace_legacy  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]

        module_upstream.validation_step(batch, batch_idx=0)
        module_marginalised.validation_step(batch, batch_idx=0)

        assert upstream_calls == ["called"], (
            "use_upstream_reconstruction=True should dispatch to "
            "_compute_reconstruction_upstream_style"
        )
        assert marginalised_calls == ["called"], (
            "use_upstream_reconstruction=False should dispatch to "
            "_compute_reconstruction"
        )


class TestSetup:
    """DiffusionModule.setup() initialises categorical stationary PMFs."""

    def test_setup_with_uniform_is_noop(self) -> None:
        """Uniform categorical setup should skip marginal counting."""
        module = _make_categorical_module(limit_distribution="uniform")
        # Clear the fixture-injected stub so setup() takes the
        # populate-from-datamodule path that this test exercises.
        module._size_distribution = None  # pyright: ignore[reportPrivateUsage]
        assert isinstance(module.noise_process, CategoricalNoiseProcess)
        assert module.noise_process._limit_x is not None
        assert module.noise_process._limit_e is not None
        original_x = module.noise_process._limit_x.clone()
        original_e = module.noise_process._limit_e.clone()

        mock_dm = MagicMock()
        mock_dm.get_size_distribution.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]
        module.setup(stage="fit")

        torch.testing.assert_close(module.noise_process._limit_x, original_x)
        torch.testing.assert_close(module.noise_process._limit_e, original_e)
        # The 2026-04-27 triplet invariant (spec §5.6) calls
        # ``train_dataloader`` once to materialise a batch and verify
        # C_x / C_e agreement; that's expected here. The marginal-
        # counting code path (which is what this test was originally
        # asserting against) uses ``train_dataloader_raw_pyg`` instead,
        # so we assert that path is the one that stays untouched.
        mock_dm.train_dataloader_raw_pyg.assert_not_called()
        mock_dm.get_size_distribution.assert_called_once_with("train")

    def test_setup_continuous_is_noop(self) -> None:
        """For a continuous noise process, setup() should be a no-op."""
        module = _make_module()
        mock_trainer = MagicMock()
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]
        # Should not raise
        module.setup(stage="fit")

    def test_setup_marginal_initialises_transition(self) -> None:
        """Empirical-marginal setup should initialise from the raw PyG loader."""
        from torch_geometric.data import Batch, Data

        module = _make_categorical_module(limit_distribution="empirical_marginal")
        # Clear the fixture-injected stub so setup() takes the
        # populate-from-datamodule path that this test exercises.
        module._size_distribution = None  # pyright: ignore[reportPrivateUsage]
        assert isinstance(module.noise_process, CategoricalNoiseProcess)
        assert module.noise_process._limit_x is None
        assert module.noise_process._limit_e is None

        # Sparse PyG batches feed the upstream-parity edge_counts port
        # (parity #13 / D-3); the dense train_dataloader is no longer
        # consulted during noise-process initialisation.
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        pyg_batch = Batch.from_data_list([Data(edge_index=edge_index, num_nodes=2)])
        mock_dm = MagicMock()
        mock_dm.train_dataloader_raw_pyg.return_value = [pyg_batch]

        mock_trainer = MagicMock()
        mock_trainer.datamodule = mock_dm
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        module.setup(stage="fit")
        assert module.noise_process._limit_x is not None
        assert module.noise_process._limit_e is not None
        assert module.noise_process.is_initialized() is True
        mock_dm.get_size_distribution.assert_called_once_with("train")

    def test_setup_marginal_no_dm_raises(self) -> None:
        """setup() must raise RuntimeError if the datamodule is None but
        a marginal transition needs initialising.
        """
        module = _make_categorical_module(limit_distribution="empirical_marginal")

        mock_trainer = MagicMock()
        mock_trainer.datamodule = None
        module._trainer = mock_trainer  # pyright: ignore[reportAttributeAccessIssue]

        with pytest.raises(RuntimeError, match="DataModule required"):
            module.setup(stage="fit")
