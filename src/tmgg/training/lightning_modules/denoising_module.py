"""Single-step denoising module for graph reconstruction experiments.

``SingleStepDenoisingModule`` extends :class:`DiffusionModule` for the T=1
denoising setting: sample a noise level, corrupt the graph, predict the clean
graph in one forward pass. This matches all existing denoising experiments
(spectral, GNN, hybrid, baseline) in the codebase.

Because T=1 denoising does not require a reverse sampler, the constructor
passes ``sampler=None`` to :class:`DiffusionModule`. An optional
``GraphEvaluator`` can be provided for distributional metrics (MMD,
uniqueness, etc.) computed per noise level at each validation round.
The class overrides ``training_step``, ``forward``, ``validation_step``, and
``test_step`` to retain the per-noise-level evaluation logic that the
multi-step parent class does not provide.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportExplicitAny=false
# pyright: reportAny=false
# pyright: reportIncompatibleMethodOverride=false
# PyTorch Lightning's self.log(), self.trainer, self.model, and self.hparams
# have incomplete type stubs. Config dicts legitimately use Any.
# reportIncompatibleMethodOverride: MRO composition with LightningModule and ABC
# causes unavoidable signature mismatches across the class hierarchy.

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from typing import Any, cast, override

import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from loguru import logger as loguru

from tmgg.data.datasets.graph_data_fields import FieldName
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphState,
)
from tmgg.diffusion.noise_process import GaussianNoiseProcess
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.models.base import GraphModel
from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
)
from tmgg.training.logging import log_figure
from tmgg.utils.noising.noise import NoiseDefinition, create_noise_definition

#: Denoising-module-specific default per-field weights. Wave 5.2 threads the
#: parent ``DiffusionModule.lambda_per_field`` into the single-step loss, but
#: the DiGress ``lambda_E = 5.0`` convention belongs to the multi-step
#: diffusion VLB, not to the single-shot denoising setting. Unit weight keeps
#: the historical loss magnitude (pre-Wave-5.2 logs) unchanged so training
#: curves and baselines remain comparable.
_DENOISING_DEFAULT_LAMBDA_PER_FIELD: dict[str, float] = {
    "X_class": 1.0,
    "E_class": 1.0,
    "X_feat": 1.0,
    "E_feat": 1.0,
}

#: The single-step denoising path is pinned to Gaussian scalar-edge diffusion;
#: any other field set would require a different forward/criterion bridge.
_DENOISING_SUPPORTED_FIELDS: frozenset[FieldName] = frozenset({"E_feat"})


def _no_edge_fill_for_denoising(state: GraphState) -> torch.Tensor | None:
    """Return a ``(d_ec,)`` no-edge one-hot fill vector for ``state.edge_class``.

    Mirrors :func:`tmgg.training.lightning_modules.diffusion_module._no_edge_fill_for`;
    duplicated locally so the denoising module can densify a sparse
    ``GraphState`` batch at its entry points without coupling to the
    diffusion module's private helpers. Returns ``None`` when the
    state carries no edge_class (continuous-only configs); the caller
    passes ``None`` to ``GraphState.to_dense`` and the helper short-circuits.
    """
    if state.edge_class is None:
        return None
    d_ec = int(state.edge_class.shape[-1])
    fill = torch.zeros(
        d_ec, dtype=state.edge_class.dtype, device=state.edge_class.device
    )
    fill[0] = 1.0
    return fill


class SingleStepDenoisingModule(DiffusionModule):
    """Single-step denoising module for graph reconstruction experiments.

    Given a noisy graph, predict the clean graph in one forward pass (T=1).
    Noise levels are sampled from a discrete set during training; at
    validation/test time every level is evaluated and per-level metrics
    are logged alongside noise-level-averaged metrics.

    Parameters
    ----------
    model : GraphModel
        Pre-constructed graph model (instantiated by Hydra from nested
        ``_target_`` in the YAML config).
    model_name : str
        Human-readable model identifier for logging and experiment naming.
    learning_rate : float
        Base learning rate.
    weight_decay : float
        Weight decay coefficient.
    optimizer_type : str
        ``"adam"`` or ``"adamw"``.
    amsgrad : bool
        Enable the AMSGrad variant.
    scheduler_config : dict[str, Any] | None
        Optional LR scheduler configuration.
    noise_type : str
        Noise generator type (``"digress"``, ``"gaussian"``, ``"edge_flip"``,
        ``"rotation"``, ``"logit"``).
    noise_levels : list[float]
        Noise levels for training. Required; the module no longer reads them
        from the datamodule at runtime.
    eval_noise_levels : list[float] | None
        Noise levels for evaluation. Falls back to ``noise_levels`` when ``None``.
    loss_type : str
        ``"mse"`` or ``"bce_logits"``.
    rotation_k : int
        Dimension for rotation noise skew matrix (only used when
        ``noise_type="rotation"``).
    seed : int | None
        Random seed for deterministic noise level sampling.
    spectral_k : int
        Number of top eigenvectors for spectral delta metrics.
    log_validation_adjacency_images : bool
        When ``True``, log side-by-side adjacency heatmaps for the original,
        noisy, and denoised validation graphs during ``on_validation_epoch_end``.
    num_validation_adjacency_images : int
        Number of reference validation graphs to visualize per noise level.
    evaluator : GraphEvaluator | None
        When provided, distributional metrics (MMD, uniqueness, etc.) are
        computed per noise level at validation/test epoch end. Reference
        graphs are pulled from the datamodule on-demand; the evaluator's
        ``eval_num_samples`` controls how many are used.
    """

    def __init__(
        self,
        *,
        # BaseGraphModule params
        model: GraphModel,
        model_name: str = "",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        amsgrad: bool = False,
        scheduler_config: dict[str, Any] | None = None,
        # Denoising-specific params
        noise_type: str = "digress",
        noise_levels: list[float],
        eval_noise_levels: list[float] | None = None,
        loss_type: str = "bce_logits",
        rotation_k: int = 20,
        seed: int | None = None,
        # Logging options
        spectral_k: int = 4,
        log_validation_adjacency_images: bool = False,
        num_validation_adjacency_images: int = 4,
        # Distributional evaluation
        evaluator: GraphEvaluator | None = None,
    ) -> None:
        if not noise_levels:
            raise ValueError("noise_levels must be a non-empty list of floats.")
        if num_validation_adjacency_images <= 0:
            raise ValueError(
                "num_validation_adjacency_images must be a positive integer."
            )

        # Build the noise definition and wrap it in a GaussianNoiseProcess
        noise_generator: NoiseDefinition = create_noise_definition(
            noise_type=noise_type, rotation_k=rotation_k, seed=seed
        )
        # T=1 schedule -- the parent's training_step is overridden anyway,
        # but NoiseSchedule is a required DiffusionModule dependency.
        noise_schedule = NoiseSchedule("linear_ddpm", timesteps=1)

        noise_process = GaussianNoiseProcess(noise_generator, noise_schedule)

        super().__init__(
            model=model,
            model_name=model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            scheduler_config=scheduler_config,
            noise_process=noise_process,
            sampler=None,
            noise_schedule=noise_schedule,
            evaluator=evaluator,
            loss_type=loss_type,
            lambda_E=1.0,
            lambda_per_field=_DENOISING_DEFAULT_LAMBDA_PER_FIELD,
        )
        self._drop_unused_diffusion_hparams()

        # Wave 5.2 invariant: the single-step denoising loss iterates
        # ``self.noise_process.fields`` and dispatches the per-field
        # tensor through ``self.criterion``. The current forward bridge
        # is scalar-edge-only, so only ``{"E_feat"}`` is supported.
        # Fail loudly if a future caller wires in a different field set;
        # there is no silent fallback on this path.
        if noise_process.fields != _DENOISING_SUPPORTED_FIELDS:
            raise ValueError(
                "SingleStepDenoisingModule requires a GaussianNoiseProcess "
                f"with fields == {sorted(_DENOISING_SUPPORTED_FIELDS)!r}; "
                f"got {sorted(noise_process.fields)!r}. Supporting additional "
                "fields requires a per-field forward / criterion bridge that "
                "does not yet exist on this module."
            )

        # Denoising-specific state
        self.noise_type: str = noise_type
        self._noise_levels: list[float] = list(noise_levels)
        self._eval_noise_levels: list[float] = (
            list(eval_noise_levels)
            if eval_noise_levels is not None
            else list(noise_levels)
        )

        # Spectral delta logging
        self.spectral_k: int = spectral_k
        self.log_validation_adjacency_images: bool = log_validation_adjacency_images
        self.num_validation_adjacency_images: int = num_validation_adjacency_images

        # Seeded RNG for noise level sampling (avoids global numpy state)
        self._noise_rng: np.random.Generator = np.random.default_rng(seed)

    def _drop_unused_diffusion_hparams(self) -> None:
        """Remove generative diffusion hparams that do not belong to T=1 denoising.

        Lightning merges module and datamodule hparams when a logger is active.
        ``SingleStepDenoisingModule`` inherits ``num_nodes`` and
        ``eval_every_n_steps`` from ``DiffusionModule`` only because the parent
        class saves its constructor kwargs wholesale. Those settings drive the
        multi-step reverse sampler, not single-step reconstruction, so keeping
        them on the denoising module both misstates its public surface and can
        create false merge conflicts with the datamodule.
        """
        self.hparams.pop("num_nodes", None)
        self.hparams.pop("eval_every_n_steps", None)
        initial_hparams = getattr(self, "_hparams_initial", None)
        if isinstance(initial_hparams, MutableMapping):
            initial_hparams.pop("num_nodes", None)
            initial_hparams.pop("eval_every_n_steps", None)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def noise_levels(self) -> list[float]:
        """Noise levels for training."""
        return self._noise_levels

    @property
    def eval_noise_levels(self) -> list[float]:
        """Noise levels for evaluation."""
        return self._eval_noise_levels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @override
    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Bridge dense edge-state tensors to the ``GraphData`` model interface.

        Wraps the input as ``GraphData``, calls the model, and extracts
        the dense edge state for BCE/MSE loss computation.

        Parameters
        ----------
        x
            Input dense edge state, shape ``(B, N, N)``.
        t
            Optional timestep tensor (unused by most single-step models).

        Returns
        -------
        torch.Tensor
            Edge logits, shape ``(B, N, N)``.
        """
        if x.dim() == 2:
            node_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            node_mask = torch.ones(
                x.shape[0], x.shape[1], dtype=torch.bool, device=x.device
            )
        data = DenseGraphState.from_structure_only(node_mask, x)
        # Wave 3 of the sparse-default refactor made ``output_dense`` opt-in;
        # the bridge here needs the dense (B, N, N, *) layout so it can read
        # the scalar adjacency back out via ``to_edge_scalar``. The model
        # returns a ``DenseGraphDistribution`` whose tensors are the raw
        # continuous logits (no thresholding); cast back to the matching
        # ``DenseGraphState`` so the state-only ``to_edge_scalar`` helper
        # can read them. Argmax is *not* appropriate here -- it would
        # discretise continuous E_feat predictions and break the loss.
        result_dist = self.model(data, t=t, output_dense=True)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        assert isinstance(result_dist, DenseGraphDistribution)
        result = DenseGraphState(
            num_nodes_per_graph=result_dist.num_nodes_per_graph,
            y=result_dist.y,
            X_class=result_dist.X_class,
            X_feat=result_dist.X_feat,
            E_class=result_dist.E_class,
            E_feat=result_dist.E_feat,
        )
        return result.to_edge_scalar(
            source="feat" if result.E_feat is not None else "class"
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @override
    def training_step(self, batch: GraphState, batch_idx: int) -> torch.Tensor:
        """Execute a single-step denoising training iteration.

        Samples a random noise level, corrupts the clean graph, and
        iterates over ``self.noise_process.fields`` computing a per-field
        weighted loss through ``self.criterion``. For the current
        scalar-edge configuration the set collapses to ``{"E_feat"}`` so
        the loop runs exactly once, but the iteration keeps the path
        symmetric with :class:`DiffusionModule` and preserves the Wave 5
        invariant that every per-field module reads targets through
        ``noise_process.fields``.

        Parameters
        ----------
        batch
            ``GraphState`` batch from the dataloader.
        batch_idx
            Index of the current batch (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss with gradient.
        """
        # Sparse-default refactor (Phase 5.3): the dataloader emits
        # ``GraphState``; the legacy denoising bridge expects the dense
        # adjacency view (``dense_adjacency`` / ``node_mask`` /
        # ``from_structure_only``), so densify once at the top and thread
        # the dense view through the existing scalar-edge flow.
        batch_dense = batch.to_dense(edge_class_fill=_no_edge_fill_for_denoising(batch))
        adj = batch_dense.dense_adjacency()
        clean_state = DenseGraphState.from_structure_only(batch_dense.node_mask, adj)

        # Sample noise level randomly from training noise levels
        eps: float = float(self._noise_rng.choice(self.noise_levels))

        # Apply noise via unified NoiseProcess interface
        noisy_gd = cast(GaussianNoiseProcess, self.noise_process).sample_at_level(
            clean_state, eps
        )
        # ``GaussianNoiseProcess.sample_at_level`` is typed against the
        # abstract GraphData base but always returns a DenseGraphState
        # via ``DenseGraphState.from_structure_only`` in ``_apply_noise``.
        assert isinstance(noisy_gd, DenseGraphState)

        # Forward pass: returns edge-state logits via GraphData bridge
        output: torch.Tensor = self.forward(noisy_gd.to_edge_scalar(source="feat"))

        loss: torch.Tensor = self._per_field_edge_loss(output, adj)

        # Training accuracy: threshold logits at 0, zero diagonal, compare
        with torch.no_grad():
            predictions: torch.Tensor = (output > 0).float()
            predictions = self._zero_diagonal(predictions)
            train_acc: torch.Tensor = (predictions == adj).float().mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train/noise_level", eps, on_step=False, on_epoch=True)

        return loss

    def _per_field_edge_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-field single-step denoising loss on the scalar-edge view.

        The constructor pins ``self.noise_process.fields`` to
        ``{"E_feat"}``, so the per-field iteration collapses to a single
        call to ``self.criterion(output, target)`` scaled by
        ``self.lambda_per_field["E_feat"]``. Wiring the loss through
        ``noise_process.fields`` and ``lambda_per_field`` keeps the
        module consistent with :class:`DiffusionModule`: the same
        weighting dict controls training loss for both the multi-step
        and single-step paths, so config-level field semantics stay
        aligned. Every additional field would require extending the
        scalar-edge forward bridge before it can carry a meaningful
        loss, which is why the invariant check in ``__init__`` is
        strict.
        """
        # Materialise the field list so the iteration order is stable
        # for test assertions; the set is a one-element frozenset today
        # and sorted() here keeps any future composition deterministic.
        fields = sorted(self.noise_process.fields)
        total: torch.Tensor | None = None
        for field in fields:
            weight = self.lambda_per_field[field]
            # ``E_feat`` is the only supported field; the scalar edge
            # view already feeds both ``output`` and ``target`` so the
            # per-field read is implicit in the caller. Future fields
            # will require routing a different (pred, target) pair
            # here; the invariant at ``__init__`` blocks that path
            # until the extension lands.
            term = self.criterion(output, target)
            contribution = weight * term
            total = contribution if total is None else total + contribution
        if total is None:  # pragma: no cover - invariant enforced at __init__
            raise RuntimeError(
                "SingleStepDenoisingModule._per_field_edge_loss: "
                "noise_process.fields is empty, cannot compute a loss."
            )
        return total

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _val_or_test(self, mode: str, batch: GraphState) -> dict[str, torch.Tensor]:
        """Evaluate across all noise levels, logging per-level and averaged metrics.

        Parameters
        ----------
        mode
            ``"val"`` or ``"test"``.
        batch
            Clean ``GraphState`` batch.

        Returns
        -------
        dict[str, torch.Tensor]
            Noise-level-averaged loss and metrics.
        """
        # Sparse-default refactor (Phase 5.3): densify once at the top so
        # the legacy adjacency-based denoising bridge keeps working.
        batch_dense = batch.to_dense(edge_class_fill=_no_edge_fill_for_denoising(batch))
        adj = batch_dense.dense_adjacency()
        clean_state = DenseGraphState.from_structure_only(batch_dense.node_mask, adj)
        target = adj

        mode_loss_sum = torch.tensor(0.0, device=adj.device)
        metrics_sum: defaultdict[str, torch.Tensor] = defaultdict(
            lambda: torch.tensor(0.0, device=adj.device)
        )
        N: int = len(self.eval_noise_levels)

        for eps in self.eval_noise_levels:
            noisy_gd = cast(GaussianNoiseProcess, self.noise_process).sample_at_level(
                clean_state, eps
            )
            assert isinstance(noisy_gd, DenseGraphState)
            batch_noisy: torch.Tensor = noisy_gd.to_edge_scalar(source="feat")
            output: torch.Tensor = self.forward(batch_noisy)
            mode_loss: torch.Tensor = self._per_field_edge_loss(output, target)

            # Predictions: threshold logits at 0, zero diagonal
            predictions: torch.Tensor = (output > 0).float()
            predictions = self._zero_diagonal(predictions)

            # Per-graph reconstruction metrics (inlined)
            diff = target - predictions
            batch_metrics: dict[str, float] = {
                "mse": (diff**2).mean().item(),
                "frobenius_error": torch.norm(diff, p="fro", dim=(-2, -1))
                .mean()
                .item(),
                "accuracy": (predictions == target).float().mean().item(),
            }

            # Per-noise-level logging
            self.log(
                f"{mode}_{eps}/loss",
                mode_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            for metric_name, value in batch_metrics.items():
                self.log(
                    f"{mode}_{eps}/{metric_name}", value, on_step=False, on_epoch=True
                )

            # Spectral deltas (always computed)
            self._log_spectral_deltas(mode, eps, target, batch_noisy, predictions)

            mode_loss_sum = mode_loss_sum + mode_loss
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum[k] + torch.tensor(v, device=adj.device)

        # Noise-level-averaged metrics
        mode_loss_mean: torch.Tensor = mode_loss_sum / N
        self.log(
            f"{mode}/loss", mode_loss_mean, on_step=False, on_epoch=True, prog_bar=True
        )
        metrics_mean: dict[str, torch.Tensor] = {
            k: v / N for k, v in metrics_sum.items()
        }
        for metric_name, value in metrics_mean.items():
            self.log(f"{mode}/{metric_name}", value, on_step=False, on_epoch=True)

        return {f"{mode}_loss": mode_loss_mean, **metrics_mean}

    @torch.no_grad()
    @override
    def validation_step(
        self, batch: GraphState, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Validation step -- delegates to :meth:`_val_or_test`."""
        return self._val_or_test(mode="val", batch=batch)

    @torch.no_grad()
    @override
    def test_step(self, batch: GraphState, batch_idx: int) -> dict[str, torch.Tensor]:
        """Test step -- delegates to :meth:`_val_or_test`."""
        return self._val_or_test(mode="test", batch=batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_diagonal(A: torch.Tensor) -> torch.Tensor:
        """Zero out diagonal entries of a batched adjacency matrix.

        Parameters
        ----------
        A
            Adjacency matrix, shape ``(B, N, N)`` or ``(N, N)``.

        Returns
        -------
        torch.Tensor
            Copy with diagonal entries set to zero.
        """
        mask = torch.eye(A.shape[-1], device=A.device, dtype=torch.bool)
        if A.ndim == 3:
            mask = mask.unsqueeze(0)
        return A.masked_fill(mask, 0)

    def _log_spectral_deltas(
        self,
        mode: str,
        eps: float,
        A_clean: torch.Tensor,
        A_noisy: torch.Tensor,
        A_denoised: torch.Tensor,
    ) -> None:
        """Log spectral delta metrics between clean, noisy, and denoised graphs.

        Computes four spectral delta metrics for both the noisy-vs-clean and
        denoised-vs-clean comparisons.

        Parameters
        ----------
        mode
            Logging mode (``"val"`` or ``"test"``).
        eps
            Noise level.
        A_clean
            Clean adjacency matrices, shape ``(batch, n, n)``.
        A_noisy
            Noisy adjacency matrices, same shape.
        A_denoised
            Denoised (predicted) adjacency matrices, same shape.
        """
        from tmgg.utils.spectral.spectral_deltas import (
            compute_spectral_deltas,
        )

        with torch.no_grad():
            try:
                deltas_noisy = compute_spectral_deltas(
                    A_clean, A_noisy, k=self.spectral_k
                )
            except Exception as exc:
                loguru.warning(
                    "Skipping spectral delta logging for {mode}_{eps} (noisy): {err}",
                    mode=mode,
                    eps=eps,
                    err=str(exc),
                )
                return
            for name, values in deltas_noisy.items():
                self.log(
                    f"{mode}_{eps}/noisy_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

            try:
                deltas_denoised = compute_spectral_deltas(
                    A_clean, A_denoised, k=self.spectral_k
                )
            except Exception as exc:
                loguru.warning(
                    "Skipping spectral delta logging for {mode}_{eps} (denoised): {err}",
                    mode=mode,
                    eps=eps,
                    err=str(exc),
                )
                return
            for name, values in deltas_denoised.items():
                self.log(
                    f"{mode}_{eps}/denoised_{name}",
                    values.mean(),
                    on_step=False,
                    on_epoch=True,
                )

    def _build_validation_adjacency_figure(
        self,
        *,
        eps: float,
        ref_graphs: list[nx.Graph[Any]],
        clean_adjs: torch.Tensor,
        noisy_adjs: torch.Tensor,
        denoised_adjs: torch.Tensor,
    ) -> matplotlib.figure.Figure:
        """Build a validation figure with original/noisy/denoised adjacency heatmaps."""
        num_graphs = min(self.num_validation_adjacency_images, len(ref_graphs))
        fig, axes = plt.subplots(
            num_graphs, 3, figsize=(9.0, 3.0 * num_graphs), squeeze=False
        )
        column_titles = ("Original", f"Noisy (eps={eps:.3g})", "Denoised")

        for col, title in enumerate(column_titles):
            axes[0, col].set_title(title)

        for row in range(num_graphs):
            num_nodes = ref_graphs[row].number_of_nodes()
            matrices = (
                clean_adjs[row, :num_nodes, :num_nodes],
                noisy_adjs[row, :num_nodes, :num_nodes],
                denoised_adjs[row, :num_nodes, :num_nodes],
            )
            for col, matrix in enumerate(matrices):
                axis = axes[row, col]
                axis.imshow(
                    matrix.detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=1
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if col == 0:
                    axis.set_ylabel(f"graph {row}\nn={num_nodes}")

        fig.tight_layout()
        return fig

    @override
    def on_validation_epoch_end(self) -> None:
        """Evaluate distributional metrics per noise level at epoch end.

        Pulls reference graphs from the datamodule, applies noise at each
        evaluation noise level, runs the model forward to get denoised
        predictions, and evaluates via the GraphEvaluator. Logs per-noise-level
        and noise-level-averaged distributional metrics.
        """
        if self.evaluator is None and not self.log_validation_adjacency_images:
            return

        # SingleStepDenoisingModule's distributional evaluation is
        # GraphEvaluator-only; the molecular path is wired up in a later
        # phase. Fail loudly if a MolecularEvaluator slips in early.
        if self.evaluator is not None and not isinstance(
            self.evaluator, GraphEvaluator
        ):
            raise NotImplementedError(
                "SingleStepDenoisingModule does not yet wire up "
                "MolecularEvaluator; only GraphEvaluator is supported here."
            )

        dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue]
        num_reference_graphs = self.evaluator.eval_num_samples if self.evaluator else 2
        # get_reference_graphs returns list[GraphData] post the
        # 2026-05-01 universal-transport refactor. The denoising
        # module is nx-native (legacy code path); convert at the call
        # site rather than threading GraphData through the rest of
        # this method's adjacency-tensor construction.
        # ``get_reference_graphs`` returns per-graph ``DenseGraphState``
        # carriers with a leading batch dim of 1 (the single-instance
        # convention noted on its docstring), so the batched
        # ``to_networkx`` requires ``batch_index=0``.
        ref_graphs: list[nx.Graph[Any]] = [
            gd.to_networkx(batch_index=0)
            for gd in dm.get_reference_graphs("val", num_reference_graphs)
        ]
        if len(ref_graphs) < 2:
            return

        # Convert reference graphs to a batched adjacency tensor for noising
        n_max = max(g.number_of_nodes() for g in ref_graphs)
        device = next(self.parameters()).device
        ref_adjs = torch.zeros(len(ref_graphs), n_max, n_max, device=device)
        for i, g in enumerate(ref_graphs):
            adj_np = nx.to_numpy_array(g)
            ref_adjs[i, : adj_np.shape[0], : adj_np.shape[1]] = torch.from_numpy(
                adj_np
            ).to(device)

        all_results: dict[float, dict[str, float | None]] = {}

        ref_node_mask = torch.ones(
            ref_adjs.shape[0], ref_adjs.shape[1], dtype=torch.bool, device=device
        )
        ref_gd = DenseGraphState.from_structure_only(ref_node_mask, ref_adjs)

        with torch.no_grad():
            for eps in self.eval_noise_levels:
                noisy_gd = cast(
                    GaussianNoiseProcess, self.noise_process
                ).sample_at_level(ref_gd, eps)
                assert isinstance(noisy_gd, DenseGraphState)
                output = self.forward(noisy_gd.to_edge_scalar(source="feat"))
                predictions = (output > 0).float()
                predictions = self._zero_diagonal(predictions)

                if self.log_validation_adjacency_images:
                    figure = self._build_validation_adjacency_figure(
                        eps=eps,
                        ref_graphs=ref_graphs,
                        clean_adjs=ref_adjs,
                        noisy_adjs=noisy_gd.to_edge_scalar(source="feat"),
                        denoised_adjs=predictions,
                    )
                    log_figure(
                        self.trainer.loggers,  # pyright: ignore[reportAttributeAccessIssue]
                        f"val_{eps}/adjacency_examples",
                        figure,
                        global_step=self.global_step,
                    )

                # Convert predictions to NetworkX graphs
                denoised_graphs: list[nx.Graph[Any]] = []
                for i in range(len(ref_graphs)):
                    ng = ref_graphs[i].number_of_nodes()
                    A_np = predictions[i, :ng, :ng].cpu().numpy()
                    denoised_graphs.append(nx.from_numpy_array(A_np))

                if self.evaluator is None:
                    continue

                # GraphEvaluator.evaluate now declares list[GraphData] but
                # runtime-duck-types nx.Graph too (legacy denoising path).
                results = self.evaluator.evaluate(
                    refs=ref_graphs,  # pyright: ignore[reportArgumentType]
                    generated=denoised_graphs,  # pyright: ignore[reportArgumentType]
                )
                if results is None:
                    continue

                metrics = results.to_dict()
                all_results[eps] = metrics
                for metric_name, value in metrics.items():
                    if value is not None:
                        self.log(
                            f"val_{eps}/{metric_name}",
                            value,
                            on_step=False,
                            on_epoch=True,
                        )

        # Noise-level-averaged distributional metrics
        if all_results:
            metric_names = next(iter(all_results.values())).keys()
            for metric_name in metric_names:
                values: list[float] = [
                    v for r in all_results.values() if (v := r[metric_name]) is not None
                ]
                if values:
                    self.log(
                        f"val/{metric_name}",
                        sum(values) / len(values),
                        on_step=False,
                        on_epoch=True,
                    )
