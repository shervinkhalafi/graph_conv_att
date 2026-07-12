"""Multi-step diffusion training loop for graph generation.

``DiffusionModule`` extends :class:`BaseGraphModule` with a complete
discrete or continuous diffusion pipeline. It composes four injected
components -- a :class:`~tmgg.diffusion.NoiseProcess` (forward diffusion),
a :class:`~tmgg.diffusion.Sampler` (reverse sampling),
a :class:`~tmgg.diffusion.NoiseSchedule` (timestep-to-noise mapping),
and a :class:`~tmgg.training.graph_evaluator.GraphEvaluator`
(generation quality metrics) -- and wires them into Lightning's
``training_step`` / ``validation_step`` / ``on_validation_epoch_end`` hooks.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportExplicitAny=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportIncompatibleMethodOverride=false
# PyTorch Lightning's self.log(), self.trainer, self.model, and self.hparams
# have incomplete type stubs. Config dicts legitimately use Any.
# reportIncompatibleMethodOverride: MRO composition with LightningModule and ABC
# causes unavoidable signature mismatches across the class hierarchy.

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm

from tmgg.data.datasets.graph_data_fields import (
    GRAPHDATA_LOSS_KIND,
    FieldName,
)
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphState,
    state_to_dense_sample,
)
from tmgg.diffusion.collectors import DiffusionLikelihoodCollector, StepMetricCollector
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    ExactDensityNoiseProcess,
    NoiseProcess,
    _read_categorical_e,  # pyright: ignore[reportPrivateUsage]
    _read_categorical_x,  # pyright: ignore[reportPrivateUsage]
)
from tmgg.diffusion.sampler import Sampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.evaluation.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.evaluation.molecular import MolecularEvaluator
from tmgg.evaluation.visualization import (
    build_validation_visualizations,
)
from tmgg.models.base import GraphModel
from tmgg.training.lightning_modules.base_graph_module import (
    BaseGraphModule,
)
from tmgg.training.lightning_modules.train_loss_discrete import (
    TrainLossDiscrete,
    masked_edge_ce,
    masked_edge_mse,
    masked_node_ce,
    masked_node_mse,
    masked_y_ce,
    masked_y_mse,
    per_graph_edge_ce,
    per_graph_edge_mse,
    per_graph_node_ce,
    per_graph_node_mse,
    per_graph_y_ce,
    per_graph_y_mse,
)
from tmgg.training.logging import log_figures
from tmgg.training.losses import masked_ce_loss, masked_mse_loss  # noqa: F401
from tmgg.utils.noising.size_distribution import SizeDistribution

_VALID_LOSS_TYPES = frozenset({"cross_entropy", "mse", "bce_logits"})
_DEFAULT_VISUALIZATION = {"enabled": True, "num_samples": 8}

#: Type alias for the dense carrier union. Both leaves expose the
#: uppercase split fields (``X_class`` / ``X_feat`` / ``E_class`` /
#: ``E_feat``) and the cached ``node_mask`` derived from
#: ``num_nodes_per_graph``; the only difference is content-semantics
#: (state vs. distribution). The training-loop helpers operate on
#: either kind, so we type them against the union rather than the
#: abstract ``GraphData`` base.
DenseGraphData = DenseGraphState | DenseGraphDistribution


def _noise_process_sample(
    proc: NoiseProcess, batch: GraphState, t_int: torch.Tensor
) -> GraphData:
    """Dispatch the sparse forward-sample entry point on a noise process.

    The sparse-default ``sample(z_0, t) -> GraphData`` entry point is
    declared only on leaf processes (``GaussianNoiseProcess`` /
    ``CategoricalNoiseProcess`` and test-side subclasses of
    :class:`NoiseProcess` that mimic the same contract); the abstract
    base and :class:`CompositeNoiseProcess` do not. Looking the method
    up via ``getattr`` would still trip pyright because
    ``nn.Module.__getattr__`` returns ``Module | Tensor`` for unknown
    attributes -- hence the spurious ``Tensor not callable`` warning at
    the call sites. Routing through this helper centralises the
    duck-typed dispatch; an absent ``sample`` raises ``TypeError`` here
    rather than masquerading as a callable.
    """
    sample_fn = getattr(proc, "sample", None)
    if not callable(sample_fn):
        raise TypeError(
            f"_noise_process_sample requires a noise process with a "
            f"callable sparse sample(z_0, t) entry point; "
            f"{type(proc).__name__} has no such attribute."
        )
    result = sample_fn(batch, t_int)
    assert isinstance(result, GraphData)
    return result


#: Default per-field loss weights for the unified per-field training loop.
#: Edge-side fields carry a 5x weight to reproduce the DiGress (Vignac et al.
#: 2023, Eq. 3) ``lambda_E`` convention; node-side and continuous node fields
#: default to unit weight. A user-supplied ``lambda_per_field`` dict on the
#: constructor merges on top of this table.
#:
#: Graph-level ``y_class`` / ``y_feat`` (parity #27 / #44 / D-13) default
#: to ``0.0`` so the y-term contributes nothing on structure-only datasets
#: (SBM, etc.) where no global target exists. Molecular datasets opt in
#: with a non-zero ``lambda_y`` constructor argument.
_DEFAULT_LAMBDA_PER_FIELD: dict[FieldName, float] = {
    "X_class": 1.0,
    "E_class": 5.0,
    "X_feat": 1.0,
    "E_feat": 5.0,
    "y_class": 0.0,
    "y_feat": 0.0,
}


def _no_edge_fill_for(state: GraphState) -> torch.Tensor | None:
    """Return a ``(d_ec,)`` no-edge one-hot fill vector for ``state.edge_class``.

    ``GraphState.to_dense(edge_class_fill=...)`` requires an explicit fill
    vector when ``edge_class`` is populated; channel-0 carries the canonical
    "no edge" class, so the fill is the channel-0 one-hot. Returns ``None``
    when the state carries no edge_class (continuous-only configs); the
    caller passes ``None`` to ``to_dense`` and the helper short-circuits.
    """
    if state.edge_class is None:
        return None
    d_ec = int(state.edge_class.shape[-1])
    fill = torch.zeros(
        d_ec, dtype=state.edge_class.dtype, device=state.edge_class.device
    )
    fill[0] = 1.0
    return fill


def _dense_to_edge_scalar(data: DenseGraphData, *, source: str) -> torch.Tensor:
    """Return a dense ``(B, n_max, n_max)`` scalar adjacency from a dense carrier.

    ``DenseGraphState`` exposes ``to_edge_scalar`` directly. The
    distribution counterpart ``DenseGraphDistribution`` does not (its
    type contract is "distribution-typed content"; ``to_edge_scalar``
    lives only on the state side per ``graph_types.py``). The
    distribution and state carriers however share identical edge tensor
    shapes -- ``E_class: (B, n_max, n_max, d_ec)`` and
    ``E_feat: (B, n_max, n_max, d_ef)`` -- so the channel-zero / squeeze
    reduction is the same operation regardless of carrier semantics.

    Sparse carriers raise: convert at the call site via
    ``state_to_dense_sample`` or request ``output_dense=True`` from the
    model.
    """
    if isinstance(data, DenseGraphState):
        return data.to_edge_scalar(source="feat" if source == "feat" else "class")
    if not isinstance(data, DenseGraphDistribution):
        raise TypeError(
            "_dense_to_edge_scalar() requires a dense carrier "
            "(DenseGraphState or DenseGraphDistribution); "
            f"got {type(data).__name__}. Convert at the call site."
        )
    if source == "feat":
        if data.E_feat is None:
            raise ValueError(
                "_dense_to_edge_scalar(source='feat') requires E_feat to be "
                "populated; got None."
            )
        e = data.E_feat
        edge_scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
    else:
        if data.E_class is None:
            raise ValueError(
                "_dense_to_edge_scalar(source='class') requires E_class to be "
                "populated; got None."
            )
        if data.E_class.shape[-1] > 1:
            edge_scalar = 1.0 - data.E_class[..., 0]
        else:
            edge_scalar = data.E_class[..., 0]
    nm = data.node_mask
    mask2d = nm.unsqueeze(-1) * nm.unsqueeze(-2)
    return edge_scalar * mask2d.to(edge_scalar.dtype)


def _normalize_visualization_config(
    visualization: dict[str, Any] | None,
) -> dict[str, bool | int]:
    """Validate and normalize validation-visualization settings."""
    merged = dict(_DEFAULT_VISUALIZATION)
    if visualization is not None:
        merged.update(visualization)

    enabled = merged["enabled"]
    num_samples = merged["num_samples"]

    if not isinstance(enabled, bool):
        raise TypeError(
            "Visualization config field 'enabled' must be a bool, "
            f"got {type(enabled).__name__}."
        )
    if not isinstance(num_samples, int):
        raise TypeError(
            "Visualization config field 'num_samples' must be an int, "
            f"got {type(num_samples).__name__}."
        )
    if num_samples <= 0 or num_samples % 2 != 0:
        raise ValueError(
            "Visualization config field 'num_samples' must be a positive even integer."
        )

    return {"enabled": enabled, "num_samples": num_samples}


def _read_field(data: DenseGraphData, field: FieldName, x_classes: int) -> torch.Tensor:
    """Read a split field from ``data`` or raise if unpopulated.

    Called from the per-field training / validation loop to fetch the
    target tensor for each field declared by the noise process. For
    ``"E_feat"`` we additionally allow lifting an ``E_class`` view into
    a single-channel scalar edge so Gaussian processes on ``E_feat``
    can still operate on categorical batches.

    ``x_classes`` controls structure-only ``X_class`` synthesis when
    ``data.X_class is None`` — must match the noise process's
    ``x_classes`` so the synthesised target shape is compatible with
    the model output it will be compared against. The synthesis itself
    is delegated to :func:`_read_categorical_x` (which in turn calls
    :meth:`GraphData.synth_structure_only_x_class`); see
    ``internal notes (not included)`` for
    the C_x regime table. ``x_classes`` is required (no default) per
    spec §5.2.
    """
    if field == "X_class":
        # Wave 9.3 (structure-only datasets): structure-only X_class is
        # synthesised from ``node_mask`` via the canonical helper; see
        # ``internal notes (not included)`` for the architecture-internal-concern rationale and
        # the 2026-04-27 spec for the unification.
        return _read_categorical_x(data, x_classes=x_classes)
    if field == "E_class":
        if data.E_class is None:
            raise ValueError(
                "DiffusionModule._read_field: E_class is None but the noise "
                "process declared 'E_class' as a target field."
            )
        return data.E_class
    if field == "X_feat":
        if data.X_feat is None:
            raise ValueError(
                "DiffusionModule._read_field: X_feat is None but the noise "
                "process declared 'X_feat' as a target field."
            )
        return data.X_feat
    if field == "E_feat":
        if data.E_feat is not None:
            return data.E_feat
        if data.E_class is None:
            raise ValueError(
                "DiffusionModule._read_field: neither E_feat nor E_class is "
                "populated; cannot derive a continuous edge view."
            )
        return _dense_to_edge_scalar(data, source="class").unsqueeze(-1)
    if field in ("y_class", "y_feat"):
        # Graph-level fields share the underlying ``y`` tensor on
        # ``GraphData`` (shape ``(bs, dy)``). The split between
        # ``y_class`` (categorical) and ``y_feat`` (continuous) is a
        # loss-dispatch concern carried by ``GRAPHDATA_LOSS_KIND``;
        # the data layer keeps a single tensor for both. Empty-class
        # batches (``y.shape[-1] == 0``, the SBM convention) are
        # handled by the masked CE/MSE helpers' ``numel == 0`` guard.
        return data.y
    raise ValueError(f"_read_field does not support field {field!r}.")


def _resolve_x_classes_for_loss(
    noise_process: NoiseProcess, pred: DenseGraphData, target: DenseGraphData
) -> int:
    """Resolve C_x for the CE-loss path.

    Per 2026-04-27 spec §5.6, a CategoricalNoiseProcess is the
    authoritative C_x source. Tests sometimes drive the CE loss with a
    non-categorical process when the data already carries X_class; in
    that case we fall back to the populated tensor's last dimension on
    either side. Raises if neither path resolves a width.
    """
    if isinstance(noise_process, CategoricalNoiseProcess):
        return noise_process.x_classes
    for candidate in (pred.X_class, target.X_class):
        if candidate is not None:
            return int(candidate.shape[-1])
    raise ValueError(
        "_resolve_x_classes_for_loss: no CategoricalNoiseProcess to source "
        "C_x from and neither pred.X_class nor target.X_class is populated. "
        "See 2026-04-27-x-class-synth-unification-spec §5.6."
    )


def _continuous_target_edge_state(data: DenseGraphData) -> torch.Tensor:
    """Extract the dense target state for continuous losses.

    Continuous predictions are one-channel edge states. For
    categorical targets we lift ``E_class`` into a scalar adjacency so
    the loss compares semantically equivalent dense states.
    """
    if data.E_feat is not None:
        return _dense_to_edge_scalar(data, source="feat")
    return _dense_to_edge_scalar(data, source="class")


def _categorical_reconstruction_log_prob(
    clean: GraphState | DenseGraphData,
    pred_probs: GraphState | DenseGraphData,
    x_classes: int,
    e_classes: int,
) -> torch.Tensor:
    """Return masked categorical reconstruction log-probabilities per graph.

    Per 2026-04-27 spec §5.4: X synthesis on missing-X_class sides is
    delegated to :func:`_read_categorical_x` (canonical helper). The
    callers pass ``x_classes`` / ``e_classes`` from the noise process so
    the synthesised side has compatible shapes with the populated side.
    """
    # ``node_mask`` is a dense-only cached property; sparse callers
    # densify here so the masking that follows operates on the
    # ``(B, n_max, ...)`` layout the categorical helpers below return.
    dense_clean: DenseGraphData = (
        state_to_dense_sample(clean) if isinstance(clean, GraphState) else clean
    )
    dense_pred_probs: DenseGraphData = (
        state_to_dense_sample(pred_probs)
        if isinstance(pred_probs, GraphState)
        else pred_probs
    )
    node_mask = dense_clean.node_mask
    inv = ~node_mask
    inv_edge = inv.unsqueeze(1) | inv.unsqueeze(2)

    clean_x = _read_categorical_x(dense_clean, x_classes=x_classes).clone()
    clean_e = _read_categorical_e(dense_clean, e_classes=e_classes).clone()
    pred_x = _read_categorical_x(dense_pred_probs, x_classes=x_classes).clone()
    pred_e = _read_categorical_e(dense_pred_probs, e_classes=e_classes).clone()

    clean_x[inv] = 0.0
    clean_e[inv_edge] = 0.0
    pred_x[inv] = 1.0
    pred_e[inv_edge] = 1.0

    log_prob_x = (
        (clean_x * pred_x.clamp(min=1e-10).log()).flatten(start_dim=1).sum(dim=1)
    )
    log_prob_e = (
        (clean_e * pred_e.clamp(min=1e-10).log()).flatten(start_dim=1).sum(dim=1)
    )
    return log_prob_x + log_prob_e


def _categorical_kl_per_graph(
    p_pmf: DenseGraphData, q_pmf: DenseGraphData
) -> torch.Tensor:
    """Sum analytic categorical KL(p || q) over all node + edge positions per graph.

    Both inputs carry per-position PMFs over the same support
    (``X_class`` for nodes, ``E_class`` for edges). Padded positions
    are replaced with delta + uniform so they contribute zero to the
    KL, matching the pattern in
    :func:`_categorical_reconstruction_log_prob`.

    Returns ``(bs,)`` per-graph KL totals.
    """
    node_mask = p_pmf.node_mask
    inv_node = ~node_mask
    inv_edge = inv_node.unsqueeze(1) | inv_node.unsqueeze(2)

    if (
        p_pmf.X_class is None
        or p_pmf.E_class is None
        or q_pmf.X_class is None
        or q_pmf.E_class is None
    ):
        raise ValueError(
            "_categorical_kl_per_graph requires X_class and E_class on both PMF inputs."
        )

    p_x = p_pmf.X_class.clone()
    q_x = q_pmf.X_class.clone()
    p_e = p_pmf.E_class.clone()
    q_e = q_pmf.E_class.clone()

    # Zero out p at masked positions (zero probability mass means zero KL
    # contribution); set q to a strictly positive constant there so the
    # downstream log() is well-defined.
    p_x[inv_node] = 0.0
    q_x[inv_node] = 1.0
    p_e[inv_edge] = 0.0
    q_e[inv_edge] = 1.0

    eps = 1e-10
    log_term = p_x.clamp(min=eps).log() - q_x.clamp(min=eps).log()
    kl_x = (p_x * log_term).flatten(start_dim=1).sum(dim=1)
    log_term_e = p_e.clamp(min=eps).log() - q_e.clamp(min=eps).log()
    kl_e = (p_e * log_term_e).flatten(start_dim=1).sum(dim=1)
    return kl_x + kl_e


class DiffusionModule(BaseGraphModule):
    """Multi-step diffusion training loop for graph generation.

    Composes ``NoiseProcess``, ``Sampler``, ``GraphEvaluator``, and
    ``NoiseSchedule`` with a ``GraphModel`` to provide ``training_step``,
    ``validation_step``, and ``test_step`` for discrete or continuous
    diffusion.

    Both this implementation and the original DiGress (Eq. 3, Vignac et al.
    2023) use masked cross-entropy with ``lambda_E``-weighted edge loss for
    training. VLB metrics (Eq. 14) are computed during validation for
    monitoring.

    Parameters
    ----------
    model : GraphModel
        Pre-constructed graph model (instantiated by Hydra from nested
        ``_target_`` in the YAML config).
    model_name : str
        Human-readable identifier for logging and experiment naming.
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
    noise_process : NoiseProcess
        Forward diffusion noise process.
    sampler : Sampler | None
        Reverse diffusion sampler for generation. ``None`` for subclasses
        (e.g. ``SingleStepDenoisingModule``) that do not use reverse sampling.
    noise_schedule : NoiseSchedule
        Timestep-to-noise mapping.
    evaluator : GraphEvaluator | None
        Optional evaluator for generation-quality metrics.
    loss_type : str
        ``"cross_entropy"`` for categorical, ``"mse"`` or ``"bce_logits"``
        for continuous diffusion.
    lambda_E : float
        Weight for the edge loss relative to node loss. Default 5.0 per
        DiGress convention. Used only when ``loss_type="cross_entropy"``.
        Kept for backward compatibility with configs that pass it; when
        set, it overrides ``lambda_per_field["E_class"]`` and
        ``lambda_per_field["E_feat"]``.
    lambda_y : float
        Weight for the graph-level loss (parity #27 / #44 / D-13). Default
        ``0.0`` so structure-only datasets (SBM, etc.) match upstream
        DiGress's behaviour bit-for-bit when no global target is
        defined. Molecular and other graph-classification datasets opt
        in by passing ``lambda_y > 0``; the value overrides
        ``lambda_per_field["y_class"]`` and ``lambda_per_field["y_feat"]``.
    lambda_per_field : dict[str, float] | None
        Per-field weights for the unified per-field training loss (Wave
        5.1). Merged on top of the default table
        ``{X_class: 1.0, E_class: 5.0, X_feat: 1.0, E_feat: 5.0,
        y_class: 0.0, y_feat: 0.0}`` so a caller can override only the
        fields they care about. ``None`` keeps the default table.
    num_nodes : int
        Node count used when sampling graphs during evaluation.
    eval_every_n_steps : int
        Generative evaluation runs when ``global_step`` is a multiple of
        this value. Decouples evaluation frequency from batch size and
        dataset size.
    visualization : dict[str, Any] | None
        Validation-visualization settings. ``enabled`` toggles figure
        logging, and ``num_samples`` sets the total number of plotted
        reference/generated graphs across both distributions.
    use_marginalised_vlb_kl : bool, default False
        When False (upstream parity, default), validation_step's KL_t
        term plugs the soft x_0 prediction directly into the Bayes
        posterior formula (matches diffusion_model_discrete.py:340-360).
        When True, marginalises over discrete x_0 classes:
        ``Σ_c p(z_s | z_t, x_0=c) · p_θ(x_0=c | z_t)``. Both estimators
        are unbiased; marginalised has lower variance early in training
        when ``p_θ`` is diffuse, plug-in is the published DiGress form.
    use_upstream_reconstruction : bool, default True
        When True (upstream parity, default), the reconstruction term in
        the VLB samples ``z_0 = x_0 @ Q_0`` and scores the model's softmax
        against the clean target — the published DiGress "first step of
        denoising" reconstruction likelihood
        (diffusion_model_discrete.py:368-405). When False, samples z_1
        and scores via the reverse-chain marginalised posterior; aligns
        with the sampler's actual first reverse step. Numerical gap is
        bounded at ~1e-3 abs for T=1000.
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
        compile_model: bool = False,
        compile_mode: str = "default",
        # Diffusion-specific params
        noise_process: NoiseProcess,
        sampler: Sampler | None = None,
        noise_schedule: NoiseSchedule,
        evaluator: GraphEvaluator | MolecularEvaluator | None = None,
        loss_type: str = "cross_entropy",
        lambda_E: float = 5.0,
        lambda_y: float = 0.0,
        lambda_per_field: dict[str, float] | None = None,
        num_nodes: int = 20,
        eval_every_n_steps: int = 5000,
        visualization: dict[str, Any] | None = None,
        use_marginalised_vlb_kl: bool = False,
        use_upstream_reconstruction: bool = True,
        n_t_bins: int = 10,
        log_optimizer_health_every_n_steps: int = 500,
    ) -> None:
        super().__init__(
            model=model,
            model_name=model_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            scheduler_config=scheduler_config,
            compile_model=compile_model,
            compile_mode=compile_mode,
        )
        # Re-save hparams at the DiffusionModule level, excluding objects
        # that cannot be reliably pickled for checkpoint reconstruction.
        self.save_hyperparameters(
            ignore=["model", "noise_process", "sampler", "noise_schedule", "evaluator"]
        )

        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type: {loss_type!r}. Expected one of {sorted(_VALID_LOSS_TYPES)}."
            )

        self.noise_process: NoiseProcess = noise_process
        self.sampler: Sampler | None = sampler
        self.noise_schedule: NoiseSchedule = noise_schedule
        self.evaluator: GraphEvaluator | MolecularEvaluator | None = evaluator

        self.num_nodes: int = num_nodes
        self.eval_every_n_steps: int = eval_every_n_steps
        self.visualization: dict[str, bool | int] = _normalize_visualization_config(
            visualization
        )
        self.use_marginalised_vlb_kl: bool = bool(use_marginalised_vlb_kl)
        self.use_upstream_reconstruction: bool = bool(use_upstream_reconstruction)

        if n_t_bins < 1:
            raise ValueError(f"n_t_bins must be a positive integer; got {n_t_bins}.")
        self.n_t_bins: int = int(n_t_bins)

        if log_optimizer_health_every_n_steps < 1:
            raise ValueError(
                "log_optimizer_health_every_n_steps must be a positive integer; "
                f"got {log_optimizer_health_every_n_steps}."
            )
        self.log_optimizer_health_every_n_steps: int = int(
            log_optimizer_health_every_n_steps
        )

        # Per-field loss weights: start from the DiGress-compatible default
        # table, merge any user overrides, then have ``lambda_E`` override
        # the two edge-side fields and ``lambda_y`` the two graph-level
        # fields so existing configs that pass only the scalar knobs keep
        # working bit-for-bit. ``lambda_y`` defaults to ``0.0`` so SBM
        # behaviour (no global target) is unchanged.
        merged_lambdas: dict[FieldName, float] = dict(_DEFAULT_LAMBDA_PER_FIELD)
        if lambda_per_field is not None:
            for field_name, weight in lambda_per_field.items():
                if field_name not in _DEFAULT_LAMBDA_PER_FIELD:
                    raise ValueError(
                        f"lambda_per_field key {field_name!r} is not a known "
                        f"FieldName; expected one of "
                        f"{sorted(_DEFAULT_LAMBDA_PER_FIELD)}."
                    )
                merged_lambdas[field_name] = float(weight)  # pyright: ignore[reportArgumentType]
        merged_lambdas["E_class"] = float(lambda_E)
        merged_lambdas["E_feat"] = float(lambda_E)
        merged_lambdas["y_class"] = float(lambda_y)
        merged_lambdas["y_feat"] = float(lambda_y)
        self.lambda_per_field: dict[FieldName, float] = merged_lambdas
        self.lambda_E: float = float(lambda_E)
        self.lambda_y: float = float(lambda_y)

        self.criterion: nn.Module
        self._train_loss_discrete: TrainLossDiscrete | None = None
        if loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
            self._train_loss_discrete = TrainLossDiscrete(lambda_E=lambda_E)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:  # bce_logits
            self.criterion = nn.BCEWithLogitsLoss()

        # Diagnostics: wall-clock timer for ``train/step_time_s``. Set in
        # ``on_train_batch_start`` and read in ``on_train_batch_end``;
        # initialized here so the attribute always exists before the first
        # training step.
        self._train_batch_start_time: float = 0.0

        # Diagnostics: current-batch batch_size for self.log calls.
        # Lightning can't unambiguously infer batch_size from GraphData
        # (multiple tensors with different leading dims) and warns on
        # every self.log invocation. Stashed at training_step /
        # validation_step / on_train_batch_start entry; threaded to all
        # per-step / hook log calls via batch_size=self._cur_bs.
        self._cur_bs: int = 0

        # VLB accumulators (only used when the process has exact densities)
        self._vlb_nll: list[torch.Tensor] = []
        self._vlb_kl_prior: list[torch.Tensor] = []
        self._vlb_kl_diffusion: list[torch.Tensor] = []
        self._vlb_reconstruction: list[torch.Tensor] = []
        self._vlb_log_pn: list[torch.Tensor] = []
        # Bits-per-edge accumulators: stored as (sum_nll, sum_edges,
        # sum_potential_edges) so the epoch-end normalisation is the
        # exact total-NLL / total-edges ratio rather than the noisier
        # mean-of-ratios. ``_vlb_nll_sum`` carries each batch's NLL × bs
        # so the eventual sum equals total NLL across the val set.
        self._vlb_nll_sum: list[torch.Tensor] = []
        self._vlb_edge_count: list[torch.Tensor] = []
        self._vlb_potential_count: list[torch.Tensor] = []
        self._size_distribution: SizeDistribution | None = None

        # Per-timestep bin accumulators (Stage 1 telemetry). Buffers stay
        # on the module device; they do NOT participate in autograd. The
        # train pair logs every ``eval_every_n_steps`` (then resets);
        # the four val pairs log on each ``on_validation_epoch_end``.
        # Each ``register_buffer`` is followed by an explicit type
        # annotation: ``LightningModule.register_buffer`` is typed as
        # storing ``Tensor | Module``, but in our use the values are
        # always plain tensors and this re-statement lets pyright check
        # ``.zero_()`` / ``.scatter_add_()`` etc. without false positives.
        self.register_buffer(
            "_train_loss_t_sum", torch.zeros(self.n_t_bins), persistent=False
        )
        self._train_loss_t_sum: torch.Tensor
        self.register_buffer(
            "_train_loss_t_count",
            torch.zeros(self.n_t_bins, dtype=torch.long),
            persistent=False,
        )
        self._train_loss_t_count: torch.Tensor
        self.register_buffer(
            "_val_loss_t_sum", torch.zeros(self.n_t_bins), persistent=False
        )
        self._val_loss_t_sum: torch.Tensor
        self.register_buffer(
            "_val_loss_t_count",
            torch.zeros(self.n_t_bins, dtype=torch.long),
            persistent=False,
        )
        self._val_loss_t_count: torch.Tensor
        self.register_buffer(
            "_val_kl_diff_t_sum", torch.zeros(self.n_t_bins), persistent=False
        )
        self._val_kl_diff_t_sum: torch.Tensor
        self.register_buffer(
            "_val_kl_diff_t_count",
            torch.zeros(self.n_t_bins, dtype=torch.long),
            persistent=False,
        )
        self._val_kl_diff_t_count: torch.Tensor
        self.register_buffer(
            "_val_kl_prior_t_sum", torch.zeros(self.n_t_bins), persistent=False
        )
        self._val_kl_prior_t_sum: torch.Tensor
        self.register_buffer(
            "_val_kl_prior_t_count",
            torch.zeros(self.n_t_bins, dtype=torch.long),
            persistent=False,
        )
        self._val_kl_prior_t_count: torch.Tensor
        self.register_buffer(
            "_val_recon_t_sum", torch.zeros(self.n_t_bins), persistent=False
        )
        self._val_recon_t_sum: torch.Tensor
        self.register_buffer(
            "_val_recon_t_count",
            torch.zeros(self.n_t_bins, dtype=torch.long),
            persistent=False,
        )
        self._val_recon_t_count: torch.Tensor

        # Optimizer-health state (Stage 2 telemetry). Allocated lazily on
        # the every-Nth-step cadence and freed in the trailing
        # ``on_train_batch_end`` hook so memory peak ≈ 2× params only on
        # logging steps. ``_prev_weights_snapshot`` doubles as the marker
        # that ``on_train_batch_end`` should compute Δθ this iteration —
        # the math needs θ_before against post-optimizer weights. The
        # cadence-gated diagnostics themselves are logged via
        # ``write_through=True`` (see ``WriteThroughLogMixin``), so no
        # cross-hook payload stashing is needed for delivery anymore.
        self._prev_grad_by_block: dict[str, torch.Tensor] | None = None
        self._prev_weights_snapshot: dict[str, torch.Tensor] | None = None

    @property
    def T(self) -> int:
        """Number of diffusion steps."""
        return self.noise_process.timesteps

    @override
    def setup(self, stage: str | None = None) -> None:
        """Initialise process state and size metadata from the datamodule."""
        dm = self.trainer.datamodule  # pyright: ignore[reportAttributeAccessIssue]

        if self.noise_process.needs_data_initialization():
            if self.noise_process.is_initialized():
                pass
            elif dm is None:
                raise RuntimeError(
                    "DataModule required for noise-process initialisation"
                )
            else:
                self.noise_process.initialize_from_data(
                    dm.train_dataloader_raw_pyg()  # pyright: ignore[reportUnknownMemberType]
                )

        if self._size_distribution is None and isinstance(
            self.noise_process, ExactDensityNoiseProcess
        ):
            if dm is None:
                raise RuntimeError(
                    "DataModule required for exact-density size distribution"
                )
            self._size_distribution = dm.get_size_distribution("train")

        # Init-time invariant (2026-04-27 spec §5.6): assert that
        # data.X_class.shape[-1], noise_process.x_classes, and
        # model.output_dims_x_class agree. Same for C_e. Catches
        # Hydra-config drift with a clear error message before the first
        # training step (replaces a CUDA bmm shape error 5 layers deep).
        # Only meaningful for categorical noise processes; continuous
        # processes don't carry x_classes / e_classes.
        if isinstance(self.noise_process, CategoricalNoiseProcess) and dm is not None:
            try:
                batch = next(iter(dm.train_dataloader()))  # pyright: ignore[reportUnknownMemberType]
            except (StopIteration, AttributeError):
                batch = None
            if batch is not None:
                materialised_x = _read_categorical_x(
                    batch, x_classes=self.noise_process.x_classes
                )
                materialised_e = _read_categorical_e(
                    batch, e_classes=self.noise_process.e_classes
                )
                assert materialised_x.shape[-1] == self.noise_process.x_classes, (
                    f"Materialised X_class width {materialised_x.shape[-1]} "
                    f"disagrees with noise_process.x_classes="
                    f"{self.noise_process.x_classes}."
                )
                assert materialised_e.shape[-1] == self.noise_process.e_classes, (
                    f"Materialised E_class width {materialised_e.shape[-1]} "
                    f"disagrees with noise_process.e_classes="
                    f"{self.noise_process.e_classes}."
                )

            model_c_x = getattr(self.model, "output_dims_x_class", None)
            if model_c_x is not None:
                assert model_c_x == self.noise_process.x_classes, (
                    f"Model output_dims_x_class={model_c_x} disagrees with "
                    f"noise_process.x_classes={self.noise_process.x_classes} "
                    f"(canonical name C_x). Configure both via the same "
                    f"Hydra source. See 2026-04-27 spec §5.6."
                )
            model_c_e = getattr(self.model, "output_dims_e_class", None)
            if model_c_e is not None:
                assert model_c_e == self.noise_process.e_classes, (
                    f"Model output_dims_e_class={model_c_e} disagrees with "
                    f"noise_process.e_classes={self.noise_process.e_classes} "
                    f"(canonical name C_e)."
                )

    @override
    def on_validation_epoch_start(self) -> None:
        """Clear VLB accumulators at the start of each validation epoch."""
        self._vlb_nll.clear()
        self._vlb_kl_prior.clear()
        self._vlb_kl_diffusion.clear()
        self._vlb_reconstruction.clear()
        self._vlb_log_pn.clear()
        self._vlb_nll_sum.clear()
        self._vlb_edge_count.clear()
        self._vlb_potential_count.clear()
        self._val_loss_t_sum.zero_()
        self._val_loss_t_count.zero_()
        self._val_kl_diff_t_sum.zero_()
        self._val_kl_diff_t_count.zero_()
        self._val_kl_prior_t_sum.zero_()
        self._val_kl_prior_t_count.zero_()
        self._val_recon_t_sum.zero_()
        self._val_recon_t_count.zero_()

    def _t_bin_idx(self, t_int: torch.Tensor) -> torch.Tensor:
        """Map per-graph timesteps ``t ∈ [0, T]`` to bin indices ``[0, n_t_bins)``.

        Detached integer tensor; safe for ``scatter_add_`` indexing
        regardless of training/eval mode.
        """
        # T+1 distinct timesteps {0,...,T}; bin width = (T+1) / n_t_bins.
        bin_idx = (t_int.detach() * self.n_t_bins) // (self.T + 1)
        return bin_idx.clamp_(max=self.n_t_bins - 1).long()

    def _per_graph_loss(
        self, pred: DenseGraphData, target: DenseGraphState
    ) -> torch.Tensor:
        """Per-graph total loss matching ``_compute_loss`` semantics.

        Returns ``(bs,)`` so the per-step / per-validation scatter into
        ``t``-bins has graph-level resolution. Mirrors
        :meth:`_compute_loss_breakdown` but uses the per-graph helpers
        in ``train_loss_discrete``: same valid-row predicate, same
        ``lambda_per_field`` weighting, same field iteration order.
        Result is ``.detach()``-ed at the call site (the binning path
        is purely diagnostic and must not contribute gradients).
        """
        bs = int(target.node_mask.shape[0])
        device = target.node_mask.device

        if self._train_loss_discrete is None:
            # Continuous path: single ``edge_state`` field, dense MSE/BCE.
            if pred.E_feat is not None:
                pred_edge_state = _dense_to_edge_scalar(pred, source="feat")
            else:
                pred_edge_state = _dense_to_edge_scalar(pred, source="class")
            target_edge_state = _continuous_target_edge_state(target).float()
            # Dense per-position squared error → per-graph mean over the
            # masked off-diagonal positions. Matches ``masked_edge_mse``
            # reduction granularity but at per-graph resolution.
            diff_sq = (pred_edge_state - target_edge_state) ** 2
            node_mask = target.node_mask
            n = node_mask.size(1)
            diag_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
            edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)) & diag_mask
            edge_mask_f = edge_mask.to(diff_sq.dtype)
            num_edges = edge_mask_f.sum(dim=(-1, -2)).clamp(min=1)
            return (diff_sq * edge_mask_f).sum(dim=(-1, -2)) / num_edges

        # Discrete branch: walk the same fields the loss does.
        target_masked = target.mask()
        x_classes_for_synth = _resolve_x_classes_for_loss(
            self.noise_process, pred, target_masked
        )

        total: torch.Tensor | None = None
        for field in sorted(self.noise_process.fields):
            pred_field = _read_field(pred, field, x_classes=x_classes_for_synth)
            target_field = _read_field(
                target_masked, field, x_classes=x_classes_for_synth
            )
            kind = GRAPHDATA_LOSS_KIND[field]
            if kind == "ce":
                if field == "X_class":
                    term = per_graph_node_ce(
                        pred_field, target_field, target_masked.node_mask
                    )
                elif field == "E_class":
                    term = per_graph_edge_ce(
                        pred_field, target_field, target_masked.node_mask
                    )
                elif field == "y_class":
                    term = per_graph_y_ce(pred_field, target_field)
                else:  # pragma: no cover - all CE fields handled above
                    raise NotImplementedError(
                        f"No per-graph CE helper registered for {field!r}."
                    )
            else:  # mse
                if field == "X_feat":
                    term = per_graph_node_mse(
                        pred_field, target_field, target_masked.node_mask
                    )
                elif field == "E_feat":
                    term = per_graph_edge_mse(
                        pred_field, target_field, target_masked.node_mask
                    )
                elif field == "y_feat":
                    term = per_graph_y_mse(pred_field, target_field)
                else:  # pragma: no cover
                    raise NotImplementedError(
                        f"No per-graph MSE helper registered for {field!r}."
                    )
            contribution = self.lambda_per_field[field] * term  # pyright: ignore[reportArgumentType]
            total = contribution if total is None else total + contribution
        if total is None:
            return torch.zeros(bs, device=device)
        return total

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Forward pass: delegates to the underlying ``GraphModel``."""
        return self.model(data, t=t)

    @override
    def training_step(self, batch: GraphState, batch_idx: int) -> torch.Tensor:
        """Single diffusion training step.

        Samples a uniform random timestep per batch element, applies forward
        noise via the noise process, runs the model to predict the clean
        graph, and computes the loss against the original batch.

        Parameters
        ----------
        batch
            Clean ``GraphState`` batch from the dataloader.
        batch_idx
            Index of the current batch (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss with gradient.
        """
        # Sparse-default refactor (Phase 5.2): the dataloader emits
        # ``GraphState``. The legacy diagnostic / loss helpers consume
        # ``DenseGraphState``-style upper-case fields, so we densify
        # the clean batch once at the top of the step and thread the
        # dense view (``target_dense``) through every downstream call.
        target_dense: DenseGraphState = batch.to_dense(
            edge_class_fill=_no_edge_fill_for(batch),
        )

        bs: int = int(target_dense.node_mask.shape[0])
        device = target_dense.node_mask.device

        # Sample random timestep per batch element: t in {0, ..., T}.
        # Upstream parity (apply_noise, diffusion_model_discrete.py:407-442):
        # training samples t in {0..T} (the t=0 term contributes to training loss),
        # validation samples t in {1..T} (t=0 handled separately by reconstruction_logp).
        t_int = torch.randint(0, self.T + 1, (bs,), device=device)

        # Apply forward noise at the sampled timesteps. The new sparse
        # entry point ``noise_process.sample(batch, t)`` returns a
        # ``GraphState``; the model accepts both carriers via the
        # input-coercion helpers (Phase 6.1) and produces the requested
        # output type when called with ``output_dense=True``.
        z_t = _noise_process_sample(self.noise_process, batch, t_int)

        condition = self.noise_process.process_state_condition_vector(t_int)

        # Model predicts clean data; request a dense distribution so the
        # downstream loss path can use ``masked_ce_loss(pred, target_dense)``
        # without an extra carrier flip. Wave 3-B/C/D add ``output_dense=True``
        # to every concrete model's ``forward``; until those land the
        # call site here is the contract that drives the model refactor.
        pred = self.model(z_t, t=condition, output_dense=True)

        # Per-component breakdown so we can log unweighted per-field losses
        # alongside the combined scalar that drives backprop. Same forward
        # pass; just keeps the intermediate terms instead of discarding them.
        # Threads ``target_dense`` (DenseGraphState) into the legacy helper
        # so its upper-case field reads (E_class / X_class / mask()) work.
        breakdown = self._compute_loss_breakdown(pred, target_dense)
        loss = self._combine_breakdown(breakdown)

        # ---- Per-timestep loss bin scatter (Stage 1 telemetry) ------------
        # Detached: the binning path is diagnostic only and must not
        # introduce gradient flow back through ``pred`` a second time.
        # ``no_grad`` keeps the autograd machinery from tracking the
        # per-graph helper ops at all. Threads the dense view so the
        # legacy per-graph helpers' upper-case field reads stay valid.
        with torch.no_grad():
            per_graph = self._per_graph_loss(pred, target_dense).detach()
            bin_idx = self._t_bin_idx(t_int)
            self._train_loss_t_sum.scatter_add_(0, bin_idx, per_graph)
            self._train_loss_t_count.scatter_add_(
                0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long)
            )

        # ---- Tier 1: per-step health metrics ------------------------------
        # batch_size=self._cur_bs threaded everywhere because Lightning
        # can't unambiguously infer it from GraphData (warns once per
        # log call). _cur_bs is stashed at on_train_batch_start /
        # validation_step entry.
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self._cur_bs,
        )
        for field, term in breakdown.items():
            self.log(
                f"train/loss_{field}",
                term,
                on_step=True,
                on_epoch=True,
                batch_size=self._cur_bs,
            )

        # Learning rate. Lightning auto-logs LR only when a scheduler is
        # attached; ours has none (upstream-parity flat LR), so emit it
        # explicitly so the dashboard shows a non-empty trace. Guarded
        # because ``self.optimizers()`` raises when called outside a
        # ``Trainer`` (e.g. unit tests that exercise ``training_step``
        # in isolation); diagnostics must never break the training path.
        try:
            opt = self.optimizers()
            if isinstance(opt, list):
                opt = opt[0]
            self.log(
                "train/lr",
                float(opt.param_groups[0]["lr"]),
                on_step=True,
                batch_size=self._cur_bs,
            )
        except (RuntimeError, AttributeError, IndexError):
            pass

        # Sampled timestep distribution per batch. Confirms the forward
        # noising is hitting the full schedule (uniform over [0, T]).
        t_float = t_int.float()
        self.log(
            "diagnostics-train/progress/t_mean",
            t_float.mean(),
            on_step=True,
            batch_size=self._cur_bs,
        )
        self.log(
            "diagnostics-train/progress/t_std",
            t_float.std(),
            on_step=True,
            batch_size=self._cur_bs,
        )

        # ---- Tier 2: stride-controlled training-dynamics metrics ----------
        # Logit max-abs as a saturation indicator. Soft-max output blows up
        # at |logit| ≈ 30+; healthy training keeps these in O(10).
        # Resolve C_x lazily — only needed when at least one CE field is
        # iterated, so non-CE training paths (MSE/BCE) don't pay the cost
        # nor trigger the assertion.
        ce_fields: list[FieldName] = [
            f for f in self.noise_process.fields if GRAPHDATA_LOSS_KIND[f] == "ce"
        ]
        if ce_fields and self.global_step % 50 == 0:
            x_classes_for_synth = _resolve_x_classes_for_loss(
                self.noise_process, pred, pred
            )
            for field in ce_fields:
                f_pred = _read_field(pred, field, x_classes=x_classes_for_synth)
                self.log(
                    f"diagnostics-train/progress/logit_max_abs_{field}",
                    f_pred.detach().abs().max(),
                    on_step=True,
                    batch_size=self._cur_bs,
                )

        # ---- Tier 3 (selected): prediction entropy ------------------------
        # Diverging entropy = collapse signal; rising entropy under noise =
        # underfit. Cheap to compute; logged at low cadence to limit wandb
        # cardinality.
        if ce_fields and self.global_step % 1000 == 0:
            x_classes_for_synth = _resolve_x_classes_for_loss(
                self.noise_process, pred, pred
            )
            for field in ce_fields:
                f_pred = _read_field(
                    pred, field, x_classes=x_classes_for_synth
                ).detach()
                log_p = F.log_softmax(f_pred, dim=-1)
                entropy = -(log_p.exp() * log_p).sum(dim=-1).mean()
                self.log(
                    f"diagnostics-train/progress/entropy_{field}",
                    entropy,
                    on_step=True,
                    batch_size=self._cur_bs,
                )

        return loss

    # ------------------------------------------------------------------
    # Diagnostic Lightning hooks
    # ------------------------------------------------------------------

    @override
    def on_train_batch_start(self, batch: GraphState, batch_idx: int) -> None:
        """Stash current batch_size + start the wall-clock timer."""
        # Sparse-default refactor: ``GraphState`` does not expose
        # ``node_mask`` (it lives on the dense carrier); the per-graph
        # node-count vector is the carrier-agnostic proxy for batch size.
        self._cur_bs = int(batch.num_nodes_per_graph.shape[0])
        self._train_batch_start_time = time.perf_counter()

    @override
    def on_train_batch_end(
        self,
        outputs: torch.Tensor | dict[str, Any],
        batch: GraphState,
        batch_idx: int,
    ) -> None:
        """Per-step wall clock + stride-controlled weight-norm logging."""
        elapsed = time.perf_counter() - self._train_batch_start_time
        self.log(
            "impl-perf/train/step_time_s",
            elapsed,
            on_step=True,
            on_epoch=False,
            batch_size=self._cur_bs,
        )

        # Per-t train-loss bins (Stage 1 telemetry). Logged on the
        # validation cadence so the train and val per-t panels share the
        # same x-axis and so the train accumulator covers a meaningful
        # window (eval_every_n_steps × bs samples ≫ n_t_bins).
        # ``write_through=True``: cadence-gated values logged via plain
        # ``self.log`` only reach the logger when the cadence iteration
        # happens to coincide with a ``log_every_n_steps`` flush boundary
        # — otherwise the epoch-end ``reset_results()`` wipes them.
        if self.global_step > 0 and self.global_step % self.eval_every_n_steps == 0:
            # Single GPU→CPU sync via ``.tolist()`` for both buffers,
            # then iterate Python-side. Was N small ``.item()`` calls
            # in a tight loop — each forced its own stream sync.
            counts = self._train_loss_t_count.tolist()
            sums = self._train_loss_t_sum.tolist()
            for i, count in enumerate(counts):
                if count == 0:
                    continue
                self.log(
                    f"diagnostics-train/progress/loss_per_t/bin_{i}",
                    sums[i] / float(count),
                    write_through=True,
                )
            self._train_loss_t_sum.zero_()
            self._train_loss_t_count.zero_()

        # Weight norm (Tier 2). Stride-controlled to keep WandB cardinality
        # bounded; 500 steps gives ~3 samples per validation window.
        if self.global_step % self.log_optimizer_health_every_n_steps == 0:
            weight_norms = self._compute_weight_norms_by_block()
            if weight_norms:
                self.log(
                    "diagnostics-train/opt-health/weight_norm_total",
                    weight_norms.pop("__total__"),
                    write_through=True,
                )
                for block_name, val in weight_norms.items():
                    self.log(
                        f"diagnostics-train/opt-health/weight_norm/{block_name}",
                        val,
                        write_through=True,
                    )

            # Stage 2: optimizer-health update-to-weight ratio. Computed
            # post-step so we can subtract the stashed θ_before snapshot
            # against the live (post-optimizer) weights. Block-bucketed
            # via ``_block_name`` to share the cadence + naming with
            # ``train/weight_norm/{block}``.
            self._log_update_to_weight(weight_norms)

            self._prev_weights_snapshot = None  # free

        # Flush write-through diagnostics collected this iteration
        # (``WriteThroughLogMixin.on_train_batch_end``).
        super().on_train_batch_end(outputs, batch, batch_idx)

    def _log_val_t_bins(
        self,
        prefix: str,
        sum_buf: torch.Tensor,
        count_buf: torch.Tensor,
    ) -> None:
        """Log per-bin averages for a validation-side scatter accumulator.

        Empty bins (count == 0) are skipped — they would otherwise log
        ``0.0`` and hide a "bin never sampled" condition behind a
        plausible-looking value.
        """
        # Single sync via ``.tolist()`` for both buffers; was N
        # ``.item()`` calls each syncing the stream.
        counts = count_buf.tolist()
        sums = sum_buf.tolist()
        for i, count in enumerate(counts):
            if count == 0:
                continue
            self.log(
                f"{prefix}/bin_{i}",
                sums[i] / float(count),
                on_epoch=True,
            )

    @override
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Gradient-norm + Adam-moment diagnostics.

        Three cadences:
        * Total grad L2 norm — every step (highest-value diagnostic; cost
          is one extra reduction over already-realised gradients).
        * Per-block grad L2 norm — every 500 steps (per-layer view of
          which transformer block is unstable).
        * Adam moment ratio mean ``|m| / (sqrt(v) + eps)`` — every 5000
          steps (effective per-parameter step magnitude; flags subspaces
          where the optimiser has stalled).
        """
        # Lightning's grad_norm helper returns a dict including the
        # ``grad_2.0_norm_total`` aggregate plus one ``grad_2.0_norm/<param>``
        # entry per parameter. We log only the total at every step and
        # collapse the per-parameter entries into per-block buckets at the
        # 500-step stride.
        norms = grad_norm(self, norm_type=2)
        total_key = "grad_2.0_norm_total"
        if total_key in norms:
            self.log(
                "diagnostics-train/opt-health/grad_norm_total",
                norms[total_key],
                on_step=True,
                batch_size=self._cur_bs,
            )

        # Cadence: fire when ``global_step + 1`` is divisible so the paired
        # post-step computation in ``on_train_batch_end`` (Δθ needs the
        # post-optimizer weights) sees ``global_step % N == 0`` — Lightning
        # increments ``global_step`` between the two hooks. Grad-derived
        # stats are computed here (gradients are alive) and logged with
        # ``write_through=True``: delivery no longer depends on this
        # iteration coinciding with a ``log_every_n_steps`` flush boundary,
        # which it generally does not in our short-epoch regime (the
        # ``2026-04-28`` smoke-run wipe bug; see
        # ``tests/training/test_optimizer_health.py``).
        if (self.global_step + 1) % self.log_optimizer_health_every_n_steps == 0:
            block_norms = self._aggregate_grad_norms_by_block(norms)
            for block_name, val in block_norms.items():
                self.log(
                    f"diagnostics-train/opt-health/grad_norm/{block_name}",
                    val,
                    write_through=True,
                )
            for key, value in self._compute_grad_health_by_block(optimizer).items():
                self.log(key, value, write_through=True)
            # Stash θ_before so ``on_train_batch_end`` can subtract it
            # from the live (post-step) weights to recover Δθ.
            self._prev_weights_snapshot = {
                name: p.detach().clone()
                for name, p in self.named_parameters()
                if p.requires_grad
            }

        if self.global_step % 5000 == 0 and self.global_step > 0:
            ratio = self._compute_adam_moment_ratio(optimizer)
            if ratio is not None:
                self.log(
                    "diagnostics-train/opt-health/adam_moment_ratio_mean",
                    ratio,
                    write_through=True,
                )

    def _compute_grad_health_by_block(
        self, optimizer: torch.optim.Optimizer
    ) -> dict[str, torch.Tensor | float]:
        """Per-block grad cosine, grad SNR, and global effective LR.

        Returns a dict ``{"diagnostics-train/opt-health/grad_snr/<block>": Tensor, ...}``
        for the caller (``on_before_optimizer_step``) to log with
        ``write_through=True``.

        * ``train/grad_snr/{block}`` — first/second-moment SNR of the
          gradient elements in the block, ``mean(g)² / var(g)``.
        * ``train/grad_cosine/{block}`` — cosine of the current
          per-block flat gradient against the previous step's stored
          flat gradient.
        * ``train/effective_lr`` — ``lr × ‖∇‖_total / ‖θ‖_total``.
        """
        out: dict[str, torch.Tensor | float] = {}
        grad_sq: dict[str, torch.Tensor] = {}
        grad_sum: dict[str, torch.Tensor] = {}
        grad_n: dict[str, int] = {}
        flat_chunks: dict[str, list[torch.Tensor]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            block = self._block_name(name)
            g = p.grad.detach()
            sq = g.pow(2).sum()
            sm = g.sum()
            grad_sq[block] = sq if block not in grad_sq else grad_sq[block] + sq
            grad_sum[block] = sm if block not in grad_sum else grad_sum[block] + sm
            grad_n[block] = grad_n.get(block, 0) + g.numel()
            flat_chunks.setdefault(block, []).append(g.flatten())

        if not grad_sq:
            return out

        new_flat: dict[str, torch.Tensor] = {
            block: torch.cat(chunks).detach() for block, chunks in flat_chunks.items()
        }

        # Grad SNR per block. ``var.clamp(min=eps)`` yields a finite,
        # large value when the gradient is constant (which is the right
        # qualitative signal: consistent direction → high SNR).
        for block, sq in grad_sq.items():
            n = grad_n[block]
            mean = grad_sum[block] / float(n)
            var = sq / float(n) - mean.pow(2)
            snr = mean.pow(2) / var.clamp(min=1e-12)
            out[f"diagnostics-train/opt-health/grad_snr/{block}"] = snr.detach()

        # Grad cosine vs previous-cadence step. First trigger has no
        # reference; we stash and skip. ``denom.clamp(min=1e-12)``
        # avoids NaN if a block briefly has all-zero grads.
        if self._prev_grad_by_block is not None:
            for block, flat in new_flat.items():
                prev = self._prev_grad_by_block.get(block)
                if prev is None or prev.shape != flat.shape:
                    continue
                num = (flat * prev).sum()
                denom = flat.norm() * prev.norm()
                cos = num / denom.clamp(min=1e-12)
                out[f"diagnostics-train/opt-health/grad_cosine/{block}"] = cos.detach()
        self._prev_grad_by_block = new_flat

        # Effective LR: lr × ‖∇‖ / ‖θ‖. Pull live LR from
        # ``param_groups[0]`` so it reflects scheduler state.
        try:
            lr = float(optimizer.param_groups[0]["lr"])  # pyright: ignore[reportAttributeAccessIssue, reportIndexIssue]
        except (KeyError, IndexError, AttributeError):
            return out
        grad_total_sq = next(iter(grad_sq.values())).clone()
        for v in list(grad_sq.values())[1:]:
            grad_total_sq = grad_total_sq + v
        param_sq = torch.zeros((), device=grad_total_sq.device)
        for p in self.parameters():
            if p.requires_grad:
                param_sq = param_sq + p.detach().pow(2).sum()
        eff = lr * grad_total_sq.sqrt() / param_sq.sqrt().clamp(min=1e-12)
        out["diagnostics-train/opt-health/effective_lr"] = eff.detach()
        return out

    def _log_update_to_weight(self, weight_norms: dict[str, torch.Tensor]) -> None:
        """Per-block ``‖Δθ‖ / ‖θ‖`` after the optimizer has stepped.

        Requires that ``on_before_optimizer_step`` stashed
        ``_prev_weights_snapshot`` on this step. Skipped silently when
        the snapshot is missing (covers the very first step on a fresh
        module and any path that bypassed the cadence guard).
        """
        if self._prev_weights_snapshot is None:
            return
        prev = self._prev_weights_snapshot
        update_sq: dict[str, torch.Tensor] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad or name not in prev:
                continue
            block = self._block_name(name)
            delta = p.detach() - prev[name]
            sq = delta.pow(2).sum()
            update_sq[block] = sq if block not in update_sq else update_sq[block] + sq
        for block, sq in update_sq.items():
            update_norm = sq.sqrt()
            wn = weight_norms.get(block)
            if wn is None:
                continue
            ratio = update_norm / wn.clamp(min=1e-12)
            self.log(
                f"diagnostics-train/opt-health/update_to_weight/{block}",
                ratio.detach(),
                write_through=True,
            )

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _block_name(param_name: str) -> str:
        """Group a parameter name into a block-level bucket.

        Examples
        --------
        ``transformer.tf_layers.5.self_attn.weight`` → ``tf_layers_5``
        ``transformer.mlp_in_X.0.weight`` → ``mlp_in_X``
        ``transformer.mlp_out_E.0.bias`` → ``mlp_out_E``

        Anything that doesn't match a known prefix lands in ``other``.
        """
        parts = param_name.split(".")
        for idx, part in enumerate(parts):
            if part == "tf_layers" and idx + 1 < len(parts):
                return f"tf_layers_{parts[idx + 1]}"
        for part in parts:
            if part.startswith("mlp_in_") or part.startswith("mlp_out_"):
                return part
        return "other"

    def _aggregate_grad_norms_by_block(
        self, norms: dict[str, float]
    ) -> dict[str, float]:
        """Reduce per-parameter grad norms into per-block L2 norms.

        Lightning's ``grad_norm`` returns one ``grad_2.0_norm/<param>``
        entry per parameter plus a ``grad_2.0_norm_total`` aggregate
        (values are Python floats, computed off-graph). We bucket the
        per-parameter entries by :meth:`_block_name` and aggregate via
        L2 (``sqrt(sum(n**2))``) to preserve the grad-norm semantics.
        """
        prefix = "grad_2.0_norm/"
        squared_sums: dict[str, float] = {}
        for key, val in norms.items():
            if not key.startswith(prefix):
                continue
            param_name = key[len(prefix) :]
            block = self._block_name(param_name)
            squared_sums[block] = squared_sums.get(block, 0.0) + float(val) ** 2
        return {block: sq**0.5 for block, sq in squared_sums.items()}

    def _compute_weight_norms_by_block(self) -> dict[str, torch.Tensor]:
        """L2 weight norms per block plus a ``__total__`` aggregate."""
        squared_sums: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            block = self._block_name(name)
            sq = param.detach().pow(2).sum()
            squared_sums[block] = (
                sq if block not in squared_sums else squared_sums[block] + sq
            )
        if not squared_sums:
            return {}
        result: dict[str, torch.Tensor] = {
            block: sq.sqrt() for block, sq in squared_sums.items()
        }
        total_sq = sum(squared_sums.values())
        result["__total__"] = total_sq.sqrt()  # pyright: ignore[reportAttributeAccessIssue]
        return result

    def _compute_adam_moment_ratio(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.Tensor | None:
        """Mean ``|m| / (sqrt(v) + eps)`` over all params with Adam state.

        Approximates the effective per-parameter step magnitude. Stable
        well-trained networks settle into a narrow band; runaway values
        signal optimiser instability. Returns ``None`` if no Adam moments
        have been materialised yet (e.g. step 0 with a non-Adam optimizer).
        """
        # ``torch.optim.Optimizer``'s public ``defaults`` / ``param_groups`` /
        # ``state`` attributes are not declared in the type stubs we have
        # available; the runtime API is documented and stable.
        eps_default = optimizer.defaults.get("eps", 1e-8)  # pyright: ignore[reportAttributeAccessIssue]
        ratios: list[torch.Tensor] = []
        for group in optimizer.param_groups:  # pyright: ignore[reportAttributeAccessIssue]
            eps = group.get("eps", eps_default)
            for param in group["params"]:
                state = optimizer.state.get(param, {})  # pyright: ignore[reportAttributeAccessIssue]
                m = state.get("exp_avg")
                v = state.get("exp_avg_sq")
                if m is None or v is None:
                    continue
                ratios.append((m.abs() / (v.sqrt() + eps)).mean())
        if not ratios:
            return None
        return torch.stack(ratios).mean()

    def _compute_loss_breakdown(
        self, pred: DenseGraphData, target: DenseGraphState
    ) -> dict[str, torch.Tensor]:
        """Per-field unweighted losses keyed by field name.

        Returns a dict whose values are the raw per-field terms before
        ``self.lambda_per_field`` weighting. The combined scalar (the
        thing actually backpropagated) is computed by :meth:`_compute_loss`
        from this dict.

        Splitting the breakdown lets ``training_step`` / ``validation_step``
        log per-component losses (e.g. ``train/loss_X_class``,
        ``train/loss_E_class``) without paying for two forward passes.

        For non-CE losses (``loss_type='mse'`` / ``'bce_logits'``) the
        breakdown has a single ``'edge_state'`` key.
        """
        if self._train_loss_discrete is None:
            if pred.E_feat is not None:
                pred_edge_state = _dense_to_edge_scalar(pred, source="feat")
            else:
                pred_edge_state = _dense_to_edge_scalar(pred, source="class")
            target_edge_state = _continuous_target_edge_state(target)
            return {
                "edge_state": self.criterion(pred_edge_state, target_edge_state.float())
            }

        # Zero padding rows in the target categorical fields so the row
        # predicate inside the CE helpers correctly excludes them. Matches
        # upstream's ``dense_data.mask(node_mask)`` before ``train_loss``.
        target = target.mask()

        # Discrete branch: ``self._train_loss_discrete is not None``
        # generally implies a CategoricalNoiseProcess owning the CE fields,
        # though tests may compose CE loss atop a non-categorical noise
        # process when the data already carries X_class. We pick C_x from
        # the noise process when available, otherwise from the populated
        # X_class on either pred or target (a defensive fallback used only
        # in test fixtures; production configs always go through
        # CategoricalNoiseProcess and the setup-time invariant in
        # ``setup()`` guarantees consistency).
        x_classes_for_synth = _resolve_x_classes_for_loss(
            self.noise_process, pred, target
        )
        breakdown: dict[str, torch.Tensor] = {}
        for field in sorted(self.noise_process.fields):
            pred_field = _read_field(pred, field, x_classes=x_classes_for_synth)
            target_field = _read_field(target, field, x_classes=x_classes_for_synth)
            kind = GRAPHDATA_LOSS_KIND[field]
            if kind == "ce":
                if field == "X_class":
                    term = masked_node_ce(pred_field, target_field, target.node_mask)
                elif field == "E_class":
                    term = masked_edge_ce(pred_field, target_field, target.node_mask)
                elif field == "y_class":
                    term = masked_y_ce(pred_field, target_field)
                else:  # pragma: no cover - all CE fields handled above
                    raise NotImplementedError(
                        f"No masked CE helper registered for categorical field "
                        f"{field!r}."
                    )
            else:  # kind == "mse"
                if field == "X_feat":
                    term = masked_node_mse(pred_field, target_field, target.node_mask)
                elif field == "E_feat":
                    term = masked_edge_mse(pred_field, target_field, target.node_mask)
                elif field == "y_feat":
                    term = masked_y_mse(pred_field, target_field)
                else:  # pragma: no cover - all MSE fields handled above
                    raise NotImplementedError(
                        f"No masked MSE helper registered for continuous field "
                        f"{field!r}."
                    )
            breakdown[field] = term
        return breakdown

    def _combine_breakdown(self, breakdown: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum a per-field breakdown weighted by ``self.lambda_per_field``.

        Shared by :meth:`_compute_loss`, :meth:`training_step`, and
        :meth:`validation_step` so the weighted sum is computed once after
        a single forward pass. Non-CE losses (``mse`` / ``bce_logits``)
        return the single ``'edge_state'`` term unweighted.
        """
        if self._train_loss_discrete is None:
            return breakdown["edge_state"]
        total: torch.Tensor | None = None
        for field, term in breakdown.items():
            contribution = self.lambda_per_field[field] * term  # pyright: ignore[reportArgumentType]
            total = contribution if total is None else total + contribution
        if total is None:  # pragma: no cover - fields non-empty by invariant
            raise RuntimeError(
                "DiffusionModule._combine_breakdown: noise_process.fields is "
                "empty; cannot compute a cross-entropy loss."
            )
        return total

    def _compute_loss(
        self, pred: DenseGraphData, target: DenseGraphState
    ) -> torch.Tensor:
        """Compute the combined per-field loss for backprop.

        Calls :meth:`_compute_loss_breakdown` to get unweighted per-field
        terms, then sums them via :meth:`_combine_breakdown` weighted by
        ``self.lambda_per_field`` (giving the upstream-DiGress
        ``lambda_E=5.0`` edge weighting by default).
        """
        return self._combine_breakdown(self._compute_loss_breakdown(pred, target))

    def _compute_reconstruction(self, batch: DenseGraphState) -> torch.Tensor:
        """Reconstruction log-probability ``log p(x_0 | z_1)``.

        Aligns with upstream DiGress's ``reconstruction_logp``: noise the
        clean batch to ``z_1``, run the model, and **pull the predicted
        ``x_0`` distribution through the per-class marginalised reverse
        kernel** at ``(t=1, s=0)`` to obtain ``p(x_0 = . | z_1)`` rather
        than scoring the raw softmax. Locally pre-2026-04-15 we scored
        the softmax directly (``_compute_reconstruction_at_t1``); the
        difference is bounded by the Q_1 transition kernel ≈ identity at
        T=1000 (a ~1e-3 shift in absolute magnitude) but matters for
        numerical parity with published DiGress results.
        """
        if not isinstance(self.noise_process, CategoricalNoiseProcess):
            raise TypeError(
                f"_compute_reconstruction requires CategoricalNoiseProcess, "
                f"got {type(self.noise_process).__name__}"
            )
        # ``node_mask`` lives only on the dense carriers; ``num_nodes_per_graph``
        # is on the abstract base and works for both sparse and dense.
        bs = int(batch.num_nodes_per_graph.shape[0])
        device = batch.num_nodes_per_graph.device

        t_int = torch.ones(bs, dtype=torch.long, device=device)
        s_int = torch.zeros_like(t_int)

        z_1 = self.noise_process.forward_sample(batch, t_int).z_t
        condition = self.noise_process.process_state_condition_vector(t_int)
        pred_logits = self.model(z_1, t=condition)
        x0_param = self.noise_process.model_output_to_posterior_parameter(pred_logits)

        # ``_posterior_probabilities_marginalised`` returns the
        # per-position PMF over ``z_s = x_0`` (since s=0). We score the
        # one-hot clean batch under this PMF.
        recon_pmf = self.noise_process._posterior_probabilities_marginalised(  # pyright: ignore[reportPrivateUsage]
            z_1, x0_param, t_int, s_int
        )
        # The categorical posterior helper returns GraphData but always
        # produces a dense distribution carrier in practice; assert so
        # pyright narrows for the dense-only reconstruction helper.
        assert isinstance(
            recon_pmf, GraphState | DenseGraphState | DenseGraphDistribution
        )
        return _categorical_reconstruction_log_prob(
            batch,
            recon_pmf,
            x_classes=self.noise_process.x_classes,
            e_classes=self.noise_process.e_classes,
        )

    def _compute_reconstruction_upstream_style(
        self, batch: DenseGraphState
    ) -> torch.Tensor:
        """Reconstruction log-probability in upstream DiGress's form.

        Direct port of
        ``digress-upstream-readonly/src/diffusion_model_discrete.py::
        reconstruction_logp`` (lines 368-405): sample
        ``z_0 = x_0 @ Q_0`` (one-hot through the t=0 transition, which is
        ≈ identity for cosine-IDDPM at T=1000), run the model, and score
        the raw softmax against the clean target. This is the published
        "first step of denoising" reconstruction likelihood. The
        marginalised reverse-posterior path (see
        :meth:`_compute_reconstruction`) integrates over an additional
        Q_1 step instead; the gap is ~1e-3 abs at T=1000 (parity #31).
        """
        if not isinstance(self.noise_process, CategoricalNoiseProcess):
            raise TypeError(
                f"_compute_reconstruction_upstream_style requires "
                f"CategoricalNoiseProcess, got {type(self.noise_process).__name__}"
            )
        # ``node_mask`` lives only on the dense carriers; ``num_nodes_per_graph``
        # is on the abstract base and works for both sparse and dense.
        bs = int(batch.num_nodes_per_graph.shape[0])
        device = batch.num_nodes_per_graph.device

        # Upstream samples z_0 by drawing from x_0 @ Q_0; ``forward_sample``
        # at t=0 does exactly that (forward noising at t=0 is the Q_0
        # mixing kernel) and returns a one-hot ``z_0`` ready for the model.
        t_zero = torch.zeros(bs, dtype=torch.long, device=device)
        z_0 = self.noise_process.forward_sample(batch, t_zero).z_t
        condition = self.noise_process.process_state_condition_vector(t_zero)
        pred_logits = self.model(z_0, t=condition)
        # ``model_output_to_posterior_parameter`` already applies softmax
        # along the class axis on both X_class and E_class.
        pred_probs = self.noise_process.model_output_to_posterior_parameter(pred_logits)
        # ``model_output_to_posterior_parameter`` is typed against the
        # abstract GraphData; categorical configs always emit a dense
        # carrier in practice, so narrow for the dense-only helper.
        assert isinstance(
            pred_probs, GraphState | DenseGraphState | DenseGraphDistribution
        )
        return _categorical_reconstruction_log_prob(
            batch,
            pred_probs,
            x_classes=self.noise_process.x_classes,
            e_classes=self.noise_process.e_classes,
        )

    @torch.no_grad()
    @override
    def validation_step(self, batch: GraphState, batch_idx: int) -> None:
        """Compute validation loss and VLB metrics for a single batch.

        Reference graphs for generative evaluation are pulled from the
        datamodule at epoch end, not accumulated here.

        Parameters
        ----------
        batch
            Clean ``GraphState`` batch.
        batch_idx
            Index of the current batch (unused).
        """
        # Sparse-default refactor: densify once at the top so the legacy
        # VLB / reconstruction / per-graph helpers (which read upper-case
        # fields and ``node_mask``) thread through ``target_dense``. The
        # noise step itself uses the new sparse ``sample`` entry point.
        target_dense: DenseGraphState = batch.to_dense(
            edge_class_fill=_no_edge_fill_for(batch),
        )

        bs: int = int(target_dense.node_mask.shape[0])
        self._cur_bs = bs
        device = target_dense.node_mask.device

        # Validation loss at a random timestep
        t_int = torch.randint(1, self.T + 1, (bs,), device=device)
        z_t = _noise_process_sample(self.noise_process, batch, t_int)
        condition = self.noise_process.process_state_condition_vector(t_int)
        pred = self.model(z_t, t=condition, output_dense=True)
        breakdown = self._compute_loss_breakdown(pred, target_dense)
        loss = self._combine_breakdown(breakdown)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self._cur_bs,
        )
        for field, term in breakdown.items():
            self.log(
                f"val/loss_{field}",
                term,
                on_step=False,
                on_epoch=True,
                batch_size=self._cur_bs,
            )

        # Per-timestep validation loss scatter. Mirrors the train-side
        # binning so the val/loss panel can be split by t. ``no_grad`` is
        # redundant under the ``@torch.no_grad`` decorator on this method
        # but kept explicit for clarity at the binning insertion point.
        with torch.no_grad():
            per_graph_val = self._per_graph_loss(pred, target_dense).detach()
            bin_idx = self._t_bin_idx(t_int)
            self._val_loss_t_sum.scatter_add_(0, bin_idx, per_graph_val)
            self._val_loss_t_count.scatter_add_(
                0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long)
            )

        # VLB estimation via single random timestep (standard DDPM
        # approach). Each batch samples one ``t`` and computes ``L_t``;
        # the epoch-end average gives an estimate of ``E_t[L_t]`` that
        # is unbiased in ``t`` and exact (closed-form categorical KL)
        # for each individual ``L_t``. This matches the DiGress
        # reference implementation (cvignac/DiGress,
        # ``src/diffusion_model.py::compute_val_loss``). For full-chain
        # VLB, see the periodic ``DiffusionLikelihoodCollector``
        # evaluation in :meth:`on_validation_epoch_end`.
        if isinstance(self.noise_process, CategoricalNoiseProcess):
            cat_process = self.noise_process

            x0_param = self.noise_process.model_output_to_posterior_parameter(pred)
            s_int = t_int - 1

            # Analytic KL_t = KL(q(z_s|z_t,x_0) || p_theta(z_s|z_t)).
            # The "true" posterior always plugs the clean batch (one-hot
            # x_0) into the Bayes formula. The "predicted" posterior
            # selection follows ``use_marginalised_vlb_kl`` (D-6):
            #
            # * False (upstream parity, default): plug the soft x_0
            #   prediction directly into the same Bayes posterior formula
            #   (diffusion_model_discrete.py:340-360).
            # * True: use the marginalised form
            #   ``sum_c p(z_s|z_t,x_0=c) * pred[c]`` — the distribution
            #   the sampler actually draws from (see the Phase C sampler
            #   change). Lower variance early in training; coincides with
            #   the plug-in form once ``p_θ`` is one-hot.
            true_posterior = cat_process._posterior_probabilities(  # pyright: ignore[reportPrivateUsage]
                z_t, target_dense, t_int, s_int
            )
            if self.use_marginalised_vlb_kl:
                pred_posterior = cat_process._posterior_probabilities_marginalised(  # pyright: ignore[reportPrivateUsage]
                    z_t, x0_param, t_int, s_int
                )
            else:
                pred_posterior = cat_process._posterior_probabilities(  # pyright: ignore[reportPrivateUsage]
                    z_t, x0_param, t_int, s_int
                )
            # CategoricalNoiseProcess posterior helpers return GraphData but
            # always emit a dense distribution carrier in practice (the
            # categorical reverse kernel works on (B, n_max, n_max, K)
            # tensors). Assert so pyright narrows for the dense-only KL helper.
            assert isinstance(true_posterior, DenseGraphState | DenseGraphDistribution)
            assert isinstance(pred_posterior, DenseGraphState | DenseGraphDistribution)
            kl_diffusion = self.T * _categorical_kl_per_graph(
                true_posterior, pred_posterior
            )

            # Analytic KL(q(z_T|x_0) || prior). At T=1000 with the
            # cosine-IDDPM schedule, ``q(z_T|x_0) ≈ stationary prior``
            # so this term is small but non-zero; computing it
            # analytically removes the per-sample MC variance the
            # log-prob-of-sample formulation carried before.
            t_T = torch.full((bs,), self.T, device=device, dtype=torch.long)
            forward_pmf_T = cat_process.forward_pmf(target_dense, t_T)
            prior_pmf = cat_process.prior_pmf(target_dense.node_mask)
            assert isinstance(forward_pmf_T, DenseGraphState | DenseGraphDistribution)
            assert isinstance(prior_pmf, DenseGraphState | DenseGraphDistribution)
            kl_prior = _categorical_kl_per_graph(forward_pmf_T, prior_pmf)

            # Reconstruction term selection follows ``use_upstream_reconstruction``
            # (D-7). Default True selects the upstream-style z_0 + raw
            # softmax form; False keeps the marginalised z_1 form.
            if self.use_upstream_reconstruction:
                reconstruction = self._compute_reconstruction_upstream_style(
                    target_dense
                )
            else:
                reconstruction = self._compute_reconstruction(target_dense)

            if self._size_distribution is None:
                raise RuntimeError(
                    "DiffusionModule.validation_step (categorical branch): "
                    "_size_distribution is None. The datamodule must populate "
                    "a SizeDistribution in setup() — see SpectreSBMDataModule / "
                    "SyntheticCategoricalDataModule."
                )
            node_counts = target_dense.node_mask.sum(dim=-1).long()  # (bs,)
            log_pn = self._size_distribution.log_prob(node_counts).mean()

            # NLL = -log_pN + kl_prior + E_t[L_t] - reconstruction_logp
            nll = (
                -log_pn + kl_prior.mean() + kl_diffusion.mean() - reconstruction.mean()
            )

            self._vlb_nll.append(nll.detach())
            self._vlb_kl_prior.append(kl_prior.mean().detach())
            self._vlb_kl_diffusion.append(kl_diffusion.mean().detach())
            self._vlb_reconstruction.append(reconstruction.mean().detach())
            self._vlb_log_pn.append(log_pn.detach())

            # Per-timestep ELBO-component bin scatter. The three terms
            # are already per-graph ``(bs,)`` tensors at this point.
            # ``detach`` is defensive — every accumulator path here is
            # under the @torch.no_grad on validation_step, so autograd
            # is already off, but the explicit detach guarantees the
            # buffer never grows a tape if upstream behaviour changes.
            bin_idx_vlb = self._t_bin_idx(t_int)
            ones_long = torch.ones_like(bin_idx_vlb, dtype=torch.long)
            self._val_kl_diff_t_sum.scatter_add_(0, bin_idx_vlb, kl_diffusion.detach())
            self._val_kl_diff_t_count.scatter_add_(0, bin_idx_vlb, ones_long)
            self._val_kl_prior_t_sum.scatter_add_(0, bin_idx_vlb, kl_prior.detach())
            self._val_kl_prior_t_count.scatter_add_(0, bin_idx_vlb, ones_long)
            self._val_recon_t_sum.scatter_add_(0, bin_idx_vlb, reconstruction.detach())
            self._val_recon_t_count.scatter_add_(0, bin_idx_vlb, ones_long)

            # Bits-per-edge: total NLL across the val set / total edges.
            # ``nll`` here is the batch-mean; multiply by bs to recover
            # the batch sum so the eventual sum across batches equals the
            # set-level NLL total.
            edge_count, potential_count = self._batch_edge_counts(target_dense)
            self._vlb_nll_sum.append(nll.detach() * float(bs))
            self._vlb_edge_count.append(edge_count.detach())
            self._vlb_potential_count.append(potential_count.detach())
        elif isinstance(self.noise_process, ExactDensityNoiseProcess):
            # Continuous (Gaussian) noise processes still use the
            # log-prob-on-sample VLB; the analytic categorical-KL path
            # above is specific to categorical PMFs.
            exact_process = self.noise_process
            x0_param = self.noise_process.model_output_to_posterior_parameter(pred)
            s_int = t_int - 1
            z_s = exact_process.posterior_sample(z_t, target_dense, t_int, s_int)
            log_q_true = exact_process.posterior_log_prob(
                z_s, z_t, target_dense, t_int, s_int
            )
            log_q_pred = exact_process.posterior_log_prob(
                z_s, z_t, x0_param, t_int, s_int
            )
            kl_diffusion = self.T * (log_q_true - log_q_pred)

            t_T = torch.full((bs,), self.T, device=device, dtype=torch.long)
            z_T = exact_process.forward_sample(target_dense, t_T).z_t
            kl_prior = exact_process.forward_log_prob(
                z_T, target_dense, t_T
            ) - exact_process.prior_log_prob(z_T)

            if self._size_distribution is None:
                raise RuntimeError(
                    "DiffusionModule.validation_step (continuous branch): "
                    "_size_distribution is None. The datamodule must populate "
                    "a SizeDistribution in setup() — see SpectreSBMDataModule / "
                    "SyntheticCategoricalDataModule."
                )
            node_counts = target_dense.node_mask.sum(dim=-1).long()
            log_pn = self._size_distribution.log_prob(node_counts).mean()

            nll = -log_pn + kl_prior.mean() + kl_diffusion.mean()

            self._vlb_nll.append(nll.detach())
            self._vlb_kl_prior.append(kl_prior.mean().detach())
            self._vlb_kl_diffusion.append(kl_diffusion.mean().detach())
            self._vlb_log_pn.append(log_pn.detach())

            # Continuous branch: per-t bins for kl_diffusion and kl_prior
            # only (no reconstruction term — Wave 5 will close that).
            bin_idx_vlb = self._t_bin_idx(t_int)
            ones_long = torch.ones_like(bin_idx_vlb, dtype=torch.long)
            self._val_kl_diff_t_sum.scatter_add_(0, bin_idx_vlb, kl_diffusion.detach())
            self._val_kl_diff_t_count.scatter_add_(0, bin_idx_vlb, ones_long)
            self._val_kl_prior_t_sum.scatter_add_(0, bin_idx_vlb, kl_prior.detach())
            self._val_kl_prior_t_count.scatter_add_(0, bin_idx_vlb, ones_long)

            edge_count, potential_count = self._batch_edge_counts(target_dense)
            self._vlb_nll_sum.append(nll.detach() * float(bs))
            self._vlb_edge_count.append(edge_count.detach())
            self._vlb_potential_count.append(potential_count.detach())

    @staticmethod
    def _batch_edge_counts(
        batch: DenseGraphState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum-over-batch (edge_count, potential_edge_count) as scalar tensors.

        ``edge_count`` is the number of present undirected edges (sum of
        the upper triangle of the binarised adjacency, masked by
        ``node_mask``). ``potential_edge_count`` is :math:`\\sum_g n_g
        (n_g - 1)/2`, the maximum number of undirected edges given each
        graph's node count. Both are returned without the leading
        graph axis so the validation accumulator collapses cleanly.
        """
        node_mask = batch.node_mask
        device = node_mask.device
        n = node_mask.size(1)

        if batch.E_class is not None and batch.E_class.shape[-1] >= 2:
            adj = (batch.E_class.argmax(dim=-1) != 0).to(torch.float32)
        elif batch.E_feat is not None:
            e_feat = batch.E_feat
            scalar = e_feat if e_feat.dim() == node_mask.dim() + 1 else e_feat[..., 0]
            adj = (scalar > 0.5).to(torch.float32)
        else:
            return (
                torch.zeros((), device=device),
                torch.zeros((), device=device),
            )

        diag_mask = ~torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
        mask_2d = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & diag_mask).to(
            adj.dtype
        )
        edges = (adj * mask_2d).sum() / 2  # undirected
        node_counts = node_mask.sum(dim=-1).to(adj.dtype)
        potential = (node_counts * (node_counts - 1) / 2).sum()
        return edges.detach(), potential.detach()

    @torch.no_grad()
    @override
    def test_step(self, batch: GraphState, batch_idx: int) -> None:
        """Test step -- delegates to ``validation_step``."""
        self.validation_step(batch, batch_idx)

    # TODO: add on_test_epoch_end that mirrors on_validation_epoch_end
    # but passes "test" stage to get_reference_graphs.

    @override
    def on_validation_epoch_end(self) -> None:
        """Log VLB metrics and run generative evaluation at configured intervals.

        Reference graphs are pulled from the datamodule's
        ``get_reference_graphs()`` rather than accumulated per-batch.
        Generative evaluation is skipped when ``sampler`` is ``None``
        (e.g. single-step denoising) or no evaluator is attached.

        Notes
        -----
        Generation cadence is gated on ``global_step``:

            ``self.global_step % self.eval_every_n_steps == 0``

        rather than on the upstream DiGress epoch counter
        (``val_counter % sample_every_val == 0``,
        ``digress-upstream-readonly/src/diffusion_model_discrete.py``).
        Step-based gating is invariant under batch-size and
        dataset-size changes, so the same ``eval_every_n_steps`` value
        keeps a stable generation cadence as data scales up or down.
        Per the parity triage (D-10 / parity #41) this is the design we
        deliberately preserve; there is no toggle.

        For runs that need to *match* an upstream epoch cadence, convert
        as follows:

            ``upstream_sample_every_val (epochs)``  ↔
            ``ours eval_every_n_steps = upstream_sample_every_val
            * len(train_dataloader)``

        where ``len(train_dataloader)`` is the steps-per-epoch count at
        the chosen batch size. Round to the nearest integer; the gate is
        an exact modulo, not a "≥" check, so off-by-one rounding skips
        the gate at the boundary epoch.
        """
        # Log VLB metrics for exact-density processes.
        if self._vlb_nll:
            self.log("val/epoch_NLL", torch.stack(self._vlb_nll).mean(), on_epoch=True)
            self.log(
                "val/kl_prior", torch.stack(self._vlb_kl_prior).mean(), on_epoch=True
            )
            self.log(
                "val/kl_diffusion",
                torch.stack(self._vlb_kl_diffusion).mean(),
                on_epoch=True,
            )
            # Wave 2.5: the ExactDensity branch (Gaussian) does not yet populate
            # _vlb_reconstruction; Wave 5 will close this by wiring per-field
            # reconstruction log-probs for Gaussian. Until then, only the
            # categorical branch contributes here.
            if self._vlb_reconstruction:
                self.log(
                    "val/reconstruction_logp",
                    torch.stack(self._vlb_reconstruction).mean(),
                    on_epoch=True,
                )
            self.log("val/log_pN", torch.stack(self._vlb_log_pn).mean(), on_epoch=True)

            # Bits-per-edge: aggregate-then-divide across the val set so
            # the ratio is exact (sum_NLL / sum_edges) rather than the
            # noisier mean-of-ratios. ``log(2)`` converts nats → bits.
            if self._vlb_edge_count:
                total_nll = torch.stack(self._vlb_nll_sum).sum()
                total_edges = torch.stack(self._vlb_edge_count).sum()
                total_potential = torch.stack(self._vlb_potential_count).sum()
                # ``log(2.0)`` as a tensor on the same device avoids a
                # spurious device transfer when computing the ratio.
                ln2 = torch.log(torch.tensor(2.0, device=total_nll.device))
                self.log(
                    "diagnostics-val/progress/bits_per_edge",
                    total_nll / (total_edges.clamp(min=1.0) * ln2),
                    on_epoch=True,
                )
                self.log(
                    "diagnostics-val/progress/bits_per_potential_edge",
                    total_nll / (total_potential.clamp(min=1.0) * ln2),
                    on_epoch=True,
                )

            # Per-t bins. Empty bins (count==0) are skipped silently so
            # the dashboard doesn't grow ghost entries when n_t_bins
            # exceeds the per-epoch ``t`` coverage. ``count.clamp(min=1)``
            # would log a zero average for empty bins which is misleading.
            self._log_val_t_bins(
                "diagnostics-val/progress/kl_diffusion_per_t",
                self._val_kl_diff_t_sum,
                self._val_kl_diff_t_count,
            )
            self._log_val_t_bins(
                "diagnostics-val/progress/kl_prior_per_t",
                self._val_kl_prior_t_sum,
                self._val_kl_prior_t_count,
            )
            self._log_val_t_bins(
                "diagnostics-val/progress/reconstruction_per_t",
                self._val_recon_t_sum,
                self._val_recon_t_count,
            )
            self._log_val_t_bins(
                "diagnostics-val/progress/loss_per_t",
                self._val_loss_t_sum,
                self._val_loss_t_count,
            )

            self._vlb_nll.clear()
            self._vlb_kl_prior.clear()
            self._vlb_kl_diffusion.clear()
            self._vlb_reconstruction.clear()
            self._vlb_log_pn.clear()
            self._vlb_nll_sum.clear()
            self._vlb_edge_count.clear()
            self._vlb_potential_count.clear()

        if self.evaluator is None or self.sampler is None:
            return

        # Skip generative evaluation on an untrained model. ``global_step == 0``
        # covers Lightning's initial sanity-check pass (where ``global_step``
        # is 0 and the model is fresh) and any other pre-training invocation.
        # Without this guard the ``eval_every_n_steps`` modulo gate passes
        # trivially (``0 % N == 0``), causing us to sample from a random-init
        # model and log meaningless metrics.
        if self.global_step == 0:
            return

        if self.global_step % self.eval_every_n_steps != 0:
            return

        # GraphEvaluator and MolecularEvaluator both consume
        # list[GraphData] post the universal-transport refactor (spec
        # 2026-05-01-graphdata-eval-pipeline-minispec.md), so the
        # branch is gone. The two evaluators differ in which metrics
        # they emit, not in how they're called.
        refs = self.trainer.datamodule.get_reference_graphs(  # pyright: ignore[reportAttributeAccessIssue]
            "val", self.evaluator.eval_num_samples
        )
        if len(refs) < 2:
            return

        generated_graphs = self.generate_graphs(len(refs))

        # Both evaluators accept ``GraphData`` at runtime per the
        # universal-transport refactor (spec
        # ``2026-05-01-graphdata-eval-pipeline-minispec.md``); the static
        # signature on :class:`MolecularEvaluator` still carries the
        # narrower ``Sequence[DenseGraphState]`` from before that refactor.
        # The cast-through-Sequence keeps the call site honest without
        # forcing a redundant per-graph densification on every eval cycle.
        results = self.evaluator.evaluate(
            refs=refs,
            generated=cast(Sequence[DenseGraphState], generated_graphs),
        )
        if results is not None:
            # All sample-quality metrics — including the new block-structure
            # ones (modularity_q, spectral_gap_l2, empirical_p_in/p_out) —
            # live under the top-level ``gen-val/`` group so the W&B view
            # cleanly separates "core training/loss" from "what does the
            # generated sample distribution look like".
            #
            # Unit note: the ``gen-val/{degree,clustering,orbit,spectral}_mmd``
            # keys hold V-statistic squared-MMD values (the literature's
            # unsuffixed "MMD"); see ``docs/eval/mmd-units-and-protocol.md``.
            for key, value in results.to_dict().items():
                if value is not None:
                    self.log(f"gen-val/{key}", value, on_epoch=True)
            if self.visualization["enabled"]:
                # build_validation_visualizations is nx-native (spring
                # layout, networkx draw helpers). Convert at the leaf
                # via GraphData.to_networkx() — cheap and lossless.
                # Both ``refs`` (per-graph batched ``DenseGraphState`` from
                # ``get_reference_graphs``) and ``generated_graphs``
                # (per-graph sparse ``GraphState`` from ``sampler.sample``)
                # carry a leading batch dim of 1, so ``batch_index=0``
                # selects the single graph in either carrier.
                figures = build_validation_visualizations(
                    refs=[g.to_networkx(batch_index=0) for g in refs],
                    generated=[g.to_networkx(batch_index=0) for g in generated_graphs],
                    num_samples=int(self.visualization["num_samples"]),
                )
                log_figures(
                    self.trainer.loggers,  # pyright: ignore[reportAttributeAccessIssue]
                    figures,
                    global_step=self.global_step,
                )

        # Full-chain VLB via DiffusionLikelihoodCollector.
        # Runs the complete reverse chain on a subset of reference graphs
        # and sums per-step KL for a lower-variance VLB estimate than the
        # single-timestep estimator used in validation_step.
        collector = DiffusionLikelihoodCollector()
        num_vlb_graphs = min(len(refs), 16)
        self.generate_graphs(num_vlb_graphs, collector=collector)
        vlb_results = collector.results()
        self.log("gen-val/full_chain_vlb", vlb_results["vlb"], on_epoch=True)
        if "mean_kl" in vlb_results:
            self.log("gen-val/mean_step_kl", vlb_results["mean_kl"], on_epoch=True)

    def generate_graphs(
        self,
        num_graphs: int,
        *,
        collector: StepMetricCollector | None = None,
        chunk_size: int | None = None,
    ) -> list[GraphState]:
        """Generate graphs using the sampler and return as ``GraphState``.

        Parameters
        ----------
        num_graphs
            Number of graphs to generate.
        collector
            Optional per-step metric collector forwarded to the sampler.

        Returns
        -------
        list[GraphState]
            Generated graphs as per-graph sparse ``GraphState`` (the
            sampler output type post the sparse-default refactor).
            Callers that need a NetworkX view convert at the consumption
            site via :meth:`GraphState.to_networkx` (cheap, lossless,
            attaches ``x_class`` / ``e_class`` per spec
            ``2026-05-01-graphdata-eval-pipeline-minispec.md``).

        Notes
        -----
        ``ChainSavingCallback`` (parity #46 / D-16a) threads its
        :class:`~tmgg.diffusion.chain_recorder.ChainRecorder` through
        this method via a one-shot stash on
        ``self._pending_chain_recorder``. The stash is consumed and
        cleared here so a subsequent ``generate_graphs`` call cannot
        accidentally re-use a stale recorder. The plumbing keeps the
        sampler call site single-sourced inside the module while
        letting the callback own the recorder lifecycle.
        """
        if self.sampler is None:
            raise RuntimeError(
                "Cannot generate graphs: no Sampler was provided. "
                "SingleStepDenoisingModule (T=1) does not use reverse sampling."
            )
        device = next(self.parameters()).device

        # Use variable-size generation if the training distribution is non-degenerate
        num_nodes_arg: int | torch.Tensor = self.num_nodes
        if (
            self._size_distribution is not None
            and not self._size_distribution.is_degenerate
        ):
            num_nodes_arg = self._size_distribution.sample(num_graphs)

        # One-shot recorder handoff from ChainSavingCallback. ``getattr``
        # with a default makes the read non-fatal when no callback set
        # the stash; ``delattr`` clears it after use to enforce single-use
        # semantics. The Sampler accepts both single-recorder and
        # dict-of-recorder forms (W2-6 composite fan-out).
        chain_recorder = getattr(self, "_pending_chain_recorder", None)
        if chain_recorder is not None:
            del self._pending_chain_recorder

        # ``chunk_size`` splits the requested ``num_graphs`` into
        # consecutive sampler.sample calls. Without chunking, a one-shot
        # eval pass at ``num_graphs=32`` runs the diffusion sample loop
        # on a (32, n_max, ...) batch — which forces a fresh
        # ``torch.compile`` trace whenever the training-time batch_size
        # was different (e.g. 12). Chunking by the train batch_size
        # keeps the compiled trace re-usable.
        if chunk_size is None or chunk_size >= num_graphs:
            return self.sampler.sample(
                model=self.model,
                noise_process=self.noise_process,
                num_graphs=num_graphs,
                num_nodes=num_nodes_arg,
                device=device,
                collector=collector,
                chain_recorder=chain_recorder,
            )

        graph_data_list: list[GraphState] = []
        n_remaining = num_graphs
        nodes_offset = 0
        while n_remaining > 0:
            this_chunk = min(chunk_size, n_remaining)
            chunk_num_nodes = (
                num_nodes_arg
                if isinstance(num_nodes_arg, int)
                else num_nodes_arg[nodes_offset : nodes_offset + this_chunk]
            )
            chunk = self.sampler.sample(
                model=self.model,
                noise_process=self.noise_process,
                num_graphs=this_chunk,
                num_nodes=chunk_num_nodes,
                device=device,
                collector=collector,
                chain_recorder=chain_recorder,
            )
            graph_data_list.extend(chunk)
            n_remaining -= this_chunk
            nodes_offset += this_chunk
        return graph_data_list
