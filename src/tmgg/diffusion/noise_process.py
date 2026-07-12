"""Unified noise process hierarchy for graph diffusion.

``NoiseProcess`` is the timestep-indexed corruption interface shared by
continuous and categorical graph diffusion. It exposes three core operations:

- ``sample_prior(node_mask)``
- ``forward_sample(x_0, t)``
- ``posterior_sample(z_t, x0_param, t, s)``

Concrete processes still carry some legacy helpers while later refactor chunks
remove the remaining transition-model and exact-density compatibility surface.
All process classes are ``nn.Module`` instances so schedules and any other
stateful subcomponents follow device placement automatically.
"""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Batch

from tmgg.data.datasets.graph_data_fields import (
    GRAPHDATA_LOSS_KIND,
    FieldName,
)
from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
    GraphState,
    state_to_dense_sample,
)
from tmgg.data.utils.edge_counts import (
    count_edge_classes_sparse,
    count_node_classes_sparse,
)
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.utils.noising.noise import NoiseDefinition

from .diffusion_math import sum_except_batch
from .diffusion_sampling import (
    _normalise_unnormalised_posterior,
    compute_posterior_distribution,
    compute_posterior_distribution_per_x0,
    sample_discrete_feature_noise,
    sample_discrete_features,
)


@dataclass(frozen=True)
class NoisedBatch:
    """Forward-noise output: corrupted graph plus per-sample schedule scalars.

    Mirrors upstream DiGress's ``apply_noise`` return dict
    (``diffusion_model_discrete.py:407-442``) field by field, with the
    noised tensors packaged as a typed :class:`GraphData` rather than
    four loose ``(X_t, E_t, y_t, node_mask)`` entries. The schedule
    scalars carry shape ``(B, 1)`` so VLB-path expressions translate
    one-to-one between this codebase and the upstream reference.

    Attributes
    ----------
    z_t
        Noised graph state at integer timestep ``t_int``.
    t_int
        Integer timesteps, shape ``(B, 1)`` with dtype ``torch.long``.
    t
        Normalised time ``t_int / T``, shape ``(B, 1)``.
    beta_t
        Schedule beta at ``t_int``, shape ``(B, 1)``.
    alpha_t_bar
        Cumulative alpha-bar at ``t_int``, shape ``(B, 1)``.
    alpha_s_bar
        Cumulative alpha-bar at ``s = t_int - 1``, shape ``(B, 1)``.

    Notes
    -----
    Composite processes share a single schedule across sub-processes, so
    the schedule scalars are well-defined for ``CompositeNoiseProcess``
    too; the composite picks any sub-process's schedule (they must agree
    by the construction-time timestep invariant).
    """

    z_t: GraphData
    t_int: Tensor
    t: Tensor
    beta_t: Tensor
    alpha_t_bar: Tensor
    alpha_s_bar: Tensor


def _build_noised_batch(
    z_t: GraphData,
    t_int: Tensor,
    schedule: NoiseSchedule,
) -> NoisedBatch:
    """Bundle a noised graph with the schedule scalars at ``t_int``.

    Reshapes ``t_int`` to ``(B, 1)`` and pulls the matching ``beta_t``,
    ``alpha_t_bar`` and ``alpha_s_bar`` off the schedule. Mirrors the
    upstream layout where ``apply_noise`` returns the ``(B, 1)`` per-
    sample scalars alongside the noised state.
    """
    if t_int.dim() == 1:
        t_int_b1 = t_int.unsqueeze(-1).long()
    elif t_int.dim() == 2 and t_int.shape[-1] == 1:
        t_int_b1 = t_int.long()
    else:
        raise ValueError(
            f"_build_noised_batch: t_int must have shape (B,) or (B, 1), "
            f"got {tuple(t_int.shape)}"
        )

    timesteps = schedule.timesteps
    t_norm = t_int_b1.float() / timesteps
    beta_t = schedule(t_int=t_int_b1)
    alpha_t_bar = schedule.get_alpha_bar(t_int=t_int_b1)

    # Upstream parity (DiGress noise_schedule.py:75-79): when ``t == 0``,
    # ``s_int = -1`` and Python negative indexing returns ``alphas_bar[T]``.
    # Our ``NoiseSchedule._resolve_index`` rejects negative indices, so we
    # branch explicitly: per-element, fold ``s = -1`` into ``s = T``.
    # Currently no consumer reads ``noised.alpha_s_bar`` at ``t = 0``; if
    # this changes, the t=0 case must be handled explicitly. ``FAIL_STRICTLY``
    # converts the warning into a hard error to surface the call site.
    # Dev-only warning: the ``where`` below already handles ``t=0``
    # numerically. Guarded by ``__debug__`` so production runs
    # (Python -O / PYTHONOPTIMIZE=1) skip the per-step ``bool(.any())``
    # sync. ``FAIL_STRICTLY`` continues to crash in debug.
    if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
        if (t_int_b1 == 0).any():
            msg = (
                "_build_noised_batch: s_int=-1 at t=0; returning alpha_bar[T] per "
                "upstream DiGress's negative-index semantics (noise_schedule.py:75-79). "
                "Currently no consumer reads noised.alpha_s_bar at t=0; if this changes, "
                "the t=0 case must be handled explicitly. Set FAIL_STRICTLY=1 to crash "
                "on this code path instead of warning."
            )
            if os.environ.get("FAIL_STRICTLY"):
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # Per-element s_int: t-1 normally, T at t=0 (= -1 mod (T+1)).
    s_int_b1 = torch.where(
        t_int_b1 == 0,
        torch.full_like(t_int_b1, timesteps),
        t_int_b1 - 1,
    )
    alpha_s_bar = schedule.get_alpha_bar(t_int=s_int_b1)
    return NoisedBatch(
        z_t=z_t,
        t_int=t_int_b1,
        t=t_norm,
        beta_t=beta_t,
        alpha_t_bar=alpha_t_bar,
        alpha_s_bar=alpha_s_bar,
    )


def _continuous_edge_state(data: GraphData) -> Tensor:
    """Return a dense edge-valued tensor for continuous-process math.

    Continuous states read from ``E_feat``; clean binary-topology
    graphs lift through ``to_edge_scalar(source="class")`` so the
    forward and reverse process stay in dense edge-state space without
    binary-topology projections. Sparse / distribution carriers
    densify on the fly to a ``DenseGraphState`` view (the two dense
    types share fields RAM-for-RAM per spec §2.1).
    """
    if isinstance(data, GraphState):
        data = state_to_dense_sample(data)
    elif isinstance(data, GraphDistribution):
        dd = data.to_dense()
        data = DenseGraphState(
            num_nodes_per_graph=dd.num_nodes_per_graph,
            y=dd.y,
            X_class=dd.X_class,
            X_feat=dd.X_feat,
            E_class=dd.E_class,
            E_feat=dd.E_feat,
        )
    elif isinstance(data, DenseGraphDistribution):
        data = DenseGraphState(
            num_nodes_per_graph=data.num_nodes_per_graph,
            y=data.y,
            X_class=data.X_class,
            X_feat=data.X_feat,
            E_class=data.E_class,
            E_feat=data.E_feat,
        )
    if data.E_feat is not None:
        return data.to_edge_scalar(source="feat")
    return data.to_edge_scalar(source="class")


def _masked_graph_log_prob(sample: GraphData, probs: GraphData) -> Tensor:
    """Sum masked categorical log-probabilities per sample.

    ``sample`` is typically a one-hot graph state and ``probs`` carries
    the corresponding categorical probabilities over the same support.
    Reads ``X_class`` / ``E_class`` directly. Both inputs MUST carry
    populated ``X_class`` / ``E_class`` already (this helper does not
    synthesise) — the C_x / C_e values are recovered from the
    populated tensors and forwarded to the readers as a defensive
    contract check.
    """
    node_mask = sample.node_mask
    inv = ~node_mask
    inv_edge = inv.unsqueeze(1) | inv.unsqueeze(2)

    if sample.X_class is None or probs.X_class is None:
        raise ValueError(
            "_masked_graph_log_prob: both sample.X_class and probs.X_class "
            "must be populated; this helper does not synthesise."
        )
    if sample.E_class is None or probs.E_class is None:
        raise ValueError(
            "_masked_graph_log_prob: both sample.E_class and probs.E_class "
            "must be populated; this helper does not synthesise."
        )
    x_classes = sample.X_class.shape[-1]
    e_classes = sample.E_class.shape[-1]

    sample_x = _read_categorical_x(sample, x_classes=x_classes).clone()
    sample_e = _read_categorical_e(sample, e_classes=e_classes).clone()
    prob_x = _read_categorical_x(probs, x_classes=x_classes).clone()
    prob_e = _read_categorical_e(probs, e_classes=e_classes).clone()

    sample_x[inv] = 0.0
    sample_e[inv_edge] = 0.0
    prob_x[inv] = 1.0
    prob_e[inv_edge] = 1.0

    log_prob_x = sum_except_batch(sample_x * prob_x.clamp(min=1e-10).log())
    log_prob_e = sum_except_batch(sample_e * prob_e.clamp(min=1e-10).log())
    return log_prob_x + log_prob_e


def _normalise_counts_or_uniform(counts: Tensor) -> Tensor:
    """Return a PMF from counts, falling back to uniform on empty support."""
    if counts.numel() == 0:
        return counts
    total = counts.sum()
    if total <= 0:
        return torch.full_like(counts, 1.0 / counts.numel())
    return counts / total


def _gaussian_log_prob_sum(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    """Sum the per-element Gaussian log-density of ``x ~ N(mean, var)``.

    Preserves the batch axis and sums every remaining dimension; the
    caller typically composes this across declared fields for a
    per-sample log-likelihood.
    """
    var_c = var.clamp(min=1e-12)
    log_prob = -0.5 * (
        ((x - mean) ** 2) / var_c
        + var_c.log()
        + torch.log(torch.tensor(2.0 * torch.pi))
    )
    return sum_except_batch(log_prob)


def _read_feature_field(data: GraphData, field: FieldName) -> Tensor:
    """Read a continuous ``X_feat`` / ``E_feat`` field.

    For ``"E_feat"`` we return ``data.E_feat`` when present and
    otherwise lift ``E_class`` into a single-channel edge-scalar view
    so binary-topology batches carried through the categorical path
    stay usable by Gaussian processes declared on ``E_feat``. For
    ``"X_feat"`` the split field MUST be populated; there is no
    canonical lift from ``X_class``.

    Accepts any of the four concrete carriers (sparse / dense × state /
    distribution). Sparse carriers densify on the fly so the returned
    tensor matches the ``(B, n_max, ...)`` dense layout every consumer
    expects.
    """
    if isinstance(data, GraphState):
        data = state_to_dense_sample(data)
    elif isinstance(data, GraphDistribution):
        data = data.to_dense()
    if field == "E_feat":
        if data.E_feat is not None:
            return data.E_feat
        if data.E_class is None:
            raise ValueError(
                "_read_feature_field('E_feat'): both E_feat and E_class are "
                "None; cannot derive a continuous edge view."
            )
        return data.to_edge_scalar(source="class").unsqueeze(-1)
    if field == "X_feat":
        if data.X_feat is None:
            raise ValueError(
                "_read_feature_field('X_feat'): X_feat is not populated; "
                "continuous node features have no legacy fallback."
            )
        return data.X_feat
    raise ValueError(f"_read_feature_field does not support field {field!r}.")


def _gaussian_graphdata(
    data: GraphData, updates: dict[FieldName, Tensor]
) -> DenseGraphState:
    """Return ``data`` with the given Gaussian fields written.

    Writes each declared split field (``X_feat`` / ``E_feat``) onto a
    copy of ``data``. Non-Gaussian fields pass through unchanged.

    Accepts any of the four concrete carriers; sparse inputs densify
    and distribution-content reinterprets as ``DenseGraphState`` so
    callers can read ``.node_mask`` / call ``.mask()`` immediately.
    The two dense types share fields RAM-for-RAM (spec §2.1) so the
    reinterpretation is structurally a no-op.
    """
    if isinstance(data, GraphState):
        dense_state: DenseGraphState = state_to_dense_sample(data)
    elif isinstance(data, GraphDistribution):
        dd = data.to_dense()
        dense_state = DenseGraphState(
            num_nodes_per_graph=dd.num_nodes_per_graph,
            y=dd.y,
            X_class=dd.X_class,
            X_feat=dd.X_feat,
            E_class=dd.E_class,
            E_feat=dd.E_feat,
        )
    elif isinstance(data, DenseGraphDistribution):
        dense_state = DenseGraphState(
            num_nodes_per_graph=data.num_nodes_per_graph,
            y=data.y,
            X_class=data.X_class,
            X_feat=data.X_feat,
            E_class=data.E_class,
            E_feat=data.E_feat,
        )
    else:
        # Already DenseGraphState (or the abstract base, which we reject implicitly via the replace below).
        dense_state = data  # type: ignore[assignment]
    replace_kwargs: dict[str, Tensor] = {}
    for field, value in updates.items():
        if field == "E_feat":
            replace_kwargs["E_feat"] = value
        elif field == "X_feat":
            replace_kwargs["X_feat"] = value
        else:
            raise ValueError(f"_gaussian_graphdata: unsupported field {field!r}.")
    return dense_state.replace(**replace_kwargs)


def _read_categorical_x(data: GraphData, x_classes: int) -> Tensor:
    """Read ``X_class`` or synthesise a structure-only X tensor of width ``x_classes``.

    Per ``internal notes (not included)`` (architecture-internal concern): structure-only datasets
    emit ``X_class=None`` and architectures that need a per-node feature
    derive one from ``node_mask``. ``CategoricalNoiseProcess`` is part
    of the DiGress-family architecture, so it synthesises the
    degenerate encoding internally rather than forcing the data
    pipeline to carry it.

    Synthesis is delegated to
    :meth:`GraphData.synth_structure_only_x_class`; see
    ``internal notes (not included)`` for
    the regime table:

    * ``x_classes = 1`` is the canonical structure-only encoding
      (ones at valid nodes); the noise process is identity on X.
    * ``x_classes = 2`` is the legacy ``[no-node, node]`` encoding,
      retained for backward compat with existing C_x = 2 model presets.
    * ``x_classes >= 3`` raises — real categorical X must come from
      the dataset.

    ``x_classes`` is required (no default): every caller passes its
    authoritative C_x explicitly. See spec §5.2.

    Accepts any of the four concrete carriers (sparse / dense × state /
    distribution). Sparse carriers densify on the fly so structure-only
    synthesis can reach for ``data.node_mask`` (a dense-only cached
    property) and the returned tensor matches the ``(B, n_max, C_x)``
    dense layout every consumer of this helper expects.
    """
    if isinstance(data, GraphState):
        data = state_to_dense_sample(data)
    elif isinstance(data, GraphDistribution):
        data = data.to_dense()
    if data.X_class is not None:
        return data.X_class
    return DenseGraphState.synth_structure_only_x_class(data.node_mask, x_classes)


def _read_categorical_e(data: GraphData, e_classes: int) -> Tensor:
    """Read ``E_class`` or raise if unpopulated.

    Unlike X, edges are an adjacency property orthogonal to
    ``node_mask`` and synthesis from the mask alone is undefined for
    any C_e. This raises with a clear message naming the X-vs-E
    asymmetry. ``e_classes`` is required (no default) so the contract
    is explicit at every call site; see
    ``internal notes (not included)``.

    Accepts any of the four concrete carriers (sparse / dense × state /
    distribution). Sparse carriers densify on the fly so the returned
    tensor matches the ``(B, n_max, n_max, C_e)`` dense layout every
    consumer of this helper expects.
    """
    if isinstance(data, GraphState):
        data = state_to_dense_sample(data)
    elif isinstance(data, GraphDistribution):
        data = data.to_dense()
    if data.E_class is None:
        raise ValueError(
            "_read_categorical_e: data.E_class is None; categorical edge "
            "features must be populated. Edges are an adjacency property "
            "orthogonal to node_mask, so synthesis from the mask alone is "
            f"undefined (C_e={e_classes} requested). See "
            "2026-04-27-x-class-synth-unification-spec §5.2 for the "
            "X-vs-E asymmetry rationale."
        )
    return data.E_class


def _categorical_graphdata(
    x: Tensor, e: Tensor, *, y: Tensor, node_mask: Tensor
) -> DenseGraphState:
    """Assemble a categorical ``DenseGraphState`` from node/edge class tensors.

    ``node_mask`` is accepted for backwards-compatibility at the call site;
    the new ``DenseGraphState`` derives ``node_mask`` from
    ``num_nodes_per_graph`` so we convert here.
    """
    return DenseGraphState(
        num_nodes_per_graph=node_mask.long().sum(dim=-1),
        y=y,
        X_class=x,
        E_class=e,
    )


def _mix_with_limit(features: Tensor, alpha: Tensor | float, limit: Tensor) -> Tensor:
    """Interpolate one-hot features with a stationary categorical PMF."""
    if isinstance(alpha, int | float):
        alpha = torch.tensor([alpha], device=features.device, dtype=torch.float32)
    alpha = alpha.to(device=features.device, dtype=torch.float32)
    alpha = alpha.view(-1, *([1] * (features.dim() - 1)))
    limit = limit.to(device=features.device, dtype=torch.float32)
    limit = limit.view(*([1] * (features.dim() - 1)), limit.numel())
    return alpha * features.float() + (1.0 - alpha) * limit


def _categorical_kernel(identity_weight: Tensor, limit: Tensor) -> Tensor:
    """Build batched categorical kernels from identity weight and a PMF."""
    if limit.numel() == 0:
        bs = int(identity_weight.reshape(-1).shape[0])
        return limit.new_zeros((bs, 0, 0))
    identity_weight = identity_weight.to(device=limit.device, dtype=torch.float32)
    identity_weight = identity_weight.reshape(-1, 1, 1)
    classes = limit.numel()
    eye = torch.eye(classes, device=limit.device, dtype=torch.float32).unsqueeze(0)
    stationary_rows = (
        limit.to(dtype=torch.float32)
        .view(1, 1, classes)
        .expand(identity_weight.shape[0], classes, classes)
    )
    return identity_weight * eye + (1.0 - identity_weight) * stationary_rows


def _assert_row_stochastic(t: Tensor, name: str, atol: float = 1e-5) -> None:
    """Assert ``t`` is row-stochastic along the last axis.

    Upstream parity (``diffusion_model_discrete.py:425-426`` asserts
    ``Q̄_t`` row-sums equal 1 inside ``apply_noise``). Our
    :func:`_mix_with_limit` closed form preserves the invariant
    algebraically; this runtime check guards against future refactors
    silently breaking it. Skips empty tensors (``numel == 0``), which
    are valid for empty-class edge fields.

    Body wrapped in ``if __debug__:`` so production runs
    (Python -O / PYTHONOPTIMIZE=1) skip the ``.item()`` + ``allclose``
    syncs. Hit 6× per call inside posterior helpers; per the 2026-04-28
    sync review (region 02) these are validation-only path, but the
    host syncs still cost during val.
    """
    if __debug__:
        if t.numel() == 0:
            return
        sums = t.sum(dim=-1)
        deviation = (sums - 1.0).abs().max().item()
        assert torch.allclose(sums, torch.ones_like(sums), atol=atol), (
            f"{name} not row-stochastic: max|row_sum - 1| = {deviation:.3e}, "
            f"shape={tuple(t.shape)}"
        )


class NoiseProcess(ABC, nn.Module):
    """Abstract base for timestep-indexed graph corruption processes.

    Subclasses declare the set of ``GraphData`` fields they noise via the
    class-level ``fields`` attribute; see
    ``internal notes (not included)``.
    """

    #: Declared set of GraphData fields the process reads and writes. Every
    #: concrete subclass MUST set this to a non-empty subset of
    #: ``FIELD_NAMES``. ``CompositeNoiseProcess`` computes the union of its
    #: children's field sets.
    fields: frozenset[FieldName]

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def timesteps(self) -> int:
        """Number of discrete diffusion steps owned by the process."""
        ...

    def initialize_from_data(self, raw_pyg_loader: Iterable[Batch]) -> None:
        """Initialise any data-dependent state from the *raw PyG* training loader.

        Subclasses that need data-dependent state should iterate the
        loader of PyG ``Batch`` objects (yielded by
        ``BaseGraphDataModule.train_dataloader_raw_pyg``) rather than
        the dense ``GraphData`` view, so the count helpers in
        :mod:`tmgg.data.utils.edge_counts` can be reused as direct
        ports of upstream DiGress (parity #13 / D-3).
        """
        _ = raw_pyg_loader

    def needs_data_initialization(self) -> bool:
        """Whether the process requires a dataloader-backed setup phase."""
        return False

    def is_initialized(self) -> bool:
        """Whether any required data-dependent state is already available."""
        return True

    @abstractmethod
    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Return the model-conditioning vector for process state ``t``."""
        ...

    def model_output_to_posterior_parameter(self, model_output: GraphData) -> GraphData:
        """Convert raw model output into the reverse-process parameterisation."""
        return model_output

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        """Decode the final latent state into the public sampling output."""
        return z_0

    @abstractmethod
    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Sample the process prior at timestep ``T``."""
        ...

    @abstractmethod
    def forward_sample(self, x_0: GraphData, t: Tensor) -> NoisedBatch:
        """Sample ``q(z_t | x_0)`` at integer timesteps ``t``.

        Returns a :class:`NoisedBatch` bundling the noised state with the
        schedule scalars at ``t_int`` (matching upstream DiGress's
        ``apply_noise`` dict).
        """
        ...

    @abstractmethod
    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample ``q(z_s | z_t, x_0)`` or its model-parameterised analogue."""
        ...

    def posterior_sample_from_model_output(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Reverse-sampler entry point used by :class:`Sampler.sample`.

        Template method: the default implementation routes straight to
        :meth:`posterior_sample`, which covers every continuous and
        mixed-composition process. :class:`CategoricalNoiseProcess`
        overrides this hook to substitute the upstream-DiGress
        per-class-marginalised posterior
        (:meth:`CategoricalNoiseProcess.posterior_sample_marginalised`)
        so the sampler stays composition-agnostic. The sampler never
        branches on ``isinstance`` checks; every subclass is
        responsible for returning a fresh ``GraphData`` with the
        declared fields updated and non-declared fields preserved.
        """
        return self.posterior_sample(z_t, x0_param, t, s)


class ExactDensityNoiseProcess(NoiseProcess, ABC):
    """Noise process with tractable forward, posterior, and prior densities."""

    @abstractmethod
    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor:
        """Log-density of the forward process at timestep ``t``."""
        ...

    @abstractmethod
    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Log-density of the reverse posterior at timestep pair ``(t, s)``."""
        ...

    @abstractmethod
    def prior_log_prob(self, x: GraphData) -> Tensor:
        """Log-density of the prior distribution."""
        ...


_GAUSSIAN_ALLOWED_FIELDS: frozenset[FieldName] = frozenset({"X_feat", "E_feat"})


class GaussianNoiseProcess(ExactDensityNoiseProcess):
    """Per-field DDPM Gaussian diffusion on continuous graph features.

    Applies the reference Gaussian forward process ``z_t = sqrt(alpha_bar_t) *
    x_0 + sqrt(1 - alpha_bar_t) * eps`` to each field in :attr:`fields`.
    Edge-field outputs are symmetrised and zero-diagonalised. Non-declared
    fields pass through unchanged on the returned ``GraphData``.

    The wrapped ``NoiseDefinition`` is retained for the Wave 2 transition:
    :meth:`sample_at_level` and :meth:`sample_prior` still delegate to it
    so single-step denoising flows and the reverse sampler see unchanged
    semantics until Wave 5 rewrites those call sites.

    Parameters
    ----------
    definition
        Existing ``NoiseDefinition`` instance. Used by
        :meth:`sample_at_level` (single-step denoising) and preserved for
        backward compatibility; :meth:`forward_sample`, :meth:`posterior_sample`
        and the log-prob methods implement pure DDPM regardless.
    schedule
        ``NoiseSchedule`` owning ``alpha_bar`` at every integer timestep.
    fields
        Optional override of the declared field set. ``None`` keeps the
        default ``frozenset({"E_feat"})``; the override must be a
        non-empty subset of ``{"X_feat", "E_feat"}``.

    Notes
    -----
    Follows the DDPM parametrisation of Ho et al., NeurIPS 2020
    (arXiv:2006.11239), applied independently per declared field. See
    ``internal notes (not included)``.
    """

    #: Default Gaussian-process fields: single-channel edge weights. Joint
    #: X_feat + E_feat diffusion is opt-in via the constructor kwarg.
    fields: frozenset[FieldName] = frozenset({"E_feat"})

    def __init__(
        self,
        definition: NoiseDefinition,
        schedule: NoiseSchedule,
        *,
        fields: frozenset[FieldName] | None = None,
    ) -> None:
        super().__init__()
        self.definition = definition
        self.noise_schedule = schedule  # nn.Module — auto-registered
        if fields is not None:
            if len(fields) == 0:
                raise ValueError(
                    f"GaussianNoiseProcess.fields must be non-empty; got {fields!r}."
                )
            if not fields.issubset(_GAUSSIAN_ALLOWED_FIELDS):
                raise ValueError(
                    "GaussianNoiseProcess.fields must be a subset of "
                    f"{set(_GAUSSIAN_ALLOWED_FIELDS)!r}; got {set(fields)!r}."
                )
            # Instance-level override: shadows the class-level default while
            # leaving the declared type stable.
            self.fields = fields

    @property
    def timesteps(self) -> int:
        """Number of discrete diffusion steps owned by the schedule."""
        return self.noise_schedule.timesteps

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Return the default model-conditioning vector for timestep ``t``."""
        return t.float() / self.timesteps

    def model_output_to_posterior_parameter(self, model_output: GraphData) -> GraphData:
        """Continuous reverse diffusion uses the model output directly."""
        return model_output

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        """Decode the final dense edge state into a binary graph."""
        edge_state = z_0.to_edge_scalar(source="feat")
        adj_final = (edge_state > 0.5).float()
        adj_final = (adj_final + adj_final.transpose(1, 2)).clamp(max=1.0)

        n_max = adj_final.shape[1]
        diag_idx = torch.arange(n_max, device=adj_final.device)
        adj_final[:, diag_idx, diag_idx] = 0.0

        mask_2d = z_0.node_mask.unsqueeze(-1) * z_0.node_mask.unsqueeze(-2)
        adj_final = adj_final * mask_2d.float()

        decoded = DenseGraphState.from_edge_scalar(
            adj_final, node_mask=z_0.node_mask, target="E_class"
        )
        return decoded.mask()

    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Sample a symmetric Gaussian prior in dense edge-state space."""
        if node_mask.dim() == 1:
            node_mask = node_mask.unsqueeze(0)

        bs, n = node_mask.shape
        edge_state = torch.randn(bs, n, n, device=node_mask.device)
        edge_state = torch.triu(edge_state, diagonal=1)
        edge_state = edge_state + edge_state.transpose(1, 2)
        mask_2d = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        edge_state = edge_state * mask_2d.float()
        return DenseGraphState.from_structure_only(node_mask, edge_state).mask()

    def forward_sample(self, x_0: GraphData, t: Tensor) -> NoisedBatch:
        """Sample ``q(z_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)``.

        For every field in :attr:`fields` draws fresh ``eps ~ N(0, I)``
        with the same shape and writes ``sqrt(alpha_bar_t) * x_0 +
        sqrt(1 - alpha_bar_t) * eps``. Edge-field outputs are
        symmetrised via ``(E + E.transpose(-2, -3)) / 2`` and
        diagonal-zeroed; non-declared fields pass through untouched.

        Returns a :class:`NoisedBatch` bundling the noised state with
        ``t_int``, ``t``, ``beta_t``, ``alpha_t_bar``, ``alpha_s_bar``
        for VLB-path callers (parity #17 / #18 / D-4).
        """
        alpha_bar = self.noise_schedule.get_alpha_bar(t_int=t)
        updates: dict[FieldName, Tensor] = {}
        for field in self.fields:
            x_val = _read_feature_field(x_0, field)
            sqrt_ab = self._broadcast_alpha(alpha_bar, x_val)
            noise = torch.randn_like(x_val)
            noised = torch.sqrt(sqrt_ab) * x_val + torch.sqrt(1.0 - sqrt_ab) * noise
            updates[field] = self._finalise_field(field, noised, x_0.node_mask)
        z_t = _gaussian_graphdata(x_0, updates).mask()
        return _build_noised_batch(z_t, t, self.noise_schedule)

    def sample_at_level(self, x_0: GraphData, level: Tensor | float) -> GraphData:
        """Sample the forward process at an explicit continuous noise level.

        Retained for single-step denoising experiments that operate
        outside the DDPM schedule. Uses the wrapped ``NoiseDefinition``;
        the ``fields`` attribute is not consulted here.
        """
        return self._apply_noise(x_0, level)

    def _schedule_to_level(self, t_int: Tensor) -> Tensor:
        """Convert timestep to noise level via ``1 - alpha_bar(t)``."""
        return self.noise_schedule.get_noise_level(t_int)

    def _apply_noise(self, data: GraphData, noise_level: Tensor | float) -> GraphData:
        """Apply continuous noise via the wrapped definition (legacy path)."""
        edge_state = _continuous_edge_state(data)
        noisy_edge_state = self.definition.add_noise(edge_state, noise_level)
        return DenseGraphState.from_structure_only(data.node_mask, noisy_edge_state)

    @staticmethod
    def _broadcast_alpha(alpha_bar: Tensor, like: Tensor) -> Tensor:
        """Broadcast a per-sample scalar schedule value to ``like``'s shape."""
        flat = alpha_bar.reshape(-1)
        return flat.view(-1, *([1] * (like.dim() - 1))).to(
            device=like.device, dtype=like.dtype
        )

    @staticmethod
    def _finalise_field(field: FieldName, value: Tensor, node_mask: Tensor) -> Tensor:
        """Apply edge symmetry + diagonal zeroing to ``E_feat`` outputs."""
        _ = node_mask  # masking is handled by ``GraphData.mask()`` downstream.
        if field == "E_feat":
            symmetrised = 0.5 * (value + value.transpose(-2, -3))
            n = symmetrised.shape[-2]
            diag_idx = torch.arange(n, device=symmetrised.device)
            symmetrised[..., diag_idx, diag_idx, :] = 0.0
            return symmetrised
        return value

    def _field_posterior_parameters(
        self,
        field: FieldName,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> dict[str, Tensor]:
        """Compute DDPM posterior mean and std for a single declared field.

        Returns tensors with the same shape as the underlying field value.
        """
        x0_val = _read_feature_field(x0_param, field)
        zt_val = _read_feature_field(z_t, field)

        alpha_bar_t = self._broadcast_alpha(
            self.noise_schedule.get_alpha_bar(t_int=t), x0_val
        )
        alpha_bar_s = self._broadcast_alpha(
            self.noise_schedule.get_alpha_bar(t_int=s), x0_val
        )
        beta_t = 1.0 - alpha_bar_t / alpha_bar_s.clamp(min=1e-8)
        one_minus_alpha_bar_t = (1.0 - alpha_bar_t).clamp(min=1e-8)

        coeff_x0 = torch.sqrt(alpha_bar_s) * beta_t / one_minus_alpha_bar_t
        coeff_xt = (
            torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_s) / one_minus_alpha_bar_t
        )
        mean = coeff_x0 * x0_val + coeff_xt * zt_val

        raw_var = beta_t * (1.0 - alpha_bar_s) / one_minus_alpha_bar_t
        # Negative-variance sanity check: ``raw_var`` is mathematically
        # non-negative whenever ``alpha_bar`` is monotone in t (the
        # constructor invariant). The ``.any()`` host branch syncs the GPU
        # stream every call. Guarded by ``__debug__`` so production runs
        # (Python -O / PYTHONOPTIMIZE=1) skip the sync; ``var.clamp(min=1e-8)``
        # below already neutralises tiny numerical underflows.
        if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
            if (raw_var < -1e-6).any():
                raise RuntimeError(
                    "Negative posterior variance detected "
                    f"(min={raw_var.min().item():.6f}); the noise schedule is "
                    "non-monotonic between the requested timesteps."
                )
        var = raw_var.clamp(min=1e-8)
        std = torch.sqrt(var).expand_as(mean)
        return {"mean": mean, "std": std}

    def _posterior_parameters(
        self, z_t: GraphData, z_0: GraphData, t: Tensor, s: Tensor
    ) -> dict[str, Tensor]:
        """Compute the Gaussian posterior ``q(x_{t-1} | x_t, x_0)`` for the
        reverse diffusion step.

        Implements the closed-form posterior from Ho et al., "Denoising
        Diffusion Probabilistic Models", NeurIPS 2020, Eq. 6-7
        (arXiv:2006.11239).  With ``alpha_bar_t = alpha_bar(t)``, ``alpha_bar_s = alpha_bar(s)``,
        and ``beta_t = 1 - alpha_bar_t / alpha_bar_s``:

        .. math::

            \\tilde{\\mu}_t = \\frac{\\sqrt{\\bar\\alpha_s}\\,\\beta_t}
                             {1 - \\bar\\alpha_t}\\, x_0
                           + \\frac{\\sqrt{1 - \\beta_t}\\,(1 - \\bar\\alpha_s)}
                             {1 - \\bar\\alpha_t}\\, x_t

            \\tilde{\\beta}_t = \\frac{\\beta_t\\,(1 - \\bar\\alpha_s)}
                               {1 - \\bar\\alpha_t}

        Parameters
        ----------
        z_t
            Noisy graph at step *t*.
        z_0
            Clean or predicted clean graph.
        t
            Current integer timestep, shape ``(bs,)``.
        s
            Previous integer timestep (< t), shape ``(bs,)``.

        Returns
        -------
        dict[str, Tensor]
            ``"mean"`` and ``"std"`` tensors of shape ``(bs, n, n)``.
        """
        adj_t = _continuous_edge_state(z_t)
        adj_0 = _continuous_edge_state(z_0)

        # Cumulative signal-retention factors from the noise schedule.
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t).view(-1, 1, 1)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s).view(-1, 1, 1)

        # Single-step noise between s and t: beta_t = 1 - alpha_bar_t / alpha_bar_s
        beta_t = 1.0 - alpha_bar_t / alpha_bar_s.clamp(min=1e-8)

        # Denominators, clamped once for numerical safety.
        one_minus_alpha_bar_t = (1.0 - alpha_bar_t).clamp(min=1e-8)

        # Posterior mean (Ho et al. 2020, Eq. 7).
        coeff_x0 = torch.sqrt(alpha_bar_s) * beta_t / one_minus_alpha_bar_t
        coeff_xt = (
            torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_s) / one_minus_alpha_bar_t
        )
        mean = coeff_x0 * adj_0 + coeff_xt * adj_t

        # Posterior variance (Ho et al. 2020, Eq. 7).
        raw_var = beta_t * (1.0 - alpha_bar_s) / one_minus_alpha_bar_t
        # Negative-variance sanity check: same monotone-schedule invariant as
        # the Gaussian helper above. Guarded by ``__debug__`` so production
        # runs (Python -O / PYTHONOPTIMIZE=1) skip the per-step ``.any()``
        # sync; ``var.clamp(min=1e-8)`` below absorbs numerical underflow.
        if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
            if (raw_var < -1e-6).any():
                raise RuntimeError(
                    f"Negative posterior variance detected "
                    f"(min={raw_var.min().item():.6f}). "
                    f"This indicates alpha_bar_s > alpha_bar_t, i.e. the noise "
                    f"schedule is non-monotonic between the requested timesteps."
                )
        var = raw_var.clamp(min=1e-8)
        std = torch.sqrt(var).expand_as(mean)

        return {"mean": mean, "std": std}

    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample ``q(z_s | z_t, x_0)`` under the DDPM closed-form posterior.

        Per declared field draws ``sample = mean + std * eps`` with
        ``eps ~ N(0, I)``; falls back to the mean when ``s <= 0`` so the
        final reverse step is deterministic. Edge-field outputs are
        symmetrised and diagonal-zeroed.

        Accepts any concrete carrier for ``z_t`` and ``x0_param``;
        sparse inputs densify on the fly so ``z_t.node_mask`` is valid.
        """
        if isinstance(z_t, GraphState):
            z_t = state_to_dense_sample(z_t)
        elif isinstance(z_t, GraphDistribution):
            z_t = z_t.to_dense()
        if isinstance(x0_param, GraphState):
            x0_param = state_to_dense_sample(x0_param)
        elif isinstance(x0_param, GraphDistribution):
            x0_param = x0_param.to_dense()
        if s.dim() == 0:
            s = s.unsqueeze(0)
        updates: dict[FieldName, Tensor] = {}
        for field in self.fields:
            params = self._field_posterior_parameters(field, z_t, x0_param, t, s)
            mean = params["mean"]
            std = params["std"]
            noise = torch.randn_like(mean)
            sample = mean + std * noise
            no_noise_mask = (s <= 0).view(-1, *([1] * (mean.dim() - 1)))
            sample = torch.where(no_noise_mask, mean, sample)
            updates[field] = self._finalise_field(field, sample, z_t.node_mask)
        return _gaussian_graphdata(z_t, updates).mask()

    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor:
        """Return ``log q(z_t | x_0)`` summed per-sample across declared fields.

        Evaluates the Gaussian forward marginal density
        ``N(z_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`` at
        ``x_t``.
        """
        alpha_bar = self.noise_schedule.get_alpha_bar(t_int=t)
        total: Tensor | None = None
        for field in self.fields:
            x0_val = _read_feature_field(x_0, field)
            xt_val = _read_feature_field(x_t, field)
            ab = self._broadcast_alpha(alpha_bar, x0_val)
            mean = torch.sqrt(ab) * x0_val
            var = (1.0 - ab).clamp(min=1e-8).expand_as(mean)
            contrib = _gaussian_log_prob_sum(xt_val, mean, var)
            total = contrib if total is None else total + contrib
        if total is None:  # pragma: no cover - fields is non-empty by invariant
            raise RuntimeError("GaussianNoiseProcess.fields is empty.")
        return total

    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Return ``log q(z_s | z_t, x_0)`` summed across declared fields."""
        if s.dim() == 0:
            s = s.unsqueeze(0)
        total: Tensor | None = None
        for field in self.fields:
            params = self._field_posterior_parameters(field, z_t, x0_param, t, s)
            xs_val = _read_feature_field(x_s, field)
            mean = params["mean"]
            var = params["std"].pow(2)
            contrib = _gaussian_log_prob_sum(xs_val, mean, var)
            total = contrib if total is None else total + contrib
        if total is None:  # pragma: no cover - fields is non-empty by invariant
            raise RuntimeError("GaussianNoiseProcess.fields is empty.")
        return total

    def prior_log_prob(self, x: GraphData) -> Tensor:
        """Return ``log N(0, I)`` per-sample summed across declared fields."""
        total: Tensor | None = None
        for field in self.fields:
            x_val = _read_feature_field(x, field)
            mean = torch.zeros_like(x_val)
            var = torch.ones_like(x_val)
            contrib = _gaussian_log_prob_sum(x_val, mean, var)
            total = contrib if total is None else total + contrib
        if total is None:  # pragma: no cover - fields is non-empty by invariant
            raise RuntimeError("GaussianNoiseProcess.fields is empty.")
        return total

    # -- 4-type GraphData grid: state-content forward sampling ---------------

    def sample_dense(self, z_0: DenseGraphState, t: Tensor) -> DenseGraphDistribution:
        """DDPM forward sample on a dense state, returned as a distribution.

        Mirrors :meth:`forward_sample`'s math but stops short of the
        :class:`NoisedBatch` bundle and re-types the output as
        :class:`DenseGraphDistribution`. Gaussian forward output has
        continuous mass at every position, so the distribution carrier is
        the right shape for downstream loss helpers; callers that want a
        state-shaped one (e.g., binarised topology) should follow with
        ``.argmax()`` or ``.sample()`` per their pipeline.

        Per declared field draws fresh ``eps ~ N(0, I)`` and writes
        ``sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps``. Edge
        outputs are symmetrised and diagonal-zeroed.
        """
        alpha_bar = self.noise_schedule.get_alpha_bar(t_int=t)
        updates: dict[FieldName, Tensor] = {}
        for field in self.fields:
            x_val = _read_feature_field(z_0, field)
            sqrt_ab = self._broadcast_alpha(alpha_bar, x_val)
            noise = torch.randn_like(x_val)
            noised = torch.sqrt(sqrt_ab) * x_val + torch.sqrt(1.0 - sqrt_ab) * noise
            updates[field] = self._finalise_field(field, noised, z_0.node_mask)
        # Re-use the dense-state plumbing to stitch updates back, then mask;
        # finally re-type as a distribution-carrier (no math change, just the
        # static type tag the caller sees).
        z_t_state = _gaussian_graphdata(z_0, updates).mask()
        return DenseGraphDistribution(
            num_nodes_per_graph=z_t_state.num_nodes_per_graph,
            y=z_t_state.y,
            X_class=z_t_state.X_class,
            X_feat=z_t_state.X_feat,
            E_class=z_t_state.E_class,
            E_feat=z_t_state.E_feat,
        )

    def sample(self, z_0: GraphState, t: Tensor) -> GraphDistribution:
        """Sparse-path forward sample, returning a sparse distribution.

        Routes via dense conversion: ``state_to_dense_sample → sample_dense
        → to_sparse``. The sparse output covers the complete off-diagonal
        edge_index per :class:`GraphDistribution`'s invariant, so every
        position carries the (Gaussian-perturbed) continuous content.
        """
        dense_state = state_to_dense_sample(z_0)
        dense_dist = self.sample_dense(dense_state, t)
        return dense_dist.to_sparse()


class CategoricalNoiseProcess(ExactDensityNoiseProcess):
    """Scheduled categorical diffusion over one-hot graph states.

    The process owns only two public configuration choices:

    - the timestep schedule
    - the stationary categorical distribution mode

    ``limit_distribution`` controls the stationary class PMF that each
    position drifts toward as ``t -> T``.

    uniform : each class is equally likely; PMFs are fixed at
        construction.
    empirical_marginal : defers PMF construction until
        :meth:`initialize_from_data` can count class frequencies from
        the train loader.
    absorbing : every node/edge collapses to a fixed class as ``t -> T``.
        ``Q_t = (1 - beta_t) * I + beta_t * u @ 1^T`` where ``u`` is a
        one-hot indicator on the absorbing class (parameterised via
        ``absorbing_class_x`` / ``absorbing_class_e``; default class 0
        maps to the "no edge"/"no node" convention). The existing
        :func:`_mix_with_limit` closed form ``alpha_bar * x +
        (1 - alpha_bar) * pi`` already evaluates the absorbing
        transition correctly once ``pi`` is one-hot, so no new math
        path is needed beyond registering a one-hot indicator as the
        stationary PMF at construction.

        Implementation note (2026-04-22): upstream DiGress's
        ``AbsorbingStateTransition``
        (``src/diffusion/noise_schedule.py:190-223``) has a copy-paste
        bug at lines 200, 203 where ``self.u_e[:, :, abs_state] = 1``
        is written twice — the second occurrence should target
        ``self.u_y``. The class is never instantiated by any shipped
        upstream config so the bug never fires, but our
        re-implementation writes ``u_x``, ``u_e`` and (implicitly) the
        ``y`` indicator as separate one-hot indicators correctly.
    """

    #: Categorical diffusion always acts jointly on the node-class and
    #: edge-class streams of a DiGress-style batch.
    fields: frozenset[FieldName] = frozenset({"X_class", "E_class"})

    _limit_x: Tensor | None
    _limit_e: Tensor | None
    _limit_y: Tensor | None

    def __init__(
        self,
        schedule: NoiseSchedule,
        x_classes: int,
        e_classes: int,
        y_classes: int = 0,
        limit_distribution: Literal[
            "uniform", "empirical_marginal", "absorbing"
        ] = "uniform",
        absorbing_class_x: int = 0,
        absorbing_class_e: int = 0,
        zero_floor: float = 1e-5,
    ) -> None:
        super().__init__()

        self.noise_schedule = schedule  # nn.Module — auto-registered
        self.x_classes = x_classes
        self.e_classes = e_classes
        self.y_classes = y_classes
        self.limit_distribution = limit_distribution
        self.absorbing_class_x = absorbing_class_x
        self.absorbing_class_e = absorbing_class_e
        # Per-row uniform floor for the marginalised reverse posterior.
        # Matches upstream DiGress's ``unnormalized_prob[sum == 0] = 1e-5``
        # in ``diffusion_model_discrete.py:629-639`` (parity #28 / D-5).
        # Flowed into ``_posterior_probabilities_marginalised`` and the
        # downstream sampling helper through a single primitive
        # (:func:`_normalise_unnormalised_posterior`).
        self.zero_floor = zero_floor
        self.register_buffer("_limit_x", None)
        self.register_buffer("_limit_e", None)
        self.register_buffer("_limit_y", None)

        if limit_distribution == "uniform":
            self._set_stationary_distribution(
                x_probs=(
                    torch.full((x_classes,), 1.0 / x_classes)
                    if x_classes > 0
                    else torch.zeros(0)
                ),
                e_probs=(
                    torch.full((e_classes,), 1.0 / e_classes)
                    if e_classes > 0
                    else torch.zeros(0)
                ),
                y_probs=(
                    torch.full((y_classes,), 1.0 / y_classes)
                    if y_classes > 0
                    else torch.zeros(0)
                ),
            )
        elif limit_distribution == "absorbing":
            # pi is a one-hot indicator on the configured absorbing
            # class; Q_t = (1 - beta_t) * I + beta_t * pi @ 1^T
            # collapses every position onto that class as t -> T. The
            # existing _mix_with_limit closed form handles the math;
            # we just need a one-hot stationary PMF.
            if x_classes > 0 and not 0 <= absorbing_class_x < x_classes:
                raise ValueError(
                    f"absorbing_class_x={absorbing_class_x} is out of range "
                    f"for x_classes={x_classes}"
                )
            if e_classes > 0 and not 0 <= absorbing_class_e < e_classes:
                raise ValueError(
                    f"absorbing_class_e={absorbing_class_e} is out of range "
                    f"for e_classes={e_classes}"
                )
            x_probs = torch.zeros(x_classes)
            if x_classes > 0:
                x_probs[absorbing_class_x] = 1.0
            e_probs = torch.zeros(e_classes)
            if e_classes > 0:
                e_probs[absorbing_class_e] = 1.0
            # y uses class 0 as the absorbing default since no separate
            # knob is exposed; upstream never exercises y for categorical
            # graph datasets (parity #12 / D-12).
            y_probs = torch.zeros(y_classes)
            if y_classes > 0:
                y_probs[0] = 1.0
            self._set_stationary_distribution(
                x_probs=x_probs, e_probs=e_probs, y_probs=y_probs
            )
        elif limit_distribution != "empirical_marginal":
            raise ValueError(
                "limit_distribution must be 'uniform', 'empirical_marginal', "
                f"or 'absorbing', got {limit_distribution!r}"
            )

    @property
    def timesteps(self) -> int:
        """Number of discrete diffusion steps owned by the schedule."""
        return self.noise_schedule.timesteps

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Return the default model-conditioning vector for timestep ``t``."""
        return t.float() / self.timesteps

    def model_output_to_posterior_parameter(self, model_output: GraphData) -> GraphData:
        """Convert model logits into categorical probabilities.

        Accepts any of the four concrete carriers; sparse inputs densify
        on the fly so the trailing ``node_mask`` lookup (a dense-only
        cached property) and the assembled ``DenseGraphState`` output
        match the dense ``(B, n_max, ...)`` layout downstream callers
        expect.
        """
        if isinstance(model_output, GraphState):
            model_output = state_to_dense_sample(model_output)
        elif isinstance(model_output, GraphDistribution):
            model_output = model_output.to_dense()
        x_logits = _read_categorical_x(model_output, x_classes=self.x_classes)
        e_logits = _read_categorical_e(model_output, e_classes=self.e_classes)
        return _categorical_graphdata(
            F.softmax(x_logits, dim=-1),
            F.softmax(e_logits, dim=-1),
            y=model_output.y,
            node_mask=model_output.node_mask,
        )

    def finalize_sample(self, z_0: GraphData) -> GraphData:
        """Decode the final one-hot state and return it as a categorical graph.

        The returned ``GraphData`` keeps the one-hot ``X_class`` /
        ``E_class`` tensors so downstream consumers (notably the
        evaluator binarisation helpers) can read them directly. Any
        continuous feature fields pass through unchanged.
        """
        return z_0.mask()

    def initialize_from_data(self, raw_pyg_loader: Iterable[Batch]) -> None:
        """Estimate empirical stationary marginals from raw PyG batches.

        Direct port of upstream DiGress's
        :meth:`AbstractDatasetInfos.edge_counts` / ``node_types``
        (``digress-upstream-readonly/src/datasets/abstract_dataset.py:34-72``).
        We sum sparse counts per class across the training PyG loader,
        then normalise to a PMF (parity #13 / D-3). The hot training
        loop is untouched and still consumes dense ``GraphData``; only
        this preprocessing pass crosses into the sparse view.
        """
        if self.limit_distribution == "uniform" or self._limit_x is not None:
            return

        x_counts = torch.zeros(self.x_classes, dtype=torch.float64)
        e_counts = torch.zeros(self.e_classes, dtype=torch.float64)

        for pyg_batch in raw_pyg_loader:
            x_counts += count_node_classes_sparse(pyg_batch, self.x_classes)
            e_counts += count_edge_classes_sparse(pyg_batch, self.e_classes)

        x_marginals = _normalise_counts_or_uniform(x_counts).to(torch.float32)
        e_marginals = _normalise_counts_or_uniform(e_counts).to(torch.float32)
        self._set_stationary_distribution(
            x_probs=x_marginals,
            e_probs=e_marginals,
            y_probs=(
                torch.full((self.y_classes,), 1.0 / self.y_classes)
                if self.y_classes > 0
                else torch.zeros(0)
            ),
        )

    def needs_data_initialization(self) -> bool:
        """Empirical-marginal mode requires training-data statistics."""
        return self.limit_distribution == "empirical_marginal"

    def is_initialized(self) -> bool:
        """Return whether the stationary marginals are already available."""
        if not self.needs_data_initialization():
            return True
        return self._limit_x is not None

    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Sample from the categorical limit distribution at timestep ``T``."""
        x_limit, e_limit, y_limit = self._stationary_distribution()
        return sample_discrete_feature_noise(x_limit, e_limit, node_mask, y_limit)

    def forward_sample(self, x_0: GraphData, t: Tensor) -> NoisedBatch:
        """Sample the forward categorical process at timestep ``t``.

        Returns a :class:`NoisedBatch` bundling the noised state with
        the schedule scalars at ``t_int`` (parity #17 / #18 / D-4).
        """
        alpha_bar = self._schedule_to_level(t)
        z_t = self._apply_noise(x_0, alpha_bar)
        return _build_noised_batch(z_t, t, self.noise_schedule)

    def _set_stationary_distribution(
        self,
        *,
        x_probs: Tensor,
        e_probs: Tensor,
        y_probs: Tensor,
    ) -> None:
        """Store stationary class PMFs as registered buffers."""
        target_device = self.noise_schedule.betas.device
        x_probs = x_probs.to(device=target_device, dtype=torch.float32)
        e_probs = e_probs.to(device=target_device, dtype=torch.float32)
        y_probs = y_probs.to(device=target_device, dtype=torch.float32)

        self._validate_stationary_distribution("x_probs", x_probs, self.x_classes)
        self._validate_stationary_distribution("e_probs", e_probs, self.e_classes)
        self._validate_stationary_distribution("y_probs", y_probs, self.y_classes)

        self._limit_x = x_probs
        self._limit_e = e_probs
        self._limit_y = y_probs

    @staticmethod
    def _validate_stationary_distribution(
        name: str, probs: Tensor, expected_classes: int
    ) -> None:
        """Validate a 1-D categorical PMF exactly once at write time."""
        if probs.dim() != 1 or probs.numel() != expected_classes:
            raise ValueError(
                f"{name} must have shape ({expected_classes},), "
                f"got {tuple(probs.shape)}"
            )
        if not torch.isfinite(probs).all():
            raise ValueError(f"{name} must be finite")
        if (probs < 0).any():
            raise ValueError(f"{name} must be non-negative")
        if probs.numel() > 0 and not torch.isclose(
            probs.sum(), torch.tensor(1.0, device=probs.device), atol=1e-5
        ):
            raise ValueError(f"{name} must sum to 1, got {probs.sum().item():.6f}")

    def _stationary_distribution(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return stationary PMFs or fail loudly if setup has not run."""
        if self._limit_x is None or self._limit_e is None or self._limit_y is None:
            raise RuntimeError(
                "Stationary categorical distribution not initialised. "
                "Call initialize_from_data() first."
            )
        return self._limit_x, self._limit_e, self._limit_y

    # -- NoiseProcess interface ----------------------------------------------

    def _schedule_to_level(self, t_int: Tensor) -> Tensor:
        """Convert timestep to alpha_bar via the schedule.

        For categorical noise the "noise level" is ``alpha_bar`` itself
        (not ``1 - alpha_bar``), since the transition matrices are
        parameterised by signal retention.
        """
        return self.noise_schedule.get_alpha_bar(t_int=t_int)

    def _apply_noise(self, data: GraphData, noise_level: Tensor | float) -> GraphData:
        """Apply categorical forward noise at the given alpha_bar level.

        Non-declared continuous fields (``X_feat`` / ``E_feat``) survive
        the call unchanged so composition with a Gaussian process on
        disjoint fields (Wave 2.4) does not drop their payload.

        Accepts sparse (``GraphState``) or dense (``DenseGraphState``)
        inputs; densifies at entry so the internal logic stays in the
        dense ``(B, n_max, ...)`` layout DiGress was written against.
        ``data.replace(X_class=..., E_class=...)`` and ``data.node_mask``
        below are both dense-only operations.
        """
        if isinstance(data, GraphState):
            data = state_to_dense_sample(data)
        x_limit, e_limit, _ = self._stationary_distribution()
        x_class = _read_categorical_x(data, x_classes=self.x_classes)
        e_class = _read_categorical_e(data, e_classes=self.e_classes)
        prob_X = _mix_with_limit(x_class, noise_level, x_limit)
        prob_E = _mix_with_limit(e_class, noise_level, e_limit)
        # Per-position PMFs are NOT row-stochastic at padding/diagonal rows
        # (E_class diagonal is zeroed by from_pyg_batch per upstream parity #4).
        # The row-stochastic invariant applies to the K×K kernel matrices in
        # _posterior_probabilities*, not to these (B, n, K) / (B, n, n, K) tensors.
        x_idx, e_idx = sample_discrete_features(prob_X, prob_E, data.node_mask)
        x_one_hot = F.one_hot(x_idx.long(), num_classes=self.x_classes).float()
        e_one_hot = F.one_hot(e_idx.long(), num_classes=self.e_classes).float()
        return data.replace(
            X_class=x_one_hot,
            E_class=e_one_hot,
        ).mask()

    def _posterior_probabilities(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Compute categorical posterior distributions for the reverse step.

        Uses the Bayes rule formulation from DiGress:
        ``p(z_s | z_t, z_0) = z_t @ Qt(t).T * z_0 @ Qsb(s) / (z_0 @ Qtb(t) @ z_t.T)``

        Parameters
        ----------
        z_t
            Noisy one-hot graph at timestep *t*.
        x0_param
            Clean graph or model-predicted clean probabilities.
        t
            Current integer timestep, shape ``(bs,)``.
        s
            Previous integer timestep (= t - 1), shape ``(bs,)``.

        Returns
        -------
        GraphData
            Posterior probability distributions with the same dense graph
            extents as ``z_t``.
        """
        # The Bayes-rule body below operates in dense layout (the (B, n, K)
        # / (B, n, n, K) tensors and ``z_t.node_mask`` are dense-only).
        # Sparse callers densify at entry; dense inputs pass through.
        if isinstance(z_t, GraphState):
            z_t = state_to_dense_sample(z_t)
        if isinstance(x0_param, GraphState):
            x0_param = state_to_dense_sample(x0_param)
        x_limit, e_limit, _ = self._stationary_distribution()

        beta_t = self.noise_schedule(t_int=t)  # (bs,)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s)
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)

        q_x = _categorical_kernel(1.0 - beta_t, x_limit)
        q_e = _categorical_kernel(1.0 - beta_t, e_limit)
        qsb_x = _categorical_kernel(alpha_bar_s, x_limit)
        qsb_e = _categorical_kernel(alpha_bar_s, e_limit)
        qtb_x = _categorical_kernel(alpha_bar_t, x_limit)
        qtb_e = _categorical_kernel(alpha_bar_t, e_limit)
        _assert_row_stochastic(q_x, "_posterior_probabilities.q_x")
        _assert_row_stochastic(q_e, "_posterior_probabilities.q_e")
        _assert_row_stochastic(qsb_x, "_posterior_probabilities.qsb_x")
        _assert_row_stochastic(qsb_e, "_posterior_probabilities.qsb_e")
        _assert_row_stochastic(qtb_x, "_posterior_probabilities.qtb_x")
        _assert_row_stochastic(qtb_e, "_posterior_probabilities.qtb_e")

        x0_x = _read_categorical_x(x0_param, x_classes=self.x_classes)
        x0_e = _read_categorical_e(x0_param, e_classes=self.e_classes)
        zt_x = _read_categorical_x(z_t, x_classes=self.x_classes)
        zt_e = _read_categorical_e(z_t, e_classes=self.e_classes)
        prob_X = compute_posterior_distribution(
            M=x0_x, M_t=zt_x, Qt_M=q_x, Qsb_M=qsb_x, Qtb_M=qtb_x, field="X_class"
        )
        prob_E = compute_posterior_distribution(
            M=x0_e, M_t=zt_e, Qt_M=q_e, Qsb_M=qsb_e, Qtb_M=qtb_e, field="E_class"
        )
        bs, n, de = zt_e.shape[0], zt_e.shape[1], zt_e.shape[-1]

        return _categorical_graphdata(
            prob_X,
            prob_E.reshape(bs, n, n, de),
            y=z_t.y,
            node_mask=z_t.node_mask,
        )

    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample the categorical reverse posterior into one-hot graph states.

        Uses the **direct** parameterisation: passes ``x0_param`` straight
        into the Bayes-rule posterior. Correct when ``x0_param`` is a
        true one-hot (training-time validation MC). For sampling from a
        soft prediction over ``x_0`` classes, use
        :meth:`posterior_sample_marginalised` to integrate over ``x_0``.

        Non-declared fields on ``z_t`` pass through unchanged so
        composition with a Gaussian process on disjoint fields (Wave
        2.4) preserves their payload.
        """
        posterior = self._posterior_probabilities(z_t, x0_param, t, s)
        x_idx, e_idx = sample_discrete_features(
            _read_categorical_x(posterior, x_classes=self.x_classes),
            _read_categorical_e(posterior, e_classes=self.e_classes),
            node_mask=z_t.node_mask,
        )
        x_one_hot = F.one_hot(x_idx.long(), num_classes=self.x_classes).float()
        e_one_hot = F.one_hot(e_idx.long(), num_classes=self.e_classes).float()
        return z_t.replace(
            X_class=x_one_hot,
            E_class=e_one_hot,
        ).mask()

    def _posterior_probabilities_marginalised(
        self,
        z_t: GraphData,
        x0_probs: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Marginalise the categorical reverse posterior over ``x_0`` classes.

        For each position computes
        ``p(z_s = k | z_t) = sum_c p(z_s = k | z_t, x_0 = c) * p(x_0 = c | z_t)``
        where the per-class posterior comes from
        :func:`compute_posterior_distribution_per_x0` and the prediction
        ``p(x_0 = c | z_t)`` comes from ``x0_probs`` (model-output
        softmax). Mirrors upstream DiGress's
        ``compute_batched_over0_posterior_distribution`` followed by a
        contraction over the predicted ``x_0`` distribution.

        Returns posterior probability tensors over ``z_s`` with the same
        graph extents as ``z_t``.
        """
        # The contraction below operates in dense layout (the (B, n, K)
        # / (B, n, n, K) tensors and ``z_t.node_mask`` are dense-only).
        # Sparse callers densify at entry; dense inputs pass through.
        if isinstance(z_t, GraphState):
            z_t = state_to_dense_sample(z_t)
        if isinstance(x0_probs, GraphState):
            x0_probs = state_to_dense_sample(x0_probs)
        x_limit, e_limit, _ = self._stationary_distribution()

        beta_t = self.noise_schedule(t_int=t)
        alpha_bar_s = self.noise_schedule.get_alpha_bar(t_int=s)
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)

        q_x = _categorical_kernel(1.0 - beta_t, x_limit)
        q_e = _categorical_kernel(1.0 - beta_t, e_limit)
        qsb_x = _categorical_kernel(alpha_bar_s, x_limit)
        qsb_e = _categorical_kernel(alpha_bar_s, e_limit)
        qtb_x = _categorical_kernel(alpha_bar_t, x_limit)
        qtb_e = _categorical_kernel(alpha_bar_t, e_limit)
        _assert_row_stochastic(q_x, "_posterior_probabilities_marginalised.q_x")
        _assert_row_stochastic(q_e, "_posterior_probabilities_marginalised.q_e")
        _assert_row_stochastic(qsb_x, "_posterior_probabilities_marginalised.qsb_x")
        _assert_row_stochastic(qsb_e, "_posterior_probabilities_marginalised.qsb_e")
        _assert_row_stochastic(qtb_x, "_posterior_probabilities_marginalised.qtb_x")
        _assert_row_stochastic(qtb_e, "_posterior_probabilities_marginalised.qtb_e")

        zt_x = _read_categorical_x(z_t, x_classes=self.x_classes)
        zt_e = _read_categorical_e(z_t, e_classes=self.e_classes)
        x0_x = _read_categorical_x(x0_probs, x_classes=self.x_classes)
        x0_e = _read_categorical_e(x0_probs, e_classes=self.e_classes)

        # Per-x0 posteriors: shape (bs, N, d_x0, d_z_s).
        per_x0_X = compute_posterior_distribution_per_x0(
            M_t=zt_x, Qt_M=q_x, Qsb_M=qsb_x, Qtb_M=qtb_x, field="X_class"
        )
        per_x0_E = compute_posterior_distribution_per_x0(
            M_t=zt_e, Qt_M=q_e, Qsb_M=qsb_e, Qtb_M=qtb_e, field="E_class"
        )

        # Contract over the x_0 axis weighted by the model's prediction.
        # x0_x is (bs, n, d_x0); flatten to (bs, N, d_x0) to align with the
        # per-x0 posterior tensor.
        pred_X_flat = x0_x.flatten(start_dim=1, end_dim=-2).to(torch.float32)
        pred_E_flat = x0_e.flatten(start_dim=1, end_dim=-2).to(torch.float32)
        unnorm_X = (per_x0_X * pred_X_flat.unsqueeze(-1)).sum(dim=2)  # (bs, n, d_z_s)
        unnorm_E_flat = (per_x0_E * pred_E_flat.unsqueeze(-1)).sum(
            dim=2
        )  # (bs, n*n, d_z_s)

        # Apply the upstream-DiGress row-sum-zero floor + renormalisation
        # (parity #28 / D-5 / W4-1). On healthy inputs the per-x0 normalised
        # contraction already sums to 1 per row, so the helper is mathematically
        # a no-op modulo float roundoff. On rows where the per-x0 floor (1e-6)
        # in compute_posterior_distribution_per_x0 has cancelled mass to zero,
        # the helper's row-sum floor (1e-5, matching upstream's
        # diffusion_model_discrete.py:631) writes a uniform PMF rather than
        # leaving NaN to surface inside torch.multinomial downstream.
        prob_X = _normalise_unnormalised_posterior(unnorm_X, zero_floor=self.zero_floor)
        prob_E_flat = _normalise_unnormalised_posterior(
            unnorm_E_flat, zero_floor=self.zero_floor
        )

        bs, n, _, de = zt_e.shape
        prob_E = prob_E_flat.reshape(bs, n, n, de)

        return _categorical_graphdata(
            prob_X,
            prob_E,
            y=z_t.y,
            node_mask=z_t.node_mask,
        )

    def posterior_sample_marginalised(
        self,
        z_t: GraphData,
        x0_probs: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Sample ``z_s`` from the per-class-marginalised reverse posterior.

        This is the upstream DiGress sampling path: the model emits a
        distribution over ``x_0`` classes, and the reverse step samples
        from ``sum_{c} p(z_s | z_t, x_0 = c) * p(x_0 = c | z_t)``
        rather than from ``p(z_s | z_t, x_0 = soft prediction)``. The
        two agree when the prediction is one-hot (converged model);
        early in training the marginalised form has no bias from the
        soft-x0-into-Bayes shortcut.

        Non-declared fields on ``z_t`` pass through unchanged so
        composition with a Gaussian process on disjoint fields (Wave
        2.4) preserves their payload.
        """
        posterior = self._posterior_probabilities_marginalised(z_t, x0_probs, t, s)
        x_idx, e_idx = sample_discrete_features(
            _read_categorical_x(posterior, x_classes=self.x_classes),
            _read_categorical_e(posterior, e_classes=self.e_classes),
            node_mask=z_t.node_mask,
        )
        x_one_hot = F.one_hot(x_idx.long(), num_classes=self.x_classes).float()
        e_one_hot = F.one_hot(e_idx.long(), num_classes=self.e_classes).float()
        return z_t.replace(
            X_class=x_one_hot,
            E_class=e_one_hot,
        ).mask()

    def posterior_sample_from_model_output(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Route the reverse-sampler hook to the marginalised posterior.

        Categorical diffusion samples from the per-class marginalised
        reverse posterior — ``sum_c p(z_s | z_t, x_0 = c) *
        p(x_0 = c | z_t)`` — to match upstream DiGress. The soft-x0
        Bayes shortcut (calling :meth:`posterior_sample` directly on
        the prediction) is a biased estimator early in training, so
        the reverse-sampler hook routes here instead.
        """
        return self.posterior_sample_marginalised(z_t, x0_param, t, s)

    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor:
        """Return ``log q(z_t | x_0)`` for sampled categorical graph states."""
        probs = self.forward_pmf(x_0, t, node_mask_template=x_t)
        return _masked_graph_log_prob(x_t, probs)

    def forward_pmf(
        self,
        x_0: GraphData,
        t: Tensor,
        *,
        node_mask_template: GraphData | None = None,
    ) -> GraphData:
        """Return the categorical forward marginal PMF ``q(z_t = . | x_0)``.

        Useful for analytic-KL VLB estimators: provides the per-position
        PMF without sampling. ``node_mask_template`` lets callers attach
        a mask that may differ from ``x_0.node_mask`` (e.g. when probing
        a freshly drawn ``z_t`` whose mask matches the original batch).

        Returns
        -------
        GraphData
            ``X``, ``E`` carry per-position categorical PMFs;
            ``node_mask`` mirrors the template (or ``x_0`` when no
            template is supplied).
        """
        alpha_bar_t = self.noise_schedule.get_alpha_bar(t_int=t)
        x_limit, e_limit, _ = self._stationary_distribution()
        prob_x = _mix_with_limit(
            _read_categorical_x(x_0, x_classes=self.x_classes), alpha_bar_t, x_limit
        )
        prob_e = _mix_with_limit(
            _read_categorical_e(x_0, e_classes=self.e_classes), alpha_bar_t, e_limit
        )
        # Per-position PMFs are NOT row-stochastic at padding/diagonal rows;
        # see comment in _apply_noise above. Kernel-level invariant lives in
        # _posterior_probabilities*.
        template = node_mask_template if node_mask_template is not None else x_0
        return _categorical_graphdata(
            prob_x, prob_e, y=template.y, node_mask=template.node_mask
        )

    def prior_pmf(self, node_mask: Tensor) -> GraphData:
        """Return the stationary categorical prior PMF tiled to ``node_mask``.

        Mirrors :meth:`prior_log_prob` but exposes the PMF directly so a
        caller can compute the analytic ``KL(q(z_T|x_0) || prior)``
        without going through a sampled state.
        """
        x_limit, e_limit, _ = self._stationary_distribution()
        bs, n = node_mask.shape
        prob_x = (
            x_limit.to(node_mask.device).view(1, 1, -1).expand(bs, n, -1).contiguous()
        )
        prob_e = (
            e_limit.to(node_mask.device)
            .view(1, 1, 1, -1)
            .expand(bs, n, n, -1)
            .contiguous()
        )
        return _categorical_graphdata(
            prob_x,
            prob_e,
            y=torch.zeros(bs, 0, device=node_mask.device),
            node_mask=node_mask,
        )

    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Return ``log q(z_s | z_t, x0_param)`` for sampled graph states."""
        posterior = self._posterior_probabilities(z_t, x0_param, t, s)
        return _masked_graph_log_prob(x_s, posterior)

    def prior_log_prob(self, x: GraphData) -> Tensor:
        """Return ``log p(z_T)`` under the stationary categorical prior."""
        x_limit, e_limit, _ = self._stationary_distribution()

        x_class = _read_categorical_x(x, x_classes=self.x_classes)
        e_class = _read_categorical_e(x, e_classes=self.e_classes)
        prob_x = x_limit.to(x_class.device).view(1, 1, -1).expand_as(x_class)
        prob_e = e_limit.to(e_class.device).view(1, 1, 1, -1).expand_as(e_class)

        probs = _categorical_graphdata(prob_x, prob_e, y=x.y, node_mask=x.node_mask)
        return _masked_graph_log_prob(x, probs)

    # -- 4-type GraphData grid: state-content forward sampling and marginal --

    def sample_dense(self, z_0: DenseGraphState, t: Tensor) -> DenseGraphState:
        """Categorical forward sample on a dense state.

        Returns a freshly drawn one-hot per position (state content) at
        timestep ``t``; this is the math :meth:`forward_sample` already
        uses internally, lifted out of the :class:`NoisedBatch` bundle and
        re-typed to the static :class:`DenseGraphState` cell.
        """
        alpha_bar = self._schedule_to_level(t)
        z_t_state = self._apply_noise(z_0, alpha_bar)
        # ``_apply_noise`` typed its argument as the legacy ``GraphData``
        # but operates on a dense state and returns the same layout; the
        # cast is safe by construction (we passed a ``DenseGraphState`` in,
        # ``data.replace(...).mask()`` preserves the type).
        assert isinstance(z_t_state, DenseGraphState)
        return z_t_state

    def sample(self, z_0: GraphState, t: Tensor) -> GraphState:
        """Sparse-path forward sample, returning a sparse one-hot state.

        Routes via dense conversion: ``state_to_dense_sample → sample_dense
        → to_sparse``. The dense intermediate is one-hot per position; the
        sparse output drops new "no-edge" positions (channel 0 argmax) per
        :meth:`DenseGraphState.to_sparse`'s default predicate.
        """
        dense_state = state_to_dense_sample(z_0)
        noised_dense = self.sample_dense(dense_state, t)
        return noised_dense.to_sparse()

    def q_marginal_dense(
        self, z_0: DenseGraphState, t: Tensor
    ) -> DenseGraphDistribution:
        """Per-position categorical marginal ``q_t(z_t | z_0) = z_0 Q̄_t``.

        Returns the distribution-content view of the forward marginal:
        each position's row is the mixture
        ``alpha_bar * z_0 + (1 - alpha_bar) * pi`` produced by
        :func:`_mix_with_limit`. No sampling is performed; ``q_marginal``
        is the loss target's distributional form, paired with
        :meth:`sample_dense` for the sample-based forward.
        """
        x_limit, e_limit, _ = self._stationary_distribution()
        alpha_bar = self._schedule_to_level(t)
        x_class = _read_categorical_x(z_0, x_classes=self.x_classes)
        e_class = _read_categorical_e(z_0, e_classes=self.e_classes)
        prob_X = _mix_with_limit(x_class, alpha_bar, x_limit)
        prob_E = _mix_with_limit(e_class, alpha_bar, e_limit)
        return DenseGraphDistribution(
            num_nodes_per_graph=z_0.num_nodes_per_graph,
            y=z_0.y,
            X_class=prob_X,
            X_feat=z_0.X_feat,
            E_class=prob_E,
            E_feat=z_0.E_feat,
        )

    def q_marginal(self, z_0: GraphState, t: Tensor) -> GraphDistribution:
        """Sparse-path forward marginal as a :class:`GraphDistribution`.

        Routes via dense conversion so the per-position categorical mixture
        materialises along the complete off-diagonal edge_index that
        :class:`GraphDistribution` requires. Continuous fields pass
        through unchanged.
        """
        dense_state = state_to_dense_sample(z_0)
        dense_dist = self.q_marginal_dense(dense_state, t)
        return dense_dist.to_sparse()


class CompositeNoiseProcess(NoiseProcess):
    """Compose several ``NoiseProcess`` instances with disjoint ``fields``.

    Every sub-process owns a strict subset of the declared ``FieldName``
    space; the composite enforces disjointness at construction time and
    forwards ``forward_sample`` / ``posterior_sample`` to each sub-process
    in list order. Disjoint fields make the order irrelevant for
    per-field semantics; the list order only fixes the concatenation
    order of ``process_state_condition_vector``.

    The composite is not an :class:`ExactDensityNoiseProcess` because a
    mixture of categorical and Gaussian log-densities does not share a
    common support; density methods delegate only when every sub-process
    is :class:`ExactDensityNoiseProcess` and raise
    :class:`NotImplementedError` otherwise.

    Parameters
    ----------
    processes
        Ordered sequence of concrete noise processes. Must be non-empty
        and have pairwise-disjoint ``fields``.

    Raises
    ------
    ValueError
        When ``processes`` is empty, when any sub-process has an empty
        ``fields`` set, or when two or more sub-processes claim the same
        field.
    """

    def __init__(self, processes: Sequence[NoiseProcess]) -> None:
        super().__init__()
        if len(processes) == 0:
            raise ValueError("CompositeNoiseProcess requires at least one sub-process.")

        self._check_disjoint_fields(processes)

        # ``nn.ModuleList`` keeps every sub-process on the composite's
        # device and surfaces them in the ``state_dict``; the parallel
        # Python list preserves ordering for iteration in a form
        # pyright can narrow without module-list indexing gymnastics.
        self.processes = nn.ModuleList(processes)
        self._process_list: list[NoiseProcess] = list(processes)
        self.fields = frozenset[FieldName]().union(*(p.fields for p in processes))

    @staticmethod
    def _check_disjoint_fields(processes: Sequence[NoiseProcess]) -> None:
        """Raise if any field is claimed by more than one sub-process."""
        seen: dict[FieldName, int] = {}
        conflicts: dict[FieldName, list[int]] = {}
        for idx, proc in enumerate(processes):
            if len(proc.fields) == 0:
                raise ValueError(
                    f"CompositeNoiseProcess sub-process at index {idx} "
                    f"({type(proc).__name__}) declares an empty fields set."
                )
            for field in proc.fields:
                if field in seen:
                    conflicts.setdefault(field, [seen[field]]).append(idx)
                else:
                    seen[field] = idx
        if conflicts:
            # Stable report order: sort by field name so tests can match
            # on the rendered string reliably.
            parts = [
                f"{field!r} (sub-processes {sorted(indices)})"
                for field, indices in sorted(conflicts.items())
            ]
            raise ValueError(
                "CompositeNoiseProcess sub-processes have overlapping fields: "
                + ", ".join(parts)
            )

    @property
    def timesteps(self) -> int:
        """Return the shared timestep count across sub-processes.

        Every sub-process must expose the same integer schedule length;
        this method raises if the sub-processes disagree so the caller
        cannot silently mix incompatible schedules.
        """
        steps = {p.timesteps for p in self._process_list}
        if len(steps) > 1:
            raise RuntimeError(
                "CompositeNoiseProcess requires every sub-process to share the "
                f"same timestep count; got {sorted(steps)}."
            )
        return next(iter(steps))

    def initialize_from_data(self, raw_pyg_loader: Iterable[Batch]) -> None:
        """Fan out the loader-backed initialisation hook to every sub-process.

        ``raw_pyg_loader`` is consumed once per sub-process. Most concrete
        loaders are re-iterable PyTorch ``DataLoader`` instances, which
        is what every datamodule's ``train_dataloader_raw_pyg`` returns.
        """
        for proc in self._process_list:
            proc.initialize_from_data(raw_pyg_loader)

    def needs_data_initialization(self) -> bool:
        """Return True when any sub-process needs a dataloader-backed setup."""
        return any(p.needs_data_initialization() for p in self._process_list)

    def is_initialized(self) -> bool:
        """Return True only when every sub-process reports itself initialised."""
        return all(p.is_initialized() for p in self._process_list)

    def process_state_condition_vector(self, t: Tensor) -> Tensor:
        """Concatenate per-process condition vectors along the feature axis.

        The sub-processes already return per-sample conditioning (shape
        ``(bs,)`` or ``(bs, d)``); this method unsqueezes scalar
        vectors to ``(bs, 1)`` before concatenating so the composite
        always returns a 2-D conditioning tensor.
        """
        pieces: list[Tensor] = []
        for proc in self._process_list:
            piece = proc.process_state_condition_vector(t)
            if piece.dim() == 1:
                piece = piece.unsqueeze(-1)
            pieces.append(piece)
        return torch.cat(pieces, dim=-1)

    def sample_prior(self, node_mask: Tensor) -> GraphData:
        """Compose priors by threading each sub-process's output through the next.

        Every sub-process returns a fully formed ``GraphData``; because
        the ``fields`` sets are disjoint, iterating in list order with
        each output feeding the next produces a single ``GraphData``
        carrying every declared field's prior sample without collisions.
        """
        data: GraphData | None = None
        for proc in self._process_list:
            if data is None:
                data = proc.sample_prior(node_mask)
            else:
                # Re-seed the sub-process on the current state: most
                # sample_prior implementations ignore the incoming
                # ``node_mask`` arg and build their output from scratch,
                # so we need a composition primitive. A fresh prior
                # draw for a disjoint field set followed by a merge
                # via ``replace`` keeps the contract local to each
                # sub-process without introducing a new API surface.
                piece = proc.sample_prior(node_mask)
                kwargs: dict[str, Tensor | None] = {}
                for field in proc.fields:
                    kwargs[field] = getattr(piece, field)
                data = data.replace(**kwargs)
        assert data is not None  # non-empty processes enforced at __init__
        return data

    def forward_sample(self, x_0: GraphData, t: Tensor) -> NoisedBatch:
        """Apply each sub-process in list order and thread the output forward.

        Returns a single :class:`NoisedBatch` whose ``z_t`` carries the
        composed per-sub-process noise and whose schedule scalars come
        from the last sub-process. The composite enforces a shared
        timestep count across sub-processes
        (:attr:`CompositeNoiseProcess.timesteps`); when sub-processes
        share an identical schedule (the canonical case) the scalars
        agree across choices.
        """
        data = x_0
        last_batch: NoisedBatch | None = None
        for proc in self._process_list:
            last_batch = proc.forward_sample(data, t)
            data = last_batch.z_t
        # ``processes`` is non-empty by ``__init__`` invariant.
        assert last_batch is not None
        return NoisedBatch(
            z_t=data,
            t_int=last_batch.t_int,
            t=last_batch.t,
            beta_t=last_batch.beta_t,
            alpha_t_bar=last_batch.alpha_t_bar,
            alpha_s_bar=last_batch.alpha_s_bar,
        )

    def posterior_sample(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Apply each sub-process's posterior sampler in list order."""
        data = z_t
        for proc in self._process_list:
            data = proc.posterior_sample(data, x0_param, t, s)
        return data

    def posterior_sample_from_model_output(
        self,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> GraphData:
        """Iterate sub-processes in list order, threading the running state.

        Each sub-process consumes the current ``z_t`` and returns a
        ``GraphData`` carrying its declared-field updates (with the
        non-declared fields preserved). Because the composite enforces
        disjoint field sets at construction time, the iteration order
        has no effect on per-field semantics; list order only matters
        for shared RNG draws and for the concatenation order of
        :meth:`process_state_condition_vector`.
        """
        data = z_t
        for proc in self._process_list:
            data = proc.posterior_sample_from_model_output(data, x0_param, t, s)
        return data

    def loss_for(self, field: FieldName) -> Literal["ce", "mse"]:
        """Return the loss kind attached to ``field`` via ``GRAPHDATA_LOSS_KIND``.

        Raises
        ------
        KeyError
            When ``field`` is not declared by any sub-process, so the
            caller does not silently ask for a loss on an un-noised
            field.
        """
        if field not in self.fields:
            raise KeyError(
                f"Field {field!r} is not declared by any sub-process of "
                f"CompositeNoiseProcess (declared: {sorted(self.fields)})."
            )
        return GRAPHDATA_LOSS_KIND[field]

    def _ensure_exact_density(self) -> list[ExactDensityNoiseProcess]:
        """Return the sub-process list as ``ExactDensityNoiseProcess`` or raise."""
        exact: list[ExactDensityNoiseProcess] = []
        for proc in self._process_list:
            if not isinstance(proc, ExactDensityNoiseProcess):
                raise NotImplementedError(
                    "CompositeNoiseProcess density methods require every "
                    "sub-process to be ExactDensityNoiseProcess; got "
                    f"{type(proc).__name__} which does not provide tractable "
                    "densities."
                )
            exact.append(proc)
        return exact

    def forward_log_prob(
        self,
        x_t: GraphData,
        x_0: GraphData,
        t: Tensor,
    ) -> Tensor:
        """Sum per-sample forward log-densities across sub-processes."""
        exact = self._ensure_exact_density()
        total = exact[0].forward_log_prob(x_t, x_0, t)
        for proc in exact[1:]:
            total = total + proc.forward_log_prob(x_t, x_0, t)
        return total

    def posterior_log_prob(
        self,
        x_s: GraphData,
        z_t: GraphData,
        x0_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> Tensor:
        """Sum per-sample posterior log-densities across sub-processes."""
        exact = self._ensure_exact_density()
        total = exact[0].posterior_log_prob(x_s, z_t, x0_param, t, s)
        for proc in exact[1:]:
            total = total + proc.posterior_log_prob(x_s, z_t, x0_param, t, s)
        return total

    def prior_log_prob(self, x: GraphData) -> Tensor:
        """Sum per-sample prior log-densities across sub-processes."""
        exact = self._ensure_exact_density()
        total = exact[0].prior_log_prob(x)
        for proc in exact[1:]:
            total = total + proc.prior_log_prob(x)
        return total


# ---------------------------------------------------------------------------
# Deprecated alias (Wave 2.1): retained for intra-branch continuity until every
# call-site is migrated to ``GaussianNoiseProcess``. Scheduled for removal when
# the refactor branch is squash-merged per the spec's clean-break stance.
# ---------------------------------------------------------------------------
ContinuousNoiseProcess = GaussianNoiseProcess
