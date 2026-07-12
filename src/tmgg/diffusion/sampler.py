"""Composable reverse-diffusion sampling for graph generation.

``Sampler`` owns one reverse-chain loop that delegates process-specific
parameterisation and final decoding back to ``NoiseProcess`` hooks.
``CategoricalSampler`` and ``ContinuousSampler`` remain as semantic aliases
so configuration and call sites can still name the intended sampling mode.

Wave 4-A note (sparse-default refactor)
---------------------------------------
The public-facing surface threads the new sparse type
:class:`tmgg.data.datasets.graph_types.GraphState` for warm-start input
and the per-graph output list. The model call uses ``output_dense=True``
so the model emits a :class:`DenseGraphDistribution`, matching the new
forward contract on every concrete model class. The internal step
delegates posterior sampling to ``NoiseProcess.posterior_sample_from_model_output``
which still consumes the legacy dense ``GraphData`` form; that call site
will be cleaned up in Wave 5 once ``noise_process.py`` adopts the new
types end-to-end. The Sampler keeps that bridge inert by funnelling all
internal traffic through the ``DenseGraphState`` / ``DenseGraphDistribution``
carriers, which are field-compatible with the legacy form
(``node_mask``, ``X_class``, ``E_class``, ``y``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import (
    DenseGraphDistribution,
    DenseGraphState,
    GraphData,
    GraphDistribution,
    GraphState,
)
from tmgg.diffusion.chain_recorder import ChainRecorder
from tmgg.diffusion.collectors import StepMetricCollector
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    CompositeNoiseProcess,
    GaussianNoiseProcess,
    NoiseProcess,
)

# ``GraphModel`` resolves at runtime via the project's editable install,
# but basedpyright fails to surface symbols added through that path
# until a fresh `uv pip install -e .` plus an editor / language-server
# reload. The same ignore is applied at every import site that hits this
# false positive (see e.g. ``training/lightning_modules/base_graph_module.py``);
# this file follows the same precedent rather than inventing a new one.
from tmgg.models.base import GraphModel  # pyright: ignore[reportAttributeAccessIssue]

# ``DenseGraphDistribution`` and ``GraphDistribution`` only appear in
# this module's docstrings; the imports keep them resolvable for
# documentation tooling and signal the intended public surface.
__all__ = [
    "CategoricalSampler",
    "ContinuousSampler",
    "DenseGraphDistribution",
    "DiffusionState",
    "GraphDistribution",
    "Sampler",
]


class _BufferingCollector:
    """One-shot collector that captures the last ``record`` call's metrics.

    Used by :meth:`Sampler._record_step_metrics` to fan out across the
    sub-processes of a :class:`CompositeNoiseProcess`: each sub-process
    writes into a fresh buffering collector and the outer method sums
    the contributions into a single record on the real collector.
    """

    def __init__(self) -> None:
        self.last: dict[str, float] = {}

    def record(self, t: int, s: int, metrics: dict[str, float]) -> None:
        """Overwrite the buffer with the latest metrics payload."""
        _ = t, s
        self.last = dict(metrics)


@dataclass(frozen=True, slots=True)
class DiffusionState:
    """A batched latent graph state together with its diffusion timestep.

    Parameters
    ----------
    graph
        Sparse :class:`GraphState` carrying the latent at timestep ``t``.
        Per the Wave 4-A contract, warm starts are presented in the
        sparse-default carrier; densification into ``DenseGraphState``
        happens inside :meth:`Sampler.sample` at the boundary into the
        legacy noise-process API.
    t
        Current diffusion timestep (``0 <= t <= max_t``).
    max_t
        Total number of diffusion steps owned by the active noise
        process.
    """

    graph: GraphState
    t: int
    max_t: int

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphState):
            raise TypeError(
                "DiffusionState.graph must be a GraphState instance, "
                f"got {type(self.graph).__name__}"
            )
        if isinstance(self.t, bool) or not isinstance(self.t, int):
            raise TypeError(
                f"DiffusionState.t must be an int, got {type(self.t).__name__}"
            )
        if isinstance(self.max_t, bool) or not isinstance(self.max_t, int):
            raise TypeError(
                f"DiffusionState.max_t must be an int, got {type(self.max_t).__name__}"
            )
        if self.max_t < 1:
            raise ValueError(f"DiffusionState.max_t must be >= 1, got {self.max_t}")
        if not 0 <= self.t <= self.max_t:
            raise ValueError(
                f"DiffusionState.t must satisfy 0 <= t <= max_t, got t={self.t}, "
                f"max_t={self.max_t}"
            )

        graph = self.graph
        # Per-graph node-count tensor fixes the batch shape; the ``y``
        # graph-level features must align with that batch axis. Other
        # invariants (sparse fields, edge_index) are checked inside
        # ``GraphState.__post_init__``.
        bs = int(graph.num_nodes_per_graph.shape[0])
        if graph.y.dim() != 2:
            raise ValueError(
                "DiffusionState.graph.y must be batched (bs, dy), "
                f"got shape {tuple(graph.y.shape)}"
            )
        if graph.y.shape[0] != bs:
            raise ValueError(
                "DiffusionState.graph.y must align with batch size, "
                f"got y shape {tuple(graph.y.shape)} and batch size {bs}"
            )


class Sampler:
    """Unified reverse-diffusion sampler loop.

    The sampler owns the reverse-chain control flow only:
    prior or warm start -> condition vector -> model -> posterior parameter
    -> posterior sample -> finalize. Process-specific math stays on the
    ``NoiseProcess`` side of the interface.

    The categorical row-sum-zero floor (``zero_floor`` in upstream
    DiGress, parity #28 / D-5) lives on
    :class:`tmgg.diffusion.noise_process.CategoricalNoiseProcess` rather
    than the sampler -- that is the object that owns the categorical
    Bayes math and needs to thread the floor into both the sampling
    helper and the VLB / KL probabilities computation. The sampler
    stays process-agnostic.

    Parameters
    ----------
    assert_symmetric_e
        When ``True`` (default), every reverse step asserts that the
        sampled ``z_t.E_class`` is symmetric across the node-pair axes,
        matching upstream DiGress's ``assert (E_s == E_s.transpose(1, 2)).all()``
        check at ``diffusion_model_discrete.py:649``. Production hot
        loops at fp32 batch sizes can disable this with ``False`` if
        the per-step overhead matters; the rest of the codebase already
        symmetrises ``E_class`` inside :func:`sample_discrete_features`,
        so the check is a guard against future drift rather than a
        load-bearing correctness step.
    """

    def __init__(
        self,
        *,
        assert_symmetric_e: bool = True,
    ) -> None:
        self._assert_symmetric_e = assert_symmetric_e

    @staticmethod
    def _build_num_nodes(
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
    ) -> Tensor:
        """Return per-graph node counts on ``device``."""
        if isinstance(num_nodes, int):
            return torch.full((num_graphs,), num_nodes, device=device, dtype=torch.long)
        return num_nodes.to(device=device, dtype=torch.long)

    @staticmethod
    def _trim_batched_graphs(
        final: DenseGraphState, n_nodes: Tensor
    ) -> list[GraphState]:
        """Split a batched final state into per-graph :class:`GraphState` items.

        Each per-graph slice is built as an unbatched
        :class:`DenseGraphState` first (because
        :meth:`NoiseProcess.finalize_sample` returns a dense-shaped
        carrier with ``X_class`` / ``E_class`` / ``X_feat`` / ``E_feat``
        as ``(B, n, ...)`` / ``(B, n, n, ...)`` tensors) and then
        :meth:`DenseGraphState.to_sparse` materialises the sparse
        :class:`GraphState`. Empty / structure-only fields pass through
        as ``None``.
        """
        results: list[GraphState] = []
        for i, n_tensor in enumerate(n_nodes, start=0):
            n = int(n_tensor.item())
            num_nodes_per_graph_i = torch.tensor([n], dtype=torch.long, device="cpu")
            y_i = final.y[i : i + 1].cpu()
            x_class_i = (
                final.X_class[i : i + 1, :n].cpu()
                if final.X_class is not None
                else None
            )
            x_feat_i = (
                final.X_feat[i : i + 1, :n].cpu() if final.X_feat is not None else None
            )
            e_class_i = (
                final.E_class[i : i + 1, :n, :n].cpu()
                if final.E_class is not None
                else None
            )
            e_feat_i = (
                final.E_feat[i : i + 1, :n, :n].cpu()
                if final.E_feat is not None
                else None
            )
            dense = DenseGraphState(
                num_nodes_per_graph=num_nodes_per_graph_i,
                y=y_i,
                X_class=x_class_i,
                X_feat=x_feat_i,
                E_class=e_class_i,
                E_feat=e_feat_i,
            )
            results.append(dense.to_sparse())
        return results

    @staticmethod
    def _record_step_metrics(
        collector: StepMetricCollector,
        noise_process: NoiseProcess,
        z_t: GraphData,
        posterior_param: GraphData,
        t: Tensor,
        s: Tensor,
    ) -> None:
        """Record per-step diagnostics for likelihood collectors.

        Composite processes fan out to every sub-process and sum the
        per-step contributions into a single collector record. The
        least-surprise choice is to report a joint ``{"kl": ...}`` that
        adds the sub-process contributions under the assumption of
        disjoint fields (the composite's construction-time invariant).
        Unknown leaf processes still raise ``TypeError`` so a future
        non-categorical/non-Gaussian leaf cannot silently drop its
        diagnostic contribution.
        """
        t_int = int(t[0].item())
        s_int = int(s[0].item())

        if isinstance(noise_process, CompositeNoiseProcess):
            merged: dict[str, float] = {}
            for sub in noise_process._process_list:  # noqa: SLF001
                sub_collector = _BufferingCollector()
                Sampler._record_step_metrics(
                    sub_collector, sub, z_t, posterior_param, t, s
                )
                for key, value in sub_collector.last.items():
                    merged[key] = merged.get(key, 0.0) + value
            if merged:
                collector.record(t_int, s_int, merged)
            return

        if isinstance(noise_process, CategoricalNoiseProcess):
            # ``_posterior_probabilities`` returns the legacy dense-shaped
            # GraphData (post-Wave-5 cleanup it will be a DenseGraphState
            # / DenseGraphDistribution); narrow to the dense state type
            # so the X_class / E_class field reads typecheck.
            posterior_probs = cast(
                DenseGraphState,
                noise_process._posterior_probabilities(z_t, posterior_param, t, s),
            )
            x_probs = posterior_probs.X_class
            e_probs = posterior_probs.E_class
            if x_probs is None or e_probs is None:
                raise RuntimeError(
                    "CategoricalNoiseProcess._posterior_probabilities must "
                    "populate X_class and E_class; got None."
                )
            # ``z_t`` is the categorical-branch state held by the sampler
            # loop, which always materialises with a dense ``node_mask``
            # (the legacy noise-process API requires it). Narrow for the
            # field read.
            z_t_dense = cast(DenseGraphState, z_t)
            node_mask = z_t_dense.node_mask
            ent_x = -torch.sum(
                x_probs * torch.log(x_probs.clamp(min=1e-30)),
                dim=-1,
            )
            ent_x = (ent_x * node_mask.float()).sum() / node_mask.sum().clamp(min=1)

            ent_e = -torch.sum(
                e_probs * torch.log(e_probs.clamp(min=1e-30)),
                dim=-1,
            )
            edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).float()
            ent_e = (ent_e * edge_mask).sum() / edge_mask.sum().clamp(min=1)

            collector.record(
                t_int,
                s_int,
                {
                    "kl": (ent_x + ent_e).item(),
                    "kl_X": ent_x.item(),
                    "kl_E": ent_e.item(),
                },
            )
            return

        if isinstance(noise_process, GaussianNoiseProcess):
            posterior = noise_process._posterior_parameters(z_t, posterior_param, t, s)
            mean = posterior["mean"]
            std = posterior["std"]
            kl = 0.5 * (std.pow(2) + mean.pow(2) - 1 - 2 * std.clamp(min=1e-30).log())
            collector.record(t_int, s_int, {"kl": kl.mean().item()})
            return

        raise TypeError(
            "Collector metrics are only implemented for categorical and "
            f"continuous diffusion, got {type(noise_process).__name__}"
        )

    @torch.no_grad()
    def sample(
        self,
        model: GraphModel,
        noise_process: NoiseProcess,
        num_graphs: int,
        num_nodes: int | Tensor,
        device: torch.device,
        *,
        start_from: DiffusionState | None = None,
        collector: StepMetricCollector | None = None,
        chain_recorder: ChainRecorder | dict[str, ChainRecorder] | None = None,
    ) -> list[GraphState]:
        """Generate graphs via reverse diffusion.

        Parameters
        ----------
        model
            Trained ``GraphModel`` that predicts clean data from noisy input.
            The forward call uses ``output_dense=True`` so the model
            returns a :class:`DenseGraphDistribution` regardless of the
            carrier of the input ``z_t``; this matches the Wave 3 model
            forward contract.
        noise_process
            Module-owned diffusion process used for the reverse chain.
        num_graphs
            Number of graphs to generate.
        num_nodes
            Either a fixed node count (int) applied to all graphs, or a
            per-graph tensor of shape ``(num_graphs,)``.
        device
            Device on which to run sampling.
        start_from
            Starting latent graph state and timestep for partial reverse
            chains. The wrapped graph is a sparse :class:`GraphState`;
            the sampler densifies it internally to feed the legacy
            noise-process posterior interface. When ``None``, sampling
            starts from the process prior at ``t=T``.
        collector
            Per-step metric collector. When provided,
            ``record(t, s, metrics)`` is called at each reverse step
            with available metrics (e.g., ``{"kl": ...}``).
        chain_recorder
            Optional recorder that snapshots ``z_t`` after each reverse
            step's symmetrisation site. The recorder's
            ``maybe_record`` is called with ``step_index`` (zero-indexed
            from the first reverse step, in capture order -- latest
            noise first). Per parity spec D-16a (resolution Q4),
            snapshots are post-symmetrisation, matching upstream
            DiGress's chain-saving behaviour.

            May also be a ``dict[str, ChainRecorder]`` for the composite
            fan-out case (D-16a Resolutions Q5 / W2-6): when the noise
            process is a :class:`~tmgg.diffusion.noise_process.CompositeNoiseProcess`,
            the :class:`~tmgg.training.callbacks.chain_saving.ChainSavingCallback`
            supplies one recorder per sub-process with a unique
            ``field_prefix``. The sampler fans the same post-step
            ``z_t`` to every recorder, so each writes its prefixed
            key namespace;
            :func:`~tmgg.diffusion.chain_recorder.merge_chain_snapshots`
            reconverges the per-sub-process artefacts at finalise. The
            dict keys only affect the emitted prefixes -- every
            recorder observes the same composed state.

        Returns
        -------
        list[GraphState]
            One :class:`GraphState` per generated graph, trimmed to its
            real node count. Wave 5 cleanup will reconcile downstream
            consumers (evaluator, callbacks) to consume the sparse
            carrier directly instead of the legacy
            :class:`GraphData`-shaped dense form.
        """
        if start_from is not None:
            if start_from.max_t != noise_process.timesteps:
                raise ValueError(
                    "Warm-start max_t must match the active noise process "
                    f"timesteps, got start_from.max_t={start_from.max_t} "
                    f"and noise_process.timesteps={noise_process.timesteps}"
                )
            sparse_start = start_from.graph.to(device)
            bs = int(sparse_start.num_nodes_per_graph.shape[0])
            if bs != num_graphs:
                raise ValueError(
                    "Warm-start batch size must match num_graphs, "
                    f"got batch size {bs} and num_graphs={num_graphs}"
                )
            n_nodes = sparse_start.num_nodes_per_graph
            t_start = start_from.t
            # Densify the sparse warm-start so the internal flow (model
            # call coercion, noise_process posterior, chain_recorder)
            # operates on the dense ``DenseGraphState`` carrier. The
            # legacy noise-process API reads ``X_class`` / ``E_class`` /
            # ``node_mask`` from this form.
            z_t: GraphData = sparse_start.to_dense(
                edge_class_fill=_default_no_edge_fill(sparse_start.edge_class),
            )
        else:
            n_nodes = self._build_num_nodes(num_graphs, num_nodes, device)
            n_max = int(n_nodes.max().item())
            arange = (
                torch.arange(n_max, device=device).unsqueeze(0).expand(num_graphs, -1)
            )
            node_mask = arange < n_nodes.unsqueeze(1)
            z_t = noise_process.sample_prior(node_mask).to(device)
            bs = num_graphs
            t_start = noise_process.timesteps

        for step_index, s_int in enumerate(reversed(range(0, t_start))):
            t_int = s_int + 1
            t_tensor = torch.full((bs,), t_int, device=device, dtype=torch.long)
            s_tensor = torch.full((bs,), s_int, device=device, dtype=torch.long)

            condition = noise_process.process_state_condition_vector(t_tensor)
            # ``output_dense=True`` selects the :class:`DenseGraphDistribution`
            # return type on every concrete model class (Wave 3 forward
            # contract). The sparse-default refactor flips the model
            # output to the dense distribution at this single call site
            # so the legacy noise-process API (which reads ``X_class`` /
            # ``E_class`` / ``node_mask``) still receives a compatible
            # carrier. Wave 5 will move the noise-process to native
            # sparse and remove the carrier mismatch entirely.
            model_output = model(z_t, t=condition, output_dense=True)
            posterior_param = noise_process.model_output_to_posterior_parameter(
                model_output
            )

            if collector is not None:
                self._record_step_metrics(
                    collector,
                    noise_process,
                    z_t,
                    posterior_param,
                    t_tensor,
                    s_tensor,
                )

            # Delegate per-field reverse sampling to the noise process.
            # ``posterior_sample_from_model_output`` is a template-method
            # hook: the default path on ``NoiseProcess`` routes to
            # ``posterior_sample``, ``CategoricalNoiseProcess`` overrides
            # to the per-class marginalised form (matching upstream
            # DiGress' ``sum_c p(z_s | z_t, x_0=c) p(x_0=c | z_t)``),
            # and ``CompositeNoiseProcess`` iterates sub-processes in
            # list order. The loop now carries no ``isinstance``
            # dispatch on the process type.
            z_t = noise_process.posterior_sample_from_model_output(
                z_t, posterior_param, t_tensor, s_tensor
            )

            # Per-reverse-step E-symmetry guard, matching upstream
            # diffusion_model_discrete.py:649 (parity #28 / #29 / D-5).
            # Inert on healthy categorical samplers because
            # sample_discrete_features already symmetrises; the check
            # protects against future code paths that bypass the
            # canonical symmetrisation primitive.
            #
            # ``z_t`` here is the legacy noise-process return — dense
            # carrier shape, with ``E_class`` of shape ``(B, n, n, d_ec)``.
            # Narrow to ``DenseGraphState`` for typed access; Wave 5
            # cleanup will retype the noise-process return so the cast
            # disappears.
            z_t_dense_post = cast(DenseGraphState, z_t)
            if self._assert_symmetric_e and z_t_dense_post.E_class is not None:
                e = z_t_dense_post.E_class
                if not torch.allclose(e, e.transpose(-3, -2)):
                    raise AssertionError(
                        "Sampler reverse step produced asymmetric E_class at "
                        f"t={t_int} (max abs deviation = "
                        f"{(e - e.transpose(-3, -2)).abs().max().item():.3e})."
                    )

            # Post-symmetrisation chain snapshot (parity D-16a / spec
            # resolution Q4). The recorder owns the snapshot-cadence
            # gate; the sampler only forwards every reverse step.
            # For the composite fan-out (D-16a Resolutions Q5 / W2-6)
            # chain_recorder may be a dict[str, ChainRecorder]; every
            # recorder observes the same composed z_t and writes to its
            # own field_prefix namespace at finalise.
            if chain_recorder is not None:
                if isinstance(chain_recorder, ChainRecorder):
                    chain_recorder.maybe_record(step_index, z_t_dense_post)
                else:
                    for sub_recorder in chain_recorder.values():
                        sub_recorder.maybe_record(step_index, z_t_dense_post)

        # ``finalize_sample`` returns the dense-carrier state from the
        # legacy noise-process API; narrow to ``DenseGraphState`` so the
        # per-graph trim sees a typed split-field carrier. Wave 5 cleanup
        # will retype the noise-process return and remove the cast.
        final = cast(DenseGraphState, noise_process.finalize_sample(z_t))
        return self._trim_batched_graphs(final, n_nodes)


class CategoricalSampler(Sampler):
    """Semantic alias for the unified sampler loop."""


class ContinuousSampler(Sampler):
    """Semantic alias for the unified sampler loop."""


def _default_no_edge_fill(edge_class: Tensor | None) -> Tensor | None:
    """Return the canonical no-edge one-hot fill for a sparse ``edge_class``.

    Mirrors :func:`tmgg.data.datasets.graph_types._no_edge_one_hot_fill`
    behaviour without importing the private helper: when ``edge_class``
    is ``None`` (structure-only graph) we pass ``None`` through, which
    instructs ``GraphState.to_dense`` to skip the edge fill. Otherwise
    we synthesise a ``(d_ec,)`` one-hot vector with a 1.0 in channel 0
    so padded / inactive (b, i, j) positions decode as the "no edge"
    class on the dense form.
    """
    if edge_class is None:
        return None
    d_ec = int(edge_class.shape[-1])
    fill = torch.zeros(d_ec, dtype=edge_class.dtype, device=edge_class.device)
    if d_ec >= 1:
        fill[0] = 1.0
    return fill
