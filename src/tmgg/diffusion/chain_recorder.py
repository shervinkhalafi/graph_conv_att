"""Per-step PMF / sample snapshots from the reverse sampling loop.

Hooked into :meth:`tmgg.diffusion.sampler.Sampler.sample` as an optional
side-effect, the :class:`ChainRecorder` captures the post-symmetrisation
``z_t`` for the first ``num_chains_to_save`` graphs of the batch on
every Kth reverse step. After the loop completes the recorder
:meth:`finalize` returns a dict suitable for ``torch.save`` whose schema
is documented in
``internal notes (not included)``.

The recorder is deliberately separate from the existing
:class:`tmgg.diffusion.collectors.StepMetricCollector` (per spec
resolution Q1): metrics aggregate scalars per step, the recorder
snapshots full per-position categorical PMFs across many steps. They
have different lifetimes and different output schemas.

Per the spec resolutions (2026-04-22):

* **Q3** -- accumulate snapshots on GPU during the reverse loop. The
  caller pays the device-to-host transfer cost once at
  :meth:`finalize`.
* **Q4** -- snapshots are taken **post-symmetrisation**, after the
  sampler's ``triu+symmetrize`` site (matches upstream DiGress).
* **Q5** -- when the recorder is wired through a
  :class:`~tmgg.diffusion.noise_process.CompositeNoiseProcess`, snapshot
  keys are prefixed with the sub-process name (e.g.
  ``categorical/E_class``). The sampler-side wiring composes one
  recorder per sub-process and merges the snapshots back into a single
  artefact at finalisation.

Storage schema (single dict, written via ``torch.save``):

================  =================  =====================================
key               dtype              shape
================  =================  =====================================
``E_chain``       ``torch.float32``  ``(S, C, n_max, n_max, de)``
``X_chain``       ``torch.float32``  ``(S, C, n_max, dx)`` -- present only
                                     when ``X_class`` is populated
``node_mask``     ``torch.bool``     ``(C, n_max)`` -- constant across
                                     snapshots
``step_indices``  ``torch.long``     ``(S,)`` -- the ``step_index`` value
                                     at which each snapshot was captured,
                                     in capture order (latest noise first)
``meta``          ``dict[str, Any]`` provenance dict supplied at
                                     construction (``global_step``,
                                     ``epoch``, ``T``,
                                     ``snapshot_step_interval``,
                                     ``noise_process``, ``ema_active``);
                                     stored unprefixed even when
                                     ``field_prefix`` namespaces other
                                     keys, so a single artefact carries
                                     one global provenance record
================  =================  =====================================

``S`` is the number of recorded snapshots, ``C`` the configured
``num_chains_to_save``. Field-prefixed keys (``<prefix>/<field>``) carry
the same semantics under a sub-process namespace.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import DenseGraphState


class ChainRecorder:
    """Capture per-step PMF / sample snapshots from the reverse loop.

    Parameters
    ----------
    num_chains_to_save
        Number of graphs from the front of the batch to track. Must be
        ``>= 1``; the "0 disables capture" mode is handled at the
        orchestration layer (the runner refrains from constructing a
        recorder).
    snapshot_step_interval
        Cadence in *reverse-loop steps*. ``maybe_record`` captures at
        ``step_index == 0`` and at every ``step_index % interval == 0``
        thereafter, plus the final step. Must be ``>= 1``. The naming
        deliberately differs from upstream's ``number_chain_steps``;
        upstream picks the write *index* via
        ``(s_int * number_chain_steps) // T``, ours simply samples every
        Kth reverse step. The two coincide for the common case of
        ``T == K * number_chain_steps``.
    meta
        Provenance dict written verbatim into ``finalize()`` output
        under the ``"meta"`` key. Required at construction (no default)
        per CLAUDE.md fail-loud spirit -- callers must explicitly
        decide what to record. Per spec D-16a Resolutions Q5 the
        canonical shape carries ``global_step``, ``epoch``, ``T``,
        ``snapshot_step_interval``, ``noise_process`` (qualname
        string), and ``ema_active``. The dict is shallow-copied at
        construction so post-construction mutations on the caller's
        copy do not affect the recorded artefact.
    field_prefix
        Optional namespace prefix for the snapshot keys. When the
        recorder is wired through a
        :class:`~tmgg.diffusion.noise_process.CompositeNoiseProcess`,
        the sampler instantiates one recorder per sub-process with that
        sub-process's name as the prefix (e.g. ``"categorical"``) so
        the final dict carries ``categorical/E_chain`` etc. Empty
        string (the default) emits unprefixed keys. The ``"meta"`` key
        is **not** prefixed; provenance is one global record per
        artefact.

    Raises
    ------
    ValueError
        ``num_chains_to_save < 1`` or ``snapshot_step_interval < 1``.
    """

    def __init__(
        self,
        num_chains_to_save: int,
        snapshot_step_interval: int,
        meta: dict[str, Any],
        field_prefix: str = "",
    ) -> None:
        if num_chains_to_save < 1:
            raise ValueError(
                "ChainRecorder.num_chains_to_save must be >= 1; "
                f"got {num_chains_to_save}. Disable capture by not "
                "constructing a recorder rather than passing 0."
            )
        if snapshot_step_interval < 1:
            raise ValueError(
                "ChainRecorder.snapshot_step_interval must be >= 1; "
                f"got {snapshot_step_interval}."
            )
        self.num_chains_to_save = num_chains_to_save
        self.snapshot_step_interval = snapshot_step_interval
        self.field_prefix = field_prefix
        # Defensive shallow copy: meta is provenance and the caller may
        # mutate their dict after construction (e.g. update epoch
        # counters). The recorded artefact must reflect the snapshot
        # taken at construction time.
        self._meta: dict[str, Any] = dict(meta)

        self._e_snapshots: list[Tensor] = []
        self._x_snapshots: list[Tensor] = []
        self._step_indices: list[int] = []
        self._node_mask: Tensor | None = None
        self._has_x: bool | None = None

    def maybe_record(self, step_index: int, z_t: DenseGraphState) -> None:
        """Capture ``z_t`` if ``step_index`` lands on a snapshot slot.

        The first call (regardless of step) and every Kth call
        thereafter snapshot the first ``num_chains_to_save`` graphs of
        the batch. The recorder accumulates Tensors on whatever device
        ``z_t`` lives on; transfer to CPU happens at :meth:`finalize`.

        ``z_t`` is the post-symmetrisation reverse-step sample (a dense
        state). The recorder stores ``E_class[:c]`` and ``X_class[:c]``
        verbatim, so it requires the dense ``(B, n_max, ...)`` carrier;
        sparse ``GraphState`` would need to be densified first.

        Raises
        ------
        ValueError
            The presented batch has fewer than ``num_chains_to_save``
            graphs. We fail at the first record call rather than
            silently truncate -- a misconfigured batch is a user bug.
        RuntimeError
            ``z_t`` lacks ``E_class`` (the recorder requires at least
            an edge-categorical field).
        """
        if step_index % self.snapshot_step_interval != 0:
            return

        if z_t.E_class is None:
            raise RuntimeError(
                "ChainRecorder requires z_t.E_class to be populated; "
                "the categorical edge field is the minimum payload."
            )

        bs = int(z_t.E_class.shape[0])
        if bs < self.num_chains_to_save:
            raise ValueError(
                f"ChainRecorder.maybe_record received a batch of {bs} graphs "
                f"but num_chains_to_save={self.num_chains_to_save}; the "
                "batch must be at least as large as the configured chain count."
            )

        c = self.num_chains_to_save
        e_snap = z_t.E_class[:c].detach().clone()
        self._e_snapshots.append(e_snap)
        self._step_indices.append(step_index)

        if self._has_x is None:
            self._has_x = z_t.X_class is not None
        elif self._has_x != (z_t.X_class is not None):
            raise RuntimeError(
                "ChainRecorder observed inconsistent X_class presence "
                "across reverse steps; the field set must be stable for "
                "a single sample call."
            )

        if z_t.X_class is not None:
            self._x_snapshots.append(z_t.X_class[:c].detach().clone())

        if self._node_mask is None:
            self._node_mask = z_t.node_mask[:c].detach().clone()

    def finalize(self) -> dict[str, Any]:
        """Stack snapshots and return a dict ready for ``torch.save``.

        Tensors are moved to CPU before stacking so a downstream
        ``torch.save`` does not pin GPU memory. Snapshot keys are
        prefixed with :attr:`field_prefix` when non-empty (separator
        ``/``); the provenance ``"meta"`` key is **always** unprefixed
        because there is one global provenance record per artefact even
        when several sub-process recorders fan out under
        :class:`~tmgg.diffusion.noise_process.CompositeNoiseProcess`.

        Returns
        -------
        dict[str, Any]
            Schema documented in the module docstring. ``X_chain`` is
            absent when ``z_t.X_class`` was ``None`` throughout the
            recorded chain. ``meta`` carries the provenance dict
            supplied at construction.

        Raises
        ------
        RuntimeError
            :meth:`maybe_record` was never called -- the recorder has
            nothing to finalise. This points at a sampler that ran
            zero reverse steps, which is itself a bug.
        """
        if not self._e_snapshots:
            raise RuntimeError(
                "ChainRecorder.finalize() called with no recorded "
                "snapshots; check that maybe_record was wired into the "
                "reverse loop and that snapshot_step_interval is "
                "compatible with the loop length."
            )
        assert self._node_mask is not None  # populated by first record call

        # Stack on the same device, then cpu() once.
        e_chain = torch.stack(self._e_snapshots, dim=0).cpu()
        node_mask = self._node_mask.cpu()
        step_indices = torch.tensor(self._step_indices, dtype=torch.long)

        out: dict[str, Any] = {}
        prefix = f"{self.field_prefix}/" if self.field_prefix else ""
        out[f"{prefix}E_chain"] = e_chain
        out[f"{prefix}node_mask"] = node_mask
        out[f"{prefix}step_indices"] = step_indices
        if self._x_snapshots:
            x_chain = torch.stack(self._x_snapshots, dim=0).cpu()
            out[f"{prefix}X_chain"] = x_chain
        # Provenance is a single global record; never prefixed. Shallow
        # copy on emit so a post-finalize mutation on the artefact
        # consumer's side does not bleed back into the recorder's
        # stored copy.
        out["meta"] = dict(self._meta)
        return out


def merge_chain_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-sub-process recorder outputs into a single artefact dict.

    Used by the sampler when the noise process is composite: each
    sub-process gets its own recorder with a unique ``field_prefix``;
    after the loop the per-sub-process dicts are concatenated into one
    artefact via key union. Disjoint-field invariant on
    :class:`~tmgg.diffusion.noise_process.CompositeNoiseProcess`
    guarantees the field-prefixed keys do not collide.

    The ``"meta"`` key is special: every recorder emits it (provenance
    is one global record per artefact, never prefixed). The merge keeps
    a single ``"meta"`` entry; if multiple recorders supply distinct
    meta dicts, the merge raises -- callers must ensure all sub-process
    recorders share the same provenance.

    Parameters
    ----------
    snapshots
        Ordered list of finalised recorder outputs.

    Returns
    -------
    dict[str, Any]
        Union of all field-prefixed keys, plus a single ``"meta"`` entry.

    Raises
    ------
    ValueError
        Two snapshots collide on a non-meta key (would only happen if a
        field-prefix wiring bug let two recorders share a prefix), or
        two snapshots carry different meta dicts.
    """
    merged: dict[str, Any] = {}
    meta: dict[str, Any] | None = None
    for snap in snapshots:
        for key, value in snap.items():
            if key == "meta":
                if meta is None:
                    meta = value
                elif meta != value:
                    raise ValueError(
                        "merge_chain_snapshots: incompatible meta dicts "
                        f"across sub-process recorders: {meta} vs {value}. "
                        "All sub-process recorders must share the same "
                        "provenance for a single artefact."
                    )
                continue
            if key in merged:
                raise ValueError(
                    f"merge_chain_snapshots: duplicate key {key!r}; "
                    "field-prefix wiring on the sampler is broken."
                )
            merged[key] = value
    if meta is not None:
        merged["meta"] = meta
    return merged
