"""Lightning callback that saves reverse-step chain snapshots per cadence.

Per parity spec D-16a Resolutions Q2: configurable cadence (every Nth
validation pass plus at-fit-end). On a snapshot validation pass the
callback constructs a :class:`~tmgg.diffusion.chain_recorder.ChainRecorder`
seeded with the full meta dict (Resolutions Q5b: ``global_step``,
``epoch``, ``T``, ``snapshot_step_interval``, ``noise_process``
qualname, ``ema_active``), threads it into the module's
``generate_graphs`` call via a one-shot stash on
``pl_module._pending_chain_recorder``, then writes the finalised
artefact to ``${chain_save_path}/epoch_{N}_chains.pt``.

Threading rationale
-------------------
Lightning's ``on_validation_epoch_end`` fires the module hook **before**
callback hooks for the same event. The module's
:meth:`tmgg.training.lightning_modules.diffusion_module.DiffusionModule.on_validation_epoch_end`
owns the sample-and-evaluate call, so the recorder must reach the
sampler through that call site rather than through a separate sampling
pass driven from the callback (which would double the work and disagree
with what the evaluator measured). The chosen mechanism: the callback
stashes the recorder on ``pl_module._pending_chain_recorder`` in
:meth:`on_validation_start`; the module's ``generate_graphs`` pops the
attribute, forwards it to ``Sampler.sample(chain_recorder=...)``, and
clears the stash. The callback retains its own reference for the
:meth:`on_validation_end` finalisation and write.

Composite fan-out (W2-6)
------------------------
When ``pl_module.noise_process`` is a
:class:`~tmgg.diffusion.noise_process.CompositeNoiseProcess`, the
callback constructs one recorder per sub-process keyed by the
sub-process's class name and stashes the dict instead of a single
recorder. ``Sampler.sample`` accepts both single-recorder and
dict-of-recorder forms; the dict form fans the same post-step ``z_t``
out to every sub-recorder so each writes to its prefixed key namespace.
:func:`~tmgg.diffusion.chain_recorder.merge_chain_snapshots` reconverges
the per-sub-process artefacts into a single dict at finalise.

Mirrors the :class:`~tmgg.training.callbacks.ema.EMACallback` /
:class:`~tmgg.training.callbacks.final_sample_dump.FinalSampleDumpCallback`
shape: getattr-based fail-loud :class:`TypeError` for missing
collaborators (no pyright suppressions).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import torch
from pytorch_lightning.callbacks import Callback

from tmgg.diffusion.chain_recorder import ChainRecorder, merge_chain_snapshots
from tmgg.diffusion.noise_process import CompositeNoiseProcess
from tmgg.training.callbacks.ema import EMACallback

if TYPE_CHECKING:
    import pytorch_lightning as pl


#: Mirror of :data:`tmgg.modal._lib.volumes.OUTPUTS_MOUNT`. Inlined to
#: avoid a tach-forbidden ``tmgg.training -> tmgg.modal`` import; pinned
#: by ``tests/training/test_chain_saving.py``.
MODAL_OUTPUTS_MOUNT = "/data/outputs"


def _require_attr(pl_module: pl.LightningModule, name: str) -> Any:
    """Return ``pl_module.<name>``, raising fail-loud if missing.

    Mirrors :func:`tmgg.training.callbacks.final_sample_dump._require_attr`
    and :func:`tmgg.training.callbacks.ema._require_backbone_parameters`.
    Lightning's ``pl_module`` is typed as the bare LightningModule base,
    which doesn't expose the diffusion-module collaborators
    (``.sampler``, ``.noise_process``, ``.T``, ...). ``getattr`` with a
    sentinel sidesteps the pyright complaint without a suppression and
    yields a clearer error than a deep ``AttributeError`` later.
    """
    sentinel = object()
    value = getattr(pl_module, name, sentinel)
    if value is sentinel or value is None:
        raise TypeError(
            f"ChainSavingCallback requires the LightningModule to expose "
            f"`.{name}`; got {type(pl_module).__name__} which does not "
            "(or which exposes None). Wire the attribute on the module "
            "before registering this callback."
        )
    return value


def _resolve_chain_path(
    *,
    configured: str | os.PathLike[str] | None,
    run_name: str,
    default_root_dir: str | os.PathLike[str],
    epoch: int,
) -> Path:
    """Resolve where ``epoch_{N}_chains.pt`` should be written.

    Priority order mirrors
    :func:`tmgg.training.callbacks.final_sample_dump._resolve_dump_path`:
    explicit ``configured`` wins; otherwise the Modal-context check
    (``MODAL_TASK_ID`` env var) routes to the canonical Modal volume
    mount; otherwise we land under ``trainer.default_root_dir / chains``.

    Parameters
    ----------
    configured
        Explicit ``chain_save_path`` from config, or ``None`` to defer
        to defaults.
    run_name
        Stable run identifier used to namespace the Modal-volume
        artefact (avoids cross-run collisions on a shared volume).
    default_root_dir
        ``trainer.default_root_dir`` -- the local fallback location.
    epoch
        Epoch index used in the filename. ``-1`` for the at-fit-end
        artefact (mirrors ``trainer.current_epoch`` after ``fit`` has
        already cleared the counter on some Lightning versions).
    """
    name = f"epoch_{epoch}_chains.pt" if epoch >= 0 else "fit_end_chains.pt"
    if configured is not None:
        return Path(configured) / name
    if os.environ.get("MODAL_TASK_ID"):
        return Path(MODAL_OUTPUTS_MOUNT) / "chains" / run_name / name
    return Path(default_root_dir) / "chains" / name


class ChainSavingCallback(Callback):
    """Snapshot reverse-loop chains on a configurable cadence.

    Parameters
    ----------
    num_chains_to_save
        Number of graphs from the front of the validation/sampling
        batch to track. Must be ``>= 1``; ``0`` disables capture and
        the orchestrator omits the callback entirely.
    snapshot_step_interval
        Reverse-loop step cadence forwarded to
        :class:`~tmgg.diffusion.chain_recorder.ChainRecorder`. Must be
        ``>= 1``.
    chain_save_every_n_val
        Validation-pass cadence: snapshot only every Nth validation
        pass (``0``-indexed; pass 0 always fires). Must be ``>= 1``.
    chain_save_at_fit_end
        When ``True``, run an extra dedicated chain-save pass at fit
        end driven directly off the sampler (independent of the
        validation cadence).
    chain_save_path
        Optional explicit output directory. When ``None`` the callback
        uses the Modal-aware resolution helper.
    run_name
        Stable run identifier for namespacing the Modal artefact.

    Raises
    ------
    ValueError
        Any positive-integer parameter is ``<= 0``.
    """

    def __init__(
        self,
        num_chains_to_save: int,
        snapshot_step_interval: int,
        chain_save_every_n_val: int,
        chain_save_at_fit_end: bool,
        chain_save_path: str | os.PathLike[str] | None = None,
        run_name: str = "run",
    ) -> None:
        super().__init__()
        if num_chains_to_save <= 0:
            raise ValueError(
                "ChainSavingCallback.num_chains_to_save must be > 0; "
                f"got {num_chains_to_save}. Disable the callback by not "
                "registering it rather than passing 0."
            )
        if snapshot_step_interval <= 0:
            raise ValueError(
                "ChainSavingCallback.snapshot_step_interval must be > 0; "
                f"got {snapshot_step_interval}."
            )
        if chain_save_every_n_val <= 0:
            raise ValueError(
                "ChainSavingCallback.chain_save_every_n_val must be > 0; "
                f"got {chain_save_every_n_val}."
            )
        self.num_chains_to_save = num_chains_to_save
        self.snapshot_step_interval = snapshot_step_interval
        self.chain_save_every_n_val = chain_save_every_n_val
        self.chain_save_at_fit_end = chain_save_at_fit_end
        self.chain_save_path = chain_save_path
        self.run_name = run_name

        # Increments on every validation pass. The snapshot-cadence
        # gate fires when ``count % chain_save_every_n_val == 0``; the
        # counter advances regardless of whether the gate fired.
        self._validation_pass_count: int = 0
        # One-shot handle on the recorder (or recorder dict) constructed
        # in on_validation_start. Cleared after on_validation_end writes
        # the artefact so a subsequent non-snapshot pass cannot
        # accidentally finalise stale state.
        self._active_recorder: ChainRecorder | dict[str, ChainRecorder] | None = None

    @staticmethod
    def _find_ema_callback(trainer: pl.Trainer) -> EMACallback | None:
        """Return a registered :class:`EMACallback`, or ``None``.

        Same getattr gate pattern as
        :meth:`FinalSampleDumpCallback._find_ema_callback`. The EMA
        active flag feeds the recorder's meta dict so the artefact
        consumer knows which weights produced the chain.
        """
        callbacks = getattr(trainer, "callbacks", None)
        if callbacks is None:
            return None
        for cb in callbacks:
            if isinstance(cb, EMACallback):
                return cb
        return None

    def _build_meta(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *, ema_active: bool
    ) -> dict[str, Any]:
        """Construct the provenance dict per spec D-16a Resolutions Q5b.

        The qualname string for ``noise_process`` lets a downstream
        artefact consumer reconstruct the producing class without
        needing the runtime object pickled into the file.
        """
        noise_process = _require_attr(pl_module, "noise_process")
        np_cls = type(noise_process)
        return {
            "global_step": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "T": int(_require_attr(pl_module, "T")),
            "snapshot_step_interval": int(self.snapshot_step_interval),
            "noise_process": f"{np_cls.__module__}.{np_cls.__qualname__}",
            "ema_active": bool(ema_active),
        }

    def _build_recorder_for_module(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> ChainRecorder | dict[str, ChainRecorder]:
        """Construct either a single recorder or a per-sub-process dict.

        Composite noise processes get one recorder per sub-process keyed
        by the sub-process's class name. The shared ``meta`` dict is
        identical across sub-recorders so
        :func:`merge_chain_snapshots` can reconverge them at finalise
        without raising the incompatible-meta guard.
        """
        ema_cb = self._find_ema_callback(trainer)
        ema_active = ema_cb is not None and ema_cb.ema is not None
        meta = self._build_meta(trainer, pl_module, ema_active=ema_active)

        noise_process = _require_attr(pl_module, "noise_process")
        if isinstance(noise_process, CompositeNoiseProcess):
            # Sub-process names are the class names; CompositeNoiseProcess
            # does not currently surface user-supplied identifiers, so the
            # class name is the most stable handle. If two sub-processes
            # share a class (rare), the prefix collides and the merge
            # would raise -- that is itself a configuration smell worth
            # surfacing loudly rather than masking with auto-numbering.
            recorders: dict[str, ChainRecorder] = {}
            seen: set[str] = set()
            for sub in noise_process._process_list:  # noqa: SLF001
                name = type(sub).__name__
                if name in seen:
                    raise RuntimeError(
                        "ChainSavingCallback: CompositeNoiseProcess has two "
                        f"sub-processes of class {name!r}; the per-sub-process "
                        "recorder dispatch keys on class name and cannot "
                        "disambiguate. Re-architect the composite to avoid "
                        "duplicate sub-process types or extend the dispatch "
                        "with a user-supplied name."
                    )
                seen.add(name)
                recorders[name] = ChainRecorder(
                    num_chains_to_save=self.num_chains_to_save,
                    snapshot_step_interval=self.snapshot_step_interval,
                    meta=meta,
                    field_prefix=name,
                )
            return recorders
        return ChainRecorder(
            num_chains_to_save=self.num_chains_to_save,
            snapshot_step_interval=self.snapshot_step_interval,
            meta=meta,
        )

    @staticmethod
    def _stash_recorder(
        pl_module: pl.LightningModule,
        recorder: ChainRecorder | dict[str, ChainRecorder],
    ) -> None:
        """Set the one-shot recorder handle on the module.

        The module's ``generate_graphs`` reads
        ``_pending_chain_recorder`` and clears it after forwarding to
        the sampler. The stash is local to a single validation pass and
        documented at the callback class level.
        """
        # ``setattr`` keeps pyright from complaining about an attribute
        # that LightningModule does not declare; using direct assignment
        # would require a pyright suppression that CLAUDE.md forbids in
        # source. The runtime semantics are identical.
        setattr(pl_module, "_pending_chain_recorder", recorder)  # noqa: B010

    @staticmethod
    def _finalize_recorder(
        recorder: ChainRecorder | dict[str, ChainRecorder],
    ) -> dict[str, Any] | None:
        """Convert the active recorder(s) into a single artefact dict.

        Returns ``None`` if no snapshots were recorded -- typically
        because the validation pass did not actually run sampling
        (e.g. the module's ``eval_every_n_steps`` gate skipped this
        epoch). A finalize-with-no-records is a wiring smell on a
        snapshot epoch, but the callback tolerates it because the
        validation cadence and the sampling cadence are independent
        gates set elsewhere.
        """
        if isinstance(recorder, ChainRecorder):
            try:
                return recorder.finalize()
            except RuntimeError:
                return None
        # dict-of-recorders: finalise each, drop those with no snapshots,
        # then merge. If every sub-recorder is empty, return None.
        parts: list[dict[str, Any]] = []
        for sub in recorder.values():
            try:
                parts.append(sub.finalize())
            except RuntimeError:
                continue
        if not parts:
            return None
        return merge_chain_snapshots(parts)

    def _write_artefact(
        self,
        payload: dict[str, Any],
        *,
        trainer: pl.Trainer,
        epoch: int,
    ) -> None:
        """Resolve the path and persist the artefact via ``torch.save``."""
        target = _resolve_chain_path(
            configured=self.chain_save_path,
            run_name=self.run_name,
            default_root_dir=trainer.default_root_dir,
            epoch=epoch,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)

    @override
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Snapshot only on cadence-matching passes. The counter
        # advances unconditionally in on_validation_end so successive
        # passes still hit the gate at the right intervals.
        if self._validation_pass_count % self.chain_save_every_n_val != 0:
            return
        recorder = self._build_recorder_for_module(trainer, pl_module)
        self._active_recorder = recorder
        self._stash_recorder(pl_module, recorder)

    @override
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        recorder = self._active_recorder
        # Always advance the pass counter, even when no recorder fired,
        # so the cadence gate stays accurate across runs.
        self._validation_pass_count += 1
        if recorder is None:
            return
        # Clear the one-shot stash on the module first so the next
        # generate_graphs call cannot accidentally re-use it.
        if hasattr(pl_module, "_pending_chain_recorder"):
            delattr(pl_module, "_pending_chain_recorder")
        self._active_recorder = None

        payload = self._finalize_recorder(recorder)
        if payload is None:
            return
        self._write_artefact(
            payload,
            trainer=trainer,
            epoch=int(trainer.current_epoch),
        )

    @override
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.chain_save_at_fit_end:
            return
        if not trainer.is_global_zero:
            # DDP rank gate -- only rank 0 writes the artefact, mirroring
            # FinalSampleDumpCallback's spec resolution Q9.
            return

        recorder = self._build_recorder_for_module(trainer, pl_module)

        # The fit-end pass owns its own sampling: there is no validation
        # epoch in flight to thread the recorder through, so we drive
        # the sampler directly via the module's generate_graphs entry
        # point. The module already validates its sampler / noise_process
        # collaborators and forwards the recorder to Sampler.sample.
        evaluator = _require_attr(pl_module, "evaluator")
        num_samples = max(int(self.num_chains_to_save), int(evaluator.eval_num_samples))

        ema_cb = self._find_ema_callback(trainer)
        ema_active = False
        if ema_cb is not None and ema_cb.ema is not None:
            backbone = _require_attr(pl_module, "model")
            ema_cb.ema.store(backbone.parameters())
            ema_cb.ema.copy_to(backbone.parameters())
            ema_active = True

        try:
            self._stash_recorder(pl_module, recorder)
            try:
                # generate_graphs returns NetworkX graphs we discard --
                # the side effect we care about is the recorder receiving
                # snapshots. Resolved via _require_attr so pyright sees
                # an Any-typed callable rather than a stubbed-out
                # LightningModule attribute.
                generate_graphs_fn = _require_attr(pl_module, "generate_graphs")
                generate_graphs_fn(num_samples)
            finally:
                if hasattr(pl_module, "_pending_chain_recorder"):
                    delattr(pl_module, "_pending_chain_recorder")
        finally:
            if ema_active and ema_cb is not None and ema_cb.ema is not None:
                backbone = _require_attr(pl_module, "model")
                ema_cb.ema.restore(backbone.parameters())

        # Override meta for fit-end provenance with ema_active reflecting
        # the swap we just performed. _build_recorder_for_module captured
        # the pre-swap state, so re-stamp the meta on the recorder(s).
        meta = self._build_meta(trainer, pl_module, ema_active=ema_active)
        if isinstance(recorder, ChainRecorder):
            recorder._meta = dict(meta)  # noqa: SLF001
        else:
            for sub in recorder.values():
                sub._meta = dict(meta)  # noqa: SLF001

        payload = self._finalize_recorder(recorder)
        if payload is None:
            return
        # epoch=-1 sentinel routes the filename to ``fit_end_chains.pt``.
        self._write_artefact(payload, trainer=trainer, epoch=-1)
