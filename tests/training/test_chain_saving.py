"""Tests for ChainSavingCallback (parity #46 / D-16a).

Test rationale
--------------
The callback constructs a
:class:`~tmgg.diffusion.chain_recorder.ChainRecorder` (or per-sub-process
dict) on snapshot validation passes, threads it through the module's
``generate_graphs`` via a one-shot stash, and writes the artefact to a
resolved path on ``on_validation_end``. We verify:

* The constructor rejects nonsensical positive-integer parameters.
* The cadence gate (``chain_save_every_n_val``) skips non-snapshot
  passes and fires on the right ones.
* ``_resolve_chain_path`` honours the explicit > Modal mount > local
  fallback priority and pins the inlined Modal mount constant against
  the canonical ``OUTPUTS_MOUNT``.
* On a snapshot pass the callback constructs the recorder, the module
  uses it via the stash, and the written artefact carries the full
  meta dict (``global_step`` / ``epoch`` / ``T`` /
  ``snapshot_step_interval`` / ``noise_process`` qualname /
  ``ema_active``).
* ``on_fit_end`` writes a final artefact when
  ``chain_save_at_fit_end=True``.
* For composite noise processes the callback builds a per-sub-process
  recorder dict; meta is shared so reconverging via
  ``merge_chain_snapshots`` does not raise.

The end-to-end test uses a real CategoricalSampler reverse loop so the
recorder actually receives snapshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from tmgg.diffusion.chain_recorder import ChainRecorder
from tmgg.diffusion.noise_process import (
    CategoricalNoiseProcess,
    CompositeNoiseProcess,
    GaussianNoiseProcess,
)
from tmgg.diffusion.sampler import CategoricalSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel
from tmgg.training.callbacks.chain_saving import (
    ChainSavingCallback,
    _resolve_chain_path,
)
from tmgg.training.callbacks.ema import EMACallback
from tmgg.utils.noising.noise import GaussianNoise

# ---------------------------------------------------------------------------
# Constructor validity
# ---------------------------------------------------------------------------


class TestConstructorValidity:
    def test_rejects_zero_chains(self) -> None:
        with pytest.raises(ValueError, match="num_chains_to_save must be > 0"):
            ChainSavingCallback(
                num_chains_to_save=0,
                snapshot_step_interval=1,
                chain_save_every_n_val=1,
                chain_save_at_fit_end=True,
            )

    def test_rejects_negative_chains(self) -> None:
        with pytest.raises(ValueError, match="num_chains_to_save must be > 0"):
            ChainSavingCallback(
                num_chains_to_save=-1,
                snapshot_step_interval=1,
                chain_save_every_n_val=1,
                chain_save_at_fit_end=True,
            )

    def test_rejects_zero_step_interval(self) -> None:
        with pytest.raises(ValueError, match="snapshot_step_interval must be > 0"):
            ChainSavingCallback(
                num_chains_to_save=1,
                snapshot_step_interval=0,
                chain_save_every_n_val=1,
                chain_save_at_fit_end=True,
            )

    def test_rejects_zero_val_cadence(self) -> None:
        with pytest.raises(ValueError, match="chain_save_every_n_val must be > 0"):
            ChainSavingCallback(
                num_chains_to_save=1,
                snapshot_step_interval=1,
                chain_save_every_n_val=0,
                chain_save_at_fit_end=True,
            )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestResolveChainPath:
    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        out = _resolve_chain_path(
            configured=str(tmp_path / "explicit"),
            run_name="myrun",
            default_root_dir=tmp_path,
            epoch=3,
        )
        assert out == tmp_path / "explicit" / "epoch_3_chains.pt"

    def test_modal_context_uses_outputs_mount(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODAL_TASK_ID", "modal-task-12345")
        out = _resolve_chain_path(
            configured=None,
            run_name="r42",
            default_root_dir=tmp_path,
            epoch=7,
        )
        # Pin the inlined constant against the canonical Modal volume
        # mount; tach forbids tmgg.training -> tmgg.modal so the value
        # is duplicated by design.
        from tmgg.modal._lib.volumes import OUTPUTS_MOUNT
        from tmgg.training.callbacks.chain_saving import MODAL_OUTPUTS_MOUNT

        assert MODAL_OUTPUTS_MOUNT == OUTPUTS_MOUNT
        assert out == Path(OUTPUTS_MOUNT) / "chains" / "r42" / "epoch_7_chains.pt"

    def test_local_fallback_when_no_modal_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        out = _resolve_chain_path(
            configured=None,
            run_name="r42",
            default_root_dir=tmp_path,
            epoch=4,
        )
        assert out == tmp_path / "chains" / "epoch_4_chains.pt"

    def test_fit_end_uses_dedicated_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """epoch=-1 routes to fit_end_chains.pt regardless of context."""
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        out = _resolve_chain_path(
            configured=None,
            run_name="r42",
            default_root_dir=tmp_path,
            epoch=-1,
        )
        assert out == tmp_path / "chains" / "fit_end_chains.pt"


# ---------------------------------------------------------------------------
# Recorder construction (single + composite fan-out)
# ---------------------------------------------------------------------------


def _make_categorical_process(timesteps: int = 5) -> CategoricalNoiseProcess:
    return CategoricalNoiseProcess(
        schedule=NoiseSchedule("cosine_iddpm", timesteps=timesteps),
        x_classes=3,
        e_classes=2,
        limit_distribution="uniform",
    )


class _StubModuleWithProcess(pl.LightningModule):
    """Minimal module exposing the attributes the callback inspects."""

    def __init__(
        self,
        noise_process: Any,
        *,
        T: int = 5,
        num_nodes: int = 4,
    ) -> None:
        super().__init__()
        self.model: nn.Linear = nn.Linear(2, 1)
        self.noise_process = noise_process
        self.T = T
        self.num_nodes = num_nodes


class TestRecorderConstruction:
    def test_single_recorder_for_categorical_process(self) -> None:
        cb = ChainSavingCallback(
            num_chains_to_save=2,
            snapshot_step_interval=1,
            chain_save_every_n_val=1,
            chain_save_at_fit_end=False,
        )
        module = _StubModuleWithProcess(_make_categorical_process(), T=5)
        trainer = MagicMock()
        trainer.global_step = 100
        trainer.current_epoch = 3
        trainer.callbacks = []
        recorder = cb._build_recorder_for_module(trainer, module)
        assert isinstance(recorder, ChainRecorder)
        # Meta dict carries the full provenance.
        meta = recorder._meta  # noqa: SLF001
        assert meta["global_step"] == 100
        assert meta["epoch"] == 3
        assert meta["T"] == 5
        assert meta["snapshot_step_interval"] == 1
        assert meta["noise_process"].endswith("CategoricalNoiseProcess")
        assert meta["ema_active"] is False

    def test_composite_dispatch_emits_per_sub_recorder(self) -> None:
        cat = _make_categorical_process(timesteps=5)
        gauss = GaussianNoiseProcess(
            definition=GaussianNoise(),
            schedule=NoiseSchedule("cosine_iddpm", timesteps=5),
            fields=frozenset({"E_feat"}),
        )
        composite = CompositeNoiseProcess([cat, gauss])
        cb = ChainSavingCallback(
            num_chains_to_save=2,
            snapshot_step_interval=1,
            chain_save_every_n_val=1,
            chain_save_at_fit_end=False,
        )
        module = _StubModuleWithProcess(composite, T=5)
        trainer = MagicMock()
        trainer.global_step = 1
        trainer.current_epoch = 0
        trainer.callbacks = []
        recorder = cb._build_recorder_for_module(trainer, module)
        assert isinstance(recorder, dict)
        assert set(recorder.keys()) == {
            "CategoricalNoiseProcess",
            "GaussianNoiseProcess",
        }
        # All sub-recorders share the same meta so merge_chain_snapshots
        # does not raise when reconverging.
        metas = [r._meta for r in recorder.values()]  # noqa: SLF001
        assert metas[0] == metas[1]
        # Each sub-recorder's field_prefix matches its dict key.
        for name, sub in recorder.items():
            assert sub.field_prefix == name


# ---------------------------------------------------------------------------
# Cadence gate: validation_pass_count
# ---------------------------------------------------------------------------


class TestCadenceGate:
    """The gate uses ``count % chain_save_every_n_val == 0``.

    Pass 0 always fires (0 % N == 0 for any positive N); subsequent
    snapshot passes land at multiples of N.
    """

    def test_skips_non_snapshot_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        cb = ChainSavingCallback(
            num_chains_to_save=1,
            snapshot_step_interval=1,
            chain_save_every_n_val=3,
            chain_save_at_fit_end=False,
            chain_save_path=str(tmp_path),
        )
        module = _StubModuleWithProcess(_make_categorical_process(), T=5)
        trainer = MagicMock()
        trainer.global_step = 0
        trainer.current_epoch = 0
        trainer.callbacks = []
        trainer.default_root_dir = str(tmp_path)
        # Pass 0: fires (0 % 3 == 0).
        cb.on_validation_start(trainer, module)
        assert cb._active_recorder is not None
        # Without driving the sampler the recorder has no snapshots and
        # _finalize_recorder returns None, so on_validation_end skips
        # the write but still advances the counter.
        cb.on_validation_end(trainer, module)
        assert cb._validation_pass_count == 1

        # Passes 1 and 2: skipped.
        cb.on_validation_start(trainer, module)
        assert cb._active_recorder is None
        cb.on_validation_end(trainer, module)
        cb.on_validation_start(trainer, module)
        assert cb._active_recorder is None
        cb.on_validation_end(trainer, module)
        assert cb._validation_pass_count == 3

        # Pass 3: fires again.
        cb.on_validation_start(trainer, module)
        assert cb._active_recorder is not None


# ---------------------------------------------------------------------------
# End-to-end smoke (sampler + module path)
# ---------------------------------------------------------------------------


class _StubEvaluator:
    eval_num_samples = 4

    def to_networkx_graphs(self, samples: Any) -> list[Any]:
        return list(samples)


class _UniformCategoricalModel(GraphModel):
    """Trivial model returning uniform PMFs for the reverse-loop test.

    Sparse-default refactor: the production sampler passes
    ``output_dense=True``, so this stub returns a
    :class:`DenseGraphDistribution` (per-position probability content)
    over ``(B, n_max, *)``. The forward also accepts the universal
    :class:`GraphData` base — the categorical sampler hands it a
    ``DenseGraphState`` after coercion at the model boundary.
    """

    def __init__(self, dx: int, de: int) -> None:
        super().__init__()
        self.dx = dx
        self.de = de

    def get_config(self) -> dict[str, Any]:
        return {"dx": self.dx, "de": self.de}

    def forward(
        self,
        data: Any,
        t: Tensor | None = None,
        *,
        output_dense: bool = False,
    ) -> Any:
        from tmgg.data.datasets.graph_types import (
            DenseGraphDistribution,
            DenseGraphState,
        )
        from tmgg.models.base import _coerce_input_to

        _ = t
        # The sampler hands us either a sparse or dense state; coerce to
        # a dense state so the body math reads (B, n_max, …) directly.
        d = _coerce_input_to(data, target=DenseGraphState)
        assert isinstance(d, DenseGraphState)
        assert d.X_class is not None
        assert d.E_class is not None
        bs, n, _ = d.X_class.shape
        X = torch.ones(bs, n, self.dx, device=d.X_class.device) / self.dx
        E = torch.ones(bs, n, n, self.de, device=d.E_class.device) / self.de
        out_dense = DenseGraphDistribution(
            num_nodes_per_graph=d.num_nodes_per_graph,
            y=d.y,
            X_class=X,
            E_class=E,
        )
        if output_dense:
            return out_dense
        return out_dense.to_sparse()


class _StubDiffusionModule(pl.LightningModule):
    """Minimal module mirroring DiffusionModule's stash protocol.

    ``generate_graphs`` reads the ``_pending_chain_recorder`` stash,
    drives a real CategoricalSampler, and clears the stash. Lets the
    callback test exercise the recorder integration end-to-end without
    hauling in the full DiffusionModule infrastructure.
    """

    num_nodes: int = 4
    T: int = 5

    def __init__(self) -> None:
        super().__init__()
        self.model: _UniformCategoricalModel = _UniformCategoricalModel(dx=3, de=2)
        self.noise_process = _make_categorical_process(timesteps=self.T)
        self.sampler = CategoricalSampler()
        self.evaluator: _StubEvaluator = _StubEvaluator()

    def generate_graphs(self, num_graphs: int) -> list[Any]:
        chain_recorder = getattr(self, "_pending_chain_recorder", None)
        if chain_recorder is not None:
            del self._pending_chain_recorder
        return self.sampler.sample(
            model=self.model,
            noise_process=self.noise_process,
            num_graphs=num_graphs,
            num_nodes=self.num_nodes,
            device=torch.device("cpu"),
            chain_recorder=chain_recorder,
        )


def test_callback_writes_artefact_with_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot pass writes the artefact with the full meta dict."""
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    cb = ChainSavingCallback(
        num_chains_to_save=2,
        snapshot_step_interval=1,
        chain_save_every_n_val=1,
        chain_save_at_fit_end=False,
        chain_save_path=str(tmp_path),
        run_name="testrun",
    )
    module = _StubDiffusionModule()
    trainer = MagicMock()
    trainer.global_step = 999
    trainer.current_epoch = 7
    trainer.callbacks = [cb]
    trainer.default_root_dir = str(tmp_path)
    trainer.is_global_zero = True

    cb.on_validation_start(trainer, module)
    # Module's generate_graphs reads the one-shot stash; emulate the
    # diffusion module's on_validation_epoch_end call site.
    module.generate_graphs(2)
    cb.on_validation_end(trainer, module)

    target = tmp_path / "epoch_7_chains.pt"
    assert target.exists()
    payload = torch.load(target, weights_only=False)
    assert "E_chain" in payload
    assert "node_mask" in payload
    assert "step_indices" in payload
    meta = payload["meta"]
    assert meta["global_step"] == 999
    assert meta["epoch"] == 7
    assert meta["T"] == 5
    assert meta["snapshot_step_interval"] == 1
    assert meta["noise_process"].endswith("CategoricalNoiseProcess")
    assert meta["ema_active"] is False


def test_callback_at_fit_end_writes_final_artefact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """on_fit_end writes a fit_end_chains.pt artefact when enabled."""
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    cb = ChainSavingCallback(
        num_chains_to_save=2,
        snapshot_step_interval=1,
        chain_save_every_n_val=1,
        chain_save_at_fit_end=True,
        chain_save_path=str(tmp_path),
        run_name="testrun",
    )
    module = _StubDiffusionModule()
    trainer = MagicMock()
    trainer.global_step = 5000
    trainer.current_epoch = 50
    trainer.callbacks = [cb]
    trainer.default_root_dir = str(tmp_path)
    trainer.is_global_zero = True

    cb.on_fit_end(trainer, module)

    target = tmp_path / "fit_end_chains.pt"
    assert target.exists()
    payload = torch.load(target, weights_only=False)
    assert "E_chain" in payload
    assert payload["meta"]["global_step"] == 5000


def test_fail_loud_when_module_lacks_noise_process(tmp_path: Path) -> None:
    """A module missing .noise_process surfaces a clear TypeError."""
    cb = ChainSavingCallback(
        num_chains_to_save=1,
        snapshot_step_interval=1,
        chain_save_every_n_val=1,
        chain_save_at_fit_end=False,
        chain_save_path=str(tmp_path),
    )
    module = pl.LightningModule()
    trainer = MagicMock()
    trainer.callbacks = []
    with pytest.raises(TypeError, match="`.noise_process`"):
        cb.on_validation_start(trainer, module)


def test_ema_active_flag_reflects_registered_ema(tmp_path: Path) -> None:
    """When EMACallback is also registered (and active), meta records it."""
    ema_cb = EMACallback(decay=0.5)
    # Simulate an active EMA (post on_fit_start).
    from tmgg.training.ema import ExponentialMovingAverage

    dummy_params = [nn.Parameter(torch.zeros(1))]
    ema_cb.ema = ExponentialMovingAverage(dummy_params, decay=0.5)

    cb = ChainSavingCallback(
        num_chains_to_save=1,
        snapshot_step_interval=1,
        chain_save_every_n_val=1,
        chain_save_at_fit_end=False,
        chain_save_path=str(tmp_path),
    )
    module = _StubModuleWithProcess(_make_categorical_process(), T=5)
    trainer = MagicMock()
    trainer.global_step = 1
    trainer.current_epoch = 0
    trainer.callbacks = [ema_cb, cb]
    recorder = cb._build_recorder_for_module(trainer, module)
    assert isinstance(recorder, ChainRecorder)
    assert recorder._meta["ema_active"] is True  # noqa: SLF001
