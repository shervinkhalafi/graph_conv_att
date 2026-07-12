"""Tests for discrete checkpoint evaluation CLI helpers.

Test rationale
--------------
The evaluation helper is a small orchestration layer: it loads a checkpoint,
optionally swaps in EMA shadow weights, fetches reference graphs from the
training-time datamodule (``--reference_set val|test``), instantiates the
trainer's saved :class:`~tmgg.evaluation.graph_evaluator.GraphEvaluator`
from the same sibling ``config.yaml``, samples generated graphs, and
delegates the full metric computation (degree/clustering/spectral MMD plus
orbit/SBM/block-structure/planarity/uniqueness when their dependencies
are available). Tests cover:

* The evaluator-loader symbol is module-level so callers can patch it.
* The evaluator drives the datamodule through ``setup`` + ``get_reference_graphs``
  rather than the deprecated synthetic ``generate_reference_graphs`` path
  (regression for parity D-16c cleanup; the synthetic shortcut bypassed
  the val/test distinction).
* ``--reference_set`` flag plumbs through to the datamodule call.
* ``--use_ema {auto,true,false}`` semantics: auto swaps when shadow
  present, true raises when shadow absent, false skips the swap.
* ``mmd_results`` carries the full ``EvaluationResults.to_dict()`` shape
  (the three core MMDs plus the optional orbit/sbm/block-structure
  fields, possibly ``None`` when their backends or train graphs are
  absent), not the stripped 3-field dict from the previous
  ``compute_mmd_metrics`` shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from tmgg.experiments.discrete_diffusion_generative import evaluate_cli


@dataclass
class _DummyEvalResults:
    """Mirrors the public surface of :class:`EvaluationResults`.

    Carries the full flat dict with sentinel values for every field;
    optional metrics are ``None`` to model the realistic case of orca /
    graph-tool unavailable and novelty unsupported by the CLI path.
    """

    degree_mmd: float = 0.1
    clustering_mmd: float = 0.2
    spectral_mmd: float = 0.3
    orbit_mmd: float | None = None
    sbm_accuracy: float | None = None
    planarity_accuracy: float | None = None
    uniqueness: float | None = None
    novelty: float | None = None
    modularity_q: float | None = None
    spectral_gap_l2: float | None = None
    empirical_p_in: float | None = None
    empirical_p_out: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "degree_mmd": self.degree_mmd,
            "clustering_mmd": self.clustering_mmd,
            "spectral_mmd": self.spectral_mmd,
            "orbit_mmd": self.orbit_mmd,
            "sbm_accuracy": self.sbm_accuracy,
            "planarity_accuracy": self.planarity_accuracy,
            "uniqueness": self.uniqueness,
            "novelty": self.novelty,
            "modularity_q": self.modularity_q,
            "spectral_gap_l2": self.spectral_gap_l2,
            "empirical_p_in": self.empirical_p_in,
            "empirical_p_out": self.empirical_p_out,
        }


class _DummyEvaluator:
    """Stand-in for ``GraphEvaluator`` exposing the surface the CLI uses."""

    def __init__(self) -> None:
        self.kernel = "gaussian_tv"
        self.sigma = 1.0
        self.evaluate_calls: list[tuple[int, int]] = []

    def evaluate(
        self, refs: list[nx.Graph[Any]], generated: list[nx.Graph[Any]]
    ) -> _DummyEvalResults:
        self.evaluate_calls.append((len(refs), len(generated)))
        return _DummyEvalResults()


class _DummyModule:
    """Stand-in for DiffusionModule with enough surface for evaluate_checkpoint."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 2)

    def to(self, device: str) -> _DummyModule:
        assert device == "cpu"
        return self

    def eval(self) -> _DummyModule:
        return self

    def generate_graphs(
        self, num_samples: int, chunk_size: int | None = None
    ) -> list[nx.Graph[Any]]:
        del chunk_size  # accepted for API parity with DiffusionModule.generate_graphs
        return [nx.path_graph(4) for _ in range(num_samples)]


class _DummyDatamodule:
    """Records setup calls and serves canned NetworkX references."""

    def __init__(self, ref_graphs: list[nx.Graph[Any]]) -> None:
        self._refs = ref_graphs
        self.setup_calls: list[str] = []
        self.get_reference_calls: list[tuple[str, int]] = []

    def setup(self, stage: str) -> None:
        self.setup_calls.append(stage)

    def get_reference_graphs(self, split: str, max_graphs: int) -> list[nx.Graph[Any]]:
        self.get_reference_calls.append((split, max_graphs))
        return self._refs[:max_graphs]


def _write_dummy_config_yaml(checkpoint_path: Path, *, _target_: str) -> Path:
    """Write a sibling config.yaml with a `data:` block containing _target_.

    The CLI walks one directory up from the checkpoint to find this file.
    Returns the config path written.
    """
    run_dir = checkpoint_path.parent.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.create({"data": {"_target_": _target_, "batch_size": 4}})
    config_path = run_dir / "config.yaml"
    OmegaConf.save(config, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Refactor regression tests: datamodule path, no synthetic fallback
# ---------------------------------------------------------------------------


class TestDatamoduleReferencePath:
    """evaluate_checkpoint must drive the training-time datamodule."""

    def test_uses_val_dataloader_when_reference_set_val(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reference_set='val' -> datamodule.setup('fit') + get_reference_graphs('val', N)."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10-val_loss=0.5.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.cycle_graph(5) for _ in range(3)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "_load_evaluator", lambda _path: _DummyEvaluator()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=3,
            reference_set="val",
        )

        assert dm.setup_calls == ["fit"]
        assert dm.get_reference_calls == [("val", 3)]
        assert result["reference_set"] == "val"
        assert result["num_reference"] == 3

    def test_uses_test_dataloader_when_reference_set_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reference_set='test' -> datamodule.setup('test') + get_reference_graphs('test', N)."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.path_graph(4) for _ in range(2)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "_load_evaluator", lambda _path: _DummyEvaluator()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=2,
            reference_set="test",
        )

        assert dm.setup_calls == ["test"]
        assert dm.get_reference_calls == [("test", 2)]
        assert result["reference_set"] == "test"

    def test_missing_config_yaml_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing sibling config.yaml fails loud; no synthetic-fallback path exists."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        # Note: no config.yaml written.

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))

        with pytest.raises(FileNotFoundError, match="No config.yaml found"):
            evaluate_cli.evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                num_samples=2,
                reference_set="val",
            )


# ---------------------------------------------------------------------------
# EMA wiring (parity D-16c Q7)
# ---------------------------------------------------------------------------


class TestEmaWiring:
    """evaluate_checkpoint honours the use_ema flag against checkpoint state."""

    def test_use_ema_true_with_no_shadow_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        # Checkpoint has callbacks but no EMA entry.
        monkeypatch.setattr(
            evaluate_cli,
            "torch",
            _build_torch_stub(callbacks={"ModelCheckpoint{...}": {}}),
        )

        with pytest.raises(RuntimeError, match="use_ema=true requested"):
            evaluate_cli.evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                num_samples=2,
                reference_set="val",
                use_ema="true",
            )

    def test_use_ema_auto_with_no_shadow_records_inactive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.cycle_graph(5)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "_load_evaluator", lambda _path: _DummyEvaluator()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=1,
            reference_set="val",
            use_ema="auto",
        )

        assert result["ema_active"] is False

    def test_use_ema_false_skips_swap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.cycle_graph(5)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        # Even with an EMA entry, use_ema='false' must not swap.
        monkeypatch.setattr(
            evaluate_cli,
            "torch",
            _build_torch_stub(
                callbacks={
                    "EMACallback{decay=0.999}": {
                        "shadow_params": [],
                        "decay": 0.999,
                    }
                }
            ),
        )
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "_load_evaluator", lambda _path: _DummyEvaluator()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=1,
            reference_set="val",
            use_ema="false",
        )

        assert result["ema_active"] is False


def _build_torch_stub(*, callbacks: dict[str, Any]) -> Any:
    """Return a `torch`-shaped namespace whose `torch.load` returns a stub ckpt.

    The CLI calls ``torch.load(checkpoint_path, ...)`` after Lightning's
    ``load_from_checkpoint`` to inspect the callback section. Stub it
    out with a dict containing the desired callbacks block; everything
    else (no_grad, the `torch` namespace) is forwarded to the real
    torch module so generate_graphs / model code keeps working.
    """
    real_torch = torch

    class _TorchStub:
        @staticmethod
        def load(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"callbacks": callbacks}

        # Forward attributes used downstream.
        def __getattr__(self, name: str) -> Any:
            return getattr(real_torch, name)

    return _TorchStub()


# ---------------------------------------------------------------------------
# Real-checkpoint integration
# ---------------------------------------------------------------------------


def _train_tiny_discrete_checkpoint(tmp_path: Path) -> Path:
    """Train one tiny diffusion step and persist a real Lightning checkpoint.

    Test rationale
    --------------
    The evaluation CLI must load the same checkpoint format produced by the
    discrete training stack. A synthetic or hand-built checkpoint would miss
    the real hyperparameter layout that previously triggered the load failure.
    """
    from tmgg.data.data_modules.synthetic_categorical import (
        SyntheticCategoricalDataModule,
    )
    from tmgg.diffusion.noise_process import CategoricalNoiseProcess
    from tmgg.diffusion.sampler import CategoricalSampler
    from tmgg.diffusion.schedule import NoiseSchedule
    from tmgg.evaluation.graph_evaluator import GraphEvaluator
    from tmgg.models.digress.transformer_model import GraphTransformer
    from tmgg.training.lightning_modules.diffusion_module import DiffusionModule

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tiny-step={step}",
        save_top_k=0,
        save_last=True,
        every_n_train_steps=1,
    )

    schedule = NoiseSchedule("cosine_iddpm", timesteps=5)
    noise_process = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=2,
        e_classes=2,
        limit_distribution="uniform",
    )

    module = DiffusionModule(
        model=GraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
            output_dims={"X": 2, "E": 2, "y": 0},
            use_timestep=True,
        ),
        noise_process=noise_process,
        sampler=CategoricalSampler(),
        noise_schedule=schedule,
        evaluator=GraphEvaluator(eval_num_samples=2, kernel="gaussian", sigma=1.0),
        loss_type="cross_entropy",
        num_nodes=8,
        eval_every_n_steps=100,
    )
    datamodule = SyntheticCategoricalDataModule(
        num_nodes=8,
        num_graphs=40,
        batch_size=4,
        seed=42,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule)

    checkpoint_path = checkpoint_dir / "last.ckpt"
    assert checkpoint_path.exists()

    # Mirror run_experiment: persist the model + data blocks as sibling
    # config.yaml so evaluate_checkpoint can hydra-instantiate the module
    # for checkpoint loading and rebuild the datamodule for reference graphs.
    config = OmegaConf.create(
        {
            "model": {
                "_target_": "tmgg.training.lightning_modules.diffusion_module.DiffusionModule",
                "model": {
                    "_target_": "tmgg.models.digress.transformer_model.GraphTransformer",
                    "n_layers": 2,
                    "input_dims": {"X": 2, "E": 2, "y": 0},
                    "hidden_mlp_dims": {"X": 16, "E": 16, "y": 16},
                    "hidden_dims": {
                        "dx": 16,
                        "de": 16,
                        "dy": 16,
                        "n_head": 2,
                    },
                    "output_dims": {"X": 2, "E": 2, "y": 0},
                    "use_timestep": True,
                },
                "noise_schedule": {
                    "_target_": "tmgg.diffusion.schedule.NoiseSchedule",
                    "schedule_type": "cosine_iddpm",
                    "timesteps": 5,
                },
                "noise_process": {
                    "_target_": "tmgg.diffusion.noise_process.CategoricalNoiseProcess",
                    "schedule": "${model.noise_schedule}",
                    "x_classes": 2,
                    "e_classes": 2,
                    "limit_distribution": "uniform",
                },
                "sampler": {
                    "_target_": "tmgg.diffusion.sampler.CategoricalSampler",
                },
                "evaluator": {
                    "_target_": "tmgg.evaluation.graph_evaluator.GraphEvaluator",
                    "eval_num_samples": 2,
                    "kernel": "gaussian",
                    "sigma": 1.0,
                },
                "loss_type": "cross_entropy",
                "num_nodes": 8,
                "eval_every_n_steps": 100,
            },
            "data": {
                "_target_": "tmgg.data.data_modules.synthetic_categorical.SyntheticCategoricalDataModule",
                "num_nodes": 8,
                "num_graphs": 40,
                "batch_size": 4,
                "seed": 42,
            },
        }
    )
    OmegaConf.save(config, tmp_path / "config.yaml")
    return checkpoint_path


def test_evaluate_checkpoint_loads_real_diffusion_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Discrete evaluation should load a real tiny training checkpoint.

    Starting state
    --------------
    ``DiffusionModule`` checkpoints do not store the nested graph model as a
    constructor hyperparameter. The evaluation helper rebuilds that model
    from checkpoint metadata before calling ``load_from_checkpoint``, then
    rebuilds the datamodule from the sibling config.yaml.

    Invariants
    ----------
    The CLI must surface the FULL :class:`EvaluationResults` dictionary
    (the three core MMDs plus the optional orbit/sbm/block-structure
    fields) on the ``mmd_results`` key. The optional fields may be
    ``None`` when their backing dependency (orca, graph-tool) or
    train-graph input is unavailable in the test environment, so the
    assertion is permissive on values but strict on the key set.
    """
    checkpoint_path = _train_tiny_discrete_checkpoint(tmp_path)

    monkeypatch.setattr(
        evaluate_cli,
        "_load_evaluator",
        lambda _path: _DummyEvaluator(),
    )

    results = evaluate_cli.evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        num_samples=2,
        reference_set="val",
        mmd_kernel="gaussian_tv",
        mmd_sigma=0.5,
        device="cpu",
    )

    assert results["checkpoint_name"] == "last.ckpt"
    assert results["num_generated"] >= 1
    assert results["reference_set"] == "val"

    mmd = results["mmd_results"]
    # Core MMDs must always be populated.
    assert {"degree_mmd", "clustering_mmd", "spectral_mmd"}.issubset(mmd)
    assert mmd["degree_mmd"] == 0.1
    assert mmd["clustering_mmd"] == 0.2
    assert mmd["spectral_mmd"] == 0.3

    # Optional metric keys must be present in the dict (value may be None
    # when the underlying dependency is missing, which mirrors the real
    # GraphEvaluator contract).
    optional_keys = {
        "orbit_mmd",
        "sbm_accuracy",
        "planarity_accuracy",
        "uniqueness",
        "novelty",
        "modularity_q",
        "spectral_gap_l2",
        "empirical_p_in",
        "empirical_p_out",
    }
    assert optional_keys.issubset(mmd)


def test_lightning_load_from_checkpoint_directly_raises_typeerror(
    tmp_path: Path,
) -> None:
    """Regression: pin the bug that ``_load_diffusion_module`` works around.

    ``DiffusionModule.save_hyperparameters(ignore=["model", "noise_process",
    "sampler", "noise_schedule", "evaluator"])`` excludes five ``nn.Module``
    constructor args from saved hparams. Lightning's
    ``DiffusionModule.load_from_checkpoint(path)`` reconstructs via
    ``cls(**hparams)`` and consequently raises::

        TypeError: DiffusionModule.__init__() missing 3 required keyword-only
        arguments: 'model', 'noise_process', and 'noise_schedule'

    The CLI's ``_load_diffusion_module`` bypasses this via
    hydra-instantiate-from-config + ``module.load_state_dict``. This test
    pins the original failure so a future refactor that re-introduces
    ``load_from_checkpoint`` would tripwire here, AND that our helper
    succeeds where the direct Lightning path fails.
    """
    from tmgg.training.lightning_modules.diffusion_module import DiffusionModule

    checkpoint_path = _train_tiny_discrete_checkpoint(tmp_path)

    # Direct Lightning path: provably broken for DiffusionModule checkpoints.
    with pytest.raises(
        TypeError, match="missing.*'(model|noise_process|noise_schedule)'"
    ):
        DiffusionModule.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    # Our helper path: succeeds via hydra-instantiate from sibling config.yaml.
    module = evaluate_cli._load_diffusion_module(checkpoint_path, device="cpu")
    assert isinstance(module, DiffusionModule)
    assert not module.training  # _load_diffusion_module returns module.eval()
