"""Stage 2 telemetry: cheap optimizer-health metrics.

``DiffusionModule._compute_grad_health_by_block`` bundles four
calculations: per-block grad cosine vs the previous step, per-block
grad SNR ``mean²/var``, global ``effective_lr = lr × ‖∇‖ / ‖θ‖``, and
the trailing update-to-weight ratio. The first eight tests pin the
*math* in isolation. Delivery — every cadence hit reaching the logger
exactly once, regardless of ``log_every_n_steps`` alignment — is pinned
end-to-end by ``test_diffusion_module_opt_health_delivery_is_cadence_exact``
(production module, via ``write_through=True``); the mechanism itself is
covered in ``test_write_through_logging.py``. The remaining test pins the
upstream Lightning flush/wipe behavior that motivated write-through.
"""

from __future__ import annotations

import torch


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (a * b).sum() / (a.norm() * b.norm()).clamp(min=eps)


def test_grad_cosine_identical_grads_returns_one() -> None:
    """Two identical gradient vectors ⇒ cosine = 1."""
    g = torch.tensor([1.0, -2.0, 3.0])
    assert torch.isclose(_cosine(g, g), torch.tensor(1.0))


def test_grad_cosine_opposite_grads_returns_minus_one() -> None:
    """Antiparallel ⇒ cosine = -1. Catches a sign-handling bug."""
    g = torch.tensor([1.0, -2.0, 3.0])
    assert torch.isclose(_cosine(g, -g), torch.tensor(-1.0))


def test_grad_cosine_orthogonal_grads_returns_zero() -> None:
    """Two orthogonal gradient vectors ⇒ cosine = 0."""
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert torch.isclose(_cosine(a, b), torch.tensor(0.0), atol=1e-7)


def test_grad_cosine_zero_norm_does_not_nan() -> None:
    """Cosine of a zero gradient against any other ⇒ 0, not NaN.

    The clamp on the denominator is what guarantees this: a literal
    ``0/0`` would otherwise propagate NaN through the dashboard.
    """
    z = torch.zeros(3)
    g = torch.tensor([1.0, 2.0, 3.0])
    cos = _cosine(z, g)
    assert torch.isfinite(cos)


def test_grad_snr_constant_grad_is_high() -> None:
    """Constant-valued gradient ⇒ var → 0 ⇒ SNR clamps to a large value.

    With ``var.clamp(min=1e-12)`` the ratio is ``mean² / 1e-12 = 1e12``
    for ``mean = 1.0``. Functions as the "all elements consistent"
    qualitative signal we want.
    """
    g = torch.full((100,), 1.0)
    n = g.numel()
    mean = g.sum() / n
    var = g.pow(2).sum() / n - mean.pow(2)
    snr = mean.pow(2) / var.clamp(min=1e-12)
    assert snr.item() > 1e9


def test_grad_snr_zero_mean_is_zero() -> None:
    """Zero-mean gradient (Σg = 0) gives SNR = 0 regardless of variance.

    Catches "noise dominates" cases — the metric is doing its job
    when it reads ~0 in this regime.
    """
    g = torch.tensor([1.0, -1.0, 1.0, -1.0])
    n = g.numel()
    mean = g.sum() / n
    var = g.pow(2).sum() / n - mean.pow(2)
    snr = mean.pow(2) / var.clamp(min=1e-12)
    assert snr.item() < 1e-6


def test_update_to_weight_zero_when_no_step() -> None:
    """If ``θ_after == θ_before`` the ratio is 0 — sanity check on the
    sign and the use of ``clamp(min=1e-12)`` in the denominator."""
    theta = torch.randn(10)
    delta = (theta - theta).norm()
    weight_norm = theta.norm()
    ratio = delta / weight_norm.clamp(min=1e-12)
    assert ratio.item() == 0.0


def test_effective_lr_formula_matches_documented_definition() -> None:
    """``lr × ‖∇‖ / ‖θ‖`` agrees with hand-computed values."""
    lr = 0.01
    g = torch.tensor([3.0, 4.0])  # ‖∇‖ = 5
    theta = torch.tensor([0.0, 1.0, 0.0, 0.0])  # ‖θ‖ = 1
    eff = lr * g.norm() / theta.norm().clamp(min=1e-12)
    assert torch.isclose(eff, torch.tensor(0.05))


def test_optimizer_health_metrics_survive_short_epochs() -> None:
    """Historical contract: the aligned stash/drain pattern does deliver.

    Documents the Lightning behavior that motivated write-through
    logging: logs written from ``on_before_optimizer_step`` get wiped by
    ``reset_results()`` at every epoch boundary unless drained from
    ``on_train_batch_end`` on an iteration that coincides with a
    ``log_every_n_steps`` flush boundary (here cadence=5 is *not* a
    multiple of log_every_n_steps=2, but lcm coincidences within 33
    steps still deliver — the pattern only works when the alignment
    holds, which is exactly the fragility that write-through removes).

    Production no longer uses this pattern: ``DiffusionModule`` logs
    cadence-gated diagnostics with ``self.log(..., write_through=True)``
    (see ``WriteThroughLogMixin`` and
    ``test_diffusion_module_opt_health_delivery_is_cadence_exact``
    below). This test keeps the upstream behavioral assumption pinned so
    a Lightning upgrade that changes the flush/reset semantics is
    noticed here, not on a dashboard.
    """
    import pytorch_lightning as pl

    class _Model(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
            self._opt_health_payload: dict[str, torch.Tensor] | None = None
            self._cadence = 5

        def training_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
            x, y = batch
            loss = ((self.linear(x) - y) ** 2).mean()
            self.log("train/loss", loss, on_step=True)
            return loss

        def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
            # Mirror the production cadence trigger.
            if (self.global_step + 1) % self._cadence == 0:
                self._opt_health_payload = {
                    "train/grad_snr/_total": torch.tensor(2.0 + self.global_step),
                    "train/effective_lr": torch.tensor(0.001 + self.global_step / 1000),
                }

        def on_train_batch_end(
            self,
            outputs: object,
            batch: object,
            batch_idx: int,
        ) -> None:
            if self.global_step % self._cadence == 0 and self._opt_health_payload:
                for k, v in self._opt_health_payload.items():
                    self.log(k, v, on_step=True)
                self._opt_health_payload = None

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    from pytorch_lightning.loggers import Logger as _PLLogger

    class _Capture(_PLLogger):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[tuple[int | None, dict[str, float]]] = []

        @property
        def name(self) -> str:
            return "capture"

        @property
        def version(self) -> str:
            return "1"

        def log_metrics(
            self,
            metrics: dict[str, float] | None = None,
            step: int | None = None,
            **kw: object,
        ) -> None:
            assert metrics is not None
            self.records.append((step, dict(metrics)))

        def log_hyperparams(self, *a: object, **k: object) -> None:
            pass

        def save(self) -> None:
            pass

    torch.manual_seed(0)
    # 33 samples / batch_size 3 = 11 batches per epoch — same as production
    # smoke-run dataset, where the bug originally manifested.
    ds = torch.utils.data.TensorDataset(torch.randn(33, 4), torch.randn(33, 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=3)
    cap = _Capture()
    # CPU-only: this test pins LightningModule cadence/flush math, not GPU
    # numerics. Earlier suite tests can leave torch.use_deterministic_algorithms
    # on, which makes CuBLAS matmul raise without CUBLAS_WORKSPACE_CONFIG; CPU
    # avoids that hazard entirely. The test runs in well under 5s either way.
    trainer = pl.Trainer(
        max_steps=33,
        log_every_n_steps=2,
        logger=cap,
        accelerator="cpu",
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_Model(), train_dataloaders=dl)

    # Cadence triggers at global_step ∈ {5, 10, 15, 20, 25, 30}.
    # Each one should appear in at least one flush event.
    flushed_keys = {k for _, m in cap.records for k in m}
    assert "train/grad_snr/_total" in flushed_keys, (
        "regression: grad_snr metric was wiped by reset_results between "
        "the cadence trigger and the next log_every_n_steps flush"
    )
    assert "train/effective_lr" in flushed_keys, (
        "regression: effective_lr metric was wiped by reset_results "
        "between the cadence trigger and the next log_every_n_steps flush"
    )


def test_diffusion_module_opt_health_delivery_is_cadence_exact() -> None:
    """End-to-end: every opt-health cadence hit reaches the logger, exactly
    once, stamped with the step it was computed at — independent of
    ``log_every_n_steps`` alignment.

    Rationale: the stash/drain workaround (ce5cc52c) only delivered
    cadence-gated diagnostics when the drain iteration coincided with a
    ``log_every_n_steps`` flush boundary; otherwise the epoch-end
    ``reset_results()`` wiped them. Here cadence (5) and
    ``log_every_n_steps`` (7) are coprime, so under the workaround
    delivery is at best incidental (lcm coincidences, Lightning-stamped);
    under write-through it is exact. This is the production-module
    counterpart of the mechanism tests in
    ``test_write_through_logging.py``.

    Expected stamps with cadence C=5 over 24 steps:

    * pre-step family (``effective_lr``, ``grad_snr/*``,
      ``grad_norm/{block}``) computed in ``on_before_optimizer_step``
      when ``(global_step + 1) % C == 0`` → stamped {4, 9, 14, 19};
    * post-step family (``weight_norm_total``, ``update_to_weight/*``)
      computed in ``on_train_batch_end`` when ``global_step % C == 0``
      → stamped {5, 10, 15, 20}.
    """
    import pytorch_lightning as pl

    from tmgg.data.data_modules.multigraph_data_module import MultiGraphDataModule
    from tmgg.diffusion.noise_process import ContinuousNoiseProcess
    from tmgg.diffusion.sampler import ContinuousSampler
    from tmgg.diffusion.schedule import NoiseSchedule
    from tmgg.evaluation.graph_evaluator import GraphEvaluator
    from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser
    from tmgg.training.lightning_modules.diffusion_module import DiffusionModule
    from tmgg.utils.noising.noise import DigressNoise

    from .test_write_through_logging import _Capture

    torch.manual_seed(0)
    schedule = NoiseSchedule(schedule_type="cosine_iddpm", timesteps=10)
    module = DiffusionModule(
        model=SelfAttentionDenoiser(k=8, d_k=16),
        noise_process=ContinuousNoiseProcess(
            definition=DigressNoise(), schedule=schedule
        ),
        sampler=ContinuousSampler(),
        noise_schedule=schedule,
        evaluator=GraphEvaluator(eval_num_samples=4, kernel="gaussian", sigma=1.0),
        loss_type="mse",
        num_nodes=16,
        eval_every_n_steps=1000,  # > max_steps: keeps loss_per_t bins out of the way
        log_optimizer_health_every_n_steps=5,
    )
    datamodule = MultiGraphDataModule(
        graph_type="sbm",
        num_nodes=16,
        num_graphs=20,
        batch_size=4,
        graph_config={"num_blocks": 2, "p_in": 0.7, "p_out": 0.1},
        seed=42,
    )
    cap = _Capture()
    trainer = pl.Trainer(
        max_steps=24,
        log_every_n_steps=7,  # coprime with the opt-health cadence of 5
        logger=cap,
        accelerator="cpu",
        limit_val_batches=0,  # plumbing test: skip sampling-heavy validation
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, datamodule)

    def stamps(key_prefix: str) -> list[int]:
        out: list[int] = []
        for step, metrics in cap.records:
            if any(k.startswith(key_prefix) for k in metrics):
                assert step is not None
                out.append(step)
        return out

    pre_step_hits = [s for s in range(24) if (s + 1) % 5 == 0]  # {4, 9, 14, 19}
    post_step_hits = [s + 1 for s in pre_step_hits]  # {5, 10, 15, 20}

    assert stamps("diagnostics-train/opt-health/effective_lr") == pre_step_hits
    assert stamps("diagnostics-train/opt-health/grad_snr/") == pre_step_hits
    assert stamps("diagnostics-train/opt-health/grad_norm/") == pre_step_hits
    assert stamps("diagnostics-train/opt-health/weight_norm_total") == post_step_hits
    assert stamps("diagnostics-train/opt-health/update_to_weight/") == post_step_hits
