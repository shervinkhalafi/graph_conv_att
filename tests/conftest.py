"""Shared pytest fixtures for integration tests.

These fixtures provide common setup for testing experiment runners and
config composition without running full training.
"""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def quick_training_overrides(tmp_path: Path) -> list[str]:
    """Hydra overrides for minimal training runs.

    Configures experiments to complete in ~10 seconds by limiting steps,
    disabling logging, and using minimal data on CPU.
    """
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=2",
        "data.num_workers=0",
        "++data.samples_per_graph=4",
        "++data.num_train_samples=8",
        "++data.num_val_samples=4",
        "++data.num_test_samples=4",
        "++data.graph_config.num_graphs=4",
        f"hydra.run.dir={tmp_path}",
    ]


@pytest.fixture
def sample_adjacency_batch() -> torch.Tensor:
    """Generate a small batch of adjacency matrices for forward pass tests.

    Creates 4 random symmetric adjacency matrices (20x20) suitable for
    testing model forward passes.
    """
    batch_size = 4
    n_nodes = 20

    matrices = []
    for _ in range(batch_size):
        # Random symmetric matrix with ~30% edge density
        A = torch.bernoulli(torch.full((n_nodes, n_nodes), 0.3))
        A = (A + A.T) / 2
        A.fill_diagonal_(0)
        A = (A > 0.5).float()
        matrices.append(A)

    return torch.stack(matrices)


@pytest.fixture
def config_path() -> Path:
    """Path to the experiment configs directory."""
    return Path(__file__).parent.parent / "src" / "tmgg" / "experiments" / "exp_configs"


@pytest.fixture
def require_modal_profile():
    """Skip unless the expected Modal profile is active and graph-denoising is deployed.

    Verifies three conditions:
    1. ``modal`` CLI is installed
    2. The expected profile, named by the ``TMGG_MODAL_PROFILE`` environment
       variable, is current (prevents accidental runs against the wrong
       account). The test is skipped when the variable is unset, so these live
       Modal tests are opt-in — set ``TMGG_MODAL_PROFILE`` to your profile to
       enable them (avoids launching real Modal containers by accident).
    3. The ``graph-denoising`` app is deployed (``modal_run_cli`` hydrates
       successfully)
    """
    import os
    import shutil
    import subprocess

    if not shutil.which("modal"):
        pytest.skip("modal CLI not installed")

    expected_profile = os.environ.get("TMGG_MODAL_PROFILE")
    if not expected_profile:
        pytest.skip("TMGG_MODAL_PROFILE not set (live Modal tests are opt-in)")

    result = subprocess.run(
        ["modal", "profile", "current"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or expected_profile not in result.stdout:
        pytest.skip(f"Wrong Modal profile: {result.stdout.strip()!r}")

    try:
        import modal

        fn = modal.Function.from_name("graph-denoising", "modal_run_cli")
        fn.hydrate()
    except Exception as exc:
        pytest.skip(f"graph-denoising not deployed: {exc}")
