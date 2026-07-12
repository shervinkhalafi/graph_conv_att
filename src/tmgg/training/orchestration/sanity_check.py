"""Sanity check utilities for verifying experimental setup.

Disabled by default. To enable, set ``sanity_check: true`` in your
experiment config (or pass ``sanity_check=true`` on the CLI). The
checks run before training starts and verify that the noise generator,
data loader, model forward pass, and loss computation all work
correctly. Diagnostic plots are saved to the output directory.

Example::

    uv run tmgg-spectral-arch sanity_check=true trainer.max_steps=0
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from tmgg.data.datasets.graph_types import GraphData
from tmgg.utils.noising.noise import NoiseDefinition


def _batch_to_tensor(batch: Any) -> torch.Tensor:
    """Extract a tensor from a dataloader batch.

    If *batch* is a ``GraphData`` instance (sparse ``GraphState`` or
    dense ``DenseGraphState``), returns its dense adjacency representation
    via :meth:`GraphData.dense_adjacency`. If it is already a ``Tensor``,
    returns it unchanged. Raises ``TypeError`` for anything else so callers
    fail fast.
    """
    if isinstance(batch, GraphData):
        return batch.dense_adjacency()
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError(
        f"Unexpected batch type {type(batch).__name__}; expected Tensor or GraphData"
    )


logger: logging.Logger = logging.getLogger(__name__)


class SanityCheckableModel(Protocol):
    """Model with attributes needed for sanity checking.

    Any model passed to ``maybe_run_sanity_check`` must expose a
    ``noise_process`` (whose ``.definition`` attribute is a
    ``NoiseDefinition``) and a ``criterion``. Callers are expected to
    also pass an ``nn.Module`` (the function asserts this at runtime).
    """

    noise_process: Any
    criterion: nn.Module


@dataclass
class SanityCheckResult:
    """Container for sanity check results."""

    passed: bool = True
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def add_check(self, name: str, passed: bool, message: str) -> None:
        """Add a check result."""
        _ = self.messages.append(f"[{'PASS' if passed else 'FAIL'}] {name}: {message}")
        if not passed:
            self.passed = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        _ = self.warnings.append(f"[WARN] {message}")

    def add_metric(self, name: str, value: float) -> None:
        """Add a metric value."""
        self.metrics[name] = value

    def __str__(self) -> str:
        """String representation of results."""
        lines: list[str] = ["=== Sanity Check Results ==="]
        _ = lines.extend(self.messages)
        if self.warnings:
            _ = lines.append("\nWarnings:")
            _ = lines.extend(self.warnings)
        if self.metrics:
            _ = lines.append("\nMetrics:")
            for k, v in self.metrics.items():
                _ = lines.append(f"  {k}: {v:.4f}")
        _ = lines.append(f"\nOverall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def _check_noise_generator(
    noise_generator: NoiseDefinition,
    sample_size: int = 10,
    noise_levels: list[float] | None = None,
) -> SanityCheckResult:
    """
    Verify noise generator is working correctly.

    Args:
        noise_generator: Noise generator to test
        sample_size: Size of test adjacency matrix
        noise_levels: Noise levels to test

    Returns:
        SanityCheckResult with test outcomes
    """
    if noise_levels is None:
        noise_levels = [0.1, 0.3, 0.5]

    result: SanityCheckResult = SanityCheckResult()

    # For rotation noise generator, use its configured k value
    from tmgg.utils.noising.noise import RotationNoise

    if isinstance(noise_generator, RotationNoise):
        sample_size = noise_generator.k

    # Create test adjacency matrix
    A: torch.Tensor = (
        torch.eye(sample_size) + torch.rand(sample_size, sample_size) * 0.3
    )
    A = (A + A.T) / 2  # Make symmetric
    A = (A > 0.5).float()  # Binarize

    # Test different noise levels
    for eps in noise_levels:
        A_noisy: torch.Tensor = noise_generator.add_noise(A, eps)

        # Check output shape
        shape_ok: bool = A_noisy.shape == A.shape
        _ = result.add_check(
            f"Output shape (eps={eps})",
            shape_ok,
            f"A_noisy: {A_noisy.shape}, expected: {A.shape}",
        )

        # Check noise was actually added
        diff: float = float(torch.abs(A_noisy - A).sum().item())
        noise_added: bool = diff > 0
        _ = result.add_check(
            f"Noise added (eps={eps})", noise_added, f"Total difference: {diff:.4f}"
        )

        # Check that noisy matrix is still valid (symmetric, etc)
        symmetry_error: float = float(torch.norm(A_noisy - A_noisy.T).item())
        is_symmetric: bool = symmetry_error < 1e-6
        _ = result.add_check(
            f"Output symmetry (eps={eps})",
            is_symmetric,
            f"Symmetry error: {symmetry_error:.6f}",
        )

        # Store metrics
        _ = result.add_metric(
            f"noise_amount_eps_{eps}", diff / (sample_size * sample_size)
        )

    return result


def _check_model_forward_pass(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    device: str | torch.device = "cpu",
) -> SanityCheckResult:
    """
    Verify model forward pass works correctly.

    Args:
        model: Model to test
        input_shape: Expected input shape
        device: Device to run on

    Returns:
        SanityCheckResult with test outcomes
    """
    result: SanityCheckResult = SanityCheckResult()

    # Move model to device
    _ = model.to(device)
    _ = model.eval()

    # Create test input
    test_input: torch.Tensor
    if len(input_shape) == 2:
        # Single adjacency matrix
        test_input = torch.eye(input_shape[0]).unsqueeze(0).to(device)
    elif len(input_shape) == 3:
        # Batch of adjacency matrices or eigenvectors
        if input_shape[2] == input_shape[1]:
            # Adjacency matrices
            test_input = (
                torch.eye(input_shape[1])
                .unsqueeze(0)
                .repeat(input_shape[0], 1, 1)
                .to(device)
            )
        else:
            # Eigenvectors
            test_input = torch.randn(*input_shape).to(device)
    else:
        test_input = torch.randn(*input_shape).to(device)

    # Test forward pass
    with torch.no_grad():
        output: torch.Tensor | tuple[torch.Tensor, ...] = model(test_input)

    # Handle models that return tuples
    if isinstance(output, tuple):
        output = output[0]

    _ = result.add_check("Forward pass", True, f"Output shape: {output.shape}")

    # Check output properties
    has_nan: bool = bool(torch.isnan(output).any().item())
    has_inf: bool = bool(torch.isinf(output).any().item())

    _ = result.add_check(
        "Output validity",
        not (has_nan or has_inf),
        f"NaN: {has_nan}, Inf: {has_inf}",
    )

    # Check output range for adjacency matrices (warning only)
    if len(output.shape) == 3 and output.shape[1] == output.shape[2]:
        in_range: bool = bool((output >= 0).all()) and bool((output <= 1).all())
        if not in_range:
            _ = result.add_warning(
                f"Output range outside [0,1]: Min: {float(output.min()):.4f}, Max: {float(output.max()):.4f}"
            )
        else:
            _ = result.add_check(
                "Output range [0,1]",
                True,
                f"Min: {float(output.min()):.4f}, Max: {float(output.max()):.4f}",
            )

    return result


def _check_data_loader(
    data_loader: torch.utils.data.DataLoader[Any],
    expected_batch_size: int,
    expected_shape: tuple[int, int],
    num_batches_to_check: int = 3,
) -> SanityCheckResult:
    """
    Verify data loader is producing correct batches.

    Args:
        data_loader: DataLoader to test
        expected_batch_size: Expected batch size
        expected_shape: Expected shape of each sample (without batch dim)
        num_batches_to_check: Number of batches to check

    Returns:
        SanityCheckResult with test outcomes
    """
    result = SanityCheckResult()

    for i, raw_batch in enumerate(data_loader):
        if i >= num_batches_to_check:
            break

        batch = _batch_to_tensor(raw_batch)

        # Check batch shape
        expected_full_shape = (expected_batch_size,) + expected_shape
        shape_ok = batch.shape == expected_full_shape or (
            # Last batch might be smaller
            batch.shape[0] <= expected_batch_size and batch.shape[1:] == expected_shape
        )

        result.add_check(
            f"Batch {i} shape",
            shape_ok,
            f"Got {batch.shape}, expected {expected_full_shape}",
        )

        # Check data validity
        has_nan = bool(torch.isnan(batch).any().item())
        has_inf = bool(torch.isinf(batch).any().item())

        result.add_check(
            f"Batch {i} validity",
            not (has_nan or has_inf),
            f"NaN: {has_nan}, Inf: {has_inf}",
        )

        # Check symmetry for adjacency matrices
        if len(batch.shape) == 3 and batch.shape[1] == batch.shape[2]:
            sym_diff = torch.abs(batch - batch.transpose(-2, -1)).max().item()
            is_symmetric = sym_diff < 1e-6
            result.add_check(
                f"Batch {i} symmetry",
                is_symmetric,
                f"Max asymmetry: {sym_diff:.6f}",
            )

    return result


def _check_loss_computation(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    device: str | torch.device = "cpu",
) -> SanityCheckResult:
    """
    Verify loss computation works correctly.

    Args:
        model: Model to test
        criterion: Loss function
        sample_input: Sample input
        sample_target: Sample target
        device: Device to run on

    Returns:
        SanityCheckResult with test outcomes
    """
    result = SanityCheckResult()

    model = model.to(device)
    sample_input = sample_input.to(device)
    sample_target = sample_target.to(device)

    with torch.no_grad():
        output = model(sample_input)
        # Handle models that return tuples
        if isinstance(output, tuple):
            output = output[0]
        loss = criterion(output, sample_target)

    result.add_check("Loss computation", True, f"Loss value: {loss.item():.4f}")

    # Check loss validity
    loss_valid = not (torch.isnan(loss).item() or torch.isinf(loss).item())
    result.add_check("Loss validity", loss_valid, f"Loss is finite: {loss_valid}")

    # Check loss is positive
    loss_positive = loss.item() >= 0
    result.add_check(
        "Loss non-negative", loss_positive, f"Loss value: {loss.item():.4f}"
    )

    result.add_metric("sample_loss", loss.item())

    return result


def _run_experiment_sanity_check(
    model: torch.nn.Module,
    noise_generator: NoiseDefinition,
    data_loader: torch.utils.data.DataLoader[Any],
    criterion: torch.nn.Module,
    device: str | torch.device = "cpu",
    save_plots: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run comprehensive sanity checks for an experiment.

    Args:
        model: Model to test
        noise_generator: Noise definition to test
        data_loader: DataLoader to test
        criterion: Loss function
        device: Device to run on
        save_plots: Whether to save diagnostic plots
        output_dir: Directory to save plots

    Returns:
        Dictionary with all check results
    """
    results = {}

    print("\n" + "=" * 50)
    print("Running Experiment Sanity Checks")
    print("=" * 50 + "\n")

    # 1. Check noise generator
    print("1. Checking noise generator...")
    noise_result = _check_noise_generator(noise_generator)
    results["noise_generator"] = noise_result
    print(noise_result)
    print()

    # 2. Check data loader
    print("2. Checking data loader...")
    # Get a sample to determine expected shape.  The batch may be a
    # GraphData subclass (production dataloaders emit GraphState /
    # DenseGraphState) or a raw Tensor (legacy test loaders).  Convert
    # once for all downstream checks.
    raw_sample = next(iter(data_loader))
    sample_batch = _batch_to_tensor(raw_sample)
    expected_shape = sample_batch.shape[1:]

    data_result = _check_data_loader(
        data_loader, sample_batch.shape[0], (expected_shape[0], expected_shape[1])
    )
    results["data_loader"] = data_result
    print(data_result)
    print()

    # 3. Check model forward pass
    print("3. Checking model forward pass...")
    # Determine input shape based on model type
    if hasattr(model, "model") and hasattr(model.model, "__class__"):
        model_name = model.model.__class__.__name__
        if "Attention" in model_name:
            # Attention models expect eigenvectors
            input_shape = (1, expected_shape[0], expected_shape[0])
        else:
            # GNN models expect adjacency matrices
            input_shape = (1,) + expected_shape
    else:
        input_shape = (1,) + expected_shape

    model_result = _check_model_forward_pass(model, input_shape, device)
    results["model_forward"] = model_result
    print(model_result)
    print()

    # 4. Check loss computation
    print("4. Checking loss computation...")
    sample_input = sample_batch[:1].to(device)
    loss_result = _check_loss_computation(
        model, criterion, sample_input, sample_input, device
    )
    results["loss_computation"] = loss_result
    print(loss_result)
    print()

    # 5. Generate diagnostic plots if requested
    if save_plots and output_dir:
        print("5. Generating diagnostic plots...")
        _generate_diagnostic_plots(
            model, noise_generator, sample_batch[0], output_dir, device
        )
        print(f"Plots saved to {output_dir}")
        print()

    # Overall result
    all_passed = all(r.passed for r in results.values() if hasattr(r, "passed"))

    print("=" * 50)
    print(f"Overall Sanity Check: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 50)

    return {"passed": all_passed, "results": results}


def maybe_run_sanity_check(
    config: DictConfig, data_module: LightningDataModule, model: SanityCheckableModel
) -> dict[str, str] | None:
    # Run sanity check if requested
    if config.get("sanity_check", False):
        print("\n" + "=" * 70)
        print("SANITY CHECK MODE - Validating experimental setup")
        print("=" * 70)

        # Setup data module (ensure prepare_data is called first)
        _ = data_module.prepare_data()
        _ = data_module.setup("fit")

        # Get data loader and other components
        train_loader: torch.utils.data.DataLoader[Any] = data_module.train_dataloader()

        # Access model attributes (typed via SanityCheckableModel Protocol).
        # Extract the NoiseDefinition from noise_process.definition for
        # sanity checks that operate on raw adjacency tensors.
        np_ = model.noise_process
        noise_gen = getattr(np_, "definition", None)
        if not isinstance(noise_gen, NoiseDefinition):
            raise TypeError(
                f"Sanity check requires noise_process.definition to be a "
                f"NoiseDefinition, got {type(noise_gen).__name__}"
            )
        crit = model.criterion
        assert isinstance(model, nn.Module), (
            f"SanityCheckableModel must also be an nn.Module, got {type(model)}"
        )
        sanity_results: dict[str, Any] = _run_experiment_sanity_check(
            model=model,
            noise_generator=noise_gen,
            data_loader=train_loader,
            criterion=crit,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            save_plots=True,
            output_dir=Path(config.paths.output_dir) / "sanity_check_plots",
        )

        if not sanity_results["passed"]:
            raise RuntimeError("Sanity check failed! See output above for details.")

        print("\nSanity check passed! Exiting without training.")
        return {"sanity_check": "passed"}
    return None


def _generate_diagnostic_plots(
    model: torch.nn.Module,
    noise_generator: NoiseDefinition,
    sample_adjacency: torch.Tensor,
    output_dir: Path,
    device: str | torch.device = "cpu",
) -> None:
    """Generate diagnostic plots for sanity checking."""
    output_dir = Path(output_dir)
    _ = output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Noise effect visualization
    fig, axes_raw = plt.subplots(2, 3, figsize=(12, 8))
    axes: np.ndarray = np.array(axes_raw)
    noise_levels: list[float] = [0.0, 0.1, 0.3]

    for i, eps in enumerate(noise_levels):
        A_noisy: torch.Tensor
        if eps == 0:
            A_noisy = sample_adjacency
        else:
            A_noisy = noise_generator.add_noise(sample_adjacency, eps)

        # Top row: adjacency matrices
        _ = axes[0, i].imshow(A_noisy.cpu().numpy(), cmap="viridis")
        _ = axes[0, i].set_title(f"Noise level: {eps}")
        _ = axes[0, i].axis("off")

        # Bottom row: difference from original
        diff: torch.Tensor = torch.abs(A_noisy - sample_adjacency)
        _ = axes[1, i].imshow(diff.cpu().numpy(), cmap="Reds")
        _ = axes[1, i].set_title(f"Difference (sum: {float(diff.sum()):.2f})")
        _ = axes[1, i].axis("off")

    _ = plt.tight_layout()
    _ = plt.savefig(output_dir / "noise_effects.png", dpi=150, bbox_inches="tight")
    _ = plt.close()

    # 2. Model input/output visualization
    fig2, axes2_raw = plt.subplots(1, 3, figsize=(12, 4))
    axes2: np.ndarray = np.array(axes2_raw)

    # Original
    _ = axes2[0].imshow(sample_adjacency.cpu().numpy(), cmap="viridis")
    _ = axes2[0].set_title("Original")
    _ = axes2[0].axis("off")

    # Noisy
    A_noisy = noise_generator.add_noise(sample_adjacency, 0.2)
    _ = axes2[1].imshow(A_noisy.cpu().numpy(), cmap="viridis")
    _ = axes2[1].set_title("Noisy (eps=0.2)")
    _ = axes2[1].axis("off")

    # Denoised
    _ = model.eval()
    with torch.no_grad():
        denoised: torch.Tensor | tuple[torch.Tensor, ...]
        denoised = model(A_noisy.unsqueeze(0).to(device))

        # Handle models that return tuples
        if isinstance(denoised, tuple):
            denoised = denoised[0]

        if len(denoised.shape) == 3:
            denoised = denoised.squeeze(0)

    _ = axes2[2].imshow(denoised.cpu().numpy(), cmap="viridis")
    _ = axes2[2].set_title("Denoised")
    _ = axes2[2].axis("off")

    _ = plt.tight_layout()
    _ = plt.savefig(output_dir / "model_io.png", dpi=150, bbox_inches="tight")
    _ = plt.close()
