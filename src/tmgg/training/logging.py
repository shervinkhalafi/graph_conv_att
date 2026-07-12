"""Logging utilities for experiment tracking.

Provides logger creation (W&B, CSV), metric logging, figure logging,
and model parameter counting.
"""

# pyright: reportCallIssue=false
# pyright: reportAny=false
# PyTorch stubs mistype numel() and parameter_count() is not in stubs.

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.figure
import matplotlib.pyplot as plt
from lightning_fabric.loggers.logger import Logger as FabricLogger
from loguru import logger as loguru
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, Logger, WandbLogger
from torch import nn

if TYPE_CHECKING:
    from rich.console import Console

type ParameterCountTree = dict[str, int | ParameterCountTree]


def _get_parameter_count_int(counts: ParameterCountTree, key: str) -> int:
    """Return a required integer leaf from a parameter-count tree."""
    value = counts[key]
    if not isinstance(value, int):
        raise TypeError(f"Expected integer parameter count at key '{key}'")
    return value


def setup_rich_logging(console: "Console") -> None:
    """Configure loguru to output through Rich console for progress bar compatibility.

    When using Rich's Progress bar (a live display), any output that bypasses
    Rich's console corrupts the progress bar rendering. This function redirects
    loguru to write through the Rich console, which coordinates with live displays.

    Parameters
    ----------
    console
        The Rich Console instance (typically from `rich.get_console()`).
    """
    loguru.remove()  # Remove default stderr handler
    loguru.add(
        lambda msg: console.print(msg, end="", highlight=False, markup=False),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        colorize=True,
    )


def _has_wandb_credentials() -> bool:
    """Check if WandB API key is set."""
    return bool(os.environ.get("WANDB_API_KEY"))


def _finish_active_wandb_run() -> None:
    """Finish an active W&B run, if one exists."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        loguru.warning("Finishing active wandb run before creating a new WandbLogger.")
        wandb.finish()


def create_loggers(config: DictConfig) -> list[Logger]:
    """Create PyTorch Lightning loggers based on configuration.

    Supported backends: W&B (primary) and CSV (lightweight alternative).

    Parameters
    ----------
    config
        Hydra configuration containing logger settings.

    Returns
    -------
    list[Logger]
        Configured PyTorch Lightning loggers.

    Raises
    ------
    OSError
        If W&B is configured but ``WANDB_API_KEY`` is not set and
        ``allow_no_wandb`` is not enabled.
    """
    loggers: list[Logger] = []

    if "logger" in config:
        logger_configs = config.logger

        # Handle single logger config or list of configs
        if not isinstance(logger_configs, list | ListConfig):
            logger_configs = [logger_configs]

        for logger_config in logger_configs:
            if isinstance(logger_config, dict) or hasattr(logger_config, "items"):
                # Get the logger type (first key in the dict)
                logger_type = list(logger_config.keys())[0]
                logger_params = logger_config[logger_type]

                if logger_type == "wandb":
                    # Hydra multirun can execute multiple jobs in the same process.
                    # Ensure each job starts with a clean wandb run state.
                    _finish_active_wandb_run()

                    if not _has_wandb_credentials():
                        if config.get("allow_no_wandb", False):
                            loguru.warning(
                                "W&B logger skipped: WANDB_API_KEY not set "
                                "(allow_no_wandb=true)"
                            )
                            continue
                        raise OSError(
                            "W&B logger configured but WANDB_API_KEY is not set. "
                            "Set the environment variable, or pass "
                            "allow_no_wandb=true to degrade to a warning."
                        )

                    loguru.info(
                        "Using wandb logger: entity={entity}, project={project}",
                        entity=logger_params.get("entity"),
                        project=logger_params.get("project"),
                    )
                    # Derive W&B run name. Order of precedence:
                    #   1. Top-level ``wandb_name`` override (added by
                    #      launchers via ``+wandb_name=...``); honors the
                    #      sweep launcher's explicit naming.
                    #   2. The logger config block's ``name`` field.
                    #   3. The auto-generated ``run_id`` (or
                    #      ``experiment_name`` as a final fallback).
                    wandb_name = (
                        config.get("wandb_name")
                        or logger_params.get("name")
                        or config.get(
                            "run_id", config.get("experiment_name", "unknown")
                        )
                    )
                    wandb_kwargs: dict[str, Any] = {
                        "project": logger_params.get("project"),
                        "name": wandb_name,
                        "tags": logger_params.get("tags", []),
                        "save_dir": logger_params.get(
                            "save_dir", config.paths.output_dir
                        ),
                        "entity": logger_params.get("entity", None),
                        "log_model": logger_params.get("log_model", False),
                    }
                    run_config: dict[str, Any] = {}

                    # Promote key sweep dimensions so they are easy to filter in W&B.
                    data_graph_type = OmegaConf.select(config, "data.graph_type")
                    if data_graph_type is not None:
                        run_config["data"] = str(data_graph_type)

                    filter_qk = OmegaConf.select(config, "model.model.filter_qk")
                    if filter_qk is not None:
                        run_config["filter_qk"] = bool(filter_qk)

                    if run_config:
                        wandb_kwargs["config"] = run_config
                    for optional_key in ("group", "notes", "offline"):
                        if optional_key in logger_params:
                            wandb_kwargs[optional_key] = logger_params.get(optional_key)

                    # Resume the same W&B run on Modal-preempt restart.
                    # ``run_experiment`` reads ``<output_dir>/wandb_run_id.txt``
                    # before this call and stashes the persisted id at
                    # ``config.wandb_run_id_resume`` when present. Passing
                    # ``id=`` to ``WandbLogger`` together with Lightning's
                    # default ``resume="allow"`` makes the W&B server append
                    # to the existing run instead of fragmenting metrics
                    # across multiple same-named runs.
                    resume_id = config.get("wandb_run_id_resume")
                    if resume_id:
                        wandb_kwargs["id"] = str(resume_id)

                    wandb_logger = WandbLogger(**wandb_kwargs)
                    loggers.append(wandb_logger)

                elif logger_type == "csv":
                    save_dir = logger_params.get(
                        "save_dir", os.path.join(config.paths.output_dir, "csv")
                    )
                    csv_logger = CSVLogger(
                        save_dir=save_dir,
                        name=logger_params.get(
                            "name", config.get("experiment_name", "default")
                        ),
                        version=logger_params.get("version", None),
                        flush_logs_every_n_steps=logger_params.get(
                            "flush_logs_every_n_steps", 100
                        ),
                    )
                    loggers.append(csv_logger)
                else:
                    raise ValueError(
                        f"Unsupported logger type {logger_type!r}. "
                        "Supported types: 'wandb', 'csv'."
                    )

    if not loggers:
        loguru.warning("No loggers configured — metrics will not be persisted")

    return loggers


def log_figure(
    loggers: Logger | list[Logger],
    tag: str,
    figure: matplotlib.figure.Figure,
    global_step: int | None = None,
    close: bool = True,
) -> None:
    """
    Log a matplotlib figure to all configured loggers.

    Args:
        loggers: Single logger or list of loggers
        tag: Tag/name for the figure
        figure: Matplotlib figure to log
        global_step: Global step for logging (optional)
        close: Whether to close the figure after logging (default: True)
    """
    # Ensure loggers is a list
    if not isinstance(loggers, list):
        loggers = [loggers]

    # Handle the case where logger might be None
    loggers = [logger for logger in loggers if logger is not None]

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            # Wandb uses Image wrapper
            import wandb

            loguru.info("Saving figure to wandb")
            logger.experiment.log({tag: wandb.Image(figure)})

        elif isinstance(logger, CSVLogger):
            # CSV logger saves figures to disk
            save_dir = Path(logger.log_dir) / "figures"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create filename with optional step
            if global_step is not None:
                filename = f"{tag}_step_{global_step}.png"
            else:
                filename = f"{tag}.png"

            # Save figure
            save_path = save_dir / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            loguru.info(f"Saving figure to {save_path=}")
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
    # Close figure to prevent memory leaks
    if close:
        plt.close(figure)


def log_figures(
    loggers: Logger | list[Logger],
    figures: dict[str, matplotlib.figure.Figure],
    global_step: int | None = None,
    close: bool = True,
) -> None:
    """
    Log multiple matplotlib figures to all configured loggers.

    Args:
        loggers: Single logger or list of loggers
        figures: Dictionary mapping tags to matplotlib figures
        global_step: Global step for logging (optional)
        close: Whether to close figures after logging (default: True)
    """
    for tag, figure in figures.items():
        log_figure(loggers, tag, figure, global_step=global_step, close=close)


def log_metrics(
    logger: Logger | list[Logger],
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """Log scalar metrics to any PyTorch Lightning logger with consistent API.

    Provides a unified interface for logging metrics across W&B and CSV
    backends, handling the different APIs transparently.

    Parameters
    ----------
    logger
        Single logger or list of loggers.
    metrics
        Metric names mapped to scalar values.
    step
        Optional global step.

    Example
    -------
    ::

        log_metrics(logger, {"loss": 0.5}, step=100)
    """
    if logger is None:
        return

    # Ensure logger is a list
    loggers = [logger] if not isinstance(logger, list) else logger

    # Filter out None loggers
    loggers = [lg for lg in loggers if lg is not None]

    for lg in loggers:
        if isinstance(lg, WandbLogger):
            # W&B can log all metrics at once
            lg.experiment.log(metrics)

        elif isinstance(lg, CSVLogger):
            # Write metrics directly to a CSV file in the log directory
            import csv

            csv_dir = Path(lg.log_dir)
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_path = csv_dir / "manual_metrics.csv"

            # Check if file exists to determine if we need headers
            file_exists = csv_path.exists()

            # Prepare row with optional step column
            row_data: dict[str, float | int | None] = (
                {"step": step} if step is not None else {}
            )
            row_data.update(metrics)

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)


def log_parameter_count(
    model: nn.Module,
    model_name: str,
    logger: FabricLogger | None,
) -> None:
    """Log the parameter count of *model* in a formatted way.

    If the model exposes a ``parameter_count`` method the output is a
    hierarchical breakdown; otherwise a simple total is printed.  Counts
    are also forwarded to *logger* when one is provided.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters are counted.
    model_name : str
        Human-readable name used in the printed header and logger tags.
    logger : Logger | None
        Optional Lightning logger.  When present, parameter counts are
        forwarded via ``log_hyperparams``.
    """
    if not hasattr(model, "parameter_count"):
        total_params: int = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"Total Trainable Parameters: {total_params:,}")
        print(f"{'=' * 50}\n")

        if logger:
            logger.log_hyperparams({"total_parameters": total_params})
        return

    param_counts: ParameterCountTree = model.parameter_count()

    def format_counts(counts: ParameterCountTree, indent: int = 0) -> list[str]:
        """Recursively format parameter counts."""
        lines: list[str] = []
        prefix: str = "  " * indent

        for key, value in counts.items():
            if key == "total" and indent == 0:
                continue
            elif key == "self":
                self_count = _get_parameter_count_int(counts, key)
                if self_count > 0:
                    _ = lines.append(f"{prefix}├─ {key}: {self_count:,}")
            elif isinstance(value, dict):
                if "total" in value:
                    child_total = _get_parameter_count_int(value, "total")
                    _ = lines.append(f"{prefix}├─ {key}: {child_total:,}")
                    sub_items: ParameterCountTree = {
                        k: v for k, v in value.items() if k != "total"
                    }
                    if sub_items:
                        sub_lines: list[str] = format_counts(sub_items, indent + 1)
                        _ = lines.extend(sub_lines)
                else:
                    _ = lines.append(f"{prefix}├─ {key}:")
                    sub_lines = format_counts(value, indent + 1)
                    _ = lines.extend(sub_lines)
            else:
                _ = lines.append(f"{prefix}├─ {key}: {value:,}")

        return lines

    print(f"\n{'=' * 60}")
    print(f"Model Parameter Count: {model_name}")
    print(f"{'=' * 60}")
    print(
        f"Total Trainable Parameters: "
        f"{_get_parameter_count_int(param_counts, 'total'):,}"
    )
    print("-" * 60)

    formatted_lines: list[str] = format_counts(param_counts)
    for line in formatted_lines:
        print(line)

    print(f"{'=' * 60}\n")

    if logger:
        logger.log_hyperparams(
            {
                "total_parameters": _get_parameter_count_int(param_counts, "total"),
                "parameter_breakdown": param_counts,
            }
        )
