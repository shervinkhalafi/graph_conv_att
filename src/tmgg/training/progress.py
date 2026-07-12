"""Step-based Rich progress bar for PyTorch Lightning.

Displays training progress as steps rather than epochs, matching
step-centric training configuration (max_steps, not max_epochs).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from rich import get_console, reconfigure
from rich.console import Console, RenderableType
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
)
from rich.style import Style
from rich.text import Text

if TYPE_CHECKING:
    import pytorch_lightning as pl

from tmgg.training.logging import (
    setup_rich_logging,
)


class StepTimeColumn(ProgressColumn):
    """Column showing elapsed and remaining time."""

    max_refresh = 0.5

    def __init__(self, style: str | Style = "dim") -> None:
        self.style = style
        super().__init__()

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_str = (
            "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        )
        remaining_str = (
            "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
        )
        return Text(f"{elapsed_str} • {remaining_str}", style=self.style)


class StepCountColumn(ProgressColumn):
    """Column showing completed/total steps with zero-padding for stable width."""

    def __init__(self, style: str | Style = "") -> None:
        self.style = style
        super().__init__()

    def render(self, task: Task) -> RenderableType:
        if task.total is None or task.total == float("inf"):
            return Text(f"{int(task.completed)}/--", style=self.style)
        total = int(task.total)
        width = len(str(total))
        current = f"{int(task.completed):0{width}d}"
        return Text(f"{current}/{total}", style=self.style)


class StepSpeedColumn(ProgressColumn):
    """Column showing steps per second."""

    def __init__(self, style: str | Style = "dim") -> None:
        self.style = style
        super().__init__()

    def render(self, task: Task) -> RenderableType:
        speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{speed}it/s", style=self.style)


class MetricsColumn(ProgressColumn):
    """Column displaying training metrics."""

    def __init__(
        self,
        style: str | Style = "italic",
        metrics_format: str = ".4f",
    ) -> None:
        self._style = style
        self._metrics_format = metrics_format
        self._metrics: dict[str, Any] = {}
        super().__init__()

    def update(self, metrics: dict[str, Any]) -> None:
        """Update displayed metrics."""
        self._metrics = metrics

    def render(self, task: Task) -> Text:
        parts = []
        for name, value in self._metrics.items():
            if isinstance(value, float):
                parts.append(f"{name}: {value:{self._metrics_format}}")
            elif isinstance(value, int | str):
                parts.append(f"{name}: {value}")
        return Text(" ".join(parts), style=self._style)


@dataclass
class StepProgressBarTheme:
    """Theme for step-based progress bar styling."""

    description: str | Style = "bold blue"
    progress_bar: str | Style = "#6206E0"
    progress_bar_finished: str | Style = "#6206E0"
    step_count: str | Style = ""
    time: str | Style = "dim"
    speed: str | Style = "dim"
    metrics: str | Style = "italic"
    metrics_format: str = ".4f"


class StepProgressBar(ProgressBar):
    """Progress bar showing global_step/max_steps with compact epoch indicator.

    Displays training progress in a step-centric format:
        Training (E2) ━━━━━━━━ 12500/50000 0:15:32 • 0:45:00 12.5it/s train_loss: 0.0234

    Parameters
    ----------
    metrics_to_show
        List of metric names to display. If None (default), shows all metrics
        logged with prog_bar=True. If empty list, shows no metrics.
    show_epoch
        Whether to show compact epoch indicator "(E2)". Default: True.
    refresh_rate
        How often to refresh the bar in steps. Default: 1.
    leave
        Keep bar visible after completion. Default: False.
    theme
        Theme for styling. Uses default if None.
    console_kwargs
        Additional kwargs for Rich Console.
    """

    def __init__(
        self,
        metrics_to_show: list[str] | None = None,
        show_epoch: bool = True,
        metrics_format: str = ".4f",
        refresh_rate: int = 1,
        leave: bool = False,
        theme: StepProgressBarTheme | None = None,
        console_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._metrics_to_show = metrics_to_show  # None means show all
        self._show_epoch = show_epoch
        self._metrics_format = metrics_format
        self._refresh_rate = refresh_rate
        self._leave = leave
        self._theme = theme or StepProgressBarTheme()
        self._console_kwargs = console_kwargs or {}

        self._console: Console | None = None
        self._enabled: bool = True
        self.progress: Progress | None = None
        self._progress_stopped: bool = False
        self._metric_component: MetricsColumn | None = None

        # Progress bar task IDs
        self.train_progress_bar_id: TaskID | None = None
        self.val_sanity_progress_bar_id: TaskID | None = None
        self.val_progress_bar_id: TaskID | None = None
        self.test_progress_bar_id: TaskID | None = None
        self.predict_progress_bar_id: TaskID | None = None

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def _reset_progress_bar_ids(self) -> None:
        """Reset all progress bar task IDs."""
        self.train_progress_bar_id = None
        self.val_sanity_progress_bar_id = None
        self.val_progress_bar_id = None
        self.test_progress_bar_id = None
        self.predict_progress_bar_id = None

    def _init_progress(self, trainer: pl.Trainer) -> None:
        """Initialize Rich progress display with safe clear_live handling."""
        if not self.is_enabled:
            return
        if self.progress is not None and not self._progress_stopped:
            return

        self._reset_progress_bar_ids()
        reconfigure(**self._console_kwargs)
        self._console = get_console()
        setup_rich_logging(self._console)

        # Lightning 2.5 / Rich >= 14.1.0 compatibility:
        # clear_live() crashes if _live_stack is empty
        if hasattr(self._console, "_live_stack"):
            if len(self._console._live_stack) > 0:
                self._console.clear_live()
        else:
            # Older Rich versions use _live attribute
            if (
                hasattr(self._console, "_live")
                and getattr(self._console, "_live", None) is not None
            ):
                self._console.clear_live()

        self._metric_component = MetricsColumn(
            style=self._theme.metrics,
            metrics_format=self._metrics_format,
        )

        self.progress = Progress(
            *self.configure_columns(trainer),
            self._metric_component,
            auto_refresh=False,
            disable=self.is_disabled,
            console=self._console,
        )
        self.progress.start()
        self._progress_stopped = False

    def _stop_progress(self) -> None:
        """Stop the progress display."""
        if self.progress is not None:
            self.progress.stop()
            self._progress_stopped = True

    def configure_columns(self, trainer: pl.Trainer) -> list[ProgressColumn]:
        """Configure progress bar columns for step-focused display."""
        return [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                complete_style=self._theme.progress_bar,
                finished_style=self._theme.progress_bar_finished,
            ),
            StepCountColumn(style=self._theme.step_count),
            StepTimeColumn(style=self._theme.time),
            StepSpeedColumn(style=self._theme.speed),
        ]

    def _estimate_max_epochs(self, trainer: pl.Trainer) -> int:
        """Estimate max epochs from max_steps and steps per epoch."""
        if trainer.max_steps <= 0:
            return 0
        steps_per_epoch = self.total_train_batches
        if steps_per_epoch <= 0:
            return 0
        return int((trainer.max_steps + steps_per_epoch - 1) // steps_per_epoch)

    def _get_train_description(self, trainer: pl.Trainer) -> str:
        """Generate training phase description with zero-padded epoch."""
        if self._show_epoch:
            max_epochs = self._estimate_max_epochs(trainer)
            width = max(1, len(str(max_epochs)))
            epoch_str = f"{trainer.current_epoch:0{width}d}"
            return f"Training (E{epoch_str})"
        return "Training"

    def _get_val_description(self, trainer: pl.Trainer) -> str:
        """Generate validation phase description with zero-padded epoch."""
        if self._show_epoch:
            max_epochs = self._estimate_max_epochs(trainer)
            width = max(1, len(str(max_epochs)))
            epoch_str = f"{trainer.current_epoch:0{width}d}"
            return f"Validating (E{epoch_str})"
        return "Validating"

    def _add_task(
        self, total: int | float, description: str, visible: bool = True
    ) -> TaskID:
        """Add a new progress task."""
        assert self.progress is not None
        styled_desc = f"[{self._theme.description}]{description}"
        return self.progress.add_task(styled_desc, total=total, visible=visible)

    def _update(
        self, progress_bar_id: TaskID | None, current: int, visible: bool = True
    ) -> None:
        """Update progress bar position."""
        if self.progress is None or not self.is_enabled or progress_bar_id is None:
            return
        total = self.progress.tasks[progress_bar_id].total
        if total is not None and not self._should_update(current, total):
            return
        self.progress.update(progress_bar_id, completed=current, visible=visible)

    def _should_update(self, current: int, total: int | float) -> bool:
        """Check if progress should be refreshed at this step."""
        return current % self._refresh_rate == 0 or current == total

    def refresh(self) -> None:
        """Refresh the progress display."""
        if self.progress:
            self.progress.refresh()

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, Any]:
        """Get filtered metrics for display.

        Merges progress_bar_metrics with key metrics from callback_metrics,
        since Lightning may not populate progress_bar_metrics immediately
        for on_step metrics like train_loss.
        """
        # Start with standard progress bar metrics
        items = super().get_metrics(trainer, pl_module)

        # Also check callback_metrics for key metrics that might be missing
        # Only include loss metrics and learning rate to avoid noise
        key_patterns = ("train_loss", "val/loss", "test/loss", "lr")
        for key, value in trainer.callback_metrics.items():
            if key not in items and key in key_patterns:
                if hasattr(value, "item"):
                    items[key] = value.item()
                elif isinstance(value, int | float):
                    items[key] = value

        # Filter out internal metrics
        items.pop("v_num", None)

        if self._metrics_to_show is None:
            return items  # Show all metrics with prog_bar=True
        if not self._metrics_to_show:
            return {}  # Empty list means show none

        return {k: v for k, v in items.items() if k in self._metrics_to_show}

    def _update_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Update the metrics column with current values."""
        if not self.is_enabled or self._metric_component is None:
            return
        metrics = self.get_metrics(trainer, pl_module)
        self._metric_component.update(metrics)

    # -------------------------------------------------------------------------
    # Callback hooks
    # -------------------------------------------------------------------------

    def on_sanity_check_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._init_progress(trainer)

    def on_sanity_check_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.progress is not None and self.val_sanity_progress_bar_id is not None:
            self.progress.update(self.val_sanity_progress_bar_id, visible=False)
        self.refresh()

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._init_progress(trainer)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.is_disabled:
            return

        # Use global steps as the primary counter
        total_steps = trainer.max_steps if trainer.max_steps > 0 else float("inf")
        description = self._get_train_description(trainer)

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)

        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total_steps, description)
            else:
                # Update description for new epoch, but keep same total (global steps)
                self.progress.update(
                    self.train_progress_bar_id,
                    description=f"[{self._theme.description}]{description}",
                    completed=trainer.global_step,
                    visible=True,
                )
        self.refresh()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.is_disabled:
            return

        # Use global steps as the primary counter
        total_steps = trainer.max_steps if trainer.max_steps > 0 else float("inf")

        if self.train_progress_bar_id is None:
            # Can happen when resuming mid-epoch
            description = self._get_train_description(trainer)
            self.train_progress_bar_id = self._add_task(total_steps, description)

        # Update with global_step (already incremented after batch)
        self._update(self.train_progress_bar_id, trainer.global_step)
        self._update_metrics(trainer, pl_module)
        self.refresh()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._update_metrics(trainer, pl_module)

    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._init_progress(trainer)

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return

        assert self.progress is not None

        if trainer.sanity_checking:
            if self.val_sanity_progress_bar_id is not None:
                self.progress.update(self.val_sanity_progress_bar_id, visible=False)
            self.val_sanity_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader,
                "Sanity Check",
                visible=True,
            )
        else:
            if self.val_progress_bar_id is not None:
                self.progress.update(self.val_progress_bar_id, visible=False)
            self.val_progress_bar_id = self._add_task(
                self.total_val_batches_current_dataloader,
                self._get_val_description(trainer),
                visible=True,
            )
        self.refresh()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled:
            return
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id, batch_idx + 1)
        else:
            self._update(self.val_progress_bar_id, batch_idx + 1)
        self.refresh()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.is_enabled and self.val_progress_bar_id is not None:
            if self.progress is not None:
                self.progress.update(self.val_progress_bar_id, visible=False)
            self.refresh()

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.state.fn == "fit":
            self._update_metrics(trainer, pl_module)
        self.reset_dataloader_idx_tracker()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._init_progress(trainer)

    def on_test_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return
        if self.test_progress_bar_id is not None:
            assert self.progress is not None
            self.progress.update(self.test_progress_bar_id, visible=False)
        self.test_progress_bar_id = self._add_task(
            self.total_test_batches_current_dataloader, "Testing"
        )
        self.refresh()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or self.test_progress_bar_id is None:
            return
        self._update(self.test_progress_bar_id, batch_idx + 1)
        self.refresh()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.reset_dataloader_idx_tracker()

    def on_predict_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._init_progress(trainer)

    def on_predict_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or not self.has_dataloader_changed(dataloader_idx):
            return
        if self.predict_progress_bar_id is not None:
            assert self.progress is not None
            self.progress.update(self.predict_progress_bar_id, visible=False)
        self.predict_progress_bar_id = self._add_task(
            self.total_predict_batches_current_dataloader, "Predicting"
        )
        self.refresh()

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled or self.predict_progress_bar_id is None:
            return
        self._update(self.predict_progress_bar_id, batch_idx + 1)
        self.refresh()

    def on_predict_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.reset_dataloader_idx_tracker()

    def teardown(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        self._stop_progress()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._stop_progress()

    def __getstate__(self) -> dict[str, object]:
        """Handle pickling by removing unpickleable objects."""
        state = self.__dict__.copy()
        state["progress"] = None
        state["_console"] = None
        return state
