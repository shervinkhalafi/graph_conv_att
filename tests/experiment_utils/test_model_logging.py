"""Tests for the standalone log_parameter_count function.

Tests the parameter-count logging logic in ``log_parameter_count``,
independently of any Lightning module. Models and loggers are mocked
to verify output formatting and logger forwarding.

Testing Strategy
----------------
- Direct calls to ``log_parameter_count(model, name, logger)`` with
  real nn.Module or mock models.
- Mock logger verifies hyperparameter forwarding.

Key Invariants
--------------
- Model without ``parameter_count()`` logs a simple total.
- Model with ``parameter_count()`` logs a hierarchical breakdown.
- Logger receives parameter counts via ``log_hyperparams`` when present.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import torch.nn as nn
from pytorch_lightning.loggers import CSVLogger

from tmgg.training.logging import log_figure, log_parameter_count

# -----------------------------------------------------------------------
# TestLogParameterCount
# -----------------------------------------------------------------------


class TestLogParameterCount:
    """Verify parameter-count logging for different model states."""

    def test_model_without_parameter_count(self) -> None:
        """Model lacking parameter_count() logs a simple total.

        Starting state: model is a plain nn.Linear(5, 3) without
        parameter_count method.
        Invariant: prints 'Total Trainable Parameters' with the correct count.
        """
        model = nn.Linear(5, 3)
        expected_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        with patch("builtins.print") as mock_print:
            log_parameter_count(model, "TestModel", None)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Total Trainable Parameters" in printed
        assert f"{expected_params:,}" in printed

    def test_model_with_parameter_count(self) -> None:
        """Model exposing parameter_count() logs a hierarchical breakdown.

        Starting state: mock model with parameter_count returning a dict.
        Invariant: prints the hierarchy including 'total' and sub-keys.
        """
        mock_model = MagicMock()
        mock_model.parameter_count.return_value = {
            "total": 1000,
            "encoder": {"total": 600, "self": 100, "linear": 500},
            "decoder": {"total": 400, "self": 400},
        }

        with patch("builtins.print") as mock_print:
            log_parameter_count(mock_model, "TestModel", None)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "1,000" in printed  # total
        assert "encoder" in printed
        assert "decoder" in printed

    def test_logger_log_hyperparams_called(self) -> None:
        """When logger is present, log_hyperparams receives parameter counts.

        Starting state: mock model with parameter_count, mock logger.
        Invariant: logger.log_hyperparams called with total_parameters key.
        """
        mock_model = MagicMock()
        mock_model.parameter_count.return_value = {"total": 42}
        mock_logger = MagicMock()

        with patch("builtins.print"):
            log_parameter_count(mock_model, "TestModel", mock_logger)

        mock_logger.log_hyperparams.assert_called_once()
        call_kwargs = mock_logger.log_hyperparams.call_args[0][0]
        assert call_kwargs["total_parameters"] == 42

    def test_no_logger_no_crash(self) -> None:
        """log_parameter_count works without a logger.

        Starting state: model is nn.Linear(3, 2), logger is None.
        Invariant: prints output, no exception.
        """
        model = nn.Linear(3, 2)
        with patch("builtins.print"):
            log_parameter_count(model, "TestModel", None)


class TestFigureLogging:
    """Figure logging should work for logger tag hierarchies."""

    def test_csv_logger_creates_nested_tag_directories(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """CSV figure logging should support slash-separated tags.

        Starting state: a CSV logger and a simple matplotlib figure.
        Invariant: logging ``gen-val/...`` creates the nested directory path
        and writes the image without raising.
        """
        logger = CSVLogger(save_dir=str(tmp_path), name="figures")
        fig = plt.figure()

        log_figure(
            logger,
            "gen-val/graph_samples",
            fig,
            global_step=7,
        )

        assert (
            tmp_path
            / "figures"
            / "version_0"
            / "figures"
            / "gen-val"
            / "graph_samples_step_7.png"
        ).exists()
