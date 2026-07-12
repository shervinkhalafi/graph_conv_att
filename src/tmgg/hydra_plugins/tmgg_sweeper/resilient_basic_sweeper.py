"""Hydra sweeper that continues a multirun after individual job failures.

Hydra's built-in :class:`hydra._internal.core_plugins.basic_sweeper.BasicSweeper`
re-raises the first task exception, which aborts the entire sweep. This subclass
logs failures and continues with remaining jobs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.utils import JobReturn
from loguru import logger as loguru
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


class ResilientBasicSweeper(BasicSweeper):
    """Same as Hydra's BasicSweeper, but failed jobs do not stop the sweep."""

    def sweep(self, arguments: list[str]) -> Any:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        params_conf = self._parse_config()
        params_conf.extend(arguments)

        parser = OverridesParser.create(config_loader=self.hydra_context.config_loader)
        overrides = parser.parse_overrides(params_conf)

        self.overrides = self.split_arguments(overrides, self.max_batch_size)
        returns: list[Sequence[JobReturn]] = []

        sweep_dir = Path(self.config.hydra.sweep.dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.config, sweep_dir / "multirun.yaml")

        initial_job_idx = 0
        while not self.is_done():
            batch = self.get_job_batch()
            tic = time.perf_counter()
            self.validate_batch_is_legal(batch)
            elapsed = time.perf_counter() - tic
            log.debug(
                "Validated configs of %d jobs in %.2f seconds, %.2f / second",
                len(batch),
                elapsed,
                len(batch) / elapsed if elapsed > 0 else 0.0,
            )
            results = self.launcher.launch(batch, initial_job_idx=initial_job_idx)

            for r in results:
                try:
                    _ = r.return_value
                except Exception:
                    loguru.exception(
                        "Hydra multirun job failed; continuing sweep. overrides={!r}",
                        r.overrides,
                    )

            initial_job_idx += len(batch)
            returns.append(results)

        return returns
