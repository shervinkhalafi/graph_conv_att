"""PyTorch Lightning callbacks owned by tmgg.training.

Callbacks defined here keep the orchestration layer slim by isolating
event-driven behaviour (e.g. EMA shadow weights) behind well-named
classes that ``run_experiment`` can register conditionally based on
config.
"""

from .chain_saving import ChainSavingCallback
from .ema import EMACallback
from .final_sample_dump import FinalSampleDumpCallback

__all__ = ["ChainSavingCallback", "EMACallback", "FinalSampleDumpCallback"]
