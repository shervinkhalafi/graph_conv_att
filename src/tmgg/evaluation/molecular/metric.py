"""Abstract base class for a single molecular generation-quality metric."""

from __future__ import annotations

import abc
from collections.abc import Sequence


class MolecularMetric(abc.ABC):
    """One metric, one ``.compute(generated, reference)`` call.

    Stateful subclasses (e.g. :class:`FCDMetric` holding a ChemNet
    embedder) initialise their state in ``__init__``; metrics live
    as long as the :class:`MolecularEvaluator` that owns them.

    The ``name`` attribute determines the W&B key under
    ``gen-val/<name>`` when the result is a single float, or
    ``gen-val/<name>/<sub>`` when ``compute`` returns a dict.
    """

    name: str

    @abc.abstractmethod
    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float | dict[str, float]:
        """Compute the metric value from generated + (optional) reference SMILES."""
