"""Per-step metric collectors for reverse diffusion sampling.

The ``StepMetricCollector`` protocol defines the interface for recording
metrics at each step of the reverse diffusion chain. The concrete
``DiffusionLikelihoodCollector`` accumulates per-step KL divergence for
variational lower bound computation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StepMetricCollector(Protocol):
    """Protocol for collecting per-step metrics during reverse diffusion."""

    def record(self, t: int, s: int, metrics: dict[str, float]) -> None:
        """Record metrics for one reverse step from timestep *t* to *s*.

        Parameters
        ----------
        t
            Current timestep (before this step).
        s
            Target timestep (after this step).
        metrics
            Step-level metrics. Standard keys include ``"kl"`` for KL
            divergence contribution at this step.
        """
        ...


class DiffusionLikelihoodCollector:
    """Collects per-step KL for full variational lower bound computation.

    After running a complete reverse chain with this collector, call
    ``vlb()`` to get the total VLB (sum of per-step KL contributions).

    Examples
    --------
    >>> collector = DiffusionLikelihoodCollector()
    >>> collector.record(10, 9, {"kl": 0.5})
    >>> collector.record(9, 8, {"kl": 0.3})
    >>> collector.vlb()
    0.8
    """

    def __init__(self) -> None:
        self._records: list[tuple[int, int, dict[str, float]]] = []

    def record(self, t: int, s: int, metrics: dict[str, float]) -> None:
        """Record metrics for one reverse step."""
        self._records.append((t, s, metrics))

    def vlb(self) -> float:
        """Total variational lower bound (sum of per-step KL contributions).

        Returns 0.0 if no steps were recorded or no ``"kl"`` keys exist.
        """
        return sum(m.get("kl", 0.0) for _, _, m in self._records)

    def results(self) -> dict[str, float]:
        """Summary statistics of the collected metrics.

        Returns
        -------
        dict[str, float]
            ``"vlb"`` (total KL), ``"num_steps"`` (count), and
            ``"mean_{key}"`` for each metric key seen across steps.
        """
        out: dict[str, float] = {
            "vlb": self.vlb(),
            "num_steps": float(len(self._records)),
        }
        if self._records:
            all_keys: set[str] = set()
            for _, _, m in self._records:
                all_keys.update(m.keys())
            for key in sorted(all_keys):
                values = [m[key] for _, _, m in self._records if key in m]
                if values:
                    out[f"mean_{key}"] = sum(values) / len(values)
        return out

    @property
    def records(self) -> list[tuple[int, int, dict[str, float]]]:
        """Raw recorded data as list of ``(t, s, metrics)`` tuples."""
        return list(self._records)
