"""Sparse representation of graph size distributions.

A ``SizeDistribution`` stores the set of distinct graph sizes and their
occurrence counts, supporting efficient sampling, JSON serialization,
and construction from raw node-count sequences or fixed-size datasets.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SizeDistribution:
    """Sparse representation of the distribution over graph sizes.

    Stores only the distinct sizes that appear and their counts, rather
    than a dense histogram over ``[0, max_size]``. This keeps the
    representation compact for datasets with a few distinct sizes (e.g.,
    fixed-size synthetic data) and still efficient for variable-size
    datasets with many distinct sizes.

    Parameters
    ----------
    sizes
        Distinct node counts, sorted ascending. Each must be positive.
    counts
        Frequency of each size. Must match ``len(sizes)``.
    max_size
        Maximum node count across the dataset. Used as the padding
        dimension when batching variable-size graphs.
    """

    sizes: tuple[int, ...]
    counts: tuple[int, ...]
    max_size: int

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.counts):
            raise ValueError(
                f"sizes and counts must have equal length, "
                f"got {len(self.sizes)} vs {len(self.counts)}"
            )
        if len(self.sizes) == 0:
            raise ValueError("sizes must be non-empty")
        if not all(s > 0 for s in self.sizes):
            raise ValueError(f"All sizes must be positive, got {self.sizes}")
        if not all(c > 0 for c in self.counts):
            raise ValueError(f"All counts must be positive, got {self.counts}")
        if self.max_size < max(self.sizes):
            raise ValueError(
                f"max_size ({self.max_size}) must be >= max(sizes) ({max(self.sizes)})"
            )
        if list(self.sizes) != sorted(self.sizes):
            raise ValueError(f"sizes must be sorted ascending, got {self.sizes}")

    @property
    def probs(self) -> list[float]:
        """Probability of each size, in the same order as ``sizes``."""
        total = sum(self.counts)
        return [c / total for c in self.counts]

    @property
    def is_degenerate(self) -> bool:
        """True if the distribution has a single size (fixed-size dataset)."""
        return len(self.sizes) == 1

    def sample(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample node counts from the distribution.

        Parameters
        ----------
        batch_size
            Number of samples to draw.
        generator
            Optional torch RNG for reproducibility.

        Returns
        -------
        Tensor
            Integer tensor of shape ``(batch_size,)`` with sampled sizes.
        """
        if self.is_degenerate:
            return torch.full((batch_size,), self.sizes[0], dtype=torch.long)
        probs_t = torch.tensor(self.probs)
        indices = torch.multinomial(
            probs_t, batch_size, replacement=True, generator=generator
        )
        sizes_t = torch.tensor(self.sizes, dtype=torch.long)
        return sizes_t[indices]

    def log_prob(self, N: Tensor) -> Tensor:
        """Log-probability of given node counts under this distribution.

        Parameters
        ----------
        N
            Integer tensor of node counts.

        Returns
        -------
        Tensor
            Log-probabilities with the same shape as ``N``.

        Notes
        -----
        For sizes with near-zero probability (outside the empirical support),
        returns approximately -69 (``log(1e-30)``), not ``-inf`` or 0, due to
        internal clamping.
        """
        probs_t = torch.tensor(self.probs, dtype=torch.float32, device=N.device)
        sizes_t = torch.tensor(self.sizes, dtype=torch.long, device=N.device)
        # For each element in N, find which index in sizes_t matches
        matches = N.unsqueeze(-1) == sizes_t.unsqueeze(0)  # (batch, num_sizes)
        p = (matches.float() * probs_t.unsqueeze(0)).sum(dim=-1)
        return torch.log(p.clamp(min=1e-30))

    def to_dict(self) -> dict[str, list[int] | int]:
        """Serialize to a sparse JSON-compatible dictionary.

        Returns
        -------
        dict
            Keys: ``sizes`` (list of int), ``counts`` (list of int),
            ``max_size`` (int).
        """
        return {
            "sizes": list(self.sizes),
            "counts": list(self.counts),
            "max_size": self.max_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, list[int] | int]) -> SizeDistribution:
        """Deserialize from a sparse dictionary (inverse of ``to_dict``).

        Parameters
        ----------
        d
            Dictionary with keys ``sizes``, ``counts``, ``max_size``.

        Returns
        -------
        SizeDistribution
        """
        sizes = d["sizes"]
        counts = d["counts"]
        max_size = d["max_size"]
        if not isinstance(sizes, list):
            raise TypeError(
                f"expected 'sizes' to be a list, got {type(sizes).__name__}"
            )
        if not isinstance(counts, list):
            raise TypeError(
                f"expected 'counts' to be a list, got {type(counts).__name__}"
            )
        if not isinstance(max_size, int):
            raise TypeError(
                f"expected 'max_size' to be an int, got {type(max_size).__name__}"
            )

        return cls(
            sizes=tuple(sizes),
            counts=tuple(counts),
            max_size=max_size,
        )

    @classmethod
    def fixed(cls, num_nodes: int) -> SizeDistribution:
        """Create a degenerate distribution where all graphs have ``num_nodes``.

        Parameters
        ----------
        num_nodes
            The fixed graph size.

        Returns
        -------
        SizeDistribution
            Single-entry distribution with ``probs == [1.0]``.
        """
        return cls(sizes=(num_nodes,), counts=(1,), max_size=num_nodes)

    @classmethod
    def from_node_counts(cls, node_counts: Sequence[int]) -> SizeDistribution:
        """Build from a sequence of per-graph node counts.

        Useful when loading a dataset (e.g. from PyG) where each graph
        may have a different number of nodes.

        Parameters
        ----------
        node_counts
            Per-graph node counts (one entry per graph).

        Returns
        -------
        SizeDistribution
            Distribution with empirical frequencies from ``node_counts``.
        """
        if len(node_counts) == 0:
            raise ValueError("node_counts must be non-empty")
        counter = Counter(node_counts)
        sorted_sizes = sorted(counter.keys())
        counts = [counter[s] for s in sorted_sizes]
        return cls(
            sizes=tuple(sorted_sizes),
            counts=tuple(counts),
            max_size=max(sorted_sizes),
        )
