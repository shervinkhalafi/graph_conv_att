"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np


def generate_sbm_adjacency(
    block_sizes: list[int],
    p_intra: float,
    p_inter: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an adjacency matrix for a stochastic block model.

    Parameters
    ----------
    block_sizes
        Number of nodes in each community.
    p_intra
        Probability of edges within the same community.
    p_inter
        Probability of edges between different communities.
    rng
        NumPy random generator. Uses a fresh default if ``None``.

    Returns
    -------
    np.ndarray
        Symmetric binary adjacency matrix with zero diagonal.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = len(block_sizes)
    n = sum(block_sizes)

    # Initialize the adjacency matrix with zeros
    adj_matrix = np.zeros((n, n))

    # Calculate the starting index of each block
    block_starts = [0]
    for i in range(n_blocks - 1):
        block_starts.append(block_starts[-1] + block_sizes[i])

    for i in range(n_blocks):
        for j in range(i, n_blocks):
            density = p_intra if i == j else p_inter
            block_start_i = block_starts[i]
            block_end_i = block_start_i + block_sizes[i]
            block_start_j = block_starts[j]
            block_end_j = block_start_j + block_sizes[j]

            # Generate random edges within or between blocks
            block_i_size = block_sizes[i]
            block_j_size = block_sizes[j]
            block = (rng.random((block_i_size, block_j_size)) < density).astype(int)

            if i == j:
                # Intra-block: symmetrize by keeping upper triangle and mirroring
                block = np.triu(block) + np.triu(block, 1).T
            adj_matrix[block_start_i:block_end_i, block_start_j:block_end_j] = block

            # Mirror inter-block edges for symmetry (undirected graph)
            if i != j:
                adj_matrix[block_start_j:block_end_j, block_start_i:block_end_i] = (
                    block.T
                )

    # Simple graphs have no self-loops
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def generate_sbm_batch(
    num_graphs: int,
    num_nodes: int,
    *,
    num_blocks: int | tuple[int, int] = 2,
    p_intra: float | tuple[float, float] = 0.7,
    p_inter: float | tuple[float, float] = 0.1,
    block_size_alpha: float | None = None,
    diversity: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate a batch of SBM adjacency matrices.

    Supports two modes:

    * **Fixed**: all hyperparameters scalar and ``diversity == 0``. Each
      graph uses the same ``(num_blocks, p_intra, p_inter)`` and equal
      block sizes. Bitwise-identical to the pre-diversity implementation.
    * **Diverse**: at least one hyperparameter is a ``(min, max)`` tuple
      and/or ``diversity > 0``. Each graph draws its own hyperparameters
      from ranges shrunk toward the midpoint by ``(1 − diversity)``; at
      ``diversity = 1`` the full range is used, at ``diversity = 0`` the
      ranges collapse to their midpoint so tuple arguments reduce to
      scalars. ``block_size_alpha``, when set, replaces the equal-block
      partitioning with a Dirichlet-sampled partition of ``num_nodes``.

    This function powers the eigenspectrum-diversity knob used by the
    improvement-gap sweep; see
    ``internal notes (not included)``.

    Parameters
    ----------
    num_graphs
        Number of graphs to generate.
    num_nodes
        Number of nodes per graph.
    num_blocks
        Number of communities (int) or inclusive integer range
        ``(min, max)``.
    p_intra
        Intra-block edge probability or inclusive range ``(min, max)``.
    p_inter
        Inter-block edge probability or inclusive range ``(min, max)``.
    block_size_alpha
        Concentration for a symmetric Dirichlet partition of
        ``num_nodes`` across blocks. ``None`` (default) keeps the
        equal-block behaviour of the fixed mode.
    diversity
        Value in ``[0, 1]``. Scales all tuple ranges toward their
        midpoint — ``0`` → midpoint only, ``1`` → full range. Must be
        ``0`` when all hyperparameters are scalar and
        ``block_size_alpha`` is ``None``.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, num_nodes, num_nodes)``,
        dtype ``float32``.

    Raises
    ------
    ValueError
        If ``diversity`` is outside ``[0, 1]``, if all hyperparameters
        are scalar and ``block_size_alpha is None`` but ``diversity > 0``
        (the knob has nothing to vary), or if any tuple is malformed.
    """
    if not 0.0 <= diversity <= 1.0:
        raise ValueError(f"diversity must be in [0, 1]; got {diversity}")

    num_blocks_is_range = isinstance(num_blocks, tuple)
    p_intra_is_range = isinstance(p_intra, tuple)
    p_inter_is_range = isinstance(p_inter, tuple)
    uses_dirichlet = block_size_alpha is not None

    all_scalars = (
        not num_blocks_is_range
        and not p_intra_is_range
        and not p_inter_is_range
        and not uses_dirichlet
    )
    if all_scalars and diversity > 0.0:
        raise ValueError(
            "diversity > 0 requires at least one tuple-ranged "
            "hyperparameter (num_blocks/p_intra/p_inter) or "
            "block_size_alpha; otherwise the knob has nothing to vary."
        )

    rng = np.random.default_rng(seed)

    # Fixed mode: all scalars and zero diversity → preserve the original
    # RNG sequence bit-for-bit so existing fixtures / seeded experiments
    # don't shift.
    if all_scalars and diversity == 0.0:
        assert isinstance(num_blocks, int)
        assert isinstance(p_intra, int | float)
        assert isinstance(p_inter, int | float)
        block_sizes = _equal_block_sizes(num_nodes, num_blocks)
        adjacencies: list[np.ndarray] = []
        for _ in range(num_graphs):
            adj = generate_sbm_adjacency(
                block_sizes=block_sizes,
                p_intra=float(p_intra),
                p_inter=float(p_inter),
                rng=rng,
            )
            adjacencies.append(adj.astype(np.float32))
        return np.stack(adjacencies, axis=0)

    # Diverse mode: sample per-graph hyperparameters, then generate.
    nb_range = _range_from_arg(num_blocks, diversity, kind="int")
    pi_range = _range_from_arg(p_intra, diversity, kind="float")
    po_range = _range_from_arg(p_inter, diversity, kind="float")

    adjacencies = []
    for _ in range(num_graphs):
        this_num_blocks = _sample_int_range(rng, nb_range)
        this_p_intra = _sample_float_range(rng, pi_range)
        this_p_inter = _sample_float_range(rng, po_range)
        if uses_dirichlet:
            assert block_size_alpha is not None  # narrow for type-checker
            this_block_sizes = _dirichlet_block_sizes(
                rng, num_nodes, this_num_blocks, block_size_alpha
            )
        else:
            this_block_sizes = _equal_block_sizes(num_nodes, this_num_blocks)
        adj = generate_sbm_adjacency(
            block_sizes=this_block_sizes,
            p_intra=this_p_intra,
            p_inter=this_p_inter,
            rng=rng,
        )
        adjacencies.append(adj.astype(np.float32))
    return np.stack(adjacencies, axis=0)


def _equal_block_sizes(num_nodes: int, num_blocks: int) -> list[int]:
    """Split ``num_nodes`` into ``num_blocks`` near-equal parts.

    The remainder is distributed across the earliest blocks, matching
    the fixed-mode behaviour relied on by pre-diversity fixtures.
    """
    block_size = num_nodes // num_blocks
    remainder = num_nodes % num_blocks
    block_sizes = [block_size] * num_blocks
    for i in range(remainder):
        block_sizes[i] += 1
    return block_sizes


def _range_from_arg(
    arg: int | float | tuple[int, int] | tuple[float, float],
    diversity: float,
    *,
    kind: str,
) -> tuple[float, float]:
    """Resolve a scalar-or-range arg into a diversity-scaled ``(lo, hi)``.

    At ``diversity = 1`` the range equals the input (width preserved).
    At ``diversity = 0`` it collapses to the midpoint. Intermediate
    values interpolate linearly. Scalar arguments return
    ``(scalar, scalar)`` regardless of diversity.
    """
    if isinstance(arg, tuple):
        lo, hi = arg
        if lo > hi:
            raise ValueError(f"{kind} range malformed: min={lo} > max={hi}")
        midpoint = (lo + hi) / 2.0
        half_width = (hi - lo) / 2.0 * diversity
        return (midpoint - half_width, midpoint + half_width)
    return (float(arg), float(arg))


def _sample_int_range(rng: np.random.Generator, rng_bounds: tuple[float, float]) -> int:
    """Draw an integer from an inclusive ``(lo, hi)`` range.

    If the range has collapsed (lo == hi) no RNG draw is taken so that
    callers with pure-integer input don't desync their stream.
    """
    lo_f, hi_f = rng_bounds
    lo = int(round(lo_f))
    hi = int(round(hi_f))
    if lo == hi:
        return lo
    return int(rng.integers(lo, hi + 1))


def _sample_float_range(
    rng: np.random.Generator, rng_bounds: tuple[float, float]
) -> float:
    """Draw a float uniformly from ``[lo, hi]``."""
    lo, hi = rng_bounds
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def _dirichlet_block_sizes(
    rng: np.random.Generator, num_nodes: int, num_blocks: int, alpha: float
) -> list[int]:
    """Sample unequal block sizes via a symmetric Dirichlet partition.

    Uses symmetric ``Dirichlet(alpha, ..., alpha)`` then rounds and
    distributes rounding error into the first blocks so the partition
    sums to ``num_nodes``. A floor of 1 prevents zero-sized blocks.

    This is the paper's synthetic-SBM block-size heterogeneity knob
    (Appendix B.2.3, Eq. 28: ``(s_1, ..., s_K) ~ n * Dirichlet(alpha, ..., alpha)``,
    rounded to sum ``n`` with a minimum block size of one). Small ``alpha``
    concentrates mass on a single block (strongly unbalanced communities);
    large ``alpha`` recovers near-equal blocks. Reached from config via
    ``graph_config.block_size_alpha`` (wired through ``graph_generation.py``).
    """
    if alpha <= 0:
        raise ValueError(f"block_size_alpha must be > 0; got {alpha}")
    if num_blocks == 1:
        return [num_nodes]
    weights = rng.dirichlet(np.full(num_blocks, alpha))
    sizes = np.maximum(np.round(weights * num_nodes).astype(int), 1)
    diff = num_nodes - int(sizes.sum())
    # Distribute any rounding slack into the first blocks.
    i = 0
    while diff != 0 and i < num_blocks * 10:
        idx = i % num_blocks
        if diff > 0:
            sizes[idx] += 1
            diff -= 1
        elif sizes[idx] > 1:
            sizes[idx] -= 1
            diff += 1
        i += 1
    return sizes.tolist()


def generate_block_sizes(
    n: int,
    min_blocks: int = 2,
    max_blocks: int = 4,
    min_size: int = 2,
    max_size: int = 15,
) -> list[list[int]]:
    """Generate all valid block size partitions for n nodes.

    Finds all ways to partition n nodes into blocks where the number of blocks
    is in ``[min_blocks, max_blocks]``, each block size is in
    ``[min_size, max_size]``, and all sizes sum to n.  The algorithm uses
    recursive backtracking, pruning branches where the remaining nodes cannot
    satisfy the min/max size constraints for the remaining blocks.

    Parameters
    ----------
    n : int
        Total number of nodes to partition.
    min_blocks : int
        Minimum number of blocks in a partition.
    max_blocks : int
        Maximum number of blocks in a partition.
    min_size : int
        Minimum size for any single block.
    max_size : int
        Maximum size for any single block.

    Returns
    -------
    list[list[int]]
        All valid block size partitions; each partition is a list of integers
        summing to n.

    Examples
    --------
    >>> generate_block_sizes(6, min_blocks=2, max_blocks=3, min_size=2, max_size=4)
    [[2, 4], [3, 3], [4, 2], [2, 2, 2]]
    """
    valid_partitions = []

    def generate_partitions(remaining, blocks_left, current_partition):
        """Recursively generate partitions for remaining nodes.

        Ensures that after choosing a size for the current block, the
        remaining nodes can still be validly partitioned into the remaining
        blocks while respecting min_size/max_size constraints.

        Parameters
        ----------
        remaining
            Number of nodes left to partition.
        blocks_left
            Number of blocks left to create (including current).
        current_partition
            Current partial partition being built (mutated in place).
        """
        if blocks_left == 0:
            if remaining == 0:
                valid_partitions.append(current_partition[:])
            return

        # Lower bound: current block must be large enough that remaining
        # blocks can fit within max_size.
        start = max(min_size, remaining - (blocks_left - 1) * max_size)

        # Upper bound: current block must be small enough that remaining
        # blocks can satisfy min_size.
        end = min(max_size, remaining - (blocks_left - 1) * min_size) + 1

        for size in range(start, end):
            if size <= remaining:
                current_partition.append(size)
                generate_partitions(
                    remaining - size, blocks_left - 1, current_partition
                )
                current_partition.pop()

    # Try different numbers of blocks
    for num_blocks in range(min_blocks, max_blocks + 1):
        generate_partitions(n, num_blocks, [])

    return valid_partitions
