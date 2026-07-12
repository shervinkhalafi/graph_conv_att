"""Synthetic graph generators for denoising experiments.

Generator functions produce batches of adjacency matrices from NetworkX graph
constructors. ``SyntheticGraphDataset`` wraps these with a uniform interface
that supports both fixed-size graphs (all graphs share node count ``n``) and
variable-size graphs (padded to ``max_n`` with ``node_counts`` tracking).
"""

# pyright: reportConstantRedefinition=false
# pyright: reportArgumentType=false
# G is reassigned across retry loops (math notation). NetworkX's
# to_numpy_array() dtype parameter has incomplete type stubs.

import math
import random

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_regular_graphs(
    n: int,
    d: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random d-regular graphs.

    A d-regular graph is a graph where every node has exactly degree d.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    d : int
        Degree of each node. Must satisfy d < n and n*d must be even.
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Raises
    ------
    ValueError
        If d >= n or n*d is odd (no d-regular graph exists).
    """

    if d >= n:
        raise ValueError(f"Degree d={d} must be less than n={n}")
    if (n * d) % 2 != 0:
        raise ValueError(f"n*d must be even, got n={n}, d={d}")

    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.random_regular_graph(d, n, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_tree_graphs(
    n: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random tree graphs using Prüfer sequences.

    A tree is a connected acyclic graph with n-1 edges. Trees are generated
    by sampling random Prüfer sequences and converting them to trees.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).
    """
    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        # Set random seed for this graph
        random.seed(base_seed + i)
        np.random.seed(base_seed + i)

        # Generate random Prüfer sequence and convert to tree
        # Prüfer sequence has length n-2, with elements in [0, n-1]
        prufer_seq = [random.randint(0, n - 1) for _ in range(n - 2)]
        G = nx.from_prufer_sequence(prufer_seq)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_lfr_graphs(
    n: int,
    num_graphs: int,
    tau1: float = 3.0,
    tau2: float = 1.5,
    mu: float = 0.1,
    average_degree: int | None = None,
    min_community: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate LFR (Lancichinetti-Fortunato-Radicchi) benchmark graphs.

    LFR graphs are synthetic networks with planted community structure,
    commonly used for testing community detection algorithms.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    tau1 : float, optional
        Power law exponent for degree distribution. Default 3.0.
    tau2 : float, optional
        Power law exponent for community size distribution. Default 1.5.
    mu : float, optional
        Mixing parameter (fraction of inter-community edges). Lower values
        mean stronger community structure. Default 0.1.
    average_degree : int, optional
        Target average degree. Default is n // 10.
    min_community : int, optional
        Minimum community size. Default is n // 10.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Notes
    -----
    The LFR generator can fail for certain parameter combinations. This
    function will retry with slightly different parameters if generation
    fails.
    """
    if average_degree is None:
        average_degree = max(3, n // 10)
    if min_community is None:
        min_community = max(5, n // 10)

    base_seed = seed if seed is not None else 42
    adjacencies = []

    max_degree = n - 1
    max_community = n // 2

    for i in range(num_graphs):
        graph_seed = base_seed + i

        # LFR can be finicky; try a few times with slight variations
        G = None
        for attempt in range(5):
            try:
                G = nx.LFR_benchmark_graph(
                    n=n,
                    tau1=tau1,
                    tau2=tau2,
                    mu=mu,
                    average_degree=average_degree,
                    max_degree=max_degree,
                    min_community=min_community,
                    max_community=max_community,
                    seed=graph_seed + attempt * 1000,
                )
                break
            except nx.ExceededMaxIterations:
                # Adjust parameters slightly and retry
                average_degree = max(3, average_degree - 1)
                continue

        if G is None:
            # Fall back to SBM-like structure if LFR fails
            # This shouldn't happen often with reasonable parameters
            G = nx.planted_partition_graph(
                l=max(2, n // min_community),
                k=min_community,
                p_in=1 - mu,
                p_out=mu,
                seed=graph_seed,
            )

        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_erdos_renyi_graphs(
    n: int,
    p: float,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Erdős-Rényi random graphs.

    Each edge exists independently with probability p.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    p : float
        Edge probability (between 0 and 1).
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).
    """
    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.fast_gnp_random_graph(n, p, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_watts_strogatz_graphs(
    n: int,
    num_graphs: int,
    k: int = 4,
    p: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Watts-Strogatz small-world graphs.

    Small-world graphs have high clustering and short path lengths,
    mimicking properties of social networks and neural networks.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    k : int, optional
        Each node is connected to k nearest neighbors in ring topology.
        Must be even. Default is 4.
    p : float, optional
        Probability of rewiring each edge. Default is 0.3.
        p=0 gives a ring lattice, p=1 gives random graph.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Raises
    ------
    ValueError
        If k >= n or k is odd.
    """
    if k >= n:
        raise ValueError(f"k={k} must be less than n={n}")
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")

    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.watts_strogatz_graph(n, k, p, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_random_geometric_graphs(
    n: int,
    num_graphs: int,
    radius: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random geometric graphs.

    Nodes are placed uniformly at random in a unit square, and edges
    connect nodes whose Euclidean distance is at most `radius`.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    radius : float, optional
        Distance threshold for edge creation. Default is 0.3.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).
    """
    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.random_geometric_graph(n, radius, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_configuration_model_graphs(
    n: int,
    num_graphs: int,
    degree_sequence: list[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random graphs with a specified degree sequence.

    The configuration model generates random graphs with a given degree
    sequence. If no sequence is provided, a default power-law-like
    sequence is used.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    degree_sequence : list of int, optional
        Desired degree for each node. Must sum to even number.
        If None, uses a default sequence based on node count.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Notes
    -----
    The configuration model may produce multigraphs or self-loops. This
    implementation removes them, so actual degrees may be slightly lower
    than specified.
    """
    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        random.seed(graph_seed)
        np.random.seed(graph_seed)

        if degree_sequence is None:
            # Generate a power-law-like degree sequence
            # Most nodes have low degree, few have high degree
            seq = np.random.zipf(2.0, n)
            seq = np.clip(seq, 1, n - 1).astype(int)
            # Ensure sum is even
            if seq.sum() % 2 != 0:
                seq[0] += 1
            deg_seq = seq.tolist()
        else:
            deg_seq = list(degree_sequence)

        # Create graph from degree sequence
        G = nx.configuration_model(deg_seq, seed=graph_seed)
        # Remove parallel edges and self-loops
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        A = nx.to_numpy_array(G, dtype=np.float32, nodelist=range(n))
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_ring_of_cliques_graphs(
    num_cliques: int,
    clique_size: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate ring-of-cliques graphs via networkx.

    The topology is deterministic for given parameters, so every graph
    in the returned batch is identical — valid for denoising where
    noise realizations differ but the clean target does not.

    Parameters
    ----------
    num_cliques : int
        Number of cliques arranged in a ring.
    clique_size : int
        Number of nodes per clique.
    num_graphs : int
        How many (identical) copies to produce.
    seed : int or None
        Unused (topology is deterministic). Accepted for interface
        consistency with the other generators.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, n, n)`` where
        ``n = num_cliques * clique_size``.
    """
    G = nx.ring_of_cliques(num_cliques, clique_size)
    A = nx.to_numpy_array(G, dtype=np.float32)
    return np.stack([A] * num_graphs, axis=0)


def generate_lollipop_graphs(
    cluster_size: int,
    path_length: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate lollipop graphs (complete graph + path) via networkx.

    Deterministic topology: every graph in the batch is identical.

    Parameters
    ----------
    cluster_size
        Number of nodes in the complete-graph portion.
    path_length
        Number of nodes in the path portion.
    num_graphs
        How many identical copies to produce.
    seed
        Unused (deterministic). Accepted for interface consistency.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, n, n)`` where
        ``n = cluster_size + path_length``.
    """
    G = nx.lollipop_graph(cluster_size, path_length)
    A = nx.to_numpy_array(G, dtype=np.float32)
    return np.stack([A] * num_graphs, axis=0)


def generate_circular_ladder_graphs(
    n: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate circular ladder graphs via networkx.

    A circular ladder has two concentric rings of ``n`` nodes each, with
    corresponding nodes connected by rungs, yielding ``2n`` total nodes.

    Parameters
    ----------
    n
        Number of nodes per ring (total graph has ``2n`` nodes).
    num_graphs
        How many identical copies to produce.
    seed
        Unused (deterministic). Accepted for interface consistency.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, 2n, 2n)``.
    """
    G = nx.circular_ladder_graph(n)
    A = nx.to_numpy_array(G, dtype=np.float32)
    return np.stack([A] * num_graphs, axis=0)


def generate_star_graphs(
    n: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate star graphs via networkx.

    A star graph has one center node connected to ``n`` leaf nodes,
    yielding ``n + 1`` total nodes.

    Parameters
    ----------
    n
        Number of arms (leaf nodes). Total graph has ``n + 1`` nodes.
    num_graphs
        How many identical copies to produce.
    seed
        Unused (deterministic). Accepted for interface consistency.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, n+1, n+1)``.
    """
    G = nx.star_graph(n)
    A = nx.to_numpy_array(G, dtype=np.float32)
    return np.stack([A] * num_graphs, axis=0)


def generate_square_grid_graphs(
    n: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate square grid graphs via networkx.

    Computes height and width to produce a grid with at least ``n`` nodes.
    The actual node count may be slightly larger than ``n``.

    Parameters
    ----------
    n
        Minimum number of nodes. Actual count is ``h * w`` where
        ``h = ceil(sqrt(n))`` and ``w`` is chosen so ``h * w >= n``.
    num_graphs
        How many identical copies to produce.
    seed
        Unused (deterministic). Accepted for interface consistency.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, h*w, h*w)``.
    """
    h = int(math.ceil(math.sqrt(n)))
    w = int(math.floor(n / h))
    while h * w < n:
        w += 1
    G = nx.grid_graph([h, w])
    A = nx.to_numpy_array(G, dtype=np.float32)
    return np.stack([A] * num_graphs, axis=0)


def generate_triangle_grid_graphs(
    n: int,
    num_graphs: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate triangular lattice graphs via networkx.

    Computes lattice dimension to produce at least ``n`` nodes. The actual
    node count may be slightly larger.

    Parameters
    ----------
    n
        Minimum number of nodes.
    num_graphs
        How many identical copies to produce.
    seed
        Unused (deterministic). Accepted for interface consistency.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, actual_n, actual_n)``.
    """
    x = int(math.ceil(math.sqrt(n)))
    while nx.number_of_nodes(nx.triangular_lattice_graph(x, x)) < n:
        x += 1
    G = nx.triangular_lattice_graph(x, x)
    A = nx.to_numpy_array(G, dtype=np.float32)
    return np.stack([A] * num_graphs, axis=0)


class SyntheticGraphDataset(Dataset[np.ndarray]):
    """Torch Dataset wrapper for synthetic adjacency matrices.

    Provides a unified interface for generating and accessing synthetic
    graph datasets of various types. Supports both fixed-size graphs (all
    graphs share the same node count) and deterministic-topology types
    where the actual node count derives from the generator parameters.
    Each item is a single adjacency matrix rather than a PyG ``Data``
    object, because the denoising experiments operate on dense adjacency
    tensors directly.

    Parameters
    ----------
    graph_type : str
        Type of graph. Supported types and their kwargs:

        ============== ===================================================
        Type           Extra kwargs
        ============== ===================================================
        regular        d (degree)
        tree           —
        lfr            tau1, tau2, mu, average_degree, min_community
        erdos_renyi    p (edge probability)
        watts_strogatz k (neighbors), p (rewiring probability)
        random_geometric radius (edge threshold)
        configuration_model degree_sequence
        ring_of_cliques num_cliques, clique_size
        lollipop       cluster_size, path_length
        circular_ladder —
        star           —
        square_grid    —
        triangle_grid  —
        ============== ===================================================

    n : int
        Number of nodes per graph. For deterministic-topology types
        (lollipop, circular_ladder, star, ring_of_cliques) the actual
        node count derives from the parameters; ``n`` is used as a hint
        or directly depending on the type. For grid types, ``n`` is the
        *minimum* node count (actual may be slightly larger).
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    adjacencies : np.ndarray
        Generated adjacency matrices, shape ``(num_graphs, max_n, max_n)``.
    node_counts : np.ndarray
        Actual node count per graph, shape ``(num_graphs,)``.
    max_n : int
        Padded matrix dimension (equals ``num_nodes`` for fixed-size types).
    """

    VALID_TYPES = {
        "regular",
        "tree",
        "lfr",
        "erdos_renyi",
        "watts_strogatz",
        "random_geometric",
        "configuration_model",
        "ring_of_cliques",
        "lollipop",
        "circular_ladder",
        "star",
        "square_grid",
        "triangle_grid",
    }

    TYPE_ALIASES = {
        "er": "erdos_renyi",
        "ws": "watts_strogatz",
        "rg": "random_geometric",
        "cm": "configuration_model",
    }

    def __init__(
        self,
        graph_type: str,
        num_nodes: int,
        num_graphs: int,
        seed: int | None = None,
        **kwargs,
    ):
        canonical_type = self.TYPE_ALIASES.get(graph_type, graph_type)
        if canonical_type not in self.VALID_TYPES:
            raise ValueError(
                f"graph_type must be one of {self.VALID_TYPES}, got '{graph_type}'"
            )

        self.graph_type = canonical_type
        self.num_nodes = num_nodes
        n = num_nodes  # local shorthand for generator calls
        self.num_graphs = num_graphs
        self.seed = seed
        self.kwargs = kwargs

        # Generate graphs — each branch sets self.adjacencies
        if canonical_type == "regular":
            d = kwargs.get("d", 3)
            self.adjacencies = generate_regular_graphs(n, d, num_graphs, seed)
        elif canonical_type == "tree":
            self.adjacencies = generate_tree_graphs(n, num_graphs, seed)
        elif canonical_type == "lfr":
            self.adjacencies = generate_lfr_graphs(n, num_graphs, seed=seed, **kwargs)
        elif canonical_type == "erdos_renyi":
            p = kwargs.get("p", 0.1)
            self.adjacencies = generate_erdos_renyi_graphs(n, p, num_graphs, seed)
        elif canonical_type == "watts_strogatz":
            k = kwargs.get("k", 4)
            p = kwargs.get("p", 0.3)
            self.adjacencies = generate_watts_strogatz_graphs(n, num_graphs, k, p, seed)
        elif canonical_type == "random_geometric":
            radius = kwargs.get("radius", 0.3)
            self.adjacencies = generate_random_geometric_graphs(
                n, num_graphs, radius, seed
            )
        elif canonical_type == "configuration_model":
            degree_sequence = kwargs.get("degree_sequence")
            self.adjacencies = generate_configuration_model_graphs(
                n, num_graphs, degree_sequence, seed
            )
        elif canonical_type == "ring_of_cliques":
            num_cliques = kwargs.get("num_cliques", 4)
            clique_size = kwargs.get("clique_size", 5)
            self.adjacencies = generate_ring_of_cliques_graphs(
                num_cliques, clique_size, num_graphs, seed
            )
        elif canonical_type == "lollipop":
            cluster_size = kwargs.get("cluster_size", n // 2)
            path_length = kwargs.get("path_length", n - n // 2)
            self.adjacencies = generate_lollipop_graphs(
                cluster_size, path_length, num_graphs, seed
            )
        elif canonical_type == "circular_ladder":
            self.adjacencies = generate_circular_ladder_graphs(n, num_graphs, seed)
        elif canonical_type == "star":
            self.adjacencies = generate_star_graphs(n, num_graphs, seed)
        elif canonical_type == "square_grid":
            self.adjacencies = generate_square_grid_graphs(n, num_graphs, seed)
        elif canonical_type == "triangle_grid":
            self.adjacencies = generate_triangle_grid_graphs(n, num_graphs, seed)

        # Derive actual node count from generated matrix dimension
        actual_n = self.adjacencies.shape[-1]
        self.max_n = actual_n
        self.node_counts = np.full(num_graphs, actual_n, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.adjacencies[idx]

    def get_adjacency_matrices(self) -> np.ndarray:
        """Return all adjacency matrices.

        Returns
        -------
        np.ndarray
            Adjacency matrices of shape ``(num_graphs, max_n, max_n)``.
        """
        return self.adjacencies

    def get_node_counts(self) -> np.ndarray:
        """Return actual node count per graph.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(num_graphs,)``.
        """
        return self.node_counts

    def get_masks(self) -> np.ndarray:
        """Return boolean masks indicating valid (non-padded) nodes.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(num_graphs, max_n)`` where
            ``True`` means the node is real, ``False`` means padding.
        """
        indices = np.arange(self.max_n)
        return indices[np.newaxis, :] < self.node_counts[:, np.newaxis]

    def to_torch(self) -> torch.Tensor:
        """Convert adjacency matrices to PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Adjacency matrices as float32 tensor.
        """
        return torch.from_numpy(self.adjacencies).float()
