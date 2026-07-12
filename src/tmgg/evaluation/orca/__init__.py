"""ORCA orbit counting wrapper with auto-compilation.

Wraps the ORCA C++ binary (Hocevar & Demsar, 2014) for computing 4-node
graphlet orbit counts. The binary is compiled on first use from the bundled
``orca.cpp`` source. Upstream DiGress uses these orbit counts for its
``motif_stats`` evaluation metric.

References
----------
.. [1] Hocevar, T. & Demsar, J. (2014). "A combinatorial approach to
   graphlet counting." Bioinformatics, 30(4), 559-565.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

# Sentinel string that ORCA prints before the orbit count matrix
_COUNT_START_STR = "orbit counts:"

# Directory containing the C++ source and compiled binary
_ORCA_DIR = Path(__file__).parent
_REBUILT_BINARY_PATHS: set[Path] = set()


def _get_binary_path() -> Path:
    """Return path to the ORCA binary, rebuilding it on first use.

    Modal containers can expose the local source tree under ``/root/tmgg``.
    If that tree contains a host-built ORCA executable, reusing it can fail
    with glibc/libstdc++ mismatches. To avoid trusting any preexisting native
    artifact, the wrapper recompiles ORCA once per process from the bundled
    source and atomically replaces any existing binary in-place.

    The compile target is fixed at ``-march=x86-64-v3`` (AVX2 + BMI2 + FMA,
    no AVX-512). Modal's cheap-tier CPU pool is heterogeneous and some
    hosts lack AVX-512; binaries baked with ``-march=native`` on an
    AVX-512 build host SIGILL when the runtime host is plainer. v3 is the
    common floor across every modern x86-64 host we have observed on Modal.
    """
    binary = _ORCA_DIR / "orca"
    source = _ORCA_DIR / "orca.cpp"
    if not source.exists():
        raise FileNotFoundError(
            f"ORCA source not found at {source}. "
            "Expected orca.cpp in the same directory as this module."
        )

    if (
        binary in _REBUILT_BINARY_PATHS
        and binary.exists()
        and os.access(binary, os.X_OK)
    ):
        return binary

    build_binary = binary.with_suffix(".tmp")

    try:
        subprocess.run(
            [
                "g++",
                "-O2",
                "-march=x86-64-v3",
                "-std=c++11",
                "-o",
                str(build_binary),
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        build_binary.replace(binary)
    except FileNotFoundError:
        raise RuntimeError(
            "g++ not found. Install a C++ compiler to use ORCA orbit counting:\n"
            "  Ubuntu/Debian: sudo apt install g++\n"
            "  macOS: xcode-select --install\n"
            "  Arch: sudo pacman -S gcc"
        ) from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to compile ORCA binary:\n{e.stderr}") from e
    finally:
        build_binary.unlink(missing_ok=True)

    _REBUILT_BINARY_PATHS.add(binary)
    return binary


def _edge_list_reindexed(G: nx.Graph[Any]) -> list[tuple[int, int]]:
    """Re-index graph nodes to contiguous 0-based integers.

    ORCA expects node IDs in [0, N). NetworkX graphs may have
    arbitrary node labels, so we remap them.
    """
    id2idx: dict[Any, int] = {}
    for idx, node in enumerate(G.nodes()):
        id2idx[node] = idx

    return [(id2idx[u], id2idx[v]) for u, v in G.edges()]


def is_available() -> bool:
    """Check whether the ORCA binary exists (or can be compiled).

    Returns True if the binary is present and executable, or if the
    source file exists and could be compiled. Does not attempt compilation.
    """
    binary = _ORCA_DIR / "orca"
    if binary.exists() and os.access(binary, os.X_OK):
        return True
    source = _ORCA_DIR / "orca.cpp"
    return source.exists()


def run_orca(graph: nx.Graph[Any]) -> np.ndarray:
    """Compute 4-node orbit counts for each node using the ORCA binary.

    Parameters
    ----------
    graph
        NetworkX graph. Must have at least one node.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_nodes, 15)`` with per-node orbit counts
        for all 15 orbit types in 4-node graphlets.

    Raises
    ------
    RuntimeError
        If the ORCA binary cannot be compiled or produces unexpected output.
    ValueError
        If the graph has no nodes.
    """
    n = graph.number_of_nodes()
    if n == 0:
        raise ValueError("Cannot compute orbit counts for an empty graph.")

    binary = _get_binary_path()

    # Write edge list to temporary file (ORCA input format)
    suffix = secrets.token_hex(4)
    tmp_path = Path(tempfile.gettempdir()) / f"orca_input_{suffix}.txt"

    try:
        edges = _edge_list_reindexed(graph)
        with open(tmp_path, "w") as f:
            f.write(f"{n} {len(edges)}\n")
            for u, v in edges:
                f.write(f"{u} {v}\n")

        cmd = [str(binary), "node", "4", str(tmp_path), "std"]
        try:
            output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            raw_output = e.output.decode("utf8", errors="replace").strip()
            input_preview = tmp_path.read_text(encoding="utf8").strip()
            raise RuntimeError(
                "ORCA subprocess failed.\n"
                f"Exit status: {e.returncode}\n"
                f"Command: {cmd!r}\n"
                f"Graph summary: nodes={n}, edges={len(edges)}\n"
                f"ORCA output:\n{raw_output}\n"
                f"Serialized ORCA input:\n{input_preview}"
            ) from e
        output_str = output.decode("utf8").strip()

        # Parse orbit counts from output
        idx = output_str.find(_COUNT_START_STR)
        if idx == -1:
            raise RuntimeError(
                f"ORCA output missing '{_COUNT_START_STR}' marker. "
                f"Raw output:\n{output_str[:500]}"
            )

        counts_str = output_str[idx + len(_COUNT_START_STR) :].strip()
        node_orbit_counts = np.array(
            [
                list(map(int, line.strip().split()))
                for line in counts_str.split("\n")
                if line.strip()
            ]
        )

        if node_orbit_counts.shape != (n, 15):
            raise RuntimeError(
                f"Expected orbit count matrix of shape ({n}, 15), "
                f"got {node_orbit_counts.shape}."
            )

        return node_orbit_counts

    finally:
        tmp_path.unlink(missing_ok=True)
