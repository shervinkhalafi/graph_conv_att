"""Noised eigenstructure collector.

Collects eigendecompositions of noised graphs at configurable noise levels,
reusing original adjacency matrices from Phase 1 collection.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from loguru import logger

from tmgg.utils.noising.noise import create_noise_definition
from tmgg.utils.spectral.laplacian import compute_laplacian
from tmgg.utils.spectral.subspace import (
    align_top_k_to_reference_batch,
    compute_frechet_mean_subspace,
    compute_procrustes_rotation_batch,
)

from .analyzer import (
    CovarianceResult,
    ImprovementGapResult,
    compute_B_invariants,
    estimate_improvement_gap,
)
from .storage import (
    iter_batches,
    load_decomposition_batch,
    load_manifest,
    save_dataset_manifest,
    save_decomposition_batch,
)

DEFAULT_SURROGATE_K_VALUES: tuple[int, ...] = (4, 8, 16, 32)

FrameMode = Literal["per_graph", "frechet"]
ConditioningFeatures = Literal["top_k_eigenvalues", "spectral_gap"]
SurrogateTarget = Literal["matrix", "invariants"]

_EPS = 1e-10  # numerical stability for relative metrics


@dataclass(frozen=True)
class MeanStdMetric:
    """Mean/std summary for one scalar metric aggregated across graphs.

    The comparison code computes many graph-wise tensors and then collapses
    them into compact statistics for reporting. This value object keeps the
    Python API typed without changing the flat JSON files that downstream
    scripts already read.
    """

    mean: float
    std: float

    @classmethod
    def from_tensor(cls, values: torch.Tensor) -> MeanStdMetric:
        """Build a summary from a one-dimensional tensor of metric values."""
        return cls(mean=values.mean().item(), std=values.std().item())

    def to_json_dict(self, *, mean_key: str, std_key: str) -> dict[str, float]:
        """Serialize under explicit field names used at the JSON boundary."""
        return {mean_key: self.mean, std_key: self.std}


@dataclass(frozen=True)
class DriftComparisonResult:
    """Relative eigenvalue drift statistics for adjacency and Laplacian spectra."""

    noise_level: float
    adjacency: MeanStdMetric
    laplacian: MeanStdMetric

    def to_json_dict(self) -> dict[str, float]:
        """Serialize to the legacy flat JSON structure."""
        return {
            "noise_level": self.noise_level,
            **self.adjacency.to_json_dict(
                mean_key="adjacency_drift_mean",
                std_key="adjacency_drift_std",
            ),
            **self.laplacian.to_json_dict(
                mean_key="laplacian_drift_mean",
                std_key="laplacian_drift_std",
            ),
        }


@dataclass(frozen=True)
class GapDeltaComparisonResult:
    """Absolute and relative delta summaries for one spectral scalar metric."""

    noise_level: float
    absolute: MeanStdMetric
    relative: MeanStdMetric

    def to_json_dict(self, prefix: str) -> dict[str, float]:
        """Serialize using a metric-specific prefix such as ``eigengap_delta``."""
        return {
            "noise_level": self.noise_level,
            **self.absolute.to_json_dict(
                mean_key=f"{prefix}_abs_mean",
                std_key=f"{prefix}_abs_std",
            ),
            **self.relative.to_json_dict(
                mean_key=f"{prefix}_rel_mean",
                std_key=f"{prefix}_rel_std",
            ),
        }


@dataclass(frozen=True)
class SubspaceDistanceResult:
    """Projection-Frobenius subspace distance summary for one noise level."""

    noise_level: float
    k: int
    distance: MeanStdMetric

    def to_json_dict(self) -> dict[str, float | int]:
        """Serialize to the legacy flat JSON structure."""
        return {
            "noise_level": self.noise_level,
            "k": self.k,
            **self.distance.to_json_dict(
                mean_key="subspace_distance_mean",
                std_key="subspace_distance_std",
            ),
        }


@dataclass(frozen=True)
class PrincipalAngleResult:
    """Principal-angle summaries for original/noised eigenspace comparisons."""

    noise_level: float
    k: int
    first_principal_angle: MeanStdMetric
    mean_principal_angle: MeanStdMetric

    def to_json_dict(self) -> dict[str, float | int]:
        """Serialize to the legacy flat JSON structure."""
        return {
            "noise_level": self.noise_level,
            "k": self.k,
            **self.first_principal_angle.to_json_dict(
                mean_key="first_principal_angle_mean",
                std_key="first_principal_angle_std",
            ),
            **self.mean_principal_angle.to_json_dict(
                mean_key="mean_principal_angle_mean",
                std_key="mean_principal_angle_std",
            ),
        }


@dataclass(frozen=True)
class ProcrustesKStats:
    """Angle/residual summaries for one Procrustes subspace dimension ``k``."""

    angle: MeanStdMetric
    residual: MeanStdMetric

    def to_json_dict(self, k: int) -> dict[str, float]:
        """Serialize using the historical ``procrustes_*_k{k}_*`` key layout."""
        return {
            **self.angle.to_json_dict(
                mean_key=f"procrustes_angle_k{k}_mean",
                std_key=f"procrustes_angle_k{k}_std",
            ),
            **self.residual.to_json_dict(
                mean_key=f"procrustes_residual_k{k}_mean",
                std_key=f"procrustes_residual_k{k}_std",
            ),
        }


@dataclass(frozen=True)
class ProcrustesRotationResult:
    """Procrustes summaries grouped by requested subspace dimension."""

    noise_level: float
    stats_by_k: dict[int, ProcrustesKStats]

    def to_json_dict(self) -> dict[str, float]:
        """Serialize to the historical flat key layout."""
        payload: dict[str, float] = {"noise_level": self.noise_level}
        for k in sorted(self.stats_by_k):
            payload.update(self.stats_by_k[k].to_json_dict(k))
        return payload


@dataclass(frozen=True)
class NoiseLevelComparisonResult:
    """Full one-pass comparison summary for a single noise level.

    The one-pass collector computes all compatible metrics while paired batches
    are already in memory. Keeping the result structured here prevents schema
    drift while still letting callers emit the existing flat JSON artifact.
    """

    noise_level: float
    k: int
    eigengap_delta_relative: MeanStdMetric
    algebraic_connectivity_delta_relative: MeanStdMetric
    eigenvalue_drift_adjacency: MeanStdMetric
    eigenvalue_drift_laplacian: MeanStdMetric
    subspace_distance: MeanStdMetric
    procrustes_by_k: dict[int, ProcrustesKStats]

    def to_json_dict(self) -> dict[str, float | int]:
        """Serialize to the legacy ``comparison.json`` layout."""
        payload: dict[str, float | int] = {
            "noise_level": self.noise_level,
            "k": self.k,
            **self.eigengap_delta_relative.to_json_dict(
                mean_key="eigengap_delta_rel_mean",
                std_key="eigengap_delta_rel_std",
            ),
            **self.algebraic_connectivity_delta_relative.to_json_dict(
                mean_key="alg_conn_delta_rel_mean",
                std_key="alg_conn_delta_rel_std",
            ),
            **self.eigenvalue_drift_adjacency.to_json_dict(
                mean_key="eigenvalue_drift_adj_mean",
                std_key="eigenvalue_drift_adj_std",
            ),
            **self.eigenvalue_drift_laplacian.to_json_dict(
                mean_key="eigenvalue_drift_lap_mean",
                std_key="eigenvalue_drift_lap_std",
            ),
            **self.subspace_distance.to_json_dict(
                mean_key="subspace_distance_mean",
                std_key="subspace_distance_std",
            ),
        }
        for k in sorted(self.procrustes_by_k):
            payload.update(self.procrustes_by_k[k].to_json_dict(k))
        return payload


@dataclass(frozen=True)
class CovarianceEvolutionItem:
    """Covariance summary and deltas for one noised dataset snapshot."""

    noise_level: float
    covariance: CovarianceResult
    frobenius_delta_relative: float
    trace_delta_relative: float
    off_diagonal_delta_relative: float

    def to_json_dict(self) -> dict[str, object]:
        """Serialize one per-noise-level covariance comparison item."""
        return {
            "noise_level": self.noise_level,
            "covariance": asdict(self.covariance),
            "frobenius_delta_relative": self.frobenius_delta_relative,
            "trace_delta_relative": self.trace_delta_relative,
            "off_diagonal_delta_relative": self.off_diagonal_delta_relative,
        }


@dataclass(frozen=True)
class CovarianceEvolutionResult:
    """Covariance evolution across all available noise levels."""

    matrix_type: str
    original: CovarianceResult
    per_noise_level: list[CovarianceEvolutionItem]

    def to_json_dict(self) -> dict[str, object]:
        """Serialize to the stable ``covariance_evolution.json`` layout."""
        return {
            "matrix_type": self.matrix_type,
            "original": asdict(self.original),
            "per_noise_level": [item.to_json_dict() for item in self.per_noise_level],
        }


class NoisedEigenstructureCollector:
    """
    Collect eigendecompositions for noised versions of graphs.

    Reads original adjacency matrices from Phase 1 output, applies noise
    at multiple levels, computes decompositions, and stores in separate
    subdirectories per noise level.

    Parameters
    ----------
    input_dir : Path
        Directory containing Phase 1 output (batch_*.safetensors and manifest.json).
    output_dir : Path
        Base directory for noised decompositions. Subdirectories eps_X.XXXX/
        will be created for each noise level.
    noise_type : str
        Type of noise: "gaussian", "digress", or "rotation".
    noise_levels : list[float]
        Noise levels (epsilon values) to apply.
    rotation_k : int, optional
        Dimension for rotation noise skew matrix. Required if noise_type="rotation".
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        noise_type: str,
        noise_levels: list[float],
        rotation_k: int | None = None,
        seed: int = 42,
        surrogate_k_values: list[int] | None = None,
        frame_mode: FrameMode = "frechet",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.noise_type = noise_type.lower()
        self.noise_levels = noise_levels
        self.rotation_k = rotation_k
        self.seed = seed
        self.surrogate_k_values: list[int] = (
            list(DEFAULT_SURROGATE_K_VALUES)
            if surrogate_k_values is None
            else list(surrogate_k_values)
        )
        self.frame_mode: FrameMode = frame_mode

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(self.input_dir)
        self._create_noise_definition()
        # Fréchet-mean references are dataset-level (single reference per k
        # shared across every batch and noise level). Compute once up-front
        # so the same frame is applied at every noise level.
        self._reference_frames: dict[int, torch.Tensor] = {}
        if self.frame_mode == "frechet":
            self._reference_frames = self._compute_frechet_references()

    def _create_noise_definition(self) -> None:
        """Initialize the noise definition."""
        if self.noise_type == "rotation" and self.rotation_k is None:
            raise ValueError("rotation_k is required for rotation noise")

        self.noise_gen = create_noise_definition(
            self.noise_type,
            rotation_k=self.rotation_k,
            seed=self.seed,
        )

    def _compute_frechet_references(self) -> dict[int, torch.Tensor]:
        """Precompute the Fréchet mean of clean top-*k* subspaces per *k*.

        Stacks every clean graph's top-*k* eigenvector block and takes the
        top-*k* left singular vectors of the resulting ``(n, N*k)`` matrix
        (extrinsic Grassmannian mean). The resulting reference is
        dataset-level and shared across every noise level, which is the
        prerequisite for ``Cov(B)`` to be well-defined as a population
        quantity — every graph's ``B_i`` ends up expressed in the same
        ``V*_k`` frame, not in its own graph-specific frame.
        """
        blocks_by_k: dict[int, list[torch.Tensor]] = {
            k: [] for k in self.surrogate_k_values
        }
        for batch_path in iter_batches(self.input_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            V_clean = tensors["eigenvectors_adj"].float()
            n = V_clean.shape[1]
            for k in self.surrogate_k_values:
                if k > n:
                    continue
                blocks_by_k[k].append(V_clean[:, :, -k:])

        means: dict[int, torch.Tensor] = {}
        for k, blocks in blocks_by_k.items():
            if not blocks:
                continue
            all_blocks = torch.cat(blocks, dim=0)  # (N_total, n, k)
            means[k] = compute_frechet_mean_subspace(all_blocks)
            logger.info(
                f"Fréchet reference for k={k}: subspace of shape "
                f"{tuple(means[k].shape)} over {all_blocks.shape[0]} graphs"
            )
        return means

    def collect(self) -> None:
        """Run noised collection for all noise levels."""
        for eps in self.noise_levels:
            eps_dir = self.output_dir / f"eps_{eps:.4f}"
            eps_dir.mkdir(exist_ok=True)

            logger.info(f"Processing noise level eps={eps}")
            self._collect_at_noise_level(eps, eps_dir)

            # Save manifest for this noise level
            save_dataset_manifest(
                eps_dir,
                f"{self.manifest['dataset_name']}_noised_{self.noise_type}",
                {
                    **self.manifest["dataset_config"],
                    "noise_type": self.noise_type,
                    "noise_level": eps,
                    "original_dataset": self.manifest["dataset_name"],
                },
                self.manifest["num_graphs"],
                self.manifest["num_batches"],
                self.seed,
            )
            logger.info(f"Completed eps={eps}, saved to {eps_dir}")

    def _collect_at_noise_level(self, eps: float, output_dir: Path) -> None:
        """Collect decompositions for a single noise level.

        In addition to the noisy adjacency/Laplacian decomposition, compute
        the improvement-gap projection ``B_k = V̂_aligned^T A_clean V̂_aligned``
        for each requested top-*k* dimension. The Procrustes step aligns each
        noisy top-*k* eigenvector block to the clean one, fixing sign and
        rotation within degenerate eigenspaces so ``B_k`` is comparable
        across graphs (see
        ``_NeurIPS_2026__Understanding_Graph_Denoising.pdf``, eq. 18).
        """
        for batch_path in iter_batches(self.input_dir):
            tensors, metadata = load_decomposition_batch(batch_path)
            A_clean = tensors["adjacency"]
            V_clean = tensors["eigenvectors_adj"]

            # Apply noise
            A_noised = self.noise_gen.add_noise(A_clean, eps)

            # Process each graph in the batch
            eig_adj_vals_list: list[torch.Tensor] = []
            eig_adj_vecs_list: list[torch.Tensor] = []
            eig_lap_vals_list: list[torch.Tensor] = []
            eig_lap_vecs_list: list[torch.Tensor] = []

            for i in range(A_noised.shape[0]):
                A = A_noised[i]
                L = compute_laplacian(A)

                # Noised adjacency matrices can be numerically
                # ill-conditioned or near-degenerate — especially on real
                # bio benchmarks like ENZYMES where the underlying graph
                # has high symmetry and small perturbations keep
                # eigenvalues effectively repeated. float32 eigh fails to
                # converge on those graphs (observed in the Phase 4 sweep
                # on ENZYMES at ε=0.01). Run eigh in float64 and downcast
                # for storage — precision is cheap, a crash here kills
                # the whole sweep.
                vals_a, vecs_a = torch.linalg.eigh(A.double())
                vals_l, vecs_l = torch.linalg.eigh(L.double())

                eig_adj_vals_list.append(vals_a.float())
                eig_adj_vecs_list.append(vecs_a.float())
                eig_lap_vals_list.append(vals_l.float())
                eig_lap_vecs_list.append(vecs_l.float())

            eig_adj_vecs = torch.stack(eig_adj_vecs_list)

            extra_tensors = self._compute_surrogate_projections(
                A_clean=A_clean.float(),
                V_clean=V_clean.float(),
                V_noisy=eig_adj_vecs.float(),
            )

            # Extract batch index from filename
            batch_idx = int(batch_path.stem.split("_")[1])

            save_decomposition_batch(
                output_dir,
                batch_idx,
                torch.stack(eig_adj_vals_list),
                eig_adj_vecs,
                torch.stack(eig_lap_vals_list),
                torch.stack(eig_lap_vecs_list),
                A_noised,
                metadata,
                extra_tensors=extra_tensors,
            )

    def _compute_surrogate_projections(
        self,
        *,
        A_clean: torch.Tensor,
        V_clean: torch.Tensor,
        V_noisy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Procrustes-aligned top-*k* projections of the clean signal.

        For each ``k`` in ``self.surrogate_k_values``, returns the per-graph
        tensor ``B_k[i] = V̂_aligned_i^T A_clean_i V̂_aligned_i``. The
        alignment reference depends on ``self.frame_mode``:

        * ``"frechet"`` (default) — every ``V̂_{k,i}`` is Procrustes-aligned
          to a single dataset-wide Fréchet mean subspace ``V*_k``. This
          puts every ``B_i`` into the same frame, the prerequisite for
          ``Cov(vec(B))`` and ``E[B | Λ̃_k]`` to be well-defined population
          quantities in eq. (18) of the NeurIPS 2026 draft.
        * ``"per_graph"`` — each ``V̂_{k,i}`` is aligned to that graph's
          own clean ``V_{k,i}``. This keeps ``B_i`` in a graph-specific
          frame and is retained as a diagnostic; it is **not** the frame
          used to claim eq. (18) validity across a heterogeneous dataset.

        Parameters
        ----------
        A_clean : torch.Tensor
            Clean adjacency matrices, shape ``(batch, n, n)``.
        V_clean : torch.Tensor
            Clean eigenvector matrices from the Phase 1 decomposition,
            columns sorted by ascending eigenvalue, shape ``(batch, n, n)``.
        V_noisy : torch.Tensor
            Noisy eigenvector matrices, same shape and column order.

        Returns
        -------
        dict[str, torch.Tensor]
            One ``B_k{k}`` entry per requested ``k`` (only those with
            ``k <= n`` are included).
        """
        n = V_clean.shape[1]
        projections: dict[str, torch.Tensor] = {}
        for k in self.surrogate_k_values:
            if k > n:
                logger.warning(f"surrogate_k_values contains k={k} > n={n}; skipping")
                continue
            if self.frame_mode == "per_graph":
                rotation = compute_procrustes_rotation_batch(
                    V_clean, V_noisy, k, return_rotation=True
                )["rotations"]  # (batch, k, k)
                V_aligned = V_noisy[:, :, -k:] @ rotation  # (batch, n, k)
            elif self.frame_mode == "frechet":
                V_ref = self._reference_frames[k]  # (n, k)
                V_aligned = align_top_k_to_reference_batch(V_ref, V_noisy, k)
            else:
                raise ValueError(f"Unknown frame_mode '{self.frame_mode}'")
            # B = V_aligned^T A_clean V_aligned  → (batch, k, k)
            B = V_aligned.transpose(-2, -1) @ A_clean @ V_aligned
            projections[f"B_k{k}"] = B
        return projections


class NoisedAnalysisComparator:
    """
    Compare eigenstructure between original and noised graphs.

    Parameters
    ----------
    original_dir : Path
        Directory with original (Phase 1) decompositions.
    noised_base_dir : Path
        Base directory with noised decompositions (contains eps_*/ subdirs).
    """

    def __init__(self, original_dir: Path, noised_base_dir: Path):
        self.original_dir = Path(original_dir)
        self.noised_base_dir = Path(noised_base_dir)

    def get_noise_levels(self) -> list[float]:
        """Get available noise levels from directory structure."""
        levels = []
        for subdir in self.noised_base_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("eps_"):
                try:
                    eps = float(subdir.name.replace("eps_", ""))
                    levels.append(eps)
                except ValueError:
                    continue
        return sorted(levels)

    def _iter_paired_batches(
        self, eps: float
    ) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
        """Yield paired (original, noised) tensor dicts for each batch.

        Parameters
        ----------
        eps : float
            Noise level identifying the noised subdirectory.

        Yields
        ------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
            ``(original_tensors, noised_tensors)`` for each corresponding batch.

        Raises
        ------
        ValueError
            If the original and noised directories contain different numbers
            of batch files (via ``strict=True`` zip).
        """
        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"
        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)
        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=True
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)
            yield orig_tensors, noised_tensors

    def compute_eigenvalue_drift(self, eps: float) -> DriftComparisonResult:
        """Compute eigenvalue drift between original and noised graphs.

        Parameters
        ----------
        eps : float
            Noise level to compare.

        Returns
        -------
        DriftComparisonResult
            Relative drift summaries for adjacency and Laplacian eigenvalues.
        """
        adj_drifts: list[torch.Tensor] = []
        lap_drifts: list[torch.Tensor] = []

        for orig, noised in self._iter_paired_batches(eps):
            orig_adj = orig["eigenvalues_adj"]
            noised_adj = noised["eigenvalues_adj"]
            orig_lap = orig["eigenvalues_lap"]
            noised_lap = noised["eigenvalues_lap"]

            adj_diff = torch.norm(noised_adj - orig_adj, dim=-1)
            adj_norm = torch.norm(orig_adj, dim=-1) + _EPS
            adj_drifts.append(adj_diff / adj_norm)

            lap_diff = torch.norm(noised_lap - orig_lap, dim=-1)
            lap_norm = torch.norm(orig_lap, dim=-1) + _EPS
            lap_drifts.append(lap_diff / lap_norm)

        adj_drift = torch.cat(adj_drifts)
        lap_drift = torch.cat(lap_drifts)

        return DriftComparisonResult(
            noise_level=eps,
            adjacency=MeanStdMetric.from_tensor(adj_drift),
            laplacian=MeanStdMetric.from_tensor(lap_drift),
        )

    def compute_eigengap_delta(self, eps: float) -> GapDeltaComparisonResult:
        """Compute eigengap delta between original and noised graphs.

        Parameters
        ----------
        eps : float
            Noise level to compare.

        Returns
        -------
        GapDeltaComparisonResult
            Absolute and relative summaries for the adjacency spectral gap.
        """
        from .analyzer import compute_eigengap_delta

        abs_deltas: list[torch.Tensor] = []
        rel_deltas: list[torch.Tensor] = []

        for orig, noised in self._iter_paired_batches(eps):
            delta = compute_eigengap_delta(
                orig["eigenvalues_adj"], noised["eigenvalues_adj"]
            )
            abs_deltas.append(delta["absolute"])
            rel_deltas.append(delta["relative"])

        abs_delta = torch.cat(abs_deltas)
        rel_delta = torch.cat(rel_deltas)

        return GapDeltaComparisonResult(
            noise_level=eps,
            absolute=MeanStdMetric.from_tensor(abs_delta),
            relative=MeanStdMetric.from_tensor(rel_delta),
        )

    def compute_algebraic_connectivity_delta(
        self, eps: float
    ) -> GapDeltaComparisonResult:
        """Compute algebraic connectivity delta between original and noised graphs.

        Parameters
        ----------
        eps : float
            Noise level to compare.

        Returns
        -------
        GapDeltaComparisonResult
            Absolute and relative summaries for the Fiedler value shift.
        """
        from .analyzer import compute_algebraic_connectivity_delta

        abs_deltas: list[torch.Tensor] = []
        rel_deltas: list[torch.Tensor] = []

        for orig, noised in self._iter_paired_batches(eps):
            delta = compute_algebraic_connectivity_delta(
                orig["eigenvalues_lap"], noised["eigenvalues_lap"]
            )
            abs_deltas.append(delta["absolute"])
            rel_deltas.append(delta["relative"])

        abs_delta = torch.cat(abs_deltas)
        rel_delta = torch.cat(rel_deltas)

        return GapDeltaComparisonResult(
            noise_level=eps,
            absolute=MeanStdMetric.from_tensor(abs_delta),
            relative=MeanStdMetric.from_tensor(rel_delta),
        )

    def compute_subspace_distance(
        self, eps: float, k: int = 10
    ) -> SubspaceDistanceResult:
        """Compute subspace distance between original and noised eigenvectors.

        Parameters
        ----------
        eps : float
            Noise level to compare.
        k : int
            Number of top eigenvectors to compare.

        Returns
        -------
        SubspaceDistanceResult
            Projection-Frobenius distance summary for the chosen ``k``.
        """
        from tmgg.utils.spectral.spectral_deltas import (
            compute_subspace_distance_from_eigenvectors,
        )

        distances: list[torch.Tensor] = []

        for orig, noised in self._iter_paired_batches(eps):
            dist = compute_subspace_distance_from_eigenvectors(
                orig["eigenvectors_adj"], noised["eigenvectors_adj"], k
            )
            distances.append(dist)

        all_distances = torch.cat(distances)

        return SubspaceDistanceResult(
            noise_level=eps,
            k=k,
            distance=MeanStdMetric.from_tensor(all_distances),
        )

    def compute_subspace_deviation(
        self, eps: float, k: int = 10
    ) -> PrincipalAngleResult:
        """Compute subspace deviation via principal angles.

        Parameters
        ----------
        eps : float
            Noise level to compare.
        k : int
            Number of top eigenvectors to compare.

        Returns
        -------
        PrincipalAngleResult
            First-angle and mean-angle summaries for the chosen ``k``.

        Notes
        -----
        For batched computation, prefer ``compute_subspace_distance`` which uses
        projection Frobenius norm and processes entire batches efficiently.
        """
        from .analyzer import compute_principal_angles

        first_angles: list[float] = []
        mean_angles: list[float] = []

        for orig, noised in self._iter_paired_batches(eps):
            orig_vecs = orig["eigenvectors_adj"]
            noised_vecs = noised["eigenvectors_adj"]

            for i in range(orig_vecs.shape[0]):
                angles = compute_principal_angles(orig_vecs[i], noised_vecs[i], k)
                first_angles.append(angles[0].item())
                mean_angles.append(angles.mean().item())

        first_angles_t = torch.tensor(first_angles)
        mean_angles_t = torch.tensor(mean_angles)

        return PrincipalAngleResult(
            noise_level=eps,
            k=k,
            first_principal_angle=MeanStdMetric.from_tensor(first_angles_t),
            mean_principal_angle=MeanStdMetric.from_tensor(mean_angles_t),
        )

    def compute_procrustes_rotation(
        self, eps: float, k_values: list[int] | None = None
    ) -> ProcrustesRotationResult:
        """Compute Procrustes rotation metrics between original and noised eigenvectors.

        Parameters
        ----------
        eps : float
            Noise level to compare.
        k_values : list[int], optional
            List of k values for which to compute Procrustes rotation.
            Defaults to [1, 2, 4, 8, 16].

        Returns
        -------
        ProcrustesRotationResult
            Procrustes angle and residual summaries indexed by ``k``.
        """
        from tmgg.utils.spectral.subspace import (
            compute_procrustes_rotation,
        )

        if k_values is None:
            k_values = [1, 2, 4, 8, 16]

        angles_by_k: dict[int, list[float]] = {k: [] for k in k_values}
        residuals_by_k: dict[int, list[float]] = {k: [] for k in k_values}

        for orig, noised in self._iter_paired_batches(eps):
            orig_vecs = orig["eigenvectors_adj"]
            noised_vecs = noised["eigenvectors_adj"]

            for i in range(orig_vecs.shape[0]):
                for k in k_values:
                    if k > orig_vecs.shape[1]:
                        continue
                    result = compute_procrustes_rotation(
                        orig_vecs[i], noised_vecs[i], k
                    )
                    angles_by_k[k].append(result["angle"].item())
                    residuals_by_k[k].append(result["residual"].item())

        stats_by_k: dict[int, ProcrustesKStats] = {}
        for k in k_values:
            if angles_by_k[k]:
                angles_t = torch.tensor(angles_by_k[k])
                residuals_t = torch.tensor(residuals_by_k[k])
                stats_by_k[k] = ProcrustesKStats(
                    angle=MeanStdMetric.from_tensor(angles_t),
                    residual=MeanStdMetric.from_tensor(residuals_t),
                )

        return ProcrustesRotationResult(noise_level=eps, stats_by_k=stats_by_k)

    def compute_full_comparison(
        self, k: int = 10, procrustes_k_values: list[int] | None = None
    ) -> list[NoiseLevelComparisonResult]:
        """Compute all delta metrics across all noise levels in a single pass.

        For each noise level, iterates the paired batches once, computing
        eigenvalue drift, eigengap delta, algebraic connectivity delta,
        subspace distance, and Procrustes rotation in one pass over disk.

        Parameters
        ----------
        k : int
            Number of eigenvectors for subspace comparison.
        procrustes_k_values : list[int], optional
            List of k values for Procrustes rotation analysis.
            Defaults to [1, 2, 4, 8, 16].

        Returns
        -------
        list[NoiseLevelComparisonResult]
            One typed result object per noise level.
        """
        from tmgg.utils.spectral.spectral_deltas import (
            compute_subspace_distance_from_eigenvectors,
        )
        from tmgg.utils.spectral.subspace import (
            compute_procrustes_rotation,
        )

        from .analyzer import (
            compute_algebraic_connectivity_delta,
            compute_eigengap_delta,
        )

        if procrustes_k_values is None:
            procrustes_k_values = [1, 2, 4, 8, 16]

        results: list[NoiseLevelComparisonResult] = []
        for eps in self.get_noise_levels():
            logger.info(f"Comparing at eps={eps}")

            adj_drifts: list[torch.Tensor] = []
            lap_drifts: list[torch.Tensor] = []
            eigengap_rel: list[torch.Tensor] = []
            alg_conn_rel: list[torch.Tensor] = []
            sub_distances: list[torch.Tensor] = []
            angles_by_k: dict[int, list[float]] = {kv: [] for kv in procrustes_k_values}
            residuals_by_k: dict[int, list[float]] = {
                kv: [] for kv in procrustes_k_values
            }

            for orig, noised in self._iter_paired_batches(eps):
                orig_adj_eig = orig["eigenvalues_adj"]
                noised_adj_eig = noised["eigenvalues_adj"]
                orig_lap_eig = orig["eigenvalues_lap"]
                noised_lap_eig = noised["eigenvalues_lap"]
                orig_vecs = orig["eigenvectors_adj"]
                noised_vecs = noised["eigenvectors_adj"]

                # Eigenvalue drift
                adj_diff = torch.norm(noised_adj_eig - orig_adj_eig, dim=-1)
                adj_norm = torch.norm(orig_adj_eig, dim=-1) + _EPS
                adj_drifts.append(adj_diff / adj_norm)
                lap_diff = torch.norm(noised_lap_eig - orig_lap_eig, dim=-1)
                lap_norm = torch.norm(orig_lap_eig, dim=-1) + _EPS
                lap_drifts.append(lap_diff / lap_norm)

                # Eigengap delta (relative only for full comparison)
                eg_delta = compute_eigengap_delta(orig_adj_eig, noised_adj_eig)
                eigengap_rel.append(eg_delta["relative"])

                # Algebraic connectivity delta (relative only)
                ac_delta = compute_algebraic_connectivity_delta(
                    orig_lap_eig, noised_lap_eig
                )
                alg_conn_rel.append(ac_delta["relative"])

                # Subspace distance
                sub_distances.append(
                    compute_subspace_distance_from_eigenvectors(
                        orig_vecs, noised_vecs, k
                    )
                )

                # Procrustes rotation (per-graph within batch)
                for i in range(orig_vecs.shape[0]):
                    for kv in procrustes_k_values:
                        if kv > orig_vecs.shape[1]:
                            continue
                        pr = compute_procrustes_rotation(
                            orig_vecs[i], noised_vecs[i], kv
                        )
                        angles_by_k[kv].append(pr["angle"].item())
                        residuals_by_k[kv].append(pr["residual"].item())

            # Aggregate the one-pass tensors into typed summaries before anything
            # crosses the JSON boundary. That keeps the in-memory API structured
            # even though the persisted artifact is still a flat dict.
            adj_drift = torch.cat(adj_drifts)
            lap_drift = torch.cat(lap_drifts)
            eg_rel = torch.cat(eigengap_rel)
            ac_rel = torch.cat(alg_conn_rel)
            sub_dist = torch.cat(sub_distances)

            procrustes_by_k: dict[int, ProcrustesKStats] = {}
            for kv in procrustes_k_values:
                if angles_by_k[kv]:
                    angles_t = torch.tensor(angles_by_k[kv])
                    residuals_t = torch.tensor(residuals_by_k[kv])
                    procrustes_by_k[kv] = ProcrustesKStats(
                        angle=MeanStdMetric.from_tensor(angles_t),
                        residual=MeanStdMetric.from_tensor(residuals_t),
                    )

            results.append(
                NoiseLevelComparisonResult(
                    noise_level=eps,
                    k=k,
                    eigengap_delta_relative=MeanStdMetric.from_tensor(eg_rel),
                    algebraic_connectivity_delta_relative=MeanStdMetric.from_tensor(
                        ac_rel
                    ),
                    eigenvalue_drift_adjacency=MeanStdMetric.from_tensor(adj_drift),
                    eigenvalue_drift_laplacian=MeanStdMetric.from_tensor(lap_drift),
                    subspace_distance=MeanStdMetric.from_tensor(sub_dist),
                    procrustes_by_k=procrustes_by_k,
                )
            )

        return results

    def compute_covariance_evolution(
        self, matrix_type: str = "adjacency"
    ) -> CovarianceEvolutionResult:
        """
        Compute eigenvalue covariance at each noise level.

        Tracks how the covariance structure evolves as noise is added:
        higher noise typically decorrelates eigenvalues, reducing off-diagonal
        terms relative to the original.

        Parameters
        ----------
        matrix_type : str
            Either "adjacency" or "laplacian".

        Returns
        -------
        CovarianceEvolutionResult
            Original covariance plus one typed delta item per noise level.
        """
        from .analyzer import compute_covariance_summary, compute_eigenvalue_covariance

        key = "eigenvalues_adj" if matrix_type == "adjacency" else "eigenvalues_lap"

        # Load original eigenvalues
        orig_eigenvalues: list[torch.Tensor] = []
        for batch_path in iter_batches(self.original_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            orig_eigenvalues.append(tensors[key])

        orig_eig = torch.cat(orig_eigenvalues, dim=0)
        orig_cov = compute_eigenvalue_covariance(orig_eig)
        orig_summary = compute_covariance_summary(orig_cov)

        original_result = CovarianceResult(
            matrix_type=matrix_type,
            num_graphs=orig_eig.shape[0],
            num_eigenvalues=orig_eig.shape[1],
            covariance_matrix=orig_cov.tolist(),
            **orig_summary,
        )

        # Compute per noise level
        per_level_results: list[CovarianceEvolutionItem] = []

        for eps in self.get_noise_levels():
            noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

            noised_eigenvalues: list[torch.Tensor] = []
            for batch_path in iter_batches(noised_dir):
                tensors, _ = load_decomposition_batch(batch_path)
                noised_eigenvalues.append(tensors[key])

            noised_eig = torch.cat(noised_eigenvalues, dim=0)
            noised_cov = compute_eigenvalue_covariance(noised_eig)
            noised_summary = compute_covariance_summary(noised_cov)

            noised_result = CovarianceResult(
                matrix_type=matrix_type,
                num_graphs=noised_eig.shape[0],
                num_eigenvalues=noised_eig.shape[1],
                covariance_matrix=noised_cov.tolist(),
                **noised_summary,
            )

            # Compute delta metrics
            frob_delta = (
                torch.norm(noised_cov - orig_cov, p="fro").item()
                / orig_summary["frobenius_norm"]
            )
            trace_delta = (noised_summary["trace"] - orig_summary["trace"]) / (
                orig_summary["trace"] + _EPS
            )
            off_diag_delta = (
                noised_summary["off_diagonal_sum"] - orig_summary["off_diagonal_sum"]
            ) / (abs(orig_summary["off_diagonal_sum"]) + _EPS)

            per_level_results.append(
                CovarianceEvolutionItem(
                    noise_level=eps,
                    covariance=noised_result,
                    frobenius_delta_relative=frob_delta,
                    trace_delta_relative=trace_delta,
                    off_diagonal_delta_relative=off_diag_delta,
                )
            )

            logger.info(
                f"eps={eps}: frob_delta={frob_delta:.4f}, "
                f"off_diag_delta={off_diag_delta:.4f}"
            )

        return CovarianceEvolutionResult(
            matrix_type=matrix_type,
            original=original_result,
            per_noise_level=per_level_results,
        )

    def compute_improvement_gap_surrogate(
        self,
        eps: float,
        *,
        k: int,
        estimator: str = "knn",
        knn_neighbours: int = 10,
        num_bins: int = 4,
        target: SurrogateTarget = "matrix",
        conditioning: ConditioningFeatures = "top_k_eigenvalues",
        permute_features: bool = False,
        permutation_seed: int = 0,
    ) -> ImprovementGapResult:
        """Estimate the improvement-gap surrogate at one noise level.

        Loads ``B_k{k}`` projections pre-computed by the collector and the
        noisy adjacency eigenvalues, then hands both to
        :func:`~tmgg.experiments.eigenstructure_study.analyzer.estimate_improvement_gap`.

        Parameters
        ----------
        eps : float
            Noise level. Must match one of the ``eps_*`` subdirectories.
        k : int
            Top-*k* eigenspace dimension. A ``B_k{k}`` tensor must have
            been saved by the collector.
        estimator : str
            Either ``"knn"`` or ``"bin"``.
        knn_neighbours : int
            Neighbourhood size for the kNN estimator.
        num_bins : int
            Bin count for the binning estimator.
        target : str
            ``"matrix"`` — run the surrogate on the raw ``B_k{k}``
            matrices (literally eq. (18); meaningful once ``B`` is in a
            common frame, e.g. Fréchet-aligned).
            ``"invariants"`` — run the surrogate on frame-invariant
            summaries ``(tr B, ||B||_F², sorted eigvals(B))``. This is
            independent of the alignment convention and serves as a
            sanity cross-check.
        conditioning : str
            ``"top_k_eigenvalues"`` — condition on the full sorted top-*k*
            noisy eigenvalues (``k``-dim feature space).
            ``"spectral_gap"`` — condition on the scalar gap
            ``λ_1 − λ_2``. The ``"bin"`` estimator always uses this.
        permute_features : bool
            If ``True``, shuffle the conditioning before estimation to
            produce the permutation-null ĝ. A well-calibrated estimator
            should return ĝ near zero under permutation.
        permutation_seed : int
            Seed for the permutation RNG; kept explicit so null runs are
            reproducible.

        Returns
        -------
        ImprovementGapResult
            Surrogate estimate and the total between-graph variance.

        Raises
        ------
        KeyError
            If the noised batches do not contain the requested ``B_k{k}``.
        ValueError
            If ``estimator="bin"`` is combined with
            ``conditioning="top_k_eigenvalues"``; multi-dimensional
            quantile binning is not supported.
        """
        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"
        B_key = f"B_k{k}"

        B_list: list[torch.Tensor] = []
        eig_list: list[torch.Tensor] = []
        for batch_path in iter_batches(noised_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            if B_key not in tensors:
                raise KeyError(
                    f"Noised batch {batch_path.name} does not contain '{B_key}'. "
                    f"Re-run the collector with k={k} in surrogate_k_values."
                )
            B_list.append(tensors[B_key])
            eig_list.append(tensors["eigenvalues_adj"])

        B = torch.cat(B_list, dim=0)
        noisy_eig = torch.cat(eig_list, dim=0)
        # Top-k eigenvalues are the last k columns (ascending order).
        top_k_eig = noisy_eig[:, -k:]

        # Binning is always 1-D by design (multi-dim quantile binning is
        # exponential in k). Refuse the combination loudly.
        if estimator == "bin" and conditioning == "top_k_eigenvalues":
            raise ValueError(
                "estimator='bin' with conditioning='top_k_eigenvalues' is not "
                "supported: quantile binning in k dimensions is exponential."
            )

        if conditioning == "spectral_gap":
            features = top_k_eig[:, -1] - top_k_eig[:, -2]
        else:
            features = top_k_eig

        # Replace B with its frame-invariant summaries under the
        # invariant target — the reviewer-2 cross-check that survives the
        # frame-convention ambiguity because (tr, ||.||_F², eigvals) are
        # all orthogonal-conjugation invariant.
        surrogate_target = compute_B_invariants(B) if target == "invariants" else B

        g_hat, trace_cov_B, fve = estimate_improvement_gap(
            surrogate_target,
            features,
            estimator=estimator,
            knn_neighbours=knn_neighbours,
            num_bins=num_bins,
            permute_features=permute_features,
            permutation_seed=permutation_seed,
        )

        return ImprovementGapResult(
            noise_level=eps,
            k=k,
            estimator=estimator,
            num_graphs=B.shape[0],
            g_hat=g_hat,
            trace_cov_B=trace_cov_B,
            fve=fve,
            knn_neighbours=knn_neighbours if estimator == "knn" else None,
            num_bins=num_bins if estimator == "bin" else None,
            target=target,
            conditioning=conditioning,
            permuted=permute_features,
        )
