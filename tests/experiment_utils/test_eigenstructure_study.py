"""Tests for the eigenstructure study module.

Test rationale:
- Laplacian computation: verify L = D - A property and symmetry preservation
- Storage: verify round-trip save/load produces identical data
- Analysis functions: verify metrics against known graph structures
- Collector: integration test with small synthetic dataset
- Noised collector: verify noise application and decomposition computation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner
from omegaconf import OmegaConf

from tmgg.experiments.eigenstructure_study import (
    EigenstructureCollector,
    NoisedAnalysisComparator,
    NoisedEigenstructureCollector,
    SpectralAnalyzer,
    compute_algebraic_connectivity,
    compute_algebraic_connectivity_delta,
    compute_eigengap_delta,
    compute_eigenvalue_entropy,
    compute_eigenvector_coherence,
    compute_laplacian,
    compute_principal_angles,
    compute_spectral_gap,
    iter_batches,
    load_decomposition_batch,
    load_manifest,
    save_dataset_manifest,
    save_decomposition_batch,
)
from tmgg.experiments.eigenstructure_study.noised_collector import (
    CovarianceEvolutionResult,
    GapDeltaComparisonResult,
    NoiseLevelComparisonResult,
)
from tmgg.utils.spectral.spectral_deltas import (
    compute_eigenvalue_drift,
    compute_subspace_distance_from_eigenvectors,
)


class TestLaplacian:
    """Tests for Laplacian computation."""

    def test_laplacian_single_matrix(self) -> None:
        """Laplacian L = D - A should satisfy row sums = 0 for symmetric A."""
        # Simple 3-node path graph: 0 -- 1 -- 2
        A = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        L = compute_laplacian(A)

        # Row sums should be zero
        row_sums = L.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros(3))

        # Diagonal should be degrees
        degrees = A.sum(dim=1)
        assert torch.allclose(L.diag(), degrees)

        # Off-diagonal should be -A
        off_diag_L = L - torch.diag(L.diag())
        off_diag_A = -A
        assert torch.allclose(off_diag_L, off_diag_A)

    def test_laplacian_batched(self) -> None:
        """Batched Laplacian should compute correctly for each graph."""
        batch_size = 4
        n = 5
        # Random symmetric adjacency matrices
        A = torch.rand(batch_size, n, n)
        A = (A + A.transpose(-2, -1)) / 2
        A = (A > 0.5).float()  # Binarize
        A = A - torch.diag_embed(A.diagonal(dim1=-2, dim2=-1))  # Zero diagonal

        L = compute_laplacian(A)

        # Row sums should be zero for each graph
        row_sums = L.sum(dim=-1)
        assert torch.allclose(row_sums, torch.zeros(batch_size, n), atol=1e-6)

    def test_laplacian_symmetry(self) -> None:
        """Laplacian should be symmetric if adjacency is symmetric."""
        A = torch.rand(10, 10)
        A = (A + A.T) / 2
        L = compute_laplacian(A)

        assert torch.allclose(L, L.T)

    def test_normalized_laplacian_eigenvalues(self) -> None:
        """Normalized Laplacian eigenvalues should be in [0, 2]."""
        # Complete graph on 5 nodes
        n = 5
        A = torch.ones(n, n) - torch.eye(n)
        L_norm = compute_laplacian(A, normalized=True)

        eigenvalues, _ = torch.linalg.eigh(L_norm)

        assert eigenvalues.min() >= -1e-6  # Should be >= 0
        assert eigenvalues.max() <= 2 + 1e-6  # Should be <= 2

    def test_single_node_graph(self) -> None:
        """Single node graph (n=1) should produce 1x1 zero matrix.

        Rationale: A single isolated node has no edges, so both its degree
        and adjacency values are zero, yielding L = D - A = 0.
        """
        A = torch.zeros(1, 1)
        L = compute_laplacian(A)

        assert L.shape == (1, 1)
        assert L.item() == 0.0

    def test_empty_graph_all_zeros(self) -> None:
        """Graph with no edges should have L = 0 (diagonal also 0).

        Rationale: An empty graph (no edges) has zero degrees for all nodes,
        so the Laplacian is the zero matrix.
        """
        n = 5
        A = torch.zeros(n, n)
        L = compute_laplacian(A)

        assert torch.allclose(L, torch.zeros(n, n))

    def test_complete_graph_structure(self) -> None:
        """Complete graph K_n: diagonal = n-1, off-diagonal = -1.

        Rationale: In K_n, each node has degree n-1 (connected to all others).
        The Laplacian has diagonal entries n-1 and off-diagonal entries -1.
        """
        n = 6
        A = torch.ones(n, n) - torch.eye(n)
        L = compute_laplacian(A)

        # Diagonal should all be n-1
        expected_diag = torch.full((n,), float(n - 1))
        assert torch.allclose(L.diag(), expected_diag)

        # Off-diagonal should all be -1
        off_diag_mask = ~torch.eye(n, dtype=torch.bool)
        off_diag_values = L[off_diag_mask]
        assert torch.allclose(off_diag_values, torch.full_like(off_diag_values, -1.0))


class TestNormalizedLaplacian:
    """Tests for normalized Laplacian computation."""

    def test_isolated_node_eps_handling(self) -> None:
        """Isolated nodes (degree=0) should use eps to avoid div-by-zero.

        Rationale: The normalized Laplacian involves D^{-1/2}, which would be
        undefined for zero-degree nodes. The eps parameter prevents this.
        """
        n = 4
        # Graph with one isolated node (node 3)
        A = torch.zeros(n, n)
        A[0, 1] = A[1, 0] = 1.0
        A[1, 2] = A[2, 1] = 1.0

        # Should not raise due to eps parameter
        L_norm = compute_laplacian(A, normalized=True, eps=1e-8)

        # Result should be finite (no inf/nan)
        assert torch.all(torch.isfinite(L_norm))

    def test_batched_eigenvalue_bounds(self) -> None:
        """Batched normalized Laplacian eigenvalues should all be in [0, 2].

        Rationale: The normalized Laplacian of any simple graph has eigenvalues
        in [0, 2], regardless of batch dimension.
        """
        batch_size = 4
        n = 6

        # Create batch of random graphs
        A = torch.rand(batch_size, n, n)
        A = (A + A.transpose(-2, -1)) / 2
        A = (A > 0.5).float()
        A = A - torch.diag_embed(A.diagonal(dim1=-2, dim2=-1))

        L_norm = compute_laplacian(A, normalized=True)

        for i in range(batch_size):
            eigenvalues, _ = torch.linalg.eigh(L_norm[i])
            assert eigenvalues.min() >= -1e-5, f"Batch {i}: eigenvalue < 0"
            assert eigenvalues.max() <= 2 + 1e-5, f"Batch {i}: eigenvalue > 2"

    def test_symmetry_preserved(self) -> None:
        """Normalized Laplacian should be symmetric for symmetric input.

        Rationale: Since L_sym = I - D^{-1/2} A D^{-1/2}, and A is symmetric,
        the result is also symmetric (conjugation preserves symmetry).
        """
        n = 8
        A = torch.rand(n, n)
        A = (A + A.T) / 2
        A = (A > 0.5).float()
        A.fill_diagonal_(0)

        L_norm = compute_laplacian(A, normalized=True)

        assert torch.allclose(L_norm, L_norm.T, atol=1e-6)


class TestStorage:
    """Tests for safetensors storage utilities."""

    def test_save_load_roundtrip(self) -> None:
        """Save and load should produce identical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            batch_size = 3
            n = 4

            # Create test data
            eigenvalues_adj = torch.randn(batch_size, n)
            eigenvectors_adj = torch.randn(batch_size, n, n)
            eigenvalues_lap = torch.randn(batch_size, n)
            eigenvectors_lap = torch.randn(batch_size, n, n)
            adjacency = torch.randn(batch_size, n, n)
            metadata = [{"graph_index": i, "node_count": n} for i in range(batch_size)]

            # Save
            save_decomposition_batch(
                output_dir,
                batch_index=0,
                eigenvalues_adj=eigenvalues_adj,
                eigenvectors_adj=eigenvectors_adj,
                eigenvalues_lap=eigenvalues_lap,
                eigenvectors_lap=eigenvectors_lap,
                adjacency_matrices=adjacency,
                metadata_list=metadata,
            )

            # Load
            batch_path = output_dir / "batch_000000.safetensors"
            tensors, loaded_metadata = load_decomposition_batch(batch_path)

            # Verify
            assert torch.allclose(tensors["eigenvalues_adj"], eigenvalues_adj)
            assert torch.allclose(tensors["eigenvectors_adj"], eigenvectors_adj)
            assert torch.allclose(tensors["eigenvalues_lap"], eigenvalues_lap)
            assert torch.allclose(tensors["eigenvectors_lap"], eigenvectors_lap)
            assert torch.allclose(tensors["adjacency"], adjacency)
            assert loaded_metadata == metadata

    def test_manifest_save_load(self) -> None:
        """Manifest should save and load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = {"num_nodes": 10, "p": 0.5}

            save_dataset_manifest(
                output_dir,
                dataset_name="test_dataset",
                dataset_config=config,
                num_graphs=100,
                num_batches=5,
                seed=42,
            )

            manifest = load_manifest(output_dir)

            assert manifest["dataset_name"] == "test_dataset"
            assert manifest["dataset_config"] == config
            assert manifest["num_graphs"] == 100
            assert manifest["num_batches"] == 5
            assert manifest["seed"] == 42

    def test_iter_batches(self) -> None:
        """iter_batches should return sorted batch paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create dummy batch files
            for i in [2, 0, 1]:  # Out of order
                (output_dir / f"batch_{i:06d}.safetensors").touch()
                (output_dir / f"batch_{i:06d}_metadata.json").write_text("[]")

            batches = iter_batches(output_dir)

            assert len(batches) == 3
            assert batches[0].name == "batch_000000.safetensors"
            assert batches[1].name == "batch_000001.safetensors"
            assert batches[2].name == "batch_000002.safetensors"


class TestAnalysisFunctions:
    """Tests for spectral analysis functions."""

    def test_spectral_gap_complete_graph(self) -> None:
        """Complete graph has spectral gap = n (largest eigenvalue is n-1)."""
        n = 5
        A = torch.ones(n, n) - torch.eye(n)
        eigenvalues, _ = torch.linalg.eigh(A)

        # Batch it
        eigenvalues_batch = eigenvalues.unsqueeze(0)
        gap = compute_spectral_gap(eigenvalues_batch)

        # Complete graph: eigenvalue n-1 with multiplicity 1, eigenvalue -1 with multiplicity n-1
        # Gap should be (n-1) - (-1) = n
        assert torch.isclose(gap[0], torch.tensor(float(n)), atol=1e-5)

    def test_algebraic_connectivity_path_graph(self) -> None:
        """Path graph has small algebraic connectivity."""
        # Path graph: 0 -- 1 -- 2 -- 3 -- 4
        n = 5
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0

        L = compute_laplacian(A)
        eigenvalues, _ = torch.linalg.eigh(L)
        eigenvalues_batch = eigenvalues.unsqueeze(0)

        alg_conn = compute_algebraic_connectivity(eigenvalues_batch)

        # Path graph has small but positive algebraic connectivity
        assert alg_conn[0] > 0
        assert alg_conn[0] < 1.0  # Known to be small for path graphs

    def test_algebraic_connectivity_disconnected(self) -> None:
        """Disconnected graph has algebraic connectivity = 0."""
        # Two isolated nodes
        A = torch.zeros(4, 4)
        A[0, 1] = A[1, 0] = 1.0  # Edge between 0-1
        # Nodes 2-3 are isolated from 0-1

        L = compute_laplacian(A)
        eigenvalues, _ = torch.linalg.eigh(L)
        eigenvalues_batch = eigenvalues.unsqueeze(0)

        alg_conn = compute_algebraic_connectivity(eigenvalues_batch)

        # Disconnected graph: second smallest eigenvalue is 0
        assert torch.isclose(alg_conn[0], torch.tensor(0.0), atol=1e-6)

    def test_eigenvector_coherence_range(self) -> None:
        """Coherence should be between 1/n (delocalized) and 1 (localized)."""
        n = 10
        batch_size = 5
        # Random orthonormal eigenvectors
        V = torch.linalg.qr(torch.randn(batch_size, n, n))[0]

        coherence = compute_eigenvector_coherence(V)

        # Coherence is max squared component, should be in [1/n, 1]
        assert (coherence >= 1.0 / n - 1e-6).all()
        assert (coherence <= 1.0 + 1e-6).all()

    def test_eigenvalue_entropy_uniform(self) -> None:
        """Uniform eigenvalues should have high entropy."""
        n = 10
        uniform_eigs = torch.ones(1, n)
        peaked_eigs = torch.zeros(1, n)
        peaked_eigs[0, 0] = 1.0

        uniform_entropy = compute_eigenvalue_entropy(uniform_eigs)
        peaked_entropy = compute_eigenvalue_entropy(peaked_eigs)

        # Uniform distribution has higher entropy
        assert uniform_entropy > peaked_entropy

    def test_principal_angles_identical_subspaces(self) -> None:
        """Identical subspaces should have near-zero principal angles."""
        n = 10
        V = torch.linalg.qr(torch.randn(n, n))[0]

        angles = compute_principal_angles(V, V, k=3)

        # Relaxed tolerance due to numerical precision in SVD
        assert torch.allclose(angles, torch.zeros(3), atol=1e-3)

    def test_principal_angles_orthogonal_subspaces(self) -> None:
        """Orthogonal subspaces should have angles close to pi/2."""
        n = 10
        k = 3
        V = torch.eye(n)

        # Take first k columns vs last k columns
        V1 = V.clone()
        V2 = torch.roll(V, shifts=k, dims=1)  # Shift columns

        angles = compute_principal_angles(V1, V2, k=k)

        # Orthogonal subspaces have angle pi/2
        assert (angles > 1.0).all()  # Greater than ~60 degrees


class TestCollector:
    """Integration tests for EigenstructureCollector."""

    def test_collect_synthetic_er(self) -> None:
        """Collector should process ER graphs and save decompositions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 10, "num_graphs": 5, "p": 0.3},
                output_dir=output_dir,
                batch_size=2,
                seed=42,
            )
            collector.collect()

            # Verify manifest
            manifest = load_manifest(output_dir)
            assert manifest["num_graphs"] == 5
            assert manifest["num_batches"] == 3  # ceil(5/2)

            # Verify batches exist
            batches = iter_batches(output_dir)
            assert len(batches) == 3

            # Load first batch and verify structure
            tensors, metadata = load_decomposition_batch(batches[0])
            assert tensors["eigenvalues_adj"].shape == (2, 10)
            assert tensors["eigenvectors_adj"].shape == (2, 10, 10)
            assert tensors["eigenvalues_lap"].shape == (2, 10)
            assert tensors["eigenvectors_lap"].shape == (2, 10, 10)
            assert len(metadata) == 2


class TestSpectralAnalyzer:
    """Tests for SpectralAnalyzer."""

    def test_analyze_collected_data(self) -> None:
        """Analyzer should compute statistics from collected data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # First collect some data
            collector = EigenstructureCollector(
                dataset_name="regular",
                dataset_config={"num_nodes": 8, "num_graphs": 10, "d": 3},
                output_dir=output_dir,
                batch_size=5,
                seed=42,
            )
            collector.collect()

            # Now analyze
            analyzer = SpectralAnalyzer(output_dir)
            result = analyzer.analyze()

            assert result.num_graphs == 10
            assert result.spectral_gap_mean > 0
            assert result.algebraic_connectivity_mean > 0
            assert result.coherence_mean > 0
            assert result.effective_rank_adj_mean > 0

    def test_execute_analyze_uses_save_results(self) -> None:
        """Hydra analyze path should delegate main JSON writing to save_results.

        Rationale
        ---------
        ``SpectralAnalyzer.save_results`` exists to own the canonical
        ``analysis.json`` write path. Both CLI and Hydra execution used to
        bypass it and duplicate the same ``json.dump(asdict(...))`` block.
        This test keeps the execute path routed through the analyzer helper.
        """
        from tmgg.experiments.eigenstructure_study import execute as execute_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            collector = EigenstructureCollector(
                dataset_name="regular",
                dataset_config={"num_nodes": 8, "num_graphs": 6, "d": 3},
                output_dir=output_dir / "original",
                batch_size=3,
                seed=42,
            )
            collector.collect()

            save_calls: list[tuple[str, Path]] = []
            original_save = SpectralAnalyzer.save_results

            def wrapped_save(
                self, result, output_path: Path
            ):  # pragma: no cover - exercised via _run_analyze
                save_calls.append((result.dataset_name, Path(output_path)))
                return original_save(self, result, output_path)

            SpectralAnalyzer.save_results = wrapped_save  # type: ignore[method-assign]
            try:
                cfg = OmegaConf.create(
                    {
                        "analysis": {
                            "subspace_k": 0,
                            "compute_covariance": False,
                            "matrix_type": "adjacency",
                        }
                    }
                )
                result = execute_module._run_analyze(cfg, output_dir)
            finally:
                SpectralAnalyzer.save_results = original_save  # type: ignore[method-assign]

            assert result["status"] == "completed"
            assert save_calls == [("regular", output_dir / "analysis")]
            assert (output_dir / "analysis" / "analysis.json").exists()

    def test_cli_analyze_uses_save_results(self) -> None:
        """CLI analyze command should use save_results for analysis.json too.

        The CLI and Hydra execution path should not drift on the main analysis
        artifact. This test invokes the Click command directly and verifies
        the CLI routes its primary JSON write through ``save_results``.
        """
        from tmgg.experiments.eigenstructure_study import cli as cli_module

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "analysis"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 6, "p": 0.3},
                output_dir=input_dir,
                batch_size=3,
                seed=42,
            )
            collector.collect()

            save_calls: list[Path] = []
            original_save = SpectralAnalyzer.save_results

            def wrapped_save(
                self, result, output_path: Path
            ):  # pragma: no cover - exercised via Click command
                save_calls.append(Path(output_path))
                return original_save(self, result, output_path)

            SpectralAnalyzer.save_results = wrapped_save  # type: ignore[method-assign]
            try:
                runner = CliRunner()
                invocation = runner.invoke(
                    cli_module.main,
                    [
                        "analyze",
                        "--input-dir",
                        str(input_dir),
                        "--output-dir",
                        str(output_dir),
                        "--subspace-k",
                        "0",
                    ],
                )
            finally:
                SpectralAnalyzer.save_results = original_save  # type: ignore[method-assign]

            assert invocation.exit_code == 0, invocation.output
            assert save_calls == [output_dir]
            assert (output_dir / "analysis.json").exists()


class TestNoisedCollector:
    """Tests for NoisedEigenstructureCollector."""

    def test_noised_collection_gaussian(self) -> None:
        """Noised collector should apply Gaussian noise and save decompositions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            # Collect original data
            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 4, "p": 0.3},
                output_dir=original_dir,
                batch_size=2,
                seed=42,
            )
            collector.collect()

            # Collect noised data
            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1, 0.2],
                seed=42,
            )
            noised_collector.collect()

            # Verify subdirectories exist
            assert (noised_dir / "eps_0.1000").exists()
            assert (noised_dir / "eps_0.2000").exists()

            # Verify manifests
            manifest_01 = load_manifest(noised_dir / "eps_0.1000")
            assert manifest_01["dataset_config"]["noise_level"] == 0.1
            assert manifest_01["dataset_config"]["noise_type"] == "gaussian"

    def test_noised_collection_digress(self) -> None:
        """Noised collector should apply Digress noise correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            collector = EigenstructureCollector(
                dataset_name="regular",
                dataset_config={"num_nodes": 6, "num_graphs": 3, "d": 2},
                output_dir=original_dir,
                batch_size=3,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="digress",
                noise_levels=[0.3],
                seed=42,
            )
            noised_collector.collect()

            # Verify noised adjacencies are different from original.
            # Uses eps=0.3 (not 0.05) to make zero-flip probability negligible:
            # P(no flips) = (0.7)^(3 graphs * 15 edges) ≈ 4e-7.
            orig_tensors, _ = load_decomposition_batch(
                list(iter_batches(original_dir))[0]
            )
            noised_tensors, _ = load_decomposition_batch(
                list(iter_batches(noised_dir / "eps_0.3000"))[0]
            )

            # Noised adjacency should differ from original
            assert not torch.allclose(
                orig_tensors["adjacency"], noised_tensors["adjacency"]
            )


class TestNoisedEigenstructureCollectorValidation:
    """Tests for NoisedEigenstructureCollector input validation."""

    def test_rotation_k_required_for_rotation_noise(self) -> None:
        """Should raise ValueError if noise_type='rotation' without rotation_k.

        Rationale: Rotation noise requires a skew matrix dimension parameter.
        Omitting it should fail early with a clear error message.
        """
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            # Create minimal original data
            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 6, "num_graphs": 2, "p": 0.3},
                output_dir=original_dir,
                batch_size=2,
                seed=42,
            )
            collector.collect()

            # Should raise ValueError when rotation_k is None for rotation noise
            with pytest.raises(ValueError, match="rotation_k is required"):
                NoisedEigenstructureCollector(
                    input_dir=original_dir,
                    output_dir=noised_dir,
                    noise_type="rotation",
                    noise_levels=[0.1],
                    rotation_k=None,  # Missing required parameter
                    seed=42,
                )

    def test_output_directory_structure(self) -> None:
        """Should create eps_X.XXXX/ subdirectories for each noise level.

        Rationale: Output organization uses noise level as subdirectory name
        in a specific format (4 decimal places) for consistency.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 6, "num_graphs": 2, "p": 0.3},
                output_dir=original_dir,
                batch_size=2,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05, 0.15, 0.25],
                seed=42,
            )
            noised_collector.collect()

            # Verify directory structure
            assert (noised_dir / "eps_0.0500").is_dir()
            assert (noised_dir / "eps_0.1500").is_dir()
            assert (noised_dir / "eps_0.2500").is_dir()

    def test_manifest_contains_noise_metadata(self) -> None:
        """Each noise level manifest should include noise_type and noise_level.

        Rationale: Manifest metadata should capture the noise configuration
        for reproducibility and analysis.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 6, "num_graphs": 2, "p": 0.3},
                output_dir=original_dir,
                batch_size=2,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="digress",
                noise_levels=[0.1],
                seed=42,
            )
            noised_collector.collect()

            manifest = load_manifest(noised_dir / "eps_0.1000")
            assert manifest["dataset_config"]["noise_type"] == "digress"
            assert manifest["dataset_config"]["noise_level"] == 0.1
            assert "original_dataset" in manifest["dataset_config"]


class TestNoisedAnalysisComparator:
    """Tests for NoisedAnalysisComparator."""

    def test_comparison_workflow(self) -> None:
        """Comparator should compute drift and deviation statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            # Setup data
            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 6, "p": 0.3},
                output_dir=original_dir,
                batch_size=3,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.05, 0.1],
                seed=42,
            )
            noised_collector.collect()

            # Compare
            comparator = NoisedAnalysisComparator(original_dir, noised_dir)

            assert set(comparator.get_noise_levels()) == {0.05, 0.1}

            drift = comparator.compute_eigenvalue_drift(0.1)
            assert drift.adjacency.mean >= 0
            assert drift.laplacian.mean >= 0

            subspace = comparator.compute_subspace_deviation(0.1, k=3)
            assert subspace.first_principal_angle.mean >= 0

    def test_get_noise_levels_parsing(self) -> None:
        """Should parse eps_0.0100, eps_0.1000 format correctly.

        Rationale: Directory names encode noise levels; parsing must extract
        the float value accurately.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            noised_dir = Path(tmpdir)

            # Create directories with specific names
            (noised_dir / "eps_0.0100").mkdir()
            (noised_dir / "eps_0.0500").mkdir()
            (noised_dir / "eps_0.1000").mkdir()

            comparator = NoisedAnalysisComparator(
                original_dir=Path("/dummy"),  # Not used for this test
                noised_base_dir=noised_dir,
            )

            levels = comparator.get_noise_levels()

            assert 0.01 in levels
            assert 0.05 in levels
            assert 0.10 in levels

    def test_get_noise_levels_sorted(self) -> None:
        """Should return noise levels in ascending order.

        Rationale: Sorted output ensures consistent iteration and display.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            noised_dir = Path(tmpdir)

            # Create directories out of order
            (noised_dir / "eps_0.3000").mkdir()
            (noised_dir / "eps_0.0500").mkdir()
            (noised_dir / "eps_0.1500").mkdir()

            comparator = NoisedAnalysisComparator(
                original_dir=Path("/dummy"),
                noised_base_dir=noised_dir,
            )

            levels = comparator.get_noise_levels()

            assert levels == [0.05, 0.15, 0.30]

    def test_get_noise_levels_skips_invalid_dirs(self) -> None:
        """Should skip directories not matching eps_* pattern.

        Rationale: Only eps_X.XXXX directories should be recognized;
        other directories (metadata, temp files) should be ignored.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            noised_dir = Path(tmpdir)

            # Create valid and invalid directories
            (noised_dir / "eps_0.1000").mkdir()
            (noised_dir / "eps_0.2000").mkdir()
            (noised_dir / "metadata").mkdir()  # Invalid
            (noised_dir / "eps_invalid").mkdir()  # Invalid (not a float)
            (noised_dir / "other_dir").mkdir()  # Invalid

            comparator = NoisedAnalysisComparator(
                original_dir=Path("/dummy"),
                noised_base_dir=noised_dir,
            )

            levels = comparator.get_noise_levels()

            assert len(levels) == 2
            assert 0.10 in levels
            assert 0.20 in levels


class TestCLI:
    """Tests for CLI commands (smoke tests)."""

    def test_cli_help(self) -> None:
        """CLI should show help without errors."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "collect" in result.output
        assert "analyze" in result.output
        assert "noised" in result.output

    def test_cli_collect_subcommand_help(self) -> None:
        """Collect subcommand should show help."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["collect", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--dataset-config" in result.output


class TestDeltaFunctions:
    """Tests for spectral delta comparison functions.

    Test rationale:
    - Delta functions measure spectral changes between clean and noisy/denoised graphs
    - Identical graphs should yield zero delta (eigengap_delta, eigenvalue_drift)
    - Different graphs should yield non-zero delta
    - Subspace distance should be bounded by 2*sqrt(k) due to projection matrix properties
    """

    def test_eigengap_delta_identical_graphs(self) -> None:
        """Identical eigenvalues should have zero eigengap delta."""
        # compute_eigengap_delta takes eigenvalues, not adjacency matrices
        batch_size = 4
        n = 10
        # Create random sorted eigenvalues
        eigenvalues = torch.sort(torch.randn(batch_size, n), dim=-1).values

        delta = compute_eigengap_delta(eigenvalues, eigenvalues)

        assert torch.allclose(delta["absolute"], torch.zeros(batch_size), atol=1e-6)
        assert torch.allclose(delta["relative"], torch.zeros(batch_size), atol=1e-6)

    def test_eigengap_delta_different_eigenvalues(self) -> None:
        """Different eigenvalues should have non-zero eigengap delta."""
        batch_size = 4
        n = 10

        # Eigenvalues 1: large spectral gap (last two values far apart)
        eig1 = torch.sort(torch.randn(batch_size, n), dim=-1).values
        eig1[:, -1] = eig1[:, -2] + 2.0  # Large gap

        # Eigenvalues 2: small spectral gap
        eig2 = torch.sort(torch.randn(batch_size, n), dim=-1).values
        eig2[:, -1] = eig2[:, -2] + 0.1  # Small gap

        delta = compute_eigengap_delta(eig1, eig2)

        # Delta should be non-zero (gaps are different)
        assert delta["absolute"].abs().mean() > 0.1

    def test_algebraic_connectivity_delta_identical(self) -> None:
        """Identical Laplacian eigenvalues should yield zero delta."""
        batch_size = 3
        n = 8

        # Create path graphs (known structure)
        L_eig = torch.zeros(batch_size, n)
        for i in range(n):
            L_eig[:, i] = 2 * (1 - np.cos(np.pi * i / n))

        delta = compute_algebraic_connectivity_delta(L_eig, L_eig)

        assert torch.allclose(delta["absolute"], torch.zeros(batch_size), atol=1e-6)

    def test_eigenvalue_drift_identical(self) -> None:
        """Identical eigenvalues should yield zero drift."""
        batch_size = 5
        n = 12
        eigenvalues = torch.randn(batch_size, n)

        drift = compute_eigenvalue_drift(eigenvalues, eigenvalues)

        assert torch.allclose(drift, torch.zeros(batch_size), atol=1e-6)

    def test_eigenvalue_drift_scaled(self) -> None:
        """Scaled eigenvalues should yield expected drift."""
        batch_size = 3
        n = 10
        eigenvalues = torch.ones(batch_size, n)
        scaled = eigenvalues * 2  # Double the values

        drift = compute_eigenvalue_drift(eigenvalues, scaled)

        # Drift = ||2λ - λ|| / ||λ|| = ||λ|| / ||λ|| = 1.0
        expected = torch.ones(batch_size)
        assert torch.allclose(drift, expected, atol=1e-5)

    def test_subspace_distance_identical(self) -> None:
        """Identical eigenvectors should yield zero subspace distance."""
        batch_size = 4
        n = 8
        k = 3

        # Create random orthonormal eigenvectors
        V = torch.randn(batch_size, n, n)
        V, _ = torch.linalg.qr(V)  # Orthonormalize

        distance = compute_subspace_distance_from_eigenvectors(V, V, k)

        assert torch.allclose(distance, torch.zeros(batch_size), atol=1e-5)

    def test_subspace_distance_bounded(self) -> None:
        """Subspace distance should be bounded by Frobenius norm of projection difference."""
        batch_size = 4
        n = 10
        k = 3

        # Create two random sets of eigenvectors
        V1 = torch.randn(batch_size, n, n)
        V1, _ = torch.linalg.qr(V1)
        V2 = torch.randn(batch_size, n, n)
        V2, _ = torch.linalg.qr(V2)

        distance = compute_subspace_distance_from_eigenvectors(V1, V2, k)

        # Max distance is 2*sqrt(k) when subspaces are orthogonal
        # (since ||P||_F = sqrt(trace(P)) = sqrt(k) for rank-k projection)
        max_distance = 2 * np.sqrt(k)
        assert (distance <= max_distance + 0.01).all()


class TestSpectralDeltasModule:
    """Tests for the spectral_deltas module used in training integration.

    Test rationale:
    - compute_spectral_deltas should compute all four metrics efficiently
    - Results should match individual analyzer.py functions
    - Should handle both 2D (single graph) and 3D (batch) inputs
    """

    def test_compute_spectral_deltas_all_metrics(self) -> None:
        """compute_spectral_deltas should return all four metrics."""
        from tmgg.utils.spectral.spectral_deltas import (
            compute_spectral_deltas,
        )

        batch_size = 4
        n = 10
        k = 3

        # Create two sets of adjacency matrices
        A1 = torch.rand(batch_size, n, n)
        A1 = (A1 + A1.transpose(-2, -1)) / 2
        A1 = (A1 > 0.5).float()
        A1 = A1 - torch.diag_embed(A1.diagonal(dim1=-2, dim2=-1))

        A2 = torch.rand(batch_size, n, n)
        A2 = (A2 + A2.transpose(-2, -1)) / 2
        A2 = (A2 > 0.5).float()
        A2 = A2 - torch.diag_embed(A2.diagonal(dim1=-2, dim2=-1))

        deltas = compute_spectral_deltas(A1, A2, k=k)

        # Check all metrics are present
        assert "eigengap_delta" in deltas
        assert "alg_conn_delta" in deltas
        assert "eigenvalue_drift" in deltas
        assert "subspace_distance" in deltas

        # Check shapes
        for key, value in deltas.items():
            assert value.shape == (batch_size,), f"{key} has wrong shape"

    def test_compute_spectral_deltas_identical_graphs(self) -> None:
        """Identical graphs should yield near-zero deltas."""
        from tmgg.utils.spectral.spectral_deltas import (
            compute_spectral_deltas,
        )

        batch_size = 3
        n = 8
        k = 2

        A = torch.rand(batch_size, n, n)
        A = (A + A.transpose(-2, -1)) / 2
        A = (A > 0.5).float()
        A = A - torch.diag_embed(A.diagonal(dim1=-2, dim2=-1))

        deltas = compute_spectral_deltas(A, A, k=k)

        # All deltas should be near zero for identical graphs
        assert torch.allclose(
            deltas["eigengap_delta"], torch.zeros(batch_size), atol=1e-5
        )
        assert torch.allclose(
            deltas["alg_conn_delta"], torch.zeros(batch_size), atol=1e-5
        )
        assert torch.allclose(
            deltas["eigenvalue_drift"], torch.zeros(batch_size), atol=1e-5
        )
        assert torch.allclose(
            deltas["subspace_distance"], torch.zeros(batch_size), atol=1e-5
        )

    def test_compute_spectral_deltas_2d_input(self) -> None:
        """Should handle 2D input (single graph) correctly."""
        from tmgg.utils.spectral.spectral_deltas import (
            compute_spectral_deltas,
        )

        n = 8
        k = 2

        A1 = torch.rand(n, n)
        A1 = (A1 + A1.T) / 2
        A1 = (A1 > 0.5).float()
        A1.fill_diagonal_(0)

        A2 = torch.rand(n, n)
        A2 = (A2 + A2.T) / 2
        A2 = (A2 > 0.5).float()
        A2.fill_diagonal_(0)

        deltas = compute_spectral_deltas(A1, A2, k=k)

        # Should return 1D tensors of shape (1,) for single graph input
        for key, value in deltas.items():
            assert value.shape == (
                1,
            ), f"{key} should have shape (1,) for 2D input, got {value.shape}"

    def test_compute_spectral_deltas_reduce_mean(self) -> None:
        """reduce='mean' should return float means."""
        from tmgg.utils.spectral.spectral_deltas import (
            compute_spectral_deltas,
        )

        batch_size = 4
        n = 8
        k = 2

        A1 = torch.rand(batch_size, n, n)
        A1 = (A1 + A1.transpose(-2, -1)) / 2
        A1 = (A1 > 0.5).float()
        A1 = A1 - torch.diag_embed(A1.diagonal(dim1=-2, dim2=-1))

        A2 = torch.rand(batch_size, n, n)
        A2 = (A2 + A2.transpose(-2, -1)) / 2
        A2 = (A2 > 0.5).float()
        A2 = A2 - torch.diag_embed(A2.diagonal(dim1=-2, dim2=-1))

        summary = compute_spectral_deltas(A1, A2, k=k, reduce="mean")

        # Should return dict of floats
        assert isinstance(summary, dict)
        for key, value in summary.items():
            assert isinstance(value, float), f"{key} should be float"


class TestNoisedAnalysisComparatorDeltaMethods:
    """Tests for new delta methods in NoisedAnalysisComparator.

    Test rationale:
    - New delta methods should compute eigengap and algebraic connectivity deltas
    - Full comparison should include all four metrics
    """

    def test_compute_eigengap_delta(self) -> None:
        """compute_eigengap_delta should return typed absolute and relative stats.

        Rationale
        ---------
        The noised comparison layer now uses typed result objects instead of
        ad-hoc dicts. This test locks the result shape at the Python API level
        and separately verifies that the legacy JSON field names remain stable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            # Setup data
            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 6, "p": 0.3},
                output_dir=original_dir,
                batch_size=3,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1],
                seed=42,
            )
            noised_collector.collect()

            comparator = NoisedAnalysisComparator(original_dir, noised_dir)
            result = comparator.compute_eigengap_delta(0.1)

            assert isinstance(result, GapDeltaComparisonResult)
            assert result.noise_level == 0.1
            assert torch.isfinite(torch.tensor(result.absolute.mean))
            assert torch.isfinite(torch.tensor(result.relative.mean))

            payload = result.to_json_dict("eigengap_delta")
            assert "eigengap_delta_abs_mean" in payload
            assert "eigengap_delta_rel_mean" in payload
            assert "eigengap_delta_rel_std" in payload

    def test_compute_algebraic_connectivity_delta(self) -> None:
        """compute_algebraic_connectivity_delta should return typed stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 6, "p": 0.3},
                output_dir=original_dir,
                batch_size=3,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1],
                seed=42,
            )
            noised_collector.collect()

            comparator = NoisedAnalysisComparator(original_dir, noised_dir)
            result = comparator.compute_algebraic_connectivity_delta(0.1)

            assert isinstance(result, GapDeltaComparisonResult)
            assert result.noise_level == 0.1
            assert torch.isfinite(torch.tensor(result.absolute.mean))
            assert torch.isfinite(torch.tensor(result.relative.mean))

            payload = result.to_json_dict("alg_conn_delta")
            assert "alg_conn_delta_abs_mean" in payload
            assert "alg_conn_delta_rel_mean" in payload
            assert "alg_conn_delta_rel_std" in payload

    def test_compute_full_comparison_all_metrics(self) -> None:
        """compute_full_comparison should return typed results with stable JSON keys.

        Rationale
        ---------
        The one-pass comparison is the public aggregation API for study output.
        We want typed fields for internal callers while preserving the current
        `comparison.json` wire shape written by the CLI and Hydra entry points.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 6, "p": 0.3},
                output_dir=original_dir,
                batch_size=3,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1],
                seed=42,
            )
            noised_collector.collect()

            comparator = NoisedAnalysisComparator(original_dir, noised_dir)
            results = comparator.compute_full_comparison(k=3)

            assert len(results) == 1
            r = results[0]
            assert isinstance(r, NoiseLevelComparisonResult)
            assert r.noise_level == 0.1
            assert r.k == 3

            payload = r.to_json_dict()
            assert "eigengap_delta_rel_mean" in payload
            assert "alg_conn_delta_rel_mean" in payload
            assert "eigenvalue_drift_adj_mean" in payload
            assert "subspace_distance_mean" in payload
            assert "procrustes_angle_k1_mean" in payload


class TestCovarianceComputation:
    """Tests for eigenvalue covariance computation.

    Test rationale:
    - Covariance matrix should be k×k for k eigenvalues
    - Covariance matrix should be symmetric (by definition)
    - IID eigenvalues should have near-diagonal covariance (no cross-correlation)
    - Correlated eigenvalues should show off-diagonal terms
    - Noise should typically decorrelate eigenvalues (reduce off-diagonal terms)
    """

    def test_covariance_shape(self) -> None:
        """Covariance matrix should be k×k for k eigenvalues."""
        from tmgg.experiments.eigenstructure_study.analyzer import (
            compute_eigenvalue_covariance,
        )

        N = 50
        k = 10
        eigenvalues = torch.randn(N, k)

        cov = compute_eigenvalue_covariance(eigenvalues)

        assert cov.shape == (k, k)

    def test_covariance_symmetry(self) -> None:
        """Covariance matrix should be symmetric."""
        from tmgg.experiments.eigenstructure_study.analyzer import (
            compute_eigenvalue_covariance,
        )

        N = 100
        k = 8
        eigenvalues = torch.randn(N, k)

        cov = compute_eigenvalue_covariance(eigenvalues)

        assert torch.allclose(cov, cov.T, atol=1e-6)

    def test_covariance_iid_is_diagonal(self) -> None:
        """IID eigenvalues should have near-diagonal covariance.

        When eigenvalue positions are independent, cross-correlation should be
        near zero. The covariance matrix should be approximately diagonal.
        """
        from tmgg.experiments.eigenstructure_study.analyzer import (
            compute_eigenvalue_covariance,
        )

        torch.manual_seed(42)
        N = 1000  # Large sample for stable statistics
        k = 5

        # Generate IID eigenvalues: each position independent
        eigenvalues = torch.randn(N, k)

        cov = compute_eigenvalue_covariance(eigenvalues)

        # Off-diagonal elements should be small relative to diagonal
        diag = torch.diag(cov)
        off_diag_mask = ~torch.eye(k, dtype=torch.bool)
        off_diag = cov[off_diag_mask]

        # Off-diagonal values should be much smaller than diagonal
        assert off_diag.abs().mean() < diag.mean() * 0.2

    def test_covariance_correlated_has_off_diagonal(self) -> None:
        """Correlated eigenvalues should have significant off-diagonal terms.

        When eigenvalue positions are correlated (e.g., larger graphs tend to
        have larger eigenvalues at all positions), off-diagonal terms should
        be non-zero.
        """
        from tmgg.experiments.eigenstructure_study.analyzer import (
            compute_eigenvalue_covariance,
        )

        torch.manual_seed(42)
        N = 500
        k = 4

        # Generate correlated eigenvalues: all positions shift together
        base = torch.randn(N, 1)  # Shared component
        eigenvalues = base.expand(N, k) + 0.1 * torch.randn(N, k)

        cov = compute_eigenvalue_covariance(eigenvalues)

        # Off-diagonal elements should be significant
        diag = torch.diag(cov)
        off_diag_mask = ~torch.eye(k, dtype=torch.bool)
        off_diag = cov[off_diag_mask]

        # Off-diagonal should be substantial (correlated)
        assert off_diag.mean() > 0.5 * diag.mean()

    def test_covariance_summary_metrics(self) -> None:
        """Covariance summary should include all expected metrics."""
        from tmgg.experiments.eigenstructure_study.analyzer import (
            compute_covariance_summary,
            compute_eigenvalue_covariance,
        )

        N = 100
        k = 6
        eigenvalues = torch.randn(N, k)
        cov = compute_eigenvalue_covariance(eigenvalues)

        summary = compute_covariance_summary(cov)

        assert "frobenius_norm" in summary
        assert "trace" in summary
        assert "condition_number" in summary
        assert "off_diagonal_sum" in summary
        assert "off_diagonal_ratio" in summary
        assert "max_eigenvalue" in summary
        assert "min_eigenvalue" in summary

        # Basic sanity checks
        assert summary["frobenius_norm"] > 0
        assert summary["trace"] > 0  # Variance is positive
        assert summary["max_eigenvalue"] >= summary["min_eigenvalue"]

    def test_covariance_requires_multiple_graphs(self) -> None:
        """Covariance computation should require at least 2 graphs."""
        import pytest

        from tmgg.experiments.eigenstructure_study.analyzer import (
            compute_eigenvalue_covariance,
        )

        eigenvalues = torch.randn(1, 5)  # Only 1 graph

        with pytest.raises(ValueError, match="at least 2 graphs"):
            compute_eigenvalue_covariance(eigenvalues)

    def test_spectral_analyzer_covariance(self) -> None:
        """SpectralAnalyzer should compute covariance from collected data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 20, "p": 0.3},
                output_dir=output_dir,
                batch_size=10,
                seed=42,
            )
            collector.collect()

            analyzer = SpectralAnalyzer(output_dir)
            result = analyzer.compute_eigenvalue_covariance(matrix_type="adjacency")

            assert result.matrix_type == "adjacency"
            assert result.num_graphs == 20
            assert result.num_eigenvalues == 8
            assert len(result.covariance_matrix) == 8
            assert len(result.covariance_matrix[0]) == 8

    def test_covariance_evolution_computation(self) -> None:
        """NoisedAnalysisComparator should compute typed covariance evolution.

        Rationale
        ---------
        The covariance evolution result is consumed by both the CLI and the
        Hydra execution path. This test locks the typed API and its JSON
        serializer together so later refactors cannot drift the file shape.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 8, "num_graphs": 20, "p": 0.3},
                output_dir=original_dir,
                batch_size=10,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1, 0.2],
                seed=42,
            )
            noised_collector.collect()

            comparator = NoisedAnalysisComparator(original_dir, noised_dir)
            evolution = comparator.compute_covariance_evolution(matrix_type="adjacency")

            assert isinstance(evolution, CovarianceEvolutionResult)
            assert evolution.matrix_type == "adjacency"
            assert len(evolution.per_noise_level) == 2

            payload = evolution.to_json_dict()
            assert "original" in payload
            assert "per_noise_level" in payload
            per_noise_level = payload["per_noise_level"]
            assert isinstance(per_noise_level, list)
            for item in per_noise_level:
                assert isinstance(item, dict)
                assert "noise_level" in item
                assert "covariance" in item
                assert "frobenius_delta_relative" in item
                assert "trace_delta_relative" in item
                assert "off_diagonal_delta_relative" in item


class TestCovarianceCLI:
    """Tests for covariance CLI command."""

    def test_cli_covariance_help(self) -> None:
        """Covariance subcommand should show help."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["covariance", "--help"])
        assert result.exit_code == 0
        assert "--original-dir" in result.output
        assert "--matrix-type" in result.output

    def test_cli_covariance_original_only(self) -> None:
        """Covariance command should work with original data only."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            output_dir = Path(tmpdir) / "output"

            # Collect data first
            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 6, "num_graphs": 10, "p": 0.3},
                output_dir=original_dir,
                batch_size=5,
                seed=42,
            )
            collector.collect()

            # Run covariance command
            result = runner.invoke(
                main,
                [
                    "covariance",
                    "--original-dir",
                    str(original_dir),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert (output_dir / "covariance.json").exists()

    def test_cli_covariance_with_noise_evolution(self) -> None:
        """Covariance command should compute evolution with noised data."""
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = Path(tmpdir) / "original"
            noised_dir = Path(tmpdir) / "noised"
            output_dir = Path(tmpdir) / "output"

            # Collect original data
            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 6, "num_graphs": 10, "p": 0.3},
                output_dir=original_dir,
                batch_size=5,
                seed=42,
            )
            collector.collect()

            # Collect noised data
            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1],
                seed=42,
            )
            noised_collector.collect()

            # Run covariance command with noised data
            result = runner.invoke(
                main,
                [
                    "covariance",
                    "--original-dir",
                    str(original_dir),
                    "--noised-dir",
                    str(noised_dir),
                    "--output-dir",
                    str(output_dir),
                ],
            )

            assert result.exit_code == 0
            assert (output_dir / "covariance_evolution.json").exists()

    def test_covariance_evolution_json_matches_cli_and_hydra_execute(self) -> None:
        """CLI and Hydra execution should serialize covariance evolution identically.

        Rationale
        ---------
        The file format is a compatibility boundary. Both execution paths
        should delegate to the same dataclass-owned serializer so the JSON
        payload cannot drift across entry points.
        """
        from click.testing import CliRunner

        from tmgg.experiments.eigenstructure_study.cli import main
        from tmgg.experiments.eigenstructure_study.execute import (
            execute_eigenstructure,
        )

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_dir = root / "original"
            noised_dir = root / "noised"

            collector = EigenstructureCollector(
                dataset_name="er",
                dataset_config={"num_nodes": 6, "num_graphs": 10, "p": 0.3},
                output_dir=original_dir,
                batch_size=5,
                seed=42,
            )
            collector.collect()

            noised_collector = NoisedEigenstructureCollector(
                input_dir=original_dir,
                output_dir=noised_dir,
                noise_type="gaussian",
                noise_levels=[0.1],
                seed=42,
            )
            noised_collector.collect()

            comparator = NoisedAnalysisComparator(original_dir, noised_dir)
            expected = comparator.compute_covariance_evolution(
                matrix_type="adjacency"
            ).to_json_dict()

            cli_output_dir = root / "cli_output"
            cli_result = runner.invoke(
                main,
                [
                    "covariance",
                    "--original-dir",
                    str(original_dir),
                    "--noised-dir",
                    str(noised_dir),
                    "--output-dir",
                    str(cli_output_dir),
                ],
            )
            assert cli_result.exit_code == 0
            with open(cli_output_dir / "covariance_evolution.json") as f:
                assert json.load(f) == expected

            hydra_output_dir = root / "hydra_output"
            (hydra_output_dir / "original").mkdir(parents=True)
            (hydra_output_dir / "noised").mkdir(parents=True)

            for source_dir, target_dir in (
                (original_dir, hydra_output_dir / "original"),
                (noised_dir, hydra_output_dir / "noised"),
            ):
                for item in source_dir.iterdir():
                    if item.is_file():
                        target_path = target_dir / item.name
                        target_path.write_bytes(item.read_bytes())
                    else:
                        nested_target = target_dir / item.name
                        nested_target.mkdir()
                        for nested in item.iterdir():
                            (nested_target / nested.name).write_bytes(
                                nested.read_bytes()
                            )

            cfg = OmegaConf.create(
                {
                    "phase": "compare",
                    "seed": 42,
                    "paths": {"output_dir": str(hydra_output_dir)},
                    "comparison": {
                        "subspace_k": 3,
                        "procrustes_k_values": [1, 2, 4],
                        "compute_covariance_evolution": True,
                    },
                    "analysis": {"matrix_type": "adjacency"},
                }
            )
            execute_eigenstructure(cfg)

            with open(
                hydra_output_dir / "comparison" / "covariance_evolution.json"
            ) as f:
                assert json.load(f) == expected
