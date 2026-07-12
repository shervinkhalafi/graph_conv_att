"""Unit tests for DiGress checkpoint format detection and remapping.

Test rationale:
    Verifies checkpoint compatibility utilities for loading DiGress models
    from different checkpoint formats (original DiGress, tmgg Lightning, tmgg raw).

Invariants:
    - Format detection correctly identifies checkpoint structure from key patterns
    - Remapping preserves tensor values while transforming key prefixes
    - Roundtrip remapping (A→B→A) returns identity for keys that exist in both formats
    - Loading works with auto-detection for all supported formats
"""

import pytest
import torch

from tmgg.training.lightning_modules.digress_checkpoint_compat import (
    CheckpointFormat,
    detect_checkpoint_format,
    load_compatible_state_dict,
    load_digress_checkpoint,
    remap_state_dict,
)


class TestFormatDetection:
    """Verify checkpoint format detection from key patterns."""

    def test_detect_original_digress_format(self) -> None:
        """Original DiGress keys (model.mlp_in_X) should be detected."""
        state_dict = {
            "model.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.mlp_in_X.0.bias": torch.randn(64),
            "model.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
            "model.lin_E.weight": torch.randn(1, 64),
        }
        fmt = detect_checkpoint_format(state_dict)
        assert fmt == CheckpointFormat.ORIGINAL_DIGRESS

    def test_detect_tmgg_lightning_format(self) -> None:
        """tmgg Lightning keys (model.transformer.*) should be detected."""
        state_dict = {
            "model.transformer.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.transformer.mlp_in_X.0.bias": torch.randn(64),
            "model.transformer.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
            "model.transformer.lin_E.weight": torch.randn(1, 64),
        }
        fmt = detect_checkpoint_format(state_dict)
        assert fmt == CheckpointFormat.TMGG_LIGHTNING

    def test_detect_tmgg_raw_format(self) -> None:
        """tmgg raw keys (transformer.*) should be detected."""
        state_dict = {
            "transformer.mlp_in_X.0.weight": torch.randn(64, 1),
            "transformer.mlp_in_X.0.bias": torch.randn(64),
            "transformer.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
            "transformer.lin_E.weight": torch.randn(1, 64),
        }
        fmt = detect_checkpoint_format(state_dict)
        assert fmt == CheckpointFormat.TMGG_RAW

    def test_detect_unknown_format(self) -> None:
        """Unrecognized key patterns should return UNKNOWN."""
        state_dict = {
            "encoder.layer1.weight": torch.randn(64, 64),
            "decoder.layer2.bias": torch.randn(64),
        }
        fmt = detect_checkpoint_format(state_dict)
        assert fmt == CheckpointFormat.UNKNOWN

    def test_detect_empty_state_dict(self) -> None:
        """Empty state dict should return UNKNOWN."""
        fmt = detect_checkpoint_format({})
        assert fmt == CheckpointFormat.UNKNOWN


class TestStateRemapping:
    """Verify state dict key remapping between formats."""

    @pytest.fixture
    def original_digress_state(self) -> dict[str, torch.Tensor]:
        """Create sample original DiGress state dict."""
        return {
            "model.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.mlp_in_X.0.bias": torch.randn(64),
            "model.mlp_in_E.0.weight": torch.randn(32, 1),
            "model.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
            "model.tf_layers.0.self_attn.k.weight": torch.randn(128, 128),
            "model.lin_E.weight": torch.randn(1, 64),
            "model.y_norm.weight": torch.randn(256),
        }

    @pytest.fixture
    def tmgg_lightning_state(self) -> dict[str, torch.Tensor]:
        """Create sample tmgg Lightning state dict."""
        return {
            "model.transformer.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.transformer.mlp_in_X.0.bias": torch.randn(64),
            "model.transformer.mlp_in_E.0.weight": torch.randn(32, 1),
            "model.transformer.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
            "model.transformer.tf_layers.0.self_attn.k.weight": torch.randn(128, 128),
            "model.transformer.lin_E.weight": torch.randn(1, 64),
            "model.transformer.y_norm.weight": torch.randn(256),
        }

    def test_remap_original_to_lightning(
        self, original_digress_state: dict[str, torch.Tensor]
    ) -> None:
        """Remapping original DiGress → tmgg Lightning should update prefixes."""
        remapped = remap_state_dict(
            original_digress_state,
            CheckpointFormat.ORIGINAL_DIGRESS,
            CheckpointFormat.TMGG_LIGHTNING,
        )

        # Check key prefixes changed
        for key in remapped:
            assert key.startswith("model.transformer."), f"Key not remapped: {key}"

        # Check tensor values preserved
        assert torch.equal(
            remapped["model.transformer.mlp_in_X.0.weight"],
            original_digress_state["model.mlp_in_X.0.weight"],
        )

    def test_remap_lightning_to_original(
        self, tmgg_lightning_state: dict[str, torch.Tensor]
    ) -> None:
        """Remapping tmgg Lightning → original DiGress should update prefixes."""
        remapped = remap_state_dict(
            tmgg_lightning_state,
            CheckpointFormat.TMGG_LIGHTNING,
            CheckpointFormat.ORIGINAL_DIGRESS,
        )

        # Check key prefixes changed
        for key in remapped:
            if (
                "mlp_in" in key
                or "tf_layers" in key
                or "lin_" in key
                or "y_norm" in key
            ):
                assert key.startswith("model."), f"Key not remapped: {key}"
                assert not key.startswith(
                    "model.transformer."
                ), f"Key still has transformer: {key}"

    def test_remap_identity(
        self, original_digress_state: dict[str, torch.Tensor]
    ) -> None:
        """Remapping to same format should be identity."""
        remapped = remap_state_dict(
            original_digress_state,
            CheckpointFormat.ORIGINAL_DIGRESS,
            CheckpointFormat.ORIGINAL_DIGRESS,
        )

        assert remapped.keys() == original_digress_state.keys()
        for key in remapped:
            assert torch.equal(remapped[key], original_digress_state[key])

    def test_remap_roundtrip(
        self, original_digress_state: dict[str, torch.Tensor]
    ) -> None:
        """Roundtrip A→B→A should preserve keys and values."""
        # Original → Lightning → Original
        to_lightning = remap_state_dict(
            original_digress_state,
            CheckpointFormat.ORIGINAL_DIGRESS,
            CheckpointFormat.TMGG_LIGHTNING,
        )
        back_to_original = remap_state_dict(
            to_lightning,
            CheckpointFormat.TMGG_LIGHTNING,
            CheckpointFormat.ORIGINAL_DIGRESS,
        )

        assert back_to_original.keys() == original_digress_state.keys()
        for key in back_to_original:
            assert torch.equal(back_to_original[key], original_digress_state[key])

    def test_remap_unknown_format_raises(
        self, original_digress_state: dict[str, torch.Tensor]
    ) -> None:
        """Remapping from UNKNOWN format should raise ValueError."""
        with pytest.raises(ValueError, match="UNKNOWN"):
            remap_state_dict(
                original_digress_state,
                CheckpointFormat.UNKNOWN,
                CheckpointFormat.TMGG_LIGHTNING,
            )


class TestCheckpointLoading:
    """Verify checkpoint loading with format detection."""

    @pytest.fixture
    def mock_checkpoint_path(self, tmp_path) -> tuple:
        """Create mock checkpoint files in different formats."""
        # Original DiGress format checkpoint
        original_state = {
            "model.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.mlp_in_E.0.weight": torch.randn(32, 1),
            "model.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
        }
        original_ckpt = {
            "state_dict": original_state,
            "hyper_parameters": {"k": 8, "n_layers": 4},
        }
        original_path = tmp_path / "original.ckpt"
        torch.save(original_ckpt, original_path)

        # tmgg Lightning format checkpoint
        lightning_state = {
            "model.transformer.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.transformer.mlp_in_E.0.weight": torch.randn(32, 1),
            "model.transformer.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
        }
        lightning_ckpt = {
            "state_dict": lightning_state,
            "hyper_parameters": {"k": 8, "n_layers": 4},
        }
        lightning_path = tmp_path / "lightning.ckpt"
        torch.save(lightning_ckpt, lightning_path)

        return original_path, lightning_path

    def test_load_original_with_autodetect(self, mock_checkpoint_path) -> None:
        """Loading original DiGress format should auto-detect and remap."""
        original_path, _ = mock_checkpoint_path

        loaded = load_digress_checkpoint(original_path)

        assert loaded.original_format == CheckpointFormat.ORIGINAL_DIGRESS
        assert loaded.target_format == CheckpointFormat.TMGG_LIGHTNING
        assert "model.transformer.mlp_in_X.0.weight" in loaded.state_dict

    def test_load_lightning_with_autodetect(self, mock_checkpoint_path) -> None:
        """Loading tmgg Lightning format should auto-detect (no remapping needed)."""
        _, lightning_path = mock_checkpoint_path

        loaded = load_digress_checkpoint(lightning_path)

        assert loaded.original_format == CheckpointFormat.TMGG_LIGHTNING
        assert loaded.target_format == CheckpointFormat.TMGG_LIGHTNING
        assert "model.transformer.mlp_in_X.0.weight" in loaded.state_dict

    def test_load_preserves_hyperparameters(self, mock_checkpoint_path) -> None:
        """Loading should preserve hyperparameters from checkpoint."""
        original_path, _ = mock_checkpoint_path

        loaded = load_digress_checkpoint(original_path)

        assert loaded.hyper_parameters == {"k": 8, "n_layers": 4}

    def test_load_raw_state_dict(self, tmp_path) -> None:
        """Loading raw state dict (no wrapper) should work."""
        raw_state = {
            "transformer.mlp_in_X.0.weight": torch.randn(64, 1),
            "transformer.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
        }
        raw_path = tmp_path / "raw.pt"
        torch.save(raw_state, raw_path)

        loaded = load_digress_checkpoint(raw_path)

        assert loaded.original_format == CheckpointFormat.TMGG_RAW
        assert loaded.hyper_parameters == {}

    def test_load_unknown_format_raises(self, tmp_path) -> None:
        """Loading unrecognized format should raise ValueError."""
        unknown_state = {
            "encoder.layer.weight": torch.randn(64, 64),
        }
        unknown_path = tmp_path / "unknown.ckpt"
        torch.save({"state_dict": unknown_state}, unknown_path)

        with pytest.raises(ValueError, match="Could not detect"):
            load_digress_checkpoint(unknown_path)


class TestConvenienceFunction:
    """Verify load_compatible_state_dict convenience function."""

    def test_load_compatible_state_dict(self, tmp_path) -> None:
        """load_compatible_state_dict should return ready-to-use state dict."""
        original_state = {
            "model.mlp_in_X.0.weight": torch.randn(64, 1),
            "model.tf_layers.0.self_attn.q.weight": torch.randn(128, 128),
        }
        ckpt_path = tmp_path / "test.ckpt"
        torch.save({"state_dict": original_state}, ckpt_path)

        state_dict = load_compatible_state_dict(ckpt_path)

        assert "model.transformer.mlp_in_X.0.weight" in state_dict
        assert "model.transformer.tf_layers.0.self_attn.q.weight" in state_dict


class TestGenerateReferenceGraphs:
    """Verify generate_reference_graphs produces valid graphs for all dataset types."""

    @pytest.mark.parametrize(
        "dataset_type",
        ["sbm", "erdos_renyi", "er", "watts_strogatz", "ws", "regular", "tree"],
    )
    def test_generates_correct_count_and_shape(self, dataset_type: str) -> None:
        """Generated graphs should have correct count and node dimensions."""
        from tmgg.evaluation.reference_graphs import (
            generate_reference_graphs,
        )

        graphs = generate_reference_graphs(
            dataset_type=dataset_type,
            num_graphs=5,
            num_nodes=12,
            seed=42,
        )

        assert len(graphs) == 5, f"Expected 5 graphs, got {len(graphs)}"
        for g in graphs:
            assert g.shape == (12, 12), f"Expected (12, 12), got {g.shape}"

    @pytest.mark.parametrize("dataset_type", ["sbm", "erdos_renyi", "regular"])
    def test_graphs_are_valid_adjacency(self, dataset_type: str) -> None:
        """Generated graphs should be binary, symmetric, zero-diagonal."""
        from tmgg.evaluation.reference_graphs import (
            generate_reference_graphs,
        )

        graphs = generate_reference_graphs(
            dataset_type=dataset_type,
            num_graphs=3,
            num_nodes=10,
            seed=42,
        )

        for g in graphs:
            # Binary
            assert torch.all((g == 0) | (g == 1)), "Non-binary values"
            # Symmetric
            assert torch.allclose(g, g.T), "Not symmetric"
            # Zero diagonal
            assert torch.all(g.diagonal() == 0), "Non-zero diagonal"

    def test_sbm_parameters_used(self) -> None:
        """SBM-specific parameters should affect generation."""
        from tmgg.evaluation.reference_graphs import (
            generate_reference_graphs,
        )

        # High intra-block probability, low inter-block
        graphs_structured = generate_reference_graphs(
            dataset_type="sbm",
            num_graphs=10,
            num_nodes=20,
            seed=42,
            p=0.9,
            q=0.01,
            num_blocks=2,
        )

        # Check that block structure exists (first half should be denser than cross-block)
        total_intra = 0.0
        total_inter = 0.0
        for g in graphs_structured:
            intra = g[:10, :10].sum() + g[10:, 10:].sum()
            inter = g[:10, 10:].sum() + g[10:, :10].sum()
            total_intra += intra.item()
            total_inter += inter.item()

        # Intra-block density should be much higher than inter-block
        assert total_intra > total_inter * 3, "SBM block structure not evident"

    def test_invalid_dataset_type_raises(self) -> None:
        """Unknown dataset type should raise ValueError."""
        from tmgg.evaluation.reference_graphs import (
            generate_reference_graphs,
        )

        with pytest.raises(ValueError, match="graph_type must be one of"):
            generate_reference_graphs(
                dataset_type="invalid_type",
                num_graphs=5,
                num_nodes=10,
            )


class TestResultsToCSV:
    """Verify CSV output formatting."""

    def test_writes_valid_csv(self, tmp_path) -> None:
        """results_to_csv should write readable CSV with correct columns."""
        import csv
        import importlib.util
        from pathlib import Path

        # Load compare_checkpoints module from scripts directory
        spec = importlib.util.spec_from_file_location(
            "compare_checkpoints",
            Path(__file__).parent.parent / "scripts" / "compare_checkpoints.py",
        )
        assert spec is not None and spec.loader is not None
        compare_checkpoints = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare_checkpoints)
        results_to_csv = compare_checkpoints.results_to_csv

        results = [
            {
                "checkpoint_name": "model1.ckpt",
                "dataset_type": "sbm",
                "num_generated": 100,
                "num_nodes": 20,
                "num_steps": 50,
                "mmd_results": {
                    "degree_mmd": 0.001,
                    "clustering_mmd": 0.002,
                    "spectral_mmd": 0.003,
                },
            },
            {
                "checkpoint_name": "model2.ckpt",
                "dataset_type": "erdos_renyi",
                "num_generated": 100,
                "num_nodes": 20,
                "num_steps": 50,
                "mmd_results": {
                    "degree_mmd": 0.004,
                    "clustering_mmd": 0.005,
                    "spectral_mmd": 0.006,
                },
            },
        ]

        output_path = tmp_path / "results.csv"
        results_to_csv(results, output_path)

        # Verify file exists and is readable
        assert output_path.exists()

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["checkpoint"] == "model1.ckpt"
        assert rows[0]["dataset"] == "sbm"
        assert float(rows[0]["degree_mmd"]) == 0.001
        assert rows[1]["checkpoint"] == "model2.ckpt"

    def test_handles_error_results(self, tmp_path) -> None:
        """results_to_csv should handle results with errors."""
        import importlib.util
        from pathlib import Path

        # Load compare_checkpoints module from scripts directory
        spec = importlib.util.spec_from_file_location(
            "compare_checkpoints",
            Path(__file__).parent.parent / "scripts" / "compare_checkpoints.py",
        )
        assert spec is not None and spec.loader is not None
        compare_checkpoints = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare_checkpoints)
        results_to_csv = compare_checkpoints.results_to_csv

        results = [
            {
                "checkpoint_name": "failed.ckpt",
                "dataset_type": "sbm",
                "error": "Load failed",
                "mmd_results": {
                    "degree_mmd": float("nan"),
                    "clustering_mmd": float("nan"),
                    "spectral_mmd": float("nan"),
                },
            },
        ]

        output_path = tmp_path / "error_results.csv"
        results_to_csv(results, output_path)

        assert output_path.exists()


class TestPrintComparisonTable:
    """Verify table formatting (smoke test)."""

    def test_does_not_crash(self, capsys) -> None:
        """print_comparison_table should handle typical results without error."""
        import importlib.util
        from pathlib import Path

        # Load compare_checkpoints module from scripts directory
        spec = importlib.util.spec_from_file_location(
            "compare_checkpoints",
            Path(__file__).parent.parent / "scripts" / "compare_checkpoints.py",
        )
        assert spec is not None and spec.loader is not None
        compare_checkpoints = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare_checkpoints)
        print_comparison_table = compare_checkpoints.print_comparison_table

        results = [
            {
                "checkpoint_name": "model1.ckpt",
                "dataset_type": "sbm",
                "mmd_results": {
                    "degree_mmd": 0.001,
                    "clustering_mmd": 0.002,
                    "spectral_mmd": 0.003,
                },
            },
            {
                "checkpoint_name": "model1.ckpt",
                "dataset_type": "erdos_renyi",
                "mmd_results": {
                    "degree_mmd": 0.004,
                    "clustering_mmd": 0.005,
                    "spectral_mmd": 0.006,
                },
            },
        ]

        # Should not raise
        print_comparison_table(results)

        captured = capsys.readouterr()
        assert "Degree MMD" in captured.out
        assert "model1.ckpt" in captured.out
