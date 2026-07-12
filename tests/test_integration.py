"""Integration tests for end-to-end training pipelines."""

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data import generate_sbm_adjacency
from tmgg.data.data_modules.multigraph_data_module import (
    GraphDataCollator,
    _adjacencies_to_pyg,
    _ListDataset,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.attention import MultiLayerAttention
from tmgg.models.gnn import GNN, GNNSymmetric, NodeVarGNN
from tmgg.models.hybrid import create_sequential_model
from tmgg.utils.noising import (
    add_edge_flip_noise,
    add_gaussian_noise,
    add_rotation_noise,
    random_skew_symmetric_matrix,
)


class TestEndToEndTraining:
    """Test complete training pipelines for different model types."""

    @pytest.fixture
    def training_data(
        self,
    ) -> tuple[DataLoader[Any], list[torch.Tensor]]:
        """Create sample training data."""
        # Generate a few SBM graphs
        block_sizes_list = [[5, 5], [3, 3, 4], [7, 3]]
        adjacency_matrices: list[torch.Tensor] = []

        for block_sizes in block_sizes_list:
            A = generate_sbm_adjacency(block_sizes, p_intra=1.0, p_inter=0.0)
            adjacency_matrices.append(torch.tensor(A, dtype=torch.float32))

        # Create dataset using PyG Data objects
        import numpy as np

        adjs_np = np.stack([a.numpy() for a in adjacency_matrices])
        data_list = _adjacencies_to_pyg(adjs_np) * 17  # ~50 samples from 3 graphs
        dataloader: DataLoader[Any] = DataLoader(
            _ListDataset(data_list),
            batch_size=10,
            shuffle=True,
            collate_fn=GraphDataCollator(),
        )

        return dataloader, adjacency_matrices

    def test_gnn_training_pipeline(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test GNN training end-to-end."""
        dataloader, adjacency_matrices = training_data

        # Initialize model
        model = GNN(num_layers=2, num_terms=2, feature_dim_in=10, feature_dim_out=5)
        _ = model.float()

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        initial_loss: float | None = None
        _ = model.train()

        for _ in range(5):
            epoch_loss = 0.0
            for graph_batch in dataloader:
                batch = graph_batch.dense_adjacency()
                # Add noise
                batch_noisy = add_edge_flip_noise(batch, p=0.1)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass - GNN returns GraphDistribution; ask for dense
                # so legacy_edge_scalar reads E_feat directly.
                result = model(edge_scalar_graphdata(batch_noisy), output_dense=True)
                logits = legacy_edge_scalar(result)
                A_pred = torch.sigmoid(logits)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if initial_loss is None:
                initial_loss = avg_loss

        # Check that loss decreased
        assert initial_loss is not None
        assert avg_loss < initial_loss

        # Test evaluation
        _ = model.eval()
        with torch.no_grad():
            test_batch = adjacency_matrices[0].unsqueeze(0)
            test_noisy = add_edge_flip_noise(test_batch, p=0.1)

            result = model(edge_scalar_graphdata(test_noisy.float()), output_dense=True)
            logits = legacy_edge_scalar(result)
            A_pred = torch.sigmoid(logits)

            assert A_pred.shape == test_batch.shape
            assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)

    def test_attention_training_pipeline(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test attention model training end-to-end.

        MultiLayerAttention now returns a single tensor (reconstructed adjacency)
        rather than (output, attention_scores) tuple.
        """
        dataloader, adjacency_matrices = training_data

        # Initialize model
        k = 10  # Number of eigenvectors
        model = MultiLayerAttention(d_model=k, num_heads=2, num_layers=2, d_k=5, d_v=5)
        _ = model.float()

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training with rotation noise
        skew = random_skew_symmetric_matrix(k, rng=np.random.default_rng(42))
        initial_loss: float | None = None
        _ = model.train()

        for _ in range(5):
            epoch_loss = 0.0
            for graph_batch in dataloader:
                batch = graph_batch.dense_adjacency()
                # Add rotation noise
                batch_noisy = add_rotation_noise(batch, eps=0.1, skew=skew)
                _, V_noisy = torch.linalg.eigh(batch_noisy.float())
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Use only top k eigenvectors
                V_input = V_noisy[:, :, :k].float()

                # Forward pass via raw attention (eigenvectors aren't adjacency)
                A_pred = model.apply_attention(V_input)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if initial_loss is None:
                initial_loss = avg_loss

        # Check that loss decreased
        assert initial_loss is not None
        assert avg_loss < initial_loss

    def test_hybrid_training_pipeline(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test hybrid model training end-to-end."""
        dataloader, adjacency_matrices = training_data

        # Create hybrid model
        gnn_config: dict[str, Any] = {
            "num_layers": 1,
            "num_terms": 2,
            "feature_dim_in": 10,
            "feature_dim_out": 5,
        }

        transformer_config: dict[str, Any] = {
            "num_heads": 2,
            "num_layers": 1,
            "d_k": 5,
            "d_v": 5,
        }

        model = create_sequential_model(gnn_config, transformer_config)
        _ = model.float()

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training loop
        initial_loss: float | None = None
        _ = model.train()

        for _ in range(5):
            epoch_loss = 0.0
            for graph_batch in dataloader:
                batch = graph_batch.dense_adjacency()
                # Add Gaussian noise
                batch_noisy = add_gaussian_noise(batch, eps=0.1)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass - model returns GraphDistribution; ask for dense.
                result = model(edge_scalar_graphdata(batch_noisy), output_dense=True)
                logits = legacy_edge_scalar(result)
                A_pred = torch.sigmoid(logits)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if initial_loss is None:
                initial_loss = avg_loss

        # Check that loss decreased
        assert initial_loss is not None
        assert avg_loss < initial_loss

        # Test evaluation
        _ = model.eval()
        with torch.no_grad():
            test_batch = adjacency_matrices[0].unsqueeze(0)
            test_noisy = add_gaussian_noise(test_batch, eps=0.1)

            result = model(edge_scalar_graphdata(test_noisy.float()), output_dense=True)
            logits = legacy_edge_scalar(result)
            A_pred = torch.sigmoid(logits)

            assert A_pred.shape == test_batch.shape
            assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)

    def test_multi_noise_level_training(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test training with multiple noise levels."""
        dataloader, _ = training_data

        # Initialize simple GNN
        model = GNNSymmetric(num_layers=1, feature_dim_in=10, feature_dim_out=5)
        _ = model.float()

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Define noise levels
        noise_levels = [0.05, 0.1, 0.2, 0.3]

        _ = model.train()
        for _ in range(3):
            for graph_batch in dataloader:
                batch = graph_batch.dense_adjacency()
                # Sample random noise level
                eps: float = np.random.choice(noise_levels)

                # Add noise
                batch_noisy = add_edge_flip_noise(batch, p=eps)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass - GNN returns GraphDistribution; ask for dense
                # so legacy_edge_scalar reads E_feat directly.
                result = model(edge_scalar_graphdata(batch_noisy), output_dense=True)
                logits = legacy_edge_scalar(result)
                A_pred = torch.sigmoid(logits)

                # Compute loss
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Test on each noise level
        _ = model.eval()
        with torch.no_grad():
            from torch_geometric.data import Batch, Data

            sample = dataloader.dataset[0]
            assert isinstance(sample, Data)
            from tmgg.data.datasets.graph_types import DenseGraphState
            test_gd = DenseGraphState.from_pyg_batch(Batch.from_data_list([sample]))
            test_batch = test_gd.dense_adjacency()

            for eps in noise_levels:
                test_noisy = add_edge_flip_noise(test_batch, p=eps)
                result = model(edge_scalar_graphdata(test_noisy.float()), output_dense=True)
                logits = legacy_edge_scalar(result)
                A_pred = torch.sigmoid(logits)

                assert A_pred.shape == test_batch.shape
                assert torch.all(A_pred >= 0) and torch.all(A_pred <= 1)

    def test_learning_rate_scheduling(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test training with learning rate scheduling."""
        dataloader, _ = training_data

        # Initialize model
        model = NodeVarGNN(num_layers=1, num_terms=2, feature_dim=10)
        _ = model.float()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Add cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2
        )

        # Track learning rates
        lrs: list[float] = []

        _ = model.train()
        for _ in range(10):
            for graph_batch in dataloader:
                batch = graph_batch.dense_adjacency()
                batch_noisy = add_gaussian_noise(batch, eps=0.1)
                batch_noisy = batch_noisy.float()
                batch = batch.float()

                # Forward pass
                result = model(edge_scalar_graphdata(batch_noisy), output_dense=True)
                A_pred = legacy_edge_scalar(result)
                loss = criterion(A_pred, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step scheduler
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Check that learning rate changed
        assert len(set(lrs)) > 1
        assert min(lrs) < 0.1  # LR decreased at some point

    def test_gradient_flow_with_regularization(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test gradients flow with eigenvalue regularization.

        Eigendecomposition gradients involve 1/(λ_i - λ_j) terms that explode
        for near-degenerate eigenvalues. Adding diagonal regularization spreads
        eigenvalues apart, preventing this numerical instability.

        This test verifies the GNN model directly. For hybrid models, gradients
        must traverse both eigendecomposition and transformer layers, resulting
        in smaller (but still non-zero) gradients that require relaxed tolerance.
        """
        dataloader, _ = training_data

        # Test GNN with eigenvalue regularization
        model = GNN(num_layers=2, feature_dim_out=5, eigenvalue_reg=1e-3)
        _ = model.float()

        batch = next(iter(dataloader)).dense_adjacency()
        batch_noisy = add_edge_flip_noise(batch, p=0.1).float()
        batch = batch.float()

        result = model(edge_scalar_graphdata(batch_noisy), output_dense=True)
        logits = legacy_edge_scalar(result)
        output = torch.sigmoid(logits)
        loss = torch.mean((output - batch) ** 2)
        loss.backward()

        # With regularization, GNN gradients should be numerically stable
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1
                assert not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=1e-10
                ), f"Zero gradient for {name}"

        assert params_with_grad > 0, "No parameters have gradients"

    def test_gradient_flow_hybrid_model(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Test gradient flow in hybrid model with relaxed tolerance.

        Hybrid models have gradients flowing through eigendecomposition then
        transformer attention, resulting in smaller gradients. We verify
        gradients exist (atol=1e-6) rather than strict non-zero check.
        """
        dataloader, _ = training_data

        gnn_config: dict[str, Any] = {
            "num_layers": 1,
            "feature_dim_out": 5,
            "eigenvalue_reg": 1e-3,
        }
        transformer_config: dict[str, Any] = {
            "num_heads": 2,
            "num_layers": 1,
        }

        model = create_sequential_model(gnn_config, transformer_config)
        _ = model.float()

        batch = next(iter(dataloader)).dense_adjacency()
        batch_noisy = add_edge_flip_noise(batch, p=0.1).float()
        batch = batch.float()

        result = model(edge_scalar_graphdata(batch_noisy), output_dense=True)
        output = legacy_edge_scalar(result)
        loss = torch.mean((output - batch) ** 2)
        loss.backward()

        # Verify most parameters have non-trivial gradients (relaxed tolerance)
        params_with_grad = 0
        params_with_nonzero_grad = 0
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad += 1
                if not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=1e-6
                ):
                    params_with_nonzero_grad += 1

        assert params_with_grad > 0, "No parameters have gradients"
        # At least 50% of parameters should have meaningful gradients
        grad_ratio = params_with_nonzero_grad / params_with_grad
        assert (
            grad_ratio >= 0.5
        ), f"Only {grad_ratio:.1%} of parameters have meaningful gradients"

    def test_gradient_flow_without_regularization(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Verify model parameter gradients work regardless of eigenvalue degeneracy.

        The eigendecomposition backward pass has 1/(λ_i - λ_j) terms that produce NaN
        when computing ∂V/∂A (gradient of eigenvectors w.r.t. input matrix). However,
        for training a neural network that uses eigenvectors, we only need ∂L/∂W
        (gradient w.r.t. model weights), which does NOT involve the ∂V/∂A path.

        The gradient path for model parameters is:
            Loss → output → GNN layers → eigenvectors (V)
        The weights receive gradients from how they transform V, not from how A produces V.

        Therefore, eigenvalue degeneracy does NOT cause NaN in model parameter gradients,
        and eigenvalue_reg is not strictly necessary for standard training. It may still
        help with numerical stability in edge cases or if input gradients are needed.
        """
        dataloader, _ = training_data

        # Model without regularization - gradients still work due to gradient path
        model = GNN(num_layers=2, feature_dim_out=5, eigenvalue_reg=0.0)
        _ = model.float()

        batch = next(iter(dataloader)).dense_adjacency()
        batch = batch.float()

        result = model(edge_scalar_graphdata(batch), output_dense=True)
        logits = legacy_edge_scalar(result)
        output = torch.sigmoid(logits)
        loss = torch.mean((output - batch) ** 2)
        loss.backward()

        # Model parameter gradients are valid even with degenerate eigenvalues
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
                assert not torch.allclose(
                    param.grad, torch.zeros_like(param.grad), atol=1e-10
                ), f"Zero gradient for {name}"

    def test_input_gradient_nan_with_degenerate_eigenvalues(
        self, training_data: tuple[DataLoader[GraphData], list[torch.Tensor]]
    ) -> None:
        """Document that input gradients (∂V/∂A) DO have NaN with degeneracy.

        While model parameter gradients work fine (see test_gradient_flow_without_regularization),
        the gradient of eigenvectors with respect to the input matrix contains 1/(λ_i - λ_j)
        terms that become NaN when eigenvalues are exactly equal.

        This test verifies this behavior as documentation. In practice, this only matters
        if you need input gradients (e.g., for adversarial attacks on graph structure).
        For standard training, input gradients are not used.
        """
        # Create a perfect block-diagonal matrix with exactly degenerate eigenvalues
        # Two K_5 complete graphs have eigenvalues: -1 (8 times), 4 (2 times)
        block = torch.ones((5, 5)) - torch.eye(5)
        A = torch.block_diag(block, block).float()

        # Enable gradient tracking on input
        batch = A.unsqueeze(0).requires_grad_(True)

        eigenvalues, V = torch.linalg.eigh(batch)

        # Verify eigenvalues are degenerate
        unique_eigs = torch.unique(eigenvalues.round(decimals=3))
        assert (
            len(unique_eigs) == 2
        ), f"Expected 2 unique eigenvalues, got {len(unique_eigs)}"

        # Backprop through eigenvectors
        loss = V.sum()
        loss.backward()

        # Input gradient should have NaN due to 1/(λ_i - λ_j) singularity
        assert batch.grad is not None
        assert torch.isnan(
            batch.grad
        ).any(), "Expected NaN in input gradient due to eigenvalue degeneracy"


if __name__ == "__main__":
    pytest.main([__file__])
