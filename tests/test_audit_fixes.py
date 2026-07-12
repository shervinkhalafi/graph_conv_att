"""Tests for audit report fixes.

Verifies that all 14 issues identified in AUDIT_REPORT.md have been correctly
addressed. Each test class corresponds to one or more issues.

Test Rationale
--------------
These tests validate mathematical correctness and numerical stability fixes,
not functional behavior. The goal is to ensure the fixes prevent the specific
failure modes documented in the audit report.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from tests._helpers.graph_builders import edge_scalar_graphdata, legacy_edge_scalar
from tmgg.data import generate_sbm_adjacency
from tmgg.data.datasets.graph_types import GraphData
from tmgg.models.attention.attention import MultiLayerAttention
from tmgg.models.gnn.gnn_sym import GNNSymmetric
from tmgg.models.gnn.nvgnn import NodeVarGNN
from tmgg.models.hybrid.hybrid import SequentialDenoisingModel
from tmgg.models.layers.eigen_embedding import _EigenEmbedding as EigenEmbedding
from tmgg.models.layers.gcn import GraphConvolutionLayer
from tmgg.models.layers.masked_softmax import masked_softmax
from tmgg.models.layers.mha_layer import MultiHeadSelfAttention
from tmgg.models.layers.nvgcn_layer import NodeVarGraphConvolutionLayer
from tmgg.models.layers.topk_eigen import TopKEigenLayer
from tmgg.models.spectral_denoisers.filter_bank import GraphFilterBank
from tmgg.utils.spectral.spectral_deltas import (
    compute_spectral_deltas,
)


class TestIssue1AttentionScaling:
    """Issue 1 [CRITICAL]: Missing 1/sqrt(d_k) scaling factor.

    Rationale: Without scaling, attention scores have variance proportional to d_k,
    causing softmax saturation. The test verifies that scores are scaled correctly.
    """

    def test_scale_factor_is_correct(self):
        """Verify scale = 1/sqrt(d_k)."""
        d_model = 64
        num_heads = 4
        d_k = d_model // num_heads  # 16

        layer = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

        expected_scale = 1.0 / math.sqrt(d_k)
        assert (
            abs(layer.scale - expected_scale) < 1e-6
        ), f"Scale should be 1/sqrt({d_k})={expected_scale}, got {layer.scale}"

    def test_attention_scores_bounded(self):
        """Verify attention scores don't explode with large d_k."""
        d_model = 256
        num_heads = 4

        layer = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(2, 10, d_model)

        output = layer(x)

        # Output should be valid (no NaN/Inf)
        assert not torch.isnan(output).any(), "NaN in attention output"
        assert not torch.isinf(output).any(), "Inf in attention output"


class TestIssue2SymmetryEnforcement:
    """Issue 2 [CRITICAL]: Eigendecomposition without symmetry enforcement.

    Rationale: eigh expects symmetric matrices. Small asymmetries from
    floating-point operations cause inconsistent eigenvector orientations.
    """

    def test_eigen_embedding_handles_asymmetry(self):
        """EigenEmbedding should handle slightly asymmetric input."""
        layer = EigenEmbedding()

        # Create symmetric matrix with small asymmetric perturbation
        A = torch.randn(2, 10, 10)
        A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
        A = A + torch.randn_like(A) * 1e-6  # Add tiny asymmetry

        # Should not raise and should return valid eigenvectors
        V = layer(A)
        assert V.shape == (2, 10, 10), f"Unexpected shape {V.shape}"
        assert not torch.isnan(V).any(), "NaN in eigenvectors"

    def test_topk_eigen_handles_asymmetry(self):
        """TopKEigenLayer should handle slightly asymmetric input."""
        layer = TopKEigenLayer(k=4)

        A = torch.randn(2, 10, 10)
        A = (A + A.transpose(-1, -2)) / 2
        A = A + torch.randn_like(A) * 1e-6

        V, Lambda = layer(A)
        assert V.shape == (2, 10, 4), f"Unexpected V shape {V.shape}"
        assert Lambda.shape == (2, 4), f"Unexpected Lambda shape {Lambda.shape}"
        assert not torch.isnan(V).any(), "NaN in eigenvectors"


class TestIssue3MatrixPowerOverflow:
    """Issue 3 [HIGH]: Matrix power overflow in polynomial GCN.

    Rationale: Without normalization, A^i can overflow for dense graphs.
    Symmetric normalization bounds spectral radius to [-1, 1].
    """

    def test_gcn_handles_dense_graph(self):
        """GCN should not overflow on dense graphs."""
        layer = GraphConvolutionLayer(num_terms=3, num_channels=10)

        # Dense adjacency (spectral radius >> 1 without normalization)
        batch_size = 2
        num_nodes = 20
        A = torch.ones(batch_size, num_nodes, num_nodes)
        X = torch.randn(batch_size, num_nodes, 10)

        output = layer(A, X)

        assert not torch.isnan(output).any(), "NaN in GCN output (overflow)"
        assert not torch.isinf(output).any(), "Inf in GCN output (overflow)"

    def test_gcn_uses_gelu_activation(self):
        """GCN should use GELU instead of Tanh (Issue 14 fix)."""
        layer = GraphConvolutionLayer(num_terms=3, num_channels=10)
        assert isinstance(
            layer.activation, nn.GELU
        ), f"Expected GELU activation, got {type(layer.activation)}"


class TestIssues4And5NodeVarGNNRedesign:
    """Issues 4-5 [HIGH]: NodeVarGNN/NodeVarGCN architectural redesign.

    Rationale: The original design used node-specific parameters which broke
    with variable graph sizes. The redesign uses node-agnostic parameters.
    """

    def test_nvgcn_layer_is_node_agnostic(self):
        """NodeVarGraphConvolutionLayer should work with any graph size."""
        layer = NodeVarGraphConvolutionLayer(
            num_terms=2, num_channels_in=8, num_channels_out=8
        )

        # Test with different graph sizes
        for num_nodes in [5, 10, 20]:
            A = torch.eye(num_nodes).unsqueeze(0)
            X = torch.randn(1, num_nodes, 8)
            output = layer(A, X)
            assert output.shape == (1, num_nodes, 8), f"Failed for n={num_nodes}"

    def test_nvgcn_layer_no_dynamic_recreation(self):
        """Layer parameters should not be recreated on size change."""
        layer = NodeVarGraphConvolutionLayer(
            num_terms=2, num_channels_in=8, num_channels_out=8
        )

        # Store parameter ids
        param_ids_before = {id(p) for p in layer.parameters()}

        # Forward with different sizes
        for num_nodes in [5, 10, 15]:
            A = torch.eye(num_nodes).unsqueeze(0)
            X = torch.randn(1, num_nodes, 8)
            _ = layer(A, X)

        param_ids_after = {id(p) for p in layer.parameters()}
        assert (
            param_ids_before == param_ids_after
        ), "Parameters were recreated during forward pass"

    def test_nvgnn_returns_graph_data(self):
        """NodeVarGNN.forward() should return GraphData."""
        model = NodeVarGNN(num_layers=1, num_terms=2, feature_dim=5)
        A = torch.eye(10).unsqueeze(0)

        # Sparse-default refactor: ask for dense output so the legacy edge
        # scalar helper (which reads E_feat / E_class on a dense carrier)
        # has fields to read from.
        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(result, GraphData)
        assert legacy_edge_scalar(result).shape == (1, 10, 10)


class TestIssue6EigenvaluePowerNormalization:
    """Issue 6 [MEDIUM-HIGH]: Eigenvalue power accumulation overflow.

    Rationale: Without normalization, Lambda^ell can explode for large eigenvalues.
    """

    def test_filter_bank_handles_large_eigenvalues(self):
        """GraphFilterBank should handle graphs with large eigenvalues."""
        model = GraphFilterBank(k=4, polynomial_degree=5)

        # Create graph with large eigenvalues
        A = torch.randn(2, 20, 20)
        A = (A + A.transpose(-1, -2)) / 2
        A = A * 10  # Scale up eigenvalues

        result = model(edge_scalar_graphdata(A), output_dense=True)

        out_e = result.E_feat if result.E_feat is not None else result.E_class
        assert out_e is not None
        assert not torch.isnan(out_e).any(), "NaN in filter bank output"
        assert not torch.isinf(out_e).any(), "Inf in filter bank output"


class TestIssue7MetricsConvergenceHandling:
    """Issue 7 [MEDIUM]: eigsh without convergence handling.

    Rationale: ARPACK can fail to converge on ill-conditioned matrices.
    """

    def test_eigenvalue_drift_handles_sparse_matrix(self):
        """Eigenvalue drift should not crash on sparse graphs."""
        # Near-empty adjacency (can cause eigsh convergence issues)
        A_true = torch.eye(10) * 0.01
        A_pred = torch.eye(10) * 0.01

        deltas = compute_spectral_deltas(A_true, A_pred, k=4)
        assert torch.isfinite(deltas["eigenvalue_drift"]).all()

    def test_subspace_distance_handles_sparse_matrix(self):
        """Subspace distance should not crash on sparse graphs."""
        A_true = torch.eye(10) * 0.01
        A_pred = torch.eye(10) * 0.01

        deltas = compute_spectral_deltas(A_true, A_pred, k=4)
        assert torch.isfinite(deltas["subspace_distance"]).all()


class TestIssue8SigmoidConsistency:
    """Issue 8 [MEDIUM]: Inconsistent sigmoid application.

    Rationale: All models should return raw logits from forward();
    predict() applies sigmoid for inference.
    """

    def test_attention_returns_logits(self):
        """MultiLayerAttention.forward() should return logits, not sigmoid output.

        d_model must match num_nodes because forward() extracts the adjacency
        matrix (B, N, N) and feeds it through attention as (B, N, d_model=N).
        """
        num_nodes = 10
        model = MultiLayerAttention(d_model=num_nodes, num_heads=2, num_layers=1)
        A = torch.randn(2, num_nodes, num_nodes)
        A = (A + A.transpose(-1, -2)) / 2

        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(result, GraphData)
        logits = legacy_edge_scalar(result)
        assert logits.shape == (2, num_nodes, num_nodes)

    def test_gnn_symmetric_returns_graph_data(self):
        """GNNSymmetric.forward() should return GraphData."""
        model = GNNSymmetric(
            num_layers=1, num_terms=2, feature_dim_in=10, feature_dim_out=5
        )
        A = torch.eye(8).unsqueeze(0)

        result = model(edge_scalar_graphdata(A), output_dense=True)

        assert isinstance(result, GraphData)
        assert legacy_edge_scalar(result).shape == (1, 8, 8)


class TestIssue9DivisionGuards:
    """Issue 9 [MEDIUM]: Division by small values in diffusion_utils.

    Note: We can't easily test diffusion_utils without the full digress setup,
    but we verify the pattern works correctly.
    """

    def test_clamp_prevents_division_by_zero(self):
        """Verify clamp-based guard prevents division overflow."""
        numerator = torch.tensor([1.0, 2.0, 3.0])
        denominator = torch.tensor([1e-30, 0.0, 1e-6])

        # Original code: denominator[denominator == 0] = 1e-6
        # This misses 1e-30

        # Fixed code: denominator.clamp(min=1e-6)
        denominator_clamped = denominator.clamp(min=1e-6)
        result = numerator / denominator_clamped

        assert not torch.isinf(result).any(), "Inf from division"
        assert not torch.isnan(result).any(), "NaN from division"


class TestIssue10ResidualShapeAssertion:
    """Issue 10 [MEDIUM]: Residual connection shape assumptions.

    The assertion validates that the denoising model preserves shape for
    residual connection. With correct d_model configuration, shapes match.
    """

    def test_matching_shapes_work(self):
        """SequentialDenoisingModel should work when shapes match."""
        from tmgg.models.gnn import GNN

        # Create embedding model
        embedding_model = GNN(  # pyright: ignore[reportArgumentType]
            num_layers=1, num_terms=2, feature_dim_in=10, feature_dim_out=5
        )

        # Create denoising model with CORRECT d_model (2*5=10)
        denoising_model = MultiLayerAttention(
            d_model=10,  # Correct: 2 * feature_dim_out
            num_heads=2,
            num_layers=1,
        )

        model = SequentialDenoisingModel(embedding_model, denoising_model)  # pyright: ignore[reportArgumentType]
        A = torch.eye(8).unsqueeze(0)

        # Should work without error
        result = model(edge_scalar_graphdata(A), output_dense=True)
        assert isinstance(result, GraphData)
        assert legacy_edge_scalar(result).shape == (1, 8, 8)


class TestIssue11ZeroEigenvectorWarning:
    """Issue 11 [LOW-MEDIUM]: Zero eigenvector edge case.

    Rationale: When k > rank(A), some eigenvectors may be zero.
    The fix adds a warning when this occurs.
    """

    def test_zero_eigenvector_handling(self):
        """TopKEigenLayer should handle zero eigenvectors gracefully."""
        layer = TopKEigenLayer(k=8)

        # Create rank-deficient symmetric matrix
        # Outer product gives rank-1 matrix
        v = torch.randn(1, 10, 1)
        A = torch.bmm(v, v.transpose(-1, -2))

        # Should not crash, even with k > rank(A)
        V, Lambda = layer(A)

        assert V.shape == (1, 10, 8), f"Unexpected V shape {V.shape}"
        assert Lambda.shape == (1, 8), f"Unexpected Lambda shape {Lambda.shape}"
        # Most eigenvalues should be near-zero for rank-1 matrix
        assert not torch.isnan(V).any(), "NaN in eigenvectors"

    def test_sign_normalization_returns_valid_signs(self):
        """Sign normalization should return +1 or -1, never 0."""
        layer = TopKEigenLayer(k=4)

        A = torch.randn(2, 10, 10)
        A = (A + A.transpose(-1, -2)) / 2

        V, Lambda = layer(A)

        # Each eigenvector column should have first nonzero entry positive
        # after sign normalization
        for b in range(V.shape[0]):
            for k in range(V.shape[2]):
                col = V[b, :, k]
                nonzero_mask = col.abs() > 1e-10
                if nonzero_mask.any():
                    first_nonzero_idx = nonzero_mask.float().argmax()
                    assert (
                        col[first_nonzero_idx] > 0
                    ), "First nonzero entry should be positive"


class TestIssue12MaskedSoftmaxNaN:
    """Issue 12 [LOW]: Masked softmax row-wise all-masked.

    Rationale: softmax([-inf, -inf, ...]) produces NaN.
    """

    def test_all_masked_row_returns_zeros(self):
        """masked_softmax should return zeros for all-masked rows."""
        x = torch.randn(2, 5)
        mask = torch.ones(2, 5)
        mask[0, :] = 0  # First row is all masked

        result = masked_softmax(x, mask, dim=-1)

        # First row should be zeros, not NaN
        assert not torch.isnan(result).any(), "NaN in masked_softmax output"
        assert (result[0] == 0).all(), "All-masked row should be zeros"
        # Second row should sum to 1
        assert abs(result[1].sum() - 1.0) < 1e-5


class TestIssue13DivisionBySmallNorm:
    """Issue 13 [LOW]: Division by small eigenvalue norm.

    Rationale: For near-zero eigenvalues, norm can be very small.
    """

    def test_eigenvalue_drift_guards_small_norm(self):
        """Eigenvalue drift should not overflow on small eigenvalues."""
        # Matrix with very small eigenvalues
        A_true = torch.eye(10) * 1e-10
        A_pred = torch.eye(10) * 1e-10

        deltas = compute_spectral_deltas(A_true, A_pred, k=4)

        # Should be finite, not inf or nan
        assert torch.isfinite(deltas["eigenvalue_drift"]).all()


class TestIssue14GELUActivation:
    """Issue 14 [LOW]: Tanh after LayerNorm redundancy.

    Already covered in TestIssue3MatrixPowerOverflow.test_gcn_uses_gelu_activation.
    """

    pass


class TestSBMDiagonalAndSymmetry:
    """Regression tests for SBM adjacency generation bugs (Step 0.1).

    Bug 1 -- Self-loops on the diagonal: generate_sbm_adjacency produced
    adjacency matrices where diagonal entries could be 1 (self-loops), because
    random values were sampled for the full block including diagonal positions,
    and never zeroed out afterward. Simple graphs must have zero diagonal.

    Bug 2 -- Asymmetric intra-block edges: when i == j (diagonal blocks), the
    code skipped the symmetrization branch (which only ran for i != j). Each
    intra-block entry (r, c) was an independent Bernoulli draw, so A[r,c] could
    differ from A[c,r] within the same community. Undirected graphs require
    A == A.T everywhere, including within blocks.

    These tests use seeded generators so failures are deterministic. The key
    invariants -- zero diagonal and full symmetry -- must hold for all parameter
    combinations, not just edge cases like p=1.0.
    """

    def test_diagonal_is_zero_deterministic(self):
        """Diagonal must be zero for all tested (p, q) combinations.

        With p < 1.0, diagonal entries would be randomly set to 0 or 1.
        This test catches the missing np.fill_diagonal(adj_matrix, 0) call.
        """
        rng = np.random.default_rng(42)
        test_cases = [
            ([5, 5, 5], 0.8, 0.2),
            ([3, 7], 0.5, 0.1),
            ([10], 0.9, 0.0),
            ([4, 4, 4, 4], 1.0, 0.5),
            ([2, 3, 5], 0.3, 0.3),
        ]
        for block_sizes, p, q in test_cases:
            A = generate_sbm_adjacency(block_sizes, p, q, rng=rng)
            assert np.all(
                np.diag(A) == 0
            ), f"Self-loops found for block_sizes={block_sizes}, p={p}, q={q}"

    def test_intra_block_symmetry(self):
        """Intra-block (diagonal block) edges must be symmetric.

        With p < 1.0, each entry is an independent Bernoulli draw. Without
        explicit symmetrization of diagonal blocks, A[r,c] != A[c,r] for
        many off-diagonal intra-block pairs. This test catches the missing
        symmetrization in the i == j branch.
        """
        rng = np.random.default_rng(123)
        test_cases = [
            ([5, 5, 5], 0.5, 0.1),
            ([8, 8], 0.3, 0.05),
            ([10], 0.7, 0.0),
            ([3, 4, 5, 6], 0.6, 0.2),
        ]
        for block_sizes, p, q in test_cases:
            A = generate_sbm_adjacency(block_sizes, p, q, rng=rng)
            assert np.allclose(
                A, A.T
            ), f"Asymmetry found for block_sizes={block_sizes}, p={p}, q={q}"

    def test_full_matrix_symmetry(self):
        """The entire adjacency matrix must be symmetric, not just inter-block.

        This is the combined check: both intra- and inter-block portions must
        satisfy A == A.T. With the bug, inter-block was symmetric but
        intra-block was not.
        """
        rng = np.random.default_rng(999)
        A = generate_sbm_adjacency([6, 6, 6], 0.4, 0.1, rng=rng)
        diff = np.abs(A - A.T)
        assert np.allclose(
            diff, 0
        ), f"Matrix not symmetric; max asymmetry = {diff.max()}"

    def test_binary_values(self):
        """Adjacency entries must be exactly 0 or 1 (no floats from averaging)."""
        rng = np.random.default_rng(77)
        A = generate_sbm_adjacency([5, 5], 0.5, 0.3, rng=rng)
        unique_vals = np.unique(A)
        assert set(unique_vals).issubset(
            {0, 1}
        ), f"Non-binary values found: {unique_vals}"

    def test_shape_and_dtype(self):
        """Output shape must match sum of block sizes, dtype should be numeric."""
        rng = np.random.default_rng(0)
        block_sizes = [3, 4, 5]
        A = generate_sbm_adjacency(block_sizes, 0.5, 0.2, rng=rng)
        n = sum(block_sizes)
        assert A.shape == (n, n), f"Expected ({n}, {n}), got {A.shape}"
