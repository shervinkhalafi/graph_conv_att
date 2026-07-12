"""Regression tests for mathematical correctness fixes (F16, F17, F18).

F17: LinearPE symmetric mode must produce symmetric output even when the
     internal weight matrix W is not itself symmetric.
F18: RotationNoise must fail loudly when the skew matrix dimension
     does not match the graph dimension.
"""

import pytest
import torch

from tests._helpers.graph_builders import edge_scalar_graphdata


def test_linear_pe_symmetric_output():
    """LinearPE in symmetric mode must produce symmetric output.

    Rationale
    ---------
    The reconstruction formula is V W V^T. If W is not symmetrized before
    use, the product V W V^T is only symmetric when W itself is symmetric --
    which is not guaranteed by `xavier_uniform_` or by gradient updates.
    Setting W to an intentionally asymmetric matrix exposes the bug.
    """
    from tmgg.models.spectral_denoisers import LinearPE

    model = LinearPE(k=4, max_nodes=10, use_bias=False, asymmetric=False)
    # Overwrite W with an intentionally asymmetric matrix.
    with torch.no_grad():
        model.W.copy_(torch.randn(4, 4))

    A = torch.randn(2, 10, 10)
    A = (A + A.transpose(-2, -1)) / 2  # symmetric input
    result = model(edge_scalar_graphdata(A), output_dense=True)
    # LinearPE writes its prediction to E_feat (single-channel scalar). The
    # result is a DenseGraphDistribution, so we read the field directly.
    assert result.E_feat is not None
    adj = result.E_feat.squeeze(-1)
    diff = (adj - adj.transpose(-2, -1)).abs().max().item()
    assert diff < 1e-5, f"LinearPE output asymmetry: {diff}"


def test_rotation_noise_dimension_mismatch():
    """RotationNoise must reject adjacency matrices whose dimension
    does not match the skew matrix.

    The skew-symmetric matrix is created at init with dimension k, but the
    eigendecomposition inside add_noise produces eigenvectors of dimension n
    (the graph size). If k != n, the matrix multiply V @ R silently produces
    wrong results. An explicit assertion catches this early.
    """
    from tmgg.utils.noising.noise import (
        RotationNoise,
    )

    gen = RotationNoise(k=5, seed=42)
    A = torch.eye(8)  # dimension 8 != k=5
    with pytest.raises(ValueError, match="Graph dimension"):
        gen.add_noise(A, eps=0.1)


def test_rotation_noise_matching_dimension():
    """RotationNoise works when dimensions match."""
    from tmgg.utils.noising.noise import (
        RotationNoise,
    )

    gen = RotationNoise(k=5, seed=42)
    A = torch.eye(5)
    result = gen.add_noise(A, eps=0.1)
    assert result.shape == (5, 5)
    # Rotation preserves eigenvalues, so trace should be close to original
    assert abs(result.trace().item() - A.trace().item()) < 1.0


def test_rotation_noise_batch_dimension_mismatch():
    """Batch variant of the dimension check."""
    from tmgg.utils.noising.noise import (
        RotationNoise,
    )

    gen = RotationNoise(k=5, seed=42)
    A = torch.eye(8).unsqueeze(0).expand(3, -1, -1)  # batch of 3, dim 8
    with pytest.raises(ValueError, match="Graph dimension"):
        gen.add_noise(A, eps=0.1)
