"""Unit tests for eigenstructure analysis functions.

Test rationale:
    These tests verify the correctness of spectral analysis functions:
    1. Procrustes rotation computes valid rotation matrices
    2. Rotation angles are extracted correctly from rotation matrices
    3. Residuals measure alignment quality accurately
    4. Batch processing works correctly

Invariants:
    - Rotation matrices are orthogonal (R @ R.T = I)
    - Rotation angle is in [0, π]
    - Residual is non-negative
    - Identity rotation has angle 0
"""

import math

import pytest
import torch

from tmgg.experiments.eigenstructure_study.analyzer import compute_principal_angles
from tmgg.utils.spectral.subspace import (
    compute_procrustes_rotation,
    compute_procrustes_rotation_batch,
)


@pytest.fixture
def identity_eigenvectors() -> torch.Tensor:
    """Create identity matrix as eigenvectors (trivial case)."""
    return torch.eye(10)


@pytest.fixture
def random_eigenvectors() -> tuple[torch.Tensor, torch.Tensor]:
    """Create two random orthogonal matrices."""
    n = 20
    # Create random orthogonal matrices via QR decomposition
    A1 = torch.randn(n, n)
    Q1, _ = torch.linalg.qr(A1)

    A2 = torch.randn(n, n)
    Q2, _ = torch.linalg.qr(A2)

    return Q1, Q2


@pytest.fixture
def rotated_eigenvectors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create eigenvectors where V2 = V1 @ R for a known rotation R.

    Returns:
        (V1, V2, R) where V2[:, -k:] = V1[:, -k:] @ R
    """
    n = 10
    k = 4

    # Create V1 as random orthogonal
    A = torch.randn(n, n)
    V1, _ = torch.linalg.qr(A)

    # Create a known rotation in k dimensions
    # Use a simple rotation angle
    theta = 0.3  # radians
    R_2d = torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    R = torch.eye(k)
    R[:2, :2] = R_2d  # Embed 2D rotation in k-dimensional space

    # Apply rotation to top-k eigenvectors
    V2 = V1.clone()
    V2[:, -k:] = V1[:, -k:] @ R

    return V1, V2, R


class TestProcrustesRotation:
    """Tests for Procrustes rotation analysis."""

    def test_identity_rotation_has_zero_angle(
        self, identity_eigenvectors: torch.Tensor
    ) -> None:
        """Identical eigenvector matrices have zero rotation angle.

        When comparing a matrix to itself, the optimal rotation is identity
        and the angle should be (close to) zero.
        """
        V = identity_eigenvectors
        k = 5

        result = compute_procrustes_rotation(V, V, k)

        assert result["angle"].item() < 1e-5
        assert result["residual"].item() < 1e-5
        # Rotation matrix should be close to identity
        assert torch.allclose(result["rotation"], torch.eye(k), atol=1e-5)

    def test_rotation_matrix_is_orthogonal(
        self, random_eigenvectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Procrustes rotation produces an orthogonal matrix."""
        V1, V2 = random_eigenvectors
        k = 8

        result = compute_procrustes_rotation(V1, V2, k)
        R = result["rotation"]

        # R @ R.T should be identity
        assert torch.allclose(R @ R.T, torch.eye(k), atol=1e-5)
        # R.T @ R should also be identity
        assert torch.allclose(R.T @ R, torch.eye(k), atol=1e-5)

    def test_angle_is_in_valid_range(
        self, random_eigenvectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Rotation angle is in [0, π]."""
        V1, V2 = random_eigenvectors
        k = 5

        result = compute_procrustes_rotation(V1, V2, k)
        angle = result["angle"].item()

        assert 0 <= angle <= math.pi

    def test_residual_is_non_negative(
        self, random_eigenvectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Residual (Frobenius norm) is non-negative."""
        V1, V2 = random_eigenvectors
        k = 5

        result = compute_procrustes_rotation(V1, V2, k)

        assert result["residual"].item() >= 0

    def test_known_rotation_is_recovered(
        self, rotated_eigenvectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Procrustes correctly recovers a known rotation.

        When V2 = V1 @ R, Procrustes finds R^T (the inverse) to align V2 back to V1.
        Thus R_computed @ R_true ≈ I.
        """
        V1, V2, R_true = rotated_eigenvectors
        k = R_true.shape[0]

        result = compute_procrustes_rotation(V1, V2, k)
        R_computed = result["rotation"]

        # Procrustes finds the inverse rotation, so R_computed @ R_true ≈ I
        # May differ by sign/reflection in degenerate cases
        product = R_computed @ R_true
        assert torch.allclose(product, torch.eye(k), atol=1e-4) or torch.allclose(
            product, -torch.eye(k), atol=1e-4
        )

        # Residual should be very small
        assert result["residual"].item() < 1e-4

    def test_larger_rotation_has_larger_angle(self) -> None:
        """Larger rotations produce larger angles.

        When we apply progressively larger rotations, the computed angle
        should increase monotonically.
        """
        n = 10
        k = 4

        # Create base eigenvectors
        A = torch.randn(n, n)
        V1, _ = torch.linalg.qr(A)

        angles_input = [0.1, 0.3, 0.5, 0.8]
        angles_output = []

        for theta in angles_input:
            # Create rotation matrix
            R = torch.eye(k)
            R[0, 0] = math.cos(theta)
            R[0, 1] = -math.sin(theta)
            R[1, 0] = math.sin(theta)
            R[1, 1] = math.cos(theta)

            # Apply rotation
            V2 = V1.clone()
            V2[:, -k:] = V1[:, -k:] @ R

            result = compute_procrustes_rotation(V1, V2, k)
            angles_output.append(result["angle"].item())

        # Angles should be monotonically increasing
        for i in range(len(angles_output) - 1):
            assert angles_output[i] < angles_output[i + 1]


class TestProcrustesRotationBatch:
    """Tests for batched Procrustes rotation."""

    def test_batch_output_shapes(self) -> None:
        """Batch processing produces correct output shapes."""
        batch_size = 5
        n = 10
        k = 4

        # Create batch of random orthogonal matrices
        V1_batch = torch.zeros(batch_size, n, n)
        V2_batch = torch.zeros(batch_size, n, n)

        for i in range(batch_size):
            A1 = torch.randn(n, n)
            V1_batch[i], _ = torch.linalg.qr(A1)

            A2 = torch.randn(n, n)
            V2_batch[i], _ = torch.linalg.qr(A2)

        result = compute_procrustes_rotation_batch(V1_batch, V2_batch, k)

        assert result["angles"].shape == (batch_size,)
        assert result["residuals"].shape == (batch_size,)

    def test_batch_matches_individual(self) -> None:
        """Batch processing gives same results as individual processing."""
        batch_size = 3
        n = 8
        k = 3

        V1_batch = torch.zeros(batch_size, n, n)
        V2_batch = torch.zeros(batch_size, n, n)

        for i in range(batch_size):
            A1 = torch.randn(n, n)
            V1_batch[i], _ = torch.linalg.qr(A1)

            A2 = torch.randn(n, n)
            V2_batch[i], _ = torch.linalg.qr(A2)

        # Batch computation
        batch_result = compute_procrustes_rotation_batch(V1_batch, V2_batch, k)

        # Individual computation
        for i in range(batch_size):
            individual_result = compute_procrustes_rotation(V1_batch[i], V2_batch[i], k)

            assert torch.isclose(
                batch_result["angles"][i], individual_result["angle"], atol=1e-5
            )
            assert torch.isclose(
                batch_result["residuals"][i], individual_result["residual"], atol=1e-5
            )


class TestPrincipalAnglesConsistency:
    """Tests for consistency between principal angles and Procrustes."""

    def test_identical_subspaces_have_zero_angles(
        self, identity_eigenvectors: torch.Tensor
    ) -> None:
        """Identical subspaces have zero principal angles."""
        V = identity_eigenvectors
        k = 5

        angles = compute_principal_angles(V, V, k)

        assert torch.allclose(angles, torch.zeros(k), atol=1e-5)

    def test_principal_angles_in_valid_range(
        self, random_eigenvectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Principal angles are in [0, π/2]."""
        V1, V2 = random_eigenvectors
        k = 5

        angles = compute_principal_angles(V1, V2, k)

        assert (angles >= 0).all()
        assert (angles <= math.pi / 2 + 1e-5).all()
