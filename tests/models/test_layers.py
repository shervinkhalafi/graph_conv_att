"""Tests for utility layers."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tmgg.models.layers import masked_softmax


class TestMaskedSoftmax:
    """Test masked_softmax function."""

    def test_no_mask(self):
        """Test softmax without mask."""
        x = torch.randn(2, 5, 5)

        output = masked_softmax(x, None, dim=-1)
        expected = torch.softmax(x, dim=-1)

        assert torch.allclose(output, expected)

    def test_full_mask(self):
        """Test with all positions valid."""
        x = torch.randn(2, 5, 5)
        mask = torch.ones_like(x)

        output = masked_softmax(x, mask, dim=-1)
        expected = torch.softmax(x, dim=-1)

        assert torch.allclose(output, expected)

    def test_partial_mask(self):
        """Test with some positions masked."""
        x = torch.randn(1, 5)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float)

        output = masked_softmax(x, mask, dim=-1)

        # Check masked positions have near-zero probability
        assert output[0, 3] < 1e-6
        assert output[0, 4] < 1e-6

        # Check unmasked positions sum to 1
        assert torch.allclose(output[0, :3].sum(), torch.tensor(1.0))

    def test_empty_mask(self):
        """All-masked branch returns the raw input (upstream parity #25).

        Upstream (cvignac/DiGress models/layers.py:41-46) returns ``x``
        when every position is masked. Behaviourally equivalent to
        returning zeros since the value-side mask zeros the downstream
        attention output, but pinning identity here guards against
        regression away from upstream form.
        """
        x = torch.randn(1, 5)
        mask = torch.zeros_like(x)

        output = masked_softmax(x, mask, dim=-1)

        assert output.shape == x.shape
        assert torch.equal(output, x)

    def test_2d_mask_on_3d_tensor(self):
        """Test 2D mask applied to 3D tensor."""
        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, seq_len)

        # Mask that masks out last position
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, -1] = 0

        output = masked_softmax(x, mask, dim=-1)

        # Check last column has near-zero values
        assert torch.all(output[:, :, -1] < 1e-6)

        # Check each row sums to 1
        row_sums = output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums))

    @given(
        batch_size=st.integers(min_value=1, max_value=5),
        seq_len=st.integers(min_value=2, max_value=10),
    )
    def test_property_probability_distribution(self, batch_size, seq_len):
        """Property test: masked softmax output is a valid probability distribution."""
        x = torch.randn(batch_size, seq_len)

        # Generate random mask ensuring each row has at least one unmasked position
        mask = torch.zeros((batch_size, seq_len))
        for i in range(batch_size):
            # Randomly select positions to unmask, ensuring at least one
            num_unmasked = torch.randint(1, seq_len + 1, (1,)).item()
            indices = torch.randperm(seq_len)[:num_unmasked]
            mask[i, indices] = 1

        output = masked_softmax(x, mask, dim=-1)

        # Check all values are non-negative
        assert torch.all(output >= 0)

        # Check all values are <= 1
        assert torch.all(output <= 1)

        # Check rows sum to 1 (allowing for numerical errors)
        row_sums = output.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

        # Check masked positions have near-zero probability
        assert torch.all(output[mask == 0] < 1e-6)

    @given(
        shape=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=4),
        dim=st.integers(
            min_value=-1, max_value=-1
        ),  # Only test last dimension for simplicity
    )
    def test_property_shape_preservation(self, shape, dim):
        """Property test: output shape matches input shape."""
        x = torch.randn(*shape)
        mask = torch.ones_like(x)

        output = masked_softmax(x, mask, dim=dim)

        assert output.shape == x.shape


class TestMaskedSoftmaxNaNHandling:
    """Test masked_softmax per-row NaN handling (formerly in digress/layers.py).

    The DiGress transformer now imports masked_softmax from the canonical
    location (tmgg.models.layers.masked_softmax). These tests verify the
    NaN handling behavior (P2.8 fix: softmax([-inf, ...]) = NaN replaced
    with zeros).
    """

    def test_per_row_nan_handling(self) -> None:
        """Rows where all positions are masked produce zeros, not NaN.

        Starting state: 2-row input where row 0 is fully masked and
        row 1 has one valid position.
        Invariant: no NaN in output; fully-masked row is all zeros;
        valid row sums to 1.
        """
        from tmgg.models.layers.masked_softmax import masked_softmax

        x = torch.randn(2, 5)
        mask = torch.zeros(2, 5)
        mask[1, 2] = 1  # only row 1, position 2 is valid

        output = masked_softmax(x, mask, dim=-1)

        assert not torch.isnan(output).any(), "NaN found in masked_softmax output"
        assert torch.all(output[0] == 0), "Fully-masked row should be all zeros"
        assert output[1, 2] == pytest.approx(
            1.0
        ), "Single valid position should get all weight"

    def test_3d_per_row_nan_handling(self) -> None:
        """Per-row NaN handling works on attention-shaped (bs, n, n) tensors.

        Starting state: (2, 3, 4) attention tensor where batch 0, row 1
        is fully masked.
        Invariant: that specific row is zeros; other rows with valid
        positions sum to 1.
        """
        from tmgg.models.layers.masked_softmax import masked_softmax

        x = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, 4)
        mask[0, 1, :] = 0  # fully mask row 1 in batch 0

        output = masked_softmax(x, mask, dim=-1)

        assert not torch.isnan(output).any(), "NaN found in output"
        assert torch.all(output[0, 1] == 0), "Fully-masked row should be zeros"
        # Other rows should be valid probability distributions
        assert torch.allclose(output[0, 0].sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(output[1, 0].sum(), torch.tensor(1.0), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
