"""Train/val/test index splitting by ratio.

This helper lives with the datamodule generation code because those callers
are the only in-repo consumers of the split contract.
"""

from __future__ import annotations

import numpy as np


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``n`` indices into train/val/test by ratio.

    Parameters
    ----------
    n
        Total number of samples.
    train_ratio
        Fraction for training.
    val_ratio
        Fraction for validation. Remainder goes to test.
    seed
        Random seed for the permutation.

    Returns
    -------
    tuple of np.ndarray
        ``(train_idx, val_idx, test_idx)`` index arrays.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    return (
        indices[:n_train],
        indices[n_train : n_train + n_val],
        indices[n_train + n_val :],
    )
