"""Utility layers for digress style graph neural networks."""

import torch


def masked_softmax(
    x: torch.Tensor, mask: torch.Tensor | None, **kwargs
) -> torch.Tensor:
    """Compute softmax with optional masking.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor
    mask : torch.Tensor, optional
        Binary mask (1 for valid positions, 0 for masked)
    **kwargs
        Additional arguments for torch.softmax (e.g., dim)

    Returns
    -------
    torch.Tensor
        Softmax output with masked positions set to 0.

    Notes
    -----
    When all positions are masked, returns the raw input ``x`` for
    upstream parity (cvignac/DiGress models/layers.py:41-46). The
    downstream attention output is zeroed by the value-side mask
    either way, so this branch is behaviourally equivalent to
    returning zeros, but matches upstream form for cross-codebase
    auditability. Per-row NaN cleanup below still applies for the
    partial-mask case.
    """
    if mask is None:
        return torch.softmax(x, **kwargs)

    # Branch-free implementation: replaces two data-dependent Python
    # ``if`` checks (``mask.sum() == 0`` early-return and
    # ``nan_mask.any()`` NaN-fixup gate) with unconditional tensor ops.
    # The originals graph-break under ``torch.compile`` because the
    # condition value is a 0-d tensor unknown at trace time.
    #
    # Semantics preserved exactly:
    # 1. Per-element ``-inf`` fill on masked positions, then softmax.
    # 2. NaN cleanup: rows where every position was masked produce
    #    ``softmax([-inf, ..., -inf]) = NaN``; we replace those with 0
    #    unconditionally (was gated on ``nan_mask.any()`` but the gate
    #    was a perf shortcut, not a behaviour change).
    # 3. Upstream parity: when *every* element of ``mask`` is zero, the
    #    upstream cvignac/DiGress implementation returns the raw input
    #    ``x``. We replicate that with a final ``torch.where`` that
    #    selects ``x`` over the cleaned ``result`` when ``mask.sum()``
    #    is zero. Cheap (one elementwise compare + select) and keeps
    #    cross-codebase audit equivalence.
    # ``masked_fill`` requires the mask to be broadcastable to ``x``.
    # Callers pass masks shaped ``(..., heads)`` while ``x`` is
    # ``(..., heads, classes)``; the original ``x[mask == 0] = -inf``
    # form broadcast implicitly across the trailing class dim. We
    # restore that by appending singleton dims to the mask until it
    # matches ``x``'s rank, then fill. ``x.dim() - mask.dim()`` is a
    # Python int known at ``torch.compile`` trace time, so the reshape
    # folds to a fixed sequence of metadata ops in the FX graph.
    mask_b = mask.reshape(mask.shape + (1,) * (x.dim() - mask.dim()))
    x_masked = x.masked_fill(mask_b == 0, -float("inf"))
    result = torch.softmax(x_masked, **kwargs)
    result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
    all_masked = mask.sum() == 0
    return torch.where(all_masked, x, result)
