"""Categorical sampling and masking utilities for graph diffusion."""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_data_fields import FieldName
from tmgg.data.datasets.graph_types import DenseGraphState


def _normalise_unnormalised_posterior(
    unnorm: torch.Tensor,
    *,
    zero_floor: float = 1e-5,
) -> torch.Tensor:
    """Normalise an unnormalised posterior PMF with the upstream zero-row floor.

    Mirrors the renormalisation half of the upstream-DiGress canonical
    pattern in ``sample_p_zs_given_zt``
    (``DiGress/src/diffusion_model_discrete.py:629-639``):

    .. code-block:: python

        unnormalized[unnormalized.sum(dim=-1) == 0] = 1e-5
        prob = unnormalized / unnormalized.sum(dim=-1, keepdim=True)

    The "row whose mass sums to zero" branch sets the *entire* row to
    ``zero_floor``, not just the divisor, so the resulting PMF is
    uniform on degenerate rows. On healthy posteriors the branch is
    inert.

    Used by both :func:`_sample_from_unnormalised_posterior` (sampling
    path) and ``CategoricalNoiseProcess._posterior_probabilities_marginalised``
    (VLB / KL probabilities path); a single primitive guarantees the
    two paths agree on the floor convention.

    Parameters
    ----------
    unnorm
        Unnormalised PMF; the last axis is the class axis. Every entry
        must be non-negative -- the helper raises a clear
        :class:`ValueError` rather than calling ``abs()`` so a real bug
        surfaces loudly per CLAUDE.md.
    zero_floor
        Floor mass written into a row whose unnormalised sum is zero,
        producing a uniform PMF for that row after normalisation.
        Defaults to ``1e-5`` to match upstream DiGress exactly.

    Returns
    -------
    torch.Tensor
        Normalised PMF with the same shape as ``unnorm``; every row
        sums to ``1`` modulo float roundoff.
    """
    # Non-negativity sanity check. Functionally an assert: the producer
    # (``_mix_with_limit`` and the marginalised-KL path) is provably
    # non-negative, so this only fires on a real upstream bug. Guarded by
    # ``__debug__`` so production runs (Python -O / PYTHONOPTIMIZE=1) skip
    # the bool(.all()) sync that fires twice per step (X, E).
    if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
        if not (unnorm >= 0).all():
            raise ValueError(
                "_normalise_unnormalised_posterior: input contains negative "
                "probability mass; cannot sample. Check upstream computation."
            )
    # Upstream uniform-fallback for zero-mass rows (parity #28 / D-5).
    fixed = unnorm.clone()
    row_sum = fixed.sum(dim=-1)
    zero_rows = row_sum == 0
    if zero_rows.any():
        fixed[zero_rows] = zero_floor
    return fixed / fixed.sum(dim=-1, keepdim=True)


def _sample_from_unnormalised_posterior(
    unnorm: torch.Tensor,
    *,
    zero_floor: float = 1e-5,
) -> torch.Tensor:
    """Normalise an unnormalised posterior PMF and sample categorical indices.

    Thin wrapper around :func:`_normalise_unnormalised_posterior` plus
    a flat ``torch.multinomial`` draw, matching the upstream-DiGress
    reverse-step pattern in ``sample_p_zs_given_zt``
    (``DiGress/src/diffusion_model_discrete.py:629-644``).

    Parameters
    ----------
    unnorm
        Unnormalised PMF; the last axis is the class axis. See
        :func:`_normalise_unnormalised_posterior` for the non-negativity
        contract.
    zero_floor
        Floor mass written into a row whose unnormalised sum is zero,
        producing a uniform PMF for that row after normalisation.
        Defaults to ``1e-5`` to match upstream DiGress exactly.

    Returns
    -------
    torch.Tensor
        Sampled class indices with shape ``unnorm.shape[:-1]``.
    """
    prob = _normalise_unnormalised_posterior(unnorm, zero_floor=zero_floor)
    flat_prob = prob.reshape(-1, prob.size(-1))
    sampled = flat_prob.multinomial(1)
    return sampled.reshape(prob.shape[:-1])


def sample_discrete_features(
    probX: torch.Tensor,
    probE: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    field: FieldName | None = None,
    zero_floor: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample node and edge class indices from per-position categorical PMFs.

    Routes the multinomial draw through
    :func:`_sample_from_unnormalised_posterior` so the upstream-DiGress
    "row-sum-zero -> uniform" floor (parity #28 / D-5 / W4-1) is the
    single way categorical sampling happens in tmgg. Masked rows are
    overwritten with a uniform PMF before the helper is called, so the
    floor stays inert on the active reverse path; it only fires if a
    caller passes a strictly-zero PMF row through.

    Parameters
    ----------
    probX : torch.Tensor
        Node class PMFs, shape ``(bs, n, dx_out)``.
    probE : torch.Tensor
        Edge class PMFs, shape ``(bs, n, n, de_out)``.
    node_mask : torch.Tensor
        Boolean node-validity mask, shape ``(bs, n)``.
    field : FieldName | None
        Informational only. Documents which split feature field of a
        :class:`tmgg.data.datasets.graph_types.GraphData` subtype the
        caller intends to populate with the returned draw; the helper
        is field-neutral.
    zero_floor
        Floor passed to :func:`_sample_from_unnormalised_posterior`.
        Defaults to ``1e-5`` to match upstream DiGress.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(X_idx, E_idx)`` integer class indices with shapes
        ``(bs, n)`` and ``(bs, n, n)`` respectively. Callers that need
        one-hot tensors wrap the result via
        :func:`torch.nn.functional.one_hot`.
    """
    _ = field  # Informational only.
    # Clone to avoid in-place mutation of gradient-tracked tensors
    probX = probX.clone()
    probE = probE.clone()

    bs, n, _ = probX.shape
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Single-primitive multinomial draw (parity #28 / D-5 / W4-1).
    X_t = _sample_from_unnormalised_posterior(probX, zero_floor=zero_floor)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    E_t = _sample_from_unnormalised_posterior(probE, zero_floor=zero_floor)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return X_t, E_t


def compute_posterior_distribution_per_x0(
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
    *,
    field: FieldName,
) -> torch.Tensor:
    """Per-x0 reverse posterior tensor — upstream-DiGress marginalisation form.

    Returns ``p(z_s = k | z_t, x_0 = c)`` for every position, every
    ``x_0`` class ``c``, and every ``z_s`` class ``k``. The caller
    marginalises over ``x_0`` by contracting along ``c`` against the
    model's predicted ``p(x_0 = c | z_t)``. Mirrors upstream
    ``compute_batched_over0_posterior_distribution`` in
    ``DiGress/src/diffusion/diffusion_utils.py``.

    Parameters
    ----------
    M_t
        Noisy one-hot at timestep ``t``. Shape ``(bs, n, d)`` for nodes
        or ``(bs, n, n, d)`` for edges.
    Qt_M, Qsb_M, Qtb_M
        Single-step, signal-retention, and cumulative transition
        kernels. All ``(bs, d, d)``.
    field
        :data:`FieldName` naming which GraphData field the caller is
        sampling for (``"X_class"`` for nodes, ``"E_class"`` for
        edges). The math is field-neutral — it operates on ``M_t``
        regardless of semantics — so the parameter is informational
        today. It becomes load-bearing in Wave 9 when the legacy
        ``X`` / ``E`` aliases are removed and callers must explicitly
        name every field they sample.

    Returns
    -------
    torch.Tensor
        Shape ``(bs, N, d, d)`` where ``N = n`` (nodes) or ``n*n``
        (edges), the third axis indexes ``x_0`` classes ``c``, and the
        fourth indexes ``z_s`` classes ``k``.
    """
    _ = field  # Informational only during the transition.
    M_t_flat = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # (bs, N, d)
    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)
    left_term = M_t_flat @ Qt_M_T  # (bs, N, d) -- common across x0 classes
    left_term = left_term.unsqueeze(2)  # (bs, N, 1, d)

    right_term = Qsb_M.unsqueeze(1)  # (bs, 1, d, d) -- per-x0 signal kernel
    numerator = left_term * right_term  # (bs, N, d, d)

    # Per-x0 normaliser z_t @ Qtb_M[c, :].T summed over the z_t support.
    # Equivalent to (Qtb_M @ M_t.T).T evaluated per-x0 row.
    M_t_transposed = M_t_flat.transpose(-1, -2)  # (bs, d, N)
    prod = (Qtb_M @ M_t_transposed).transpose(-1, -2)  # (bs, N, d)
    denominator = prod.unsqueeze(-1)  # (bs, N, d, 1)
    # Numerical floor for masked / impossible classes; matches upstream.
    denominator = denominator.clone()
    denominator[denominator == 0] = 1e-6

    return numerator / denominator


def compute_posterior_distribution(
    M: torch.Tensor,
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
    *,
    field: FieldName,
) -> torch.Tensor:
    """Direct reverse-posterior tensor used when ``x_0`` is already one-hot.

    Computes ``xt @ Qt.T * x_0 @ Qsb / (x_0 @ Qtb @ xt.T)`` — the
    DiGress Bayes-rule form that treats ``M`` (``x_0``) as a
    distribution over classes (commonly the one-hot clean target, or
    the softmax of a model prediction).

    Parameters
    ----------
    M, M_t
        Clean and noisy categorical tensors. ``M`` is either a one-hot
        target or a soft prediction; ``M_t`` is the noisy state at
        timestep ``t``. Shapes ``(bs, n, d)`` for node fields,
        ``(bs, n, n, d)`` for edge fields.
    Qt_M, Qsb_M, Qtb_M
        Single-step, signal-retention, and cumulative transition
        kernels. All ``(bs, d, d)``.
    field
        :data:`FieldName` naming which GraphData field the caller is
        sampling for (``"X_class"`` for nodes, ``"E_class"`` for
        edges). Informational today; see
        :func:`compute_posterior_distribution_per_x0` for the rationale.
    """
    _ = field  # Informational only during the transition.
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(
        torch.float32
    )  # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

    left_term = M_t @ Qt_M_T  # (bs, N, d)
    right_term = M @ Qsb_M  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    # Upstream parity (compute_posterior_distribution, diffusion_utils.py:269-290)
    # divides without any guard and relies on mask_distributions downstream
    # to overwrite masked positions before KL. We removed mask_distributions
    # (commit 59e9593f), so handle the masked-position cleanup here:
    #   - At valid positions (both M and M_t carry one-hot mass), assert
    #     denom > 0 per CLAUDE.md fail-loud — a zero here is a real bug.
    #   - At structural-zero positions (M or M_t zeroed by padding/diagonal),
    #     replace denom with 1 so the division yields 0 (since product is
    #     also zero there) instead of NaN. Downstream masking discards them.
    valid = (M.sum(dim=-1) > 0) & (M_t.sum(dim=-1) > 0)  # (bs, N)
    # Degenerate-denominator sanity assert. Validation-only path
    # (every val_check_interval), but the ``valid.any()`` host branch
    # plus inner ``.item()`` calls each sync the GPU stream. Guarded by
    # ``__debug__`` so production runs (Python -O / PYTHONOPTIMIZE=1)
    # skip the check; ``torch.where`` below already neutralises masked
    # positions.
    if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
        if valid.any():
            valid_denom = denom[valid]
            assert (valid_denom > 0).all(), (
                f"compute_posterior_distribution: degenerate posterior denominator "
                f"at valid (non-masked) position (field={field}, "
                f"min_at_valid={valid_denom.min().item():.3e}, "
                f"valid_count={int(valid.sum().item())}, "
                f"zero_at_valid={int((valid_denom == 0).sum().item())})"
            )
    denom = torch.where(valid, denom, torch.ones_like(denom))

    prob = product / denom.unsqueeze(-1)  # (bs, N, d)

    return prob


def sample_discrete_feature_noise(
    x_limit: torch.Tensor,
    e_limit: torch.Tensor,
    node_mask: torch.Tensor,
    y_limit: torch.Tensor | None = None,
) -> DenseGraphState:
    """Sample a one-hot categorical prior :class:`DenseGraphState` from stationary PMFs.

    Returns
    -------
    DenseGraphState
        Dense + state instance with ``X_class`` and ``E_class``
        populated as one-hot samples drawn from the supplied stationary
        PMFs and masked against ``node_mask``.
    """
    bs, n_max = node_mask.shape
    x_limit = x_limit.to(node_mask.device)[None, None, :].expand(bs, n_max, -1)
    e_limit = e_limit.to(node_mask.device)[None, None, None, :].expand(
        bs, n_max, n_max, -1
    )
    ux_idx = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    ue_idx = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)

    if y_limit is None or y_limit.numel() == 0:
        uy = torch.empty((bs, 0), device=node_mask.device)
    else:
        y_limit = y_limit.to(node_mask.device)[None, :].expand(bs, -1)
        uy_idx = y_limit.multinomial(1).squeeze(-1)
        uy = F.one_hot(uy_idx, num_classes=y_limit.shape[-1]).float()

    long_mask = node_mask.long()
    ux_idx = ux_idx.type_as(long_mask)
    ue_idx = ue_idx.type_as(long_mask)
    # Note: ``uy`` deliberately keeps its Float dtype from line 349
    # (``F.one_hot(...).float()``). Previously this line also applied
    # ``uy.type_as(long_mask)``, which silently coerced a Float one-hot
    # vector to Long by piggy-backing on ``long_mask`` (introduced for the
    # integer indices ``ux_idx`` / ``ue_idx``). That cast was vestigial:
    # ``uy`` is a one-hot probability channel, not an index, and the
    # downstream ``torch.cat([y, extra_y])`` in the GraphTransformer
    # requires Float because ``extra_y`` is Float. The Long form happened
    # to work in eager mode via silent widening, but it diverged from the
    # training-path producer (``GraphData.from_pyg_batch``) which emits
    # Float ``y``. The dtype split forced ``torch.compile`` to retrace
    # per-call; keeping ``uy`` Float here makes both paths emit the same
    # canonical dtype.

    ux_one_hot = F.one_hot(ux_idx, num_classes=x_limit.shape[-1]).float()
    ue_one_hot = F.one_hot(ue_idx, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(ue_one_hot)
    indices = torch.triu_indices(
        row=ue_one_hot.size(1), col=ue_one_hot.size(2), offset=1
    )
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    ue_one_hot = ue_one_hot * upper_triangular_mask
    ue_one_hot = ue_one_hot + torch.transpose(ue_one_hot, 1, 2)

    # Symmetry sanity check: ``ue_one_hot`` is mathematically symmetric by
    # construction (upper-tri masked + its own transpose, two lines above).
    # The ``.all()`` host branch syncs the GPU stream every sample-loop
    # init. Guarded by ``__debug__`` so production runs (Python -O /
    # PYTHONOPTIMIZE=1) skip the sync.
    if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
        if not (torch.transpose(ue_one_hot, 1, 2) == ue_one_hot).all():
            raise AssertionError("Edge noise is not symmetric")

    # Derive ``num_nodes_per_graph`` from the boolean ``node_mask`` (1 ==
    # valid). DenseGraphState exposes ``node_mask`` as a cached_property
    # over ``num_nodes_per_graph``, so the carrier identity is preserved
    # by construction.
    num_nodes_per_graph = node_mask.long().sum(dim=1)
    return DenseGraphState(
        num_nodes_per_graph=num_nodes_per_graph,
        y=uy,
        X_class=ux_one_hot,
        E_class=ue_one_hot,
    ).mask()
