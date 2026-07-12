"""Pure mathematical utilities for discrete and continuous diffusion.

Contains noise schedule computation, KL divergence, signal-to-noise ratio,
and tensor manipulation helpers. None of these functions use randomness or
depend on domain-specific graph types.
"""

import math
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional as F


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).sum(dim=-1)


def clip_noise_schedule(
    alphas2: NDArray[np.floating[Any]], clip_value: float = 0.001
) -> NDArray[np.floating[Any]]:
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(
    timesteps: int, s: float = 0.008, raise_to_power: float = 1
) -> NDArray[np.floating[Any]]:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(
    timesteps: int, s: float = 0.008
) -> NDArray[np.floating[Any]]:
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(
    timesteps: int,
    average_num_nodes: int = 50,
    s: float = 0.008,
    num_edge_classes: int = 5,
) -> NDArray[np.floating[Any]]:
    """Cosine beta schedule with edge-class-dependent floor.

    Uses the schedule from https://openreview.net/forum?id=-NEXDKk8gZ with a
    minimum beta that depends on the number of edge classes K. The stationary
    probability of a non-null edge type is ``p = 1 - 1/K``, which sets the
    floor so that early steps produce ~1.2 edge updates per graph.

    Parameters
    ----------
    timesteps : int
        Number of diffusion steps (must be >= 100).
    average_num_nodes : int
        Mean graph size, used to estimate the number of edges.
    s : float
        Offset for the cosine schedule.
    num_edge_classes : int
        Number of categorical edge classes K. For binary (edge/no-edge)
        graphs K=2 gives p=1/2; for molecular datasets K=5 gives p=4/5.
        Default is 5: K=5 is the upstream-parity default — recovers
        ``p = 1 - 1/5 = 4/5`` from upstream's hardcoded molecular case
        (``custom_beta_schedule_discrete``, ``diffusion_utils.py:77-97``).
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    if not timesteps >= 100:
        raise AssertionError(f"timesteps must be >= 100, got {timesteps}")

    p = 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def gaussian_KL(q_mu: torch.Tensor, q_sigma: torch.Tensor) -> torch.Tensor:
    """KL divergence from N(q_mu, q_sigma^2) to the standard normal N(0, 1).

    Parameters
    ----------
    q_mu
        Mean of distribution q.
    q_sigma
        Standard deviation of distribution q.

    Returns
    -------
    torch.Tensor
        KL divergence, summed over all dimensions except the batch dim.
    """
    return sum_except_batch(torch.log(1 / q_sigma) + 0.5 * (q_sigma**2 + q_mu**2) - 0.5)


def cdf_std_gaussian(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def SNR(gamma: torch.Tensor) -> torch.Tensor:
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


def inflate_batch_array(
    array: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape_tuple: tuple[int, ...] = (array.size(0),) + (1,) * (
        len(target_shape) - 1
    )
    return array.view(target_shape_tuple)


def sigma(
    gamma: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(
    gamma: torch.Tensor, target_shape: torch.Size | tuple[int, ...]
) -> torch.Tensor:
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_issues_norm_values(
    gamma: torch.nn.Module,
    norm_val1: float,
    norm_val2: float,
    num_stdevs: int = 8,
) -> None:
    """Check if 1 / norm_value is still larger than 10 * standard deviation."""
    zeros = torch.zeros((1, 1))
    gamma_0 = gamma(zeros)
    sigma_0 = sigma(gamma_0, target_shape=zeros.size()).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sigma_0 * num_stdevs > 1.0 / max_norm_value:
        raise ValueError(
            f"Value for normalization value {max_norm_value} probably too "
            f"large with sigma_0 {sigma_0:.5f} and "
            f"1 / norm_value = {1.0 / max_norm_value}"
        )


def sigma_and_alpha_t_given_s(
    gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)),  # pyright: ignore[reportAttributeAccessIssue]  # F.softplus exists at runtime
        target_size,
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.size(0) - 1, -1, -1)]
