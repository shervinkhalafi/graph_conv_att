"""Test that custom_beta_schedule_discrete uses the correct p for K edge classes.

Rationale
---------
The original DiGress code hardcodes p=4/5 (for K=5 molecular edge classes).
The default mirrors upstream (K=5 → p=4/5); per-call overrides recover
SBM-binary behaviour (K=2 → p=1/2). Using a K-mismatched schedule
under-noises early diffusion steps.

Starting state: custom_beta_schedule_discrete accepts num_edge_classes.
Invariant: p = 1 - 1/num_edge_classes, so different K produce different betas.
Default value is K=5 (upstream parity #8). Reference: DiGress paper
(Vignac et al. 2023), Eq. 4 and Section 3.3.
"""

import inspect

import numpy as np


def test_binary_edge_schedule_uses_p_half():
    """For K=2 edge classes, p should be 1 - 1/2 = 0.5."""
    from tmgg.diffusion.diffusion_math import custom_beta_schedule_discrete

    betas_k2 = custom_beta_schedule_discrete(500, num_edge_classes=2)
    betas_k5 = custom_beta_schedule_discrete(500, num_edge_classes=5)

    # K=2 should produce different betas than K=5 since p differs
    assert not np.allclose(
        betas_k2, betas_k5
    ), "K=2 and K=5 produced identical beta schedules -- p is likely still hardcoded"


def test_default_matches_upstream_parity():
    """Default num_edge_classes is 5 (upstream parity #8 — recovers p=4/5).

    Upstream's custom_beta_schedule_discrete hardcodes p=4/5 for the
    K=5 molecular case. Our parameterised form 1 - 1/K recovers this
    when K=5; the default is set accordingly. SBM/binary callers must
    pass num_edge_classes=2 explicitly.
    """
    from tmgg.diffusion.diffusion_math import custom_beta_schedule_discrete

    sig = inspect.signature(custom_beta_schedule_discrete)
    default = sig.parameters["num_edge_classes"].default
    assert default == 5, f"Default num_edge_classes is {default}, expected 5"


def test_p_value_matches_formula():
    """Verify that the beta floor changes proportionally with K.

    For K edge classes the stationary probability of a non-null edge type is
    p = 1 - 1/K. The floor beta_first = updates_per_graph / (p * num_edges),
    so larger K yields smaller beta_first (since p is larger). We check that
    K=3 produces a floor between K=2 and K=10.
    """
    from tmgg.diffusion.diffusion_math import custom_beta_schedule_discrete

    betas_k2 = custom_beta_schedule_discrete(500, num_edge_classes=2)
    betas_k3 = custom_beta_schedule_discrete(500, num_edge_classes=3)
    betas_k10 = custom_beta_schedule_discrete(500, num_edge_classes=10)

    # The floor (minimum beta) should decrease as K increases
    floor_k2 = betas_k2.min()
    floor_k3 = betas_k3.min()
    floor_k10 = betas_k10.min()

    assert floor_k2 > floor_k3 > floor_k10, (
        f"Beta floors should decrease with K: "
        f"K=2 floor={floor_k2:.6f}, K=3 floor={floor_k3:.6f}, K=10 floor={floor_k10:.6f}"
    )
