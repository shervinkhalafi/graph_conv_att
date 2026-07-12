"""Cross-yaml consistency check for the discrete-diffusion repro panel.

For every ``configs/experiment/discrete_*_repro.yaml`` we resolve the
config through Hydra and assert that the four class-cardinality knobs
agree pairwise:

::

    model.input_dims.X        == model.output_dims_x_class == model.noise_process.x_classes
    model.input_dims.E        == model.output_dims_e_class == model.noise_process.e_classes
    model.output_dims.X       == model.output_dims_x_class
    model.output_dims.E       == model.output_dims_e_class

Plus the structural invariant that ``model.num_nodes`` is a positive
integer.

Why this test exists
--------------------
We have repeatedly hit Modal-only failures where:

- A molecular yaml inherits from ``discrete_sbm_official`` (which sets
  ``noise_process.x_classes=2``, ``e_classes=2`` for the abstract-
  graph SBM convention).
- The molecular yaml overrides ``input_dims``/``output_dims`` to the
  vocab cardinality (e.g. 4/5 for QM9, 8/5 for MOSES, 12/5 for
  GuacaMol) but forgets to override ``noise_process``.
- ``CategoricalNoiseProcess.initialize_from_data`` then asserts on the
  per-class histogram width (4-vs-2 mismatch) ~3 minutes into a
  billed Modal cold-start.

This test fails fast at unit-test time so the next person who adds a
panel yaml cannot ship that exact bug class to production. ``planar``
fixed via 05e8a406; QM9/MOSES/GuacaMol fixed via the commit landing
this test.

Testing strategy
----------------
- Iterate every ``discrete_*_repro.yaml`` matching the panel naming.
- Compose the full config tree through Hydra so all interpolations and
  ``override /models/discrete@model:`` overlays resolve.
- Read out the four cardinality knobs and assert the chain.
- Pin in a single test rather than per-file because the failure cost
  is global (any new panel inherits the same drift risk) and the
  per-file overhead is minimal (~50ms each).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPRO_NAMES = (
    "discrete_sbm_vignac_repro",
    "discrete_planar_digress_repro",
    "discrete_qm9_digress_repro",
    "discrete_moses_digress_repro",
    "discrete_guacamol_digress_repro",
)


@pytest.fixture(autouse=True)
def _hydra_global_clear():
    """Hydra global state survives across tests; clear before + after."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    yield
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _config_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tmgg"
        / "experiments"
        / "exp_configs"
    )


@pytest.mark.parametrize("name", REPRO_NAMES)
def test_repro_yaml_class_cardinality_chain_matches(name: str) -> None:
    """The data → noise_process → model dim chain must agree on K."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(_config_dir()),
    ):
        cfg = compose(
            config_name="base_config_discrete_diffusion_generative",
            overrides=[f"+experiment={name}"],
        )

    model = cfg.model
    nm = model.model
    np_ = model.noise_process

    x_in = int(nm.input_dims.X)
    e_in = int(nm.input_dims.E)
    x_out_total = int(nm.output_dims.X)
    e_out_total = int(nm.output_dims.E)
    x_out_class = int(nm.output_dims_x_class)
    e_out_class = int(nm.output_dims_e_class)
    x_np = int(np_.x_classes)
    e_np = int(np_.e_classes)

    # Atom / node-class chain.
    assert x_in == x_out_class, (
        f"[{name}] model.input_dims.X={x_in} disagrees with "
        f"model.output_dims_x_class={x_out_class}"
    )
    assert x_out_total == x_out_class, (
        f"[{name}] model.output_dims.X={x_out_total} disagrees with "
        f"model.output_dims_x_class={x_out_class}"
    )
    assert x_np == x_out_class, (
        f"[{name}] model.noise_process.x_classes={x_np} disagrees with "
        f"model.output_dims_x_class={x_out_class} — typical fingerprint "
        "of a yaml that inherited the SBM preset and forgot to override "
        "the noise_process block."
    )

    # Bond / edge-class chain.
    assert e_in == e_out_class, (
        f"[{name}] model.input_dims.E={e_in} disagrees with "
        f"model.output_dims_e_class={e_out_class}"
    )
    assert e_out_total == e_out_class, (
        f"[{name}] model.output_dims.E={e_out_total} disagrees with "
        f"model.output_dims_e_class={e_out_class}"
    )
    assert e_np == e_out_class, (
        f"[{name}] model.noise_process.e_classes={e_np} disagrees with "
        f"model.output_dims_e_class={e_out_class}"
    )

    # Sanity invariant on graph-size knob.
    assert (
        int(model.num_nodes) > 0
    ), f"[{name}] model.num_nodes={model.num_nodes!r} must be a positive int"
