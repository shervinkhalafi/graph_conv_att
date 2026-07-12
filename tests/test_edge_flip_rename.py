"""Regression test: edge-flip noise uses correct names."""


def test_edge_flip_function_exists():
    from tmgg.utils.noising.noise import add_edge_flip_noise

    assert callable(add_edge_flip_noise)


def test_edge_flip_definition_exists():
    from tmgg.utils.noising.noise import (
        EdgeFlipNoise,
    )

    assert EdgeFlipNoise is not None


def test_factory_creates_edge_flip():
    from tmgg.utils.noising.noise import (
        create_noise_definition,
    )

    gen = create_noise_definition("edge_flip")
    assert gen is not None


def test_factory_digress_still_works():
    """Existing 'digress' key must still work (backward compat)."""
    from tmgg.utils.noising.noise import (
        create_noise_definition,
    )

    gen = create_noise_definition("digress")
    assert gen is not None
