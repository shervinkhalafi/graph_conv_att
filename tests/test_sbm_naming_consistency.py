"""Regression test: SBM parameter naming uses p_intra/p_inter consistently."""

import inspect


def test_generate_sbm_batch_uses_p_intra_p_inter():
    from tmgg.data.datasets.sbm import generate_sbm_batch

    sig = inspect.signature(generate_sbm_batch)
    param_names = list(sig.parameters.keys())
    assert "p_intra" in param_names, f"Expected p_intra in {param_names}"
    assert "p_inter" in param_names, f"Expected p_inter in {param_names}"
    assert "p_in" not in param_names, "p_in should be renamed to p_intra"
    assert "p_out" not in param_names, "p_out should be renamed to p_inter"


def test_generate_sbm_batch_callable():
    """Verify the renamed parameters work."""
    from tmgg.data.datasets.sbm import generate_sbm_batch

    result = generate_sbm_batch(
        num_graphs=2,
        num_nodes=10,
        num_blocks=2,
        p_intra=0.7,
        p_inter=0.1,
        seed=42,
    )
    assert result.shape == (2, 10, 10)
