"""Guard: SBM config files must use keys the datamodule actually reads.

Rationale: Prior to this fix, configs used ``q_inter`` but
``data_module.py`` reads ``p_inter``, silently falling back to 0.0.
The grid_gaussian/grid_digress/grid_rotation configs intended
q_inter=0.05 but actually produced 0.0 inter-block edges.
"""

import glob

import yaml

# Keys the SBM datamodule dispatch actually reads from graph_config
# (data_module.py plus graph_generation.py: _resolve_sbm_partition_mode,
# _generate_partitioned_sbm_split, and _resolve_sbm_batch_kwargs).
VALID_SBM_KEYS = {
    "num_nodes",
    "p_intra",
    "p_inter",
    "num_blocks",
    "min_blocks",
    "max_blocks",
    "min_block_size",
    "max_block_size",
    "block_sizes",
    "partition_mode",
    "num_train_partitions",
    "num_test_partitions",
    # Paper's synthetic-SBM diversity knobs (Appendix B.2.3, Eq. 28), forwarded
    # to generate_sbm_batch's diverse mode by _resolve_sbm_batch_kwargs.
    "block_size_alpha",
    "diversity",
}

DEAD_SBM_KEYS = {"q_inter", "p_out", "q_intra"}


def test_no_dead_keys_in_sbm_configs():
    """No SBM data config should use q_inter, p_out, or q_intra."""
    config_dir = "src/tmgg/experiments/exp_configs/data/"
    violations = []
    for path in glob.glob(f"{config_dir}*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            continue
        # Check top-level and nested dataset_config
        for section in [cfg, cfg.get("dataset_config", {})]:
            if section is None:
                continue
            found = [k for k in DEAD_SBM_KEYS if k in section]
            if found:
                violations.append(f"{path}: {found}")
    assert not violations, (
        "Dead SBM keys found in config files (code reads p_inter, not q_inter):\n"
        + "\n".join(violations)
    )
