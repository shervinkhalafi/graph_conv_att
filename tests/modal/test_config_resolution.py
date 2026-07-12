"""Tests for Modal config-resolution helpers.

Test rationale
--------------
Modal dispatch depends on the `_cli_cmd` declarations embedded in the base
Hydra configs. A missing declaration silently removes an experiment type from
the discoverable CLI map. This test keeps the embedding study on that surface.
"""

from tmgg.modal._lib.config_resolution import discover_cli_cmd_map


def test_discover_cli_cmd_map_includes_embedding_study() -> None:
    """Embedding study config should advertise its Modal CLI command."""
    cmd_map = discover_cli_cmd_map()

    assert cmd_map["tmgg-embedding-study-exp"] == "base_config_embedding_study"
