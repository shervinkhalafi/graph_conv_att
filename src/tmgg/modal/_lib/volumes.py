"""Modal volume management for dataset caching.

No ``import modal`` at module level. Volume objects are created lazily
via ``_create_volume()`` — called at decoration time inside
``_functions.py`` or at runtime inside a Modal container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal

# Mount paths within containers (pure constants, no modal dependency)
DATASETS_MOUNT = "/data/datasets"
OUTPUTS_MOUNT = "/data/outputs"

# Volume names in Modal
DATASETS_VOLUME_NAME = "graph-denoising-datasets"
OUTPUTS_VOLUME_NAME = "graph-denoising-outputs"


def _create_volume(name: str) -> modal.Volume:
    """Create or retrieve a Modal volume by name.

    Parameters
    ----------
    name
        Modal volume name.

    Returns
    -------
    modal.Volume
        Volume handle (creates if missing).
    """
    import modal as _modal

    return _modal.Volume.from_name(name, create_if_missing=True)


def get_volume_mounts() -> dict[str, Any]:
    """Get volume mount configuration for Modal functions.

    Only call at decoration time (inside ``_functions.py``) or at runtime
    inside a Modal container.

    Returns
    -------
    dict
        Mapping of mount paths to volumes.
    """
    return {
        DATASETS_MOUNT: _create_volume(DATASETS_VOLUME_NAME),
        OUTPUTS_MOUNT: _create_volume(OUTPUTS_VOLUME_NAME),
    }
