"""Compatibility helpers for the legacy RIFE video reader.

The bundled RIFE/sk-video stack still references NumPy aliases such as
``np.float`` that were removed in NumPy 1.24.  Keep the compatibility shim
local to the RIFE subprocess instead of pinning/downgrading NumPy globally.
"""

from __future__ import annotations


def patch_legacy_numpy_aliases() -> None:
    """Restore aliases required by legacy sk-video/RIFE code."""
    import numpy as np

    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)
