"""Scene loaders — both Sionna RT built-ins and custom OSM-derived scenes.

Importing this package eagerly selects the Mitsuba variant (LLVM by
default, override via ``SIM_VARIANT``). This must happen *before* Sionna
RT is imported anywhere downstream, because Sionna RT's radio-material
plugins are registered per-variant at import time.
"""

from libs.scenes.builtin import BUILTIN_SCENES, _ensure_variant, load_builtin_scene

_ensure_variant()

__all__ = ["BUILTIN_SCENES", "load_builtin_scene"]
