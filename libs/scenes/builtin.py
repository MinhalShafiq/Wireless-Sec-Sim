"""Wrapper around Sionna RT's built-in example scenes.

Sionna RT ships several ready-made scenes; the OSM-derived urban ones
(Munich, Etoile, Florence, San Francisco) are the most useful starting
points for wireless-network experiments because they have realistic
building geometry and material assignments.

Backend selection
-----------------
Sionna RT's CUDA backend requires OptiX (libnvoptix.so.1). On systems
where OptiX is missing, ray tracing falls back to the LLVM (CPU) backend.
Set ``SIM_VARIANT`` to override (``llvm_ad_mono_polarized`` /
``cuda_ad_mono_polarized``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneInfo:
    key: str
    description: str
    kind: str  # "urban" | "primitive"


BUILTIN_SCENES: dict[str, SceneInfo] = {
    "munich": SceneInfo("munich", "Area around the Frauenkirche, Munich (OSM)", "urban"),
    "etoile": SceneInfo("etoile", "Area around the Arc de Triomphe, Paris (OSM)", "urban"),
    "florence": SceneInfo("florence", "Area around the Florence cathedral (OSM)", "urban"),
    "san_francisco": SceneInfo("san_francisco", "Portion of San Francisco cityscape (OSM)", "urban"),
    "simple_street_canyon": SceneInfo(
        "simple_street_canyon", "Two parallel buildings forming a street canyon", "primitive"
    ),
    "simple_street_canyon_with_cars": SceneInfo(
        "simple_street_canyon_with_cars", "Street canyon with vehicles", "primitive"
    ),
    "floor_wall": SceneInfo("floor_wall", "Ground plane plus a vertical wall", "primitive"),
    "box": SceneInfo("box", "Closed metallic box (debug)", "primitive"),
}


_DEFAULT_VARIANT = "llvm_ad_mono_polarized"


def _ensure_variant() -> str:
    """Pick the Mitsuba variant before Sionna RT initializes.

    Sionna RT registers radio-material plugins for the variant that is
    active at first import; switching variants afterward breaks scene
    loading. We therefore set the variant deterministically here, and
    callers must call this before importing ``sionna.rt`` for the first
    time. Override via ``SIM_VARIANT`` (e.g. ``cuda_ad_mono_polarized``).
    """
    import mitsuba as mi

    requested = os.environ.get("SIM_VARIANT", _DEFAULT_VARIANT)
    if mi.variant() != requested:
        mi.set_variant(requested)
    return requested


def load_builtin_scene(key: str):
    """Load a Sionna RT built-in scene by its short key.

    Returns the loaded :class:`sionna.rt.Scene`. Sionna RT is imported
    lazily so that ``libs.scenes`` is cheap to import without GPU/CUDA
    libraries available (useful for tests and CI).
    """
    if key not in BUILTIN_SCENES:
        raise KeyError(
            f"Unknown scene '{key}'. Available: {', '.join(sorted(BUILTIN_SCENES))}"
        )

    _ensure_variant()

    import sionna.rt as rt
    from sionna.rt import scene as scene_assets

    asset_path = getattr(scene_assets, key)
    return rt.load_scene(asset_path)
