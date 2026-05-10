"""Phase 1 smoke tests — confirm built-in scenes can be loaded."""

from __future__ import annotations

import pytest

from libs.scenes import BUILTIN_SCENES, load_builtin_scene


def test_builtin_registry_nonempty():
    assert "munich" in BUILTIN_SCENES
    assert BUILTIN_SCENES["munich"].kind == "urban"


def test_unknown_scene_raises():
    with pytest.raises(KeyError):
        load_builtin_scene("not_a_real_scene")


@pytest.mark.gpu
def test_load_munich():
    """Loads the Munich scene. Requires sionna-rt installed."""
    sionna_rt = pytest.importorskip("sionna.rt")
    scene = load_builtin_scene("munich")
    assert scene is not None
    bbox = scene.mi_scene.bbox()
    extent = bbox.max - bbox.min
    # Munich scene is on the order of hundreds of meters across.
    assert float(extent.x) > 100.0
    assert float(extent.y) > 100.0
