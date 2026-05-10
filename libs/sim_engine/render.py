"""Photo-realistic 3D rendering of a scenario.

Wraps :func:`sionna.rt.Scene.render_to_file` to emit a small gallery of
images per run:

- ``view_overview.png``     — buildings + TX/RX markers, no overlays.
- ``view_coverage.png``     — radio-map (path gain) projected onto the
  ground for the primary legitimate transmitter.
- ``view_sinr.png``         — radio-map SINR (incorporates all TXs as
  interference / aggressors).
- ``view_paths.png``        — ray paths drawn between every TX and RX.

These are the "visual simulation" outputs — they show the actual 3D
geometry, not just an XY heatmap, so attack effects (rogue gNB, jammer)
are immediately visible against the city skyline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from libs.schemas import Scenario


def _bbox_camera(scene, *, azimuth: float = 0.5, zoom: float = 0.55):
    """Pick a bird's-eye-ish camera looking down at the scene centre."""
    import mitsuba as mi
    import sionna.rt as rt

    bbox = scene.mi_scene.bbox()
    cx = float((bbox.min.x + bbox.max.x) * 0.5)
    cy = float((bbox.min.y + bbox.max.y) * 0.5)
    cz = float(bbox.max.z) * 0.4
    diag = float(max(bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y))
    pos = mi.Point3f(cx + diag * zoom, cy - diag * zoom * azimuth, float(bbox.max.z) + diag * 0.55)
    return rt.Camera(position=pos, look_at=mi.Point3f(cx, cy, cz))


def _primary_legit_idx(scenario: Scenario) -> int | None:
    for i, t in enumerate(scenario.transmitters):
        if t.role == "legitimate":
            return i
    return None


def render_views(
    scene: Any,
    scenario: Scenario,
    out_dir: Path,
    *,
    paths: Any | None = None,
    radio_map: Any | None = None,
    num_samples: int = 64,
    resolution: tuple[int, int] = (1280, 800),
) -> dict[str, float]:
    """Render the standard 3D view gallery into ``out_dir``.

    Returns a dict of per-view render times (seconds).
    """
    import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    camera = _bbox_camera(scene)
    timings: dict[str, float] = {}

    def _render(name: str, **kwargs):
        t0 = time.time()
        scene.render_to_file(
            camera=camera,
            filename=str(out_dir / name),
            num_samples=num_samples,
            resolution=resolution,
            **kwargs,
        )
        timings[name] = round(time.time() - t0, 2)

    _render("view_overview.png")

    if radio_map is not None:
        primary = _primary_legit_idx(scenario)
        if primary is not None:
            _render(
                "view_coverage.png",
                radio_map=radio_map,
                rm_tx=primary,
                rm_metric="path_gain",
            )
        _render("view_sinr.png", radio_map=radio_map, rm_metric="sinr")

        # If the scenario contains an adversarial TX, render its
        # footprint as well so attacks are immediately visible.
        for i, t in enumerate(scenario.transmitters):
            if t.role in ("rogue", "jammer"):
                _render(
                    f"view_{t.role}_{t.name}.png",
                    radio_map=radio_map,
                    rm_tx=i,
                    rm_metric="path_gain",
                )

    if paths is not None:
        _render("view_paths.png", paths=paths)

    return timings
