"""End-to-end Scenario runner.

Loads the scene, populates TXs/RXs (legitimate and adversarial), runs
both the path solver (per-link KPIs) and optionally the radio-map
solver (area coverage / SINR maps), and persists all artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from libs.schemas import Scenario
from libs.scenes import load_builtin_scene
from libs.sim_engine.io import save_kpis, save_meta, save_radio_map, save_scenario
from libs.sim_engine.kpis import RunKpis, compute_kpis
from libs.sim_engine.render import render_views


@dataclass
class RunResult:
    scenario: Scenario
    out_dir: Path
    kpis: RunKpis
    timings: dict[str, float]


def run_scenario(scenario: Scenario, out_dir: str | Path, *, render: bool = True) -> RunResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import mitsuba as mi
    import sionna.rt as rt

    timings: dict[str, float] = {}

    t0 = time.time()
    scene = load_builtin_scene(scenario.scene)
    timings["load_scene_s"] = time.time() - t0
    scene.frequency = scenario.band.carrier_hz

    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    for txc in scenario.transmitters:
        scene.add(rt.Transmitter(
            name=txc.name,
            position=mi.Point3f(*txc.position.to_list()),
            power_dbm=txc.power_dbm,
        ))
    for rxc in scenario.receivers:
        scene.add(rt.Receiver(
            name=rxc.name,
            position=mi.Point3f(*rxc.position.to_list()),
        ))

    # --- Per-link path solver (always run; cheap and gives exact KPIs) ---
    t0 = time.time()
    path_solver = rt.PathSolver()
    paths = path_solver(
        scene=scene,
        max_depth=scenario.propagation.max_depth,
        samples_per_src=scenario.propagation.samples_per_src_paths,
        los=scenario.propagation.los,
        specular_reflection=scenario.propagation.specular_reflection,
        diffuse_reflection=scenario.propagation.diffuse_reflection,
        refraction=scenario.propagation.refraction,
        diffraction=scenario.propagation.diffraction,
        seed=scenario.propagation.seed,
    )
    timings["path_solver_s"] = time.time() - t0

    t0 = time.time()
    kpis = compute_kpis(scenario, paths)
    timings["kpi_extraction_s"] = time.time() - t0

    # --- Optional radio-map solver (area coverage + SINR map) ---
    radio_map = None
    if scenario.radio_map is not None:
        t0 = time.time()
        rm_solver = rt.RadioMapSolver()
        cell = mi.Point2f(scenario.radio_map.cell_size_m, scenario.radio_map.cell_size_m)
        radio_map = rm_solver(
            scene=scene,
            cell_size=cell,
            max_depth=scenario.propagation.max_depth,
            samples_per_tx=scenario.propagation.samples_per_tx,
            los=scenario.propagation.los,
            specular_reflection=scenario.propagation.specular_reflection,
            diffuse_reflection=scenario.propagation.diffuse_reflection,
            refraction=scenario.propagation.refraction,
            diffraction=scenario.propagation.diffraction,
            seed=scenario.propagation.seed,
        )
        timings["radio_map_solver_s"] = time.time() - t0

    # --- 3D photo-realistic views ---
    if render:
        t0 = time.time()
        view_timings = render_views(
            scene, scenario, Path(out_dir),
            paths=paths, radio_map=radio_map,
        )
        timings["render_3d_s"] = round(time.time() - t0, 3)
        timings["render_per_view_s"] = view_timings  # type: ignore[assignment]

    # --- Persist ---
    t0 = time.time()
    save_scenario(out_dir, scenario)
    save_kpis(out_dir, kpis)
    save_radio_map(out_dir, radio_map)
    save_meta(out_dir, {
        "timings": timings,
        "scene": scenario.scene,
        "frequency_hz": scenario.band.carrier_hz,
        "bandwidth_hz": scenario.band.bandwidth_hz,
        "num_tx": len(scenario.transmitters),
        "num_rx": len(scenario.receivers),
        "variant": _current_variant(),
    })
    timings["save_s"] = time.time() - t0

    return RunResult(scenario=scenario, out_dir=out_dir, kpis=kpis, timings=timings)


def _current_variant() -> str | None:
    import mitsuba as mi
    return mi.variant()
