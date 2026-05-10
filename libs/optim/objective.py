"""In-memory study evaluator.

Holds a single Sionna RT scene + receiver layout + RX/TX arrays alive
across trials, swapping out transmitter positions / powers per
candidate. Each evaluation runs the path solver and computes the
aggregate KPIs without touching disk — ~10× faster than calling the
full :func:`run_scenario`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import libs.scenes  # noqa: F401  # ensure Mitsuba variant set first

from libs.schemas import Scenario
from libs.scenes import load_builtin_scene
from libs.sim_engine.kpis import compute_kpis


@dataclass
class TrialResult:
    objective: float
    kpis: dict[str, float | int | None]


class StudyEvaluator:
    """Caches the loaded scene + RX layout. Re-used across trials."""

    def __init__(self, base: Scenario):
        import mitsuba as mi
        import sionna.rt as rt

        self._base = base
        self._mi = mi
        self._rt = rt
        self._scene = load_builtin_scene(base.scene)
        self._scene.frequency = base.band.carrier_hz

        self._scene.tx_array = rt.PlanarArray(
            num_rows=1, num_cols=1, pattern="iso", polarization="V",
        )
        self._scene.rx_array = rt.PlanarArray(
            num_rows=1, num_cols=1, pattern="iso", polarization="V",
        )
        for r in base.receivers:
            self._scene.add(rt.Receiver(name=r.name, position=mi.Point3f(*r.position.to_list())))

        self._tx_names: set[str] = set()
        self._path_solver = rt.PathSolver()

    def evaluate(self, scenario: Scenario, weights: dict[str, float]) -> TrialResult:
        """Run the path solver for ``scenario`` and return the weighted objective."""
        self._sync_tx(scenario)

        paths = self._path_solver(
            scene=self._scene,
            max_depth=scenario.propagation.max_depth,
            samples_per_src=scenario.propagation.samples_per_src_paths,
            los=scenario.propagation.los,
            specular_reflection=scenario.propagation.specular_reflection,
            diffuse_reflection=scenario.propagation.diffuse_reflection,
            refraction=scenario.propagation.refraction,
            diffraction=scenario.propagation.diffraction,
            seed=scenario.propagation.seed,
        )
        kpis = compute_kpis(scenario, paths)
        agg = {k: kpis.aggregate.get(k) for k in kpis.aggregate}
        obj = self._scalarize(agg, weights)
        return TrialResult(objective=obj, kpis=agg)

    # -- internals -------------------------------------------------------------
    def _sync_tx(self, scenario: Scenario) -> None:
        """Match the scene's TX set to the scenario's. Removes/adds as needed."""
        wanted = {t.name: t for t in scenario.transmitters}
        for name in list(self._tx_names):
            if name not in wanted:
                self._scene.remove(name)
                self._tx_names.discard(name)
        for name, txc in wanted.items():
            if name in self._tx_names:
                self._scene.remove(name)  # repopulate with new pos/power
            self._scene.add(self._rt.Transmitter(
                name=name,
                position=self._mi.Point3f(*txc.position.to_list()),
                power_dbm=txc.power_dbm,
            ))
            self._tx_names.add(name)

    @staticmethod
    def _scalarize(agg: dict, weights: dict[str, float]) -> float:
        score = 0.0
        for key, w in weights.items():
            v = agg.get(key)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                # Treat missing metrics as a strong penalty so the
                # optimizer learns to avoid configurations that produce
                # them (e.g., zero-coverage deployments).
                score += w * (-1e3 if w > 0 else 1e3)
            else:
                score += w * float(v)
        return score
