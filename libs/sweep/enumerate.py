"""Enumerate a SweepCfg into a concrete list of (run_id, params, Scenario)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

from libs.schemas import Scenario
from libs.sweep.schema import SweepCfg
from libs.sweep.templates import apply_template


@dataclass(frozen=True)
class SweepRun:
    run_id: str
    params: dict
    seed: int
    scenario: Scenario


def enumerate_scenarios(sweep: SweepCfg, *, base_dir: Path | None = None) -> list[SweepRun]:
    """Expand ``sweep`` into a list of concrete runs (cross product × seeds)."""
    base_dir = base_dir or Path(".")
    base = Scenario.from_yaml(base_dir / sweep.base)

    if sweep.grid:
        keys = list(sweep.grid.keys())
        values = [sweep.grid[k] for k in keys]
        combos = list(itertools.product(*values))
    else:
        keys = []
        combos = [()]

    runs: list[SweepRun] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        for seed in sweep.seeds:
            run_id = f"{sweep.name}_{len(runs):04d}"
            scenario = apply_template(base, sweep.template, params, seed=seed, run_id=run_id)
            runs.append(SweepRun(run_id=run_id, params=params, seed=seed, scenario=scenario))
    return runs
