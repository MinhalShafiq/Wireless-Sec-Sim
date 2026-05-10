"""Phase 3 sweep tests: schema, enumeration, template application."""

from __future__ import annotations

from pathlib import Path

import pytest

from libs.sweep import SweepCfg, enumerate_scenarios
from libs.sweep.schema import SweepCfg as _SweepCfg


def test_sweep_loads():
    cfg = SweepCfg.from_yaml("configs/sweeps/jammer_position.yaml")
    assert cfg.name == "jammer_position"
    assert cfg.template == "jammer"


def test_grid_cardinality():
    cfg = SweepCfg.from_yaml("configs/sweeps/jammer_position.yaml")
    assert cfg.grid_cardinality() == 5 * 5 * 1 * 1 * 1
    cfg2 = SweepCfg.from_yaml("configs/sweeps/rogue_position.yaml")
    assert cfg2.grid_cardinality() == 3 * 3 * 1 * 2


def test_enumerate_jammer_sweep():
    cfg = SweepCfg.from_yaml("configs/sweeps/jammer_position.yaml")
    runs = enumerate_scenarios(cfg)
    assert len(runs) == 25
    # Each scenario should have base TXs + the jammer.
    assert any(t.role == "jammer" for t in runs[0].scenario.transmitters)
    # Names should be unique and properly numbered.
    assert len({r.run_id for r in runs}) == 25


def test_enumerate_rogue_sweep_powers_apply():
    cfg = SweepCfg.from_yaml("configs/sweeps/rogue_position.yaml")
    runs = enumerate_scenarios(cfg)
    # Verify both power levels actually produced rogue TXs at each level.
    powers = {r.scenario.transmitters[-1].power_dbm for r in runs
              if r.scenario.transmitters[-1].role == "rogue"}
    assert powers == {40.0, 50.0}


def test_position_applied_to_jammer_tx():
    cfg = _SweepCfg(
        name="t", base="configs/munich_baseline.yaml", template="jammer",
        grid={"position_x": [123.0], "position_y": [-45.0], "position_z": [9.0],
              "power_dbm": [33.0]},
        seeds=[7],
    )
    runs = enumerate_scenarios(cfg)
    assert len(runs) == 1
    jam = runs[0].scenario.transmitters[-1]
    assert jam.role == "jammer"
    assert (jam.position.x, jam.position.y, jam.position.z) == (123.0, -45.0, 9.0)
    assert jam.power_dbm == 33.0
    assert runs[0].scenario.propagation.seed == 7
