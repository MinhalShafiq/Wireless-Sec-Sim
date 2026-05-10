"""Phase 2 tests: scenario schema, FSPL agreement, adversary injection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libs.schemas import (
    FrequencyBand,
    ReceiverCfg,
    Scenario,
    TransmitterCfg,
    Vec3,
)
from libs.sim_engine.adversaries import add_eavesdropper, add_jammer, add_rogue_gnb


def _baseline() -> Scenario:
    return Scenario(
        name="t",
        scene="floor_wall",
        band=FrequencyBand(),
        transmitters=[
            TransmitterCfg(name="bs0", position=Vec3(x=0, y=0, z=25), power_dbm=44.0)
        ],
        receivers=[ReceiverCfg(name="ue0", position=Vec3(x=50, y=0, z=1.5))],
    )


def test_scenario_yaml_roundtrip(tmp_path: Path):
    s = _baseline()
    p = tmp_path / "s.yaml"
    s.to_yaml(p)
    loaded = Scenario.from_yaml(p)
    assert loaded == s


def test_yaml_loads(tmp_path: Path):
    cfg = Path("configs/fspl_validation.yaml")
    assert cfg.exists()
    s = Scenario.from_yaml(cfg)
    assert s.scene == "floor_wall"
    assert len(s.transmitters) == 1
    assert len(s.receivers) == 3


def test_add_jammer_appends_jammer():
    s = _baseline()
    s2 = add_jammer(s, (10, 10, 5), power_dbm=50)
    assert len(s2.transmitters) == 2
    assert s2.transmitters[-1].role == "jammer"
    # Original untouched
    assert len(s.transmitters) == 1


def test_add_rogue_mimics_power():
    s = _baseline()
    s2 = add_rogue_gnb(s, (10, 10, 5))
    assert s2.transmitters[-1].role == "rogue"
    assert s2.transmitters[-1].power_dbm == s.transmitters[0].power_dbm


def test_add_eavesdropper():
    s = _baseline()
    s2 = add_eavesdropper(s, (5, 5, 1.5))
    assert s2.receivers[-1].role == "eavesdropper"


@pytest.mark.gpu
def test_fspl_agreement(tmp_path: Path):
    """Path-gain from the engine must match Friis FSPL within 0.5 dB."""
    pytest.importorskip("sionna.rt")
    from libs.sim_engine import run_scenario

    s = Scenario.from_yaml("configs/fspl_validation.yaml")
    result = run_scenario(s, tmp_path / "fspl")

    # Expected FSPL per RX (path_gain in dB = -FSPL).
    c = 3e8
    lam = c / s.band.carrier_hz
    measured = {r.name: None for r in s.receivers}
    for row in result.kpis.per_link_rows:
        if row.tx_name == "tx0":
            measured[row.rx_name] = row.path_gain_db

    for rx in s.receivers:
        d = abs(rx.position.x)
        expected = -20 * np.log10(4 * np.pi * d / lam)
        assert measured[rx.name] is not None, f"no measurement for {rx.name}"
        err = abs(measured[rx.name] - expected)
        assert err < 0.5, (
            f"{rx.name}: measured {measured[rx.name]:.2f} dB vs Friis {expected:.2f} dB "
            f"(err {err:.2f} dB)"
        )
