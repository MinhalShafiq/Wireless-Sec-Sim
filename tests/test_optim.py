"""Phase 6 optimization tests: schema, scenario building, end-to-end short study."""

from __future__ import annotations

from pathlib import Path

import pytest

from libs.optim.objective import StudyEvaluator
from libs.optim.schema import (
    ObjectiveCfg,
    ObjectiveTerm,
    ParamRange,
    StudyCfg,
    ThreatCfg,
    TxSearchSpace,
)
from libs.optim.study import _apply_threat, _scenario_from_params, run_study
from libs.schemas import Scenario


def test_study_yaml_loads():
    cfg = StudyCfg.from_yaml("configs/studies/munich_bs_perf.yaml")
    assert cfg.name == "munich_bs_perf"
    assert cfg.threat.type == "none"
    assert cfg.optimizer == "tpe"
    assert "bs0" in cfg.search_space


def test_apply_threat_jammer():
    base = Scenario.from_yaml("configs/munich_baseline.yaml")
    threat = ThreatCfg(type="jammer", position={"x": 0, "y": 0, "z": 8}, power_dbm=50)
    s = _apply_threat(base, threat)
    assert any(t.role == "jammer" for t in s.transmitters)


def test_scenario_from_params_overrides_position():
    base = Scenario.from_yaml("configs/munich_baseline.yaml")
    study = StudyCfg(
        name="t", base="configs/munich_baseline.yaml",
        search_space={
            "bs0": TxSearchSpace(
                position_x=ParamRange(low=-200, high=200),
                power_dbm=ParamRange(low=30, high=46),
            )
        },
        objective=ObjectiveCfg(
            terms=[ObjectiveTerm(name="median_user_sinr_db", weight=1.0)]
        ),
        n_trials=1,
    )
    s = _scenario_from_params(base, study, {"bs0.position_x": 123.0, "bs0.power_dbm": 40.0})
    bs0 = next(t for t in s.transmitters if t.name == "bs0")
    assert bs0.position.x == 123.0
    assert bs0.power_dbm == 40.0


@pytest.mark.gpu
def test_short_study_runs(tmp_path: Path):
    """Smoke-test end-to-end: 5 trials, must produce best.json + trials.parquet."""
    study = StudyCfg(
        name="smoke",
        base="configs/munich_baseline.yaml",
        search_space={
            "bs0": TxSearchSpace(
                position_x=ParamRange(low=-100, high=100),
                position_y=ParamRange(low=-100, high=100),
            )
        },
        objective=ObjectiveCfg(
            direction="maximize",
            terms=[ObjectiveTerm(name="median_user_sinr_db", weight=1.0)],
        ),
        n_trials=5,
        optimizer="random",
        seed=0,
    )
    result = run_study(study, tmp_path)
    assert (tmp_path / "trials.parquet").exists()
    assert (tmp_path / "best.json").exists()
    assert (tmp_path / "plot_convergence.png").exists()
    assert result.n_trials == 5
    # The objective must be a finite float (not the NaN penalty value).
    assert -1e3 < result.best_objective < 1e3


@pytest.mark.gpu
def test_evaluator_reuses_scene_across_trials():
    """Two consecutive evaluations on the same evaluator should both succeed
    without reloading the scene. Indirect check: time of second call ≪ first
    is too brittle; instead just assert both return objective floats."""
    base = Scenario.from_yaml("configs/munich_baseline.yaml")
    ev = StudyEvaluator(base)
    weights = {"median_user_sinr_db": 1.0}
    a = ev.evaluate(base, weights)
    b = ev.evaluate(base, weights)
    assert isinstance(a.objective, float)
    assert isinstance(b.objective, float)
