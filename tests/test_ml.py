"""Phase 4 ML tests: dataset construction, classifier sanity, regressor RMSE bound."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libs.ml.coverage_predictor import build_link_dataset, train as train_cov
from libs.ml.dataset import (
    SCENARIO_FEATURES,
    build_dataset,
    extract_scenario_features,
    label_for_scenario,
    load_run,
)
from libs.ml.security_clf import predict_run, train as train_clf
from libs.schemas import (
    FrequencyBand,
    ReceiverCfg,
    Scenario,
    TransmitterCfg,
    Vec3,
)


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


def test_label_for_scenario_benign():
    assert label_for_scenario(_baseline()) == 0


def test_label_for_scenario_attack():
    s = _baseline()
    s.transmitters.append(
        TransmitterCfg(name="jam0", position=Vec3(x=10, y=10, z=5), power_dbm=50, role="jammer")
    )
    assert label_for_scenario(s) == 1


def _synthetic_per_link(n_users: int = 4, n_tx: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(n_users):
        for t in range(n_tx):
            rows.append({
                "rx_name": f"ue{u}",
                "rx_role": "user",
                "rx_x": float(u * 10), "rx_y": 0.0, "rx_z": 1.5,
                "tx_name": f"bs{t}",
                "tx_role": "legitimate",
                "tx_x": float(t * 50 - 50), "tx_y": 0.0, "tx_z": 25.0,
                "tx_power_dbm": 44.0,
                "distance_m": float(np.linalg.norm([u * 10 - (t * 50 - 50), 0, 25 - 1.5])),
                "path_gain_db": float(-80 - rng.normal(0, 2)),
                "rx_power_dbm": float(-36 - rng.normal(0, 2)),
            })
    return pd.DataFrame(rows)


def test_extract_features_returns_full_vector():
    s = _baseline()
    s.receivers = [ReceiverCfg(name=f"ue{i}", position=Vec3(x=i*10, y=0, z=1.5)) for i in range(4)]
    pl = _synthetic_per_link()
    feats = extract_scenario_features(s, pl)
    assert set(feats.keys()) == set(SCENARIO_FEATURES)
    assert all(np.isfinite(v) or v == 0.0 for v in feats.values())


@pytest.mark.gpu  # tagged because it relies on Phase 3 sweep outputs being on disk
def test_build_dataset_from_sweeps_balanced():
    """If the bundled sweeps have been run, classes should both appear."""
    benign = Path("data/sweeps/legit_bs_position")
    attack = Path("data/sweeps/jammer_position")
    if not (benign.exists() and attack.exists()):
        pytest.skip("sweep outputs not present")
    X, y, ids = build_dataset(benign, attack)
    assert (y == 0).sum() > 0
    assert (y == 1).sum() > 0
    assert X.shape[1] == len(SCENARIO_FEATURES)


@pytest.mark.gpu
def test_classifier_separates_jammer_from_benign():
    benign = Path("data/sweeps/legit_bs_position")
    attack = Path("data/sweeps/jammer_position")
    if not (benign.exists() and attack.exists()):
        pytest.skip("sweep outputs not present")
    X, y, _ = build_dataset(benign, attack)
    pipe, report = train_clf(X, y, test_size=0.3, random_state=0)
    # Jammer-vs-benign should be near-trivially separable.
    assert report.accuracy >= 0.85, report
    assert report.f1 >= 0.85, report


@pytest.mark.gpu
def test_coverage_regressor_rmse_bounded():
    benign = Path("data/sweeps/legit_bs_position")
    if not benign.exists():
        pytest.skip("benign sweep not present")
    X, y, ids = build_link_dataset(benign)
    pipe, report = train_cov(X, y, ids, test_size=0.25, random_state=0)
    # Loose bound — for an in-scene tabular baseline on Munich, RMSE should
    # comfortably beat the path-gain standard deviation.
    assert report.rmse < float(y.std()), report


@pytest.mark.gpu
def test_predict_run_roundtrip(tmp_path: Path):
    benign = Path("data/sweeps/legit_bs_position")
    attack = Path("data/sweeps/jammer_position")
    if not (benign.exists() and attack.exists()):
        pytest.skip("sweep outputs not present")
    X, y, _ = build_dataset(benign, attack)
    pipe, _ = train_clf(X, y, test_size=0.2, random_state=0)

    # Pick one attack run and check inference flags it.
    one_attack = next((attack / "runs").iterdir())
    out = predict_run(pipe, one_attack / "scenario.json", one_attack / "per_link.parquet")
    assert out["predicted_label"] == 1
    assert 0.0 <= out["p_attack"] <= 1.0
