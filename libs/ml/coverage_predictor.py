"""Coverage / path-gain regressor.

Predicts ``path_gain_db`` from purely geometric features (TX/RX
positions, distance, frequency). Trained on per-link rows aggregated
across many sweep runs. A tabular GradientBoosting baseline — cheap,
strong, easy to swap out for a CNN/GNN later (Phase 4 future work).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from libs.ml.dataset import discover_runs, load_run

PATH_GAIN_FEATURES: list[str] = [
    "tx_x", "tx_y", "tx_z",
    "rx_x", "rx_y", "rx_z",
    "dx", "dy", "dz",
    "distance_m", "log_distance_m",
    "horizontal_distance_m",
    "height_diff_m",
    "carrier_hz",
]


@dataclass
class CoverageReport:
    rmse: float
    mae: float
    n_train: int
    n_test: int


def featurize_links(per_link: pd.DataFrame, carrier_hz: float) -> pd.DataFrame:
    df = per_link.copy()
    df["dx"] = df["tx_x"] - df["rx_x"]
    df["dy"] = df["tx_y"] - df["rx_y"]
    df["dz"] = df["tx_z"] - df["rx_z"]
    df["log_distance_m"] = np.log10(df["distance_m"].clip(lower=1e-3))
    df["horizontal_distance_m"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["height_diff_m"] = df["dz"]
    df["carrier_hz"] = carrier_hz
    return df


def build_link_dataset(*roots: Path | str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Concatenate per-link rows from many runs into one regression dataset."""
    runs = discover_runs(*roots)
    parts: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    src_ids: list[str] = []
    for run_dir in runs:
        run = load_run(run_dir)
        f = featurize_links(run.per_link, carrier_hz=run.scenario.band.carrier_hz)
        # Drop rows with non-finite path gain (no measurable signal).
        f = f[np.isfinite(f["path_gain_db"])]
        if f.empty:
            continue
        parts.append(f[PATH_GAIN_FEATURES].copy())
        targets.append(f["path_gain_db"].to_numpy())
        src_ids.extend([run.run_id] * len(f))
    if not parts:
        raise FileNotFoundError(f"no usable per-link rows under {roots}")
    X = pd.concat(parts, ignore_index=True)
    y = pd.Series(np.concatenate(targets), name="path_gain_db")
    return X, y, src_ids


def make_pipeline(*, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=random_state,
        )),
    ])


def train(
    X: pd.DataFrame,
    y: pd.Series,
    src_ids: list[str],
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[Pipeline, CoverageReport]:
    """Fit on one split, evaluate on a held-out one. Splits at the link level
    (good enough as a baseline; cross-scene generalisation is future work)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    pipe = make_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    return pipe, CoverageReport(rmse=rmse, mae=mae, n_train=len(X_train), n_test=len(X_test))


def save_model(pipe: Pipeline, report: CoverageReport, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "coverage_predictor.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(asdict(report), indent=2))


def load_model(model_dir: Path) -> Pipeline:
    return joblib.load(Path(model_dir) / "coverage_predictor.joblib")
