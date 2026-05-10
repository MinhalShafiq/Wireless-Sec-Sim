"""Security classifier — binary attack-detection on scenario features.

A RandomForest baseline on the role-free per-scenario features from
:mod:`libs.ml.dataset`. Reports accuracy / F1 / confusion matrix on a
scenario-level held-out split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from libs.ml.dataset import SCENARIO_FEATURES, extract_scenario_features


@dataclass
class TrainReport:
    accuracy: float
    f1: float
    roc_auc: float | None
    confusion: list[list[int]]
    n_train: int
    n_test: int
    feature_importances: dict[str, float]


def make_pipeline(*, n_estimators: int = 200, random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[Pipeline, TrainReport]:
    """Fit a classifier, return it and a held-out evaluation report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    pipe = make_pipeline(random_state=random_state)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    try:
        roc = float(roc_auc_score(y_test, proba)) if proba is not None else None
    except ValueError:
        roc = None

    rf: RandomForestClassifier = pipe.named_steps["clf"]
    importances = dict(zip(SCENARIO_FEATURES, rf.feature_importances_))

    report = TrainReport(
        accuracy=float(accuracy_score(y_test, pred)),
        f1=float(f1_score(y_test, pred)),
        roc_auc=roc,
        confusion=confusion_matrix(y_test, pred).tolist(),
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        feature_importances={
            k: float(v) for k, v in
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        },
    )
    return pipe, report


def save_model(pipe: Pipeline, report: TrainReport, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "security_clf.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(asdict(report), indent=2))


def load_model(model_dir: Path) -> Pipeline:
    return joblib.load(Path(model_dir) / "security_clf.joblib")


def predict_run(pipe: Pipeline, scenario_path: Path, per_link_path: Path) -> dict:
    """Run inference on a single per-run directory's artifacts."""
    from libs.schemas import Scenario as _S
    scenario = _S.model_validate_json(Path(scenario_path).read_text())
    per_link = pd.read_parquet(per_link_path)
    feats = extract_scenario_features(scenario, per_link)
    X = pd.DataFrame([feats], columns=SCENARIO_FEATURES)
    pred = int(pipe.predict(X)[0])
    proba = float(pipe.predict_proba(X)[0, 1])
    return {
        "scenario": scenario.name,
        "predicted_label": pred,
        "predicted": "attack" if pred == 1 else "benign",
        "p_attack": proba,
        "n_features": len(feats),
    }


def text_classification_report(y_true: pd.Series, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, target_names=["benign", "attack"])
