"""Scenario embedder.

A single-line "encoder" — :class:`sklearn.preprocessing.StandardScaler`
applied to the 63 scenario features from :mod:`libs.ml.dataset`. It's
intentionally simple for the prototype: same feature space the security
classifier uses, so a high-precision retriever doubles as a
classification consistency check.

Forward path to the proposal's contrastive / autoencoder embedder is
clean — only ``embed_run`` and the persisted artefact need to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from libs.ml.dataset import SCENARIO_FEATURES, extract_scenario_features
from libs.schemas import Scenario

if TYPE_CHECKING:
    pass


class ScenarioEmbedder:
    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler
        self.dim: int = int(scaler.n_features_in_)

    @classmethod
    def fit(cls, X: pd.DataFrame) -> "ScenarioEmbedder":
        scaler = StandardScaler()
        scaler.fit(X[SCENARIO_FEATURES].fillna(0.0).to_numpy())
        return cls(scaler)

    def _embed(self, feats: dict) -> np.ndarray:
        df = pd.DataFrame([feats], columns=SCENARIO_FEATURES).fillna(0.0)
        return self.scaler.transform(df.to_numpy())[0].astype(np.float32)

    def embed_features(self, feats: dict) -> np.ndarray:
        return self._embed(feats)

    def embed_run(self, scenario: Scenario, per_link: pd.DataFrame) -> np.ndarray:
        return self._embed(extract_scenario_features(scenario, per_link))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "embedder.joblib")

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioEmbedder":
        scaler = joblib.load(Path(path) / "embedder.joblib")
        return cls(scaler)
