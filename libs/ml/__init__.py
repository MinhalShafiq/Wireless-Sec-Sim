"""Phase 4 — ML/AI layer.

This package intentionally does not import ``libs.scenes`` at module
load time, so it can be used without GPU/Sionna present (training and
inference run on the parquet artifacts produced by Phases 2–3).
"""

from libs.ml.dataset import (
    PER_RX_FEATURES,
    SCENARIO_FEATURES,
    build_dataset,
    extract_scenario_features,
    label_for_scenario,
    load_run,
)

__all__ = [
    "PER_RX_FEATURES",
    "SCENARIO_FEATURES",
    "build_dataset",
    "extract_scenario_features",
    "label_for_scenario",
    "load_run",
]
