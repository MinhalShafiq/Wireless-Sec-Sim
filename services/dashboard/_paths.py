"""Helpers for the Streamlit dashboard.

Every page imports this module first; it does the ``sys.path``
bookkeeping so ``from libs... import ...`` works under
``streamlit run``, and provides discovery functions for runs / sweeps /
studies / datasets / models that the pages read from disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"


def list_run_dirs(under: Path) -> list[Path]:
    """A "run" is a dir containing scenario.json + per_link.parquet."""
    if not under.exists():
        return []
    out: list[Path] = []
    if (under / "scenario.json").exists():
        out.append(under)
        return out
    if (under / "runs").exists():
        for sub in sorted((under / "runs").iterdir()):
            if (sub / "scenario.json").exists():
                out.append(sub)
    return out


def list_single_runs() -> list[Path]:
    """Direct per-run dirs under data/runs/ (Phase 2 outputs)."""
    runs_dir = DATA / "runs"
    if not runs_dir.exists():
        return []
    return [
        d for d in sorted(runs_dir.iterdir())
        if d.is_dir() and (d / "scenario.json").exists()
    ]


def list_sweep_dirs() -> list[Path]:
    sweeps = DATA / "sweeps"
    if not sweeps.exists():
        return []
    return [
        d for d in sorted(sweeps.iterdir())
        if d.is_dir() and (d / "summary.parquet").exists()
    ]


def list_study_dirs() -> list[Path]:
    studies = DATA / "studies"
    if not studies.exists():
        return []
    return [
        d for d in sorted(studies.iterdir())
        if d.is_dir() and (d / "best.json").exists()
    ]


def list_dataset_dirs() -> list[Path]:
    datasets = DATA / "datasets"
    if not datasets.exists():
        return []
    return [
        d for d in sorted(datasets.iterdir())
        if d.is_dir() and (d / "dataset.parquet").exists()
    ]


def list_model_dirs() -> list[Path]:
    models = DATA / "models"
    if not models.exists():
        return []
    return [d for d in sorted(models.iterdir()) if d.is_dir()]


def list_embedder_dirs() -> list[Path]:
    embedders = DATA / "embedders"
    if not embedders.exists():
        return []
    return [
        d for d in sorted(embedders.iterdir())
        if d.is_dir() and (d / "embedder.joblib").exists()
    ]


def vector_store_path() -> Path | None:
    p = DATA / "vector_store"
    return p if p.exists() else None


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def all_runs_under_sweeps_and_singles() -> list[Path]:
    out: list[Path] = list(list_single_runs())
    for sd in list_sweep_dirs():
        out.extend(list_run_dirs(sd))
    return out
