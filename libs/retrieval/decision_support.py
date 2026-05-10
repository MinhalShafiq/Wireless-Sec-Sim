"""Retrieval-augmented decision support.

Given a freshly completed run, surface the *k* most similar past
scenarios and what happened in them. Use cases:

- A new deployment plan → "find similar past plans and how their
  security held up."
- A new attack scenario → "have we seen this attack pattern before?"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from libs.ml.dataset import label_for_scenario, load_run
from libs.retrieval.embed import ScenarioEmbedder
from libs.retrieval.store import Match, ScenarioStore


def _scenario_payload(run, run_dir: Path) -> dict[str, Any]:
    s = run.scenario
    kpis_path = run_dir / "kpis.json"
    kpis_aggregate: dict[str, Any] = {}
    if kpis_path.exists():
        kpis_aggregate = json.loads(kpis_path.read_text()).get("aggregate", {})
    return {
        "scenario_name": s.name,
        "scene": s.scene,
        "label": run.label,
        "has_jammer": any(t.role == "jammer" for t in s.transmitters),
        "has_rogue": any(t.role == "rogue" for t in s.transmitters),
        "has_eavesdropper": any(r.role == "eavesdropper" for r in s.receivers),
        "n_tx": len(s.transmitters),
        "n_users": sum(1 for r in s.receivers if r.role == "user"),
        "carrier_hz": float(s.band.carrier_hz),
        "median_user_sinr_db": kpis_aggregate.get("median_user_sinr_db"),
        "users_with_jammer_dominant": kpis_aggregate.get("users_with_jammer_dominant"),
        "users_with_rogue_dominant": kpis_aggregate.get("users_with_rogue_dominant"),
        "source_root": str(run_dir.parent.parent),
    }


def ingest_run(
    run_dir: Path, embedder: ScenarioEmbedder, store: ScenarioStore
) -> None:
    """Embed a single run and upsert it into ``store``."""
    run = load_run(run_dir)
    vec = embedder.embed_run(run.scenario, run.per_link)
    payload = _scenario_payload(run, run_dir)
    store.upsert(run.run_id, vec, payload)


@dataclass
class SummaryRow:
    run_id: str
    score: float
    label: int
    scenario_name: str
    has_jammer: bool
    has_rogue: bool
    median_user_sinr_db: float | None


def summarize_similar(
    query_run_dir: Path,
    embedder: ScenarioEmbedder,
    store: ScenarioStore,
    *,
    k: int = 5,
    filters: dict | None = None,
) -> dict[str, Any]:
    """Build a "k nearest past scenarios + their outcomes" summary."""
    query_run_dir = Path(query_run_dir)
    run = load_run(query_run_dir)
    vec = embedder.embed_run(run.scenario, run.per_link)

    matches: list[Match] = store.find_similar(
        vec, k=k, filters=filters, exclude_run_id=run.run_id,
    )

    rows = [
        SummaryRow(
            run_id=m.run_id,
            score=m.score,
            label=int(m.payload.get("label", -1)),
            scenario_name=str(m.payload.get("scenario_name", "")),
            has_jammer=bool(m.payload.get("has_jammer", False)),
            has_rogue=bool(m.payload.get("has_rogue", False)),
            median_user_sinr_db=m.payload.get("median_user_sinr_db"),
        )
        for m in matches
    ]

    sinrs = [r.median_user_sinr_db for r in rows if r.median_user_sinr_db is not None]
    return {
        "query": {
            "run_id": run.run_id,
            "scenario_name": run.scenario.name,
            "label": label_for_scenario(run.scenario),
        },
        "matches": [asdict(r) for r in rows],
        "neighbour_summary": {
            "k": len(rows),
            "share_attack": float(np.mean([r.label for r in rows])) if rows else None,
            "median_sinr_db_across_neighbors": float(np.median(sinrs)) if sinrs else None,
            "any_jammer": bool(any(r.has_jammer for r in rows)),
            "any_rogue": bool(any(r.has_rogue for r in rows)),
        },
    }


def precision_at_k(
    embedder: ScenarioEmbedder,
    store: ScenarioStore,
    runs: list[Path],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Hold-one-out retrieval evaluation against scenario labels."""
    hits = total = 0
    per_run: list[float] = []
    for run_dir in runs:
        run = load_run(run_dir)
        true = run.label
        vec = embedder.embed_run(run.scenario, run.per_link)
        ms = store.find_similar(vec, k=k, exclude_run_id=run.run_id)
        if not ms:
            continue
        match = sum(1 for m in ms if int(m.payload.get("label", -1)) == true) / len(ms)
        per_run.append(match)
        hits += int(round(match * len(ms)))
        total += len(ms)
    return {
        "k": k,
        "n_queries": len(per_run),
        "precision_at_k_mean": float(np.mean(per_run)) if per_run else 0.0,
        "precision_at_k_pooled": hits / total if total else 0.0,
    }


def export_per_link(run_dir: Path) -> pd.DataFrame:
    """Convenience: load a run's per_link.parquet."""
    return pd.read_parquet(Path(run_dir) / "per_link.parquet")
