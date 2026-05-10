"""Warm-start the optimizer from prior similar runs in the vector store.

Given a study config (which fully specifies the search space), find the
*k* most similar past scenarios in the corpus and pull their TX
parameters into seed trials. Optuna's ``enqueue_trial`` will run them
first, giving the sampler good initial coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from libs.ml.dataset import SCENARIO_FEATURES, discover_runs, load_run
from libs.optim.schema import StudyCfg, TxSearchSpace
from libs.retrieval import ScenarioEmbedder, ScenarioStore
from libs.schemas import Scenario


def _aggregate_query_features(base: Scenario) -> np.ndarray | None:
    """Pull a feature vector from the base's most-similar past run.

    We don't have a real run for the base scenario, so we use a uniform
    zero vector — the StandardScaler-fit embedder maps unseen "average"
    inputs to the origin, which then retrieves a representative slice of
    the corpus. Good enough as a seed; refine later if we add a true
    "expected feature vector for this scene" model.
    """
    return np.zeros(len(SCENARIO_FEATURES), dtype=np.float32)


def _params_from_scenario(scenario: Scenario, search_space: dict[str, TxSearchSpace]) -> dict:
    """Extract param dict for trial seeding by looking up TXs in the past scenario."""
    by_name = {t.name: t for t in scenario.transmitters}
    out: dict[str, float] = {}
    for tx_name, sp in search_space.items():
        if tx_name not in by_name:
            continue  # past scenario didn't have this TX; skip
        tx = by_name[tx_name]
        if sp.position_x is not None:
            out[f"{tx_name}.position_x"] = float(np.clip(tx.position.x, sp.position_x.low, sp.position_x.high))
        if sp.position_y is not None:
            out[f"{tx_name}.position_y"] = float(np.clip(tx.position.y, sp.position_y.low, sp.position_y.high))
        if sp.position_z is not None:
            out[f"{tx_name}.position_z"] = float(np.clip(tx.position.z, sp.position_z.low, sp.position_z.high))
        if sp.power_dbm is not None:
            out[f"{tx_name}.power_dbm"] = float(np.clip(tx.power_dbm, sp.power_dbm.low, sp.power_dbm.high))
    return out


def warm_start_params(study: StudyCfg, *, max_runs_root: Path | None = None) -> list[dict]:
    """Return up to ``study.warm_start.k`` parameter dicts to enqueue.

    ``max_runs_root`` (optional) limits which sweep root the warm-start
    pulls examples from — useful when the study's threat model differs
    from most of the corpus.
    """
    ws = study.warm_start
    if not ws.enabled:
        return []
    if not ws.embedder or not ws.store:
        raise ValueError("warm_start.enabled requires both embedder and store paths")

    embedder = ScenarioEmbedder.load(ws.embedder)
    store = ScenarioStore(path=ws.store, dim=embedder.dim)

    if store.count() == 0:
        return []

    q = _aggregate_query_features(Scenario.from_yaml(study.base))
    matches = store.find_similar(q, k=max(ws.k * 3, ws.k))

    seeds: list[dict] = []
    for m in matches:
        if max_runs_root is not None and not m.payload.get("source_root", "").startswith(str(max_runs_root)):
            continue
        # Try to find the original run dir to read scenario.json.
        source_root = m.payload.get("source_root", "")
        run_dir = Path(source_root) / "runs" / m.run_id
        if not (run_dir / "scenario.json").exists():
            continue
        scenario = Scenario.model_validate_json((run_dir / "scenario.json").read_text())
        params = _params_from_scenario(scenario, study.search_space)
        if not params:
            continue
        seeds.append(params)
        if len(seeds) >= ws.k:
            break
    return seeds
