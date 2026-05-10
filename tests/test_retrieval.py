"""Phase 5 retrieval tests: embedder + store + retrieval quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libs.ml.dataset import SCENARIO_FEATURES, discover_runs
from libs.retrieval.decision_support import (
    ingest_run,
    precision_at_k,
    summarize_similar,
)
from libs.retrieval.embed import ScenarioEmbedder
from libs.retrieval.store import ScenarioStore, run_id_to_uuid


def test_run_id_uuid_deterministic():
    a = run_id_to_uuid("foo_0001")
    b = run_id_to_uuid("foo_0001")
    c = run_id_to_uuid("foo_0002")
    assert a == b
    assert a != c


def test_store_upsert_and_query_in_memory():
    store = ScenarioStore(":memory:", dim=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    store.upsert("a", v, {"label": 0, "scenario_name": "a"})
    store.upsert("b", np.array([0.99, 0.01, 0.0, 0.0], dtype=np.float32),
                 {"label": 0, "scenario_name": "b"})
    store.upsert("c", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                 {"label": 1, "scenario_name": "c"})
    assert store.count() == 3
    matches = store.find_similar(v, k=2, exclude_run_id="a")
    # Closest to v=[1,0,0,0] should be b, then c.
    assert [m.run_id for m in matches] == ["b", "c"]
    assert matches[0].score > matches[1].score


def test_store_filter_by_label():
    store = ScenarioStore(":memory:", dim=2)
    for i in range(5):
        store.upsert(f"benign_{i}", np.array([1.0, i * 0.1], dtype=np.float32),
                     {"label": 0})
    for i in range(5):
        store.upsert(f"attack_{i}", np.array([0.0, i * 0.1], dtype=np.float32),
                     {"label": 1})
    matches = store.find_similar(
        np.array([1.0, 0.5], dtype=np.float32), k=10, filters={"label": 1}
    )
    assert all(m.payload["label"] == 1 for m in matches)


def test_embedder_fit_dim_matches():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, len(SCENARIO_FEATURES))), columns=SCENARIO_FEATURES)
    e = ScenarioEmbedder.fit(df)
    assert e.dim == len(SCENARIO_FEATURES)


@pytest.mark.gpu
def test_full_ingest_and_precision_at_k(tmp_path: Path):
    """Embed all sweep runs, query each one, precision@5 should be high."""
    benign = Path("data/sweeps/legit_bs_position")
    attack_j = Path("data/sweeps/jammer_position")
    attack_r = Path("data/sweeps/rogue_position")
    if not all(p.exists() for p in (benign, attack_j, attack_r)):
        pytest.skip("sweep outputs not present")

    df = pd.read_parquet(Path("data/datasets/v1/dataset.parquet"))
    embedder = ScenarioEmbedder.fit(df[SCENARIO_FEATURES])

    store = ScenarioStore(path=tmp_path / "store", dim=embedder.dim)
    runs = discover_runs(benign, attack_j, attack_r)
    for r in runs:
        ingest_run(r, embedder, store)

    assert store.count() == len(runs)

    p = precision_at_k(embedder, store, runs, k=5)
    # Embedding uses the same role-aware features the classifier uses, so
    # neighbours should overwhelmingly share the query's label.
    assert p["precision_at_k_mean"] >= 0.85, p


@pytest.mark.gpu
def test_summarize_similar_round_trip(tmp_path: Path):
    """With the full corpus (benign + jammer + rogue) in the store, the
    nearest neighbours of an *effective* jammer run should be predominantly
    other attacks."""
    import json

    benign = Path("data/sweeps/legit_bs_position")
    attack_j = Path("data/sweeps/jammer_position")
    attack_r = Path("data/sweeps/rogue_position")
    if not all(p.exists() for p in (benign, attack_j, attack_r)):
        pytest.skip("sweep outputs not present")

    df = pd.read_parquet(Path("data/datasets/v1/dataset.parquet"))
    embedder = ScenarioEmbedder.fit(df[SCENARIO_FEATURES])
    store = ScenarioStore(path=tmp_path / "store", dim=embedder.dim)
    for r in discover_runs(benign, attack_j, attack_r):
        ingest_run(r, embedder, store)

    # Find a run where the jammer actually dominates >= 1 UE.
    effective_run = None
    for run_dir in sorted((attack_j / "runs").iterdir()):
        kpis = json.loads((run_dir / "kpis.json").read_text())
        if kpis["aggregate"].get("users_with_jammer_dominant", 0) >= 1:
            effective_run = run_dir
            break
    assert effective_run is not None, "no effective jammer in sweep"

    summary = summarize_similar(effective_run, embedder, store, k=5)
    assert summary["query"]["label"] == 1
    assert summary["neighbour_summary"]["k"] == 5
    # With the full attack corpus available, neighbours should be mostly attacks.
    assert summary["neighbour_summary"]["share_attack"] >= 0.6
