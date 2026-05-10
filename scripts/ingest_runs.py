"""CLI: ingest runs into the scenario vector store.

Usage:
    # Fit the embedder from an existing dataset and ingest every run found.
    python -m scripts.ingest_runs \
        --dataset data/datasets/v1 \
        --embedder data/embedders/v1 \
        --store data/vector_store \
        data/sweeps/legit_bs_position \
        data/sweeps/jammer_position \
        data/sweeps/rogue_position
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from libs.ml.dataset import SCENARIO_FEATURES, discover_runs, load_run  # noqa: E402
from libs.retrieval.decision_support import _scenario_payload  # noqa: E402
from libs.retrieval.embed import ScenarioEmbedder  # noqa: E402
from libs.retrieval.store import ScenarioStore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest runs into the scenario vector store.")
    p.add_argument("roots", nargs="+", help="Sweep dirs or per-run dirs to ingest.")
    p.add_argument("--dataset", required=True, help="Existing dataset.parquet dir to fit embedder.")
    p.add_argument("--embedder", required=True, help="Where to load/save the embedder artifact.")
    p.add_argument("--store", required=True, help="Local Qdrant store path (will be created).")
    p.add_argument("--collection", default="scenarios", help="Collection name.")
    p.add_argument("--refit", action="store_true", help="Always refit the embedder, even if it exists.")
    args = p.parse_args()

    embedder_dir = Path(args.embedder)
    embedder_artifact = embedder_dir / "embedder.joblib"

    if args.refit or not embedder_artifact.exists():
        df = pd.read_parquet(Path(args.dataset) / "dataset.parquet")
        print(f"[embed] fitting embedder on {len(df)} runs from {args.dataset}")
        embedder = ScenarioEmbedder.fit(df[SCENARIO_FEATURES])
        embedder.save(embedder_dir)
        print(f"[embed] saved → {embedder_artifact}")
    else:
        embedder = ScenarioEmbedder.load(embedder_dir)
        print(f"[embed] loaded {embedder_artifact} (dim={embedder.dim})")

    store = ScenarioStore(path=args.store, collection=args.collection, dim=embedder.dim)
    print(f"[store] {args.store}::{args.collection} initially has {store.count()} points")

    runs = discover_runs(*args.roots)
    if not runs:
        print(f"[ingest] no runs found under {args.roots}")
        return 1

    t0 = time.time()
    batch: list = []
    for run_dir in runs:
        run = load_run(run_dir)
        vec = embedder.embed_run(run.scenario, run.per_link)
        payload = _scenario_payload(run, run_dir)
        batch.append((run.run_id, vec, payload))
    store.upsert_batch(batch)
    print(f"[ingest] upserted {len(batch)} runs in {time.time() - t0:.2f}s")
    print(f"[store] now has {store.count()} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
