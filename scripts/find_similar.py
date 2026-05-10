"""CLI: find the k most similar past scenarios for a given run.

Usage:
    python -m scripts.find_similar data/runs/munich_jammed \
        --embedder data/embedders/v1 \
        --store data/vector_store

    # Filter to jammer-only neighbours:
    python -m scripts.find_similar data/runs/munich_jammed \
        --embedder data/embedders/v1 \
        --store data/vector_store \
        --filter has_jammer=true --k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.retrieval.decision_support import summarize_similar  # noqa: E402
from libs.retrieval.embed import ScenarioEmbedder  # noqa: E402
from libs.retrieval.store import ScenarioStore  # noqa: E402


def _parse_filter(items: list[str]) -> dict:
    """Parse `--filter key=value` pairs. Strings stay strings; ``true``/``false``
    become booleans; otherwise tries int → float → string."""
    out: dict = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--filter must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Find similar past scenarios.")
    p.add_argument("run_dir", help="Per-run directory to use as the query.")
    p.add_argument("--embedder", required=True, help="Embedder artifact dir.")
    p.add_argument("--store", required=True, help="Local Qdrant store path.")
    p.add_argument("--collection", default="scenarios", help="Collection name.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--filter", action="append", default=[], help="key=value filter (repeatable).")
    args = p.parse_args()

    embedder = ScenarioEmbedder.load(args.embedder)
    store = ScenarioStore(path=args.store, collection=args.collection, dim=embedder.dim)
    filters = _parse_filter(args.filter) or None

    summary = summarize_similar(
        Path(args.run_dir), embedder, store, k=args.k, filters=filters,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
