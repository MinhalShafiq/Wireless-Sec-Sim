"""Phase 5 — vector retrieval over the scenario library."""

from libs.retrieval.decision_support import (
    ingest_run,
    precision_at_k,
    summarize_similar,
)
from libs.retrieval.embed import ScenarioEmbedder
from libs.retrieval.store import Match, ScenarioStore, run_id_to_uuid

__all__ = [
    "Match",
    "ScenarioEmbedder",
    "ScenarioStore",
    "ingest_run",
    "precision_at_k",
    "run_id_to_uuid",
    "summarize_similar",
]
