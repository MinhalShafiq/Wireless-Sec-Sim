"""Qdrant-backed scenario store.

Wraps ``qdrant-client`` in **local persistence mode** (no server). The
same client API works against cloud Qdrant when we lift this into the
deployment phase — only the constructor argument changes.

Point IDs are deterministic UUID5s derived from the run id, so
re-ingesting the same sweep dir is idempotent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-0000000077af")


def run_id_to_uuid(run_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, run_id))


@dataclass
class Match:
    run_id: str
    score: float
    payload: dict[str, Any]


class ScenarioStore:
    """Thin wrapper over a single Qdrant collection of scenario embeddings."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        collection: str = "scenarios",
        dim: int | None = None,
    ):
        self._path = path
        self.collection = collection
        # `path=":memory:"` works only via location kwarg; otherwise pass path.
        if str(path) == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

        if dim is not None:
            self._ensure_collection(dim)

    # -- collection lifecycle --------------------------------------------------
    def _ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    # -- writes ----------------------------------------------------------------
    def upsert(self, run_id: str, vector: np.ndarray, payload: dict[str, Any]) -> None:
        self._ensure_collection(int(len(vector)))
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(
                    id=run_id_to_uuid(run_id),
                    vector=vector.astype(float).tolist(),
                    payload={"run_id": run_id, **payload},
                )
            ],
        )

    def upsert_batch(self, runs: list[tuple[str, np.ndarray, dict[str, Any]]]) -> None:
        if not runs:
            return
        self._ensure_collection(int(len(runs[0][1])))
        points = [
            qm.PointStruct(
                id=run_id_to_uuid(run_id),
                vector=vec.astype(float).tolist(),
                payload={"run_id": run_id, **payload},
            )
            for run_id, vec, payload in runs
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    # -- reads -----------------------------------------------------------------
    def find_similar(
        self,
        vector: np.ndarray,
        k: int = 5,
        *,
        filters: dict[str, Any] | None = None,
        exclude_run_id: str | None = None,
    ) -> list[Match]:
        qfilter = _build_filter(filters, exclude_run_id=exclude_run_id)
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector.astype(float).tolist(),
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        )
        return [
            Match(
                run_id=str(p.payload.get("run_id", "")),
                score=float(p.score),
                payload=dict(p.payload or {}),
            )
            for p in result.points
        ]

    def all_run_ids(self) -> list[str]:
        ids: list[str] = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            ids.extend(str(p.payload.get("run_id", "")) for p in batch)
            if offset is None:
                break
        return ids


def _build_filter(
    filters: dict[str, Any] | None, *, exclude_run_id: str | None = None
) -> qm.Filter | None:
    must: list[qm.FieldCondition] = []
    must_not: list[qm.FieldCondition] = []
    if filters:
        for k, v in filters.items():
            if isinstance(v, bool):
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
            elif isinstance(v, (int, float, str)):
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))
            elif isinstance(v, list):
                must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
    if exclude_run_id:
        must_not.append(
            qm.FieldCondition(key="run_id", match=qm.MatchValue(value=exclude_run_id))
        )
    if not must and not must_not:
        return None
    return qm.Filter(must=must or None, must_not=must_not or None)
