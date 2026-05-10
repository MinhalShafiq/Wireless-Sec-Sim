"""Persist run artifacts: scenario.json, kpis.json, per_rx.parquet, radio_map.npz/png."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from libs.schemas import Scenario
from libs.sim_engine.kpis import RunKpis


def save_scenario(out_dir: Path, scenario: Scenario) -> None:
    (out_dir / "scenario.json").write_text(scenario.model_dump_json(indent=2))


def save_kpis(out_dir: Path, kpis: RunKpis) -> None:
    summary = _sanitize({
        "aggregate": kpis.aggregate,
        "per_rx_summary": [asdict(s) for s in kpis.per_rx_summary],
    })
    (out_dir / "kpis.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    df = pd.DataFrame([asdict(r) for r in kpis.per_link_rows])
    # Flatten xyz tuples into separate columns for easier downstream use.
    for col in ("rx_xyz", "tx_xyz"):
        df[[f"{col[:2]}_x", f"{col[:2]}_y", f"{col[:2]}_z"]] = pd.DataFrame(
            df[col].tolist(), index=df.index
        )
    df = df.drop(columns=["rx_xyz", "tx_xyz"])
    df.to_parquet(out_dir / "per_link.parquet", index=False)


def save_radio_map(out_dir: Path, radio_map) -> None:
    if radio_map is None:
        return
    np.savez_compressed(
        out_dir / "radio_map.npz",
        path_gain=np.asarray(radio_map.path_gain),
        sinr=np.asarray(radio_map.sinr),
        rss=np.asarray(radio_map.rss),
        cell_centers=np.asarray(radio_map.cell_centers),
        cell_size=np.asarray(radio_map.cell_size),
        center=np.asarray(radio_map.center),
        size=np.asarray(radio_map.size),
    )
    fig = radio_map.show(metric="path_gain", tx=0)
    fig.savefig(out_dir / "radio_map.png", dpi=120, bbox_inches="tight")
    fig.clf()


def save_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    (out_dir / "meta.json").write_text(
        json.dumps(_sanitize(meta), indent=2, default=_json_default)
    )


def _sanitize(obj):
    """Recursively convert non-JSON-safe values (±inf, NaN, numpy scalars)."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _json_default(obj):
    return _sanitize(obj)
