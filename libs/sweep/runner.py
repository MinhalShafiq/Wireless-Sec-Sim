"""Execute a Sweep: iterate runs, persist outputs, write summary parquet."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from libs.sim_engine import run_scenario
from libs.sweep.enumerate import enumerate_scenarios
from libs.sweep.schema import SweepCfg


@dataclass
class SweepResult:
    sweep: SweepCfg
    out_dir: Path
    summary_path: Path
    n_runs: int
    total_time_s: float


def run_sweep(
    sweep: SweepCfg,
    out_dir: str | Path,
    *,
    render: bool = False,
    progress: bool = True,
) -> SweepResult:
    """Run every (grid × seed) combo. Per-run outputs live under ``runs/<run_id>/``;
    a sweep-level ``summary.parquet`` aggregates KPIs and grid params for analysis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    (out_dir / "sweep.json").write_text(sweep.model_dump_json(indent=2))

    runs = enumerate_scenarios(sweep)
    n = len(runs)
    rows: list[dict] = []
    t_start = time.time()

    for i, sr in enumerate(runs):
        if progress:
            elapsed = time.time() - t_start
            eta = (elapsed / i * (n - i)) if i > 0 else 0.0
            print(f"[sweep {sweep.name}] {i + 1}/{n}  run={sr.run_id}  "
                  f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)

        run_dir = runs_dir / sr.run_id
        result = run_scenario(sr.scenario, run_dir, render=render)

        row = {
            "run_id": sr.run_id,
            "seed": sr.seed,
            **{f"param.{k}": v for k, v in sr.params.items()},
            **{f"kpi.{k}": v for k, v in result.kpis.aggregate.items()},
            **{f"timing.{k}": v for k, v in result.timings.items() if not isinstance(v, dict)},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_path = out_dir / "summary.parquet"
    df.to_parquet(summary_path, index=False)

    total = time.time() - t_start
    summary_json = {
        "sweep": sweep.name,
        "n_runs": n,
        "total_time_s": round(total, 2),
        "mean_time_per_run_s": round(total / max(n, 1), 2),
        "kpi_means": {
            c.removeprefix("kpi."): float(df[c].dropna().mean())
            for c in df.columns
            if c.startswith("kpi.") and pd.api.types.is_numeric_dtype(df[c])
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    return SweepResult(
        sweep=sweep, out_dir=out_dir, summary_path=summary_path,
        n_runs=n, total_time_s=total,
    )
