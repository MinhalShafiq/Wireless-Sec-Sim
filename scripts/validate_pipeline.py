"""Cross-phase validation report.

Walks the on-disk artefacts produced by phases 1–6 and checks the key
numerical invariants:

  Phase 1: Munich scene loads with sensible bbox extents.
  Phase 2: FSPL run agrees with Friis to within 0.5 dB.
  Phase 3: All three reference sweeps populated; runs/sec sane.
  Phase 4: Trained classifier metrics + coverage RMSE under threshold.
  Phase 5: Vector store populated; precision@5 above threshold.
  Phase 6: Both reference studies produced a best.json with sane KPIs.

Emits a structured JSON report and a one-line textual summary, suitable
for CI and for inclusion in the technical report.

Usage:
    python -m scripts.validate_pipeline --out data/validation/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    note: str | None = None


def _agg(path: Path) -> dict:
    return json.loads(path.read_text())["aggregate"]


def check_phase1() -> Check:
    """Munich loads with realistic bbox extents."""
    try:
        import numpy as np  # noqa: F401

        from libs.scenes import load_builtin_scene
        scene = load_builtin_scene("munich")
        bbox = scene.mi_scene.bbox()
        ext = (
            float(bbox.max.x - bbox.min.x),
            float(bbox.max.y - bbox.min.y),
            float(bbox.max.z - bbox.min.z),
        )
        ok = ext[0] > 500 and ext[1] > 500 and ext[2] > 50
        return Check(
            "phase1.scene_load",
            ok,
            detail={"munich_extent_m": [round(v, 1) for v in ext], "n_objects": len(scene.objects)},
        )
    except Exception as e:  # noqa: BLE001
        return Check("phase1.scene_load", False, note=str(e))


def check_phase2() -> Check:
    """FSPL run on disk agrees with Friis."""
    run = ROOT / "data" / "runs" / "fspl_validation"
    if not run.exists():
        return Check("phase2.fspl_agreement", False, note="data/runs/fspl_validation not found — run scripts.run_scenario configs/fspl_validation.yaml")
    import numpy as np
    import pandas as pd

    df = pd.read_parquet(run / "per_link.parquet")
    scenario = json.loads((run / "scenario.json").read_text())
    f = scenario["band"]["carrier_hz"]
    lam = 3e8 / f
    df = df[df["tx_name"] == "tx0"]
    expected = -20 * np.log10(4 * np.pi * df["distance_m"] / lam)
    err = (df["path_gain_db"].to_numpy() - expected.to_numpy())
    max_err = float(np.max(np.abs(err)))
    return Check(
        "phase2.fspl_agreement",
        max_err < 0.5,
        detail={"max_abs_error_db": round(max_err, 4), "tolerance_db": 0.5, "n_links": int(len(df))},
    )


def check_phase3() -> Check:
    sweeps = {
        "legit_bs_position": 50,
        "jammer_position": 25,
        "rogue_position": 18,
    }
    out: dict[str, Any] = {}
    ok = True
    for name, expected in sweeps.items():
        d = ROOT / "data" / "sweeps" / name
        runs_dir = d / "runs"
        n = sum(1 for p in runs_dir.iterdir() if p.is_dir() and (p / "scenario.json").exists()) if runs_dir.exists() else 0
        out[name] = {"runs": n, "expected": expected}
        if n != expected:
            ok = False
    return Check("phase3.sweeps_populated", ok, detail=out)


def check_phase4() -> Check:
    out: dict[str, Any] = {}
    metrics_path = ROOT / "data" / "models" / "security_clf_v1" / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        out["classifier"] = {"accuracy": m["accuracy"], "f1": m["f1"], "n_test": m["n_test"]}
        clf_ok = m["accuracy"] >= 0.85 and m["f1"] >= 0.85
    else:
        out["classifier"] = "missing"
        clf_ok = False

    cov_path = ROOT / "data" / "models" / "coverage_predictor_v1" / "metrics.json"
    if cov_path.exists():
        c = json.loads(cov_path.read_text())
        out["coverage_regressor"] = {"rmse_db": c["rmse"], "mae_db": c["mae"], "n_test": c["n_test"]}
        cov_ok = c["rmse"] < 5.0
    else:
        out["coverage_regressor"] = "missing"
        cov_ok = False

    return Check("phase4.ml_models", clf_ok and cov_ok, detail=out)


def check_phase5() -> Check:
    """Hold-one-out precision@5 on the loaded vector store."""
    embedder_dir = ROOT / "data" / "embedders" / "v1"
    store_dir = ROOT / "data" / "vector_store"
    if not (embedder_dir.exists() and store_dir.exists()):
        return Check("phase5.retrieval_quality", False, note="run scripts.ingest_runs first")
    from libs.ml.dataset import discover_runs
    from libs.retrieval import ScenarioEmbedder, ScenarioStore, precision_at_k

    embedder = ScenarioEmbedder.load(embedder_dir)
    store = ScenarioStore(path=str(store_dir), dim=embedder.dim)
    runs = discover_runs(
        ROOT / "data" / "sweeps" / "legit_bs_position",
        ROOT / "data" / "sweeps" / "jammer_position",
        ROOT / "data" / "sweeps" / "rogue_position",
    )
    p = precision_at_k(embedder, store, runs, k=5)
    return Check(
        "phase5.retrieval_quality",
        p["precision_at_k_mean"] >= 0.7,
        detail={"precision_at_5_mean": round(p["precision_at_k_mean"], 4),
                "n_queries": p["n_queries"], "store_size": store.count()},
    )


def check_phase6() -> Check:
    out: dict[str, Any] = {}
    ok = True
    for name in ("munich_bs_perf", "munich_bs_perf_sec"):
        bp = ROOT / "data" / "studies" / name / "best.json"
        if not bp.exists():
            out[name] = "missing"
            ok = False
            continue
        b = json.loads(bp.read_text())
        sinr = b["kpis"].get("kpi.median_user_sinr_db")
        out[name] = {"best_objective": b["objective"],
                     "best_median_sinr_db": sinr,
                     "n_trials": b["n_trials"], "optimizer": b["optimizer"]}
    perf = out.get("munich_bs_perf", {})
    perf_sec = out.get("munich_bs_perf_sec", {})
    if isinstance(perf, dict) and perf.get("best_median_sinr_db") is not None:
        ok = ok and perf["best_median_sinr_db"] >= 30
    if isinstance(perf_sec, dict) and perf_sec.get("best_median_sinr_db") is not None:
        # Under jammer, just require a positive recovery margin
        ok = ok and perf_sec["best_median_sinr_db"] > 0
    return Check("phase6.studies", ok, detail=out)


def run_all() -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    t0 = time.time()
    for fn in (check_phase1, check_phase2, check_phase3, check_phase4, check_phase5, check_phase6):
        c = fn()
        print(f"[{('OK' if c.passed else 'FAIL'):4s}] {c.name}  {c.detail or c.note or ''}")
        checks.append(c)
    summary = {
        "passed": sum(1 for c in checks if c.passed),
        "total": len(checks),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"\n[summary] {summary['passed']}/{summary['total']} passed in {summary['elapsed_s']}s")
    return checks, summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/validation/report.json")
    args = p.parse_args()

    checks, summary = run_all()
    report = {
        "summary": summary,
        "checks": [asdict(c) for c in checks],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[wrote] {out}")
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
