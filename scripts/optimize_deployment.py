"""CLI: run an optimization study described by a YAML config.

Usage:
    python -m scripts.optimize_deployment configs/studies/munich_bs_perf.yaml
    python -m scripts.optimize_deployment configs/studies/munich_bs_perf_sec.yaml --warm-start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.optim import StudyCfg, run_study  # noqa: E402
from libs.optim.warm_start import warm_start_params  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run an optimization study.")
    p.add_argument("config", help="Path to study YAML")
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--warm-start", action="store_true",
                   help="Force-enable warm start (overrides study config).")
    p.add_argument("--no-warm-start", action="store_true",
                   help="Disable warm start (overrides study config).")
    args = p.parse_args()

    study = StudyCfg.from_yaml(args.config)
    if args.warm_start:
        study.warm_start.enabled = True
    if args.no_warm_start:
        study.warm_start.enabled = False

    out_dir = Path(args.out) if args.out else Path("data/studies") / study.name

    print(f"[study] {study.name}")
    print(f"[study] base       : {study.base}")
    print(f"[study] threat     : {study.threat.type}")
    print(f"[study] optimizer  : {study.optimizer}")
    print(f"[study] n_trials   : {study.n_trials}")
    print(f"[study] direction  : {study.objective.direction}")
    print(f"[study] objective  : "
          + " + ".join(f"{t.weight:+.2f}*{t.name}" for t in study.objective.terms))
    print(f"[study] out        : {out_dir}")
    print()

    seeds = warm_start_params(study) if study.warm_start.enabled else []
    if seeds:
        print(f"[study] warm-start : {len(seeds)} seed trials from vector store")

    result = run_study(study, out_dir, warm_start_params=seeds)

    print()
    print(f"[done] {result.n_trials} trials in {result.total_time_s:.1f}s "
          f"({result.total_time_s / max(result.n_trials, 1):.2f}s/trial)")
    print(f"[done] best objective: {result.best_objective:.3f}")
    print(f"[done] best params:")
    for k, v in result.best_params.items():
        print(f"   {k:30s} {v:10.3f}")
    print(f"[done] best KPIs:")
    for k, v in result.best_kpis.items():
        print(f"   {k:40s} {v}")
    print(f"[done] outputs in {result.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
