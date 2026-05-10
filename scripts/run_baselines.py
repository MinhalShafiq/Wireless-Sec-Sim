"""CLI: head-to-head optimizer comparison on a single study config.

Runs the same study (same scenario, same threat, same search space, same
objective, same n_trials) once per optimizer, and writes:

- ``<out>/<optimizer>/`` — full study artefacts per optimizer.
- ``<out>/comparison.json`` — best-objective summary.
- ``<out>/plot_convergence.png`` — overlaid running-best curves.

Usage:
    python -m scripts.run_baselines configs/studies/munich_bs_perf_sec.yaml \
        --optimizers tpe cmaes random \
        --out data/baselines/munich_bs_perf_sec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from libs.optim import StudyCfg, run_study  # noqa: E402
from libs.optim.objective import StudyEvaluator  # noqa: E402
from libs.schemas import Scenario  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Compare optimizers on the same study.")
    p.add_argument("config", help="Study YAML")
    p.add_argument("--optimizers", nargs="+", default=["tpe", "cmaes", "random"])
    p.add_argument("--out", required=True, help="Output directory.")
    args = p.parse_args()

    cfg = StudyCfg.from_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build the evaluator once and reuse across optimizers.
    base = Scenario.from_yaml(cfg.base)
    evaluator = StudyEvaluator(base)

    summary = {"study": cfg.name, "n_trials": cfg.n_trials, "direction": cfg.objective.direction,
               "results": {}}

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"tpe": "C0", "cmaes": "C1", "random": "C2"}

    for opt in args.optimizers:
        sub_cfg = cfg.model_copy(deep=True)
        sub_cfg.optimizer = opt  # type: ignore[assignment]
        sub_cfg.name = f"{cfg.name}__{opt}"
        sub_out = out / opt
        print(f"\n=== {opt.upper()} ===")
        result = run_study(sub_cfg, sub_out, evaluator=evaluator)
        df = pd.read_parquet(sub_out / "trials.parquet")
        running_best = (
            df["objective"].cummax()
            if cfg.objective.direction == "maximize"
            else df["objective"].cummin()
        )
        ax.plot(df["trial"], running_best, "-", label=opt, color=palette.get(opt))
        summary["results"][opt] = {
            "best_objective": result.best_objective,
            "best_kpis": result.best_kpis,
            "total_time_s": round(result.total_time_s, 2),
        }

    ax.set_xlabel("trial")
    ax.set_ylabel("running best objective")
    ax.set_title(f"Optimizer comparison — {cfg.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (out / "comparison.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=== summary ===")
    for opt, r in summary["results"].items():
        print(f"  {opt:<8s}  best={r['best_objective']:8.3f}  time={r['total_time_s']:6.2f}s")
    print(f"\nwrote {out / 'plot_convergence.png'}")
    print(f"wrote {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
