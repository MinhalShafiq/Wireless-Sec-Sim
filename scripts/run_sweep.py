"""CLI: run a parameter sweep described by a YAML config.

Usage:
    python -m scripts.run_sweep configs/sweeps/jammer_position.yaml
    python -m scripts.run_sweep configs/sweeps/rogue_position.yaml --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.sweep import SweepCfg, run_sweep  # noqa: E402
from libs.sweep.plot import auto_plots  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run a parameter sweep.")
    p.add_argument("config", help="Path to sweep YAML")
    p.add_argument("--out", default=None, help="Output directory (default: data/sweeps/<name>)")
    p.add_argument("--render", action="store_true", help="Enable per-run 3D rendering (slow).")
    p.add_argument("--no-plots", action="store_true", help="Skip sweep-level plot generation.")
    args = p.parse_args()

    sweep = SweepCfg.from_yaml(args.config)
    out_dir = Path(args.out) if args.out else Path("data/sweeps") / sweep.name

    n = sweep.grid_cardinality()
    print(f"[sweep] {sweep.name}")
    print(f"[sweep] base     : {sweep.base}")
    print(f"[sweep] template : {sweep.template}")
    print(f"[sweep] grid     : {sweep.grid}")
    print(f"[sweep] seeds    : {sweep.seeds}")
    print(f"[sweep] runs     : {n}")
    print(f"[sweep] out      : {out_dir}")
    print()

    result = run_sweep(sweep, out_dir, render=args.render)

    print()
    print(f"[done] {result.n_runs} runs in {result.total_time_s:.1f}s "
          f"({result.total_time_s / max(result.n_runs, 1):.2f}s/run)")
    print(f"[done] summary  : {result.summary_path}")

    if not args.no_plots:
        plots = auto_plots(result.summary_path)
        if plots:
            print("[plot] wrote:")
            for pp in plots:
                print(f"   {pp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
