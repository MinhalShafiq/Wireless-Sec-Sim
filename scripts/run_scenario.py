"""CLI: run a Scenario described by a YAML config.

Usage:
    python -m scripts.run_scenario configs/munich_baseline.yaml
    python -m scripts.run_scenario configs/munich_jammed.yaml --out data/runs/munich_jam2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.schemas import Scenario  # noqa: E402
from libs.sim_engine import run_scenario  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run a wireless simulation scenario.")
    p.add_argument("config", help="Path to scenario YAML")
    p.add_argument("--out", default=None, help="Output directory (default: data/runs/<scenario_name>)")
    p.add_argument("--no-render", action="store_true", help="Skip 3D photo-realistic views (faster).")
    args = p.parse_args()

    scenario = Scenario.from_yaml(args.config)
    out_dir = Path(args.out) if args.out else Path("data/runs") / scenario.name

    n_tx = len(scenario.transmitters)
    n_rx = len(scenario.receivers)
    n_adv = sum(1 for t in scenario.transmitters if t.role != "legitimate")
    n_eve = sum(1 for r in scenario.receivers if r.role == "eavesdropper")

    print(f"[run] scenario : {scenario.name}")
    print(f"[run] scene    : {scenario.scene}")
    print(f"[run] band     : {scenario.band.name} @ {scenario.band.carrier_hz / 1e9:.2f} GHz")
    print(f"[run] tx       : {n_tx} ({n_adv} adversarial)")
    print(f"[run] rx       : {n_rx} ({n_eve} eavesdroppers)")
    print(f"[run] out      : {out_dir}")

    result = run_scenario(scenario, out_dir, render=not args.no_render)

    print()
    print("[done] timings (s):")
    for k, v in result.timings.items():
        if isinstance(v, dict):
            print(f"   {k}:")
            for kk, vv in v.items():
                print(f"      {kk:30s} {vv:7.2f}")
        else:
            print(f"   {k:24s} {v:7.3f}")
    print()
    print("[done] aggregate KPIs:")
    for k, v in result.kpis.aggregate.items():
        print(f"   {k:30s} {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
