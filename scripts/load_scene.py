"""CLI: load a built-in scene, place a transmitter, render a radio map.

Usage:
    python -m scripts.load_scene --scene munich --out data/runs/munich_coverage.png

Sionna RT 2.x API: Scene + RadioMapSolver -> PlanarRadioMap.show().
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.scenes import BUILTIN_SCENES, load_builtin_scene  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Load a Sionna RT scene and render a radio map.")
    p.add_argument(
        "--scene",
        default="munich",
        choices=sorted(BUILTIN_SCENES),
        help="Built-in scene key (default: munich).",
    )
    p.add_argument("--frequency", type=float, default=3.5e9, help="Carrier frequency in Hz.")
    p.add_argument("--tx-height", type=float, default=25.0, help="Transmitter height (m).")
    p.add_argument(
        "--out",
        default="data/runs/coverage.png",
        help="Output PNG for the radio map.",
    )
    p.add_argument("--cell-size", type=float, default=2.0, help="Radio map cell size (m).")
    p.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max ray-tracing interaction depth (reflections).",
    )
    p.add_argument(
        "--samples-per-tx",
        type=int,
        default=1_000_000,
        help="Samples per transmitter for the radio map.",
    )
    args = p.parse_args()

    info = BUILTIN_SCENES[args.scene]
    print(f"[scene] loading '{info.key}' — {info.description}")
    t0 = time.time()
    scene = load_builtin_scene(args.scene)
    print(f"[scene] loaded in {time.time() - t0:.2f}s")

    import mitsuba as mi
    import sionna.rt as rt

    scene.frequency = args.frequency

    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    bbox = scene.mi_scene.bbox()
    cx = float((bbox.min.x + bbox.max.x) * 0.5)
    cy = float((bbox.min.y + bbox.max.y) * 0.5)
    tx_pos = mi.Point3f(cx, cy, args.tx_height)
    print(f"[scene] tx position: ({cx:.1f}, {cy:.1f}, {args.tx_height})")

    tx = rt.Transmitter(name="tx0", position=tx_pos)
    scene.add(tx)

    print(
        f"[trace] computing radio map @ {args.frequency / 1e9:.2f} GHz "
        f"(cell={args.cell_size}m, depth={args.max_depth}, samples={args.samples_per_tx:,})…"
    )
    solver = rt.RadioMapSolver()
    t0 = time.time()
    rm = solver(
        scene=scene,
        cell_size=mi.Point2f(args.cell_size, args.cell_size),
        max_depth=args.max_depth,
        samples_per_tx=args.samples_per_tx,
    )
    print(f"[trace] done in {time.time() - t0:.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = rm.show(metric="path_gain", tx=0)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[out]   radio map written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
