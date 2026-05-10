"""CLI: run the security classifier on a per-run directory.

Usage:
    python -m scripts.predict_attack \
        --model data/models/security_clf_v1 \
        data/runs/munich_jammed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ml.security_clf import load_model, predict_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Predict attack/benign for a run.")
    p.add_argument("run_dir", help="Per-run directory (with scenario.json + per_link.parquet).")
    p.add_argument("--model", required=True, help="Trained model directory.")
    args = p.parse_args()

    pipe = load_model(Path(args.model))
    run_dir = Path(args.run_dir)
    out = predict_run(
        pipe,
        run_dir / "scenario.json",
        run_dir / "per_link.parquet",
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
