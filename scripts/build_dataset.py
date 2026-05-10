"""CLI: build a labelled dataset from one or more sweep / single-run roots.

Usage:
    python -m scripts.build_dataset \
        data/sweeps/legit_bs_position \
        data/sweeps/jammer_position \
        data/sweeps/rogue_position \
        --out data/datasets/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ml.dataset import build_dataset  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Build an ML dataset from sweep outputs.")
    p.add_argument("roots", nargs="+", help="Sweep dirs or per-run dirs to harvest.")
    p.add_argument("--out", required=True, help="Output dataset directory.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[dataset] discovering runs under: {args.roots}")
    X, y, ids = build_dataset(*args.roots)
    print(f"[dataset] {len(X)} runs   features={X.shape[1]}   "
          f"benign={(y == 0).sum()}   attack={(y == 1).sum()}")

    feats = X.copy()
    feats.insert(0, "label", y.values)
    feats.insert(0, "run_id", ids)
    feats.to_parquet(out / "dataset.parquet", index=False)
    print(f"[dataset] wrote {out / 'dataset.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
