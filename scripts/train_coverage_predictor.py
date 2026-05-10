"""CLI: train the path-gain regressor from sweep outputs.

Usage:
    python -m scripts.train_coverage_predictor \
        data/sweeps/legit_bs_position \
        --out data/models/coverage_predictor_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.ml.coverage_predictor import build_link_dataset, save_model, train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Train the path-gain regressor.")
    p.add_argument("roots", nargs="+", help="Sweep dirs or per-run dirs to use.")
    p.add_argument("--out", required=True, help="Where to save the trained model.")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"[coverage] harvesting per-link rows from {args.roots} …")
    X, y, src_ids = build_link_dataset(*args.roots)
    print(f"[coverage] rows={len(X)}   "
          f"target range: [{y.min():.1f}, {y.max():.1f}] dB   mean={y.mean():.1f}")

    pipe, report = train(X, y, src_ids, test_size=args.test_size, random_state=args.seed)

    print(f"[coverage] RMSE: {report.rmse:.2f} dB")
    print(f"[coverage] MAE:  {report.mae:.2f} dB")
    print(f"[coverage] n_train={report.n_train}  n_test={report.n_test}")

    save_model(pipe, report, Path(args.out))
    print(f"[coverage] saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
