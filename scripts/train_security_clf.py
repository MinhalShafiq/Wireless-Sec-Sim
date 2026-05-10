"""CLI: train the security classifier from a dataset built by ``build_dataset``.

Usage:
    python -m scripts.train_security_clf data/datasets/v1 --out data/models/security_clf_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from libs.ml.dataset import SCENARIO_FEATURES  # noqa: E402
from libs.ml.security_clf import save_model, train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Train the security classifier.")
    p.add_argument("dataset_dir", help="Dir containing dataset.parquet")
    p.add_argument("--out", required=True, help="Where to save the trained model.")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_parquet(Path(args.dataset_dir) / "dataset.parquet")
    X = df[SCENARIO_FEATURES]
    y = df["label"]
    print(f"[train] dataset rows: {len(df)}   benign={(y == 0).sum()}   attack={(y == 1).sum()}")

    pipe, report = train(X, y, test_size=args.test_size, random_state=args.seed)

    print(f"[train] accuracy: {report.accuracy:.3f}")
    print(f"[train] F1:       {report.f1:.3f}")
    print(f"[train] ROC AUC:  {report.roc_auc}")
    print(f"[train] confusion (rows=true, cols=pred):")
    print(f"   benign: {report.confusion[0]}")
    print(f"   attack: {report.confusion[1]}")
    print(f"[train] top features:")
    for k, v in list(report.feature_importances.items())[:10]:
        print(f"   {k:40s} {v:.4f}")

    save_model(pipe, report, Path(args.out))
    print(f"[train] saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
