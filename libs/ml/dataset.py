"""ML dataset construction from sweep run artifacts.

Loads ``per_link.parquet`` and ``scenario.json`` from each run directory
and produces:

- **per-RX features** that a real defender could observe at the UE
  (max RX power, second-best, serving margin, sum, n_strong_tx, …) —
  *no* TX role information is used so the classifier doesn't cheat.
- **per-scenario features** by aggregating the per-RX features (mean,
  std, min, max, etc.) into one feature vector per scenario.
- **labels** derived from the scenario's TX/RX roles (whether any
  adversary is present); these are *only* used during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from libs.schemas import Scenario

# ---------------------------------------------------------------------------
# Feature column names (kept stable so models can be saved/reloaded reliably).
# ---------------------------------------------------------------------------

PER_RX_FEATURES: list[str] = [
    "best_rx_power_dbm",
    "second_rx_power_dbm",
    "serving_margin_db",
    "sum_rx_power_dbm",
    "interferer_power_dbm",
    "n_strong_tx",            # # TXs with rx_power_dbm > -100 dBm
    "max_path_gain_db",
    "min_path_gain_db",
    "path_gain_spread_db",
    "max_tx_distance_m",
    "min_tx_distance_m",
    "n_tx_visible",
]

# Aggregations applied per-scenario over the per-RX feature columns.
_AGGS = ("mean", "std", "min", "max", "median")

SCENARIO_FEATURES: list[str] = (
    [f"{f}_{a}" for f in PER_RX_FEATURES for a in _AGGS]
    + ["n_users", "n_tx", "n_links_with_signal"]
)


@dataclass
class Run:
    run_id: str
    scenario: Scenario
    per_link: pd.DataFrame
    label: int


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> Run:
    """Load one per-run directory into a :class:`Run`."""
    run_dir = Path(run_dir)
    scenario = Scenario.model_validate_json((run_dir / "scenario.json").read_text())
    per_link = pd.read_parquet(run_dir / "per_link.parquet")
    label = label_for_scenario(scenario)
    return Run(run_id=run_dir.name, scenario=scenario, per_link=per_link, label=label)


def label_for_scenario(scenario: Scenario) -> int:
    """1 if the scenario contains any adversary (jammer or rogue), else 0."""
    has_adv = any(t.role in ("jammer", "rogue") for t in scenario.transmitters)
    return 1 if has_adv else 0


def discover_runs(*roots: Path | str) -> list[Path]:
    """Find every per-run directory under any of the given sweep roots.

    Each ``root`` can be either a sweep dir (``runs/<run_id>/`` underneath)
    or a single-run dir directly.
    """
    runs: list[Path] = []
    for r in roots:
        r = Path(r)
        if (r / "scenario.json").exists():
            runs.append(r)
            continue
        runs_dir = r / "runs"
        if runs_dir.exists():
            for sub in sorted(runs_dir.iterdir()):
                if (sub / "scenario.json").exists():
                    runs.append(sub)
    return runs


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

_STRONG_TX_THRESHOLD_DBM = -100.0
_VISIBLE_TX_THRESHOLD_DBM = -120.0


def _per_rx_features(per_link: pd.DataFrame) -> pd.DataFrame:
    """Compute role-free per-RX features from the long-format per-link table."""
    feats: dict[str, list[float]] = {f: [] for f in PER_RX_FEATURES}
    rx_names: list[str] = []

    for rx_name, group in per_link.groupby("rx_name", sort=False):
        rx_pow = group["rx_power_dbm"].to_numpy(dtype=np.float64)
        path_gain = group["path_gain_db"].to_numpy(dtype=np.float64)
        dist = group["distance_m"].to_numpy(dtype=np.float64)
        rx_finite = rx_pow[np.isfinite(rx_pow)]
        pg_finite = path_gain[np.isfinite(path_gain)]

        if rx_finite.size == 0:
            continue

        sorted_desc = np.sort(rx_finite)[::-1]
        best = float(sorted_desc[0])
        second = float(sorted_desc[1]) if sorted_desc.size > 1 else float("nan")

        # Linear-domain sum and interference.
        rx_w = 10 ** ((rx_finite - 30) / 10)
        sum_dbm = 10 * np.log10(rx_w.sum()) + 30
        interf_w = max(rx_w.sum() - 10 ** ((best - 30) / 10), 1e-30)
        interf_dbm = 10 * np.log10(interf_w) + 30

        rx_names.append(rx_name)
        feats["best_rx_power_dbm"].append(best)
        feats["second_rx_power_dbm"].append(second)
        feats["serving_margin_db"].append(best - second if np.isfinite(second) else 80.0)
        feats["sum_rx_power_dbm"].append(float(sum_dbm))
        feats["interferer_power_dbm"].append(float(interf_dbm))
        feats["n_strong_tx"].append(int((rx_finite > _STRONG_TX_THRESHOLD_DBM).sum()))
        feats["max_path_gain_db"].append(float(pg_finite.max()) if pg_finite.size else float("nan"))
        feats["min_path_gain_db"].append(float(pg_finite.min()) if pg_finite.size else float("nan"))
        feats["path_gain_spread_db"].append(
            float(np.ptp(pg_finite)) if pg_finite.size > 1 else 0.0
        )
        feats["max_tx_distance_m"].append(float(dist.max()))
        feats["min_tx_distance_m"].append(float(dist.min()))
        feats["n_tx_visible"].append(int((rx_finite > _VISIBLE_TX_THRESHOLD_DBM).sum()))

    df = pd.DataFrame(feats, index=rx_names)
    df.index.name = "rx_name"
    return df


def extract_scenario_features(scenario: Scenario, per_link: pd.DataFrame) -> dict[str, float]:
    """Build a single scenario-level feature dict from per-link rows."""
    # Restrict to UE rows (eavesdroppers shouldn't be visible to a defender).
    user_rxs = [r.name for r in scenario.receivers if r.role == "user"]
    pl = per_link[per_link["rx_name"].isin(user_rxs)] if user_rxs else per_link
    rx_feats = _per_rx_features(pl)

    out: dict[str, float] = {}
    for col in PER_RX_FEATURES:
        if col not in rx_feats.columns or rx_feats[col].dropna().empty:
            for agg in _AGGS:
                out[f"{col}_{agg}"] = float("nan")
            continue
        vals = rx_feats[col].dropna().to_numpy(dtype=np.float64)
        out[f"{col}_mean"] = float(vals.mean())
        out[f"{col}_std"] = float(vals.std()) if vals.size > 1 else 0.0
        out[f"{col}_min"] = float(vals.min())
        out[f"{col}_max"] = float(vals.max())
        out[f"{col}_median"] = float(np.median(vals))

    out["n_users"] = float(len(user_rxs))
    # The defender doesn't know TX *roles* but does know how many radio
    # sources it's measuring — use total #unique TX names observed.
    out["n_tx"] = float(per_link["tx_name"].nunique())
    out["n_links_with_signal"] = float((per_link["rx_power_dbm"] > -130).sum())
    return out


def build_dataset(*roots: Path | str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build (X, y, run_ids) from one or more sweep / single-run roots."""
    rows: list[dict] = []
    labels: list[int] = []
    ids: list[str] = []
    for run_dir in discover_runs(*roots):
        run = load_run(run_dir)
        feats = extract_scenario_features(run.scenario, run.per_link)
        rows.append(feats)
        labels.append(run.label)
        ids.append(run.run_id)
    if not rows:
        raise FileNotFoundError(f"no runs found under {roots}")
    X = pd.DataFrame(rows, columns=SCENARIO_FEATURES)
    y = pd.Series(labels, name="label", index=ids)
    return X, y, ids
