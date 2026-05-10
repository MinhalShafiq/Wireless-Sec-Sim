"""Run a study: enumerate trials via Optuna, evaluate, persist results.

The runner is deliberately decoupled from disk: it asks Optuna for the
next set of parameter values, asks the StudyEvaluator to score them,
and reports back. All artifacts (trials.parquet, best.json,
plot_convergence.png) are written at the end.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from libs.optim.objective import StudyEvaluator
from libs.optim.schema import StudyCfg, TxSearchSpace
from libs.schemas import Scenario, TransmitterCfg, Vec3
from libs.sim_engine.adversaries import add_jammer, add_rogue_gnb


@dataclass
class StudyResult:
    out_dir: Path
    best_params: dict[str, float]
    best_objective: float
    best_kpis: dict[str, Any]
    n_trials: int
    total_time_s: float


# ---------------------------------------------------------------------------
# Scenario assembly
# ---------------------------------------------------------------------------

def _apply_threat(scenario: Scenario, threat) -> Scenario:
    if threat.type == "none":
        return scenario
    pos = (
        float(threat.position.get("x", 0.0)),
        float(threat.position.get("y", 0.0)),
        float(threat.position.get("z", 8.0)),
    )
    if threat.type == "jammer":
        power = threat.power_dbm if threat.power_dbm is not None else 50.0
        return add_jammer(scenario, pos, power_dbm=power, name="jammer0")
    if threat.type == "rogue":
        out = add_rogue_gnb(scenario, pos, name="rogue0")
        if threat.power_dbm is not None:
            out.transmitters[-1].power_dbm = threat.power_dbm
        return out
    raise ValueError(f"unknown threat type: {threat.type!r}")


def _scenario_from_params(base: Scenario, study: StudyCfg, params: dict[str, float]) -> Scenario:
    """Build a Scenario for a trial: clone base, apply param overrides, inject threat."""
    s = base.model_copy(deep=True)
    s.name = "trial"
    s.propagation.seed = study.seed
    by_name = {t.name: t for t in s.transmitters}

    for tx_name, _ in study.search_space.items():
        if tx_name not in by_name:
            raise KeyError(f"search_space targets unknown TX {tx_name!r}")
        tx = by_name[tx_name]
        if (v := params.get(f"{tx_name}.position_x")) is not None:
            tx.position.x = float(v)
        if (v := params.get(f"{tx_name}.position_y")) is not None:
            tx.position.y = float(v)
        if (v := params.get(f"{tx_name}.position_z")) is not None:
            tx.position.z = float(v)
        if (v := params.get(f"{tx_name}.power_dbm")) is not None:
            tx.power_dbm = float(v)

    return _apply_threat(s, study.threat)


# ---------------------------------------------------------------------------
# Optuna glue
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, search_space: dict[str, TxSearchSpace]) -> dict[str, float]:
    params: dict[str, float] = {}
    for tx_name, sp in search_space.items():
        for var in ("position_x", "position_y", "position_z", "power_dbm"):
            rng = getattr(sp, var)
            if rng is None:
                continue
            params[f"{tx_name}.{var}"] = trial.suggest_float(f"{tx_name}.{var}", rng.low, rng.high)
    return params


def _make_sampler(study: StudyCfg) -> optuna.samplers.BaseSampler:
    if study.optimizer == "tpe":
        return optuna.samplers.TPESampler(seed=study.seed)
    if study.optimizer == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=study.seed)
    if study.optimizer == "random":
        return optuna.samplers.RandomSampler(seed=study.seed)
    raise ValueError(f"unknown optimizer {study.optimizer!r}")


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def _plot_convergence(trials_df: pd.DataFrame, direction: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    if direction == "maximize":
        running_best = trials_df["objective"].cummax()
    else:
        running_best = trials_df["objective"].cummin()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials_df["trial"], trials_df["objective"], "o-", alpha=0.4, label="trial")
    ax.plot(trials_df["trial"], running_best, "r-", linewidth=2, label=f"running best ({direction})")
    ax.set_xlabel("trial")
    ax.set_ylabel("objective")
    ax.set_title("Optimization convergence")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_study(
    study: StudyCfg, out_dir: str | Path, *, evaluator: StudyEvaluator | None = None,
    progress: bool = True, warm_start_params: list[dict] | None = None,
) -> StudyResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Scenario.from_yaml(study.base)
    weights = {t.name: t.weight for t in study.objective.terms}

    if evaluator is None:
        evaluator = StudyEvaluator(base)

    sampler = _make_sampler(study)
    optuna_study = optuna.create_study(direction=study.objective.direction, sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if warm_start_params:
        for p in warm_start_params:
            optuna_study.enqueue_trial(p)

    rows: list[dict[str, Any]] = []
    t_start = time.time()

    def _objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, study.search_space)
        scenario = _scenario_from_params(base, study, params)
        result = evaluator.evaluate(scenario, weights)
        row = {
            "trial": trial.number,
            "objective": result.objective,
            **{f"param.{k}": v for k, v in params.items()},
            **{f"kpi.{k}": v for k, v in result.kpis.items() if not isinstance(v, dict)},
        }
        rows.append(row)
        if progress:
            elapsed = time.time() - t_start
            best = (
                max(r["objective"] for r in rows)
                if study.objective.direction == "maximize"
                else min(r["objective"] for r in rows)
            )
            print(
                f"[study {study.name}] trial {trial.number + 1}/{study.n_trials}  "
                f"obj={result.objective:8.3f}  best={best:8.3f}  elapsed={elapsed:5.1f}s",
                flush=True,
            )
        return result.objective

    optuna_study.optimize(_objective, n_trials=study.n_trials)

    total = time.time() - t_start
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "trials.parquet", index=False)

    best = optuna_study.best_trial
    best_kpis = {k: v for k, v in rows[best.number].items() if k.startswith("kpi.")}
    (out_dir / "best.json").write_text(json.dumps({
        "params": best.params,
        "objective": float(best.value),
        "trial_index": int(best.number),
        "kpis": best_kpis,
        "direction": study.objective.direction,
        "n_trials": int(study.n_trials),
        "optimizer": study.optimizer,
    }, indent=2))
    (out_dir / "study.json").write_text(study.model_dump_json(indent=2))

    _plot_convergence(df, study.objective.direction, out_dir / "plot_convergence.png")

    return StudyResult(
        out_dir=out_dir,
        best_params=dict(best.params),
        best_objective=float(best.value),
        best_kpis=best_kpis,
        n_trials=study.n_trials,
        total_time_s=total,
    )
