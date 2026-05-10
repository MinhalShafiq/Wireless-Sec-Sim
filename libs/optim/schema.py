"""Optimization study configuration.

A ``StudyCfg`` is the declarative description of one optimization run:
a base scenario, an optional fixed threat, the search space (per-TX
parameter ranges), the objective (weighted KPI sum), and the optimizer
choice. The runner consumes it without further code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

OptimizerKind = Literal["tpe", "cmaes", "random"]


class ParamRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    low: float
    high: float


class TxSearchSpace(BaseModel):
    """Per-TX continuous ranges to search over. All keys are optional."""
    model_config = ConfigDict(extra="forbid")
    position_x: ParamRange | None = None
    position_y: ParamRange | None = None
    position_z: ParamRange | None = None
    power_dbm: ParamRange | None = None


class ThreatCfg(BaseModel):
    """Optional fixed threat injected into every trial's scenario."""
    model_config = ConfigDict(extra="forbid")
    type: Literal["none", "jammer", "rogue"] = "none"
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 8.0})
    power_dbm: float | None = None


class ObjectiveTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    weight: float = 1.0


class ObjectiveCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["maximize", "minimize"] = "maximize"
    terms: list[ObjectiveTerm] = Field(default_factory=list)


class WarmStartCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    embedder: str | None = None
    store: str | None = None
    k: int = 5


class StudyCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    base: str = Field(description="Path to the base Scenario YAML.")
    threat: ThreatCfg = Field(default_factory=ThreatCfg)
    search_space: dict[str, TxSearchSpace] = Field(
        default_factory=dict,
        description="Map of TX name → continuous ranges over its mutable params.",
    )
    objective: ObjectiveCfg = Field(default_factory=ObjectiveCfg)
    n_trials: int = 30
    optimizer: OptimizerKind = "tpe"
    seed: int = 42
    warm_start: WarmStartCfg = Field(default_factory=WarmStartCfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StudyCfg":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
