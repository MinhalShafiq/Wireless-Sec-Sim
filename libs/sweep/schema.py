"""Sweep configuration schema.

A ``SweepCfg`` is a declarative description of a parameter sweep: a base
scenario, a template that says how each grid point becomes a Scenario,
the parameter grid itself, and the random seeds. The cross product of
``grid × seeds`` is the run set.

Templates (Phase 3):

- ``none``    — no adversary; vary base scenario knobs (e.g., seeds).
- ``jammer``  — append a jammer; grid keys: ``position_x``, ``position_y``,
  ``position_z``, ``power_dbm``.
- ``rogue``   — append a rogue gNB; grid keys: ``position_x``,
  ``position_y``, ``position_z``, ``power_dbm`` (optional, mimics base if omitted).

Add new templates in ``libs.sweep.templates`` when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

SweepTemplate = Literal["none", "jammer", "rogue", "legit_tx"]


class SweepCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    base: str = Field(description="Path to base Scenario YAML (relative to repo root).")
    template: SweepTemplate = "none"
    grid: dict[str, list[float | int]] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [42])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepCfg":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def grid_cardinality(self) -> int:
        n = 1
        for v in self.grid.values():
            n *= max(len(v), 1)
        return n * max(len(self.seeds), 1)
