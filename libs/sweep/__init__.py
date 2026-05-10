"""Phase 3 — sweep over parameter grids of scenarios."""

import libs.scenes  # noqa: F401  # ensure Mitsuba variant is set first

from libs.sweep.enumerate import SweepRun, enumerate_scenarios
from libs.sweep.runner import SweepResult, run_sweep
from libs.sweep.schema import SweepCfg

__all__ = ["SweepCfg", "SweepRun", "SweepResult", "enumerate_scenarios", "run_sweep"]
