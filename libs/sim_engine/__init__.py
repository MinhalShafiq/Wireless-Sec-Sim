"""Phase 2 simulation engine — runs a Scenario end-to-end."""

import libs.scenes  # noqa: F401  # ensure Mitsuba variant is set before sionna.rt loads

from libs.sim_engine.runner import RunResult, run_scenario

__all__ = ["RunResult", "run_scenario"]
