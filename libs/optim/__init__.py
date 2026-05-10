"""Phase 6 — optimization & security modules."""

import libs.scenes  # noqa: F401  # ensure Mitsuba variant set first

from libs.optim.objective import StudyEvaluator, TrialResult
from libs.optim.schema import StudyCfg
from libs.optim.study import StudyResult, run_study
from libs.optim.warm_start import warm_start_params

__all__ = [
    "StudyCfg",
    "StudyEvaluator",
    "StudyResult",
    "TrialResult",
    "run_study",
    "warm_start_params",
]
