"""Phase 7 smoke tests — every dashboard module must be importable.

Streamlit modules execute the entire script on import (page logic at
top level), so a successful import is itself a basic smoke test that
the page builds without exceptions, given the discovery helpers
return empty lists when no data is present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "services" / "dashboard"


def _load_as_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_paths_helper_imports():
    """The shared helper must import without Streamlit being initialised."""
    from services.dashboard import _paths
    assert callable(_paths.list_sweep_dirs)
    assert callable(_paths.list_study_dirs)


@pytest.mark.parametrize(
    "page",
    [
        "Home.py",
        "pages/1_Run_Browser.py",
        "pages/2_Sweep_Results.py",
        "pages/3_Similar_Scenarios.py",
        "pages/4_Optimization_Studies.py",
    ],
)
def test_page_compiles(page: str):
    """Compile each page's source — catches syntax errors and bad imports
    without actually launching Streamlit."""
    path = DASHBOARD_DIR / page
    src = path.read_text()
    compile(src, str(path), "exec")
