"""Convenience launcher: ``python -m scripts.dashboard`` runs the Streamlit app.

Equivalent to ``streamlit run services/dashboard/Home.py``; provided so users
don't need to remember the path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    home = ROOT / "services" / "dashboard" / "Home.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(home), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
