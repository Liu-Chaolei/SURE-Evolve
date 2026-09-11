#!/usr/bin/env python3
"""Compatibility shim for the integrated Xcientist SurveyAgent phase.

The supported XLab entrypoint is scripts/run_survey_phase.py, which builds
run-local state and artifacts without an external checkout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    entrypoint = Path(__file__).resolve().parents[2] / "run_survey_phase.py"
    completed = subprocess.run([sys.executable, str(entrypoint), *sys.argv[1:]], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
