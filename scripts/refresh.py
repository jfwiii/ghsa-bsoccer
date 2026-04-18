"""
Refresh script — called by GitHub Actions cron.

Thin wrapper around run_full_pipeline that handles playoff-season detection
and failure reporting.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def is_playoff_season() -> bool:
    """Rough heuristic: playoffs run April–May."""
    from datetime import date
    today = date.today()
    return today.month in (4, 5)


def main():
    playoffs = is_playoff_season()
    cmd = [sys.executable, "scripts/run_full_pipeline.py"]
    if playoffs:
        cmd.append("--playoffs")

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode != 0:
        print(f"Pipeline failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print("Refresh complete.", flush=True)


if __name__ == "__main__":
    main()
