"""Background loop: re-runs export_dashboard_data.py every N seconds.

Keeps data.json fresh (Elo, recent matches, model metrics, predictions)
without hammering external APIs. Run alongside the live tracker.

Usage:
    python refresh_loop.py [interval_seconds]   # default 120
"""
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent
VENV_PY = REPO / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def run_export():
    result = subprocess.run(
        [PYTHON, "-m", "cricket_pipeline.work.export_dashboard_data"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"[refresh_loop] export failed: {result.stderr[-300:]}", flush=True)
    else:
        print(f"[refresh_loop] data.json refreshed", flush=True)


print(f"[refresh_loop] Starting — interval {INTERVAL}s", flush=True)
run_export()  # run immediately on start
while True:
    time.sleep(INTERVAL)
    run_export()
