"""Stop a running orchestrator (graceful)."""
from __future__ import annotations

import os, signal, sys, time
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "cricket_pipeline" / "work" / "runs"
PID_FILE = RUNS_DIR / "orchestrator.pid"


def main():
    if not PID_FILE.exists():
        print("No PID file. Orchestrator probably not running.")
        return
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        print("Bad PID file."); return

    import psutil
    if not psutil.pid_exists(pid):
        print(f"PID {pid} not alive. Removing stale PID file.")
        PID_FILE.unlink(); return

    p = psutil.Process(pid)
    print(f"Stopping orchestrator pid {pid} ({p.name()}) ...")
    try:
        p.terminate()
        try:
            p.wait(timeout=8)
            print("Stopped.")
        except psutil.TimeoutExpired:
            print("Did not exit in 8s — killing.")
            p.kill()
    finally:
        try: PID_FILE.unlink()
        except Exception: pass


if __name__ == "__main__":
    main()
