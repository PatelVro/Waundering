"""Spawn the orchestrator as a fully detached background process.

Use this if you want it to survive your shell exiting. The orchestrator
itself is a regular Python script you can also run interactively.

Usage:
    .venv/Scripts/python.exe -m cricket_pipeline.work.start_orchestrator

Idempotent: if an orchestrator is already running, prints its PID and exits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "cricket_pipeline" / "work" / "runs"
PID_FILE = RUNS_DIR / "orchestrator.pid"
LOG_FILE = RUNS_DIR / "orchestrator.log"


def _is_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in (psutil.Process(pid).name() or "").lower()
    except Exception:
        return False


def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if _is_alive(old_pid):
                print(f"Orchestrator already running (pid {old_pid}). Tail the log:")
                print(f"  tail -f {LOG_FILE}")
                sys.exit(0)
            else:
                print(f"Stale PID {old_pid} — removing.")
                PID_FILE.unlink()
        except Exception:
            pass

    py = sys.executable
    cmd = [py, "-m", "cricket_pipeline.work.orchestrator"]

    log = open(LOG_FILE, "ab", buffering=0)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}

    creation = 0
    if os.name == "nt":
        DETACHED_PROCESS         = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        env=env, close_fds=True,
        creationflags=creation if os.name == "nt" else 0,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"Spawned orchestrator pid {proc.pid}")
    print(f"Log: {LOG_FILE}")
    print(f"Dashboard: http://127.0.0.1:4173/")
    # Brief wait so we can confirm it didn't crash immediately
    time.sleep(2.0)
    if not _is_alive(proc.pid):
        print("WARNING: orchestrator died within 2s. Check the log:")
        try:
            print(LOG_FILE.read_text()[-2000:])
        except Exception: pass
        sys.exit(2)
    print("Running. Safe to close this shell.")


if __name__ == "__main__":
    main()
