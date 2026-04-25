"""Verify GPU + CUDA install for the cricket pipeline.

Prints:
  * NVIDIA driver / CUDA version (from nvidia-smi)
  * PyTorch CUDA availability + device list
  * LightGBM GPU/CUDA support (probed via a tiny train run)
  * DuckDB version (CPU-only; reported for completeness)

Exit code:
  0 — at least PyTorch reports a usable device (CPU or CUDA)
  1 — something is broken
"""

from __future__ import annotations

import shutil
import subprocess
import sys


GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
DIM = "\033[2m"
RESET = "\033[0m"


def header(msg: str) -> None:
    print(f"\n{DIM}─ {msg} ─{RESET}")


def ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")


def err(msg: str) -> None:
    print(f"{RED}✘{RESET} {msg}")


def check_nvidia_smi() -> dict:
    if shutil.which("nvidia-smi") is None:
        warn("nvidia-smi not found — no NVIDIA GPU on this machine, or driver missing.")
        return {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=15,
        )
    except subprocess.CalledProcessError as e:
        err(f"nvidia-smi failed: {e}")
        return {}
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    info = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        info.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    for i, gpu in enumerate(info):
        ok(f"GPU{i}: {gpu['name']}  driver {gpu['driver']}  vram {gpu['memory']}")
    return {"gpus": info}


def check_pytorch() -> bool:
    try:
        import torch
    except ImportError:
        err("torch not installed.")
        return False
    cuda = torch.cuda.is_available()
    print(f"  torch:        {torch.__version__}")
    print(f"  cuda built:   {torch.version.cuda or 'cpu-only'}")
    print(f"  cuda usable:  {cuda}")
    if cuda:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory // (1 << 20)
            ok(f"  device {i}: {name} ({mem} MiB)")
        # Tiny op to make sure CUDA actually runs
        try:
            x = torch.randn(1024, 1024, device="cuda")
            (x @ x).sum().item()
            ok("  matmul on GPU works.")
        except Exception as e:
            err(f"  CUDA matmul failed: {e}")
            return False
    else:
        warn("  PyTorch is CPU-only — sequence-model training will be ~30x slower.")
    return True


def check_lightgbm() -> None:
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        err("lightgbm not installed.")
        return
    print(f"  lightgbm:     {lgb.__version__}")
    # Probe whether the wheel was built with GPU support by trying to train.
    X = np.random.rand(2000, 8); y = (X.sum(axis=1) > 4).astype(int)
    for device in ("gpu", "cuda"):
        try:
            ds = lgb.Dataset(X, label=y)
            lgb.train({"device": device, "objective": "binary",
                       "verbose": -1, "num_leaves": 7},
                      ds, num_boost_round=5)
            ok(f"  LightGBM accepts device='{device}' — GPU acceleration available.")
            return
        except Exception:
            continue
    warn("  LightGBM is CPU-only (this is fine — GPU LightGBM is a niche speedup).")


def check_duckdb() -> None:
    try:
        import duckdb
    except ImportError:
        err("duckdb not installed.")
        return
    print(f"  duckdb:       {duckdb.__version__}  (CPU only — no GPU path expected)")


def main() -> int:
    header("nvidia-smi")
    check_nvidia_smi()

    header("PyTorch")
    torch_ok = check_pytorch()

    header("LightGBM")
    check_lightgbm()

    header("DuckDB")
    check_duckdb()

    print()
    if torch_ok:
        ok("Pipeline is runnable.")
        return 0
    err("Some core deps are missing — re-run scripts/install.sh.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
