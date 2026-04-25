#!/usr/bin/env bash
# Cricket pipeline installer.
#
# What this does:
#   1. Detects OS (Linux / macOS / Windows-WSL).
#   2. Detects an NVIDIA GPU + the installed CUDA driver version.
#   3. Creates a Python venv at .venv/ (unless --no-venv).
#   4. Installs base CPU-only deps (DuckDB, pandas, sklearn, LightGBM, …).
#   5. Installs PyTorch with the matching CUDA wheel if a GPU was detected,
#      else CPU PyTorch.
#   6. Optionally tries to install LightGBM with GPU/CUDA support — this
#      requires building from source and is best-effort.
#   7. Runs scripts/gpu_check.py to verify the install.
#
# Usage:
#   ./scripts/install.sh                   # auto-detect, recommended
#   ./scripts/install.sh --cpu             # force CPU
#   ./scripts/install.sh --no-venv         # install into the active env
#   ./scripts/install.sh --lightgbm-gpu    # also build LightGBM GPU
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- args ----------
USE_VENV=1
FORCE_CPU=0
LIGHTGBM_GPU=0
for arg in "$@"; do
    case "$arg" in
        --no-venv)        USE_VENV=0 ;;
        --cpu)            FORCE_CPU=1 ;;
        --lightgbm-gpu)   LIGHTGBM_GPU=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

# ---------- helpers ----------
say()  { printf "\033[1;36m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }
fail() { printf "\033[1;31m✘ %s\033[0m\n" "$*"; exit 1; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$*"; }

# ---------- 1. detect OS ----------
OS="$(uname -s)"
say "Detected OS: $OS"

# ---------- 2. detect GPU ----------
GPU_NAME=""
CUDA_VER=""
if [[ "$FORCE_CPU" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    CUDA_VER="$(nvidia-smi --query 2>/dev/null | grep 'CUDA Version' | sed -E 's/.*CUDA Version *: *([0-9]+\.[0-9]+).*/\1/' | head -n1 || true)"
fi
if [[ -n "$GPU_NAME" ]]; then
    ok "GPU detected: $GPU_NAME (driver CUDA $CUDA_VER)"
else
    warn "No NVIDIA GPU detected — installing CPU-only stack."
fi

# ---------- 3. venv ----------
PY="${PYTHON:-python3}"
if [[ "$USE_VENV" -eq 1 ]]; then
    if [[ ! -d .venv ]]; then
        say "Creating venv at .venv/"
        "$PY" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="$(command -v python)"
    say "Using venv $PY"
fi
"$PY" -m pip install --quiet --upgrade pip wheel setuptools

# ---------- 4. base deps (CPU) ----------
say "Installing base dependencies …"
"$PY" -m pip install --quiet -r cricket_pipeline/requirements.txt
ok "Base deps installed."

# ---------- 5. PyTorch (CUDA or CPU) ----------
say "Installing PyTorch …"
TORCH_INDEX=""
if [[ -n "$GPU_NAME" ]]; then
    # Map driver CUDA version → wheel index. PyTorch publishes wheels for
    # specific minor versions only (cu118, cu121, cu124 at time of writing).
    case "$CUDA_VER" in
        12.4*|12.5*|12.6*|12.7*|12.8*|12.9*) TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
        12.1*|12.2*|12.3*)                   TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        11.7*|11.8*|11.9*|12.0*)             TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        *)
            warn "CUDA $CUDA_VER not directly mapped — defaulting to cu121 wheels"
            TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    esac
fi
if [[ -n "$TORCH_INDEX" ]]; then
    "$PY" -m pip install --quiet --index-url "$TORCH_INDEX" torch
else
    "$PY" -m pip install --quiet torch
fi
ok "PyTorch installed."

# ---------- 6. LightGBM GPU (optional, best-effort) ----------
if [[ "$LIGHTGBM_GPU" -eq 1 ]]; then
    say "Building LightGBM with GPU support (best-effort) …"
    if "$PY" -m pip install --quiet --no-binary lightgbm "lightgbm>=4.0.0" \
            --install-option=--gpu 2>/dev/null; then
        ok "LightGBM GPU build OK"
    else
        warn "LightGBM GPU build failed — falling back to CPU build."
        warn "If you want LightGBM-GPU, install OpenCL + Boost dev headers"
        warn "and re-run: pip install lightgbm --install-option=--gpu"
    fi
fi

# ---------- 7. verify ----------
say "Verifying install …"
"$PY" scripts/gpu_check.py || warn "GPU check returned non-zero — see output above."

ok "Done. Try:"
echo "    source .venv/bin/activate"
echo "    python -m cricket_pipeline.pipeline daily-refresh --datasets ipl_json"
echo "    python -m cricket_pipeline.pipeline match-forecast --home … --away … --venue …"
