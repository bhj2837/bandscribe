"""Filesystem layout and child-process environment.

Everything durable lives under ``ROOT``: the env var BANDSCRIBE_ROOT if set, else the folder holding this package
(the repo root). Nothing goes to %APPDATA%/%LOCALAPPDATA%:
processes launched from the Claude desktop app get those paths redirected into the app sandbox,
so files written there would be invisible from the user's own terminal.

Importable from both the core and the GPU worker env (stdlib only).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(os.environ.get("BANDSCRIBE_ROOT") or Path(__file__).resolve().parents[1])

DATA: Path = ROOT / "data"
TOOLS: Path = ROOT / "tools"
MODELS: Path = DATA / "models"
JOBS: Path = DATA / "jobs"
DOWNLOADS: Path = DATA / "downloads"
LOGS: Path = DATA / "logs"
HF_HOME: Path = DATA / "hf"
TORCH_HOME: Path = DATA / "torch"
UV_CACHE: Path = DATA / "uv-cache"

CORE_PYTHON: Path = ROOT / ".venv" / "Scripts" / "python.exe"
GPU_PYTHON: Path = ROOT / "envs" / "gpu" / ".venv" / "Scripts" / "python.exe"
# Basic Pitch env (Python 3.10, onnxruntime). bandscribe is not installed there; workers import it via PYTHONPATH.
BP310_PYTHON: Path = ROOT / "envs" / "bp310" / ".venv" / "Scripts" / "python.exe"
NODE_DIR: Path = ROOT / "node"  # package.json + node_modules (alphaTab) for GT import / render

# M1a/M2 data layout (M1_M2_SPEC 1.2).
EVAL: Path = DATA / "eval"  # tierA/, tierB/, external/, reftx_cache/
RUNS: Path = DATA / "runs"  # evaluation run directories
DATASETS: Path = DATA / "datasets"  # <name>/raw|extracted|local + datasets.lock.json
BENCH: Path = DATA / "bench"  # bandscribe bench gpu output + vram_table.json
MODELS_LOCK: Path = MODELS / "models.lock.json"
VRAM_TABLE: Path = BENCH / "vram_table.json"
# npm's default cache is under %LOCALAPPDATA%, which the app sandbox redirects -> keep it under DATA.
NPM_CACHE: Path = DATA / "npm-cache"

YTDLP_EXE: Path = TOOLS / "yt-dlp.exe"
YTDLP_CONF: Path = TOOLS / "yt-dlp.conf"

GPU_LOCK: Path = DATA / "gpu.lock"
USER_CONFIG: Path = DATA / "config.toml"

_DIRS = (DATA, MODELS, JOBS, DOWNLOADS, LOGS, HF_HOME, TORCH_HOME, UV_CACHE, EVAL, RUNS, DATASETS, BENCH, NPM_CACHE)


def _managed_env() -> dict[str, str]:
    return {
        "BANDSCRIBE_ROOT": str(ROOT),
        "HF_HOME": str(HF_HOME),
        "TORCH_HOME": str(TORCH_HOME),
        "UV_CACHE_DIR": str(UV_CACHE),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        # synctoolbox imports matplotlib.pyplot at import time and reports plot with matplotlib: never let
        # either look for a GUI backend (workers, node children and, via apply_process_env, the CLI itself).
        "MPLBACKEND": "Agg",
    }


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for every subprocess we start (workers, ffmpeg, yt-dlp, node)."""
    env = dict(os.environ)
    env.update(_managed_env())
    if extra:
        env.update(extra)
    return env


def apply_process_env() -> None:
    """Apply the managed variables to the current process (called once by the CLI)."""
    os.environ.update(_managed_env())


def ensure_dirs() -> None:
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)
