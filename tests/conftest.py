"""Shared pytest config: auto-skip `gpu` / `network` tests when the resource is missing."""
from __future__ import annotations

import os
import socket
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

GTAB_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = GTAB_ROOT / "envs" / "gpu" / ".venv" / "Scripts" / "python.exe"


@lru_cache(maxsize=1)
def _gpu_available() -> bool:
    if not GPU_PYTHON.exists():
        return False
    try:
        out = subprocess.run(
            [str(GPU_PYTHON), "-c", "import torch;print(torch.cuda.is_available())"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.strip().endswith("True")


@lru_cache(maxsize=1)
def _network_available() -> bool:
    if os.environ.get("GTAB_OFFLINE"):
        return False
    try:
        with socket.create_connection(("www.youtube.com", 443), timeout=3):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not _gpu_available():
            item.add_marker(pytest.mark.skip(reason="GPU worker env / CUDA not available"))
        if "network" in item.keywords and not _network_available():
            item.add_marker(pytest.mark.skip(reason="network not available"))
