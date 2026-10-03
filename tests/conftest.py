"""Shared pytest config: auto-skip tests whose resource is missing (M0_SPEC 0, M1_M2_SPEC 0).

Markers (registered in pyproject.toml):
- ``gpu``: CUDA works in the gpu env (M0).
- ``gpuenv``: the gpu venv exists (CPU torch is enough).
- ``bp310``: the Basic Pitch venv (Python 3.10) exists.
- ``node``: node is on PATH and ``node/node_modules/@coderline/alphatab`` is installed.
- ``data(name)``: ``gtab.datasets.is_available(name)`` (complete, not partial) and its folder exists.
- ``model(name)``: model ``name`` is recorded in ``data/models/models.lock.json`` and its file exists.
- ``network``: internet reachable (``GTAB_OFFLINE`` forces "no").
- ``slow``: runs by default (as in M0); deselect with ``-m "not slow"``.

The default ``pytest -q`` stays green without GPU, network, datasets or models. Resource checks read the real
``GTAB_ROOT`` layout directly (JSON / file checks) instead of going through gtab modules, so a broken module
can never hide or unhide whole groups of tests.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

GTAB_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = GTAB_ROOT / "envs" / "gpu" / ".venv" / "Scripts" / "python.exe"
BP310_PYTHON = GTAB_ROOT / "envs" / "bp310" / ".venv" / "Scripts" / "python.exe"
ALPHATAB_PKG = GTAB_ROOT / "node" / "node_modules" / "@coderline" / "alphatab" / "package.json"
DATASETS_LOCK = GTAB_ROOT / "data" / "datasets" / "datasets.lock.json"
MODELS_LOCK = GTAB_ROOT / "data" / "models" / "models.lock.json"


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


@lru_cache(maxsize=1)
def _node_available() -> bool:
    return shutil.which("node") is not None and ALPHATAB_PKG.is_file()


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=None)
def _dataset_available(name: str) -> bool:
    # The DATA owner's own check: a partial fetch (e.g. EGDB with only some amp folders) is "partial", not
    # available, even though the lock lists files for it.
    try:
        from gtab import datasets
    except Exception:
        return False
    try:
        return bool(datasets.is_available(name)) and (DATASETS_LOCK.parent / name).is_dir()
    except Exception:  # unknown name / unreadable lock -> skip, never error at collection
        return False


@lru_cache(maxsize=None)
def _model_available(name: str) -> bool:
    entry = (_read_json(MODELS_LOCK).get("models") or {}).get(name)
    if not isinstance(entry, dict) or not entry.get("path"):
        return False
    return (GTAB_ROOT / str(entry["path"])).is_file()


def _marker_args(item: pytest.Item, marker: str) -> list[str]:
    names: list[str] = []
    for m in item.iter_markers(marker):
        names += [str(a) for a in m.args]
    return names


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not _gpu_available():
            item.add_marker(pytest.mark.skip(reason="GPU worker env / CUDA not available"))
        if "gpuenv" in item.keywords and not GPU_PYTHON.exists():
            item.add_marker(pytest.mark.skip(reason=f"gpu venv not found: {GPU_PYTHON}"))
        if "bp310" in item.keywords and not BP310_PYTHON.exists():
            item.add_marker(pytest.mark.skip(reason=f"bp310 venv not found: {BP310_PYTHON}"))
        if "node" in item.keywords and not _node_available():
            item.add_marker(pytest.mark.skip(reason="node + node/node_modules/@coderline/alphatab needed"))
        for name in _marker_args(item, "data"):
            if not _dataset_available(name):
                item.add_marker(pytest.mark.skip(reason=f"dataset {name!r} not available (gtab data fetch {name})"))
        for name in _marker_args(item, "model"):
            if not _model_available(name):
                item.add_marker(pytest.mark.skip(reason=f"model {name!r} not recorded (gtab models fetch {name})"))
        if "network" in item.keywords and not _network_available():
            item.add_marker(pytest.mark.skip(reason="network not available"))
