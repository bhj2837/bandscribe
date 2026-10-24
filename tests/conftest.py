"""Shared pytest config: auto-skip tests whose resource is missing (M0_SPEC 0, M1_M2_SPEC 0).

Markers (registered in pyproject.toml):
- ``gpu``: CUDA works in the gpu env (M0).
- ``gpuenv``: the gpu venv exists (CPU torch is enough).
- ``bp310``: the Basic Pitch venv (Python 3.10) exists.
- ``node``: node is on PATH and ``node/node_modules/@coderline/alphatab`` is installed.
- ``data(name)``: ``bandscribe.datasets.is_available(name)`` (complete, not partial) and its folder exists.
- ``model(name)``: model ``name`` is recorded in ``data/models/models.lock.json`` and its file exists.
- ``network``: internet reachable (``BANDSCRIBE_OFFLINE`` forces "no").
- ``slow``: runs by default (as in M0); deselect with ``-m "not slow"``.
- ``smoke`` (mid-term plan 1-1): the fast subset of ``SMOKE_MODULES`` the pre-push hook runs (``-m smoke``).

``gpu`` and ``network`` tests run only with ``--run-gpu`` / ``--run-network`` (plan 1-1): they take the GPU or the
internet for minutes, so a plain run never starts them by accident.

The default ``pytest -q`` stays green without GPU, network, datasets or models. Resource checks read the real
``BANDSCRIBE_ROOT`` layout directly (JSON / file checks) instead of going through bandscribe modules, so a broken module
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

BANDSCRIBE_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = BANDSCRIBE_ROOT / "envs" / "gpu" / ".venv" / "Scripts" / "python.exe"
BP310_PYTHON = BANDSCRIBE_ROOT / "envs" / "bp310" / ".venv" / "Scripts" / "python.exe"
ALPHATAB_PKG = BANDSCRIBE_ROOT / "node" / "node_modules" / "@coderline" / "alphatab" / "package.json"
DATASETS_LOCK = BANDSCRIBE_ROOT / "data" / "datasets" / "datasets.lock.json"
MODELS_LOCK = BANDSCRIBE_ROOT / "data" / "models" / "models.lock.json"


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
    if os.environ.get("BANDSCRIBE_OFFLINE"):
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
        from bandscribe import datasets
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
    return (BANDSCRIBE_ROOT / str(entry["path"])).is_file()


# Fast modules that cover the CLI, the config, the run flow on fake stages, the song health, the viewer server, the
# backup and the copyright guard (`pytest -m smoke`, well under 2 minutes; the pre-push hook runs it).
SMOKE_MODULES = frozenset({
    "test_repo_guard", "test_config", "test_cli", "test_pipeline_cli", "test_jobs_hints", "test_tab_health",
    "test_tab_summary", "test_viewer_home", "test_backup", "test_tab_cleanup", "test_datasets_registry",
})


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", default=False, help="run tests marked gpu (CUDA worker)")
    parser.addoption("--run-network", action="store_true", default=False, help="run tests marked network")


def _marker_args(item: pytest.Item, marker: str) -> list[str]:
    names: list[str] = []
    for m in item.iter_markers(marker):
        names += [str(a) for a in m.args]
    return names


def pytest_collection_modifyitems(config, items):
    run_gpu, run_net = config.getoption("--run-gpu"), config.getoption("--run-network")
    for item in items:
        if item.path.stem in SMOKE_MODULES:
            item.add_marker(pytest.mark.smoke)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(pytest.mark.skip(reason="GPU test: run with --run-gpu"))
        elif "gpu" in item.keywords and not _gpu_available():
            item.add_marker(pytest.mark.skip(reason="GPU worker env / CUDA not available"))
        if "gpuenv" in item.keywords and not GPU_PYTHON.exists():
            item.add_marker(pytest.mark.skip(reason=f"gpu venv not found: {GPU_PYTHON}"))
        if "bp310" in item.keywords and not BP310_PYTHON.exists():
            item.add_marker(pytest.mark.skip(reason=f"bp310 venv not found: {BP310_PYTHON}"))
        if "node" in item.keywords and not _node_available():
            item.add_marker(pytest.mark.skip(reason="node + node/node_modules/@coderline/alphatab needed"))
        for name in _marker_args(item, "data"):
            if not _dataset_available(name):
                item.add_marker(pytest.mark.skip(reason=f"dataset {name!r} not available (bandscribe data fetch {name})"))
        for name in _marker_args(item, "model"):
            if not _model_available(name):
                item.add_marker(pytest.mark.skip(reason=f"model {name!r} not recorded (bandscribe models fetch {name})"))
        if "network" in item.keywords and not run_net:
            item.add_marker(pytest.mark.skip(reason="network test: run with --run-network"))
        elif "network" in item.keywords and not _network_available():
            item.add_marker(pytest.mark.skip(reason="network not available"))
