"""CLI start-up stays light (M1_M2_SPEC 0, 1.7): heavy analysis libraries are imported inside functions only.

``import bandscribe.cli`` registers every ``bandscribe.commands.*`` module, so ``bandscribe --help`` would otherwise pay for
librosa/numba JIT set-up, synctoolbox (matplotlib.pyplot at import time), libfmp (IPython 8), music21 and
pandas. Checked in a fresh interpreter so this test process's own imports do not count.
"""
from __future__ import annotations

import json
import pkgutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAVY = (
    "torch", "librosa", "synctoolbox", "libfmp", "IPython", "music21", "pandas", "sklearn", "matplotlib",
    "numba", "mir_eval", "pretty_midi", "guitarpro",
)


def _command_modules() -> list[str]:
    return sorted(f"bandscribe.commands.{m.name}" for m in pkgutil.iter_modules([str(ROOT / "bandscribe" / "commands")]))


def test_cli_and_commands_import_no_heavy_modules() -> None:
    mods = ["bandscribe.cli", *_command_modules()]
    code = (
        "import importlib, json, sys\n"
        f"for m in {mods!r}:\n"
        "    importlib.import_module(m)\n"
        f"print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=120)
    assert run.returncode == 0, run.stderr
    loaded = json.loads(run.stdout.strip().splitlines()[-1])
    assert loaded == [], f"heavy modules imported at CLI start-up: {loaded}"


def test_every_spec_command_module_exists() -> None:
    assert {"bandscribe.commands.run", "bandscribe.commands.bench", "bandscribe.commands.eval", "bandscribe.commands.gt",
            "bandscribe.commands.data", "bandscribe.commands.models"} <= set(_command_modules())
