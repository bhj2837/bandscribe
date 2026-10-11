"""Part diarizer experiments for ``bandscribe eval exp`` (``bandscribe.eval.experiments``): E25 (M7a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _run_e25(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.parts.eval import run_e25

    return run_e25(out_dir, cfg, args, **kw)


EXPERIMENTS = {"E25": _run_e25}
