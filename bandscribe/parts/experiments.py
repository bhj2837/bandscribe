"""Part diarizer experiments for ``bandscribe eval exp`` (``bandscribe.eval.experiments``): E25 (M7a), E27 (M7b), E28 (same-position count), E29 (M8 repeat consensus), E30 (M8 bleed flags)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _run_e25(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.parts.eval import run_e25

    return run_e25(out_dir, cfg, args, **kw)


def _run_e27(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.parts.eval_e27 import run_e27

    return run_e27(out_dir, cfg, args, **kw)


def _run_e28(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.parts.eval_e28 import run_e28

    return run_e28(out_dir, cfg, args, **kw)


def _run_e29(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.parts.eval_e29 import run_e29

    return run_e29(out_dir, cfg, args, **kw)


def _run_e30(out_dir: Path, cfg: Any, args: dict, **kw: Any) -> list[dict]:
    from bandscribe.parts.eval_e30 import run_e30

    return run_e30(out_dir, cfg, args, **kw)


EXPERIMENTS = {"E25": _run_e25, "E27": _run_e27, "E28": _run_e28, "E29": _run_e29, "E30": _run_e30}
