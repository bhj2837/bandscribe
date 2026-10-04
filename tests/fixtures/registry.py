"""A copy of docs/decisions.md with every experiment reset to '사전 등록' (pre-registered).

Tests exercise the experiment state machine; the real registry moves on as experiments get decided, and a decided
entry (correctly) refuses to run again. Copying it and resetting the states keeps the tests independent of that.
"""
from __future__ import annotations

import re
from pathlib import Path

from gtab import paths

_STATE = re.compile(r"^- 상태: .*$", re.MULTILINE)


def fresh_registry(dest: Path) -> Path:
    text = (paths.GTAB_ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    head, *sections = text.split("\n## ")
    sections = [_STATE.sub("- 상태: 사전 등록", s, count=1) for s in sections]
    dest.write_text("\n## ".join([head, *sections]), encoding="utf-8")
    return dest
