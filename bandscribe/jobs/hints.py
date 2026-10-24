"""A job's ``hints.toml`` (per-song settings: ``[hints] title``, ``[grid] tempo_bpm``, ``[tab] guitar_tuning`` ...):
read it, and set one key without losing the user's comments or the other keys (mid-term plan 1-6: the CLI keeps
``--tempo``, ``--title``, ``--parts`` and the tempo answer here, so nobody edits the file by hand).

``set_hint`` edits the text line by line: it replaces ``key = ...`` inside ``[section]`` (the first one), or adds the
key after that section's last key, or appends the section. Values are written as TOML (strings quoted with JSON rules, which
TOML basic strings accept)."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from bandscribe import atomic

FILE = "hints.toml"
_SECTION = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")


def path(job_dir: Path) -> Path:
    return Path(job_dir) / FILE


def read_hints(job_dir: Path) -> dict[str, Any]:
    p = path(job_dir)
    if not p.is_file():
        return {}
    try:
        return tomllib.loads(p.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return {}


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    return json.dumps(str(v), ensure_ascii=False)


def set_hint(job_dir: Path, section: str, key: str, value: Any, *, comment: str | None = None) -> None:
    """``[section] key = value`` in the job's hints.toml (module doc). ``comment`` goes on the line above a new key."""
    p = path(job_dir)
    lines = p.read_bytes().decode("utf-8-sig").splitlines() if p.is_file() else []  # Notepad's BOM is dropped
    line = f"{key} = {_toml_value(value)}"
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")
    start = end = None
    for i, ln in enumerate(lines):
        m = _SECTION.match(ln)
        if m and start is None and m.group(1).strip() == section:
            start, end = i, len(lines)
        elif m and start is not None and end == len(lines) and i > start:
            end = i
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines += ([f"# {comment}"] if comment else []) + [f"[{section}]", line]
    else:
        for i in range(start + 1, end):
            if key_re.match(lines[i]):
                lines[i] = line
                break
        else:
            k = end
            # after the section's last key: blank and comment lines before the next header belong to that header
            while k > start + 1 and (not lines[k - 1].strip() or lines[k - 1].lstrip().startswith("#")):
                k -= 1
            lines[k:k] = ([f"# {comment}"] if comment else []) + [line]
    text = "\n".join(lines) + "\n"
    tomllib.loads(text)  # never write a file the config loader would reject
    atomic.write_text(p, text)
