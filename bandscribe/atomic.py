"""Atomic file writes and hashing.

Readers must never see a half-written file: write to a temp file in the same directory, then os.replace.
Importable from both envs (stdlib only).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from bandscribe import formats

# Windows refuses os.replace() onto a file that another process has open without FILE_SHARE_DELETE
# (Python's open() never sets it), so a reader such as `bandscribe status` or an antivirus scan can make a
# rewrite fail with WinError 5. Those handles live for milliseconds: retry for up to ~2 s.
_REPLACE_RETRIES = 20
_REPLACE_BACKOFF_S = 0.01


def _replace(src: str, dst: Path) -> None:
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))


def write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text(path: Path, text: str) -> None:
    write_bytes(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n")


def read_json(path: Path) -> Any:
    """Parsed JSON; a legacy top-level ``"format": "gtab.<name>/N"`` comes back as ``"bandscribe.<name>/N"``
    (``formats.normalize_doc``: files from before the rename are read, never rewritten)."""
    with open(path, encoding="utf-8") as f:
        return formats.normalize_doc(json.load(f))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
