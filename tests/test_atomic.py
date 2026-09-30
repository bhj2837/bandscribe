"""gtab.atomic: atomic rewrites, hashing, and the Windows open-reader retry."""
from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path

import pytest

from gtab import atomic


def _temps(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.name.endswith(".tmp")]


def test_json_round_trip_keeps_unicode(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "meta.json"
    obj = {"title": "합주 녹음", "n": 1.5, "flags": {"a": True}}
    atomic.write_json(path, obj)
    assert atomic.read_json(path) == obj
    assert "합주 녹음" in path.read_text(encoding="utf-8")  # ensure_ascii=False
    assert _temps(path.parent) == []


def test_hashes_match_hashlib(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 + 7)
    path = tmp_path / "blob.bin"
    atomic.write_bytes(path, data)
    expected = hashlib.sha256(data).hexdigest()
    assert atomic.sha256_bytes(data) == expected
    assert atomic.sha256_file(path, chunk=1000) == expected


def test_failed_replace_leaves_old_file_and_no_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.json"
    atomic.write_json(path, {"v": 1})

    def boom(src: object, dst: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(atomic.os, "replace", boom)
    with pytest.raises(OSError, match="disk gone"):
        atomic.write_json(path, {"v": 2})
    assert atomic.read_json(path) == {"v": 1}
    assert _temps(tmp_path) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_rewrite_waits_for_a_short_lived_reader(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    atomic.write_json(path, {"v": 1})
    reader = open(path, "rb")  # noqa: SIM115 - held open on purpose, closed by the timer
    try:
        # Precondition: this is exactly the failure the retry exists for.
        probe = tmp_path / "probe.tmp"
        probe.write_bytes(b"{}")
        with pytest.raises(PermissionError):
            os.replace(probe, path)
        probe.unlink()

        timer = threading.Timer(0.15, reader.close)
        timer.start()
        atomic.write_json(path, {"v": 2})
        timer.join()
    finally:
        reader.close()
    assert atomic.read_json(path) == {"v": 2}
    assert _temps(tmp_path) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing semantics")
def test_rewrite_gives_up_on_a_reader_that_never_closes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atomic, "_REPLACE_BACKOFF_S", 0.0)
    path = tmp_path / "held.json"
    atomic.write_json(path, {"v": 1})
    with open(path, "rb"):
        with pytest.raises(PermissionError):
            atomic.write_json(path, {"v": 2})
    assert atomic.read_json(path) == {"v": 1}
    assert _temps(tmp_path) == []
