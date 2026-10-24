"""Copyright guard (mid-term plan 1-1): the public repository holds code and documents only. No recording, sheet,
Guitar Pro file, MIDI or large note dump may be tracked outside ``tests/fixtures``, and nothing under ``data/`` may be
tracked or staged (songs, stems, jobs, models and the user's sheets all live there). Runs in the pre-push hook."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".ogg", ".opus", ".webm", ".aac", ".aif", ".aiff", ".wma",
         ".pdf", ".gp", ".gpx", ".gp3", ".gp4", ".gp5", ".mid", ".midi", ".sf2", ".ckpt", ".pth", ".onnx", ".npz", ".npy"}
MAX_JSON = 100_000       # bytes: a song's note list is far bigger, the repo's own JSON far smaller
MAX_FILE = 1_000_000     # bytes: docs/DESIGN.md is the largest tracked file (about 240 kB)
ALLOWED = ("tests/fixtures/",)

pytestmark = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(),
                                reason="not a git checkout")


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def problems(tracked: list[str], staged: list[str], size_of) -> list[str]:
    """What the guard refuses, as readable lines (empty = fine)."""
    bad = []
    for path in tracked:
        p = path.replace("\\", "/")
        if p.startswith("data/"):
            bad.append(f"{p}: data/ 는 올리지 않습니다(음원·작업·모델·악보)")
            continue
        if p.startswith(ALLOWED):
            continue
        ext = Path(p).suffix.lower()
        if ext in MEDIA:
            bad.append(f"{p}: 음원·악보·MIDI·모델 파일은 tests/fixtures 밖에 둘 수 없습니다")
            continue
        size = size_of(p)
        if size is None:
            continue
        if ext == ".json" and size > MAX_JSON:
            bad.append(f"{p}: {size:,} 바이트 JSON (음표 덤프?) — {MAX_JSON:,} 바이트까지")
        elif size > MAX_FILE:
            bad.append(f"{p}: {size:,} 바이트 — {MAX_FILE:,} 바이트까지")
    bad += [f"{p}: data/ 아래가 스테이지되어 있습니다" for p in staged if p.replace("\\", "/").startswith("data/")]
    return bad


def _size(rel: str) -> int | None:
    try:
        return (ROOT / rel).stat().st_size
    except OSError:
        return None


def test_nothing_copyrighted_or_private_is_tracked_or_staged():
    bad = problems(_git("ls-files"), _git("diff", "--cached", "--name-only"), _size)
    assert not bad, "\n".join(bad)


def test_the_guard_catches_what_it_should():
    sizes = {"bandscribe/x.json": 5_000, "docs/notes.json": 400_000, "docs/big.md": 2_000_000}
    bad = problems(["data/jobs/f-1/input/source.mp3", "tests/fixtures/tone.wav", "docs/lulu.gp", "Score.PDF",
                    "bandscribe/x.json", "docs/notes.json", "docs/big.md", "README.md"],
                   ["data/config.toml", "bandscribe/cli.py"], sizes.get)
    assert [b.split(":")[0] for b in bad] == ["data/jobs/f-1/input/source.mp3", "docs/lulu.gp", "Score.PDF",
                                              "docs/notes.json", "docs/big.md", "data/config.toml"]
