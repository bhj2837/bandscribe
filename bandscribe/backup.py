"""Local backup of what bandscribe cannot make again (mid-term plan 1-1; D: is one USB disk).

``backup(dest)`` mirrors into ``dest``: the models (the SW checkpoint has no download anywhere), the evaluation
references and sheet extractions, each job's own inputs (``hints.toml``, ``edits.jsonl``, ``input/meta.json``), the
user config, and the folders listed in ``[backup] extra`` (the original recordings and the band scores, which live
outside the repository). Incremental: a file whose size and modification time match the last manifest is not read
again. Every copied file is hashed before and after the copy. Nothing is deleted from ``dest``.

``dest/manifest.json`` lists every file with its sha256; ``data/backup.json`` records the last backup for
``bandscribe doctor`` (warns after ``[backup] max_age_days`` or when the SW checkpoint is not in it). Stages, stems and
datasets are left out: they can be made or fetched again.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

MANIFEST = "manifest.json"
RECORD = "backup.json"           # in paths.DATA
JOB_FILES = ("hints.toml", "edits.jsonl", "input/meta.json")
SW_NAME = "BS-Rofo-SW-Fixed.ckpt"


class BackupError(RuntimeError):
    """User-facing (Korean) problem of a backup."""


def sources(cfg: Any) -> list[tuple[str, Path]]:
    """(name in the backup, folder) of everything mirrored whole."""
    data = Path(paths.DATA)
    out = [("models", data / "models"), ("eval_tierA", data / "eval" / "tierA"), ("eval_sheets", data / "eval" / "sheets")]
    out += [(f"tmp_{n}", data / "tmp" / n) for n in ("m9", "m10", "midterm")]
    tmp = data / "tmp"
    out += [(f"tmp_{p.name}", p) for p in sorted(tmp.glob("sheets_*")) if p.is_dir()] if tmp.is_dir() else []
    for i, extra in enumerate(cfg.backup.extra):
        p = Path(extra)
        out.append((f"extra_{i + 1}_{p.name or 'root'}", p))
    return out


def _walk(base: Path) -> Iterator[Path]:
    for p in sorted(base.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            yield p


def plan_files(cfg: Any) -> tuple[list[tuple[str, Path]], list[str]]:
    """([(path in the backup, source file)], [missing ``[backup] extra`` folders])."""
    files: list[tuple[str, Path]] = []
    missing: list[str] = []
    for name, base in sources(cfg):
        if not base.is_dir():
            if name.startswith("extra_"):  # bandscribe's own folders appear as the work goes on
                missing.append(str(base))
            continue
        files += [(f"{name}/{p.relative_to(base).as_posix()}", p) for p in _walk(base)]
    jobs = Path(paths.JOBS)
    for job in sorted(jobs.iterdir()) if jobs.is_dir() else []:
        for fn in JOB_FILES:
            if (job / fn).is_file():
                files.append((f"jobs/{job.name}/{fn}", job / fn))
    if Path(paths.USER_CONFIG).is_file():
        files.append(("config/config.toml", Path(paths.USER_CONFIG)))
    return files, missing


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def backup(dest: Path, cfg: Any, *, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Mirror into ``dest`` (module doc); returns the record written to ``data/backup.json``."""
    dest = Path(dest)
    try:
        dest.resolve().relative_to(Path(paths.ROOT).resolve())
    except ValueError:
        pass
    else:
        raise BackupError(f"백업 위치가 bandscribe 폴더 안입니다: {dest}. 다른 디스크나 폴더를 고르세요.")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        old = {f["path"]: f for f in (atomic.read_json(dest / MANIFEST).get("files") or [])}
    except (OSError, ValueError):
        old = {}
    files, missing = plan_files(cfg)
    t0 = time.monotonic()
    entries, copied, copied_bytes, total = [], 0, 0, 0
    for i, (rel, src) in enumerate(files):
        st = src.stat()
        total += st.st_size
        prev, dst = old.get(rel), dest / rel
        if (prev and prev.get("bytes") == st.st_size and prev.get("mtime_ns") == st.st_mtime_ns
                and dst.is_file() and dst.stat().st_size == st.st_size):
            entries.append(prev)
            continue
        h = sha256(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copy2(src, tmp)
        if sha256(tmp) != h:
            tmp.unlink(missing_ok=True)
            raise BackupError(f"복사한 파일의 해시가 원본과 다릅니다: {src} -> {dst}")
        os.replace(tmp, dst)
        entries.append({"path": rel, "source": str(src), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h})
        copied += 1
        copied_bytes += st.st_size
        if progress and copied % 50 == 0:
            progress(f"{i + 1}/{len(files)} 확인, {copied}개 복사")
    date = time.strftime("%Y-%m-%d %H:%M")
    atomic.write_json(dest / MANIFEST, {"date": date, "files": entries})
    sw = next((e for e in entries if e["path"].endswith("/" + SW_NAME)), None)
    record = {"date": date, "dest": str(dest), "files": len(entries), "bytes": total, "copied": copied,
              "copied_bytes": copied_bytes, "missing_sources": missing, "sw_sha256": sw["sha256"] if sw else None,
              "seconds": round(time.monotonic() - t0, 1)}
    atomic.write_json(Path(paths.DATA) / RECORD, record)
    return record


def last_backup() -> dict[str, Any] | None:
    try:
        doc = atomic.read_json(Path(paths.DATA) / RECORD)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None
