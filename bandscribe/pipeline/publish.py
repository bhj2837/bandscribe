"""Publish a finished run into ``jobs/<song_key>/export/`` (M1_M2_SPEC 3.3; DESIGN 7.1).

``export/`` holds what a user opens: ``midi/guitar_all.mid``, ``midi/bass_raw.mid``, ``midi/b0_guitar.mid``
(if present), the M3 score (``tab/score.alphatex``, ``tab/score.gp5`` when a track is fretted, ``tab/*.txt``,
``tab/tab.json``, ``midi/*_quantized.mid``, ``score.json``; ``bandscribe view`` reads these), ``practice/mix_minus_guitar.wav``, ``practice/mix_minus_bass.wav`` (``stems`` stage),
``practice/bass_stem.wav`` (``sep`` stage's ``stems/bass.wav``), ``instrumentation.json`` and ``export.json``
(``{"format": "bandscribe.export/1", "files": {rel: {"stage", "key12", "sha256"}}}``).

- Small files (MIDI, JSON) are **copied**, so editing an exported MIDI in a DAW never alters a cached stage
  file. WAVs are **hardlinked** (fallback: copy) and marked read-only; the attribute is shared with the stage
  file, which protects the cache from in-place edits (M0's ``rmtree_quiet`` clears it on delete).
- The new export is built in ``.tmp-export-*`` and swapped in like ingest's rebuild: old ``export/`` renamed
  aside to ``.tmp-export-old-*``, new one renamed in, old one removed. If the old one cannot be removed
  (a player holds a file open) it stays as ``.tmp-export-old-*`` for ``bandscribe gc``; if it cannot even be renamed
  aside, the old export is kept, the new build stays as ``.tmp-export-*`` (also for ``bandscribe gc``) and publishing
  fails with a Korean message (the stages stay cached, so a rerun is quick).
- Idempotent: same stage keys (same ``export.json``) and all files present -> nothing is rewritten.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.jobs.store import MANIFEST, make_temp_dir, rmtree_quiet

log = logging.getLogger(__name__)

EXPORT_DIR = "export"
EXPORT_JSON = "export.json"
EXPORT_FORMAT = "bandscribe.export/1"

# (export rel path, stage, rel path in the stage dir, hardlink (WAV) or copy, required)
EXPORT_MAP: tuple[tuple[str, str, str, bool, bool], ...] = (
    ("midi/guitar_all.mid", "notes", "midi/guitar_all.mid", False, True),
    ("midi/bass_raw.mid", "notes", "midi/bass_raw.mid", False, True),
    ("midi/b0_guitar.mid", "notes", "midi/b0_guitar.mid", False, False),
    ("practice/mix_minus_guitar.wav", "stems", "practice/mix_minus_guitar.wav", True, True),
    ("practice/mix_minus_bass.wav", "stems", "practice/mix_minus_bass.wav", True, True),
    ("practice/bass_stem.wav", "sep", "stems/bass.wav", True, True),
    ("instrumentation.json", "instr", "instrumentation.json", False, True),
    # M3 (optional: `--until notes` publishes without them)
    ("tab/score.alphatex", "score", "tab/score.alphatex", False, False),
    ("tab/score.gp5", "score", "tab/score.gp5", False, False),
    ("tab/bass_raw.txt", "score", "tab/bass_raw.txt", False, False),
    ("tab/guitar_all.txt", "score", "tab/guitar_all.txt", False, False),
    ("tab/tab.json", "tab", "tab.json", False, False),
    ("midi/guitar_all_quantized.mid", "score", "midi/guitar_all_quantized.mid", False, False),
    ("midi/bass_raw_quantized.mid", "score", "midi/bass_raw_quantized.mid", False, False),
    ("score.json", "score", "score.json", False, False),
)

_RENAME_RETRIES = 8
_RENAME_BACKOFF_S = 0.05


class PublishError(RuntimeError):
    """Publishing failed (Korean, user-facing message); the stage results stay cached."""


def _stage_sha(stage_dir: Path, rel: str) -> str:
    try:
        m = atomic.read_json(stage_dir / MANIFEST)
        sha = ((m.get("outputs") or {}).get(rel) or {}).get("sha256")
        if isinstance(sha, str) and sha:
            return sha
    except (OSError, ValueError):
        pass
    return atomic.sha256_file(stage_dir / rel)


def plan_export(results: dict[str, Path]) -> list[dict[str, Any]]:
    """What would be exported: [{rel, stage, key12, src, link, sha256}] (missing optional files skipped)."""
    out = []
    for rel, stage, src_rel, link, required in EXPORT_MAP:
        d = results.get(stage)
        src = Path(d) / src_rel if d is not None else None
        if src is None or not src.is_file():
            if required:
                raise PublishError(f"내보낼 파일이 없습니다: {stage} 단계의 {src_rel}")
            continue
        out.append({"rel": rel, "stage": stage, "key12": Path(d).name, "src": src, "link": link,
                    "sha256": _stage_sha(Path(d), src_rel)})
    return out


def _doc(entries: list[dict[str, Any]]) -> dict[str, Any]:
    files = {e["rel"]: {"stage": e["stage"], "key12": e["key12"], "sha256": e["sha256"]}
             for e in sorted(entries, key=lambda e: e["rel"])}
    return {"format": EXPORT_FORMAT, "files": files}


def _up_to_date(export: Path, doc: dict[str, Any], entries: list[dict[str, Any]]) -> bool:
    try:
        if atomic.read_json(export / EXPORT_JSON) != doc:
            return False
    except (OSError, ValueError):
        return False
    for e in entries:
        p = export / e["rel"]
        try:
            if p.stat().st_size != e["src"].stat().st_size:
                return False
            if e["link"] and not os.path.samefile(p, e["src"]) and (os.access(p, os.W_OK) or p.stat().st_nlink > 1):
                # The stage was recomputed under the same key (--force): the export still holds the old inode,
                # now writable (deleting the old stage dir cleared the shared read-only bit) and no longer
                # sharing storage with the cache. Re-link. (A read-only single-link file is the copy fallback
                # of a volume without hardlinks: content is pinned by export.json's sha, nothing to do.)
                return False
        except OSError:
            return False
    return True


def _set_readonly(p: Path) -> None:
    try:
        os.chmod(p, stat.S_IREAD)
    except OSError as e:
        log.debug("could not mark %s read-only: %s", p, e)


def _rename(src: Path, dst: Path) -> None:
    for attempt in range(_RENAME_RETRIES):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if attempt == _RENAME_RETRIES - 1:
                raise
            time.sleep(_RENAME_BACKOFF_S * (attempt + 1))


def publish(job_dir: Path, results: dict[str, Path]) -> dict[str, Any]:
    """Build and swap in ``export/``. Returns {"export_dir", "changed", "files": {rel: abs path}}."""
    job_dir = Path(job_dir)
    export = job_dir / EXPORT_DIR
    entries = plan_export(results)
    doc = _doc(entries)
    files = {e["rel"]: export / e["rel"] for e in entries}
    if _up_to_date(export, doc, entries):
        log.info("export is up to date: %s", export)
        return {"export_dir": export, "changed": False, "files": files}

    tmp = make_temp_dir(job_dir, "export")
    try:
        for e in entries:
            dst = tmp / e["rel"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if e["link"]:
                try:
                    os.link(e["src"], dst)
                except OSError as err:  # other volume, FAT, ...: a copy is fine, just bigger
                    log.debug("hardlink failed (%s); copying %s", err, e["src"])
                    shutil.copyfile(e["src"], dst)
                _set_readonly(dst)
            else:
                shutil.copyfile(e["src"], dst)
        atomic.write_json(tmp / EXPORT_JSON, doc)
    except BaseException:
        rmtree_quiet(tmp)
        raise

    aside: Path | None = None
    if export.exists():
        aside = make_temp_dir(job_dir, "export-old")
        aside.rmdir()  # os.rename needs a free target name
        try:
            _rename(export, aside)
        except OSError as err:
            log.warning("could not move the old export aside (%s); new build left at %s", err, tmp)
            raise PublishError(f"export 폴더의 파일이 다른 프로그램에서 열려 있습니다: {export}. "
                               "닫고 다시 실행하세요.") from err
    try:
        _rename(tmp, export)
    except OSError as err:
        if aside is not None and not export.exists():
            try:
                _rename(aside, export)  # put the old export back
            except OSError:
                pass
        raise PublishError(f"export 폴더를 만들지 못했습니다: {export} ({err}). 다시 실행하세요.") from err
    if aside is not None and not rmtree_quiet(aside):
        log.warning("이전 export 폴더를 지우지 못했습니다(열린 파일). 나중에 'bandscribe gc' 가 지웁니다: %s", aside)
    # removing the old export may have cleared the read-only attribute shared by the hardlinks
    for e in entries:
        if e["link"]:
            _set_readonly(export / e["rel"])
    log.info("published %d files to %s", len(entries), export)
    return {"export_dir": export, "changed": True, "files": files}
