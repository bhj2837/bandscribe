"""Auxiliary backing check (``bandscribe eval run --suite backing``): guitar notes of an original song that also
appear on its guitar-removed backing track.

Folded in from the 2026-10-04 scratch scripts (``data/scratch/speed*/analyze*.py``, ``retention*.py``). Without a
ground truth for a commercial song, a guitar note that the same pipeline also finds on a backing whose guitar was
removed (same pitch, onset within 50 ms) is most likely **not** guitar: bleed of another instrument into the guitar
stem. The share is a symptom, not a cause; the ``distractors`` suite measures causes on multitracks.

Pairs come from a local TOML file (default ``data/eval/backing_pairs.toml``; ``--arg pairs=<file>``)::

    [[pair]]
    label = "song1"            # ASCII; used in file names and the report
    original = "<path of the original audio>"
    backing = "<path of the backing (guitar removed)>"

Both files must have been run (``bandscribe run <file>``); the suite finds their jobs by content key (the files are
only read, never changed) and uses the ``notes`` stage made with the current settings (else the newest one).
``bleed_share`` compares with the backing's current notes; ``bleed_share_any`` with the union of every ``notes``
result the backing job has (any mask or settings), so it does not depend on which mask the backing used.
Outputs stay under ``data/runs`` (the songs are copyrighted; nothing here may be committed).
"""

from __future__ import annotations

import bisect
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

log = logging.getLogger(__name__)

SUITE = "backing"
GUITAR = ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar")
ONSET_TOL = 0.05


class BackingError(RuntimeError):
    """User-facing (Korean) problem of the backing suite."""


def default_pairs_file() -> Path:
    return Path(paths.EVAL) / "backing_pairs.toml"


def load_pairs(path: Path) -> list[dict[str, str]]:
    import tomllib

    p = Path(path)
    if not p.is_file():
        raise BackingError(f"반주 쌍 목록이 없습니다: {p}. [[pair]] label/original/backing 을 적은 TOML 을 만드세요 "
                           "(bandscribe/eval/backing.py 설명 참고).")
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    out = []
    for i, d in enumerate(doc.get("pair") or []):
        lab = str(d.get("label") or "")
        if not lab or not lab.isascii() or not d.get("original") or not d.get("backing"):
            raise BackingError(f"{p}: pair[{i}] 에 ASCII label, original, backing 이 모두 있어야 합니다")
        out.append({"label": lab, "original": str(d["original"]), "backing": str(d["backing"])})
    if not out:
        raise BackingError(f"{p}: [[pair]] 항목이 없습니다")
    return out


def song_key_of(path: Path, store: Any) -> str | None:
    """Job of an audio file via the ingest index (content key; the file is only read)."""
    from bandscribe.ingest.filekey import content_key

    p = Path(path)
    if not p.is_file():
        return None
    return store.index_get(f"file:{content_key(p)}")


def notes_dirs(song_key: str) -> list[Path]:
    d = Path(paths.JOBS) / song_key / "stages" / "notes"
    if not d.is_dir():
        return []
    return sorted((x for x in d.iterdir() if (x / "guitar_all.json").is_file() and (x / "manifest.json").is_file()),
                  key=lambda x: x.stat().st_mtime)


def guitar_notes(notes_dir: Path) -> list[dict[str, Any]]:
    doc = atomic.read_json(Path(notes_dir) / "guitar_all.json")
    return [n for n in doc.get("notes") or [] if n.get("instrument") in GUITAR]


def hits(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]], tol: float = ONSET_TOL) -> list[bool]:
    """For each note of ``a``: does ``b`` have a note of the same pitch with |Δonset| <= tol?"""
    idx: dict[int, list[float]] = {}
    for n in b:
        idx.setdefault(int(n["pitch"]), []).append(float(n["onset_s"]))
    for v in idx.values():
        v.sort()
    out = []
    for n in a:
        arr = idx.get(int(n["pitch"]), [])
        i = bisect.bisect_left(arr, float(n["onset_s"]) - tol)
        out.append(i < len(arr) and arr[i] <= float(n["onset_s"]) + tol)
    return out


def pair_stats(orig: Sequence[Mapping[str, Any]], back: Sequence[Mapping[str, Any]],
               back_any: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    h = hits(orig, back)
    ha = hits(orig, back_any)
    by_cls: Counter = Counter()
    tot_cls: Counter = Counter()
    for n, x in zip(orig, ha):
        tot_cls[n.get("instrument")] += 1
        by_cls[n.get("instrument")] += int(x)
    n = len(orig)
    return {"orig_n": n, "backing_n": len(back), "on_backing": sum(h), "share": sum(h) / n if n else None,
            "on_backing_any": sum(ha), "share_any": sum(ha) / n if n else None,
            "by_class": {c: {"n": tot_cls[c], "on_backing_any": by_cls[c]} for c in GUITAR if tot_cls[c]}}


def run(run_dir: Path, cfg: Any, args: Mapping[str, Any] | None = None,
        progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    from bandscribe.eval import runs
    from bandscribe.eval.suites import JobStage
    from bandscribe.jobs.store import JobStore

    args = dict(args or {})
    say = progress or (lambda s: log.info("%s", s))
    pairs = load_pairs(Path(args.get("pairs") or default_pairs_file()))
    store = JobStore(paths.JOBS)
    lookup = JobStage("notes", cfg=cfg)
    rows: list[dict[str, Any]] = []
    out_pairs = []
    for p in pairs:
        say(f"반주 대조: {p['label']}")
        ko, kb = song_key_of(Path(p["original"]), store), song_key_of(Path(p["backing"]), store)
        if not ko or not kb:
            out_pairs.append({"label": p["label"], "error": "작업 없음: 두 파일을 먼저 bandscribe run 으로 돌리세요"})
            continue
        do, db = lookup._stage_dir(ko, "notes", "guitar_all.json"), lookup._stage_dir(kb, "notes", "guitar_all.json")
        if do is None or db is None:
            out_pairs.append({"label": p["label"], "error": "notes 단계 결과 없음"})
            continue
        orig, back = guitar_notes(do), guitar_notes(db)
        back_any = [n for d in notes_dirs(kb) for n in guitar_notes(d)]
        st = pair_stats(orig, back, back_any)
        st.update(label=p["label"], original_job=ko, backing_job=kb, original_notes=do.name, backing_notes=db.name)
        out_pairs.append(st)
        common = {"suite": SUITE, "system": "job:notes", "dataset": "backing", "item": p["label"],
                  "group": p["label"], "line": "guitar"}
        rows.append({**common, "metric": "bleed_share", "value": st["share"], "tp": st["on_backing"], "n": st["orig_n"]})
        rows.append({**common, "metric": "bleed_share_any", "value": st["share_any"], "tp": st["on_backing_any"],
                     "n": st["orig_n"]})
        rows.append({**common, "metric": "backing_notes", "value": st["backing_n"]})
    summary = {"suite": SUITE, "pairs": out_pairs}
    runs.write_metrics_csv(Path(run_dir) / "metrics.csv", rows)
    runs.write_summary(Path(run_dir) / "summary.json", summary)
    lines = ["# 반주 대조 (보조 지표)", "",
             "원곡 기타 음 중 기타를 지운 반주에서도 같은 음높이·onset ±50 ms 로 나오는 음의 비율. 높을수록 기타 스템에 "
             "다른 악기가 섞였을 가능성이 크다(원인은 distractors 세트가 잰다).", "",
             "| 쌍 | 원곡 기타 음 | 반주 음 | 반주에도 있음 | 비율 | 반주의 모든 결과 기준 | 비율 |", "|---|---|---|---|---|---|---|"]
    for st in out_pairs:
        if "error" in st:
            lines.append(f"| {st['label']} | {st['error']} | | | | | |")
            continue
        f = (lambda x: "–" if x is None else f"{100 * x:.1f} %")
        lines.append(f"| {st['label']} | {st['orig_n']} | {st['backing_n']} | {st['on_backing']} | {f(st['share'])} | "
                     f"{st['on_backing_any']} | {f(st['share_any'])} |")
    atomic.write_text(Path(run_dir) / "report.md", "\n".join(lines) + "\n")
    return summary
