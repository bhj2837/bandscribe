"""The run summary card (mid-term plan 1-7): what ``run`` prints last, in a few Korean lines - the tempo and whether
anyone set or kept it, the meter, the tunings and their margins, the guitar parts, the health counts, how to open the
song and where the Guitar Pro file is. Read from the export and the stage outputs behind it; nothing is computed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.jobs import hints as hints_mod
from bandscribe.tab import health as HL


def _read(p: Path) -> dict[str, Any]:
    try:
        doc = atomic.read_json(p)
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _meter(grid: dict[str, Any]) -> str:
    segs = grid.get("meter") or []
    if not segs:
        return ""
    m = max(segs, key=lambda s: int(s.get("end_bar", 0)) - int(s.get("start_bar", 0)))
    return f"{m.get('numerator')}/{m.get('denominator')}" + (" (박자 바뀜 있음)" if len(segs) > 2 else "")


def _tempo(doc: dict[str, Any], job_dir: Path) -> str:
    m = doc.get("metrics") or {}
    bpm = m.get("tempo_bpm")
    if not isinstance(bpm, (int, float)):
        return "-"
    if m.get("tempo_by") == "hint":
        return f"{bpm:.0f} BPM (정함)"
    if (hints_mod.read_hints(job_dir).get("hints") or {}).get("tempo_confirmed") is True:
        return f"{bpm:.0f} BPM (자동, 확인함)"
    item = next((i for i in doc.get("items") or [] if i.get("id") == "tempo"), {})
    cands = ((item.get("data") or {}).get("candidates") or {})
    alts = sorted(float(v) for v in cands.values() if isinstance(v, (int, float)) and abs(float(v) - bpm) > 0.5)
    return f"{bpm:.0f} BPM (자동, 확인 안 됨" + (f" — 다른 후보 {' · '.join(f'{a:.0f}' for a in alts)}" if alts else "") + \
        ". 틀렸으면 --tempo)"


def _tuning(tab: dict[str, Any]) -> str:
    out = []
    for key, ko in (("guitar", "기타"), ("bass", "베이스")):
        t = (tab.get("tuning") or {}).get(key)
        if not isinstance(t, dict) or not t.get("label"):
            continue
        s = f"{ko} {t['label']}" + (f", 카포 {t['capo']}" if t.get("capo") else "")
        margin = t.get("margin")
        if t.get("user"):
            s += " (지정)"
        elif isinstance(margin, (int, float)):
            s += f" (여유 {margin:.2f}" + (", 확인 필요)" if margin < HL.TUNING_MARGIN_LOW else ")")
        out.append(s)
    return " · ".join(out) or "-"


def _parts(score: dict[str, Any]) -> str:
    lines = score.get("lines") or []
    gtr = [ln for ln in lines if ln.get("kind") == "guitar"]
    bits = [f"{ln.get('name') or ln.get('id')} {len(ln.get('notes') or [])}음" for ln in gtr]
    bass = next((ln for ln in lines if ln.get("kind") == "bass"), None)
    if bass is not None:
        bits.append(f"베이스 {len(bass.get('notes') or [])}음")
    head = f"기타 {len(gtr)}파트" if gtr else "기타 파트 없음"
    return head + (" — " + " · ".join(bits) if bits else "")


def card(job_dir: Path, doc: dict[str, Any] | None = None, *, view_arg: str | None = None) -> list[tuple[str, str]]:
    """[(label, text)] of the summary card. ``doc`` = the song's health (``HL.load_health`` when None); ``view_arg`` =
    what to type after ``bandscribe view`` (default: the job key)."""
    job_dir = Path(job_dir)
    doc = doc if doc is not None else HL.load_health(job_dir)
    export = job_dir / "export"
    tab = _read(export / "tab" / "tab.json")
    score = _read(export / "score.json")
    grid_dir = HL.stage_dirs(job_dir).get("grid")
    grid = _read(grid_dir / "grid.json") if grid_dir is not None else {}
    meter = _meter(grid)
    s = doc.get("summary") or {}
    rows = [("템포", _tempo(doc, job_dir) + (f" · {meter}" if meter else "")),
            ("튜닝", _tuning(tab)),
            ("파트", _parts(score)),
            ("점검", f"경고 {s.get('warn', 0)} · 알림 {s.get('info', 0)}"
                     + (" (위 목록, 뷰어 위쪽에도 나옵니다)" if s.get("warn") or s.get("info") else ""))]
    arg = view_arg or job_dir.name
    rows.append(("보기", "bandscribe view " + (f'"{arg}"' if any(c.isspace() for c in arg) else arg)
                 + "   (곡 목록: bandscribe view)"))
    gp, gp5 = export / "tab" / "score.gp", export / "tab" / "score.gp5"
    files = [str(p) for p in (gp, gp5) if p.is_file()]
    if files:
        rows.append(("파일", files[0] + (f"  (GP5: {Path(files[1]).name})" if len(files) > 1 else "")))
    return rows
