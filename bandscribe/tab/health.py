"""Song health (mid-term plan phase 1: items 1-4, 1-5, 1-8, 1-9 of docs/MIDTERM_REVIEW.md): the conditions a run must not
keep silent, collected from the stage outputs behind the exported score into ``export/health.json``. The CLI prints the
warnings after a run; the viewer shows them as a banner. Nothing here changes a note: every item is a display.

Items (``level`` = ``warn`` or ``info``; ``spans`` = [[start_s, end_s], ...] where the item applies):
- ``guitar_removed``: the separated guitar stem is quieter than ``GUITAR_REMOVED_DB`` against the mix (``stem_stats``
  ``rel_db``). The user's originals sit at -5.3 to -11.5 dB, the guitar-removed backings at -20.6 to -42.0 dB
  (mid-term review, robustness area). A file name with "backing" is an info line.
- ``tempo``: the grid's tempo level and whether a hint set it; without a hint, the x2 / /2 alternatives with their scores
  (an info line on every unhinted song: the review found a score gap too small to tell on 10 of 13 songs).
- ``burst_<track>``: 10 s windows of the stage notes (before the quant clean-up) where notes shorter than 50 ms are at
  least half, or the density is over 3x the song's median window - a transcriber degeneration (lulu.'s bass).
- ``unplaced_<track>``: the tab could not place more than ``UNPLACED_WARN`` of a track's notes (``tab.json`` check).
- ``tuning_<instrument>`` (info): the chosen tuning is not the instrument's own cheapest (the guitar/bass link overrode it),
  or it wins by less than ``TUNING_MARGIN_LOW``. Info, not a warning: the link is right as often as not (逆光 is
  half-step down by its band score, SNN is not), so only the owner can tell.
- ``bleed_risk``: bars where the separated piano+other stems are at least ``BLEED_RISK_DB`` against the guitar stem
  (E37: such sounds land in the guitar tab and the transcriber never labels them; E38/E41: no note rule removes them).
- ``merged_parts``: non-silent router windows outside R3 (wide double / centred guitars): their notes go to Gtr 1 unsplit.
- ``run``: the run's own warnings (``pipeline.runner.run_warnings``: degraded GPU rungs, grid repairs, leftover energy...).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic

FORMAT = "bandscribe.health/1"
CODE_VERSION = "1"
HEALTH_FILE = "health.json"
GUITAR_REMOVED_DB = -16.0
BURST_WINDOW_S = 10.0
BURST_TINY_S = 0.05
BURST_TINY_SHARE = 0.5
BURST_DENSITY_X = 3.0
BURST_MIN_NOTES = 20
UNPLACED_WARN = 0.2
TUNING_MARGIN_LOW = 0.1
BLEED_RISK_DB = -3.0
BLEED_INFO_SHARE = 0.1
MERGED_INFO_SHARE = 0.2
TRACK_KO = {"bass": "베이스", "guitar_all": "기타(합침)", "gtr_1": "기타 1", "gtr_2": "기타 2", "gtr_3": "기타 3",
            "gtr_4": "기타 4"}


def _manifest(d: Path) -> dict[str, Any]:
    try:
        m = atomic.read_json(d / "manifest.json")
    except (OSError, ValueError):
        return {}
    return m if isinstance(m, dict) else {}


def stage_dirs(job_dir: Path) -> dict[str, Path]:
    """{stage: dir} of every stage behind the exported score (the score stage's deps, transitively)."""
    from bandscribe.tab import edits as E

    job_dir = Path(job_dir)
    stages = job_dir / "stages"
    key = E.export_score_key(job_dir)
    if key is None:
        return {}
    out: dict[str, Path] = {"score": stages / "score" / key}
    todo = [("score", out["score"])]
    while todo:
        _name, d = todo.pop()
        for dep, k in (_manifest(d).get("deps") or {}).items():
            if dep in out:
                continue
            dd = stages / dep / str(k)[:12]
            if dd.is_dir():
                out[dep] = dd
                todo.append((dep, dd))
    return out


def _read(dirs: Mapping[str, Path], stage: str, name: str) -> Any:
    d = dirs.get(stage)
    if d is None or not (Path(d) / name).is_file():
        return None
    try:
        return atomic.read_json(Path(d) / name)
    except (OSError, ValueError):
        return None


def _item(iid: str, level: str, title: str, detail: str = "", spans: Sequence[Sequence[float]] = (),
          **data: Any) -> dict[str, Any]:
    return {"id": iid, "level": level, "title": title, "detail": detail,
            "spans": [[round(float(a), 2), round(float(b), 2)] for a, b in spans], **({"data": data} if data else {})}


def _merge(spans: list[list[float]], gap: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _fmt_spans(spans: Sequence[Sequence[float]], limit: int = 3) -> str:
    def t(x: float) -> str:
        return f"{int(x // 60)}:{int(x % 60):02d}"
    parts = [f"{t(a)}–{t(b)}" for a, b in spans[:limit]]
    return ", ".join(parts) + (f" 외 {len(spans) - limit}곳" if len(spans) > limit else "")


def burst_spans(notes: Sequence[Mapping[str, Any]], duration_s: float) -> tuple[list[list[float]], dict[str, Any]]:
    """10 s windows where the notes are a degeneration burst (module doc), merged into spans."""
    if not notes:
        return [], {}
    n_win = max(1, int(math.ceil(max(duration_s, max(float(n["onset_s"]) for n in notes)) / BURST_WINDOW_S)))
    count = [0] * n_win
    tiny = [0] * n_win
    for n in notes:
        k = min(n_win - 1, int(float(n["onset_s"]) // BURST_WINDOW_S))
        count[k] += 1
        off = n.get("offset_s")
        if off is not None and float(off) - float(n["onset_s"]) < BURST_TINY_S:
            tiny[k] += 1
    busy = sorted(c for c in count if c > 0)
    med = busy[len(busy) // 2] if busy else 0
    spans = []
    for k, c in enumerate(count):
        if c < BURST_MIN_NOTES:
            continue
        if tiny[k] / c >= BURST_TINY_SHARE or (med and c > BURST_DENSITY_X * med):
            spans.append([k * BURST_WINDOW_S, (k + 1) * BURST_WINDOW_S])
    flagged = [k for k in range(n_win) if any(a <= k * BURST_WINDOW_S < b for a, b in spans)]
    stats = {"median_notes_per_window": med, "tiny_in_flagged": sum(tiny[k] for k in flagged),
             "notes_in_flagged": sum(count[k] for k in flagged)}
    return _merge(spans), stats


def _bar_spans(grid: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(b["start_s"]), float(b["end_s"])) for b in grid.get("bars") or []]


def bleed_risk(energy: Mapping[str, Any], grid: Mapping[str, Any]) -> tuple[list[list[float]], float]:
    """Bars where 10 log10(piano + other) - guitar >= BLEED_RISK_DB over the bar's frames where the guitar sounds."""
    import numpy as np

    fps = float(np.asarray(energy["fps"]).reshape(-1)[0])
    g = np.asarray(energy["guitar"], dtype=float)
    po = 10 * np.log10(10 ** (np.asarray(energy["piano"], dtype=float) / 10) + 10 ** (np.asarray(energy["other"], dtype=float) / 10))
    spans, n_bars = [], 0
    for a, b in _bar_spans(grid):
        i, j = int(a * fps), max(int(a * fps) + 1, int(b * fps))
        gg, pp = g[i:j], po[i:j]
        on = gg > -60.0
        if not on.any():
            continue
        n_bars += 1
        rel = 10 * np.log10(np.mean(10 ** (pp[on] / 10)) / np.mean(10 ** (gg[on] / 10)))
        if rel >= BLEED_RISK_DB:
            spans.append([a, b])
    return _merge(spans), (len(spans) / n_bars if n_bars else 0.0)


def song_health(job_dir: Path, *, results: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """The health document of a job's exported score (module doc)."""
    import numpy as np

    job_dir = Path(job_dir)
    dirs = dict(results) if results else stage_dirs(job_dir)
    items: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    grid = _read(dirs, "grid", "grid.json") or {}
    duration = float(grid.get("duration_s") or 0.0)

    st = _read(dirs, "stems", "stem_stats.json") or {}
    g_rel = ((st.get("stems") or {}).get("guitar") or {}).get("rel_db")
    if g_rel is not None:
        metrics["guitar_rel_db"] = round(float(g_rel), 2)
        if float(g_rel) < GUITAR_REMOVED_DB:
            items.append(_item("guitar_removed", "warn", "이 음원에는 기타 소리가 거의 없습니다 — 기타를 뺀 반주일 수 있습니다",
                               f"분리된 기타가 믹스보다 {float(g_rel):.1f} dB 작습니다(원곡은 보통 −5 ~ −12 dB). 탭 전체를 믿기 어렵습니다.",
                               guitar_rel_db=round(float(g_rel), 2)))
    try:
        meta = atomic.read_json(job_dir / "input" / "meta.json")
    except (OSError, ValueError):
        meta = {}
    fname = str(((meta or {}).get("source") or {}).get("filename") or "")
    if "backing" in fname.lower():
        items.append(_item("file_backing", "info", f"파일 이름에 'backing' 이 있습니다({fname})"))

    to = (grid.get("hypotheses") or {}).get("tempo_octave") or {}
    sc = to.get("scores") or {}
    keep = (sc.get(to.get("decision") or "keep") or sc.get("keep") or {})
    bpm = keep.get("bpm")
    if bpm is not None:
        metrics["tempo_bpm"] = round(float(bpm), 1)
        metrics["tempo_by"] = to.get("by")
        if to.get("by") == "hint":
            items.append(_item("tempo", "info", f"템포 {float(bpm):.0f} BPM (힌트로 정함)"))
        else:
            alts = sorted(((k, v) for k, v in sc.items() if k != (to.get("decision") or "keep")),
                          key=lambda kv: -float(kv[1].get("score") or 0))
            alt_txt = ", ".join(f"{float(v.get('bpm') or 0):.0f}({float(v.get('score') or 0):.2f})" for k, v in alts)
            items.append(_item("tempo", "info", f"템포 {float(bpm):.0f} BPM 은 자동 판단입니다 — 확인 안 됨",
                               f"점수 {float(keep.get('score') or 0):.2f}. 다른 후보: {alt_txt}. 틀렸으면 --tempo 로 다시 돌리세요.",
                               candidates={k: v.get("bpm") for k, v in sc.items()}))

    lines = _read(dirs, "parts", "guitar_lines.json") or {}
    gnotes = lines.get("notes") or (_read(dirs, "notes", "guitar_all.json") or {}).get("notes") or []
    bnotes = (_read(dirs, "bass", "bass.json") or {}).get("notes") or []
    for tid, ns in (("guitar_all", gnotes), ("bass", bnotes)):
        spans, stats = burst_spans(ns, duration)
        metrics[f"burst_{tid}_s"] = round(sum(b - a for a, b in spans), 1)
        if spans:
            items.append(_item(f"burst_{tid}", "warn",
                               f"{TRACK_KO[tid]}: 전사가 무너진 구간으로 보입니다 ({_fmt_spans(spans)})",
                               f"아주 짧은 음이나 비정상적으로 많은 음이 몰렸습니다(창당 중앙값 {stats['median_notes_per_window']}개). "
                               "이 구간의 음은 믿기 어렵습니다.", spans, **stats))

    tab = None
    try:
        tab = atomic.read_json(job_dir / "export" / "tab" / "tab.json")
    except (OSError, ValueError):
        tab = None
    for tid, tr in ((tab or {}).get("tracks") or {}).items():
        ck = (tr or {}).get("check") or {}
        n, dropped = int(ck.get("notes") or 0), int(ck.get("dropped") or 0)
        if n and dropped / n > UNPLACED_WARN:
            items.append(_item(f"unplaced_{tid}", "warn", f"{TRACK_KO.get(tid, tid)}: 음 {n}개 중 {dropped}개를 탭에 놓지 못했습니다",
                               f"옥타브를 옮겨 놓은 음 {int(ck.get('octave_shifted') or 0)}개. 오선보·MIDI 에는 남아 있습니다.",
                               notes=n, dropped=dropped))
    for inst in ("guitar", "bass"):
        tu = ((tab or {}).get("tuning") or {}).get(inst) or {}
        if not tu or tu.get("user"):
            continue
        margin = tu.get("margin")
        top = tu.get("top") or []
        if margin is None or not top:
            continue
        ko = "기타" if inst == "guitar" else "베이스"
        if float(margin) < 0 and top[0].get("label") != tu.get("label"):
            items.append(_item(f"tuning_{inst}", "info", f"{ko} 튜닝 확인 필요: {ko} 음만 보면 1순위는 {top[0]['label']}, 기타·베이스를 함께 맞춰 {tu.get('label')} 로 적었습니다",
                               "다른 튜닝이면 --set tab.guitar_tuning=… (또는 bass_tuning) 으로 다시 돌리세요.",
                               chosen=tu.get("label"), own_best=top[0].get("label"), margin=margin))
        elif float(margin) < TUNING_MARGIN_LOW:
            ru = (tu.get("runner_up") or {}).get("label")
            items.append(_item(f"tuning_{inst}", "info", f"{ko} 튜닝 {tu.get('label')} 은 확신이 낮습니다(2위 {ru})",
                               chosen=tu.get("label"), runner_up=ru, margin=margin))

    if "stems" in dirs and (Path(dirs["stems"]) / "energy.npz").is_file() and grid:
        with np.load(Path(dirs["stems"]) / "energy.npz") as z:
            spans, share = bleed_risk({k: z[k] for k in z.files}, grid)
        metrics["bleed_risk_bar_share"] = round(share, 3)
        if share >= BLEED_INFO_SHARE:
            items.append(_item("bleed_risk", "info",
                               f"건반·현악·신스가 기타만큼 큰 마디가 {share * 100:.0f} % 입니다 — 이 구간 기타 탭에 다른 악기 음이 섞였을 수 있습니다",
                               "전사기는 이런 음을 기타가 아니라고 표시하지 못합니다(E37). 표시만 합니다.", spans, share=round(share, 3)))

    router = _read(dirs, "router", "router.json") or {}
    wins = [w for w in router.get("windows") or [] if w.get("label") not in (None, "silent")]
    if wins:
        out = [w for w in wins if w.get("label") != "R3"]
        tot = sum(float(w["end_s"]) - float(w["start_s"]) for w in wins)
        share = sum(float(w["end_s"]) - float(w["start_s"]) for w in out) / tot if tot else 0.0
        metrics["merged_parts_share"] = round(share, 3)
        if share >= MERGED_INFO_SHARE:
            spans = _merge([[float(w["start_s"]), float(w["end_s"])] for w in out], gap=0.05)
            items.append(_item("merged_parts", "info",
                               f"기타가 가운데·넓은 더블로 깔린 구간({share * 100:.0f} %)은 파트를 나누지 못하고 기타 1 에 합쳤습니다",
                               "그 구간의 기타 1 은 한 사람의 파트가 아닐 수 있습니다.", spans, share=round(share, 3)))

    try:
        from bandscribe.pipeline import runner

        for w in runner.run_warnings(dirs):
            items.append(_item("run", "info", w))
    except Exception:  # noqa: BLE001 - the run's own warnings are a bonus here
        pass

    order = {"warn": 0, "info": 1}
    items.sort(key=lambda it: order.get(it["level"], 2))
    return {"format": FORMAT, "code_version": CODE_VERSION, "job": job_dir.name,
            "score_key": Path(dirs["score"]).name if "score" in dirs else None, "duration_s": duration,
            "summary": {"warn": sum(1 for i in items if i["level"] == "warn"),
                        "info": sum(1 for i in items if i["level"] == "info")},
            "metrics": metrics, "items": items}


def write_health(job_dir: Path, *, results: Mapping[str, Path] | None = None) -> dict[str, Any]:
    doc = song_health(job_dir, results=results)
    atomic.write_json(Path(job_dir) / "export" / HEALTH_FILE, doc)
    return doc


def load_health(job_dir: Path) -> dict[str, Any]:
    """``export/health.json`` when it belongs to the exported score (same score key and code version), else rebuilt."""
    p = Path(job_dir) / "export" / HEALTH_FILE
    try:
        doc = atomic.read_json(p)
    except (OSError, ValueError):
        doc = None
    from bandscribe.tab import edits as E

    if (isinstance(doc, dict) and doc.get("code_version") == CODE_VERSION
            and doc.get("score_key") == E.export_score_key(Path(job_dir))):
        return doc
    return write_health(job_dir)
