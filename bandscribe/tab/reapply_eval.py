"""E26 (M4b): how many edit anchors made on the M4a-era result still land after M5 / M6 / M7a changed it.

For each job with today's export and a ``quant`` result made on the M4a day (``M4A_DAY``, UTC), synthetic edits are
anchored on that old result (``docs/decisions.md`` E26): ``N_NOTES`` guitar and bass notes (a ``pitch`` edit each)
and ``N_BARS`` bar lines (moved one beat later). Every edit is matched on its own against the job's current base,
with the matching rules of ``bandscribe.tab.edits`` (``find`` for notes; ``_beat_of`` and "a bar line starts at that
beat" for bar lines), and classified (exact / within the tolerance / missed, and why it was missed).

Rows (metrics.csv): ``system`` = ``anchors``, ``item`` = ``<job>:<kind>:<k>``, ``group`` = job (bootstrap block),
``line`` = kind (``guitar`` / ``bass`` / ``bar``), ``metric`` = ``reapply_rate`` (1 = it landed), ``note`` = class.
"""

from __future__ import annotations

import bisect
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths

M4A_DAY = "2026-10-05"
N_NOTES = 50
N_BARS = 20
SEED = 20260930
THRESHOLD = 0.90
WIDE_TOL_S = 0.06   # descriptive only: the rate a 60 ms tolerance would give
NEAR_S = 0.10       # a missed note with its pitch this close is a tolerance miss
KINDS = (("guitar", "guitar_all"), ("bass", "bass"))
OLD_IDS = {"bass_raw": "bass"}


def _manifest(d: Path) -> dict[str, Any]:
    return atomic.read_json(d / "manifest.json")


def old_quant(job_dir: Path, day: str = M4A_DAY) -> Path | None:
    """The last ``quant`` result made on ``day`` (UTC), or None."""
    best = None
    for d in sorted((Path(job_dir) / "stages" / "quant").glob("*/")):
        try:
            m = _manifest(d)
        except (OSError, ValueError):
            continue
        c = str(m.get("created_utc") or "")
        if c.startswith(day) and (d / "quant.json").is_file() and (best is None or c > best[0]):
            best = (c, d)
    return best[1] if best else None


def jobs(root: Path | None = None) -> list[tuple[Path, Path]]:
    """(job dir, old quant dir) of every job E26 can measure, in folder-name order."""
    from bandscribe.tab import edits as E

    out = []
    for j in sorted(Path(root or paths.JOBS).glob("*/")):
        old = old_quant(j)
        if old is None or E.export_score_key(j) is None:
            continue
        try:
            E.load_base(j)
        except E.EditError:
            continue
        out.append((j, old))
    return out


def old_tracks(old: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """(track -> notes, grid) of an old quant result (today's track ids)."""
    q = atomic.read_json(old / "quant.json")
    notes = {OLD_IDS.get(t, t): list(tr.get("notes") or []) for t, tr in (q.get("tracks") or {}).items()}
    gkey = str(_manifest(old)["deps"]["grid"])[:12]
    grid = atomic.read_json(old.parent.parent / "grid" / gkey / "grid.json")
    return notes, grid


def sample_edits(notes: Mapping[str, Sequence[Mapping[str, Any]]], grid: Mapping[str, Any], rng: random.Random
                 ) -> list[dict[str, Any]]:
    """The registered synthetic edits of one job (notes first, then bar lines; ``kind`` and ``k`` added)."""
    from bandscribe.tab import edits as E

    out: list[dict[str, Any]] = []
    for kind, tid in KINDS:
        ns = list(notes.get(tid) or [])
        for k, i in enumerate(sorted(rng.sample(range(len(ns)), min(N_NOTES, len(ns))))):
            n = ns[i]
            out.append({"kind": kind, "k": k, "op": "pitch", "id": f"{kind}{k}", "note": E.anchor(tid, n),
                        "to": int(n["pitch"]) + 1})
    times = [float(b["t_s"]) for b in grid["beats"]]
    starts = E.bar_starts(grid)[1:]
    cand = [s for s in starts if s + 1 < len(times)]
    for k, s in enumerate(sorted(rng.sample(cand, min(N_BARS, len(cand))))):
        out.append({"kind": "bar", "k": k, "op": "barline", "id": f"bar{k}", "at": s, "to": s + 1,
                    "at_s": times[s], "to_s": times[s + 1]})
    return out


def _nearest(ns: Sequence[Mapping[str, Any]], pitch: int, t: float, tol: float) -> list[float]:
    return sorted(abs(float(n["onset_s"]) - t) for n in ns
                  if int(n["pitch"]) == pitch and not n.get("added") and abs(float(n["onset_s"]) - t) <= tol)


def classify_note(notes: Mapping[str, list[dict[str, Any]]], e: Mapping[str, Any]) -> dict[str, Any]:
    from bandscribe.tab import edits as E

    a = e["note"]
    hit = E.find(notes, a)
    ns = notes.get(a["track"]) or []
    p, t = int(a["pitch"]), float(a["onset_s"])
    wide = _nearest(ns, p, t, WIDE_TOL_S)
    out = {"landed": hit is not None, "wide": bool(wide), "wide_ambiguous": len(wide) >= 2}
    if hit is not None:
        out["class"] = "exact" if hit[2] else "within_tol"
    elif _nearest(ns, p, t, NEAR_S):
        out["class"] = "near_miss"
    elif _nearest(ns, p + 12, t, E.MATCH_TOL_S) or _nearest(ns, p - 12, t, E.MATCH_TOL_S):
        out["class"] = "octave"
    else:
        out["class"] = "gone"
    return out


def classify_bar(grid: Mapping[str, Any], e: Mapping[str, Any]) -> dict[str, Any]:
    from bandscribe.tab import edits as E

    times = [float(b["t_s"]) for b in grid["beats"]]
    starts = set(E.bar_starts(grid))
    at, to = E._beat_of(times, e, "at"), E._beat_of(times, e, "to")
    if at is not None and to is not None and at in starts:
        j = bisect.bisect_left(times, float(e["at_s"]))
        exact = any(0 <= x < len(times) and abs(times[x] - float(e["at_s"])) <= E.EXACT_TOL_S for x in (j - 1, j))
        return {"landed": True, "class": "exact" if exact else "within_tol"}
    return {"landed": False, "class": "beat_not_bar" if at is not None else "no_beat"}


def measure_job(job_dir: Path, old: Path, index: int) -> list[dict[str, Any]]:
    """One row dict per edit: kind, k, landed, class, (notes) wide / wide_ambiguous."""
    from bandscribe.tab import edits as E

    base = E.load_base(job_dir)
    state = E.apply(base, [])
    from bandscribe.tab import quantize as Q

    new_notes = {tid: Q.notes_of(state.quant, tid) for tid in E.TRACKS}
    notes, grid = old_tracks(old)
    edits = sample_edits(notes, grid, random.Random(SEED + index))
    out = []
    for e in edits:
        c = classify_note(new_notes, e) if e["op"] == "pitch" else classify_bar(state.grid, e)
        out.append({"kind": e["kind"], "k": e["k"], **c})
    return out


def block_ci(per_song: Mapping[str, tuple[int, int]], *, n: int = 10_000, seed: int = SEED) -> tuple[float, float, float]:
    """(rate, lo, hi): pooled rate with a song-block bootstrap 95 % interval."""
    keys = sorted(per_song)
    hit = sum(per_song[k][0] for k in keys)
    tot = sum(per_song[k][1] for k in keys)
    rate = hit / tot if tot else float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        h = sum(per_song[k][0] for k in pick)
        t = sum(per_song[k][1] for k in pick)
        vals.append(h / t if t else float("nan"))
    vals.sort()
    return rate, vals[int(0.025 * n)], vals[min(n - 1, int(0.975 * n))]


def run_e26(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.amt.experiments import ExperimentDataMissing

    found = jobs(Path(args["jobs_root"]) if args.get("jobs_root") else None)
    if not found:
        raise ExperimentDataMissing(f"E26: {M4A_DAY} 에 만든 quant 결과와 지금 export 가 둘 다 있는 작업이 없습니다 "
                                    f"({paths.JOBS})")
    rows: list[dict] = []
    detail: dict[str, Any] = {}
    per_song: dict[str, tuple[int, int]] = {}
    per_kind: dict[str, dict[str, tuple[int, int]]] = {"guitar": {}, "bass": {}, "bar": {}}
    for i, (j, old) in enumerate(found):
        if progress:
            progress(f"E26: {j.name}")
        res = measure_job(j, old, i)
        for r in res:
            rows.append({"system": "anchors", "dataset": "tierA-jobs", "item": f"{j.name}:{r['kind']}:{r['k']}",
                         "group": j.name, "line": r["kind"], "metric": "reapply_rate",
                         "value": 1.0 if r["landed"] else 0.0, "n": 1, "note": r["class"]})
        per_song[j.name] = (sum(r["landed"] for r in res), len(res))
        for kind in per_kind:
            rs = [r for r in res if r["kind"] == kind]
            per_kind[kind][j.name] = (sum(r["landed"] for r in rs), len(rs))
        notes = [r for r in res if r["kind"] != "bar"]
        detail[j.name] = {"old_quant": old.name, "classes": dict(Counter(f"{r['kind']}:{r['class']}" for r in res)),
                          "landed": per_song[j.name][0], "total": per_song[j.name][1],
                          "wide_landed": sum(r["wide"] for r in notes), "wide_ambiguous": sum(r["wide_ambiguous"]
                                                                                              for r in notes)}
    rate, lo, hi = block_ci(per_song)
    kinds = {k: dict(zip(("rate", "ci_low", "ci_high"), block_ci(v))) for k, v in per_kind.items()}
    decision = {"rate": rate, "ci_low": lo, "ci_high": hi, "threshold": THRESHOLD,
                "verdict": "PASS" if rate >= THRESHOLD else "FAIL", "kinds": kinds,
                "landed": sum(v[0] for v in per_song.values()), "total": sum(v[1] for v in per_song.values())}
    atomic.write_json(Path(out_dir) / "e26_items.json", detail)
    atomic.write_json(Path(out_dir) / "e26_decision.json", decision)
    return rows
