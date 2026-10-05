"""Tab experiments of the registry (docs/decisions.md): ``E12`` fretting vs tuttut on GuitarSet.

E12 rows (metrics.csv): ``system`` = arm (``tuttut`` or ``dp_s<shift>_h<height>``), ``item`` = GuitarSet track,
``group`` = composition (style + progression + tempo + key, e.g. ``BN2-131-B``; the bootstrap block),
``section`` = ``dev`` (progression 1) / ``test`` (progressions 2, 3), ``scenario`` = ``comp`` / ``solo``.
Metrics: ``string_agreement`` (points, per track), ``playability_violations`` (events breaking the invariant),
``dropped_notes``, ``seconds``. Input notes are the GT notes (oracle pitches), grouped into events by onset
within ``EVENT_TOL_S`` of the event's first note, the same events for both systems.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths
from bandscribe.tab import fretting as F

log = logging.getLogger(__name__)

E12_SHIFTS = (0.3, 0.6, 1.2)
E12_HEIGHTS = (0.05, 0.15, 0.4)
EVENT_TOL_S = 0.05
_ID = re.compile(r"^(\d+)_([A-Za-z]+)(\d)-([^_]+)_(comp|solo)$")


def e12_arms() -> dict[str, dict[str, float]]:
    return {f"dp_s{s}_h{h}": {"shift": s, "height": h} for s in E12_SHIFTS for h in E12_HEIGHTS}


def track_meta(track_id: str) -> dict[str, str]:
    """{split, group, mode} of a GuitarSet track id like ``00_BN1-129-Eb_comp``."""
    m = _ID.match(track_id)
    if not m:
        raise ValueError(f"not a GuitarSet track id: {track_id}")
    _player, style, prog, rest, mode = m.groups()
    return {"split": "dev" if prog == "1" else "test", "group": f"{style}{prog}-{rest}", "mode": mode}


def _gt(track: Any) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    (line, ns), = track.notes.items()
    sf = (track.string_fret or {}).get(line)
    if sf is None:
        raise ValueError(f"{track.track_id}: no string/fret labels")
    notes = [{"onset_s": float(n.onset_s), "offset_s": float(n.offset_s), "pitch": int(n.pitch)} for n in ns.notes]
    return notes, [(int(s), int(f)) for s, f in sf]


def _agreement(pred: dict[int, tuple[int, int] | None], gt: Sequence[tuple[int, int]]) -> float:
    return 100.0 * sum(1 for i, g in enumerate(gt) if pred.get(i) == g) / len(gt) if gt else float("nan")


def run_tuttut(tracks: list[dict[str, Any]], work: Path, *, tuning: Sequence[int] = F.GUITAR_STD,
               processes: int | None = None, timeout_s: float = 3600.0) -> dict[str, dict[str, Any]]:
    """tuttut over ``tracks`` = [{"id", "events": [[onset, [pitches]]]}] in parallel processes of the tuttut env;
    returns {id: {"events": [[[string, fret], ...] or None per event] or None, "error", "seconds"}}."""
    py = Path(paths.TUTTUT_PYTHON)
    if not py.is_file():
        raise RuntimeError(f"tuttut 환경이 없습니다: {py} (envs\\tuttut 에서 ..\\..\\tools\\uvw.cmd sync)")
    runner = Path(__file__).with_name("tuttut_runner.py")
    k = max(1, min(processes or max(1, (os.cpu_count() or 2) - 1), len(tracks)))
    work.mkdir(parents=True, exist_ok=True)
    procs = []
    for j in range(k):
        chunk = tracks[j::k]
        req, res = work / f"tuttut_req_{j}.json", work / f"tuttut_res_{j}.json"
        atomic.write_json(req, {"tuning": list(tuning), "tracks": chunk})
        env = {**paths.child_env(), "PYTHONWARNINGS": "ignore"}
        procs.append((subprocess.Popen([str(py), str(runner), str(req), str(res)], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE), res))
    out: dict[str, dict[str, Any]] = {}
    for p, res in procs:
        try:
            _o, err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            raise
        if p.returncode != 0 or not res.is_file():
            raise RuntimeError(f"tuttut 실행 실패 (코드 {p.returncode}): {err.decode('utf-8', 'replace')[-800:]}")
        for t in json.loads(res.read_text(encoding="utf-8"))["tracks"]:
            out[t["id"]] = t
    return out


def run_e12(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe import datasets

    ids = sorted(datasets.track_ids("guitarset"))
    if args.get("limit"):
        ids = ids[: int(args["limit"])]
    inst = F.Instrument(F.GUITAR_STD, max_fret=24)
    data = []
    for tid in ids:
        tr = datasets.load_track("guitarset", tid)
        notes, gt = _gt(tr)
        events = F.group_events(notes, key="onset_s", tol=EVENT_TOL_S)
        data.append((tid, track_meta(tid), notes, gt, events))
    rows: list[dict] = []

    def row(arm: str, tid: str, meta: dict, metric: str, value: float, n: int | None = None) -> None:
        rows.append({"system": arm, "dataset": "guitarset", "item": tid, "group": meta["group"],
                     "scenario": meta["mode"], "section": meta["split"], "metric": metric, "value": value,
                     **({"n": n} if n is not None else {})})

    # tuttut (baseline)
    req = [{"id": tid, "events": [[ev.onset_s, ev.pitches] for ev in events]} for tid, _m, _n, _g, events in data]
    t0 = time.monotonic()
    tt = run_tuttut(req, Path(out_dir) / "tuttut")
    failed = sorted(t for t, r in tt.items() if r.get("events") is None)
    log.info("tuttut: %d tracks in %.0f s, %d failed", len(tt), time.monotonic() - t0, len(failed))
    atomic.write_json(Path(out_dir) / "tuttut_failures.json",
                      {"failed": failed, "errors": {t: tt[t].get("error") for t in failed}})
    keep = [d for d in data if d[0] not in failed]
    for tid, meta, _notes, gt, events in keep:
        res = tt[tid]["events"]
        pred: dict[int, tuple[int, int] | None] = {}
        for ev, got in zip(events, res):
            avail = list(got or [])
            for pitch, nid in zip(ev.pitches, ev.ids):
                hit = next((sf for sf in avail if inst.open_pitch(sf[0]) + sf[1] == pitch), None)
                if hit is not None:
                    avail.remove(hit)
                    pred[nid] = (int(hit[0]), int(hit[1]))
        row("tuttut", tid, meta, "string_agreement", _agreement(pred, gt), len(gt))
        row("tuttut", tid, meta, "seconds", float(tt[tid].get("seconds") or 0.0))

    # bandscribe Viterbi arms
    for arm, w in e12_arms().items():
        if progress:
            progress(arm)
        for tid, meta, _notes, gt, events in keep:
            t1 = time.monotonic()
            res = F.assign(events, inst, w)
            dt = time.monotonic() - t1
            pred = {a.note_id: (a.string, a.fret) for a in res if a.string is not None}
            chk = F.playable(res, events, inst)
            row(arm, tid, meta, "string_agreement", _agreement(pred, gt), len(gt))
            row(arm, tid, meta, "playability_violations", float(len(chk["bad_events"])))
            row(arm, tid, meta, "dropped_notes", float(chk["dropped"]))
            row(arm, tid, meta, "seconds", round(dt, 4))
    atomic.write_json(Path(out_dir) / "e12_tracks.json", {"tracks": len(data), "tuttut_failed": failed,
                                                          "event_tol_s": EVENT_TOL_S})
    return rows


EXPERIMENTS: dict[str, Callable[[Path, Any, dict], list[dict]]] = {"E12": run_e12}
