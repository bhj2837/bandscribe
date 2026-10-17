"""E29 (docs/decisions.md): repeat consensus (``bandscribe.amt.consensus``) on the merged guitar notes.

Items: E27's whole-song remixes (P, A1, B2) and E28's one-guitar control D1 (``eval_track``; caches only, no GPU).
The guitar notes of a remix are its ``sys`` notes (MuScriptor on the separated guitar stem's mono); the reference is
the union of the included takes' own transcriptions (one note per pitch within 50 ms). Arms: ``off`` and
``consensus`` (dev-chosen ``GRID`` point). Metric: note F1 x 100 (onset 50 ms, +-50 cents, one-to-one).
"""

from __future__ import annotations

import statistics as stx
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.parts.eval_e28 import SEED, block_bootstrap

GRID = [{"add_frac": af, "sim_min": sm, "tol_beats": tb}
        for af in (1.0, 2 / 3) for sm in (0.7, 0.5) for tb in (0.125, 0.0625)]
SCENARIOS = ("P", "A1", "B2", "D1")
DEDUPE_S = 0.05


def _arrays(notes: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np

    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[float(n["onset_s"]), max(float(n["onset_s"]), float(n.get("offset_s") or n["onset_s"]))]
                   for n in notes])
    return iv, np.array([float(n["pitch"]) for n in notes])


def note_f1(ref: Sequence[Mapping[str, Any]], est: Sequence[Mapping[str, Any]]) -> Any:
    from bandscribe.eval.metrics import note_prf

    return note_prf(_arrays(ref), _arrays(est))


def union_ref(lists: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """The takes' notes together, a note dropped when a kept one has its pitch within ``DEDUPE_S`` (a double)."""
    alln = sorted((dict(n) for ns in lists for n in ns), key=lambda n: (int(n["pitch"]), float(n["onset_s"])))
    out: list[dict[str, Any]] = []
    for n in alln:
        if out and int(out[-1]["pitch"]) == int(n["pitch"]) and float(n["onset_s"]) - float(out[-1]["onset_s"]) <= DEDUPE_S:
            continue
        out.append(n)
    return sorted(out, key=lambda n: float(n["onset_s"]))


def take_notes(song: str, takes: Sequence[str], cfg: Any) -> dict[str, list[dict[str, Any]]]:
    """Each take's own guitar transcription from the eval cache (a missing one is an error: no GPU here)."""
    from bandscribe import audio as AU
    from bandscribe.amt import experiments as AE
    from bandscribe.amt import instruments as I
    from bandscribe.amt import stage as S
    from bandscribe.amt.experiments import ExperimentDataMissing
    from bandscribe.parts import eval_track as ET

    def no_gpu(*a: Any, **k: Any) -> Any:
        raise ExperimentDataMissing(f"E29: {song} 테이크 전사가 캐시에 없습니다(GPU 를 쓰지 않습니다). 먼저 E27 캐시를 채우세요.")

    params = S.ms_params(cfg)
    work = ET.cache_dir() / "work" / song
    views = []
    for t in takes:
        wav = work / f"take{t}_mono.wav"
        if not wav.is_file():
            raise ExperimentDataMissing(f"E29: {wav} 가 없습니다")
        views.append({"id": str(t), "wav": wav, "pcm_sha256": AU.pcm_sha256(wav), "instruments": I.mask_guitar_only()})
    raws, _ = S.transcribe_views_ms(views, params, cfg, work_dir=work / "_e29", cache=AE._eval_cache(),
                                    force_rung=AE.ms_rung(params), run_gpu_stage=no_gpu)
    g = set(I.GUITAR_CLASSES)
    return {str(t): [n for n in raws[str(t)].get("notes") or [] if n.get("instrument") in g] for t in takes}


def score(it: Mapping[str, Any], ref: Sequence[Mapping[str, Any]], params: Mapping[str, Any] | None) -> dict[str, Any]:
    from bandscribe.amt import consensus as C

    sys_notes = list(it["sys"])
    if params is None:
        return {"f1": round(100.0 * note_f1(ref, sys_notes).f1, 3)}
    r = C.consensus(sys_notes, it["grid"], it["repeat"], params)
    added = r["added"]
    rep = r["report"]
    return {"f1": round(100.0 * note_f1(ref, sys_notes + added).f1, 3), "added": len(added),
            "added_precision": round(note_f1(ref, added).precision, 4) if added else None, "lone": rep["lone"],
            "voted_cells": rep["voted_cells"], "cells": rep["cells"]}


def choose(dev: Sequence[tuple[Mapping[str, Any], list]]) -> tuple[dict[str, Any], list[dict]]:
    base = {id(it): score(it, ref, None)["f1"] for it, ref in dev}
    trace, best = [], None
    for gi, g in enumerate(GRID):
        d = [score(it, ref, g)["f1"] - base[id(it)] for it, ref in dev]
        m = stx.mean(d) if d else 0.0
        trace.append({**g, "dev_mean_diff": round(m, 4)})
        if best is None or (-m, gi) < best[0]:
            best = ((-m, gi), g)
    return dict(best[1]), trace


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diffs: dict[str, list[float]] = {}
    for r in rows:
        diffs.setdefault(r["song"], []).append(r["consensus"]["f1"] - r["off"]["f1"])
    mean, lo, hi = block_bootstrap(diffs, seed=SEED)
    rule1 = mean == mean and mean >= 2.0 and lo > 0.0
    per = {}
    for sc in SCENARIOS:
        d = [r["consensus"]["f1"] - r["off"]["f1"] for r in rows if r["scenario"] == sc]
        if d:
            per[sc] = {"n": len(d), "mean": round(stx.mean(d), 3)}
    rule2 = all(v["mean"] >= -1.0 for v in per.values())
    added = sum(r["consensus"]["added"] for r in rows)
    precs = [(r["consensus"]["added_precision"], r["consensus"]["added"]) for r in rows if r["consensus"]["added"]]
    return {"n": len(rows), "diff": {"mean": round(mean, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)},
            "rule1": rule1, "per_scenario": per, "rule2_no_regression": rule2, "added": added,
            "added_precision": round(sum(p * n for p, n in precs) / added, 4) if added else None,
            "touched_items": sum(1 for r in rows if r["consensus"]["voted_cells"]),
            "verdict": "PASS" if rule1 and rule2 else "FAIL"}


def run_e29(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.parts import eval_track as ET

    items = [*ET.load_items(), *ET.load_items(pattern="*.d1.json")]
    refs: dict[str, dict[str, list]] = {}
    pairs = []
    for it in items:
        if it["song"] not in refs:
            refs[it["song"]] = {}
        need = [t for t in sorted(it["takes"]) if t not in refs[it["song"]]]
        if need:
            if progress:
                progress(f"E29: {it['song']} 테이크 {need}")
            refs[it["song"]].update(take_notes(it["song"], need, cfg))
        pairs.append((it, union_ref([refs[it["song"]][t] for t in sorted(it["takes"])])))
    chosen, trace = choose([(it, ref) for it, ref in pairs if it["split"] == "dev"])
    per, rows_out = [], []
    for it, ref in pairs:
        r = {"song": it["song"], "scenario": it["scenario"], "split": it["split"], "ref_notes": len(ref),
             "sys_notes": len(it["sys"]), "off": score(it, ref, None), "consensus": score(it, ref, chosen)}
        per.append(r)
        for arm in ("off", "consensus"):
            rows_out.append({"system": arm, "dataset": "cambridge_mt", "item": f"{it['song']}:{it['scenario']}",
                             "group": it["song"], "scenario": it["scenario"], "section": it["split"], "line": "full",
                             "metric": "f1", "value": r[arm]["f1"]})
    atomic.write_json(Path(out_dir) / "e29_items.json", per)
    atomic.write_json(Path(out_dir) / "e29_decision.json", {
        "chosen": chosen, "trace": trace, "test": decide([r for r in per if r["split"] == "test"]),
        "dev": decide([r for r in per if r["split"] == "dev"])})
    return rows_out
