"""E27 (M7b, docs/decisions.md): identity tracking on whole-song Tier B remixes (``eval_track``'s items).

Arms: ``v0`` (M7a's ``diarize`` with the pipeline's ``[parts]`` settings), ``track`` (the same plus the M7b tracker
with the dev-chosen params and the remix's repeat map), and the diagnostics ``v0|oracle_k`` / ``track|oracle_k``
(the true part count forced). Per remix with two or more true Lines: ``assign_macro`` (E25's), ``id_switches``
(``idmetrics.identity_switches`` total), ``block_der``, ``changed`` (notes the tracker relabelled).

Dev selection (dev songs only): ``GRID`` in order (conservative first); the first combination that keeps the dev
mean macro within ``MACRO_SLACK`` of v0's and has the fewest dev identity switches in total (ties: the higher mean
macro, then grid order). The decision rules are the registration's (``decide``).
"""

from __future__ import annotations

import random
import statistics as stx
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic

GRID = [{"p_floor": pf, "w_link": wl, "boundary": bd, "w_rep": wr}
        for pf in (0.0, 0.1) for wl in (1.0, 2.0, 4.0) for bd in (0.5, 0.25) for wr in (0.0, 1.0)]
MACRO_SLACK = 0.01
ARMS = ("v0", "track", "v0|oracle_k", "track|oracle_k")
SEED = 20260930


def preds(it: Mapping[str, Any], count: Mapping[str, Any], tparams: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from bandscribe.parts import diarize as DZ

    out = {}
    for kf, sfx in ((None, ""), (it["k_true"], "|oracle_k")):
        for arm, tr in (("v0", None), ("track", dict(tparams))):
            d = DZ.diarize(it["sys"], it["sigs"], it["windows"], params=count, k_force=kf, tracking=tr,
                           repeat=it.get("repeat") if tr is not None else None)
            out[arm + sfx] = {"pred": [None if n["line"] is None else str(n["line"]) for n in d["note"]],
                              "k": d["count"]["chosen"], "changed": (d["count"].get("track") or {}).get("changed", 0)}
    return out


def score(it: Mapping[str, Any], count: Mapping[str, Any], tparams: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from bandscribe.parts import eval as PE
    from bandscribe.parts import idmetrics as IM

    on = [float(n["onset_s"]) for n in it["sys"]]
    out = {}
    for arm, pr in preds(it, count, tparams).items():
        macro, ex = PE.macro_hungarian(it["truth"], pr["pred"])
        sw = IM.identity_switches(it["truth"], pr["pred"], on, ex["match"])
        der = IM.block_der(it["truth"], pr["pred"], on, ex["match"])
        out[arm] = {"assign_macro": macro, "id_switches": sw["total"], "block_der": der["der"], "k_pred": pr["k"],
                    "changed": pr["changed"], "per_line": ex["per_line"], "switch_lines": sw["per_line"]}
    return out


def choose(dev: Sequence[Mapping[str, Any]], count: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    trace = []
    base = [score(it, count, GRID[0])["v0"] for it in dev]
    m0 = stx.mean(b["assign_macro"] for b in base) if base else 0.0
    best = None
    for gi, g in enumerate(GRID):
        rs = [score(it, count, g)["track"] for it in dev]
        m = stx.mean(r["assign_macro"] for r in rs) if rs else 0.0
        sw = sum(r["id_switches"] for r in rs)
        ok = m >= m0 - MACRO_SLACK
        trace.append({**g, "macro": round(m, 4), "switches": sw, "ok": ok})
        key = (sw, -m, gi)
        if ok and (best is None or key < best[0]):
            best = (key, g)
    return (dict(best[1]) if best else dict(GRID[0])), trace


def block_bootstrap(diffs: Mapping[str, Sequence[float]], n: int = 10_000, seed: int = SEED
                    ) -> tuple[float, float, float]:
    """Mean of the per-remix differences with a song-block bootstrap 95 % interval."""
    songs = sorted(diffs)
    allv = [v for s in songs for v in diffs[s]]
    if not allv:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        pick = [v for _ in songs for v in diffs[songs[rng.randrange(len(songs))]]]
        vals.append(sum(pick) / len(pick))
    vals.sort()
    return sum(allv) / len(allv), vals[int(0.025 * n)], vals[min(n - 1, int(0.975 * n))]


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The registered rules on the test remixes (``rows``: one per remix with both arms' scores)."""
    sw_v0 = [r["v0"]["id_switches"] for r in rows]
    sw_tr = [r["track"]["id_switches"] for r in rows]
    med_v0, med_tr = (stx.median(sw_v0), stx.median(sw_tr)) if rows else (None, None)
    rule1 = bool(rows) and med_tr <= 2 and med_tr < med_v0
    diffs: dict[str, list[float]] = {}
    by_sc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        d = 100.0 * (r["track"]["assign_macro"] - r["v0"]["assign_macro"])
        diffs.setdefault(r["song"], []).append(d)
        by_sc.setdefault(r["scenario"], {}).setdefault(r["song"], []).append(d)
    mean, lo, hi = block_bootstrap(diffs)
    rule2 = bool(rows) and hi >= -2.0
    scen = {sc: dict(zip(("mean", "ci_low", "ci_high"), block_bootstrap(v))) for sc, v in sorted(by_sc.items())}
    regress = sorted(sc for sc, v in scen.items() if v["ci_high"] < -2.0)
    return {"n": len(rows), "median_switches": {"v0": med_v0, "track": med_tr}, "rule1_switches": rule1,
            "macro_diff_points": {"mean": mean, "ci_low": lo, "ci_high": hi}, "rule2_macro": rule2,
            "scenarios": scen, "regressing_scenarios": regress,
            "verdict": "PASS" if rule1 and rule2 else "FAIL",
            "der_by_song": {s: {a: _mean_or_none([r[a]["block_der"] for r in rows if r["song"] == s])
                                for a in ("v0", "track")} for s in sorted({r["song"] for r in rows})}}


def _mean_or_none(xs: Sequence[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return round(stx.mean(v), 4) if v else None


def run_e27(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.amt.experiments import ExperimentDataMissing
    from bandscribe.parts import eval_track as ET
    from bandscribe.parts.stage import parts_params

    items = [it for it in ET.load_items() if it["k_true"] >= 2]
    if not items:
        raise ExperimentDataMissing(f"E27: 곡 전체 리믹스 캐시가 없습니다({ET.cache_dir()}). data/tmp/m7b/fill_e27.py "
                                    "로 곡마다 채우세요(GPU).")
    count = parts_params(cfg)["count"]
    dev = [it for it in items if it["split"] == "dev"]
    chosen, trace = choose(dev, count)
    rows_out: list[dict] = []
    per_remix = []
    for it in items:
        if progress:
            progress(f"E27: {it['song']} {it['scenario']}")
        sc = score(it, count, chosen)
        per_remix.append({"song": it["song"], "scenario": it["scenario"], "split": it["split"],
                          "k_true": it["k_true"], "notes": len(it["sys"]), "duration_s": it.get("duration_s"),
                          **sc})
        for arm in ARMS:
            for m in ("assign_macro", "id_switches", "block_der"):
                v = sc[arm][m]
                if v is not None:
                    rows_out.append({"system": arm, "dataset": "cambridge_mt", "item": f"{it['song']}:{it['scenario']}",
                                     "group": it["song"], "scenario": it["scenario"], "section": it["split"],
                                     "line": "guitar", "metric": m, "value": round(float(v), 4)})
    test = [r for r in per_remix if r["split"] == "test"]
    devr = [r for r in per_remix if r["split"] == "dev"]
    atomic.write_json(Path(out_dir) / "e27_items.json", per_remix)
    atomic.write_json(Path(out_dir) / "e27_decision.json", {"chosen": chosen, "trace": trace, "test": decide(test),
                                                            "dev": decide(devr), "count_params": count})
    return rows_out
