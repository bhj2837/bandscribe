"""E28 (docs/decisions.md): the part count at the same position - the ``width`` evidence (``reid``: a centred part
next to a wide double in R2 windows; ``arbitration``: centred vs wide notes) against v0.

Items: E25's 32 s remixes (P, ADT, A1, B2, N1; rebuilt from the caches only), E27's whole-song remixes (P, A1, B2)
and the whole-song one-guitar control ``D1`` (A's double alone, ``eval_track`` ``<song>.d1.json``). Arms: ``v0``
(the pipeline's ``[parts]``) and ``width`` (the same with ``use_width`` and the dev-chosen thresholds). Per remix:
``k_pred``, ``k_correct``, ``assign_macro`` (E25's; remixes with one true Line score the count only).

Dev selection (dev songs only): ``GRID`` in order (conservative first); objective, lexicographic: one-guitar remixes
(ADT, N1, D1) counted 1, then part counts right over all dev remixes, then the mean A1 macro; ties -> grid order.
"""

from __future__ import annotations

import random
import statistics as stx
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic

GRID = [{"width_ms_hi": hi, "width_coh_hi": ch, "width_ms_lo": lo, "width_n_min": nm}
        for hi in (10.0, 6.0) for ch in (0.95, 0.9) for lo in (0.0, 2.0) for nm in (6, 4)]
ONE_GUITAR = ("ADT", "N1", "D1")
SEED = 20260930


def predict(it: Mapping[str, Any], count: Mapping[str, Any]) -> dict[str, Any]:
    from bandscribe.parts import diarize as DZ
    from bandscribe.parts import eval as PE

    d = DZ.diarize(it["sys"], it["sigs"], it["windows"], params=count)
    pred = [None if n["line"] is None else str(n["line"]) for n in d["note"]]
    k = d["count"]["chosen"]
    macro = PE.macro_hungarian(it["truth"], pred)[0] if it["k_true"] >= 2 else None
    return {"k_pred": k, "k_correct": float(k == it["k_true"]), "assign_macro": macro,
            "evidence": d["count"].get("evidence"), "support": d["count"].get("support")}


def objective(items: Sequence[Mapping[str, Any]], count: Mapping[str, Any]) -> tuple[int, float, float]:
    rs = [(it, predict(it, count)) for it in items]
    one = sum(1 for it, r in rs if it["scenario"] in ONE_GUITAR and r["k_pred"] == 1)
    kc = stx.mean(r["k_correct"] for _it, r in rs) if rs else 0.0
    a1 = [r["assign_macro"] for it, r in rs if it["scenario"] == "A1" and r["assign_macro"] is not None]
    return one, kc, stx.mean(a1) if a1 else 0.0


def choose(dev: Sequence[Mapping[str, Any]], base: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    trace, best = [], None
    for gi, g in enumerate(GRID):
        one, kc, a1 = objective(dev, {**base, "use_width": True, **g})
        trace.append({**g, "one_guitar_1": one, "k_correct": round(kc, 4), "a1_macro": round(a1, 4)})
        key = (-one, -kc, -a1, gi)
        if best is None or key < best[0]:
            best = (key, g)
    return dict(best[1]), trace


def block_bootstrap(diffs: Mapping[str, Sequence[float]], n: int = 10_000, seed: int = SEED
                    ) -> tuple[float, float, float]:
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
    """The registered rules on the test remixes (one row per remix with both arms)."""
    one = [r for r in rows if r["scenario"] in ONE_GUITAR]
    rule1 = all(r["width"]["k_pred"] == 1 for r in one) and bool(one)
    a1 = [r for r in rows if r["scenario"] == "A1"]
    a1_k = {a: sum(1 for r in a1 if r[a]["k_pred"] == r["k_true"]) for a in ("v0", "width")}
    diffs: dict[str, list[float]] = {}
    for r in a1:
        diffs.setdefault(r["song"], []).append(100.0 * (r["width"]["assign_macro"] - r["v0"]["assign_macro"]))
    a1_diff = dict(zip(("mean", "ci_low", "ci_high"), block_bootstrap(diffs)))
    rule2 = a1_k["width"] > a1_k["v0"] and a1_diff["mean"] >= 10.0 and a1_diff["ci_low"] > 0.0
    other: dict[str, dict[str, float]] = {}
    for sc in sorted({r["scenario"] for r in rows if r["scenario"] not in ONE_GUITAR + ("A1",)}):
        dd: dict[str, list[float]] = {}
        for r in rows:
            if r["scenario"] == sc and r["v0"]["assign_macro"] is not None:
                dd.setdefault(r["song"], []).append(100.0 * (r["width"]["assign_macro"] - r["v0"]["assign_macro"]))
        other[sc] = dict(zip(("mean", "ci_low", "ci_high"), block_bootstrap(dd)))
    rule3 = all(v["ci_high"] >= -2.0 for v in other.values() if v["ci_high"] == v["ci_high"])
    return {"one_guitar": {"n": len(one), "counted_1": sum(1 for r in one if r["width"]["k_pred"] == 1),
                           "v0_counted_1": sum(1 for r in one if r["v0"]["k_pred"] == 1)},
            "rule1_one_guitar": rule1, "a1_k_correct": a1_k, "a1_n": len(a1), "a1_macro_diff": a1_diff,
            "rule2_a1": rule2, "other_scenarios": other, "rule3_no_regression": rule3,
            "verdict": "PASS" if rule1 and rule2 and rule3 else "FAIL"}


def load_all(cfg: Any) -> list[dict[str, Any]]:
    """E25's 32 s items (cache-only rebuild), E27's whole-song items and the D1 controls, each with ``set``."""
    from bandscribe.parts import eval_track as ET

    out = []
    for it in e25_items(cfg):
        out.append({**it, "set": "32s"})
    for it in ET.load_items():
        out.append({**it, "set": "full"})
    for it in ET.load_items(pattern="*.d1.json"):
        out.append({**it, "set": "full"})
    return out


def e25_items(cfg: Any) -> list[dict[str, Any]]:
    """E25's prepared remixes; GPU workers are refused (the caches must hold every view)."""
    import tempfile

    from bandscribe import gpu, vram
    from bandscribe.amt.experiments import ExperimentDataMissing
    from bandscribe.parts import eval as PE

    def no_gpu(*a: Any, **k: Any) -> Any:
        raise ExperimentDataMissing("E28: E25 의 캐시에 없는 뷰가 있습니다(GPU 를 쓰지 않습니다). 먼저 bandscribe eval exp E25 의 "
                                    "캐시를 채우세요.")

    saved = (gpu.run_worker, vram.run_gpu_stage)
    gpu.run_worker, vram.run_gpu_stage = no_gpu, no_gpu
    try:
        with tempfile.TemporaryDirectory(prefix="e28_") as tmp:
            items, _info = PE._prepare(Path(tmp), cfg, {}, None)
    finally:
        gpu.run_worker, vram.run_gpu_stage = saved
    return [it for it in items if it["k_true"] >= 1]


def run_e28(out_dir: Path, cfg: Any, args: dict, *, progress: Callable[[str], None] | None = None) -> list[dict]:
    from bandscribe.parts.stage import parts_params

    items = load_all(cfg)
    base = parts_params(cfg)["count"]
    dev = [it for it in items if it["split"] == "dev"]
    chosen, trace = choose(dev, base)
    rows_out: list[dict] = []
    per: list[dict] = []
    for it in items:
        if progress:
            progress(f"E28: {it['song']} {it['scenario']} {it['set']}")
        r = {"song": it["song"], "scenario": it["scenario"], "set": it["set"], "split": it["split"],
             "k_true": it["k_true"], "v0": predict(it, base), "width": predict(it, {**base, "use_width": True, **chosen})}
        per.append(r)
        for arm in ("v0", "width"):
            for m in ("k_correct", "assign_macro"):
                v = r[arm][m]
                if v is not None:
                    rows_out.append({"system": arm, "dataset": "cambridge_mt",
                                     "item": f"{it['song']}:{it['scenario']}:{it['set']}", "group": it["song"],
                                     "scenario": it["scenario"], "section": it["split"], "line": it["set"],
                                     "metric": m, "value": round(float(v), 4)})
    atomic.write_json(Path(out_dir) / "e28_items.json", per)
    atomic.write_json(Path(out_dir) / "e28_decision.json", {
        "chosen": chosen, "trace": trace, "base": base,
        "test": decide([r for r in per if r["split"] == "test"]),
        "dev": decide([r for r in per if r["split"] == "dev"])})
    return rows_out
