"""``gtab eval compare A B``: paired song-block bootstrap of B − A over the items both runs scored."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gtab.eval import stats
from gtab.eval.experiments import items_from_rows
from gtab.eval.runs import read_metrics_csv

COMPARE_METRICS = ("note_f1_onset50", "note_f1_onset100", "note_f1_offset50", "tick_f1", "chroma_f1_onset50",
                   "assign_macro", "assign_weighted", "unassigned_ratio", "shared_f1")
DECISION_KO = {"adopt": "B가 나음", "reject": "B가 낫지 않음", "hold": "판단 보류"}


class CompareError(RuntimeError):
    """User-facing (Korean)."""


def _item_rows(rows: Sequence[Mapping[str, str]], metric: str) -> dict[str, Mapping[str, str]]:
    return {r["item"]: r for r in rows if r["metric"] == metric and r["item"] != "*" and not r.get("line")}


def compare_runs(a: Path, b: Path, cfg: Any = None) -> dict[str, Any]:
    fa, fb = Path(a) / "metrics.csv", Path(b) / "metrics.csv"
    if not fa.is_file() or not fb.is_file():
        raise CompareError("두 실행 폴더 모두에 metrics.csv 가 있어야 합니다")
    ra, rb = read_metrics_csv(fa), read_metrics_csv(fb)
    n = int(cfg.get("eval.bootstrap_iterations", stats.DEFAULT_ITERATIONS)) if cfg is not None else stats.DEFAULT_ITERATIONS
    seed = int(cfg.get("eval.seed", stats.DEFAULT_SEED)) if cfg is not None else stats.DEFAULT_SEED
    mde = float(cfg.get("eval.mde_points", 1.0)) if cfg is not None else 1.0
    out: dict[str, Any] = {"a": Path(a).name, "b": Path(b).name, "metrics": {}, "warnings": [], "n_common_items": 0}
    common_all: set[str] = set()
    for m in COMPARE_METRICS:
        ia, ib = _item_rows(ra, m), _item_rows(rb, m)
        common = sorted(set(ia) & set(ib))
        if not common:
            continue
        common_all |= set(common)
        diff_rung = [i for i in common if (ia[i].get("rung") or "") != (ib[i].get("rung") or "")]
        if diff_rung:
            out["warnings"].append(f"{m}: 사다리 칸(rung)이 다른 항목 {len(diff_rung)}개 — 칸 차이가 섞인 비교입니다")
        xa = items_from_rows([ia[i] for i in common])
        xb = items_from_rows([ib[i] for i in common])
        stat = stats.f1_points if xa and "tp" in xa[0].counts else stats.ratio_points
        d, lo, hi = stats.paired_bootstrap(xa, xb, stat, n=n, seed=seed)
        pa = stats.block_bootstrap(xa, stat, n=0)[0]
        pb = stats.block_bootstrap(xb, stat, n=0)[0]
        subsets = {}
        for sc in sorted({ia[i].get("scenario") or "" for i in common} - {""}):
            sel = [i for i in common if (ia[i].get("scenario") or "") == sc]
            subsets[sc] = stats.paired_bootstrap(items_from_rows([ia[i] for i in sel]), items_from_rows([ib[i] for i in sel]),
                                                 stat, n=n, seed=seed)
        dec = stats.decide(d, lo, hi, mde, subsets)
        if m == "unassigned_ratio":  # lower is better: no adopt/reject semantics, report only
            dec_ko = "참고"
        else:
            dec_ko = DECISION_KO[dec]
        out["metrics"][m] = {"a": pa / 100, "b": pb / 100, "delta": d, "ci": [lo, hi], "decision": dec,
                             "decision_ko": dec_ko, "n_items": len(common),
                             "subsets": {k: list(v) for k, v in subsets.items()}}
    out["n_common_items"] = len(common_all)
    if not out["metrics"]:
        raise CompareError("두 실행에 공통 항목이 없습니다 (같은 평가 세트인지 확인하세요)")
    return out
