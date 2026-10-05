"""report.md + report.html of the distractor suite (Korean). Both come from the same table data (summary.json).

No template engine: the report is a handful of tables, rendered by ``_md_table`` / ``_html_table`` from the same rows
so the two files never disagree. Numbers are rounded for reading; the exact values are in summary.json/metrics.csv.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bandscribe import atomic
from bandscribe.eval import stem_taxonomy as T

CAUSE_KO: dict[str, str] = {
    "guitar_timing": "기타 자신: 타이밍·중복",
    "guitar_octave": "기타 자신: 옥타브·배음",
    "guitar_near": "기타 자신: 반음·온음 어긋남",
    "guitar_unref": "기타 소리지만 참조에 없음",
    "unknown": "에너지 없음(불명)",
    "est_timing": "추정 음 있음: 타이밍 어긋남",
    "est_octave": "추정 음 있음: 옥타브",
    "est_near": "추정 음 있음: 반음·온음",
    "guitar_clear": "가림 없이 놓침(기타 우세)",
    "guitar": "기타",
    "none": "없음",
}


def cause_ko(k: str) -> str:
    return CAUSE_KO.get(k) or T.label_ko(k)


def cond_ko(c: str) -> str:
    if c.startswith("+"):
        return f"+{T.label_ko(c[1:])}"
    return {"base": "기본(기타·드럼·베이스·보컬)", "full": "전체 믹스", "guitar_alone": "기타만",
            "base_null": "기본 + 들리지 않는 잡음(null)"}.get(c, c)


def _f(x: Any, nd: int = 1, *, signed: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(v):
        return "–"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _ci(d: Mapping[str, Any] | None, nd: int = 1) -> str:
    if not d:
        return "–"
    lo, hi = (d.get("ci") or [None, None])[:2]
    pt = d.get("delta", d.get("value"))
    signed = "delta" in d
    if lo is None or hi is None or not all(math.isfinite(float(v)) for v in (lo, hi)):
        return f"{_f(pt, nd, signed=signed)} (CI 없음)"
    return f"{_f(pt, nd, signed=signed)} [{_f(lo, nd, signed=signed)}, {_f(hi, nd, signed=signed)}]"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return "\n".join(out)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def _top_causes(causes: Mapping[str, Mapping[str, Any]], prefix: str, n: int = 3) -> str:
    items = [(k[len(prefix):], v["delta"]) for k, v in causes.items()
             if k.startswith(prefix) and v.get("delta") is not None and math.isfinite(float(v["delta"]))]
    items = [it for it in items if it[1] > 0.05]
    items.sort(key=lambda kv: -kv[1])
    return ", ".join(f"{cause_ko(k)} {_f(v, 1, signed=True)}" for k, v in items[:n]) or "–"


_GUITAR_FIRST = ("guitar_timing", "guitar_octave", "guitar_near", "guitar_unref")


def cause_vs_dominant_rows(cvd: Mapping[str, Any]) -> list[tuple[str, str, str, dict[str, int]]]:
    """(arm, condition, cause, {dominant sound: count}) rows of ``summary["fp_cause_vs_dominant"]``.

    Current layout ``{arm: {condition: {cause: {dominant: n}}}}``; summaries written before 2026-10-05 hold one
    table pooled over every condition of the production arm (``{cause: {dominant: n}}``), shown as arm
    ``production``, condition ``*``."""
    def is_counts(d: Any) -> bool:
        return isinstance(d, Mapping) and all(isinstance(v, (int, float)) for v in d.values())

    if cvd and all(is_counts(v) for v in cvd.values()):  # the old pooled layout
        cvd = {"production": {"*": dict(cvd)}}
    from bandscribe.eval.distractors import cond_order

    out = []
    for arm in sorted(cvd, key=lambda a: (a != "production", a)):
        for cond in sorted(cvd[arm], key=cond_order):
            by = cvd[arm][cond] or {}
            for cause in sorted(by, key=lambda k: (k not in _GUITAR_FIRST, _GUITAR_FIRST.index(k)
                                                   if k in _GUITAR_FIRST else 0, k)):
                if any(by[cause].values()):
                    out.append((arm, cond, cause, dict(by[cause])))
    return out


def sections(s: Mapping[str, Any]) -> list[tuple[str, str, list[str], list[list[str]]]]:
    """(title, intro, headers, rows) per report table."""
    out: list[tuple[str, str, list[str], list[list[str]]]] = []
    arm = "production"
    deltas = (s.get("deltas") or {}).get(arm) or {}
    leak = s.get("leakage") or {}

    rows = []
    for r in s.get("ranking") or []:
        if r["kind"] == "category":
            what = f"{T.label_ko(r['source'])} 추가"
            how = (f"FP {_ci(r['fp'])}/분, FN {_ci(r['fn'])}/분 · 구간 {r['n_items']}개({r['n_groups']}곡)"
                   + (f" · 기타만 마스크로는 {_f(r['guitar_only_errors_per_min'], 1, signed=True)}/분"
                      if r.get("guitar_only_errors_per_min") is not None else ""))
        elif r["kind"] == "null":
            what = "디코딩 불안정(들리지 않는 잡음만으로 바뀌는 오류)"
            wst = r.get("worst") or {}
            how = (f"구간 평균 (ΔFP 절댓값 + ΔFN 절댓값)/분; 가장 큰 구간 "
                   f"{str(wst.get('item', '')).split(':')[1:2][0] if wst else '–'} "
                   f"FP {_f(wst.get('d_fp_per_min'), 1, signed=True)}, FN {_f(wst.get('d_fn_per_min'), 1, signed=True)}/분")
        else:
            what = cause_ko(r["source"])
            how = "전체 믹스에서 " + ("FP" if r["kind"] == "fp_cause" else "FN")
        rows.append([what, _f(r["errors_per_min"], 1), _f(r["weighted"], 1), how, r.get("hint", "")])
    out.append(("먼저 고칠 것", "오류/분이 큰 순서. 범주 행은 그 소리를 더했을 때 늘어난 FP+FN/분(점추정)에 그 소리가 "
                "있는 구간 비율을 곱한 값으로 정렬한다. 원인 행은 전체 믹스에 남은 그 원인의 오류/분이다. CI 가 0 을 "
                "포함하는 범주는 순위를 믿지 말 것.", ["오류 원천", "오류/분", "정렬 값", "근거", "고칠 곳 후보"], rows))

    rows = []
    for cond, d in deltas.items():
        cat = cond[1:] if cond.startswith("+") else cond
        lk = leak.get(cat) or {}
        rows.append([cond_ko(cond), f"{d['n_items']}({d['n_groups']}곡)", _ci(d["fp_per_min"]), _ci(d["fn_per_min"]),
                     _ci(d.get("fp_per_active_min")), _ci(d.get("fn_per_active_min")),
                     _f(d.get("active_min"), 1) if d.get("active_min") is not None else "–",
                     _ci(d["recall"]), _ci(d["precision"]), _ci(d["f1"]),
                     _ci((((s.get("deltas") or {}).get("guitar_only") or {}).get(cond) or {}).get("f1")),
                     _f(lk.get("own_leak_db"), 1) if cond.startswith("+") else "–",
                     _f(100 * lk["own_share"], 1) if cond.startswith("+") and lk.get("own_share") is not None else "–",
                     _top_causes(d.get("causes") or {}, "fp:")])
    out.append(("범주별 효과 (기본 대비, 제품 경로)",
                "+범주 조건 − 기본 조건. 괄호는 곡 블록 부트스트랩 95 % CI(10,000회). 재현율·정밀도·F1 은 점(×100). "
                "/분 = 구간 1분당, /활성분 = 그 소리가 실제로 나는 1분당(참 스템 기준, 띄엄띄엄 나오는 소리를 묽히지 않는 값). "
                "누출 = 그 소리가 지배하는 시간-주파수 칸에서 SW 기타 스템이 가져간 에너지 비율(dB, 0 dB = 전부 통과). "
                "기타 스템 비중 = 분리된 기타 스템 에너지 중 그 소리가 지배하는 칸의 몫(%).",
                ["조건", "구간(곡)", "ΔFP/분", "ΔFN/분", "ΔFP/활성분", "ΔFN/활성분", "활성 분", "Δ재현율", "Δ정밀도",
                 "ΔF1", "ΔF1 기타만 마스크", "누출 dB", "기타 스템 비중 %", "늘어난 FP 원인(/분)"], rows))

    nul = s.get("null") or {}
    if nul.get("windows"):
        med = nul.get("median_abs") or {}
        rows = [[w["item"].split(":")[1], _f(w["d_fp_per_min"], 1, signed=True), _f(w["d_fn_per_min"], 1, signed=True),
                 _f(w["d_f1"], 1, signed=True)] for w in nul["windows"]]
        rows.append(["중앙값(절댓값)", _f(med.get("d_fp_per_min"), 1), _f(med.get("d_fn_per_min"), 1),
                     _f(med.get("d_f1"), 1)])
        out.append(("잡음 기준 (null): 들리지 않는 변화만으로 생기는 차이",
                    f"기본 조건의 SW 기타 뷰에 {nul.get('noise_rms_dbfs')} dBFS RMS 백색 잡음만 더해 다시 전사한 값 − 기본. "
                    "MuScriptor 디코딩이 입력의 아주 작은 변화에 얼마나 흔들리는지 보여 준다. 범주별 차이가 이 크기 안에 있으면 "
                    "그 소리의 효과라고 말할 수 없다. (첫 전체 실행 결과를 본 뒤 더한 진단: 사전 등록 밖)",
                    ["곡", "ΔFP/분", "ΔFN/분", "ΔF1(점)"], rows))

    rows = []
    for a in s.get("arms") or []:
        for cond, d in ((s.get("conditions") or {}).get(a) or {}).items():
            if not d:
                continue
            rows.append([a, cond_ko(cond), f"{d['n_items']}", _ci(d["precision"]), _ci(d["recall"]), _ci(d["f1"]),
                         _ci(d["fp_per_min"]), _ci(d["fn_per_min"]), f"{int(d['tp'])}/{int(d['fp'])}/{int(d['fn'])}"])
    out.append(("조건별 전체 값", "모든 구간을 합친 값(풀링), 95 % CI. production = 제품 마스크, guitar_only = 기타만 마스크(진단).",
                ["경로", "조건", "구간", "정밀도", "재현율", "F1", "FP/분", "FN/분", "TP/FP/FN"], rows))

    fpm = s.get("fp_cause_matrix") or {}
    causes = sorted({k for d in fpm.values() for k in d}, key=lambda k: (k not in ("guitar_timing", "guitar_octave",
                                                                                     "guitar_near", "guitar_unref"), k))
    rows = [[cond_ko(c)] + [_f(fpm[c].get(k), 2) for k in causes] for c in fpm]
    out.append(("FP 원인 혼동표 (조건 × 원인, FP/분)", "각 오검출 음의 원인 라벨(attribution.py 규칙).",
                ["조건"] + [cause_ko(k) for k in causes], rows))

    fnm = s.get("fn_cause_matrix") or {}
    causes = sorted({k for d in fnm.values() for k in d})
    rows = [[cond_ko(c)] + [_f(fnm[c].get(k), 2) for k in causes] for c in fnm]
    out.append(("FN 원인표 (조건 × 원인, 놓친 음/분)", "놓친 기준 음마다: 추정 음과의 관계, 아니면 그 음의 대역을 가린 가장 센 "
                "비기타 소리(기타보다 6 dB 이상 약하면 '가림 없이 놓침').", ["조건"] + [cause_ko(k) for k in causes], rows))

    cvd = cause_vs_dominant_rows(s.get("fp_cause_vs_dominant") or {})
    doms = sorted({d for _a, _c, _k, row in cvd for d in row})
    rows = [[a, cond_ko(c), cause_ko(k)] + [str(row.get(d, "")) for d in doms] for a, c, k, row in cvd]
    out.append(("FP 원인 라벨 × 대역 에너지 1위 소리 (경로·조건별, 개수)",
                "원인 라벨이 '기타 자신'이어도 그 음높이 대역을 실제로 가장 크게 울린 소리가 무엇이었는지 조건과 경로마다 "
                "보여 준다. 주의: '기타 자신: 타이밍·중복'(같은 음높이 참조 음이 200 ms 안)은 대역 1위 소리를 보기 전에 "
                "붙는 라벨이다. 그래서 이 라벨의 FP 라도 1위가 방해 소리면, 기타 자신의 분할이 아니라 그 소리가 같은 "
                "음높이에서 다시 잡히게 만든 것일 수 있다(어느 쪽인지 이 측정으로는 가를 수 없다).",
                ["경로", "조건", "원인 라벨"] + [cause_ko(d) for d in doms], rows))

    rows = []
    for w in s.get("per_window") or []:
        conds = w["conditions"]
        base = (conds.get("base") or {}).get(arm) or {}
        full = (conds.get("full") or {}).get(arm) or {}
        adds = []
        for cn, cd in conds.items():
            if cn.startswith("+"):
                d = cd[arm]
                adds.append(f"{T.label_ko(cn[1:])}: F1 {_f(100 * (d['f1'] - base.get('f1', float('nan'))), 1, signed=True)}, "
                            f"FP {_f(d['fp_per_min'] - base.get('fp_per_min', float('nan')), 1, signed=True)}/분")
        masks = sorted({"+".join(m for m in cd[arm]["mask"] if "guitar" not in m) or "기타만"
                        for cd in conds.values()})
        rows.append([w["song"], f"{_f(w['start_s'], 0)}–{_f(w['end_s'], 0)} s", ", ".join(T.label_ko(c) for c in w["present"]),
                     str(w["n_ref"]), _f(100 * base.get("f1", float("nan")), 1), _f(100 * full.get("f1", float("nan")), 1),
                     _f(base.get("fp_per_min"), 1), _f(full.get("fp_per_min"), 1), _f(base.get("fn_per_min"), 1),
                     _f(full.get("fn_per_min"), 1), "; ".join(adds), " / ".join(masks)])
    out.append(("곡별", "구간마다 기본·전체 믹스와 범주를 하나씩 더한 효과(제품 경로). 마스크 = 조건별 기타 패스 마스크의 비기타 클래스.",
                ["곡", "구간", "있는 방해 소리", "참조 음", "F1 기본", "F1 전체", "FP/분 기본", "FP/분 전체",
                 "FN/분 기본", "FN/분 전체", "범주별 변화", "마스크"], rows))

    rows = []
    for w in s.get("per_window") or []:
        for cn, cd in w["conditions"].items():
            pm = [m for m in cd["production"]["mask"] if "guitar" not in m]
            if not pm:
                continue
            p, g = cd["production"], cd["guitar_only"]
            rows.append([w["song"], cond_ko(cn), "+".join(pm), str(p.get("non_guitar_dropped", "–")),
                         f"{_f(100 * g['f1'], 1)} → {_f(100 * p['f1'], 1)}",
                         f"{_f(g['fp_per_min'], 1)} → {_f(p['fp_per_min'], 1)}",
                         f"{_f(g['fn_per_min'], 1)} → {_f(p['fn_per_min'], 1)}"])
    out.append(("제품 마스크가 기타만이 아닌 조건", "편성 추정이 건반·신스 클래스를 마스크에 넣은 조건. 같은 기타 뷰를 기타만 마스크(왼쪽)와 "
                "제품 마스크(오른쪽)로 전사한 값. 빠진 음 = 비기타 클래스로 표시되어 guitar_all 에서 빠진 음 수.",
                ["곡", "조건", "추가된 클래스", "빠진 음", "F1 기타만 → 제품", "FP/분", "FN/분"], rows))

    rows = []
    for cat in T.CATEGORIES + ("guitar",):
        lk = leak.get(cat)
        if not lk:
            continue
        rows.append([T.label_ko(cat) if cat != "guitar" else "기타(유지율)", _f(lk.get("base_leak_db"), 1),
                     _f(lk.get("own_leak_db"), 1), _f(lk.get("full_leak_db"), 1),
                     _f(100 * lk["full_share"], 1) if lk.get("full_share") is not None else "–"])
    out.append(("분리 누출 (SW 기타 스템)", "그 소리가 지배하는 칸의 기타 스템/참 신호 에너지비(dB, 구간 평균). 기타 행은 유지율.",
                ["소리", "기본 조건", "+그 소리 조건", "전체 믹스", "전체 믹스에서 기타 스템 비중 %"], rows))
    return out


def _intro(s: Mapping[str, Any]) -> list[str]:
    gpu = s.get("gpu") or {}
    lines = [
        f"- 구간 {s.get('n_windows')}개, 곡 {s.get('n_songs')}개 (Cambridge-MT 멀티트랙). rung `{s.get('rung')}`.",
        "- 참조 = 참 기타 스템 합의 MuScriptor 전사(기타만 마스크). 그래서 이 수치는 믹스·분리가 더한 오류이고, "
        "전사기 자체의 절대 정확도가 아니다(참조가 MuScriptor 파생).",
        "- 조건 오디오는 모두 전체 믹스의 마스터링 게인 곡선을 똑같이 곱했다: 기타 신호는 조건마다 같다.",
        f"- GPU 시간: 분리 {_f((gpu.get('sep_s') or 0) / 60, 1)}분, MuScriptor {_f((gpu.get('ms_s') or 0) / 60, 1)}분 "
        f"(락 대기 {_f((gpu.get('waited_s') or 0) / 60, 1)}분 포함, 캐시된 것은 0).",
    ]
    if s.get("skipped"):
        lines.append("- 뺀 곡: " + ", ".join(f"{k}({v})" for k, v in sorted(s["skipped"].items())))
    if s.get("unknown_stem_names"):
        lines.append("- 분류 규칙이 모르는 스템 이름(미분류로 처리): " + ", ".join(
            f"{k}: {', '.join(v)}" for k, v in sorted(s["unknown_stem_names"].items())))
    return lines


def write(run_dir: Path, summary: Mapping[str, Any]) -> tuple[Path, Path]:
    secs = sections(summary)
    md = ["# 방해 소리별 기타 전사 오류 (distractors)", "", *_intro(summary), ""]
    for title, intro, headers, rows in secs:
        md += [f"## {title}", "", intro, "", _md_table(headers, rows) if rows else "(행 없음)", ""]
    p_md = Path(run_dir) / "report.md"
    atomic.write_text(p_md, "\n".join(md) + "\n")
    body = ["<h1>방해 소리별 기타 전사 오류 (distractors)</h1>", "<ul>"]
    body += [f"<li>{html.escape(x[2:])}</li>" for x in _intro(summary)]
    body.append("</ul>")
    for title, intro, headers, rows in secs:
        body += [f"<h2>{html.escape(title)}</h2>", f"<p>{html.escape(intro)}</p>",
                 _html_table(headers, rows) if rows else "<p>(행 없음)</p>"]
    doc = ("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><title>distractors</title><style>"
           "body{font-family:'Malgun Gothic',sans-serif;margin:24px;max-width:1400px}"
           "table{border-collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #ccc;padding:3px 6px;"
           "font-size:13px;vertical-align:top}th{background:#f3f3f3}</style></head><body>"
           + "\n".join(body) + "</body></html>\n")
    p_html = Path(run_dir) / "report.html"
    atomic.write_text(p_html, doc)
    return p_md, p_html
