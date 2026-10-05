"""Pre-registered experiment runner (``bandscribe eval exp <id>``, spec §9.4 item 8, DESIGN §8.1/§8.6).

``docs/decisions.md`` is the registry: each ``## <id>`` heading is a CLI id; ``- 상태:`` is the state machine
(``사전 등록`` → ``실행 중`` → ``결정: …``); ``- 판정 설정:`` is the machine-readable decision spec
(``rule=…; metric=…; baseline=…; mde=…``). The runner:

1. refuses ids that are missing or already decided (a decided experiment needs a new pre-registered id);
2. flips ``사전 등록`` to ``실행 중`` (reruns stay allowed while undecided);
3. calls ``EXPERIMENTS[id](out_dir, cfg, args) -> list[dict]`` (metrics.csv rows with ``rung`` filled) from
   ``bandscribe.sep.experiments`` (``stereo-preservation``), ``bandscribe.tab.experiments`` (``E12``) or
   ``bandscribe.amt.experiments`` (all others);
4. computes the decision with ``bandscribe.eval.stats`` against the pre-registration, writes the run dir and appends a
   result line to the entry. The final ``결정:`` status is set by a person (``--record``), not silently.

Row conventions (documented in decisions.md): ``system`` = arm, ``item``/``group`` = item and bootstrap block,
``scenario`` = subset tag, ``section`` = split (dev/test) for tuning / latency experiments, ``line`` = part.
Rows with different ``rung`` values are never pooled: an arm whose rows mix rungs makes the decision ``hold``.
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bandscribe import atomic, paths
from bandscribe.eval import stats
from bandscribe.eval.runs import METRIC_COLUMNS

log = logging.getLogger(__name__)

SEP_IDS = frozenset({"stereo-preservation"})
AMT_IDS = frozenset({"E1", "E2", "E3", "E3b", "E3c", "E23", "E24-sep", "E24-amt", "latency", "gate-m2"})
TAB_IDS = frozenset({"E12"})
ALL_IDS = tuple(sorted(SEP_IDS | AMT_IDS | TAB_IDS))

STATE_PRE = "사전 등록"
STATE_RUNNING = "실행 중"
STATE_DECIDED = "결정:"
DECISION_KO = {"adopt": "채택", "reject": "기각", "hold": "판단 보류"}


def decisions_path() -> Path:
    return paths.ROOT / "docs" / "decisions.md"


def contamination_path() -> Path:
    return paths.ROOT / "docs" / "eval" / "contamination.md"


class ExperimentStateError(RuntimeError):
    """The registry does not allow this run (Korean, user-facing)."""


class ExperimentRunError(RuntimeError):
    """The experiment function failed before producing results; ``data_missing`` = the pre-registered data
    is not on this PC (the honest record is then ``판단 보류 (데이터 대기)``)."""

    def __init__(self, exp_id: str, cause: BaseException, run_dir: Path | None = None) -> None:
        self.exp_id = exp_id
        self.cause = cause
        self.run_dir = run_dir
        # AMT's ExperimentDataMissing, DATA's DatasetUnavailable: matched by name (no import of either module)
        self.data_missing = type(cause).__name__ in ("ExperimentDataMissing", "DatasetUnavailable")
        super().__init__(f"{exp_id}: {cause}")


def _leave_failure_note(run_dir: Path, exp_id: str, err: BaseException) -> None:
    """A failed run keeps its folder (worker logs live there) plus a short note saying why it has no results."""
    try:
        atomic.write_json(Path(run_dir) / "failed.json", {"experiment": exp_id, "error_type": type(err).__name__,
                                                          "error": str(err)})
    except OSError:
        log.warning("could not write failed.json in %s", run_dir)


# ------------------------------------------------------------------------------------------ registry


@dataclass
class Entry:
    id: str
    status: str
    fields: dict[str, str]
    start: int                  # line index of the heading
    end: int                    # line index after the entry
    full: dict[str, str] = field(default_factory=dict)  # field text incl. indented continuation lines (display)

    @property
    def spec(self) -> dict[str, str]:
        return parse_spec(self.fields.get("판정 설정", ""))

    @property
    def decided(self) -> bool:
        return self.status.startswith(STATE_DECIDED)


_FIELD = re.compile(r"^- ([^:：]+)[:：]\s?(.*)$")


def load_registry(path: Path | None = None) -> dict[str, Entry]:
    text = Path(path or decisions_path()).read_text(encoding="utf-8")
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    out: dict[str, Entry] = {}
    in_code = False
    code_lines: set[int] = set()
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_code = not in_code
        if in_code:
            code_lines.add(i)
    heads = [h for h in heads if h not in code_lines]
    for k, h in enumerate(heads):
        end = heads[k + 1] if k + 1 < len(heads) else len(lines)
        eid = lines[h][3:].strip().split()[0] if lines[h][3:].strip() else ""
        fields: dict[str, str] = {}
        full: dict[str, str] = {}
        cur: str | None = None
        for i in range(h + 1, end):
            if i in code_lines:  # the template block shows field lines too
                cur = None
                continue
            m = _FIELD.match(lines[i])
            if m:
                cur = m.group(1).strip()
                if cur in fields:  # a repeated field keeps its first value
                    cur = None
                    continue
                fields[cur] = m.group(2).strip()
                full[cur] = fields[cur]
            elif cur is not None and lines[i].startswith("  ") and lines[i].strip():
                full[cur] = (full[cur] + " " + lines[i].strip()).strip()
            else:
                cur = None
        if eid and "상태" in fields:
            out[eid] = Entry(eid, fields["상태"], fields, h, end, full)
    return out


def parse_spec(text: str) -> dict[str, str]:
    """``rule=superiority; metric=onset_f1_50; baseline=off; mde=1.0`` (backticks optional) -> dict."""
    t = text.strip().strip("`").strip()
    out: dict[str, str] = {}
    for part in t.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def check_runnable(registry: Mapping[str, Entry], exp_id: str) -> Entry:
    e = registry.get(exp_id)
    if e is None:
        raise ExperimentStateError(
            f"'{exp_id}' 는 docs/decisions.md 에 사전 등록되어 있지 않습니다. 실험 전에 주 지표·MDE·데이터·분석·결정 규칙을 "
            f"등록하세요 (등록된 id: {', '.join(sorted(registry)) or '없음'}).")
    if e.decided:
        raise ExperimentStateError(
            f"'{exp_id}' 는 이미 결정되었습니다 ({e.status}). 결정된 실험을 다시 보려면 새 id로 사전 등록하세요.")
    if e.status not in (STATE_PRE, STATE_RUNNING):
        raise ExperimentStateError(f"'{exp_id}' 의 상태 '{e.status}' 를 알 수 없습니다 ({STATE_PRE} / {STATE_RUNNING} / 결정: …).")
    return e


def _rewrite(path: Path, fn: Callable[[list[str]], list[str]]) -> None:
    raw = Path(path).read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = fn(raw.splitlines())
    atomic.write_text(Path(path), nl.join(lines) + (nl if raw.endswith(("\n", "\r\n")) else ""))


def set_status(exp_id: str, status: str, path: Path | None = None) -> None:
    p = Path(path or decisions_path())
    e = load_registry(p)[exp_id]

    def fn(lines: list[str]) -> list[str]:
        for i in range(e.start + 1, e.end):
            if _FIELD.match(lines[i]) and _FIELD.match(lines[i]).group(1).strip() == "상태":
                lines[i] = f"- 상태: {status}"
                break
        return lines

    _rewrite(p, fn)


def append_result(exp_id: str, text: str, path: Path | None = None) -> None:
    p = Path(path or decisions_path())
    e = load_registry(p)[exp_id]

    def fn(lines: list[str]) -> list[str]:
        for i in range(e.start + 1, e.end):
            m = _FIELD.match(lines[i])
            if m and m.group(1).strip() == "결과":
                if "아직 없음" in m.group(2):
                    lines[i] = "- 결과:"
                j = i + 1
                while j < e.end and lines[j].startswith("  "):
                    j += 1
                lines.insert(j, f"  - {text}")
                return lines
        # no "결과" field: add it after the entry's last non-blank line (one list item per line, so the file
        # keeps a single line-ending style)
        j = e.end
        while j > e.start + 1 and not lines[j - 1].strip():
            j -= 1
        lines[j:j] = ["- 결과:", f"  - {text}"]
        return lines

    _rewrite(p, fn)


# -------------------------------------------------------------------------------------- contamination


def contamination_status(model: str, dataset: str, path: Path | None = None) -> str:
    """Cell of contamination.md: ``학습 포함`` | ``학습 제외`` | ``불명`` (missing row/column -> ``불명``)."""
    p = Path(path or contamination_path())
    if not p.is_file():
        return "불명"
    header: list[str] | None = None
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.startswith("|"):
            header = None if not ln.strip() else header
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        ids = [re.findall(r"`([^`]+)`", c) for c in header]
        row_id = re.findall(r"`([^`]+)`", cells[0])
        if not row_id or row_id[0] != dataset:
            continue
        for j, idl in enumerate(ids):
            if idl and idl[0] == model and j < len(cells):
                for v in ("학습 포함", "학습 제외", "불명"):
                    if cells[j].startswith(v):
                        return v
    return "불명"


# ------------------------------------------------------------------------------------------ decisions


def _is_num(v: Any) -> bool:
    try:
        return v not in (None, "") and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def items_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[stats.ItemCounts]:
    """Per-item counts: tp/fp/fn when present (pooled F1), else the value as a mean (sum/n)."""
    out = []
    for r in rows:
        item = str(r.get("item") or "")
        if item == "*":
            continue
        group = str(r.get("group") or item)
        if all(_is_num(r.get(k)) for k in ("tp", "fp", "fn")):
            out.append(stats.ItemCounts.prf(item, group, float(r["tp"]), float(r["fp"]), float(r["fn"])))
        elif _is_num(r.get("value")):
            out.append(stats.ItemCounts.mean(item, group, float(r["value"]), 1.0))
    return out


def _stat_for(items: Sequence[stats.ItemCounts]) -> Callable:
    return stats.f1_points if items and "tp" in items[0].counts else stats.mean_value


def _find_arm(arms: Iterable[str], label: str) -> str | None:
    arms = list(arms)
    if label in arms:
        return label
    cands = [a for a in arms if a.endswith(f"={label}") or a.endswith(f":{label}") or a.split("|")[0] == label]
    return cands[0] if len(cands) == 1 else None


def _by_arm(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("metric") == metric and r.get("item") != "*":
            out[str(r.get("system") or "")].append(r)
    return dict(sorted(out.items()))


def _mixed_rungs(arm_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    return [a for a, rs in arm_rows.items() if len({str(r.get("rung") or "") for r in rs}) > 1]


def _paired(base: Sequence[Mapping[str, Any]], cand: Sequence[Mapping[str, Any]], n: int, seed: int,
            subset_col: str) -> dict[str, Any]:
    ib, ic = items_from_rows(base), items_from_rows(cand)
    stat = _stat_for(ib)
    d, lo, hi = stats.paired_bootstrap(ib, ic, stat, n=n, seed=seed)
    subsets: dict[str, tuple[float, float, float]] = {}
    for s in sorted({str(r.get(subset_col) or "") for r in base} - {""}):
        sb = items_from_rows([r for r in base if str(r.get(subset_col) or "") == s])
        sc = items_from_rows([r for r in cand if str(r.get(subset_col) or "") == s])
        if sb and sc:
            subsets[s] = stats.paired_bootstrap(sb, sc, stat, n=n, seed=seed)
    pb = stats.block_bootstrap(ib, stat, n=n, seed=seed)
    pc = stats.block_bootstrap(ic, stat, n=n, seed=seed)
    return {"delta": d, "ci": [lo, hi], "baseline": {"point": pb[0], "ci": [pb[1], pb[2]]},
            "candidate": {"point": pc[0], "ci": [pc[1], pc[2]]}, "subsets": {k: list(v) for k, v in subsets.items()},
            "n_items": len(ic), "n_groups": len({i.group for i in ic})}


def _resource_ok(rows: Sequence[Mapping[str, Any]], base: str, cand: str, spec_text: str) -> tuple[bool | None, dict]:
    """``reserved_peak_mb:-20%|rtf:+20%`` -> any listed resource improves by the stated relative amount."""
    info: dict[str, Any] = {}
    ok_any: bool | None = None
    for part in [p for p in spec_text.split("|") if p]:
        name, _, pct = part.partition(":")
        target = float(pct.strip().rstrip("%")) / 100.0
        vals = {}
        for arm in (base, cand):
            v = [float(r["value"]) for r in rows if r.get("metric") == name and str(r.get("system")) == arm and _is_num(r.get("value"))]
            vals[arm] = sum(v) / len(v) if v else None
        if vals[base] in (None, 0) or vals[cand] is None:
            info[name] = {"baseline": vals[base], "candidate": vals[cand], "ok": None}
            continue
        rel = (vals[cand] - vals[base]) / abs(vals[base])
        ok = rel <= target if target < 0 else rel >= target
        info[name] = {"baseline": vals[base], "candidate": vals[cand], "relative": rel, "target": target, "ok": ok}
        ok_any = bool(ok_any) or ok
    return ok_any, info


def _violations(rows: Iterable[Mapping[str, Any]], constraint: str | None, arm: str) -> int:
    """Rows of ``arm`` breaking ``constraint`` (``<metric>==<value>``, e.g. ``guitar_class_omissions==0``)."""
    if not constraint:
        return 0
    cname, _, cval = constraint.partition("==")
    return sum(1 for r in rows if r.get("metric") == cname.strip() and str(r.get("system")) == arm
               and _is_num(r.get("value")) and float(r["value"]) != float(cval))


def excluded_arms(arms: Iterable[str], spec: Mapping[str, str], *, baseline: str | None = None) -> list[str]:
    """Arms that may be reported but never chosen: ``exclude=<prefix>[|<prefix>...]`` in 판정 설정 (e.g. E23's
    ``exclude=f12_``: the 12-frame arms were registered "비교용으로만"). The baseline always stays eligible."""
    prefixes = [p.strip() for p in str(spec.get("exclude") or "").split("|") if p.strip()]
    return sorted(a for a in arms if a != baseline and any(a.startswith(p) for p in prefixes))


def _pooled_point(rows: Sequence[Mapping[str, Any]]) -> float:
    its = items_from_rows(rows)
    return stats.block_bootstrap(its, _stat_for(its), n=0)[0] if its else float("nan")


def _select_then_confirm(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, str], *, n: int,
                         seed: int) -> dict[str, Any]:
    """Superiority with selection (E3: "dev 에서 제약 아래 F1 최대, test 에서 확인").

    The candidate is chosen on the ``select`` split (pooled point, arms breaking the constraint there are not
    eligible; ties keep the baseline), and only that one candidate is confirmed against the baseline on the
    ``split`` split with the DESIGN 8.1 rule (+ the constraint on the confirmation rows). Choosing among many
    arms by their test result would grade the selection with the data that made it.
    """
    metric = spec.get("metric", "")
    col = spec.get("split_col", "section")
    sel_split, test_split = spec["select"], spec.get("split", "test")
    subset_col = spec.get("subset_col", "scenario")
    constraint = spec.get("constraint")
    in_split = lambda rs, s: [r for r in rs if str(r.get(col) or "") == s]  # noqa: E731
    sel_arms = _by_arm(in_split(rows, sel_split), metric)
    test_arms = _by_arm(in_split(rows, test_split), metric)
    out: dict[str, Any] = {"rule": "superiority", "metric": metric, "arms": sorted(set(sel_arms) | set(test_arms)),
                           "warnings": [], "select_split": sel_split, "confirm_split": test_split}
    mixed = sorted(set(_mixed_rungs(sel_arms)) | set(_mixed_rungs(test_arms)))
    if mixed:
        out["warnings"].append(f"사다리 칸(rung)이 섞인 arm: {', '.join(mixed)} — 합치지 않고 판단 보류")
    base = _find_arm(sel_arms, spec.get("baseline", "")) or _find_arm(test_arms, spec.get("baseline", ""))
    if base is None:
        out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (기준 arm '{spec.get('baseline', '')}' 행 없음)")
        return out
    sel_rows = in_split(rows, sel_split)
    scores = {a: _pooled_point(rs) for a, rs in sel_arms.items()}
    excluded = excluded_arms(scores, spec, baseline=base)
    eligible = {a: s for a, s in scores.items() if math.isfinite(s) and _violations(sel_rows, constraint, a) == 0
                and a not in excluded}
    out.update(baseline=base, select_scores=scores, eligible=sorted(eligible), excluded=excluded)
    if not eligible:
        out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} ({sel_split} 행 없음 또는 모든 arm 이 제약 위반)")
        return out
    best = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0] != base, kv[0]))[0][0]
    out["chosen_on_select"] = best
    if best == base:
        out.update(decision="reject", decision_ko=f"기각 ({sel_split} 최적이 기본값 {base})")
        return out
    if base not in test_arms or best not in test_arms:
        out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} ({test_split} 행 없음)")
        return out
    res = _paired(test_arms[base], test_arms[best], n, seed, subset_col)
    d = stats.decide(res["delta"], res["ci"][0], res["ci"][1], float(spec.get("mde", "1.0")),
                     {k: tuple(v) for k, v in res["subsets"].items()})
    viol = _violations(in_split(rows, test_split), constraint, best)
    if viol:
        res["constraint_violations"] = viol
        if d == "adopt":
            d = "reject"
    out.update(comparisons={best: res}, per_candidate={best: d})
    if mixed:
        out.update(decision="hold", decision_ko="판단 보류 (rung 혼합)")
    elif d == "adopt":
        out.update(decision="adopt", chosen=best, decision_ko=f"채택 ({best})")
    elif d == "hold":
        out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (기본값 {base} 유지)")
    else:
        out.update(decision="reject", decision_ko=f"기각 (기본값 {base} 유지)")
    return out


def evaluate(rows: Sequence[Mapping[str, Any]], spec: Mapping[str, str], *, n: int = stats.DEFAULT_ITERATIONS,
             seed: int = stats.DEFAULT_SEED, contamination_file: Path | None = None) -> dict[str, Any]:
    """Decision summary for one experiment run according to its pre-registered ``판정 설정``."""
    rule = spec.get("rule", "descriptive")
    metric = spec.get("metric", "")
    subset_col = spec.get("subset_col", "scenario")
    if rule == "superiority" and spec.get("select"):
        return _select_then_confirm(rows, spec, n=n, seed=seed)
    if rule in ("superiority", "noninferiority") and spec.get("split"):
        # decided on the held-out split only (e.g. E3: thresholds chosen on dev, confirmed on test)
        col = spec.get("split_col", "section")
        rows = [r for r in rows if r.get("metric") != metric or str(r.get(col) or "") == spec["split"]]
    arm_rows = _by_arm(rows, metric)
    out: dict[str, Any] = {"rule": rule, "metric": metric, "arms": sorted(arm_rows), "warnings": []}
    mixed = _mixed_rungs(arm_rows)
    if mixed:
        out["warnings"].append(f"사다리 칸(rung)이 섞인 arm: {', '.join(mixed)} — 합치지 않고 판단 보류")
    if rule == "descriptive":
        split_col = spec.get("split_col")
        desc = {}
        for arm, rs in arm_rows.items():
            parts = {"all": rs} if not split_col else {s or "all": [r for r in rs if str(r.get(split_col) or "") == s]
                                                         for s in sorted({str(r.get(split_col) or "") for r in rs})}
            for part, prs in parts.items():
                its = items_from_rows(prs)
                p, lo, hi = stats.block_bootstrap(its, _stat_for(its), n=n, seed=seed)
                desc[f"{arm}|{part}"] = {"point": p, "ci": [lo, hi], "n_items": len(its)}
        out.update(decision="descriptive", decision_ko="서술(판정 없음)", values=desc)
        return out
    base_label = spec.get("baseline", "")
    base = _find_arm(arm_rows, base_label)
    if base is None:
        out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (기준 arm '{base_label}' 행 없음)")
        return out
    if rule == "gate":
        cand = _find_arm(arm_rows, spec.get("candidate", ""))
        subset = spec.get("subset")
        if cand is None:
            out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (후보 arm 행 없음)")
            return out
        sel = (lambda rs: [r for r in rs if str(r.get(subset_col) or "") == subset]) if subset else (lambda rs: list(rs))
        res = _paired(sel(arm_rows[base]), sel(arm_rows[cand]), n, seed, subset_col="__none__")
        thr = float(spec.get("threshold", "5"))
        passed = res["delta"] >= thr
        out.update(comparison={cand: res}, threshold=thr, passed=passed)
        model, _, dataset = spec.get("contamination", "").partition(":")
        cstat = contamination_status(model, dataset, contamination_file) if model else "학습 제외"
        out["contamination"] = {"model": model, "dataset": dataset, "status": cstat}
        if cstat == "학습 포함":
            out.update(decision="hold", decision_ko="판단 보류 (오염)")
        elif cstat == "불명":
            out.update(decision="hold", decision_ko="판단 보류 (오염 불명)")
        elif mixed:
            out.update(decision="hold", decision_ko="판단 보류 (rung 혼합)")
        else:
            out.update(decision="adopt" if passed else "reject",
                       decision_ko="통과" if passed else "주 전사기 재검토")
        return out
    if rule == "tuning":
        split_col = spec.get("split_col", "section")
        dev = {a: [r for r in rs if str(r.get(split_col) or "") == "dev"] for a, rs in arm_rows.items()}
        test = {a: [r for r in rs if str(r.get(split_col) or "") == "test"] for a, rs in arm_rows.items()}
        scores = {}
        for a, rs in dev.items():
            its = items_from_rows(rs)
            if its:
                scores[a] = float(_stat_for(its)({k: sum(i.counts.get(k, 0.0) for i in its) for k in its[0].counts}))
        if not scores:
            out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (dev 행 없음)")
            return out
        # ``exclude=``: arms registered as comparison-only are scored and reported but never picked
        excluded = excluded_arms(scores, spec, baseline=base)
        eligible = {a: s for a, s in scores.items() if a not in excluded}
        if not eligible:
            out.update(dev_scores=scores, excluded=excluded, decision="hold",
                       decision_ko=f"{DECISION_KO['hold']} (선택 가능한 arm 없음)")
            return out
        best = sorted(eligible.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        res = _paired(test.get(base, []), test.get(best, []), n, seed, subset_col) if best != base else None
        out.update(dev_scores=scores, excluded=excluded, tuned=best, test=res)
        if best == base:
            out.update(decision="reject", decision_ko="기각 (dev 최적이 기본값)")
        elif res is None or not math.isfinite(res["delta"]):
            out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (test 행 없음)")
        elif mixed:
            out.update(decision="hold", decision_ko="판단 보류 (rung 혼합)")
        else:
            ok = res["delta"] >= 0
            out.update(decision="adopt" if ok else "reject",
                       decision_ko=f"{'채택' if ok else '기각'} (튜닝값 {best}, test Δ {res['delta']:+.2f}점)")
        return out
    comps = {}
    decisions = {}
    for cand in [a for a in arm_rows if a != base]:
        res = _paired(arm_rows[base], arm_rows[cand], n, seed, subset_col)
        comps[cand] = res
        lo, hi = res["ci"]
        if rule == "noninferiority":
            margin = float(spec.get("margin", "-0.5"))
            ni = stats.noninferior(lo, margin)
            r_ok, r_info = _resource_ok(rows, base, cand, spec.get("resource", ""))
            res["noninferior"] = ni
            res["resource"] = r_info
            decisions[cand] = "adopt" if ni and r_ok else ("hold" if ni and r_ok is None else "reject")
        else:
            d = stats.decide(res["delta"], lo, hi, float(spec.get("mde", "1.0")),
                             {k: tuple(v) for k, v in res["subsets"].items()})
            viol = _violations(rows, spec.get("constraint"), cand)
            if d == "adopt" and viol:
                d = "reject"
                res["constraint_violations"] = viol
            decisions[cand] = d
    out.update(baseline=base, comparisons=comps, per_candidate=decisions)
    if mixed:
        out.update(decision="hold", decision_ko="판단 보류 (rung 혼합)")
        return out
    adopted = [c for c, d in decisions.items() if d == "adopt"]
    if adopted:
        best = max(adopted, key=lambda c: (comps[c]["delta"], c))
        out.update(decision="adopt", chosen=best, decision_ko=f"채택 ({best})")
    elif any(d == "hold" for d in decisions.values()) or not decisions:
        out.update(decision="hold", decision_ko=f"{DECISION_KO['hold']} (기본값 {base} 유지)")
    else:
        out.update(decision="reject", decision_ko=f"기각 (기본값 {base} 유지)")
    return out


# ------------------------------------------------------------------------------------------- running


def experiment_fn(exp_id: str) -> Callable[[Path, Any, dict], list[dict]]:
    mod_name = ("bandscribe.sep.experiments" if exp_id in SEP_IDS else
                "bandscribe.tab.experiments" if exp_id in TAB_IDS else "bandscribe.amt.experiments")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise ExperimentStateError(f"실험 모듈 {mod_name} 을(를) 불러오지 못했습니다: {e}") from e
    table = getattr(mod, "EXPERIMENTS", {})
    if exp_id not in table:
        raise ExperimentStateError(f"{mod_name}.EXPERIMENTS 에 '{exp_id}' 가 아직 구현되지 않았습니다.")
    return table[exp_id]


@dataclass
class ExpRun:
    run_dir: Path
    summary: dict[str, Any]
    rows: list[dict[str, Any]] = field(repr=False, default_factory=list)


def aggregate_rows(rows: Sequence[Mapping[str, Any]], suite: str, *, n: int, seed: int) -> list[dict[str, Any]]:
    """Aggregate rows (item "*", CI from the block bootstrap) per (system, dataset, metric, line, rung)."""
    groups: dict[tuple, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("item") == "*":
            continue
        groups[(str(r.get("system") or ""), str(r.get("dataset") or ""), str(r["metric"]), str(r.get("line") or ""),
                str(r.get("rung") or ""), str(r.get("section") or ""))].append(r)
    out = []
    for (system, dataset, metric, line, rung, section), rs in sorted(groups.items()):
        its = items_from_rows(rs)
        if not its:
            continue
        stat = _stat_for(its)
        p, lo, hi = stats.block_bootstrap(its, stat, n=n, seed=seed)
        row = {"suite": suite, "system": system, "dataset": dataset, "item": "*", "metric": metric, "line": line,
               "rung": rung, "section": section, "value": p / 100.0 if stat is stats.f1_points else p,
               "ci_low": lo / 100.0 if stat is stats.f1_points else lo, "ci_high": hi / 100.0 if stat is stats.f1_points else hi,
               "n": len(its)}
        if "tp" in its[0].counts:
            row.update(tp=sum(i.counts["tp"] for i in its), fp=sum(i.counts["fp"] for i in its),
                       fn=sum(i.counts["fn"] for i in its))
        out.append(row)
    return out


def run_experiment(exp_id: str, cfg: Any, *, args: dict | None = None, dry_run: bool = False,
                   registry_path: Path | None = None, runs_root: Path | None = None,
                   now: dt.datetime | None = None, gitsha: str | None = None) -> ExpRun | Entry:
    """Run a pre-registered experiment; returns the Entry on ``dry_run``."""
    from bandscribe.eval import report, runs

    reg_path = Path(registry_path or decisions_path())
    entry = check_runnable(load_registry(reg_path), exp_id)
    if entry.spec.get("suite"):
        raise ExperimentStateError(f"'{exp_id}' 는 평가 세트로 실행합니다: bandscribe eval run --suite {entry.spec['suite']}")
    fn = experiment_fn(exp_id)
    if dry_run:
        return entry
    n = int(cfg.get("eval.bootstrap_iterations", stats.DEFAULT_ITERATIONS)) if cfg is not None else stats.DEFAULT_ITERATIONS
    seed = int(cfg.get("eval.seed", stats.DEFAULT_SEED)) if cfg is not None else stats.DEFAULT_SEED
    run_dir = runs.new_run_dir(f"exp-{exp_id}", root=runs_root, now=now, gitsha=gitsha)
    started = runs.utc_now()
    try:
        rows = [dict(r) for r in fn(run_dir, cfg, dict(args or {}))]
    except Exception as e:
        # The experiment did not produce results (data not on this PC, model missing, VRAM, worker failure):
        # the registry stays as it was, so "실행 중" only ever means "has at least one computed result".
        _leave_failure_note(run_dir, exp_id, e)
        raise ExperimentRunError(exp_id, e, run_dir) from e
    if not rows:
        _leave_failure_note(run_dir, exp_id, RuntimeError("no metric rows"))
        raise ExperimentRunError(exp_id, RuntimeError("실험이 지표 행을 하나도 내지 않았습니다"), run_dir)
    if entry.status == STATE_PRE:
        set_status(exp_id, STATE_RUNNING, reg_path)
    suite = f"exp:{exp_id}"
    for r in rows:
        r.setdefault("suite", suite)
        unknown = set(r) - set(METRIC_COLUMNS)
        if unknown:
            raise ValueError(f"{exp_id}: unknown metrics.csv columns {sorted(unknown)}")
    missing_rung = [r for r in rows if r.get("rung") in (None, "") and str(r.get("metric", "")).endswith(("f1_50", "_f1"))]
    summary = evaluate(rows, entry.spec, n=n, seed=seed)
    if missing_rung:
        summary["warnings"].append(f"rung 열이 빈 전사 지표 행 {len(missing_rung)}개 (실험은 rung을 채워야 합니다)")
    summary = {"experiment": exp_id, "status_before": entry.status, "spec": entry.spec, **summary}
    all_rows = rows + aggregate_rows(rows, suite, n=n, seed=seed)
    runs.write_metrics_csv(run_dir / "metrics.csv", all_rows)
    runs.write_summary(run_dir / "summary.json", summary)
    runs.write_run_json(run_dir / "run.json", {
        "suite": suite, "system": f"exp-{exp_id}", "experiment": exp_id, "args": args or {},
        "git_sha": run_dir.name.split("_")[1], "started_utc": started, "finished_utc": runs.utc_now(),
        "config": cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else None, **runs.provenance()})
    report.write_experiment_report(run_dir, summary, all_rows)
    append_result(exp_id, f"{(now or dt.datetime.now()).strftime('%Y-%m-%d')} `{run_dir.name}`: "
                          f"계산된 판정 = {summary.get('decision_ko')}", reg_path)
    return ExpRun(run_dir, summary, all_rows)


def record_decision(exp_id: str, decision_ko: str, *, registry_path: Path | None = None) -> None:
    """Set ``결정: …`` after a person has checked the run (``bandscribe eval exp <id> --record "채택 (…)"``)."""
    reg_path = Path(registry_path or decisions_path())
    entry = check_runnable(load_registry(reg_path), exp_id)
    del entry
    set_status(exp_id, f"{STATE_DECIDED} {decision_ko}", reg_path)
