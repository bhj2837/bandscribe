"""report.html for suite runs and experiment runs (jinja2 + inline matplotlib SVG).

The report is for people; the byte-stable artefacts are metrics.csv and summary.json. jinja2 and matplotlib are
imported inside the functions (spec §0 lazy imports); matplotlib runs on the Agg backend (MPLBACKEND=Agg is also
set in the managed env).
"""

from __future__ import annotations

import io
import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gtab import atomic

log = logging.getLogger(__name__)

TEMPLATES = Path(__file__).with_name("templates")
ITEM_METRICS = ("note_f1_onset50", "note_f1_onset100", "note_f1_offset50", "tick_f1", "octave_error_rate",
                "assign_macro", "assign_weighted", "unassigned_ratio", "shared_f1")
SCENARIO_METRICS = ("note_f1_onset50", "tick_f1", "assign_macro", "unassigned_ratio")


def fmt(v: Any) -> str:
    if v is None or v == "":
        return "–"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "–"
    return f"{f:.3f}" if abs(f) < 10 else f"{f:.1f}"


def _env():
    import jinja2

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), autoescape=jinja2.select_autoescape(["html", "j2"]),
                             undefined=jinja2.ChainableUndefined, keep_trailing_newline=True)
    env.globals["fmt"] = fmt
    return env


def coverage_svg(points: Sequence[tuple[float, float, float]]) -> str:
    """Coverage–accuracy curve as an inline SVG (no timestamp metadata)."""
    pts = [(t, c, a) for t, c, a in points if math.isfinite(c) and math.isfinite(a)]
    if len(pts) < 2:
        return ""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.4), dpi=100)
    try:
        ax.plot([c for _t, c, _a in pts], [a for _t, _c, a in pts], marker="o", color="#2563eb")
        for t, c, a in pts:
            ax.annotate(f"τ={t:g}", (c, a), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel("coverage (assigned share)")
        ax.set_ylabel("accuracy | assigned")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", metadata={"Date": None})
    finally:
        plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


def _pivot(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("item") == "*" or r.get("line"):
            continue
        d = by.setdefault(str(r["item"]), {"item": r["item"], "scenario": r.get("scenario", ""), "vals": {}})
        d["vals"][r["metric"]] = r.get("value")
    return [by[k] for k in sorted(by)]


def _by_scenario(items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        for m in SCENARIO_METRICS:
            v = it["vals"].get(m)
            if v not in (None, "") and math.isfinite(float(v)):
                acc[it["scenario"]][m].append(float(v))
        acc[it["scenario"]]["_n"].append(1.0)
    out = {}
    for sc in sorted(acc):
        d = {m: (sum(v) / len(v) if v else None) for m, v in acc[sc].items() if m != "_n"}
        d["n"] = len(acc[sc]["_n"])
        out[sc] = d
    return out


def write_suite_report(run_dir: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                       details: Sequence[Mapping[str, Any]], worst: Sequence[Mapping[str, Any]], *,
                       max_worst: int = 40, max_confusions: int = 30) -> Path:
    from gtab.eval.suites import coverage_points

    items = _pivot(rows)
    html = _env().get_template("suite_report.html.j2").render(
        summary=summary, run_name=Path(run_dir).name, items=items, item_metrics=ITEM_METRICS,
        by_scenario=_by_scenario(items), scenario_metrics=SCENARIO_METRICS,
        line_rows=[r for r in rows if r.get("metric") == "line_f1_onset50" and r.get("item") != "*"],
        details=list(details), max_confusions=max_confusions, coverage_svg=coverage_svg(coverage_points(details)),
        worst=sorted(worst, key=lambda r: (-r["total"], r["item"], r["bar"]))[:max_worst])
    out = Path(run_dir) / "report.html"
    atomic.write_text(out, html)
    return out


def write_experiment_report(run_dir: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Path:
    from gtab.eval.runs import round_floats

    html = _env().get_template("experiment_report.html.j2").render(
        summary=summary, agg_rows=[r for r in rows if r.get("item") == "*"],
        summary_json=json.dumps(round_floats(summary), ensure_ascii=False, indent=1))
    out = Path(run_dir) / "report.html"
    atomic.write_text(out, html)
    return out
