"""report.html renders; the byte-stable files carry no timestamps; the rung column is present."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from gtab import config
from gtab.eval import report, runs, suites

pytest.importorskip("mir_eval")


def test_suite_report_renders_and_metric_files_have_no_timestamps(tmp_path):
    res = suites.run_suite("quick", "oracle-noisy", config.load_config(), out=tmp_path,
                           now=dt.datetime(2026, 9, 30, 9, 8, 7), gitsha="abcdef1")
    html = (res.run_dir / "report.html").read_text(encoding="utf-8")
    for needle in ("집계", "시나리오별", "파트 혼동 행렬", "coverage–accuracy", "최악 마디", "<svg"):
        assert needle in html, needle
    run_json = json.loads((res.run_dir / "run.json").read_text(encoding="utf-8"))
    stamp = run_json["started_utc"][:10]
    for name in ("metrics.csv", "summary.json", "worst_bars.csv"):
        text = (res.run_dir / name).read_text(encoding="utf-8")
        assert stamp not in text and "utc" not in text.lower(), name
    header = (res.run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(runs.METRIC_COLUMNS) and "rung" in header
    worst = (res.run_dir / "worst_bars.csv").read_text(encoding="utf-8").splitlines()
    assert worst[0] == "item,song_key,bar,missed,extra,wrong_line,total" and len(worst) > 1


def test_coverage_svg_and_experiment_report(tmp_path):
    svg = report.coverage_svg([(0.0, 1.0, 0.8), (0.5, 0.7, 0.9), (0.9, 0.3, 0.97)])
    assert svg.startswith("<svg") and "Date" not in svg[:500]
    assert report.coverage_svg([(0.0, 1.0, 0.8)]) == ""
    summary = {"experiment": "E1", "decision_ko": "판단 보류 (기본값 off 유지)", "rule": "superiority",
               "metric": "onset_f1_50", "arms": ["off", "-14"], "warnings": [], "baseline": "off",
               "comparisons": {"-14": {"delta": 0.4, "ci": [-0.3, 1.1], "baseline": {"point": 70.0},
                                       "candidate": {"point": 70.4}, "n_items": 4}}, "per_candidate": {"-14": "hold"}}
    out = report.write_experiment_report(tmp_path, summary, [{"system": "off", "item": "*", "metric": "onset_f1_50",
                                                              "value": 0.7, "ci_low": 0.6, "ci_high": 0.8, "n": 4}])
    html = out.read_text(encoding="utf-8")
    assert "판단 보류" in html and "-14" in html
