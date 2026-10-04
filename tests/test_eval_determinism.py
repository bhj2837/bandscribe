"""a9: `bandscribe eval run --suite quick --system oracle-noisy` twice -> byte-identical metrics.csv / summary.json,
with bootstrap CIs filled on the aggregate rows."""

from __future__ import annotations

import datetime as dt
import time

import pytest

from bandscribe import atomic, config
from bandscribe.eval import runs, suites

pytest.importorskip("mir_eval")


def test_quick_suite_twice_is_byte_identical(tmp_path):
    cfg = config.load_config()
    t0 = time.monotonic()
    a = suites.run_suite("quick", "oracle-noisy", cfg, out=tmp_path / "a", now=dt.datetime(2026, 9, 30, 12, 0, 0),
                         gitsha="abcdef1")
    elapsed = time.monotonic() - t0
    b = suites.run_suite("quick", "oracle-noisy", cfg, out=tmp_path / "b", now=dt.datetime(2026, 9, 30, 12, 0, 5),
                         gitsha="abcdef1")
    for name in ("metrics.csv", "summary.json", "worst_bars.csv"):
        assert atomic.sha256_file(a.run_dir / name) == atomic.sha256_file(b.run_dir / name), name
    assert elapsed < 60, f"quick suite took {elapsed:.1f} s"
    rows = runs.read_metrics_csv(a.run_dir / "metrics.csv")
    agg = [r for r in rows if r["item"] == "*"]
    assert agg and all(r["ci_low"] != "" and r["ci_high"] != "" for r in agg if r["metric"] == "note_f1_onset50")
    assert "rung" in rows[0]
    run_json = atomic.read_json(a.run_dir / "run.json")
    assert run_json["started_utc"] and "started_utc" not in (a.run_dir / "summary.json").read_text(encoding="utf-8")
    assert a.run_dir.name == "20260930-120000_abcdef1_oracle-noisy"
    assert (a.run_dir / "report.html").is_file()


def test_oracle_scores_one(tmp_path):
    res = suites.run_suite("quick", "oracle", config.load_config(), out=tmp_path, gitsha="abcdef1")
    agg = res.summary["aggregates"]
    for m in ("note_f1_onset50", "note_f1_offset50", "tick_f1", "assign_macro"):
        assert agg[m]["value"] == pytest.approx(1.0), m
    assert agg["unassigned_ratio"]["value"] == pytest.approx(0.0)
    assert res.summary["baselines"]["majority_macro"]["value"] < 1.0
