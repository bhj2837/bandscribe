"""slug(), run dirs, metrics.csv format, the pre-registration state machine and the decision engine."""

from __future__ import annotations

import datetime as dt
import shutil
import sys
import types

import numpy as np
import pytest

from gtab import paths
from gtab.eval import experiments, runs

# ------------------------------------------------------------------------------------------------ slug


@pytest.mark.parametrize(("raw", "expected"), [
    ("job:amt_gtr", "job-amt_gtr"),
    ("guitarset:00_BN1", "guitarset-00_bn1"),
    ("guitarset:00_BN1-129-Eb_comp", "guitarset-00_bn1-129-eb_comp"),
    ("external:klangio", "external-klangio"),
    ("synth:A_s0_lead-6db", "synth-a_s0_lead-6db"),
    ("a  b//c", "a_b_c"),
    ("x::y", "x-y"),
])
def test_slug(raw, expected):
    assert runs.slug(raw) == expected


@pytest.mark.parametrize("raw", ["", "기타", "___", ":"])
def test_slug_rejects_empty(raw):
    with pytest.raises(ValueError):
        runs.slug(raw)


def test_run_dir_name_and_no_colon(tmp_path):
    d = runs.new_run_dir("job:amt_gtr", root=tmp_path, now=dt.datetime(2026, 9, 30, 1, 2, 3), gitsha="1234567abc")
    assert d.name == "20260930-010203_1234567_job-amt_gtr"
    d2 = runs.new_run_dir("job:amt_gtr", root=tmp_path, now=dt.datetime(2026, 9, 30, 1, 2, 3), gitsha="1234567")
    assert d2.name.endswith("-2")
    assert all(":" not in p.name for p in tmp_path.rglob("*"))


def test_metrics_csv_is_sorted_and_stable(tmp_path):
    rows = [{"suite": "s", "item": "b", "metric": "m", "value": 0.1 + 0.2},
            {"suite": "s", "item": "a", "metric": "m", "value": -0.0, "tp": np.int64(3)},
            {"suite": "s", "item": "a", "metric": "k", "value": float("nan")}]
    text = runs.metrics_csv_text(rows)
    lines = text.splitlines()
    assert lines[0] == ",".join(runs.METRIC_COLUMNS)
    assert [ln.split(",")[3] + ln.split(",")[9] for ln in lines[1:]] == ["ak", "am", "bm"]
    assert "0.3," in lines[3] and ",0,3," in lines[2] and "\r" not in text
    assert text == runs.metrics_csv_text(list(reversed(rows)))
    with pytest.raises(ValueError):
        runs.metrics_csv_text([{"metric": "m", "bogus": 1}])


# ---------------------------------------------------------------------------------------- registry


@pytest.fixture
def registry(tmp_path):
    p = tmp_path / "decisions.md"
    shutil.copyfile(paths.GTAB_ROOT / "docs" / "decisions.md", p)
    return p


def test_real_registry_has_every_cli_id_preregistered():
    reg = experiments.load_registry()
    for eid in ("E1", "E2", "E3", "E23", "E24-sep", "E24-amt", "gate-m2", "latency", "stereo-preservation"):
        assert eid in reg, eid
        assert reg[eid].status in (experiments.STATE_PRE, experiments.STATE_RUNNING) or reg[eid].decided
        assert reg[eid].spec.get("rule"), eid
    assert "양식" not in reg  # the template block is not an entry
    assert set(experiments.ALL_IDS) == set(reg) & set(experiments.ALL_IDS)


def _fake_rows(out_dir, cfg, args):
    rng = np.random.default_rng(1)
    base = {(s, line): int(rng.binomial(200, 0.7)) for s in range(12) for line in ("guitar", "bass")}
    rows = []
    for arm, gain in (("off", 0), ("-14", 6), ("-16", 1)):
        for s in range(12):
            for line in ("guitar", "bass"):
                n = 200
                tp = base[(s, line)] + gain + int(rng.integers(-1, 2))
                rows.append({"system": arm, "dataset": "cambridge_mt", "item": f"song{s}", "group": f"song{s}",
                             "scenario": "rock" if s % 2 else "pop", "line": line, "rung": "chunk=588800|medium",
                             "metric": "onset_f1_50", "value": 2 * tp / (2 * tp + (n - tp) * 2),
                             "tp": tp, "fp": n - tp, "fn": n - tp})
    return rows


@pytest.fixture
def fake_amt(monkeypatch):
    mod = types.ModuleType("gtab.amt.experiments")
    mod.EXPERIMENTS = {"E1": _fake_rows}
    monkeypatch.setitem(sys.modules, "gtab.amt.experiments", mod)
    return mod


def test_state_machine_runs_preregistered_and_refuses_decided(registry, tmp_path, fake_amt):
    assert experiments.load_registry(registry)["E1"].status == "사전 등록"
    r = experiments.run_experiment("E1", None, registry_path=registry, runs_root=tmp_path / "runs", gitsha="abcdef1")
    reg = experiments.load_registry(registry)
    assert reg["E1"].status == "실행 중"
    assert r.summary["decision"] == "adopt" and r.summary["chosen"] == "-14"
    assert (r.run_dir / "metrics.csv").is_file() and (r.run_dir / "report.html").is_file()
    assert "계산된 판정" in registry.read_text(encoding="utf-8")
    r2 = experiments.run_experiment("E1", None, registry_path=registry, runs_root=tmp_path / "runs", gitsha="abcdef1")
    assert r2.run_dir != r.run_dir  # reruns allowed while running
    experiments.record_decision("E1", "채택 (-14)", registry_path=registry)
    assert experiments.load_registry(registry)["E1"].status == "결정: 채택 (-14)"
    with pytest.raises(experiments.ExperimentStateError, match="이미 결정"):
        experiments.run_experiment("E1", None, registry_path=registry, runs_root=tmp_path / "runs")
    with pytest.raises(experiments.ExperimentStateError, match="사전 등록되어 있지 않"):
        experiments.run_experiment("E99", None, registry_path=registry, runs_root=tmp_path / "runs")
    # other entries untouched
    assert experiments.load_registry(registry)["E2"].status == "사전 등록"


def test_dry_run_changes_nothing(registry, fake_amt):
    before = registry.read_bytes()
    e = experiments.run_experiment("E1", None, registry_path=registry, dry_run=True)
    assert e.id == "E1" and registry.read_bytes() == before


def test_missing_backend_is_clean_error(registry, monkeypatch):
    mod = types.ModuleType("gtab.amt.experiments")
    mod.EXPERIMENTS = {}
    monkeypatch.setitem(sys.modules, "gtab.amt.experiments", mod)
    with pytest.raises(experiments.ExperimentStateError, match="구현되지"):
        experiments.run_experiment("E2", None, registry_path=registry, dry_run=True)


# ---------------------------------------------------------------------------------------- decisions


def _rows(arms: dict[str, float], *, metric="onset_f1_50", groups=10, rung="r", scenario=None, section=None, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 3, groups)  # per-song noise shared by every arm (paired design)
    out = []
    for arm, p in arms.items():
        for g in range(groups):
            n = 300
            tp = int(round(n * p + noise[g]))
            out.append({"system": arm, "item": f"t{g}", "group": f"t{g}", "metric": metric, "tp": tp, "fp": n - tp,
                        "fn": n - tp, "rung": rung, "scenario": scenario or "", "section": section or ""})
    return out


def test_gate_contamination_rule_holds():
    rows = _rows({"muscriptor": 0.85, "basicpitch": 0.70}, scenario="distortion")
    spec = experiments.parse_spec("`rule=gate; metric=onset_f1_50; candidate=muscriptor; baseline=basicpitch; "
                                  "threshold=5.0; subset=distortion; contamination=muscriptor-medium:egdb`")
    s = experiments.evaluate(rows, spec, n=500)
    assert s["passed"] is True
    assert s["decision_ko"] == "판단 보류 (오염 불명)"
    spec2 = dict(spec, contamination="basic-pitch:egdb")  # a clean cell -> the gate result counts
    assert experiments.evaluate(rows, spec2, n=500)["decision_ko"] == "통과"


def test_contamination_table_cells():
    assert experiments.contamination_status("muscriptor-medium", "egdb") == "불명"
    assert experiments.contamination_status("basic-pitch", "guitarset") == "학습 포함"
    assert experiments.contamination_status("basic-pitch", "egdb") == "학습 제외"
    assert experiments.contamination_status("beat-this-final0", "guitarset") == "학습 포함"
    assert experiments.contamination_status("nope", "egdb") == "불명"


def test_noninferiority_needs_resource_gain():
    rows = _rows({"upstream": 0.80, "fp16": 0.80})
    rows += [{"system": "upstream", "item": "*", "metric": "reserved_peak_mb", "value": 1800.0},
             {"system": "fp16", "item": "*", "metric": "reserved_peak_mb", "value": 1300.0}]
    spec = experiments.parse_spec("rule=noninferiority; metric=onset_f1_50; baseline=upstream; margin=-0.5; "
                                  "resource=reserved_peak_mb:-20%|rtf:+20%")
    s = experiments.evaluate(rows, spec, n=500)
    assert s["per_candidate"]["fp16"] == "adopt"
    rows2 = [r for r in rows if r["metric"] != "reserved_peak_mb"]
    assert experiments.evaluate(rows2, spec, n=500)["per_candidate"]["fp16"] == "hold"


def test_tuning_rule_and_mixed_rungs():
    rows = (_rows({"f4_o0.5_fr0.3": 0.70, "f5_o0.4_fr0.3": 0.75, "f3_o0.6_fr0.2": 0.60}, section="dev")
            + _rows({"f4_o0.5_fr0.3": 0.70, "f5_o0.4_fr0.3": 0.74, "f3_o0.6_fr0.2": 0.61}, section="test", seed=3))
    spec = experiments.parse_spec("rule=tuning; metric=onset_f1_50; baseline=f4_o0.5_fr0.3; split_col=section")
    s = experiments.evaluate(rows, spec, n=500)
    assert s["tuned"] == "f5_o0.4_fr0.3" and s["decision"] == "adopt"
    mixed = _rows({"off": 0.7}, rung="chunk=588800") + _rows({"off": 0.7}, rung="chunk=262144") + _rows({"-14": 0.8})
    s2 = experiments.evaluate(mixed, experiments.parse_spec("rule=superiority; metric=onset_f1_50; baseline=off; mde=1"), n=200)
    assert s2["decision"] == "hold" and "rung" in s2["decision_ko"]


def test_constraint_blocks_adoption():
    rows = _rows({"guitar+present": 0.70, "all": 0.80})
    rows.append({"system": "all", "item": "t1", "group": "t1", "metric": "guitar_class_omissions", "value": 2})
    spec = experiments.parse_spec("rule=superiority; metric=onset_f1_50; baseline=guitar+present; mde=1.0; "
                                  "constraint=guitar_class_omissions==0")
    s = experiments.evaluate(rows, spec, n=500)
    assert s["per_candidate"]["all"] == "reject" and s["comparisons"]["all"]["constraint_violations"] == 1


def test_select_on_dev_confirm_on_test_only_the_chosen_arm():
    """E3 style: the arm is chosen on dev (constraint respected) and only that arm is confirmed on test.
    'b' wins on test but not on dev, so it is never picked; 'c' wins on dev but breaks the constraint there."""
    dev = _rows({"base": 0.70, "a": 0.76, "b": 0.71, "c": 0.80}, section="dev")
    test = _rows({"base": 0.70, "a": 0.75, "b": 0.85, "c": 0.86}, section="test", seed=4)
    rows = dev + test
    rows.append({"system": "c", "item": "t0", "group": "t0", "metric": "guitar_class_omissions", "value": 1,
                 "section": "dev"})
    spec = experiments.parse_spec("rule=superiority; metric=onset_f1_50; baseline=base; mde=1.0; "
                                  "constraint=guitar_class_omissions==0; split_col=section; select=dev; split=test")
    s = experiments.evaluate(rows, spec, n=500)
    assert s["chosen_on_select"] == "a" and "c" not in s["eligible"]
    assert list(s["comparisons"]) == ["a"]  # b and c are never compared on test
    assert s["decision"] == "adopt" and s["chosen"] == "a"
    # the dev optimum is the default -> reject (keep default), no test comparison
    s2 = experiments.evaluate(_rows({"base": 0.80, "a": 0.70}, section="dev") + test, spec, n=200)
    assert s2["decision"] == "reject" and "comparisons" not in s2
    # the chosen arm breaks the constraint on test -> not adopted
    rows3 = rows + [{"system": "a", "item": "t1", "group": "t1", "metric": "guitar_class_omissions",
                           "value": 2, "section": "test"}]
    s3 = experiments.evaluate(rows3, spec, n=200)
    assert s3["decision"] == "reject" and s3["comparisons"]["a"]["constraint_violations"] == 1


def test_real_e3_entry_selects_on_dev():
    spec = experiments.load_registry()["E3"].spec
    assert spec["select"] == "dev" and spec["split"] == "test" and spec["constraint"] == "guitar_class_omissions==0"


def test_failed_experiment_keeps_status_and_leaves_note(registry, tmp_path, monkeypatch):
    class ExperimentDataMissing(RuntimeError):
        pass

    def no_data(out_dir, cfg, args):
        raise ExperimentDataMissing("Tier B 데이터가 없습니다")

    mod = types.ModuleType("gtab.amt.experiments")
    mod.EXPERIMENTS = {"E1": no_data, "E2": lambda out, cfg, args: []}
    monkeypatch.setitem(sys.modules, "gtab.amt.experiments", mod)
    before = registry.read_bytes()
    with pytest.raises(experiments.ExperimentRunError) as ei:
        experiments.run_experiment("E1", None, registry_path=registry, runs_root=tmp_path / "runs", gitsha="abcdef1")
    assert ei.value.data_missing and "Tier B" in str(ei.value)
    assert registry.read_bytes() == before  # still 사전 등록, no result line
    note = ei.value.run_dir / "failed.json"
    assert note.is_file() and "ExperimentDataMissing" in note.read_text(encoding="utf-8")
    with pytest.raises(experiments.ExperimentRunError, match="지표 행"):
        experiments.run_experiment("E2", None, registry_path=registry, runs_root=tmp_path / "runs", gitsha="abcdef1")
    assert registry.read_bytes() == before


def test_experiment_run_json_has_provenance(registry, tmp_path, fake_amt):
    import json

    r = experiments.run_experiment("E1", None, registry_path=registry, runs_root=tmp_path / "runs", gitsha="abcdef1")
    info = json.loads((r.run_dir / "run.json").read_text(encoding="utf-8"))
    assert "models" in info and "datasets_lock_sha256" in info and "models_lock_sha256" in info
    if paths.MODELS_LOCK.is_file():
        assert info["models"] and all(len(v) == 64 for v in info["models"].values())


def test_append_result_without_result_field(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("# x\n\n## E9\n- 상태: 사전 등록\n- 판정 설정: `rule=descriptive; metric=m`\n\n## E10\n- 상태: 사전 등록\n",
                 encoding="utf-8")
    experiments.append_result("E9", "2026-10-03 `run`: 계산된 판정 = x", p)
    text = p.read_text(encoding="utf-8")
    assert "- 판정 설정: `rule=descriptive; metric=m`\n- 결과:\n  - 2026-10-03" in text
    reg = experiments.load_registry(p)
    assert set(reg) == {"E9", "E10"} and reg["E9"].fields["결과"] == ""
