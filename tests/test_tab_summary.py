"""The run summary card (mid-term plan 1-7)."""

from __future__ import annotations

from bandscribe import atomic
from bandscribe.jobs import hints as HI
from bandscribe.tab import health as H
from bandscribe.tab import summary as SM
from test_tab_health import _job


def _score(job):
    atomic.write_json(job / "export" / "score.json", {"lines": [
        {"id": "gtr_1", "name": "Gtr 1 - Rhythm (dist., L)", "kind": "guitar", "notes": [{}] * 3},
        {"id": "gtr_2", "name": "Gtr 2 - Lead (dist., R)", "kind": "guitar", "notes": [{}] * 2},
        {"id": "guitar_all", "name": "All guitars (reference)", "kind": "reference", "notes": [{}] * 5},
        {"id": "bass", "name": "Bass", "kind": "bass", "notes": [{}] * 4}]})
    (job / "export" / "tab" / "score.gp").write_bytes(b"PK")


def test_card_reads_tempo_tuning_parts_and_health(tmp_path, monkeypatch):
    job, dirs = _job(tmp_path, bass_burst=True)
    _score(job)
    atomic.write_json(dirs["grid"] / "grid.json", {**atomic.read_json(dirs["grid"] / "grid.json"),
                                                   "meter": [{"start_bar": 0, "end_bar": 90, "numerator": 4,
                                                              "denominator": 4}]})
    monkeypatch.setattr(H, "stage_dirs", lambda job_dir: dirs)
    doc = H.song_health(job, results=dirs)
    rows = dict(SM.card(job, doc, view_arg="my song"))
    assert rows["템포"] == "120 BPM (자동, 확인 안 됨 — 다른 후보 60 · 240. 틀렸으면 --tempo) · 4/4"
    assert rows["튜닝"] == "기타 Eb standard (여유 -0.05, 확인 필요) · 베이스 E standard (여유 0.30)"
    assert rows["파트"] == "기타 2파트 — Gtr 1 - Rhythm (dist., L) 3음 · Gtr 2 - Lead (dist., R) 2음 · 베이스 4음"
    assert rows["점검"].startswith(f"경고 {doc['summary']['warn']} · 알림 {doc['summary']['info']}")
    assert rows["보기"].startswith('bandscribe view "my song"') and rows["파일"].endswith("score.gp")
    HI.set_hint(job, "hints", "tempo_confirmed", True)
    assert dict(SM.card(job, doc))["템포"] == "120 BPM (자동, 확인함) · 4/4"
    assert dict(SM.card(job, doc))["보기"].startswith(f"bandscribe view {job.name} ")
