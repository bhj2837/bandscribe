"""``bandscribe backup`` and the doctor's backup check (mid-term plan 1-1)."""

from __future__ import annotations

import os
import types

import pytest

from bandscribe import atomic, paths
from bandscribe import backup as B


@pytest.fixture()
def layout(tmp_path, monkeypatch):
    root = tmp_path / "bs"
    data = root / "data"
    for name, p in (("ROOT", root), ("DATA", data), ("JOBS", data / "jobs"), ("USER_CONFIG", data / "config.toml")):
        monkeypatch.setattr(paths, name, p)
    (data / "models" / "bs_roformer_sw").mkdir(parents=True)
    (data / "models" / "bs_roformer_sw" / B.SW_NAME).write_bytes(b"sw" * 1000)
    (data / "eval" / "tierA").mkdir(parents=True)
    (data / "eval" / "tierA" / "sc.json").write_text("{}", encoding="utf-8")
    (data / "eval" / "big_runs").mkdir()
    (data / "eval" / "big_runs" / "x.bin").write_bytes(b"0" * 100)          # not irreplaceable: left out
    (data / "tmp" / "sheets_sc").mkdir(parents=True)
    (data / "tmp" / "sheets_sc" / "tab.json").write_text("[]", encoding="utf-8")
    (data / "tmp" / "sheets_sc" / "__pycache__").mkdir()
    (data / "tmp" / "sheets_sc" / "__pycache__" / "a.pyc").write_bytes(b"x")
    job = data / "jobs" / "f-1"
    (job / "input").mkdir(parents=True)
    (job / "hints.toml").write_text('[grid]\ntempo_bpm = 88\n', encoding="utf-8")
    (job / "edits.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    atomic.write_json(job / "input" / "meta.json", {"duration_s": 1.0})
    (job / "input" / "source.wav").write_bytes(b"RIFF")                          # a copy of an extra: left out
    (data / "config.toml").write_text("[backup]\n", encoding="utf-8")
    music = tmp_path / "음원"
    music.mkdir()
    (music / "lulu.mp3").write_bytes(b"ID3")
    cfg = types.SimpleNamespace(backup=types.SimpleNamespace(extra=[str(music), str(tmp_path / "gone")]))
    return root, cfg, tmp_path / "backup"


def test_backup_mirrors_what_cannot_be_made_again_and_is_incremental(layout):
    root, cfg, dest = layout
    files, missing = B.plan_files(cfg)
    rels = sorted(r for r, _ in files)
    assert rels == ["config/config.toml", "eval_tierA/sc.json", "extra_1_음원/lulu.mp3", "jobs/f-1/edits.jsonl",
                    "jobs/f-1/hints.toml", "jobs/f-1/input/meta.json", f"models/bs_roformer_sw/{B.SW_NAME}",
                    "tmp_sheets_sc/tab.json"]
    assert missing == [str(root.parent / "gone")]
    rec = B.backup(dest, cfg)
    assert rec["files"] == rec["copied"] == 8 and rec["sw_sha256"] == B.sha256(dest / f"models/bs_roformer_sw/{B.SW_NAME}")
    assert (dest / "extra_1_음원" / "lulu.mp3").read_bytes() == b"ID3" and B.last_backup()["dest"] == str(dest)
    assert B.backup(dest, cfg)["copied"] == 0                                    # nothing changed
    hints = root / "data" / "jobs" / "f-1" / "hints.toml"
    hints.write_text('[grid]\ntempo_bpm = 176\n', encoding="utf-8")
    os.utime(hints, ns=(1, 1_000_000_000))
    rec = B.backup(dest, cfg)
    assert rec["copied"] == 1 and "176" in (dest / "jobs/f-1/hints.toml").read_text(encoding="utf-8")
    man = atomic.read_json(dest / B.MANIFEST)
    assert len(man["files"]) == 8 and all(len(f["sha256"]) == 64 for f in man["files"])


def test_backup_refuses_a_folder_inside_bandscribe(layout):
    root, cfg, _ = layout
    with pytest.raises(B.BackupError, match="bandscribe 폴더 안"):
        B.backup(root / "data" / "bk", cfg)


def test_doctor_warns_without_a_recent_backup_holding_the_sw_checkpoint(layout, monkeypatch):
    from bandscribe import doctor

    cfg = {"backup": {"max_age_days": 14}}
    monkeypatch.setattr(doctor, "_today", lambda: __import__("datetime").date(2026, 10, 24))
    (c,) = doctor._check_backup(cfg)
    assert not c.ok and not c.required and "백업 기록이 없습니다" in c.detail
    atomic.write_json(paths.DATA / B.RECORD, {"date": "2026-10-01 10:00", "dest": "X", "sw_sha256": None})
    (c,) = doctor._check_backup(cfg)
    assert not c.ok and "23일 전" in c.detail and "SW 체크포인트" in c.detail
    atomic.write_json(paths.DATA / B.RECORD, {"date": "2026-10-20 10:00", "dest": "X", "sw_sha256": "ab" * 32})
    (c,) = doctor._check_backup(cfg)
    assert c.ok and c.data["age_days"] == 4
