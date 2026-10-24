"""``bandscribe eval sheets`` (mid-term plan 1-2): the sheet-song measures on synthetic sheets, and a snapshot of the
local songs (skipped on a PC without them: the sheets and jobs are never in the repository)."""

from __future__ import annotations

from collections import Counter

import pytest

from bandscribe import paths
from bandscribe.eval import sheets as SH


class _Tempo:  # one beat per second
    def beat_to_time(self, b):
        return float(b)


def _bars(pitches_per_bar, beats=4):
    return [SH.SheetBar(i + 1, i * beats, beats, list(ps)) for i, ps in enumerate(pitches_per_bar)]


def _est(spec):  # [(bar index, pitch)] -> notes at the bar's start + 0.5 s
    return sorted((4.0 * i + 0.5, p, "1") for i, p in spec)


def test_sheet_tuning_reads_down_tuning_retuned_strings_and_capo():
    std = SH.GUITAR
    assert SH.sheet_tuning("Standard tuning", std) == std
    assert SH.sheet_tuning("Tune down 1/2 step", std) == tuple(x - 1 for x in std)
    drop_d = SH.sheet_tuning(f"{chr(0xE834 + 5)} = D", std)          # string 6 to D
    assert drop_d == (38,) + std[1:]
    assert SH.sheet_tuning("Capo. fret 3", std) == tuple(x + 3 for x in std)
    assert SH.sheet_tuning(f"Tune down 1/2 step\n{chr(0xE834 + 5)} = D{SH.FLAT}", std)[0] == 37


def test_bag_counts_unites_the_guitars_and_uses_the_bar_span():
    g1 = _bars([[40, 52], [45]])
    g2 = _bars([[52, 64], []])
    est = _est([(0, 40), (0, 52), (0, 64), (0, 70), (1, 45), (1, 45)])
    c = SH.bag_counts(_Tempo(), [g1, g2], est, lambda _b: 0)
    assert (c.tp, c.fp, c.fn) == (4, 2, 0)                           # 52 once (the union), 70 and the second 45 extra
    assert SH.bag_counts(_Tempo(), [g1], est, lambda _b: 0).fn == 0
    early = [(3.95, 45, "1")]                                        # 60 ms early still counts in bar 2
    assert SH.bag_counts(_Tempo(), [_bars([[], [45]])], early, lambda _b: 0).tp == 1


def test_offsets_global_and_drifting():
    pitch = [40 + (i % 5) for i in range(32)]
    sheet = _bars([[p] * 8 for p in pitch])                           # 8 notes a bar: 64 a window
    # sheet beat b is our beat b in bars 1-16, then b + 1 (one spurious beat): offset 0, then -1
    est = sorted((4.0 * i + 0.25 + 0.5 * k + (1 if i >= 16 else 0), pitch[i], "1") for i in range(32) for k in range(8))
    off = SH.best_offset(_Tempo(), [sheet], est)
    local, changes = SH.drifting_offsets(_Tempo(), [sheet], est, off)
    assert off == 0 and [(c["sheet_bar"], c["to"]) for c in changes] == [(17, -1)]
    assert local[16] == 0 and local[17] == -1
    drift = SH.bag_counts(_Tempo(), [sheet], est, lambda b: local[b])
    assert drift.f1 == 100.0 > SH.bag_counts(_Tempo(), [sheet], est, lambda _b: off).f1


def test_load_songs_refuses_an_invalid_job(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    f = tmp_path / "songs.yaml"
    f.write_text("songs:\n  - {id: a, job: f-1, sheets: s}\ninvalid:\n  - {job: f-2, why: backing}\n", encoding="utf-8")
    songs, invalid = SH.load_songs(f)
    assert [s.id for s in songs] == ["a"] and songs[0].sheets == tmp_path / "s" and invalid == {"f-2": "backing"}
    f.write_text("songs:\n  - {id: a, job: f-2, sheets: s}\ninvalid:\n  - {job: f-2}\n", encoding="utf-8")
    with pytest.raises(SH.SheetEvalError, match="무효"):
        SH.load_songs(f)
    with pytest.raises(SH.SheetEvalError, match="곡 목록이 없습니다"):
        SH.load_songs(tmp_path / "none.yaml")


# The local songs on 2026-10-24 (after E39/E40 and the 2276733 re-run; docs/MIDTERM_REVIEW.md 2.1): song -> merged,
# part gtr1, part gtr2, bass bar F1. A move of more than 0.5 point fails: re-measure on purpose, then update here
# and in decisions.md's floor.
SNAPSHOT = {"sc": (73.98, 65.71, 58.14, 90.41), "gka": (60.65, 48.42, 59.64, 70.91), "anb": (64.20, 55.65, 43.50, 85.93),
            "snn": (71.74, 67.55, 54.64, 91.32), "hbt": (71.52, 57.64, 57.89, 81.34), "hnn": (59.21, 50.77, 45.61, 70.84)}
POOLED = {"merged": 66.84, "merged_drift": 68.86, "parts": 55.92, "bass": 81.52, "bass_drift": 85.56}


@pytest.mark.skipif(not SH.songs_file().is_file(), reason="the sheet songs are local only")
def test_snapshot_of_the_local_sheet_songs():
    songs, _ = SH.load_songs()
    have = [s for s in songs if (paths.JOBS / s.job).is_dir() and s.sheets.is_dir()]
    if {s.id for s in have} != set(SNAPSHOT):
        pytest.skip("not every sheet song is on this PC")
    doc = SH.evaluate(have)
    moved = {}
    for row in doc["songs"]:
        got = (row["merged"]["F1"], row["part_gtr1"]["F1"], row["part_gtr2"]["F1"], row["bass"]["F1"])
        if any(abs(a - b) > 0.5 for a, b in zip(got, SNAPSHOT[row["song"]])):
            moved[row["song"]] = got
    moved.update({k: doc["pooled"][k]["F1"] for k, v in POOLED.items() if abs(doc["pooled"][k]["F1"] - v) > 0.5})
    assert not moved, f"sheet scores moved by more than 0.5 point: {moved}"
    assert round(100 * doc["pooled"]["downbeats"]["share"], 1) == 82.4
    assert sum(Counter(r["part_map"].values()).total() for r in doc["songs"]) == 12
