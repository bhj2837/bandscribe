"""E26 helpers (bandscribe.tab.reapply_eval): old-result choice, sampling, note and bar-line classes."""

from __future__ import annotations

import json
import random
from pathlib import Path

from bandscribe.tab import reapply_eval as R


def _grid(n_bars: int = 6, beats: int = 4, ibi: float = 0.5) -> dict:
    bs, bars = [], []
    for b in range(n_bars):
        for k in range(beats):
            i = b * beats + k
            bs.append({"t_s": round(i * ibi, 4), "bar": b + 1, "beat": k + 1, "is_downbeat": k == 0})
        bars.append({"bar": b + 1, "start_s": b * beats * ibi, "end_s": (b + 1) * beats * ibi, "n_beats": beats,
                     "numerator": beats, "denominator": 4, "beat_unit": "quarter", "pickup": False})
    return {"beats": bs, "bars": bars}


def _note(t: float, p: int, **kw) -> dict:
    return {"onset_s": t, "offset_s": t + 0.2, "pitch": p, **kw}


def test_old_quant_takes_the_last_result_of_the_m4a_day(tmp_path: Path) -> None:
    for name, when in (("a", "2026-10-05T07:00:00Z"), ("b", "2026-10-05T12:50:00Z"), ("c", "2026-10-08T10:00:00Z")):
        d = tmp_path / "stages" / "quant" / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"created_utc": when, "deps": {}}), encoding="utf-8")
        (d / "quant.json").write_text("{}", encoding="utf-8")
    assert R.old_quant(tmp_path).name == "b"
    assert R.old_quant(tmp_path, "2026-10-09") is None


def test_sampling_is_deterministic_and_registered_size() -> None:
    notes = {"guitar_all": [_note(i * 0.1, 60) for i in range(80)], "bass": [_note(i * 0.2, 40) for i in range(30)]}
    a = R.sample_edits(notes, _grid(30), random.Random(R.SEED))
    b = R.sample_edits(notes, _grid(30), random.Random(R.SEED))
    assert a == b
    kinds = [e["kind"] for e in a]
    assert kinds.count("guitar") == R.N_NOTES and kinds.count("bass") == 30 and kinds.count("bar") == R.N_BARS
    assert all(e["at"] > 0 and e["to"] == e["at"] + 1 for e in a if e["kind"] == "bar")  # never the first bar


def test_note_classes() -> None:
    new = {"guitar_all": [_note(1.0, 60), _note(2.05, 62), _note(3.0, 76), _note(4.01, 65), _note(4.04, 65)],
           "bass": []}

    def cls(t: float, p: int) -> dict:
        return R.classify_note(new, {"note": {"track": "guitar_all", "onset_s": t, "pitch": p}})

    assert cls(1.0, 60)["class"] == "exact"
    assert cls(1.02, 60)["class"] == "within_tol"
    assert cls(2.0, 62)["class"] == "near_miss" and cls(2.0, 62)["wide"]
    assert cls(3.0, 64)["class"] == "octave"
    assert cls(5.0, 60)["class"] == "gone" and not cls(5.0, 60)["landed"]
    assert cls(4.0, 65)["wide_ambiguous"]


def test_bar_classes() -> None:
    g = _grid(6)
    times = [b["t_s"] for b in g["beats"]]
    e = {"at_s": times[4], "to_s": times[5]}
    assert R.classify_bar(g, e) == {"landed": True, "class": "exact"}
    assert R.classify_bar(g, {"at_s": times[4] + 0.1, "to_s": times[5] + 0.1})["class"] == "within_tol"
    assert R.classify_bar(g, {"at_s": times[5], "to_s": times[6]})["class"] == "beat_not_bar"
    assert R.classify_bar(g, {"at_s": 99.0, "to_s": 99.5})["class"] == "no_beat"


def test_block_ci_pools_counts() -> None:
    rate, lo, hi = R.block_ci({"a": (9, 10), "b": (1, 10)}, n=200)
    assert rate == 0.5 and 0.1 <= lo <= rate <= hi <= 0.9
