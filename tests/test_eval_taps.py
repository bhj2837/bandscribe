"""Audacity label parsing and tap -> bar assignment."""

from __future__ import annotations

import pytest

from gtab.eval import taps

LABELS = "\n".join([
    "1.000000\t1.000000\tb1",
    "\\\t100.0\t2000.0",           # spectral-label continuation line: ignored
    "3.010000\t3.010000\t",       # unlabeled -> nearest GT downbeat
    "5,000000\t5,000000\t3",       # comma decimal (some locales), bare bar number
    "junk line",
    "7.020000\t7.020000\t박자",    # non-numeric label -> assigned
])


def test_parse_labels():
    rows = taps.parse_audacity_labels(LABELS)
    assert [r["t_s"] for r in rows] == [1.0, 3.01, 5.0, 7.02]
    assert [r["bar"] for r in rows] == [1, None, 3, None]
    assert rows[3]["label"] == "박자"


def test_read_label_text_cp949(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_bytes("2.5\t2.5\t다운비트\n".encode("cp949"))
    assert taps.parse_audacity_labels(taps.read_label_text(p))[0]["label"] == "다운비트"
    p.write_bytes("﻿2.5\t2.5\tb4\n".encode("utf-8"))
    assert taps.parse_audacity_labels(taps.read_label_text(p))[0]["bar"] == 4


def test_assign_bars_nearest_and_monotonic():
    from gtab.eval.refscore import tempo_map_dict, tempo_map_from_dict

    times = [1.0 + 0.5 * i for i in range(4 * 6 + 1)]  # bar 1 at 1.0 s, 2 s per bar
    bars = [1 + i // 4 for i in range(len(times))]
    beats = [1 + i % 4 for i in range(len(times))]
    tm = tempo_map_from_dict(tempo_map_dict(times, bars, beats, {b: 4 for b in set(bars)}, source="x"))
    rows = taps.assign_bars(taps.parse_audacity_labels(LABELS), tm, range(1, 7))
    assert [r["bar"] for r in rows] == [1, 2, 3, 4]
    js = taps.taps_json("labels.txt", rows)
    assert js["format"] == "gtab.taps/1" and js["taps"][1] == {"t_s": 3.01, "bar": 2, "label": ""}


@pytest.mark.parametrize("label,bar", [("b17", 17), ("B 3", 3), ("  12 ", 12), ("x17", None)])
def test_bar_label_forms(label, bar):
    assert taps.parse_audacity_labels(f"1\t1\t{label}")[0]["bar"] == bar
