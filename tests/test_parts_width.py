"""E28 width evidence: a centred part next to a wide double (R2 windows) - counting and splitting."""

from __future__ import annotations

from bandscribe.parts import diarize as DZ
from bandscribe.parts import layers as LY
from bandscribe.parts import reid as R


def _scene(n_win: int = 4, label: str = "R2", centre: bool = True):
    """Per 4 s window: 8 wide double notes (ms ~0 dB, coh 0.4, low chords) and, with ``centre``, 8 centred notes
    (ms 25 dB, coh 0.99, a higher line)."""
    notes, sigs, truth = [], [], []
    for w in range(n_win):
        for k in range(8):
            t = w * 4.0 + k * 0.5
            notes.append({"onset_s": t, "offset_s": t + 0.4, "pitch": 45 + k % 3, "instrument": "distorted_electric_guitar"})
            sigs.append({"ild_db": 0.2, "coh": 0.4, "ms_db": 0.0})
            truth.append("wide")
            if centre:
                notes.append({"onset_s": t + 0.25, "offset_s": t + 0.45, "pitch": 67 + k % 4,
                              "instrument": "distorted_electric_guitar"})
                sigs.append({"ild_db": 0.0, "coh": 0.99, "ms_db": 25.0})
                truth.append("centre")
    wins = [{"start_s": w * 4.0, "end_s": w * 4.0 + 4.0, "label": label, "section": f"S{1 + w // 2}"}
            for w in range(n_win)]
    return notes, sigs, wins, truth


ON = {"n_min": 6, "min_windows": 2, "min_sections": 2, "side_db": 3.0, "use_class": False, "use_symbolic": False,
      "use_width": True}


def test_width_split_reads_mid_side_and_coherence():
    sigs = [{"ms_db": 25.0, "coh": 0.99}, {"ms_db": 25.0, "coh": 0.5}, {"ms_db": 0.0, "coh": 0.3},
            {"ms_db": 4.0, "coh": 0.99}, {"ms_db": None, "coh": None}]
    cen, wide = R.width_split(range(5), sigs, ON)
    assert cen == [0] and wide == [2]  # incoherent "centred" and in-between notes count for neither


def test_a_centred_part_next_to_a_double_makes_two_parts_and_is_split():
    notes, sigs, wins, truth = _scene()
    d = DZ.diarize(notes, sigs, wins, params=ON)
    assert d["count"]["chosen"] == 2 and d["count"]["evidence"] == "width"
    by = {}
    for t, n in zip(truth, d["note"]):
        by.setdefault(t, set()).add(n["line"])
    assert len(by["centre"]) == 1 and len(by["wide"]) == 1 and by["centre"] != by["wide"]
    off = DZ.diarize(notes, sigs, wins, params={**ON, "use_width": False})
    assert off["count"]["chosen"] == 1  # v0 without the width evidence


def test_a_double_alone_or_outside_r2_stays_one_part():
    notes, sigs, wins, _t = _scene(centre=False)
    assert DZ.diarize(notes, sigs, wins, params=ON)["count"]["chosen"] == 1  # D1: the double alone
    notes, sigs, wins, _t = _scene(label="R1")
    assert DZ.diarize(notes, sigs, wins, params=ON)["count"]["chosen"] == 1  # an ADT window never reads width
    feats = LY.note_features(notes)
    ws = R.window_streams(list(range(16)), notes, feats, sigs, "R2", ON)
    assert ws["width"] == 2 and R.window_streams(list(range(16)), notes, feats, sigs, "R0", ON)["width"] == 0
