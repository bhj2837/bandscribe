"""M3 string/fret assignment (DESIGN 6.3): textbook fingerings, invariants, dropped notes, confidence."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.tab import fretting as F

STD = F.Instrument()


def ev(t: float, pitches, dur: float = 0.25) -> F.Event:
    return F.Event(list(pitches), t, [t + dur] * len(pitches), list(range(len(pitches))))


def fingering(res, event: int) -> list[tuple[int, int, int]]:
    return sorted((a.pitch, a.string, a.fret) for a in res if a.event == event)


def test_open_position_scale_and_open_chords():
    evs = [ev(0.25 * i, [p]) for i, p in enumerate([48, 50, 52, 53, 55, 57, 59, 60])]
    evs += [ev(2.5, [48, 52, 55, 60, 64], 1.0), ev(3.5, [43, 47, 50, 55, 59, 67], 1.0)]
    res = F.assign(evs, STD)
    assert [(a.string, a.fret) for a in res[:8]] == [(5, 3), (4, 0), (4, 2), (4, 3), (3, 0), (3, 2), (2, 0), (2, 1)]
    assert fingering(res, 8) == [(48, 5, 3), (52, 4, 2), (55, 3, 0), (60, 2, 1), (64, 1, 0)]  # C: x32010
    assert fingering(res, 9) == [(43, 6, 3), (47, 5, 2), (50, 4, 0), (55, 3, 0), (59, 2, 0), (67, 1, 3)]  # G


def test_ringing_arpeggio_takes_separate_strings():
    """Am arpeggio with every note ringing on: no string reused for another fret while a note still sounds."""
    evs = [F.Event([p], 0.2 * i, [2.0], [0]) for i, p in enumerate([45, 52, 57, 60, 64])]
    res = F.assign(evs, STD)
    assert [(a.string, a.fret) for a in res] == [(5, 0), (4, 2), (3, 2), (2, 1), (1, 0)]


@pytest.mark.xfail(strict=True, reason="known limit: v1 plays this 12th-position line (B4-A5) up and down string 1 "
                   "(frets 7-17); the window model (E12b) keeps positions but shifts from 10 to 14 instead of "
                   "stretching 12-17")
def test_high_melody_stays_in_position():
    """A line around the 12th fret stays in one position (frets 12-17 on strings 1-2)."""
    line = [76, 74, 72, 71, 72, 74, 76, 79, 81, 79, 76]
    res = F.assign([ev(0.2 * i, [p]) for i, p in enumerate(line)], STD)
    frets = [a.fret for a in res]
    assert min(frets) >= 7 and max(frets) - min(frets) <= 5


def test_every_assignment_is_playable_and_correct_pitch():
    rng = np.random.default_rng(0)
    evs = []
    t = 0.0
    for _ in range(300):
        k = int(rng.choice([1, 1, 1, 2, 3, 4]))
        root = int(rng.integers(40, 70))
        evs.append(ev(t, sorted({root + int(x) for x in rng.choice([0, 3, 4, 7, 12, 15, 16], size=k)})))
        t += float(rng.choice([0.125, 0.25, 0.5]))
    res = F.assign(evs, STD)
    chk = F.playable(res, evs, STD)
    assert chk["bad_events"] == []
    assert all(STD.open_pitch(a.string) + a.fret == a.pitch for a in res if a.string is not None)
    assert all(0.0 <= a.confidence <= 1.0 for a in res)


def test_unplayable_chords_drop_notes_and_out_of_range_pitches_are_dropped():
    # 8 notes at once (two guitars merged): at most 6 can be placed, octave doublings go first
    big = [40, 47, 52, 55, 59, 64, 67, 76]
    res = F.assign([ev(0.0, big), ev(1.0, [30]), ev(2.0, [100])], STD)
    placed = [a for a in res if a.event == 0 and a.string is not None]
    assert len(placed) <= 6 and len({a.string for a in placed}) == len(placed)
    assert {a.pitch for a in res if a.event == 0 and a.string is None} <= set(big)
    low, high = [a for a in res if a.event == 1][0], [a for a in res if a.event == 2][0]
    assert low.string is None and high.string is None  # below E2 / above fret 24 on the top string


def test_bass_tuning_and_capo():
    bass = F.Instrument(F.BASS_STD, max_fret=20)
    evs = [ev(0.0, [28]), ev(0.5, [33]), ev(1.0, [35]), ev(1.5, [40])]
    res = F.assign(evs, bass)
    assert res[0].string == 4 and res[0].fret == 0  # E1 only exists as the open 4th string
    fretted = [a.fret for a in res if a.fret]
    assert max(fretted) - min(fretted) <= 3  # one hand position (open position or 5th position both fine)
    v1 = F.assign(evs, bass, F.V1_WEIGHTS, model="v1")
    assert [(a.string, a.fret) for a in v1] == [(4, 0), (3, 0), (3, 2), (2, 2)]  # v1 likes open strings
    capo = F.Instrument(F.GUITAR_STD, capo=2)
    a = F.assign([ev(0.0, [42])], capo)[0]  # F#2 = open low string under a capo on 2: written fret 0
    assert (a.string, a.fret) == (6, 0)
    assert F.assign([ev(0.0, [41])], capo)[0].string is None  # below the capo


def test_confidence_reflects_alternatives():
    sure = F.assign([ev(0.0, [40])], STD)[0]          # low E: only one place
    unsure = F.assign([ev(0.0, [64])], STD)[0]        # E4: open 1st string or 5th fret B string or 9th G ...
    assert sure.confidence == pytest.approx(1.0, abs=1e-4)  # one candidate: margin 10, rounded
    assert unsure.confidence < sure.confidence


def test_group_events_by_tick_and_by_onset_window():
    notes = [{"gtick": 0, "onset_s": 0.0, "offset_s": 0.5, "pitch": 40}, {"gtick": 0, "onset_s": 0.02, "offset_s": 0.5, "pitch": 47},
             {"gtick": 24, "onset_s": 0.26, "offset_s": 0.5, "pitch": 52}]
    evs = F.group_events(notes)
    assert [e.pitches for e in evs] == [[40, 47], [52]] and evs[0].ids == [0, 1]
    evs2 = F.group_events(notes, key="onset_s", tol=0.05)
    assert [e.pitches for e in evs2] == [[40, 47], [52]]
