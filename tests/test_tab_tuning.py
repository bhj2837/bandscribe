"""M3 tuning search: "[판정] 튜닝 픽스처(전조 렌더, 오픈 튜닝 포함)를 맞힌다" (DESIGN 10 M3)."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.tab import fretting as F
from bandscribe.tab import tuning as T

E_STD = (40, 45, 50, 55, 59, 64)
# chord shapes as frets for strings 6..1 (None = not played)
OPEN_SHAPES = {
    "G": (3, 2, 0, 0, 0, 3), "C": (None, 3, 2, 0, 1, 0), "D": (None, None, 0, 2, 3, 2), "Em": (0, 2, 2, 0, 0, 0),
    "Am": (None, 0, 2, 2, 1, 0), "E": (0, 2, 2, 1, 0, 0), "A": (None, 0, 2, 2, 2, 0),
}


def shape_pitches(tuning, frets) -> list[int]:
    return [tuning[i] + f for i, f in enumerate(frets) if f is not None]


def song(tuning, seed: int = 0, transpose: int = 0, n_bars: int = 16) -> list[F.Event]:
    """Open-chord strums + a single-note bass-string line + a melody, played in ``tuning`` (6 strings, low
    first) and then transposed (a transposed render keeps the shapes, shifts the pitches)."""
    rng = np.random.default_rng(seed)
    evs, t = [], 0.0
    names = list(OPEN_SHAPES)
    for _ in range(n_bars):
        sh = OPEN_SHAPES[names[int(rng.integers(len(names)))]]
        for _k in range(4):
            ps = [p + transpose for p in shape_pitches(tuning, sh)]
            evs.append(F.Event(ps, t, [t + 0.45] * len(ps), list(range(len(ps)))))
            t += 0.5
        for _k in range(4):  # a line on the low strings, frets 0-5
            s = int(rng.integers(0, 3))
            p = tuning[s] + int(rng.integers(0, 6)) + transpose
            evs.append(F.Event([p], t, [t + 0.2], [0]))
            t += 0.25
    return evs


def bass_line(tuning, seed: int = 1, transpose: int = 0, n: int = 64) -> list[F.Event]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = int(rng.integers(0, len(tuning)))
        p = tuning[s] + int(rng.integers(0, 5)) + transpose
        out.append(F.Event([p], 0.25 * i, [0.25 * i + 0.2], [0]))
    out.insert(0, F.Event([tuning[0] + transpose], 0.0, [0.2], [0]))  # the open low string once
    return out


def riff(tuning, frets_on_low3: list[int], transpose: int = 0) -> list[F.Event]:
    """One-finger power chords on the three lowest strings (a drop-tuning riff)."""
    evs, t = [], 0.0
    for _rep in range(8):
        for f in frets_on_low3:
            ps = [tuning[i] + f + transpose for i in range(3)]
            evs.append(F.Event(ps, t, [t + 0.2] * 3, [0, 1, 2]))
            t += 0.25
    return evs


@pytest.mark.parametrize("transpose,label", [(0, "E standard"), (-1, "Eb standard"), (-2, "D standard")])
def test_transposed_renders(transpose, label):
    res = T.choose(song(E_STD, transpose=transpose), bass_line((28, 33, 38, 43), transpose=transpose))
    assert res["guitar"]["label"] == label, res["guitar"]["top"]
    assert res["guitar"]["downtune"] == res["bass"]["downtune"] == transpose
    assert res["guitar"]["unplayable"] == 0.0 and not res["guitar"]["unknown"]


def test_drop_d_riff():
    drop = (38, 45, 50, 55, 59, 64)
    evs = riff(drop, [0, 0, 3, 5, 0, 0, 7, 5]) + song(drop, seed=3, n_bars=4)
    res = T.choose(evs, bass_line((26, 33, 38, 43)))
    assert res["guitar"]["label"] == "Drop D", res["guitar"]["top"]
    assert res["bass"]["label"] in ("Drop D", "E standard") and res["bass"]["unplayable"] == 0.0


def test_open_g():
    og = (38, 43, 50, 55, 59, 62)
    shapes = [(0, 0, 0, 0, 0, 0), (5, 5, 5, 5, 5, 5), (7, 7, 7, 7, 7, 7), (0, 0, 0, 0, 0, 0)]
    evs, t = [], 0.0
    for _rep in range(12):
        for sh in shapes:
            ps = shape_pitches(og, sh)
            evs.append(F.Event(ps, t, [t + 0.45] * 6, list(range(6))))
            t += 0.5
    res = T.choose(evs, [])
    assert res["guitar"]["label"] == "Open G", res["guitar"]["top"]
    assert res["bass"] is None


def test_fake_low_notes_keep_drop_roots_and_drop_bleed_and_stray_octaves():
    def n(t, p):
        return {"onset_s": t, "offset_s": t + 0.3, "pitch": p}

    g = [n(0.0, 33),                              # 0 lone A1 at the bass's exact pitch -> bleed
         n(1.0, 38),                              # 1 lone D2, the bass plays D1 (an octave below) -> kept
         n(5.0, 38), n(5.0, 45), n(5.0, 50),      # 2 D2 root of a power chord, the bass plays D2 too -> kept
         n(20.0, 36), n(20.0, 48),                # 5 a bare octave with no other low note near -> octave error
         n(40.0, 38), n(40.0, 50), n(40.5, 38), n(40.5, 50), n(41.0, 38), n(41.0, 50)]  # a repeated octave riff
    b = [n(0.0, 33), n(1.0, 26), n(5.0, 38)]
    assert T.fake_low_notes(g, b) == {0, 5}


def _notes(events):
    return [{"onset_s": e.onset_s, "offset_s": off, "pitch": p} for e in events for p, off in zip(e.pitches, e.offsets_s)]


@pytest.mark.parametrize("octave_down", [12, 0])
def test_drop_d_survives_the_guard_with_a_doubling_bass(octave_down):
    """Review 2026-10-05: the first guard removed every drop-D root the bass doubled (pitch class) and every
    power-chord root (octave pair), and the Drop D fixture came out E standard."""
    drop = (38, 45, 50, 55, 59, 64)
    g = _notes(riff(drop, [0, 0, 3, 5, 0, 0, 7, 5]) + song(drop, seed=3, n_bars=4))
    b = [{**x, "pitch": x["pitch"] - octave_down} for x in g if x["pitch"] in (38, 41, 43, 45)]
    fake = T.fake_low_notes(g, b)
    res = T.choose(F.group_events([x for i, x in enumerate(g) if i not in fake], key="onset_s", tol=0.01),
                   F.group_events(b, key="onset_s", tol=0.01))
    assert res["guitar"]["label"] == "Drop D", res["guitar"]["top"]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("transpose,label", [(0, "E standard"), (-1, "Eb standard"), (-2, "D standard")])
def test_transposed_renders_through_the_guard(seed, transpose, label):
    g = _notes(song(E_STD, seed=seed, transpose=transpose))
    b = _notes(bass_line((28, 33, 38, 43), transpose=transpose))
    fake = T.fake_low_notes(g, b)
    res = T.choose(F.group_events([x for i, x in enumerate(g) if i not in fake], key="onset_s", tol=0.01),
                   F.group_events(b, key="onset_s", tol=0.01))
    assert (res["guitar"]["label"], res["bass"]["label"]) == (label, label)


def test_runner_up_and_margin_are_reported():
    res = T.choose(song(E_STD, seed=5), [])
    g = res["guitar"]
    assert g["runner_up"] is not None and g["margin"] is not None and g["margin"] >= 0
    assert len(g["top"]) == 5 and g["top"][0]["label"] == g["label"]
