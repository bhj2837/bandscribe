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


def test_fake_low_notes_from_bass_bleed_and_octave_errors():
    g = [{"onset_s": 0.0, "offset_s": 0.4, "pitch": 36}, {"onset_s": 0.0, "offset_s": 0.4, "pitch": 48},
         {"onset_s": 1.0, "offset_s": 1.4, "pitch": 33}, {"onset_s": 2.0, "offset_s": 2.4, "pitch": 38},
         {"onset_s": 3.0, "offset_s": 3.4, "pitch": 45}]
    b = [{"onset_s": 0.98, "offset_s": 1.5, "pitch": 33}]
    # 36 has its octave (48) at the same onset; 33 is the bass's A1; 38 (D2) stays: a real low note
    assert T.fake_low_notes(g, b) == {0, 2}


def test_runner_up_and_margin_are_reported():
    res = T.choose(song(E_STD, seed=5), [])
    g = res["guitar"]
    assert g["runner_up"] is not None and g["margin"] is not None and g["margin"] >= 0
    assert len(g["top"]) == 5 and g["top"][0]["label"] == g["label"]
