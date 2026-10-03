"""alphaTab (Node) importer and renderer: same expanded GT as PyGuitarPro; capo pitches checked against alphaTab's own
pitch (a file PyGuitarPro wrote cannot prove PyGuitarPro's capo semantics); headless render is audible."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gtab.eval import node

pytestmark = [pytest.mark.node,
              pytest.mark.skipif(not node.alphatab_available(), reason="node + node/node_modules/@coderline/alphatab needed")]

pytest.importorskip("guitarpro")
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
import gp_make  # noqa: E402

from gtab.eval import gp_import, refscore  # noqa: E402


@pytest.fixture(scope="module")
def gp5(tmp_path_factory):
    return gp_make.make_fixture_gp5(tmp_path_factory.mktemp("at") / "fixture.gp5")


def _key(n):
    return (n.line, n.bar, round(n.tick, 3), round(n.dur_ticks, 3), n.pitch, n.string, n.fret, n.shared)


def test_alphatab_import_equals_pyguitarpro(gp5):
    at = gp_import.import_reference(gp5, engine="alphatab")
    py = gp_import.import_reference(gp5, engine="pyguitarpro")
    assert at.source["importer"] == "alphatab-1.8.4"
    assert [t.name for t in at.tracks] == [t.name for t in py.tracks] == [gp_make.TRACK1_NAME, "Lead"]
    assert [t.tuning for t in at.tracks] == [t.tuning for t in py.tracks]
    assert [t.capo for t in at.tracks] == [t.capo for t in py.tracks]
    lines = {1: "L1", 2: "L2"}
    ga = refscore.expand(at, song_id="fx", track_lines=lines)
    gp = refscore.expand(py, song_id="fx", track_lines=lines)
    assert sorted(_key(n) for n in ga.notes) == sorted(_key(n) for n in gp.notes) == sorted(gp_make.EXPECTED)
    assert ga.bars == gp.bars


def test_capo_pitch_matches_alphatab_reference(gp5):
    """alphaTab computes pitch itself (fret + capo + tuning); our convention must agree note by note."""
    at = gp_import.import_reference(gp5, engine="alphatab")
    tracks = {t.index: t for t in at.tracks}
    assert tracks[1].capo == 2
    for n in at.notes:
        t = tracks[n.track]
        assert n.pitch == t.tuning[n.string - 1] + t.capo + n.fret, n
    py = gp_import.read_gp345(gp5)
    key = lambda n: (n.track, n.bar, n.start, n.string)  # noqa: E731
    assert {key(n): n.pitch for n in at.notes} == {key(n): n.pitch for n in py.notes}


def test_alphatab_render_is_audible(gp5, tmp_path):
    import soundfile as sf

    out = tmp_path / "render.wav"
    info = node.alphatab_render(gp5, out)
    x, sr = sf.read(str(out), dtype="float32")
    assert sr == 44100 and x.ndim == 2 and x.shape[1] == 2
    dur = len(x) / sr
    assert 12.0 <= dur <= 12.0 + 3.0  # 0.5 + 4×2 + 1.5 + 2 s of music (repeat expanded) + release tail
    assert float(np.sqrt(np.mean(x ** 2))) > 1e-3
    assert info.get("frames") == len(x)
