"""M3 export: rhythm spelling, alphaTex (parsed by alphaTab itself when Node is there), GP5 round trip, viewer."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from bandscribe.schema.grid import TempoMap
from bandscribe.tab import alphatex as ATX
from bandscribe.tab import gp5 as GP5
from bandscribe.tab import notation as N
from bandscribe.tab import quantize as Q

from test_tab_quantize import beat_times, grid_doc, performance


def _bar(n_beats: int = 4, unit: str = "quarter", start: int = 0) -> N.WBar:
    num, den = (3 * n_beats, 8) if unit == "dotted_quarter" else (n_beats, 4)
    return N.WBar(bar=1, start_gtick=start, n_beats=n_beats, beat_unit=unit, numerator=num, denominator=den,
                  start_s=0.0, feel=None)


def spelled(segs, *, unit="quarter", n_beats=4, fam=None):
    bar = _bar(n_beats, unit)
    beats = N.spell_bar(bar, segs, fam or {})
    assert N.check_bar(bar, beats) == []
    return [(b.start, b.value, b.dots, b.tuplet, bool(b.notes), [n.tie for n in b.notes]) for b in beats]


NOTE = [N.WNote(60)]


def test_binary_spelling_is_conventional():
    # 16th + 8th inside a beat, never an 8th tied across the middle of the beat
    assert spelled([(0, 12, None, False), (12, 48, NOTE, False), (48, 192, None, False)])[:3] == [
        (0, 16, 0, None, False, []), (12, 16, 0, None, True, [False]), (24, 8, 0, None, True, [True])]
    # dotted 8th + 16th, dotted quarter from a beat, half on beat 3, whole bar
    assert spelled([(0, 36, NOTE, False), (36, 48, NOTE, False), (48, 192, None, False)])[:2] == [
        (0, 8, 1, None, True, [False]), (36, 16, 0, None, True, [False])]
    assert spelled([(0, 72, NOTE, False), (72, 192, None, False)])[0] == (0, 4, 1, None, True, [False])
    assert spelled([(0, 96, None, False), (96, 192, NOTE, False)]) == [(0, 2, 0, None, False, []),
                                                                       (96, 2, 0, None, True, [False])]
    assert spelled([(0, 192, NOTE, False)]) == [(0, 1, 0, None, True, [False])]


def test_ternary_beats_are_triplets_and_a_syncopated_half_is_tied():
    fam = {0: "ter"}
    s = spelled([(0, 16, NOTE, False), (16, 32, NOTE, False), (32, 48, NOTE, False), (48, 192, None, False)], fam=fam)
    assert [x[:4] for x in s[:3]] == [(0, 8, 0, (3, 2)), (16, 8, 0, (3, 2)), (32, 8, 0, (3, 2))]
    s2 = spelled([(0, 8, NOTE, False), (8, 48, NOTE, False), (48, 192, None, False)], fam=fam)
    assert all(x[3] == (6, 4) for x in s2 if x[0] < 48)  # a 16th-triplet point -> sextuplets for the beat
    s3 = spelled([(0, 48, None, False), (48, 144, NOTE, False), (144, 192, None, False)])
    assert [(x[0], x[1]) for x in s3] == [(0, 4), (48, 4), (96, 4), (144, 4)] and s3[2][5] == [True]


def test_compound_meter_needs_no_tuplets():
    s = spelled([(0, 16, NOTE, False), (16, 32, NOTE, False), (32, 48, NOTE, False), (48, 96, NOTE, False),
                 (96, 192, None, False)], unit="dotted_quarter", fam={0: "ter", 1: "ter"})
    assert [x[:4] for x in s[:4]] == [(0, 8, 0, None), (16, 8, 0, None), (32, 8, 0, None), (48, 4, 1, None)]


def _song_score(fretted: bool):
    times = beat_times(12 * 4 + 4, warp=True)
    grid = grid_doc(times)
    notes, _ = performance(times, 12, seed=3)
    q = Q.quantize({"guitar_all": notes, "bass_raw": notes[::4]}, grid)
    tm = TempoMap.from_dict(grid)
    bars = N.bars_from_quant(q, tm, 1, 12, [{"id": "S1", "label": "A", "start_bar": 1}, {"id": "S2", "label": "B", "start_bar": 7}])
    fam = {int(b): f for b, f in q["beat_families"]["*"]}
    score = ATX.WScore(title='Test "song"', bars=bars, tempo_qpm=ATX.written_tempi(bars, section_starts=[1, 7]))
    for tid, name in (("guitar_all", "Guitar (all parts)"), ("bass_raw", "Bass (raw)")):
        evs = {}
        for n in Q.notes_of(q, tid):
            evs.setdefault(n["gtick"], []).append(n)
        events = []
        for g in sorted(evs):
            ns = evs[g][:1] if fretted else evs[g]
            wn = [N.WNote(int(n["pitch"]), 3 if fretted else None, 5 if fretted else None) for n in ns]
            events.append((g, max(int(n["dur"]) for n in ns), wn))
        beats = N.spell_track(events, bars, fam)
        for b, bb in zip(bars, beats):
            assert N.check_bar(b, bb) == []
        score.tracks.append(ATX.WTrack(name=name, short=name[:4], instrument="electricbassfinger", fretted=fretted,
                                       tuning=[40, 45, 50, 55, 59, 64] if fretted else None, capo=0, beats=beats))
    return score


def test_alphatex_text_and_alphatab_parse(tmp_path):
    score = _song_score(fretted=False)
    tex = ATX.write(score)
    assert "\\section (\"A\")" in tex and "\\section (\"B\")" in tex and tex.count("\\sync (") == 12
    assert "\\staff {score}" in tex and "'song'" in tex  # quotes inside the title are neutralised
    from bandscribe.eval import node

    if not node.alphatab_available():
        pytest.skip("alphaTab (node) not installed")
    f = tmp_path / "s.alphatex"
    f.write_text(tex, encoding="utf-8")
    r = node.alphatab_check_tex(f)
    assert r["ok"], r["error"]
    assert r["master_bars"] == 12 and r["sync_points"] == 12
    assert [t["score"] for t in r["tracks"]] == [True, True]


def test_gp5_round_trip(tmp_path):
    import guitarpro as gp

    score = _song_score(fretted=True)
    path = tmp_path / "s.gp5"
    assert GP5.write(score, path)
    song = gp.parse(str(path))
    assert [t.name for t in song.tracks] == ["Guitar (all parts)", "Bass (raw)"]
    assert len(song.measureHeaders) == 12 and song.measureHeaders[6].marker.title == "B"
    n_written = sum(len(be.notes) for bars in score.tracks[0].beats for be in bars)
    n_read = sum(len(b.notes) for m in song.tracks[0].measures for v in m.voices for b in v.beats)
    assert n_written == n_read
    assert not GP5.write(_song_score(fretted=False), tmp_path / "none.gp5")  # no fretted track, no file
    kr = _song_score(fretted=True)
    kr.title = "괴수의 꽃노래"
    assert GP5.text_encoding(kr) == "cp949" and GP5.write(kr, tmp_path / "kr.gp5")
    assert gp.parse(str(tmp_path / "kr.gp5"), encoding="cp949").title == "괴수의 꽃노래"


def test_viewer_serves_the_job_with_ranges(tmp_path):
    from bandscribe.tab import viewer

    job = tmp_path / "f-test"
    (job / "export" / "tab").mkdir(parents=True)
    (job / "input").mkdir()
    (job / "export" / "tab" / "score.alphatex").write_text("\\title (\"x\")\n", encoding="utf-8")
    (job / "export" / "tab" / "tab.json").write_text(json.dumps({"tuning": {"guitar": {"label": "Drop D"}, "params": {}}}),
                                                     encoding="utf-8")
    (job / "input" / "source.wav").write_bytes(bytes(range(256)) * 4)
    (job / "input" / "meta.json").write_text(json.dumps({"source": {"filename": "곡.wav"}}), encoding="utf-8")
    at = tmp_path / "at"
    at.mkdir()
    (at / "alphaTab.min.js").write_text("//", encoding="utf-8")
    httpd, t = viewer.serve(job, alphatab=at)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode("utf-8")
        assert "곡 · bandscribe" in page and "기타 Drop D" in page
        assert urllib.request.urlopen(base + "/score.alphatex").read().startswith(b"\\title")
        req = urllib.request.Request(base + "/audio", headers={"Range": "bytes=10-19"})
        r = urllib.request.urlopen(req)
        assert r.status == 206 and r.read() == bytes(range(10, 20)) and r.headers["Content-Range"] == "bytes 10-19/1024"
        assert urllib.request.urlopen(base + "/alphatab/alphaTab.min.js").read() == b"//"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/alphatab/../input/meta.json")
    finally:
        httpd.shutdown()
        t.join(timeout=5)
    assert isinstance(t, threading.Thread)


def test_bars_before_a_pickup_tile_the_timeline_once():
    """Notes before the first beat of a song with a 2-beat pickup: the extrapolated bars must have the TempoMap's
    length (bar starts -192, -96, 0 here), or the spelled bars overlap and a note is written twice (2026-10-05)."""
    from bandscribe.schema.grid import Grid

    times = beat_times(2 + 8 * 4, warp=False)
    beats, bars = [], [{"bar": 0, "start_s": times[0], "end_s": times[2], "n_beats": 2, "numerator": 2,
                        "denominator": 4, "beat_unit": "quarter", "pickup": True}]
    for i, t in enumerate(times):
        bar, pos = (0, i + 1) if i < 2 else (1 + (i - 2) // 4, 1 + (i - 2) % 4)
        beats.append({"t_s": t, "t_raw_s": t, "bar": bar, "beat": pos, "is_downbeat": i >= 2 and pos == 1,
                      "shift_ms": 0.0, "activation": None})
    for b in range(8):
        s = times[2 + 4 * b]
        bars.append({"bar": b + 1, "start_s": s, "end_s": s + 2.0, "n_beats": 4, "numerator": 4, "denominator": 4,
                     "beat_unit": "quarter", "pickup": False})
    g = {"format": "bandscribe.grid/1", "tpb": 48, "duration_s": times[-1] + 2, "beats": beats, "bars": bars,
         "tempo": [{"start_bar": 0, "end_bar": 9, "bpm": 120.0}], "meter": [], "beat_unit": "quarter",
         "pickup": {"present": True, "beats": 2}, "hypotheses": {"compound": {"decision": "simple"}},
         "confidence": {}, "params": {}}
    Grid.model_validate(g)
    tm = TempoMap.from_dict(g)
    notes = [{"onset_s": times[0] - 1.5, "offset_s": times[0] - 1.0, "pitch": 50},
             {"onset_s": times[0] - 0.75, "offset_s": times[0] - 0.5, "pitch": 52},
             {"onset_s": times[3], "offset_s": times[4], "pitch": 55}]
    q = Q.quantize({"g": notes}, g)
    first = min(n["bar"] for n in q["tracks"]["g"]["notes"])
    wb = N.bars_from_quant(q, tm, first, 2)
    assert [(b.bar, b.start_gtick, b.n_beats, b.numerator) for b in wb] == \
        [(-2, -192, 2, 2), (-1, -96, 2, 2), (0, 0, 2, 2), (1, 96, 4, 4), (2, 288, 4, 4)]
    events = [(n["gtick"], n["dur"], [N.WNote(n["pitch"])]) for n in q["tracks"]["g"]["notes"]]
    spelled = N.spell_track(events, wb, {})
    starts = [(b.start_gtick + be.start, be.notes[0].pitch) for b, bb in zip(wb, spelled) for be in bb
              if be.notes and not be.notes[0].tie]
    assert starts == [(-144, 50), (-72, 52), (144, 55)]
    for b, bb in zip(wb, spelled):
        assert N.check_bar(b, bb) == []
