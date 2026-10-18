"""Chord chart (``bandscribe.tab.chords``, M9 / E31): template matching per beat, change cost, the bass as tie
breaker, empty beats, the bass cross-check and the text chart."""

from __future__ import annotations

from bandscribe.tab import chords as C

BEAT = 0.5


def grid(n_beats=16):
    return {"beats": [{"t_s": k * BEAT, "bar": 1 + k // 4} for k in range(n_beats)]}


def chord(pitches, b0, b1):
    return [{"onset_s": b0 * BEAT, "offset_s": b1 * BEAT, "pitch": p} for p in pitches]


def run(guitar, bass=(), n=16, **kw):
    g = grid(n)
    return C.recognize(guitar, list(bass), C.beat_edges(g), kw or None)


def test_two_chords_and_a_passing_note():
    notes = chord((48, 52, 55), 0, 4) + chord((43, 47, 50), 4, 8) + [{"onset_s": 1.0, "offset_s": 1.2, "pitch": 50}]
    r = run(notes, n=8)
    assert [g["name"] for g in r["segments"]] == ["C", "G"] and r["segments"][1]["start"] == 4
    assert r["beats"][2] == "C"  # the short D on beat 3 does not flip the chord


def test_the_bass_breaks_a_tie():
    sus = chord((48, 50, 55), 0, 4)  # C D G: Csus2 = Gsus4
    assert run(sus, chord((36,), 0, 4), n=4)["segments"][0]["name"] == "Csus2"
    assert run(sus, chord((31,), 0, 4), n=4)["segments"][0]["name"] == "Gsus4"


def test_qualities_and_empty_beats():
    notes = chord((45, 48, 52, 55), 0, 4) + chord((40, 47), 8, 12)  # Am7, rest, E5
    r = run(notes, n=12)
    assert [g["name"] for g in r["segments"]] == ["Am7", "N", "E5"]


def test_the_cross_check_flags_a_bass_outside_the_chord():
    r = run(chord((48, 52, 55), 0, 4), chord((42,), 2, 4), n=4)  # F# under C
    assert [c["beat"] for c in r["checks"]] == [2, 3] and r["checks"][0]["bass"] == "F#"


def test_text_chart_has_sections_and_changes():
    notes = chord((48, 52, 55), 0, 6) + chord((43, 47, 50), 6, 8)
    g = grid(8)
    r = C.recognize(notes, [], C.beat_edges(g))
    txt = C.text_chart(r["segments"], g, [{"start_bar": 1, "label": "Verse"}])
    assert txt.splitlines()[0] == "[Verse] (bar 1)" and "| C . . . | . . G . |" in txt


def test_the_score_stage_writes_the_chart_and_names_chords_only_when_asked(tmp_path):
    import json

    import guitarpro as gp

    from test_parts_tracks import make_parts_job

    from bandscribe.tab import edits as E
    from bandscribe.tab import stage as T

    job, d = make_parts_job(tmp_path)
    chart = d["score"] / "chords" / "chords.txt"
    assert chart.is_file() and chart.read_text(encoding="utf-8").startswith("# 자동 추정 코드 차트")
    doc = json.loads((d["score"] / "chords" / "chords.json").read_text(encoding="utf-8"))
    assert doc["format"] == "bandscribe.chords/1" and doc["segments"]
    assert ' ch "' not in (d["score"] / "tab" / "score.alphatex").read_text(encoding="utf-8")  # default: chart only
    st = E.apply(E.load_base(job), [])
    params = {**T.score_params(None), "chords": {"in_score": True}}
    T.write_score(tmp_path / "on", song_key="s", meta={}, title="t", quant=st.quant, tab=st.tab, grid=st.grid,
                  sections=st.sections, params=params)
    tex = (tmp_path / "on" / "tab" / "score.alphatex").read_text(encoding="utf-8")
    named = [g["name"] for g in doc["segments"] if g["name"] != C.N]
    assert named and f'ch "{named[0]}"' in tex
    song = gp.parse(str(tmp_path / "on" / "tab" / "score.gp5"))
    names = {b.effect.chord.name for tr in song.tracks for m in tr.measures for v in m.voices for b in v.beats
             if b.effect.chord is not None}
    assert named[0] in names


def test_e33_options_tension_templates_and_the_bass_in_the_histogram():
    # D F A C E without the root on the guitar, D on the bass: Fmaj7-ish alone; Dm9 once the bass counts
    voicing = chord((53, 57, 60, 64), 0, 4)
    bass = chord((38,), 0, 4)
    assert run(voicing, bass, n=4)["segments"][0]["name"] == "Fmaj7"
    r = run(voicing, bass, n=4, extended=True, bass_weight=1.0)
    assert r["segments"][0]["name"] == "Dm9" and C.CLASS["m9"] == "min"
    assert run(voicing, bass, n=4)["segments"][0]["name"] == "Fmaj7"  # defaults unchanged (E31's recogniser)
