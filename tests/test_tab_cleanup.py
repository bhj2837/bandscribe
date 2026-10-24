"""Note clean-up rules (E39 / E40): short-note drop and same-pitch duplicates within a part."""

from __future__ import annotations

from bandscribe.tab import cleanup as C


def _n(on, off, pitch, line=None, inst="clean_electric_guitar"):
    return {"onset_s": on, "offset_s": off, "pitch": pitch, "line": line, "instrument": inst}


def test_drop_short():
    notes = [_n(0.0, 0.01, 40), _n(1.0, 1.3, 40), _n(2.0, 2.049, 41), _n(3.0, None, 42)]
    kept, dropped = C.drop_short(notes, 0.05)
    assert [n["onset_s"] for n in kept] == [1.0] and len(dropped) == 3
    assert C.drop_short(notes, 0.0) == ([dict(n) for n in notes], [])


def test_dedupe_within_line_keeps_the_earlier_note_and_the_later_end():
    notes = [_n(1.000, 1.20, 64, "1", "acoustic_guitar"), _n(1.010, 1.50, 64, "1"),  # one note, two classes
             _n(1.005, 1.30, 64, "2"),                                               # the other part doubles it
             _n(1.200, 1.40, 64, "1"),                                               # a real repeat
             _n(1.000, 1.20, 67, "1")]                                               # another pitch
    kept, dropped = C.dedupe_within_line(notes, 0.03)
    assert len(dropped) == 1 and dropped[0]["onset_s"] == 1.010
    first = next(n for n in kept if n["onset_s"] == 1.000 and n["pitch"] == 64)
    assert first["offset_s"] == 1.50 and first["instrument"] == "acoustic_guitar"
    assert sum(1 for n in kept if n["pitch"] == 64) == 3
    # chained duplicates collapse onto the first; notes without a part dedupe among themselves
    chain = [_n(0.0, 0.1, 50), _n(0.02, 0.2, 50), _n(0.04, 0.3, 50), _n(0.5, 0.6, 50)]
    k2, d2 = C.dedupe_within_line(chain, 0.03)
    assert [n["onset_s"] for n in k2] == [0.0, 0.04, 0.5] and len(d2) == 1
    assert C.dedupe_within_line(chain, 0.0)[1] == []


def test_quant_stage_applies_the_cleanup(tmp_path):
    import json

    from bandscribe.tab import stage as TS

    parts = tmp_path / "parts"
    parts.mkdir()
    gtr = [_n(1.0, 1.3, 64, "1", "acoustic_guitar"), _n(1.01, 1.4, 64, "1"), _n(2.0, 2.01, 60, "1")]
    (parts / TS.LINES_JSON).write_text(json.dumps({"notes": gtr}), encoding="utf-8")
    bass_dir = tmp_path / "bass"
    bass_dir.mkdir()
    (bass_dir / TS.BASS_JSON).write_text(json.dumps({"notes": [_n(0.5, 0.51, 40), _n(1.0, 1.5, 40)]}), encoding="utf-8")
    dd = {"parts": parts, "bass": bass_dir, "notes": tmp_path / "notes"}
    raw = TS.quant_tracks(dd)
    assert len(raw["guitar_all"]) == 3 and len(raw[TS.BASS]) == 2
    clean = TS.quant_tracks(dd, {"dedupe_s": 0.03, "min_dur_s": 0.05})
    assert [n["onset_s"] for n in clean["guitar_all"]] == [1.0] and clean["guitar_all"][0]["offset_s"] == 1.4
    assert [n["onset_s"] for n in clean[TS.BASS]] == [1.0]
    # the defaults (E39 / E40 adopted) put the clean-up into the quant params; off leaves the old params
    from bandscribe.config import load_config

    p = TS.quant_params(load_config())
    assert p["cleanup"] == {"dedupe_s": 0.03, "min_dur_s": 0.05}
    assert "cleanup" not in TS.quant_params({"tab.dedupe_s": 0.0, "tab.min_dur_s": 0.0})


def test_cleanup_keeps_the_source_index_for_parts_and_repeat_links():
    # the part-count switch and the repeat links find notes by quant's ``i`` = the parts stage's index
    from test_tab_quantize import grid_doc

    from bandscribe.tab import quantize as Q

    notes = [_n(0.0, 0.3, 60, "1"), _n(0.01, 0.4, 60, "1"), _n(0.5, 0.51, 62, "1"), _n(1.0, 1.3, 64, "2")]
    dd_notes = [{**n, Q.SRC_INDEX: k} for k, n in enumerate(notes)]
    kept = C.dedupe_within_line(C.drop_short(dd_notes, 0.05)[0], 0.03)[0]
    assert [n[Q.SRC_INDEX] for n in kept] == [0, 3]
    q = Q.quantize({"guitar_all": kept}, grid_doc([0.5 * b for b in range(16)]))
    assert sorted(n["i"] for n in Q.notes_of(q, "guitar_all")) == [0, 3]
    q2 = Q.quantize({"guitar_all": notes}, grid_doc([0.5 * b for b in range(16)]))
    assert sorted(n["i"] for n in Q.notes_of(q2, "guitar_all")) == [0, 1, 2, 3]  # no filter: the position, as before


def test_editor_base_applies_the_runs_cleanup(monkeypatch):
    from bandscribe.tab import edits as ED
    from bandscribe.tab import stage as TS

    seen = {}

    def fake_quant_tracks(dirs, cleanup=None):
        seen["cleanup"] = cleanup
        raise KeyError("stop here")

    monkeypatch.setattr(TS, "quant_tracks", fake_quant_tracks)
    monkeypatch.setattr(ED, "export_score_key", lambda job: "s" * 12)
    manifests = {"score": {"deps": {"tab": "t" * 12, "quant": "q" * 12, "grid": "g" * 12, "sections": "x" * 12}},
                 "tab": {"params": {}}, "quant": {"deps": {"notes": "n" * 12, "parts": "p" * 12},
                                                  "params": {"cleanup": {"dedupe_s": 0.03, "min_dur_s": 0.05}}}}
    monkeypatch.setattr(ED, "_manifest", lambda d: manifests[d.parent.name])
    monkeypatch.setattr(ED.atomic, "read_json", lambda p: {"sections": [], "bars": []})
    import pytest

    with pytest.raises(ED.EditError):
        ED.load_base(__import__("pathlib").Path("D:/nonexistent_job"))
    assert seen["cleanup"] == {"dedupe_s": 0.03, "min_dur_s": 0.05}
