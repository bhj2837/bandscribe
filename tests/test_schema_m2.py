"""M1a/M2 file formats (M1_M2_SPEC 6): round trips, invariants and TempoMap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

import gtab.schema as S
from gtab.schema import (
    FAMILIES,
    MT3_GROUPS,
    BackendInfo,
    BeatsRaw,
    DatasetTrack,
    Grid,
    GtLine,
    GtNote,
    GtNotes,
    GtSpec,
    Instrumentation,
    NoteEvent,
    NoteSet,
    Part,
    PitchBend,
    Prediction,
    PredNote,
    RepeatMap,
    Sections,
    TempoMap,
    notes_to_arrays,
)


def roundtrip(m: BaseModel) -> None:
    cls = type(m)
    js = m.model_dump_json()
    again = cls.model_validate_json(js)
    assert again == m
    assert cls.model_validate(json.loads(js)) == m
    assert again.model_dump_json() == js  # stable bytes


# ------------------------------------------------------------------------------------------- notes


def noteset(**kw) -> NoteSet:
    notes = kw.pop("notes", None) or [
        NoteEvent(onset_s=0.5, offset_s=1.0, pitch=40, instrument="clean_electric_guitar", confidence=0.9),
        NoteEvent(onset_s=0.5, offset_s=1.0, pitch=52, instrument="clean_electric_guitar"),
        NoteEvent(onset_s=1.25, offset_s=1.5, pitch=45, amplitude=0.3,
                  bend=PitchBend(t_s=[1.25, 1.3], cents=[0.0, 33.3])),
    ]
    return NoteSet(view="guitar_mono", view_sha256="ab" * 32,
                   backend=BackendInfo(name="muscriptor", version="0.3.0", params={"beam_size": 1},
                                       instruments=["acoustic_guitar"]),
                   audio_duration_s=10.0, notes=notes, **kw)


def test_noteset_roundtrip_and_defaults() -> None:
    ns = noteset()
    roundtrip(ns)
    assert ns.format == "gtab.notes/1" and ns.time_offset_applied_s == 0.0
    assert NoteSet.model_validate({"view": "x", "backend": {"name": "gt"}, "notes": []}).backend.params == {}


def test_noteset_must_be_sorted() -> None:
    a = NoteEvent(onset_s=1.0, offset_s=2.0, pitch=50)
    b = NoteEvent(onset_s=0.5, offset_s=2.0, pitch=50)
    with pytest.raises(ValidationError, match="sorted"):
        noteset(notes=[a, b])
    # ties broken by pitch, then offset, then instrument ("" for None)
    c1 = NoteEvent(onset_s=1.0, offset_s=2.0, pitch=50, instrument=None)
    c2 = NoteEvent(onset_s=1.0, offset_s=2.0, pitch=50, instrument="drums")
    noteset(notes=[c1, c2])
    with pytest.raises(ValidationError):
        noteset(notes=[c2, c1])
    assert S.sort_notes([c2, a, b, c1]) == [b, a, c1, c2]  # a == c1 (same fields); stable sort


def test_note_event_invariants() -> None:
    with pytest.raises(ValidationError, match="offset_s"):
        NoteEvent(onset_s=1.0, offset_s=0.5, pitch=40)
    with pytest.raises(ValidationError):
        NoteEvent(onset_s=0.0, offset_s=0.5, pitch=128)
    with pytest.raises(ValidationError):
        NoteEvent(onset_s=float("nan"), offset_s=0.5, pitch=40)  # allow_inf_nan=False
    with pytest.raises(ValidationError):
        NoteEvent(onset_s=0.0, offset_s=0.5, pitch=40, colour="red")  # extra="forbid"
    with pytest.raises(ValidationError, match="same length"):
        PitchBend(t_s=[0.0, 0.1], cents=[1.0])


def test_notes_to_arrays() -> None:
    iv, p = notes_to_arrays(noteset())
    assert iv.shape == (3, 2) and p.tolist() == [40, 52, 45] and iv.dtype == np.float64
    iv2, p2 = notes_to_arrays(noteset(), instruments={"clean_electric_guitar"})
    assert p2.tolist() == [40, 52]
    iv3, p3 = notes_to_arrays(noteset(notes=[NoteEvent(onset_s=0, offset_s=0, pitch=1)]), instruments=set())
    assert iv3.shape == (0, 2) and p3.shape == (0,)


# -------------------------------------------------------------------------------------------- grid


def grid_doc() -> dict:
    beats = []
    t = 0.37
    bar_beats = [(0, 2)] + [(b, 4) for b in range(1, 4)]
    for bar, n in bar_beats:
        for k in range(1, n + 1):
            beats.append({"t_s": round(t, 6), "t_raw_s": round(t, 6), "bar": bar, "beat": k,
                          "is_downbeat": k == 1 and bar > 0, "shift_ms": 0.0, "activation": 0.8})
            t += 0.5
    bars = [{"bar": bar, "start_s": 0.37 + 0.5 * i, "end_s": 0.37 + 0.5 * (i + n), "n_beats": n, "numerator": 4,
             "denominator": 4, "beat_unit": "quarter", "pickup": bar == 0}
            for bar, n, i in [(0, 2, 0), (1, 4, 2), (2, 4, 6), (3, 4, 10)]]
    return {"format": "gtab.grid/1", "tpb": 48, "duration_s": 8.0, "beats": beats, "bars": bars,
            "tempo": [{"start_bar": 0, "end_bar": 4, "bpm": 120.0}],
            "meter": [{"start_bar": 0, "end_bar": 4, "numerator": 4, "denominator": 4, "beat_unit": "quarter"}],
            "beat_unit": "quarter", "pickup": {"present": True, "beats": 2},
            "hypotheses": {"tempo_octave": {"decision": "keep"}}, "confidence": {"beats": 0.9, "downbeats": None},
            "params": {"refine_window_ms": 35.0}}


def test_grid_and_beats_roundtrip() -> None:
    g = Grid.model_validate(grid_doc())
    roundtrip(g)
    roundtrip(BeatsRaw(format="gtab.beats/1", tracker="beat_this/final0", checkpoint_sha256=None, dbn=False, fps=50.0,
                       beats_s=[0.5, 1.0], downbeats_s=[0.5], duration_s=2.0))
    with pytest.raises(ValidationError):
        Grid.model_validate({**grid_doc(), "format": "gtab.grid/2"})
    bad = grid_doc()
    bad["beats"][3]["t_s"] = bad["beats"][2]["t_s"]
    with pytest.raises(ValidationError, match="strictly increasing"):
        Grid.model_validate(bad)
    bad2 = grid_doc()
    bad2["tempo"] = [{"start_bar": 2, "end_bar": 2, "bpm": 120.0}]
    with pytest.raises(ValidationError, match="half-open"):
        Grid.model_validate(bad2)


def _map(times, bars, beats, bpb, tpb=48) -> TempoMap:
    return TempoMap(times, bars, beats, bpb, tpb)


def test_tempomap_round_trip_below_1e9() -> None:
    rng = np.random.default_rng(1)
    times = np.cumsum(rng.uniform(0.35, 0.65, 41))
    tm = _map(times, [1 + i // 4 for i in range(41)], [1 + i % 4 for i in range(41)], {b: 4 for b in range(1, 12)})
    t = np.concatenate([rng.uniform(times[0] - 3, times[-1] + 3, 2000), times])
    bar, tick = tm.time_to_bar_tick(t)
    assert bar.dtype.kind == "i"
    back = tm.bar_tick_to_time(bar, tick)
    assert np.max(np.abs(back - t)) < 1e-9


def test_tempomap_tempo_change_and_ticks_per_second() -> None:
    # 4 beats at 120 BPM, then 4 beats at 60 BPM
    times = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    tm = _map(times, [1, 1, 1, 1, 2, 2, 2, 2, 3], [1, 2, 3, 4, 1, 2, 3, 4, 1], {1: 4, 2: 4, 3: 4})
    np.testing.assert_allclose(tm.bar_tick_to_time([1, 1, 2, 2, 3], [0, 96, 0, 72, 0]), [0, 1.0, 2.0, 3.5, 6.0])
    np.testing.assert_allclose(tm.ticks_per_second([0.25, 2.5]), [96.0, 48.0])
    bar, tick = tm.time_to_bar_tick([0.75, 2.75])
    assert bar.tolist() == [1, 2] and tick.tolist() == pytest.approx([72.0, 36.0])
    np.testing.assert_allclose(tm.time_to_beat([0.0, 0.25, 2.5, 6.0]), [0.0, 0.5, 4.5, 8.0])
    np.testing.assert_allclose(tm.beat_to_time([0.5, 4.5]), [0.25, 2.5])


def test_tempomap_pickup_bar_zero_from_grid() -> None:
    tm = TempoMap.from_grid(Grid.model_validate(grid_doc()))
    assert tm.tpb == 48
    np.testing.assert_allclose(tm.bar_tick_to_time([0, 0, 1, 2], [0, 48, 0, 0]), [0.37, 0.87, 1.37, 3.37])
    bar, tick = tm.time_to_bar_tick([0.37, 0.87, 1.37, 1.62])
    assert bar.tolist() == [0, 0, 1, 1] and tick.tolist() == pytest.approx([0, 48, 0, 24])


def test_tempomap_extrapolation_outside_listed_beats() -> None:
    times = [1.0, 1.5, 2.0, 2.5, 3.0]
    tm = _map(times, [1, 1, 1, 1, 2], [1, 2, 3, 4, 1], {1: 4, 2: 4})
    # before the first beat: first interval; after the last: last interval, bars continue with their lengths
    np.testing.assert_allclose(tm.time_to_beat([0.0, 4.0]), [-2.0, 6.0])
    np.testing.assert_allclose(tm.bar_tick_to_time([0, 3, 4], [0, 0, 0]), [-1.0, 5.0, 7.0])
    bar, tick = tm.time_to_bar_tick([0.0, 5.25, 7.0])
    assert bar.tolist() == [0, 3, 4] and tick.tolist() == pytest.approx([96.0, 24.0, 0.0])
    # a note end tick past its bar continues into the next beats
    np.testing.assert_allclose(tm.bar_tick_to_time(1, 240), 3.5)


def test_tempomap_grid_starting_mid_bar() -> None:
    # first listed beat is beat 2 of bar 1: bar 1 starts one (extrapolated) beat earlier
    tm = _map([0.5, 1.0, 1.5, 2.0], [1, 1, 1, 2], [2, 3, 4, 1], {1: 4, 2: 4})
    np.testing.assert_allclose(tm.bar_tick_to_time([1, 2], [0, 0]), [0.0, 2.0])


def test_tempomap_from_file_and_dict(tmp_path: Path) -> None:
    doc = {"format": "gtab.tempomap/1", "source": "dtw", "tpb": 48,
           "beats": [{"t_s": 0.0, "bar": 1, "beat": 1}, {"t_s": 0.5, "bar": 1, "beat": 2},
                     {"t_s": 1.0, "bar": 2, "beat": 1}],
           "beats_per_bar": {"1": 2, "2": 2}}
    (tmp_path / "tempo_map.json").write_text(json.dumps(doc), encoding="utf-8")
    tm = TempoMap.from_file(tmp_path / "tempo_map.json")
    np.testing.assert_allclose(tm.bar_tick_to_time([2], [0]), [1.0])
    assert tm.to_dict(source="dtw") == doc
    (tmp_path / "grid.json").write_text(json.dumps(grid_doc()), encoding="utf-8")
    assert TempoMap.from_file(tmp_path / "grid.json").bar_tick_to_time(1, 0) == pytest.approx(1.37)
    with pytest.raises(ValueError):
        TempoMap.from_dict({"format": "other"})


def test_tempomap_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        _map([0.0], [1], [1], {1: 4})
    with pytest.raises(ValueError):
        _map([0.0, 0.0, 1.0], [1, 1, 1], [1, 2, 3], {1: 4})
    with pytest.raises(ValueError):
        _map([0.0, 0.5, 1.0], [2, 1, 1], [1, 1, 2], {1: 4, 2: 4})


# ----------------------------------------------------------------------------------------- sections


def test_sections_and_repeat_map_roundtrip() -> None:
    sec = Sections.model_validate({"format": "gtab.sections/1", "method": "laplacian", "params": {}, "sections": [
        {"id": "s1", "label": "A", "start_bar": 1, "end_bar": 9, "start_s": 0.0, "end_s": 16.0, "confidence": 0.8}]})
    roundtrip(sec)
    rm = RepeatMap.model_validate({"format": "gtab.repeat_map/1", "params": {}, "groups": [
        {"label": "A", "reference": "s1", "occurrences": [{"section": "s1", "start_bar": 1, "n_bars": 8, "offset_s": 0.0}]}],
        "bars": [{"bar": 1, "matches": [{"bar": 17, "score": 0.9, "offset_s": 0.01}]}]})
    roundtrip(rm)
    with pytest.raises(ValidationError, match="half-open"):
        Sections.model_validate({"format": "gtab.sections/1", "method": "x", "params": {}, "sections": [
            {"id": "s", "label": "A", "start_bar": 5, "end_bar": 5, "start_s": 0, "end_s": 1, "confidence": 1}]})


# ---------------------------------------------------------------------------------- instrumentation


def test_mt3_groups_and_families() -> None:
    assert len(MT3_GROUPS) == 35 and sorted(MT3_GROUPS.values()) == list(range(34)) + [36]
    flat = [c for cls in FAMILIES.values() for c in cls]
    assert sorted(flat) == sorted(MT3_GROUPS)  # every class in exactly one family
    assert S.GUITAR_CLASSES == ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar")
    assert S.BASS_CLASSES == ("acoustic_bass", "electric_bass")
    assert S.FAMILY_OF["synth_strings"] == "synth"


def instrumentation_doc() -> dict:
    classes = {c: {"p": 0.9 if c in S.GUITAR_CLASSES else 0.01, "present": c in S.GUITAR_CLASSES,
                   "sources": {"stem_energy": {"ratio": 0.5}}} for c in MT3_GROUPS}
    return {"format": "gtab.instrumentation/1", "profile": "quality", "classes": classes,
            "families": {f: 0.5 for f in FAMILIES},
            "masks": {"guitar_mono": list(S.GUITAR_CLASSES), "mix_mono": None},
            "per_section": [{"section": "s1", "family_active": {"guitar": 1.0}}],
            "thresholds": {"threshold": 0.5}, "hints": {"instruments": "auto"}, "warnings": []}


def test_instrumentation_roundtrip_and_checks() -> None:
    roundtrip(Instrumentation.model_validate(instrumentation_doc()))
    d = instrumentation_doc()
    d["classes"] = dict(reversed(list(d["classes"].items())))
    with pytest.raises(ValidationError, match="MT3_GROUPS order"):
        Instrumentation.model_validate(d)
    d = instrumentation_doc()
    del d["classes"]["organ"]
    with pytest.raises(ValidationError, match="organ"):
        Instrumentation.model_validate(d)
    d = instrumentation_doc()
    d["masks"]["guitar_mono"] = ["guitar"]
    with pytest.raises(ValidationError, match="unknown MT3"):
        Instrumentation.model_validate(d)
    d = instrumentation_doc()
    d["families"]["kazoo"] = 1.0
    with pytest.raises(ValidationError, match="kazoo"):
        Instrumentation.model_validate(d)


# --------------------------------------------------------------------------------------------- gt


def gt_spec_doc() -> dict:
    return {
        "format": "gtab.gt/1", "song_id": "sc_test-01", "title": "青春コンプレックス", "artist": "結束バンド",
        "audio": {"song_key": "f-0123", "note": ""}, "version_flags": {"version": "album", "trimmed": False, "note": ""},
        "reference": {"file": "ref.gp5", "format": "gp5", "expanded_bars": 80, "bar_order": None, "encoding": None},
        "families_present": ["guitar", "bass", "vocals", "drums"],
        "tracks": [{"ref_track": 1, "name": "기타 1", "kind": "guitar", "line": "L1", "tuning": [64, 59, 55, 50, 45, 40],
                    "capo": 0, "player": None},
                   {"ref_track": 3, "name": "Drums", "kind": "other", "line": None, "tuning": None}],
        "lines": [{"id": "L1", "name": "Lead", "kind": "guitar", "takes": [1],
                   "relation": {"double": False, "unison_with": [], "twin_with": None, "overdub": False}}],
        "players": [],
        "sections": [{"id": "v1", "start_bar": 17, "end_bar": 25, "scenario": "A", "roles": {"L1": "lead"},
                      "split": "dev"}],
        "alignment": {"method": None, "approved": False, "approved_utc": None, "minutes_spent": None},
    }


def test_gt_spec_roundtrip_and_checks() -> None:
    spec = GtSpec.model_validate(gt_spec_doc())
    roundtrip(spec)
    assert spec.tracks[1].capo == 0 and spec.lines[0].relation["double"] is False
    for key, val, match in [("song_id", "SC 01", "song_id"), ("families_present", ["guitar", "piano"], "piano")]:
        with pytest.raises(ValidationError, match=match):
            GtSpec.model_validate({**gt_spec_doc(), key: val})
    d = gt_spec_doc()
    d["sections"][0]["end_bar"] = 17
    with pytest.raises(ValidationError, match="half-open"):
        GtSpec.model_validate(d)
    d = gt_spec_doc()
    d["sections"][0]["roles"] = {"L1": "solo"}
    with pytest.raises(ValidationError):
        GtSpec.model_validate(d)
    d = gt_spec_doc()
    d["lines"].append(dict(d["lines"][0]))
    with pytest.raises(ValidationError, match="duplicate"):
        GtSpec.model_validate(d)


def test_gt_notes_roundtrip() -> None:
    gn = GtNotes(format="gtab.gtnotes/1", song_id="x", tpb=48,
                 bars=[{"bar": 1, "numerator": 4, "denominator": 4, "beat_unit": "quarter", "nominal_bpm": 120.0}],
                 notes=[GtNote(line="L1", ref_track=1, bar=1, tick=24.0, dur_ticks=48.0, pitch=64, string=1, fret=0,
                               shared=True),
                        GtNote(line="L1", ref_track=1, bar=1, tick=72.0, dur_ticks=24.0, pitch=60, string=None,
                               fret=None, onset_s=1.5, offset_s=1.75)])
    roundtrip(gn)
    with pytest.raises(ValidationError):
        GtNote(line="L1", ref_track=1, bar=1, tick=0, dur_ticks=1, pitch=60)  # string/fret are required keys
    with pytest.raises(ValidationError):
        GtNote(line="L1", ref_track=1, bar=1, tick=0, dur_ticks=1, pitch=60, string=0, fret=0)


# ------------------------------------------------------------------------------------------ dataset


def dataset_doc() -> dict:
    ns = noteset(notes=[NoteEvent(onset_s=0.1, offset_s=0.4, pitch=40), NoteEvent(onset_s=0.5, offset_s=0.9, pitch=45)])
    return {"format": "gtab.dataset_track/1", "dataset": "guitarset", "track_id": "00_BN1-129-Eb_comp", "split": None,
            "group": "00", "tier": "C", "license": "CC BY 4.0", "mix": "audio_mono-mic/00_BN1-129-Eb_comp_mic.wav",
            "parts": [{"id": "gtr", "kind": "guitar", "audio": "audio_mono-mic/x.wav", "channels": 1, "line": "L1"}],
            "lines": [GtLine(id="L1", name="Guitar", kind="guitar", takes=[1]).model_dump()],
            "families_present": ["guitar"],
            "notes": {"L1": json.loads(ns.model_dump_json())},
            "string_fret": {"L1": [(6, 0), (5, 0)]},
            "beats_s": [0.0, 0.5], "downbeats_s": [0.0], "tuning": [64, 59, 55, 50, 45, 40], "meta": {}}


def test_dataset_track_roundtrip_and_checks() -> None:
    t = DatasetTrack.model_validate(dataset_doc())
    roundtrip(t)
    assert isinstance(t.parts[0], Part) and t.string_fret["L1"][0] == (6, 0)
    d = dataset_doc()
    d["string_fret"] = {"L1": [(6, 0)]}
    with pytest.raises(ValidationError, match="same order"):
        DatasetTrack.model_validate(d)
    d = dataset_doc()
    d["string_fret"] = {"L2": []}
    with pytest.raises(ValidationError, match="no notes"):
        DatasetTrack.model_validate(d)
    with pytest.raises(ValidationError, match="kazoo"):
        DatasetTrack.model_validate({**dataset_doc(), "families_present": ["kazoo"]})
    with pytest.raises(ValidationError):
        DatasetTrack.model_validate({**dataset_doc(), "tier": "A"})


# ------------------------------------------------------------------------------------------- evalio


def test_prediction_roundtrip() -> None:
    p = Prediction(format="gtab.prediction/1", system="job:amt_gtr", item="guitarset:00_BN1-129-Eb_comp",
                   lines=[{"id": "L1"}],
                   notes=[PredNote(onset_s=0.1, offset_s=0.2, pitch=40, line="L1", posterior=0.7),
                          PredNote(onset_s=0.3, offset_s=0.4, pitch=41, line="unassigned", bar=1, tick=12.0)])
    roundtrip(p)
    assert p.meta == {}
    with pytest.raises(ValidationError):
        PredNote(onset_s=0.3, offset_s=0.2, pitch=41, line="L1")


def test_file_models_carry_format_and_no_schema_field() -> None:
    for cls in (NoteSet, BeatsRaw, Grid, Sections, RepeatMap, Instrumentation, GtSpec, GtNotes, DatasetTrack, Prediction):
        assert "format" in cls.model_fields, cls
        assert "schema" not in cls.model_fields, cls
