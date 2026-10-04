"""Dataset loaders on tiny synthetic trees that mimic each real layout (verified 2026-09-30 on the real files).

Checks the §6 conventions: string 1 = highest-pitched string, tuning[i] = open pitch of string i+1,
pitch = tuning + fret, notes sorted, families_present filled, paths relative to the dataset root.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fixtures.datasets_fakes import write_midi, write_wav
from bandscribe.datasets import store
from bandscribe.datasets.loaders import _common as C
from bandscribe.datasets.loaders import cambridge_mt, egdb, filobass, guitarset, idmt_bass, medleydb


def _check_track(t, root: Path) -> None:
    assert t.families_present
    for p in t.parts:
        assert "\\" not in p.audio and not Path(p.audio).is_absolute()
        assert (root / p.audio).is_file()
    for ns in t.notes.values():
        keys = [(n.onset_s, n.pitch, n.offset_s, n.instrument or "") for n in ns.notes]
        assert keys == sorted(keys)
        assert all(n.offset_s > n.onset_s for n in ns.notes)
    if t.string_fret and t.tuning:
        for lid, sf in t.string_fret.items():
            for (string, fret), n in zip(sf, t.notes[lid].notes):
                assert t.tuning[string - 1] + fret == n.pitch


# ------------------------------------------------------------------------------------------- GuitarSet


def _jams(notes_by_source: dict[int, list[tuple[float, float, float]]], columnar: bool = False) -> dict:
    anns = []
    for src, rows in notes_by_source.items():
        if columnar:
            data = {"time": [r[0] for r in rows], "duration": [r[1] for r in rows], "value": [r[2] for r in rows],
                    "confidence": [None] * len(rows)}
        else:
            data = [{"time": r[0], "duration": r[1], "value": r[2], "confidence": None} for r in rows]
        anns.append({"namespace": "note_midi", "annotation_metadata": {"data_source": str(src)}, "data": data})
    anns.append({"namespace": "beat_position", "annotation_metadata": {"data_source": ""}, "data": [
        {"time": 0.0, "duration": 0, "value": {"position": 1, "measure": 1, "num_beats": 4, "beat_units": 4}},
        {"time": 0.5, "duration": 0, "value": {"position": 2, "measure": 1, "num_beats": 4, "beat_units": 4}},
        {"time": 2.0, "duration": 0, "value": {"position": 1, "measure": 2, "num_beats": 4, "beat_units": 4}}]})
    anns.append({"namespace": "tempo", "annotation_metadata": {}, "data": [{"time": 0, "duration": 3, "value": 120.0}]})
    return {"annotations": anns, "file_metadata": {"duration": 3.0}, "sandbox": {}}


def test_guitarset_two_strings_numbering(tmp_path):
    root = tmp_path / "guitarset"
    # JAMS data_source 0 = low E (string 6), 5 = high E (string 1)
    j = _jams({0: [(0.10, 0.4, 43.02)], 5: [(0.05, 0.3, 66.97), (1.0, 0.5, 64.0)]})
    (root / "extracted" / "annotation").mkdir(parents=True)
    (root / "extracted" / "annotation" / "03_Rock1-90-C#_comp.jams").write_text(json.dumps(j), encoding="utf-8")
    write_wav(root / "extracted" / "audio_mono-mic" / "03_Rock1-90-C#_comp_mic.wav", 3.0)
    j2 = _jams({2: [(0.0, 0.5, 55.0)]}, columnar=True)
    (root / "extracted" / "annotation" / "01_BN2-100-Eb_solo.jams").write_text(json.dumps(j2), encoding="utf-8")
    tracks = list(guitarset.iter_tracks(root))
    assert [t.track_id for t in tracks] == ["01_BN2-100-Eb_solo", "03_Rock1-90-C#_comp"]
    t = guitarset.load_track(root, "03_Rock1-90-C#_comp")
    _check_track(t, root)
    assert t.split == "test" and t.group == "03" and t.families_present == ["guitar"]
    assert t.tuning == [64, 59, 55, 50, 45, 40]
    notes = t.notes["gtr"].notes
    assert [n.pitch for n in notes] == [67, 43, 64]
    assert t.string_fret["gtr"] == [(1, 3), (6, 3), (1, 0)]
    assert t.beats_s == [0.0, 0.5, 2.0] and t.downbeats_s == [0.0, 2.0]
    assert t.meta["pair"] == "03_Rock1-90-C#_solo" and t.meta["mode"] == "comp"
    assert t.mix == "extracted/audio_mono-mic/03_Rock1-90-C#_comp_mic.wav"
    solo = tracks[0]
    assert solo.split == "dev" and solo.string_fret["gtr"] == [(4, 5)] and solo.meta["audio_missing"]
    assert [x.track_id for x in guitarset.iter_tracks(root, split="dev")] == ["01_BN2-100-Eb_solo"]
    assert guitarset.track_ids(root) == [t.track_id for t in tracks]
    assert guitarset.track_ids(root, split="dev") == ["01_BN2-100-Eb_solo"]


# ------------------------------------------------------------------------------------------------ IDMT

_IDMT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<instrumentRecording><globalParameter><audioFileName>001.wav</audioFileName><instrument>EB</instrument>
<instrumentModel>X</instrumentModel><pickUpSetting>N</pickUpSetting><instrumentTuning>28,33,38,43</instrumentTuning>
<recordingArtist>A</recordingArtist></globalParameter><transcription>
<event><pitch>36</pitch><onsetSec>0.78</onsetSec><offsetSec>1.04</offsetSec><fretNumber>3</fretNumber>
<stringNumber>2</stringNumber><excitationStyle>FS</excitationStyle><expressionStyle>NO</expressionStyle>
<modulationFrequencyRange>0</modulationFrequencyRange><modulationFrequency>0</modulationFrequency></event>
<event><pitch>43</pitch><onsetSec>0</onsetSec><offsetSec>0.71</offsetSec><fretNumber>5</fretNumber>
<stringNumber>3</stringNumber><excitationStyle>MU</excitationStyle><expressionStyle>VI</expressionStyle>
<modulationFrequencyRange>25</modulationFrequencyRange><modulationFrequency>4</modulationFrequency></event>
</transcription></instrumentRecording>"""


def test_idmt_string_conversion(tmp_path):
    root = tmp_path / "idmt"
    base = root / "extracted" / "IDMT-SMT-BASS-SINGLE-TRACKS"
    (base / "annotation").mkdir(parents=True)
    (base / "annotation" / "001.xml").write_text(_IDMT_XML, encoding="utf-8")
    write_wav(base / "audio" / "001.wav", 1.5)
    (base / "misc" / "beats_csv").mkdir(parents=True)
    (base / "misc" / "beats_csv" / "001_beats.csv").write_text("0,1.1\n0.5,1.2\n1.0,1.3\n1.5,1.4\n2.0,2.1\n")
    t = idmt_bass.load_track(root, "001")
    _check_track(t, root)
    assert idmt_bass.track_ids(root) == ["001"] and idmt_bass.track_ids(root, split="test") == []
    assert t.tuning == [43, 38, 33, 28]            # highest first
    assert [n.pitch for n in t.notes["bass"].notes] == [43, 36]
    # IDMT string 3 (D, counted from the low E) -> §6 string 2; IDMT string 2 (A) -> §6 string 3
    assert t.string_fret["bass"] == [(2, 5), (3, 3)]
    assert t.meta["styles"][0]["expression"] == "VI"  # styles follow the sorted note order
    assert t.downbeats_s == [0.0, 2.0] and len(t.beats_s) == 5
    assert t.families_present == ["bass"]


# ------------------------------------------------------------------------------------------------ EGDB


def test_egdb_tracks_strings_split_and_tuning(tmp_path):
    root = tmp_path / "egdb"
    raw = root / "raw"
    # track index = string (1 = high E): string 1 note 64 fret 0, string 6 note 45 fret 5
    write_midi(raw / "audio_label" / "7.midi", [("1", [(0.5, 1.0, 64, 90)]), ("2", []), ("3", []), ("4", []),
                                                 ("5", []), ("6", [(0.0, 0.5, 45, 80)])], bpm=119)
    # clip 220 (test split): unnamed tracks; string 1 plays 62, below its open E4 -> a label error, not a tuning
    write_midi(raw / "audio_label" / "220.midi", [("", [(0.1, 0.2, 62, 90)]), ("", [(0.3, 0.4, 60, 90)])],
               name_tracks=False)
    for timbre in ("DI", "Plexi"):
        write_wav(raw / f"audio_{timbre}" / "7.wav", 1.2)
        write_wav(raw / f"audio_{timbre}" / "220.wav", 0.6)
    tracks = list(egdb.iter_tracks(root))
    assert [t.track_id for t in tracks] == ["di_007", "di_220", "plexi_007", "plexi_220"]
    t = egdb.load_track(root, "plexi_007")
    _check_track(t, root)
    assert t.split == "train" and t.group == "007" and t.meta["timbre_split"] == "test"
    assert t.meta["drive_guess"] == "distorted" and t.meta["string_label_errors"] == 0
    assert t.meta["scenario"] == "distortion" and egdb.scenario_of("JCjazz") == "clean"
    assert [n.pitch for n in t.notes["gtr"].notes] == [45, 64]
    assert t.string_fret["gtr"] == [(6, 5), (1, 0)]
    assert abs(t.notes["gtr"].notes[1].onset_s - 0.5) < 2e-3  # tempo map applied (119 BPM)
    t2 = egdb.load_track(root, "di_220")
    _check_track(t2, root)
    assert t2.split == "test" and t2.tuning == [64, 59, 55, 50, 45, 40]  # standard, never inferred
    assert t2.string_fret is None and t2.meta["string_label_errors"] == 1
    assert [n.pitch for n in t2.notes["gtr"].notes] == [62, 60]  # pitches (the GT) are kept
    assert t2.parts[0].kind == "di" and t2.meta["scenario"] == "di"
    assert [x.track_id for x in egdb.iter_tracks(root, split="test")] == ["di_220", "plexi_220"]
    assert egdb.track_ids(root) == [t.track_id for t in tracks]
    assert egdb.track_ids(root, split="test") == ["di_220", "plexi_220"]
    with pytest.raises(KeyError):
        egdb.load_track(root, "7")


# -------------------------------------------------------------------------------------------- FiloBass


def test_filobass_notes_and_downbeats(tmp_path):
    root = tmp_path / "filobass"
    base = root / "extracted" / "FiloBass_v1.0.0" / "FiloBass ISMIR Publication"
    write_midi(base / "midi_fully_aligned" / "Song-A.mid", [("", [(1.0, 1.5, 40, 100), (1.5, 2.0, 43, 100)])],
               bpm=120, name_tracks=False)
    write_midi(base / "midi_from_score" / "Song-A.mid", [("", [(0.0, 0.5, 40, 100)])])
    (base / "syncpoints").mkdir(parents=True)
    (base / "syncpoints" / "Song-A.json").write_text(json.dumps([[0, 0.5], [1, 2.5], [2, 4.5], [2, 5.0, 120]]))
    write_wav(base / "audio_bass_stems" / "Song-A.wav", 5.0, channels=2)
    t = filobass.load_track(root, "Song-A")
    _check_track(t, root)
    assert filobass.track_ids(root) == ["Song-A"]
    assert [n.pitch for n in t.notes["bass"].notes] == [40, 43]
    assert t.notes["bass"].notes[0].onset_s == pytest.approx(1.0, abs=1e-3)  # fully aligned MIDI, not score
    assert t.downbeats_s == [0.5, 2.5, 4.5]
    assert t.beats_s[:5] == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert t.string_fret is None and t.families_present == ["bass"] and t.parts[0].channels == 2


# ---------------------------------------------------------------------------------------- Cambridge-MT


def test_cambridge_lines_yaml_skeleton_and_loader(tmp_path):
    pytest.importorskip("yaml")
    root = tmp_path / "cambridge_mt"
    song = root / "extracted" / "Band_Song_Full" / "Band_Song_Full"
    names = ["01_Kick.wav", "02_BassDI.wav", "03_BassAmp.wav", "04_ElecGtr1a.wav", "05_ElecGtr1b.wav",
             "06_ElecGtr2_VoxSM57.wav", "07_ElecGtr2_OrangeMD421.wav", "08_Hammond.wav", "09_LeadVox.wav"]
    for n in names:
        write_wav(song / n, 0.3, channels=1 if "Bass" in n else 2)
    store.write_lines_skeleton(song, names, song="band_song", title="Band - Song")
    t = cambridge_mt.load_track(root, "band_song")
    _check_track(t, root)
    assert cambridge_mt.track_ids(root) == ["band_song"] and cambridge_mt.track_ids(root, split="test") == []
    lines = {ln.id: ln for ln in t.lines}
    assert set(lines) == {"G1", "G2", "B1"}
    assert lines["G1"].takes == [4, 5] and lines["G2"].takes == [6, 7] and lines["B1"].takes == [2, 3]
    takes = {p.id: p.take for p in t.parts}
    assert takes["p04"] == 1 and takes["p05"] == 2      # ElecGtr1a/1b: separate takes (double)
    assert takes["p06"] == 1 and takes["p07"] == 1      # two mics of one performance
    kinds = {p.id: p.kind for p in t.parts}
    assert kinds["p02"] == "di" and kinds["p08"] == "keys" and kinds["p01"] == "drums"
    assert t.families_present == ["guitar", "bass", "keys", "vocals", "drums"]
    assert t.tier == "B" and t.meta["reviewed"] is False and t.notes == {}
    # an existing (user-edited) lines.yaml is never overwritten
    text = (song / "lines.yaml").read_text(encoding="utf-8").replace("reviewed: false", "reviewed: true")
    (song / "lines.yaml").write_text(text, encoding="utf-8")
    store.write_lines_skeleton(song, names, song="band_song")
    assert cambridge_mt.load_track(root, "band_song").meta["reviewed"] is True


def test_cambridge_rejects_bad_labels(tmp_path):
    pytest.importorskip("yaml")
    root = tmp_path / "cambridge_mt"
    song = root / "local" / "x"
    write_wav(song / "a.wav", 0.1)
    (song / "lines.yaml").write_text("format: bandscribe.cmt_lines/1\nsong: x\nfamilies_present: [kazoo]\nparts:\n"
                                     "  - {id: p1, file: a.wav, kind: guitar, line: G1}\nlines: []\n", encoding="utf-8")
    with pytest.raises(cambridge_mt.LinesYamlError):
        cambridge_mt.load_track(root, "x")


# -------------------------------------------------------------------------------------------- MedleyDB


def test_medleydb_local_copy(tmp_path):
    pytest.importorskip("yaml")
    root = tmp_path / "medleydb"
    song = root / "local" / "Audio" / "Artist_Song"
    stems = song / "Artist_Song_STEMS"
    for i in range(1, 5):
        write_wav(stems / f"Artist_Song_STEM_0{i}.wav", 0.2, channels=2)
    write_wav(song / "Artist_Song_MIX.wav", 0.2, channels=2)
    (song / "Artist_Song_METADATA.yaml").write_text(
        "artist: Artist\ntitle: Song\nmix_filename: Artist_Song_MIX.wav\nstem_dir: Artist_Song_STEMS\n"
        "has_bleed: 'no'\ngenre: Rock\nstems:\n"
        "  S01: {filename: Artist_Song_STEM_01.wav, instrument: distorted electric guitar, component: melody}\n"
        "  S02: {filename: Artist_Song_STEM_02.wav, instrument: clean electric guitar, component: ''}\n"
        "  S03: {filename: Artist_Song_STEM_03.wav, instrument: electric bass, component: bass}\n"
        "  S04: {filename: Artist_Song_STEM_04.wav, instrument: drum set, component: ''}\n", encoding="utf-8")
    t = medleydb.load_track(root, "Artist_Song")
    _check_track(t, root)
    assert medleydb.track_ids(root) == ["Artist_Song"]
    assert [ln.id for ln in t.lines] == ["G1", "G2", "B1"]
    assert t.families_present == ["guitar", "bass", "drums"]
    assert t.mix.endswith("Artist_Song_MIX.wav") and t.meta["has_bleed"] is False
    assert medleydb.family_of("bass drum") == "drums" and medleydb.family_of("synthesizer") == "synth"


# --------------------------------------------------------------------------------------------- helpers


def test_midi_reader_tempo_change_and_overlaps(tmp_path):
    from fixtures.datasets_fakes import write_midi as wm

    p = tmp_path / "t.mid"
    wm(p, [("x", [(0.0, 1.0, 60, 90), (0.5, 0.75, 60, 80)])], bpm=60)
    m = C.read_midi(p)
    assert m.ticks_per_beat == 480 and m.tempos == [(0, 1_000_000)]
    assert [(round(n.onset, 3), round(n.offset, 3)) for n in m.notes] == [(0.0, 0.75), (0.5, 1.0)]


def test_build_noteset_fixes_zero_length(monkeypatch):
    ns, ordered = C.build_noteset("v", [C.RawNote(1.0, 1.0, 60), C.RawNote(0.5, 0.7, 62)], duration=2.0, source="t")
    assert [n.pitch for n in ns.notes] == [62, 60]
    assert ns.notes[1].offset_s == pytest.approx(1.001)
    assert ns.backend.name == "gt" and ns.backend.params["zero_length_fixed"] == 1
