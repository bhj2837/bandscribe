"""MuScriptor event stream -> raw NoteSet (gtab.workers.amt_muscriptor pure helpers; no torch needed)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from gtab.workers import amt_muscriptor as W


@dataclass
class Start:  # mirrors muscriptor.events.NoteStartEvent
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class End:  # mirrors muscriptor.events.NoteEndEvent
    end_time: float
    start_event: Start


@dataclass
class Progress:  # mirrors muscriptor.events.ProgressEvent
    completed: int
    total: int


def _stream():
    a = Start(64, 1.5, 0, "clean_electric_guitar")
    b = Start(40, 0.25, 1, "electric_bass")
    c = Start(64, 0.25, 2, "acoustic_guitar")
    d = Start(36, 2.0, 3, "drums")
    e = Start(70, 2.5, 4, "distorted_electric_guitar")  # never ended
    return [Progress(0, 2), a, b, c, End(1.9, b), End(2.4, a), Progress(1, 2), d, End(2.01, d), End(0.9, c), e,
            Progress(2, 2)]


def test_assemble_pairs_sorts_and_closes_unterminated():
    progress = []
    notes = W.assemble_notes(_stream(), 3.0, on_progress=lambda c, t: progress.append((c, t)))
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert [(n["onset_s"], n["pitch"]) for n in notes] == [(0.25, 40), (0.25, 64), (1.5, 64), (2.0, 36), (2.5, 70)]
    by_pitch = {(n["onset_s"], n["pitch"]): n for n in notes}
    assert by_pitch[(0.25, 40)]["offset_s"] == 1.9 and by_pitch[(0.25, 40)]["instrument"] == "electric_bass"
    assert by_pitch[(0.25, 64)]["offset_s"] == 0.9
    assert by_pitch[(2.5, 70)]["offset_s"] == 3.0  # unterminated -> audio end
    assert by_pitch[(2.0, 36)]["instrument"] == "drums"


def test_zero_length_notes_get_minimum_duration_except_drums():
    s = Start(60, 1.0, 0, "acoustic_piano")
    k = Start(36, 1.0, 1, "drums")
    notes = W.assemble_notes([s, End(1.0, s), k, End(1.0, k)], 5.0)
    piano = next(n for n in notes if n["pitch"] == 60)
    drum = next(n for n in notes if n["pitch"] == 36)
    assert piano["offset_s"] == pytest.approx(1.01)
    assert drum["offset_s"] == 1.0


def test_same_onset_sort_key_is_total():
    s1 = Start(60, 1.0, 0, "organ")
    s2 = Start(60, 1.0, 1, "acoustic_piano")
    notes = W.assemble_notes([s1, s2, End(2.0, s1), End(2.0, s2)], 3.0)
    assert [n["instrument"] for n in notes] == ["acoustic_piano", "organ"]


def test_noteset_doc_shape():
    notes = W.assemble_notes(_stream(), 3.0)
    doc = W.noteset_doc(view_id="guitar_mono", view_sha256="ab" * 32,
                        backend={"name": "muscriptor", "version": "0.3.0"}, instruments=["acoustic_guitar"],
                        audio_s=3.0, notes=notes)
    assert doc["format"] == "gtab.notes/1" and doc["time_offset_applied_s"] == 0.0
    assert doc["backend"]["instruments"] == ["acoustic_guitar"]
    try:
        from gtab.schema.notes import NoteSet
    except ImportError:
        pytest.skip("P0 schema not present yet")
    NoteSet.model_validate(doc)


def test_sort_instruments_is_mt3_order_and_rejects_unknown():
    assert W.sort_instruments(["distorted_electric_guitar", "acoustic_guitar", "organ", "acoustic_guitar"]) == [
        "organ", "acoustic_guitar", "distorted_electric_guitar"]
    assert W.sort_instruments(None) is None
    with pytest.raises(ValueError):
        W.sort_instruments(["banjo"])


def test_worker_mt3_table_matches_schema():
    from gtab.amt.instruments import MT3_GROUPS

    assert W.MT3_IDS == dict(MT3_GROUPS)
    assert len(W.MT3_IDS) == 35


def test_rung_labels_and_ladder():
    model = {"size": "medium", "weights_path": "x", "fallback": [{"size": "small", "repo": "MuScriptor/muscriptor-small"}]}
    assert W.rung_labels(model, "upstream") == ["medium", "small"]
    assert W.rung_labels(model, "lean") == ["medium-lean", "small"]
    rungs = W.build_rungs(model, "upstream")
    assert rungs[-1]["label"] == "medium@cpu" and rungs[-1]["device"] == "cpu"
    assert W.rung_labels({"size": "medium", "fallback": []}, "upstream") == ["medium"]
