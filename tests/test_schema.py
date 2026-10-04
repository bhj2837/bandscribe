from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

import bandscribe
from bandscribe.schema import (
    NOTE_FLAGS,
    Confidence,
    Line,
    Note,
    RoleSegment,
    Score,
    SongMeta,
    export_json_schema,
    load_score,
    save_score,
)


def make_score() -> Score:
    return Score(
        song=SongMeta(
            song_key="yt-YE7VzlLtp-4",
            title="빅 벅 버니",
            artist=None,
            duration_s=33.5,
            source_kind="youtube",
            source_ref="https://www.youtube.com/watch?v=YE7VzlLtp-4",
        ),
        lines=[
            Line(
                id="g1",
                name="Guitar 1 (L)",
                kind="guitar",
                role_segments=[RoleSegment(start_s=0.0, end_s=12.5, role="rhythm")],
                tuning=[40, 45, 50, 55, 59, 64],
                capo=2,
                notes=[
                    Note(
                        onset_s=0.5,
                        offset_s=0.75,
                        pitch=64,
                        velocity=90,
                        string=1,
                        fret=0,
                        techniques=["palm_mute"],
                        confidence=Confidence(pitch=0.9, onset=0.8, part=0.7, fret=None),
                        sources=["muscriptor_medium:view_L"],
                        part_posterior={"g1": 0.7, "g2": 0.3},
                        flags={"double": True, "low_conf": False},
                    ),
                    Note(onset_s=1.0, offset_s=1.0, pitch=52),  # zero length allowed, no tab position yet
                ],
            ),
            Line(id="b", name="Bass", kind="bass", tuning=[28, 33, 38, 43]),
            Line(id="u", name="Unassigned", kind="unassigned"),
        ],
    )


def test_round_trip_json_equal() -> None:
    score = make_score()
    text = score.model_dump_json()
    again = Score.model_validate_json(text)
    assert again == score
    assert json.loads(text)["song"]["title"] == "빅 벅 버니"


def test_defaults() -> None:
    score = Score(song=SongMeta(song_key="f-0123456789abcdef", duration_s=1.0, source_kind="file", source_ref="a.flac"))
    assert score.schema_version == 0
    assert score.lines == []
    assert score.bandscribe_version == bandscribe.__version__
    assert score.created_utc.endswith("Z") and len(score.created_utc) == 20
    note = Note(onset_s=0, offset_s=1, pitch=60)
    assert note.confidence == Confidence()
    assert note.flags == {} and note.techniques == [] and note.string is None
    assert "unison_undecided" in NOTE_FLAGS


def test_save_and_load(tmp_path: Path) -> None:
    score = make_score()
    path = tmp_path / "score" / "score.json"
    save_score(path, score)
    assert load_score(path) == score
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"onset_s": 1.0, "offset_s": 0.5, "pitch": 60},  # ends before it starts
        {"onset_s": -0.1, "offset_s": 0.5, "pitch": 60},
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 128},
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 60, "string": 1},  # fret missing
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 60, "fret": 3},  # string missing
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 60, "string": 0, "fret": 3},
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 60, "confidence": {"pitch": 1.5}},
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 60, "part_posterior": {"g1": -0.1}},
        {"onset_s": 0.0, "offset_s": math.nan, "pitch": 60},
        {"onset_s": 0.0, "offset_s": 0.5, "pitch": 60, "stringg": 1},  # typo -> extra forbidden
    ],
)
def test_invalid_notes_rejected(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        Note(**kwargs)


def test_invalid_lines_and_score_rejected() -> None:
    with pytest.raises(ValidationError):
        Line(id="x", name="x", kind="keys")
    with pytest.raises(ValidationError):
        Line(id="", name="x", kind="guitar")
    with pytest.raises(ValidationError):
        Line(id="x", name="x", kind="guitar", tuning=[])
    with pytest.raises(ValidationError):
        RoleSegment(start_s=2.0, end_s=1.0, role="lead")
    data = make_score().model_dump()
    data["lines"][1]["id"] = "g1"
    with pytest.raises(ValidationError, match="duplicate line ids"):
        Score.model_validate(data)
    data = make_score().model_dump()
    data["schema_version"] = 1
    with pytest.raises(ValidationError):
        Score.model_validate(data)
    with pytest.raises(ValidationError):
        SongMeta(song_key="k", duration_s=1.0, source_kind="spotify", source_ref="x")


def test_export_json_schema(tmp_path: Path) -> None:
    path = tmp_path / "schema" / "score.v0.schema.json"
    returned = export_json_schema(path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == returned
    assert on_disk["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert on_disk["title"] == "Score"
    assert on_disk["properties"]["schema_version"]["const"] == 0
    assert set(on_disk["required"]) == {"song"}
    defs = on_disk["$defs"]
    assert {"Note", "Line", "SongMeta", "Confidence", "RoleSegment"} <= set(defs)
    note = defs["Note"]
    assert note["additionalProperties"] is False
    assert {"onset_s", "offset_s", "pitch"} <= set(note["required"])
    assert note["properties"]["pitch"]["maximum"] == 127
    assert defs["Line"]["properties"]["kind"]["enum"] == ["guitar", "bass", "reference", "unassigned"]
