"""score.json v0: the contract between transcription / part stages, the tab exporter and the web UI.

Exported as JSON Schema (export_json_schema) so the TypeScript viewer shares the same definitions.
Models forbid unknown fields: a misspelt key in a stage's output should fail here, not vanish silently.
Importable from both envs (pydantic + stdlib only).
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gtab import __version__, atomic

SCHEMA_VERSION = 0

# Keys later stages put into Note.flags (DESIGN 4.x / S7). The dict stays open; this is the shared vocabulary.
NOTE_FLAGS: tuple[str, ...] = ("shared", "double", "octave", "variant", "low_conf", "unison_undecided")

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
Seconds = Annotated[float, Field(ge=0.0)]
Midi = Annotated[int, Field(ge=0, le=127)]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Model(BaseModel):
    # allow_inf_nan=False: JSON has no NaN; a NaN from a model output must fail loudly, not become null.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Confidence(_Model):
    """Per-aspect confidence in [0, 1]; None = that aspect was not estimated."""

    pitch: Prob | None = None
    onset: Prob | None = None
    part: Prob | None = None
    fret: Prob | None = None


class Note(_Model):
    onset_s: Seconds
    offset_s: Seconds
    pitch: Midi = Field(description="MIDI note number")
    velocity: int | None = Field(None, ge=0, le=127)
    string: int | None = Field(
        None, ge=1, le=12, description="1 = highest-pitched string (tab / Guitar Pro convention)"
    )
    fret: int | None = Field(None, ge=0, le=36)
    techniques: list[str] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    sources: list[str] = Field(default_factory=list, description="backends / views that produced this note")
    part_posterior: dict[str, Prob] = Field(default_factory=dict, description="Line id -> probability")
    flags: dict[str, bool] = Field(default_factory=dict, description=f"known keys: {', '.join(NOTE_FLAGS)}")

    @model_validator(mode="after")
    def _check(self) -> Note:
        if self.offset_s < self.onset_s:
            raise ValueError(f"offset_s ({self.offset_s}) < onset_s ({self.onset_s})")
        if (self.string is None) != (self.fret is None):
            raise ValueError("string and fret must be given together (a tab position needs both)")
        return self


class RoleSegment(_Model):
    start_s: Seconds
    end_s: Seconds
    role: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> RoleSegment:
        if self.end_s < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) < start_s ({self.start_s})")
        return self


class Line(_Model):
    id: str = Field(min_length=1)
    name: str
    kind: Literal["guitar", "bass", "reference", "unassigned"]
    role_segments: list[RoleSegment] = Field(default_factory=list)
    tuning: list[Midi] | None = Field(
        None,
        min_length=1,
        max_length=12,
        description="open-string MIDI pitches, lowest string first (Note.string 1 = last entry)",
    )
    capo: int | None = Field(None, ge=0, le=24)
    notes: list[Note] = Field(default_factory=list)


class SongMeta(_Model):
    song_key: str = Field(min_length=1)
    title: str | None = None
    artist: str | None = None
    duration_s: Seconds
    source_kind: Literal["file", "youtube"]
    source_ref: str = Field(description="file path / name or YouTube URL the job was created from")


class Score(_Model):
    schema_version: Literal[0] = SCHEMA_VERSION
    song: SongMeta
    lines: list[Line] = Field(default_factory=list)
    created_utc: str = Field(default_factory=utc_now)
    gtab_version: str = __version__

    @model_validator(mode="after")
    def _unique_line_ids(self) -> Score:
        counts = Counter(ln.id for ln in self.lines)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate line ids: {dupes}")
        return self


def json_schema() -> dict[str, Any]:
    schema = Score.model_json_schema()
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def export_json_schema(path: Path) -> dict[str, Any]:
    """Write the score.json JSON Schema (draft 2020-12) to ``path`` atomically and return it."""
    schema = json_schema()
    atomic.write_json(Path(path), schema)
    return schema


def save_score(path: Path, score: Score) -> None:
    atomic.write_text(Path(path), score.model_dump_json(indent=1) + "\n")


def load_score(path: Path) -> Score:
    return Score.model_validate_json(Path(path).read_bytes())
