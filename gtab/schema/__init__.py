"""Data contracts shared across stages, the exporter and the web UI (score.json v0)."""

from __future__ import annotations

from gtab.schema.score import (
    NOTE_FLAGS,
    SCHEMA_VERSION,
    Confidence,
    Line,
    Note,
    RoleSegment,
    Score,
    SongMeta,
    export_json_schema,
    json_schema,
    load_score,
    save_score,
)

__all__ = [
    "NOTE_FLAGS",
    "SCHEMA_VERSION",
    "Confidence",
    "Line",
    "Note",
    "RoleSegment",
    "Score",
    "SongMeta",
    "export_json_schema",
    "json_schema",
    "load_score",
    "save_score",
]
