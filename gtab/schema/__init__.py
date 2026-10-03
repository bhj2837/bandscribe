"""Data contracts shared across stages, the exporter, the evaluation harness and the web UI.

- ``score``: score.json v0 (M0).
- M1a/M2 file formats (M1_M2_SPEC 6): ``notes`` (NoteSet), ``grid`` (beats.json, grid.json, TempoMap),
  ``sections`` (sections.json, repeat_map.json), ``instrumentation`` (MT3 classes, instrumentation.json),
  ``gt`` (gt.yaml, gt_notes.json), ``dataset`` (DatasetTrack), ``evalio`` (Prediction).

Workers never import this package (M1_M2_SPEC 0): they write plain JSON that the core validates.
"""

from __future__ import annotations

from gtab.schema._common import BEAT_UNITS, Midi, Prob, Seconds
from gtab.schema.dataset import DatasetTrack, Part
from gtab.schema.evalio import PredNote, Prediction
from gtab.schema.grid import BeatsRaw, Grid, GridBar, GridBeat, TempoMap, TempoSegment
from gtab.schema.gt import GtLine, GtNote, GtNotes, GtSection, GtSpec, GtTrack
from gtab.schema.instrumentation import (
    BASS_CLASSES,
    FAMILIES,
    FAMILY_OF,
    GUITAR_CLASSES,
    MT3_GROUPS,
    ClassEvidence,
    Instrumentation,
)
from gtab.schema.notes import BackendInfo, NoteEvent, NoteSet, PitchBend, notes_to_arrays, sort_notes
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
from gtab.schema.sections import BarMatch, Occurrence, RepeatGroup, RepeatMap, Section, Sections

__all__ = [
    # score.json v0 (M0)
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
    # shared field types
    "BEAT_UNITS",
    "Midi",
    "Prob",
    "Seconds",
    # notes (6.1)
    "BackendInfo",
    "NoteEvent",
    "NoteSet",
    "PitchBend",
    "notes_to_arrays",
    "sort_notes",
    # grid (6.2)
    "BeatsRaw",
    "Grid",
    "GridBar",
    "GridBeat",
    "TempoMap",
    "TempoSegment",
    # sections (6.3)
    "BarMatch",
    "Occurrence",
    "RepeatGroup",
    "RepeatMap",
    "Section",
    "Sections",
    # instrumentation (6.5)
    "BASS_CLASSES",
    "FAMILIES",
    "FAMILY_OF",
    "GUITAR_CLASSES",
    "MT3_GROUPS",
    "ClassEvidence",
    "Instrumentation",
    # ground truth (6.7)
    "GtLine",
    "GtNote",
    "GtNotes",
    "GtSection",
    "GtSpec",
    "GtTrack",
    # datasets (6.8)
    "DatasetTrack",
    "Part",
    # evaluation (6.9)
    "PredNote",
    "Prediction",
]
