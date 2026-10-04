"""Predictions handed to the evaluation harness (M1_M2_SPEC 6.9).

A :class:`Prediction` is one system's output for one item (a Tier A song id, ``<dataset>:<track>`` or a
synthetic scene id), already split into lines. Ids keep their original spelling inside JSON; file names use
``bandscribe.eval.runs.slug()`` (``:`` is illegal on NTFS).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from bandscribe.schema._common import Midi, Prob, Seconds, _Model


class PredNote(_Model):
    onset_s: Seconds
    offset_s: Seconds
    pitch: Midi
    line: str  # line id or "unassigned"
    posterior: Prob | None = None  # max part posterior (coverage-accuracy curve)
    shared: bool = False
    bar: int | None = None  # bar/tick only if the system quantized
    tick: float | None = None

    @model_validator(mode="after")
    def _check(self) -> PredNote:
        if self.offset_s < self.onset_s:
            raise ValueError(f"offset_s ({self.offset_s}) < onset_s ({self.onset_s})")
        return self


class Prediction(_Model):
    format: Literal["bandscribe.prediction/1"]
    system: str  # e.g. "job:amt_gtr", "external:klangio", "oracle"
    item: str  # song_id / dataset:track / scene id
    lines: list[dict]
    notes: list[PredNote]
    meta: dict = Field(default_factory=dict)
