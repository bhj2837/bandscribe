"""Sections and the repeat map (M1_M2_SPEC 6.3): ``sections.json`` and ``repeat_map.json``.

Bar ranges are half-open (``end_bar`` exclusive) like every other bar range in bandscribe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from bandscribe.schema._common import Prob, Seconds, _Model, check_half_open


class Section(_Model):
    id: str
    label: str  # A, B, C... in order of first appearance
    start_bar: int
    end_bar: int  # exclusive
    start_s: Seconds
    end_s: Seconds
    confidence: Prob

    @model_validator(mode="after")
    def _range(self) -> Section:
        check_half_open(self.start_bar, self.end_bar, f"section {self.id}")
        if self.end_s < self.start_s:
            raise ValueError(f"section {self.id}: end_s ({self.end_s}) < start_s ({self.start_s})")
        return self


class Sections(_Model):
    format: Literal["bandscribe.sections/1"]
    method: str
    sections: list[Section]
    params: dict


class Occurrence(_Model):
    section: str  # section id
    start_bar: int
    n_bars: int = Field(ge=1)
    offset_s: float  # vs the group's reference occurrence


class RepeatGroup(_Model):
    label: str
    occurrences: list[Occurrence]
    reference: str  # reference section id


class BarMatch(_Model):
    bar: int
    matches: list[dict]  # [{"bar": int, "score": Prob, "offset_s": float}], best first


class RepeatMap(_Model):
    format: Literal["bandscribe.repeat_map/1"]
    groups: list[RepeatGroup]
    bars: list[BarMatch]
    params: dict
