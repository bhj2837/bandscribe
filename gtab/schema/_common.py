"""Shared pydantic base and field types of the M1a/M2 file formats (M1_M2_SPEC 6).

Conventions (M1_M2_SPEC 6): times in seconds (rounded to 1e-6 in files), MIDI pitch ints, musical bar numbers
(1 = first full bar, 0 = pickup), half-open bar ranges, ticks = 48 per beat of the bar's beat unit, string 1 =
highest-pitched string.

Every model forbids unknown fields (a misspelt key in a stage's output fails here instead of vanishing) and
rejects NaN/inf (JSON has neither; a NaN from a model output must fail loudly). File-level models carry
``format: Literal["gtab.<name>/1"]`` - a field called ``schema`` would shadow ``BaseModel.schema``.

pydantic + stdlib only, Python-3.10-safe. Workers never import ``gtab.schema`` (M1_M2_SPEC 0, A13): they write
plain JSON and the core validates it.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Prob = Annotated[float, Field(ge=0.0, le=1.0)]
Seconds = Annotated[float, Field(ge=0.0)]
Midi = Annotated[int, Field(ge=0, le=127)]

# The dominant / per-bar beat unit of a meter (DESIGN 6.2): 4/4 -> quarter, 6/8 -> dotted_quarter, 7/8 -> eighth.
BEAT_UNITS: tuple[str, ...] = ("quarter", "dotted_quarter", "eighth")


def check_half_open(start: int, end: int, what: str) -> None:
    """Bar ranges are half-open everywhere (start inclusive, end exclusive), so an empty range is an error."""
    if end <= start:
        raise ValueError(f"{what}: end_bar ({end}) must be > start_bar ({start}) (half-open range)")
