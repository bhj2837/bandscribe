"""Instrumentation (S3.5 편성 추정, M1_M2_SPEC 6.5): MT3 classes, families and ``instrumentation.json``.

``MT3_GROUPS`` equals the installed muscriptor 0.3.0 MT3_FULL_PLUS group table (verified: 35 entries, ids 0-33
plus drums 36). MuScriptor's ``instruments`` argument is a hard mask **and** an order-dependent conditioning
string, so every mask is sorted by these ids before it is sent or hashed (``bandscribe.amt.instruments.sort_mt3``).
Workers keep their own copy of the table (they never import bandscribe.schema); AMT's ``probe`` test compares them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from bandscribe.schema._common import Prob, _Model

MT3_GROUPS: dict[str, int] = {
    "acoustic_piano": 0, "electric_piano": 1, "chromatic_percussion": 2, "organ": 3, "acoustic_guitar": 4,
    "clean_electric_guitar": 5, "distorted_electric_guitar": 6, "acoustic_bass": 7, "electric_bass": 8,
    "violin": 9, "viola": 10, "cello": 11, "contrabass": 12, "orchestral_harp": 13, "timpani": 14,
    "string_ensemble": 15, "synth_strings": 16, "voice": 17, "orchestra_hit": 18, "trumpet": 19,
    "trombone": 20, "tuba": 21, "french_horn": 22, "brass_section": 23, "soprano_and_alto_sax": 24,
    "tenor_sax": 25, "baritone_sax": 26, "oboe": 27, "english_horn": 28, "bassoon": 29, "clarinet": 30,
    "flutes": 31, "synth_lead": 32, "synth_pad": 33, "drums": 36,
}

FAMILIES: dict[str, tuple[str, ...]] = {
    "guitar": ("acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"),
    "bass": ("acoustic_bass", "electric_bass"),
    "keys": ("acoustic_piano", "electric_piano", "organ", "chromatic_percussion"),
    "synth": ("synth_lead", "synth_pad", "synth_strings"),
    "strings": ("violin", "viola", "cello", "contrabass", "orchestral_harp", "string_ensemble"),
    "brass": ("trumpet", "trombone", "tuba", "french_horn", "brass_section", "orchestra_hit"),
    "winds": ("soprano_and_alto_sax", "tenor_sax", "baritone_sax", "oboe", "english_horn", "bassoon",
              "clarinet", "flutes"),
    "vocals": ("voice",),
    "drums": ("drums", "timpani"),
}

GUITAR_CLASSES: tuple[str, ...] = FAMILIES["guitar"]
BASS_CLASSES: tuple[str, ...] = FAMILIES["bass"]
FAMILY_OF: dict[str, str] = {c: fam for fam, classes in FAMILIES.items() for c in classes}


def check_families(names: list[str], what: str) -> None:
    unknown = [f for f in names if f not in FAMILIES]
    if unknown:
        raise ValueError(f"{what}: unknown families {unknown} (known: {', '.join(FAMILIES)})")


class ClassEvidence(_Model):
    p: Prob
    present: bool
    sources: dict[str, dict]  # "mix_transcription" | "stem_energy" | "hint" -> evidence details


class Instrumentation(_Model):
    """``instrumentation.json``."""

    format: Literal["bandscribe.instrumentation/1"]
    profile: str
    classes: dict[str, ClassEvidence]  # every MT3 group, in MT3_GROUPS order
    families: dict[str, Prob]
    masks: dict[str, list[str] | None]  # view id -> MuScriptor hard mask (None = all classes)
    per_section: list[dict]  # [{"section", "family_active": {family: ratio}}]
    thresholds: dict
    hints: dict
    warnings: list[str]
    # Summary of the piano+other presence transcription that fed the estimate (mode sampled|full, windows,
    # per-class note counts); None when no presence pass ran (older stage dirs, experiments).
    presence: dict | None = None

    @model_validator(mode="after")
    def _check(self) -> Instrumentation:
        # Deterministic file order and completeness: a class missing here would silently never be masked in.
        if list(self.classes) != list(MT3_GROUPS):
            missing = [c for c in MT3_GROUPS if c not in self.classes]
            extra = [c for c in self.classes if c not in MT3_GROUPS]
            raise ValueError(
                f"classes must list every MT3 group in MT3_GROUPS order (missing {missing}, unknown {extra})"
            )
        check_families(list(self.families), "families")
        for view, mask in self.masks.items():
            if mask is None:
                continue
            bad = [c for c in mask if c not in MT3_GROUPS]
            if bad:
                raise ValueError(f"masks[{view!r}]: unknown MT3 classes {bad}")
        return self
