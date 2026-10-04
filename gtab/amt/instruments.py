"""MT3 instrument classes, families, user hints and MuScriptor hard masks (M1_M2_SPEC 9.2 item 4, DESIGN S3.5/S6).

MuScriptor's ``instruments`` argument is both a conditioning string and a **hard mask** (every other program
token is forbidden), and the conditioning string is built in list order. So:

- every mask sent to the worker and every cache key goes through ``sort_mt3`` (MT3 id order, no duplicates);
- a guitar class is never dropped from a guitar-view mask (omitting clean guitar deletes those notes);
- the piano+other view is masked with the keys/synth/strings classes (``PRESENCE_CLASSES``), plus the guitar
  classes by default (``amt.piano_other_instruments = "keys+guitar"``, M1_M2_SPEC A5): with a keys-only mask,
  guitar bleed in the "other" stem would be forced into keys classes and later penalise true guitar notes in S7.
  Strings joined the mask on 2026-10-04: on 怪獣の花唄 / 青と夏 SW's guitar stem is mostly non-guitar content
  (very likely strings/pads), and a mask without strings could only call it synth or keys.
- the guitar-pass mask ``guitar+present`` (default) adds the **keys/synth** classes the instrumentation marks
  present (the M2 set), so MuScriptor can label keyboard bleed non-guitar instead of forcing it into a guitar
  class. Strings are **not** added by default (review 2026-10-04): on 怪獣の花唄 / 青と夏 / SC the strings-widened
  mask removed no measurable bleed (share of guitar notes also found on the guitar-removed backing 17.7 -> 17.7 %
  and 15.3 -> 20.2 %) but reshuffled the guitar sub-class labels (MuScriptor's labels follow the prompt) and
  added notes on SC. ``guitar+present+strings`` keeps that variant for E3b; it never adds ``orchestral_harp``
  (plucked, the most guitar-like class) or ``contrabass`` (guitar's low register). Brass/winds stay out of every
  mask: each extra class is one more sink a real guitar note can be mislabelled into.

The class and family tables are ``gtab.schema.instrumentation.MT3_GROUPS`` / ``FAMILIES`` (P0).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from gtab.config import ConfigError
from gtab.schema.instrumentation import FAMILIES, FAMILY_OF, MT3_GROUPS

log = logging.getLogger(__name__)

GUITAR_CLASSES: tuple[str, ...] = tuple(FAMILIES["guitar"])
BASS_CLASSES: tuple[str, ...] = tuple(FAMILIES["bass"])
KEYS_SYNTH_CLASSES: tuple[str, ...] = tuple(FAMILIES["keys"]) + tuple(FAMILIES["synth"])
# Non-guitar classes of the piano+other (presence) transcription (evidence for instrumentation.json).
PRESENCE_FAMILIES: tuple[str, ...] = ("keys", "synth", "strings")
PRESENCE_CLASSES: tuple[str, ...] = tuple(c for fam in PRESENCE_FAMILIES for c in FAMILIES[fam])
# Present classes a guitar-pass mask may add: guitar+present = keys/synth; guitar+present+strings (E3b) also the
# bowed strings, never orchestral_harp / contrabass (guitar-like timbre / register).
GUITAR_MASK_STRINGS: tuple[str, ...] = ("violin", "viola", "cello", "string_ensemble")
GUITAR_MASK_ADDS: dict[str, tuple[str, ...]] = {
    "guitar+present": KEYS_SYNTH_CLASSES,
    "guitar+present+strings": KEYS_SYNTH_CLASSES + GUITAR_MASK_STRINGS,
}

GUITAR_MASK_POLICIES = ("guitar_only", "guitar+present", "guitar+present+strings", "all")
PIANO_OTHER_MODES = ("keys", "keys+guitar")
GUITAR_VIEWS = ("guitar_mono", "mix_mono", "nonvox_mono", "guitar_other_mono")


def family_of(cls: str) -> str:
    try:
        return FAMILY_OF[cls]
    except KeyError:
        raise ValueError(f"unknown MT3 instrument class {cls!r}") from None


def sort_mt3(classes: Iterable[str]) -> list[str]:
    """MT3-id order without duplicates; the one order used for requests and cache keys."""
    names = list(dict.fromkeys(str(c) for c in classes))
    unknown = [c for c in names if c not in MT3_GROUPS]
    if unknown:
        raise ValueError(f"unknown MT3 instrument class(es): {unknown}")
    return sorted(names, key=lambda c: MT3_GROUPS[c])


def classes_of(families: Iterable[str]) -> list[str]:
    out: list[str] = []
    for fam in families:
        out.extend(FAMILIES[fam])
    return sort_mt3(out)


def parse_hint(hint: Any) -> list[str] | None:
    """``hints.instruments``: ``"auto"`` -> None; ``"guitar, Keys"`` -> ``["guitar", "keys"]`` (FAMILIES order).

    An unknown family is a user error (Korean ConfigError, CLI exit 2).
    """
    if hint is None:
        return None
    if isinstance(hint, (list, tuple)):
        parts = [str(p) for p in hint]
    else:
        text = str(hint).strip()
        if text.lower() in ("", "auto"):
            return None
        parts = text.replace(";", ",").split(",")
    fams = [p.strip().lower() for p in parts if p.strip()]
    if not fams:
        return None
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        raise ConfigError(
            f"hints.instruments 에 알 수 없는 악기 계열이 있습니다: {', '.join(unknown)} "
            f"(쓸 수 있는 값: auto 또는 {', '.join(FAMILIES)} 의 쉼표 목록)")
    order = list(FAMILIES)
    return sorted(dict.fromkeys(fams), key=order.index)


# -------------------------------------------------------------------------------------------- masks


def mask_bass() -> list[str]:
    return sort_mt3(BASS_CLASSES)


def mask_guitar_only() -> list[str]:
    return sort_mt3(GUITAR_CLASSES)


def mask_piano_other(mode: str = "keys+guitar") -> list[str]:
    """Mask of the piano+other view (S3.5 presence evidence; S7 bleed later): keys/synth/strings (+ guitar, A5)."""
    if mode not in PIANO_OTHER_MODES:
        raise ConfigError(f"amt.piano_other_instruments 는 {' | '.join(PIANO_OTHER_MODES)} 중 하나여야 합니다 "
                            f"(입력값: {mode!r})")
    classes = list(PRESENCE_CLASSES)
    if mode == "keys+guitar":
        classes += list(GUITAR_CLASSES)
    return sort_mt3(classes)


def _present(instrumentation: Any) -> dict[str, bool]:
    """{class: present} from an Instrumentation model or its JSON dict."""
    classes = instrumentation.get("classes") if isinstance(instrumentation, Mapping) else getattr(
        instrumentation, "classes", None)
    out: dict[str, bool] = {}
    for name, ev in (classes or {}).items():
        present = ev.get("present") if isinstance(ev, Mapping) else getattr(ev, "present", False)
        out[str(name)] = bool(present)
    return out


def mask_guitar_view(instrumentation: Any, policy: str = "guitar+present") -> list[str] | None:
    """Mask for the guitar transcription (``amt.guitar_mask``, E3).

    ``guitar_only`` = the three guitar classes; ``guitar+present`` = those plus every keys/synth class the
    instrumentation marks present; ``guitar+present+strings`` = also the present bowed strings
    (``GUITAR_MASK_STRINGS``); ``all`` = None (no mask). The guitar classes are always in a list mask, whatever
    the instrumentation says.
    """
    if policy not in GUITAR_MASK_POLICIES:
        raise ConfigError(f"amt.guitar_mask 는 {' | '.join(GUITAR_MASK_POLICIES)} 중 하나여야 합니다 "
                            f"(입력값: {policy!r})")
    if policy == "all":
        return None
    classes = list(GUITAR_CLASSES)
    if policy in GUITAR_MASK_ADDS:
        present = _present(instrumentation)
        classes += [c for c in GUITAR_MASK_ADDS[policy] if present.get(c)]
    return sort_mt3(classes)


def pass1_masks(piano_other_mode: str) -> dict[str, list[str] | None]:
    """Masks of the pass-1 views of ``amt_ms1`` (the mix view is transcribed with every class, B0).

    ``piano_other_presence`` (sampled windows of the piano+other view) uses the same mask as the full view.
    """
    po = mask_piano_other(piano_other_mode)
    return {"mix_mono": None, "bass_mono": mask_bass(), "piano_other_mono": po, "piano_other_presence": list(po)}


def presence_evidence_classes(piano_other_mode: str) -> list[str]:
    """Classes the piano+other transcription is presence evidence for: its mask minus the guitar classes
    (guitar notes there are bleed; guitar presence comes from the guitar stem rule)."""
    return [c for c in mask_piano_other(piano_other_mode) if c not in GUITAR_CLASSES]


def guitar_omissions(mask: list[str] | None) -> int:
    """Number of guitar classes missing from a mask (must be 0; E3 constraint, b13)."""
    if mask is None:
        return 0
    return sum(1 for c in GUITAR_CLASSES if c not in mask)
