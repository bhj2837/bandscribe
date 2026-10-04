"""Stem file name -> sound category, for the distractor measurement (``bandscribe eval run --suite distractors``).

Why a second mapper next to ``bandscribe.datasets.store.guess_part``: that one decides a multitrack *part kind*
(guitar / bass / keys / other ...) for ``lines.yaml``. The distractor suite needs finer **sound** categories, because
the question is *which kind of sound* puts false notes into the guitar transcription or hides real ones: a synth pad
and a synth lead, an electric piano and an acoustic piano, a riser and a string section behave differently in the
separator and in MuScriptor.

``CATEGORIES`` (fixed order, also the report order):

- guitar_electric, guitar_acoustic: the true guitars (reference transcription, ``base`` condition)
- bass, drums, vocals: the rest of ``base`` (synth bass is bass; claps, shakers, percussion are drums; choir is vocals)
- keys_piano, keys_epiano (Rhodes, Wurlitzer, electric piano, clavinet), organ (Hammond, organ)
- synth_pad, synth_lead (an explicit "lead" synth, stylophone), synth_other (a synth whose name says nothing more)
- strings (violin, viola, cello, strings, "StringPad"), brass_winds
- sfx (SFX, FX, risers, sweeps, sub drops, reverses, noise)
- other: anything the rules do not know (listed by ``unknown_names``) and loops (a loop can be tonal)

Rules work on lower-case tokens (CamelCase, digits and separators split: ``SynthPad1`` -> synth, pad, 1) in a fixed
precedence; the first rule that matches wins. Precedence (why):

1. guitar (``gtr``/``guitar``/``guit``): ``GtrFX`` or ``ElecGtrSFX`` are still played on the guitar.
2. bass before sfx (``BassFX`` is the bass), but ``bassdrum`` is drums and ``bassoon`` is a wind.
3. sfx before every keys/synth rule: ``SynthFX``, ``RhodesSFX`` are effects whatever made them.
4. drums, vocals, strings (before pads: ``StringPad`` sounds like strings), brass/winds, electric piano (before
   piano: ``ElecPiano``), piano, organ, synth pad, synth lead, synth other.

Deterministic, stdlib only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

CATEGORIES: tuple[str, ...] = (
    "guitar_electric", "guitar_acoustic", "bass", "drums", "vocals", "keys_piano", "keys_epiano", "organ",
    "synth_pad", "synth_lead", "synth_other", "strings", "brass_winds", "sfx", "other",
)
GUITAR: tuple[str, ...] = ("guitar_electric", "guitar_acoustic")
BASE: tuple[str, ...] = GUITAR + ("bass", "drums", "vocals")
DISTRACTORS: tuple[str, ...] = tuple(c for c in CATEGORIES if c not in BASE)

LABEL_KO: dict[str, str] = {
    "guitar_electric": "일렉 기타", "guitar_acoustic": "어쿠스틱 기타", "bass": "베이스", "drums": "드럼·타악기",
    "vocals": "보컬", "keys_piano": "피아노", "keys_epiano": "일렉트릭 피아노", "organ": "오르간",
    "synth_pad": "신스 패드", "synth_lead": "신스 리드", "synth_other": "신스(기타 종류)", "strings": "현악",
    "brass_winds": "관악", "sfx": "효과음(라이저·스윕 등)", "other": "미분류",
}

_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

_GUITAR = {"gtr", "gtrs", "guitar", "guitars", "guit", "gt"}
_ACOUSTIC = {"ac", "acoustic", "acc", "acous", "nylon", "classical", "steelstring"}
_BASS = {"bass", "basses", "bassdi", "bassamp"}
_SFX = {"sfx", "fx", "riser", "risers", "rise", "rising", "sweep", "sweeps", "subdrop", "drop", "reverse",
        "reversed", "rev", "noise", "whoosh", "impact", "impacts", "uplifter", "downlifter", "swell", "ambience",
        "ambient", "atmos", "suck", "vinyl", "laugh", "wolf"}
_DRUMS = {"kick", "kik", "snare", "snr", "hat", "hats", "hihat", "hh", "tom", "toms", "overhead", "overheads", "oh",
          "ohs", "room", "rooms", "cymbal", "cymbals", "crash", "ride", "drum", "drums", "kit", "perc", "percussion",
          "tamb", "tambourine", "shaker", "shakers", "clap", "claps", "handclap", "handclaps", "cowbell", "conga",
          "congas", "bongo", "bongos", "woodblock", "maracas", "triangle", "djembe", "cajon", "timbale", "timbales",
          "trigger", "sample", "samples", "stick", "sticks", "snap", "snaps", "fingersnap"}
_VOCALS = {"vox", "vocal", "vocals", "voc", "voice", "voices", "bv", "bvs", "bvox", "choir", "harmony", "harmonies",
           "adlib", "adlibs", "whisper", "singer", "sing", "chant", "chants"}
_STRINGS = {"string", "strings", "violin", "violins", "viola", "violas", "cello", "cellos", "fiddle", "orchestra",
            "pizz", "pizzicato"}
_BRASS_WINDS = {"trumpet", "trumpets", "trombone", "trombones", "horn", "horns", "brass", "tuba", "flugelhorn",
                "sax", "saxophone", "flute", "flutes", "clarinet", "oboe", "bassoon", "harmonica", "whistle",
                "recorder", "piccolo"}
_EPIANO = {"rhodes", "wurli", "wurlitzer", "epiano", "ep", "clav", "clavinet", "elecpiano", "rhodespiano"}
_PIANO = {"piano", "pianos", "pno", "grand", "keys", "key", "keyboard", "keyboards", "celesta", "celeste", "glock",
          "glockenspiel", "vibes", "vibraphone", "marimba", "xylophone"}
_ORGAN = {"organ", "organs", "hammond", "leslie", "b3"}
_PAD = {"pad", "pads", "drone", "drones"}
_LEAD = {"lead", "leads"}
_LEAD_ALONE = {"stylophone"}
_SYNTH = {"synth", "synths", "synthesizer", "moog", "arp", "arps", "seq", "sequence", "pluck", "plucks", "juno",
          "korg", "prophet", "mellotron", "theremin"}
_OTHER_TONAL = {"loop", "loops", "mandolin", "banjo", "ukulele", "sitar", "accordion", "bells", "bell", "chimes"}
_EXCLUDE = {"click", "metronome", "guide", "unused", "mix", "master", "premaster", "mixdown"}


def tokens(name: str) -> list[str]:
    """Lower-case tokens of a stem file name (``"12_ElecGtr2DI.wav"`` -> ``["12", "elec", "gtr", "2", "di"]``)."""
    stem = Path(str(name)).stem
    return [t.lower() for t in _TOKEN_RE.findall(stem)]


def _joined(toks: list[str]) -> str:
    return "".join(t for t in toks if not t.isdigit())


def is_excluded(name: str) -> bool:
    """A file that never belongs in a mix of the song (click track, guide, a provided mix/master)."""
    return any(t in _EXCLUDE for t in tokens(name))


def categorize(name: str) -> str:
    """Sound category of a stem file name (one of ``CATEGORIES``; unknown -> ``other``)."""
    toks = tokens(name)
    ts = set(toks)
    joined = _joined(toks)
    if ts & _GUITAR or "guitar" in joined:
        return "guitar_acoustic" if ts & _ACOUSTIC else "guitar_electric"
    if "bassdrum" in joined or "bassdrm" in joined:
        return "drums"
    if "bassoon" in ts:
        return "brass_winds"
    if ts & _BASS or joined.startswith("bass") or "synthbass" in joined or "subbass" in joined:
        return "bass"
    if ts & _SFX:
        return "sfx"
    if ts & _DRUMS:
        return "drums"
    if ts & _VOCALS:
        return "vocals"
    if ts & _STRINGS:
        return "strings"
    if ts & _BRASS_WINDS:
        return "brass_winds"
    if ts & _EPIANO or ("elec" in ts and "piano" in ts) or ("electric" in ts and "piano" in ts):
        return "keys_epiano"
    if ts & _ORGAN:
        return "organ"
    if ts & _PIANO:
        return "keys_piano"
    if ts & _PAD:
        return "synth_pad"
    if ts & _LEAD_ALONE or (ts & _LEAD and ts & _SYNTH):
        return "synth_lead"
    if ts & _SYNTH:
        return "synth_other"
    return "other"


def known(name: str) -> bool:
    """True when a rule (not the ``other`` fallback) placed the name; loops etc. count as known."""
    toks = set(tokens(name))
    return categorize(name) != "other" or bool(toks & _OTHER_TONAL)


def unknown_names(names: Iterable[str]) -> list[str]:
    """Names no rule recognises (they fall back to ``other``), sorted; excluded files are not listed."""
    return sorted({str(n) for n in names if not is_excluded(str(n)) and not known(str(n))})


def label_ko(category: str) -> str:
    return LABEL_KO.get(category, category)
