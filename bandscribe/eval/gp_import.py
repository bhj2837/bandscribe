"""Reference tab import: Guitar Pro files -> ``RefScore`` (spec §9.4 item 6).

- ``.gp3/.gp4/.gp5``: PyGuitarPro 0.11 (LGPL-3.0). String 1 is already the highest-pitched string there;
  ``Note.realValue`` omits the capo (``Track.offset``), so the pitch is computed as tuning + capo + fret.
  GP3-5 strings carry no encoding: PyGuitarPro decodes cp1252 by default, which garbles Korean track names.
  We use ``reference.encoding`` from gt.yaml if given, else try ``cp949`` (strict), then ``cp1252``, then
  ``latin-1`` (never fails), and record the encoding used (names only; notes are unaffected).
- ``.gp/.gpx`` (and any format as a cross-check): alphaTab in Node (``bandscribe.eval.node.alphatab_import``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from bandscribe.eval.refscore import RefScore, RsMasterBar, RsNote, RsTrack

log = logging.getLogger(__name__)

PYGP_EXTS = {".gp3", ".gp4", ".gp5"}
ALPHATAB_EXTS = {".gp", ".gpx", ".musicxml", ".xml", ".mxl", ".cap", ".tex"}
DEFAULT_ENCODINGS = ("cp949", "cp1252", "latin-1")


class ReferenceImportError(RuntimeError):
    """Reference could not be imported (message is Korean, user-facing)."""


def _parse_song(path: Path, encoding: str | None):
    import guitarpro

    tried: list[str] = []
    for enc in ([encoding] if encoding else list(DEFAULT_ENCODINGS)):
        try:
            return guitarpro.parse(str(path), encoding=enc), enc
        except Exception as e:  # PyGuitarPro wraps decode errors in GPException (cause = UnicodeDecodeError)
            if isinstance(e, (UnicodeDecodeError, LookupError)) or isinstance(e.__cause__, UnicodeDecodeError):
                tried.append(f"{enc}: {e}")
                continue
            raise ReferenceImportError(f"참조 탭을 읽지 못했습니다 ({path.name}): {e}") from e
    raise ReferenceImportError(f"문자 인코딩을 알 수 없어 참조 탭을 읽지 못했습니다 ({'; '.join(tried)}). "
                       "gt.yaml 의 reference.encoding 을 지정해 보세요.")


def _bar_tempos(song) -> dict[int, float]:
    """Tempo changes that sit on the first beat of a bar (mix-table changes), by written bar (1-based)."""
    out: dict[int, float] = {}
    for track in song.tracks:
        for mi, measure in enumerate(track.measures, 1):
            for voice in measure.voices:
                for beat in voice.beats:
                    mtc = getattr(beat.effect, "mixTableChange", None)
                    tempo = getattr(mtc, "tempo", None) if mtc is not None else None
                    val = getattr(tempo, "value", None) if tempo is not None else None
                    if val and beat.start == measure.start:
                        out.setdefault(mi, float(val))
    return out


def read_gp345(path: Path, encoding: str | None = None) -> RefScore:
    """PyGuitarPro importer (GP3/4/5)."""
    from guitarpro import models as gpm

    path = Path(path)
    song, used = _parse_song(path, encoding)
    tracks: list[RsTrack] = []
    notes: list[RsNote] = []
    for ti, track in enumerate(song.tracks, 1):
        strings = sorted(track.strings, key=lambda s: s.number)
        tuning = [int(s.value) for s in strings]
        capo = int(track.offset or 0)
        tracks.append(RsTrack(index=ti, name=track.name, percussion=bool(track.isPercussionTrack),
                              n_strings=len(tuning), tuning=tuning, capo=capo,
                              program=int(track.channel.instrument) if track.channel else None))
        for mi, measure in enumerate(track.measures, 1):
            for vi, voice in enumerate(measure.voices, 1):
                for beat in voice.beats:
                    if beat.status == gpm.BeatStatus.rest or not beat.notes:
                        continue
                    start = float(beat.start - measure.start)
                    dur = float(beat.duration.time)
                    tup = beat.duration.tuplet
                    tuplet = [int(tup.enters), int(tup.times)] if (tup.enters, tup.times) != (1, 1) else None
                    for note in beat.notes:
                        if note.type == gpm.NoteType.rest:
                            continue
                        s = int(note.string)
                        if not 1 <= s <= len(tuning):
                            log.warning("%s: track %d bar %d has string %d outside the tuning", path.name, ti, mi, s)
                            continue
                        fret = int(note.value)
                        notes.append(RsNote(track=ti, bar=mi, voice=vi, start=start, dur=dur, string=s, fret=fret,
                                            pitch=tuning[s - 1] + capo + fret, tie=note.type == gpm.NoteType.tie,
                                            dead=note.type == gpm.NoteType.dead, tuplet=tuplet))
    tempos = _bar_tempos(song)
    mbs = []
    for hi, h in enumerate(song.measureHeaders, 1):
        mbs.append(RsMasterBar(index=hi, numerator=int(h.timeSignature.numerator),
                               denominator=int(h.timeSignature.denominator.value),
                               repeat_open=bool(h.isRepeatOpen),
                               # PyGuitarPro keeps the GP3 meaning (number of repeats, -1 = none); RefScore
                               # stores the total play count like alphaTab.
                               repeat_count=int(h.repeatClose) + 1 if h.repeatClose > -1 else 0,
                               alternate=int(h.repeatAlternative or 0),
                               tempo_bpm=tempos.get(hi, float(song.tempo) if hi == 1 else None),
                               start=int(h.start - 960), length=int(h.length)))
    return RefScore(source={"file": path.name, "format": path.suffix.lower().lstrip("."),
                            "importer": "pyguitarpro-0.11", "encoding": used},
                    title=song.title or "", artist=song.artist or "", tempo_bpm=float(song.tempo),
                    tracks=tracks, masterbars=mbs, notes=notes)


def import_reference(path: Path, *, encoding: str | None = None, engine: str = "auto",
                     timeout_s: float = 300) -> RefScore:
    """Import any supported reference file. ``engine``: auto | pyguitarpro | alphatab."""
    path = Path(path)
    ext = path.suffix.lower()
    if engine == "auto":
        engine = "pyguitarpro" if ext in PYGP_EXTS else "alphatab"
    if engine == "pyguitarpro":
        if ext not in PYGP_EXTS:
            raise ReferenceImportError(f"PyGuitarPro 는 .gp3/.gp4/.gp5 만 읽습니다: {path.name}")
        return read_gp345(path, encoding)
    if engine == "alphatab":
        from bandscribe.eval.node import alphatab_import

        return alphatab_import(path, encoding=encoding or ("cp949" if ext in PYGP_EXTS else None), timeout_s=timeout_s)
    raise ValueError(f"unknown import engine {engine!r}")
