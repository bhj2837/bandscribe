"""Guitar Pro 5 writer (PyGuitarPro, LGPL-3.0; DESIGN 7.1 ``.gp5``).

GP5 stores notes only as string and fret, so only fretted tracks go in (the bass tab, and the merged guitar
with ``--parts 1``); a staff-only guitar stays in alphaTex / MIDI (DESIGN 4.2). Same spelled bars as the
alphaTex writer: time signature, triplet feel and section marker per measure header, a tempo change as a mix
table change on the first beat of the first track, ties as ``NoteType.tie``, tuplets as ``Tuplet(3, 2)`` /
``Tuplet(6, 4)``, capo as the track offset.
"""

from __future__ import annotations

from pathlib import Path

import guitarpro as gp

from bandscribe.tab.alphatex import WScore

GM_PROGRAM = {"electricbassfinger": 33, "acousticguitarsteel": 25, "electricguitarclean": 27,
              "overdrivenguitar": 29, "distortionguitar": 30}


def build(score: WScore) -> gp.Song:
    song = gp.Song()
    song.title = score.title
    song.artist = score.artist or ""
    song.tempo = int(score.tempo_qpm[0]) if score.tempo_qpm else 120
    song.measureHeaders = []
    song.tracks = []
    for i, bar in enumerate(score.bars):
        h = gp.MeasureHeader(number=i + 1, timeSignature=gp.TimeSignature(numerator=bar.numerator,
                                                                           denominator=gp.Duration(value=bar.denominator)),
                             tripletFeel=gp.TripletFeel.eighth if bar.feel == "triplet8th" else gp.TripletFeel.none)
        if bar.section:
            h.marker = gp.Marker(title=bar.section)
        song.measureHeaders.append(h)
    fretted = [t for t in score.tracks if t.fretted]
    for ti, tr in enumerate(fretted, start=1):
        program = GM_PROGRAM.get(tr.instrument, 25)
        ch = ti - 1 if ti - 1 < 9 else ti  # skip the drum channel 9
        track = gp.Track(song, number=ti, name=tr.name, fretCount=24, offset=int(tr.capo),
                         strings=[gp.GuitarString(number=s, value=int(p))
                                  for s, p in enumerate(reversed(tr.tuning or []), start=1)],
                         channel=gp.MidiChannel(channel=ch, effectChannel=ch, instrument=program))
        track.measures = []
        prev_tempo = song.tempo
        for bi, (h, beats) in enumerate(zip(song.measureHeaders, tr.beats)):
            m = gp.Measure(track, h)
            voice = m.voices[0]
            for k, be in enumerate(beats):
                tup = gp.Tuplet(enters=be.tuplet[0], times=be.tuplet[1]) if be.tuplet else gp.Tuplet(enters=1, times=1)
                beat = gp.Beat(voice, duration=gp.Duration(value=be.value, isDotted=be.dots == 1, tuplet=tup),
                               status=gp.BeatStatus.normal if be.notes else gp.BeatStatus.rest)
                for n in be.notes:
                    beat.notes.append(gp.Note(beat, value=int(n.fret), string=int(n.string), velocity=95,
                                              type=gp.NoteType.tie if n.tie else gp.NoteType.normal))
                if ti == 1 and k == 0 and bi < len(score.tempo_qpm) and score.tempo_qpm[bi] != prev_tempo:
                    beat.effect.mixTableChange = gp.MixTableChange(
                        tempo=gp.MixTableItem(value=int(score.tempo_qpm[bi]), duration=0))
                    prev_tempo = score.tempo_qpm[bi]
                voice.beats.append(beat)
            track.measures.append(m)
        song.tracks.append(track)
    return song


def text_encoding(score: WScore) -> str:
    """GP5 stores strings in a single-byte code page: cp1252 (Guitar Pro's default) when the texts fit, else
    cp949 for Korean titles (a Korean Windows reads it; alphaTab: ``settings.importer.encoding = "euc-kr"``)."""
    texts = [score.title, score.artist or ""] + [t.name for t in score.tracks] + [b.section or "" for b in score.bars]
    for enc in ("cp1252", "cp949"):
        try:
            for t in texts:
                t.encode(enc)
            return enc
        except UnicodeEncodeError:
            continue
    return "cp1252"


def write(score: WScore, path: Path) -> bool:
    """Write ``path``; False (nothing written) when the score has no fretted track."""
    if not any(t.fretted for t in score.tracks):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    enc = text_encoding(score)
    song = build(score)
    # characters outside the chosen code page (rare symbols) become "?" rather than failing the stage
    song.title = song.title.encode(enc, "replace").decode(enc)
    song.artist = song.artist.encode(enc, "replace").decode(enc)
    gp.write(song, str(path), version=(5, 1, 0), encoding=enc)
    return True
