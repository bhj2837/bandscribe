"""Guitar Pro 5 fixture for the GT importers (tests/test_eval_gp_import.py, tests/test_eval_alphatab.py).

Written with PyGuitarPro so no binary fixture is committed. The song exercises everything the expansion must
handle: a 1/4 pickup bar, a repeat (played twice) with alternate endings 1./2., a 6/8 bar, a capo on one
track, eighth-note triplets, a tie inside a bar, a unison bar between the two guitars and a Korean track
name stored in cp949.

Written bars (1-based) and play order: 1(pickup) 2 3 | 2 4 | 5 6  ->  expanded bars 0..6.
"""

from __future__ import annotations

from pathlib import Path

TEMPO = 120
TRACK1_NAME = "리듬 기타"          # stored cp949 in the file
TRACK2_NAME = "Lead"
CAPO1 = 2
STANDARD = [64, 59, 55, 50, 45, 40]  # string 1 (highest) .. string 6

# (numerator, denominator, repeat_open, repeat_close (pyguitarpro: number of repeats, -1 none), alternative mask)
HEADERS = [
    (1, 4, False, -1, 0),   # 1: pickup
    (4, 4, True, -1, 0),    # 2: repeat open
    (4, 4, False, 1, 1),    # 3: 1st ending, repeat close (play twice)
    (4, 4, False, -1, 2),   # 4: 2nd ending
    (6, 8, False, -1, 0),   # 5: 6/8
    (4, 4, False, -1, 0),   # 6: unison bar
]

# beat = (duration value, dotted, tuplet(enters, times) | None, [(string, fret, tie)] or None for a rest)
T1_BARS = [
    [(4, False, None, [(6, 0, False)])],
    [(4, False, None, [(5, 2, False)]), (4, False, None, [(4, 2, False)]), (4, False, None, [(3, 2, False)]),
     (4, False, None, [(2, 3, False)])],
    [(1, False, None, [(6, 0, False)])],
    [(1, False, None, [(6, 3, False)])],
    [(4, True, None, [(5, 0, False)]), (4, True, None, [(5, 2, False)])],
    [(4, False, None, [(1, 3, False)]), (4, False, None, [(1, 5, False)]), (2, False, None, [(1, 7, False)])],
]
T2_BARS = [
    [(4, False, None, None)],
    [(8, False, (3, 2), [(1, 5, False)]), (8, False, (3, 2), [(1, 7, False)]), (8, False, (3, 2), [(1, 8, False)]),
     (4, False, None, [(1, 10, False)]), (4, False, None, [(1, 10, True)]), (4, False, None, None)],
    [(2, False, None, [(2, 5, False)]), (2, False, None, None)],
    [(1, False, None, None)],
    [(8, False, None, [(3, 2, False)]), (8, False, None, [(3, 4, False)]), (8, False, None, [(3, 5, False)]),
     (4, True, None, None)],
    [(4, False, None, [(1, 5, False)]), (4, False, None, [(1, 7, False)]), (2, False, None, [(1, 9, False)])],
]

# Expected expanded notes: (line, bar, tick@48 per beat unit, dur_ticks, pitch, string, fret, shared)
EXPECTED = [
    ("L1", 0, 0, 48, 42, 6, 0, False),
    *[("L1", b, t, 48, p, s, f, False) for b in (1, 3)
      for t, p, s, f in ((0, 49, 5, 2), (48, 54, 4, 2), (96, 59, 3, 2), (144, 64, 2, 3))],
    *[("L2", b, t, 16, p, 1, f, False) for b in (1, 3) for t, p, f in ((0, 69, 5), (16, 71, 7), (32, 72, 8))],
    *[("L2", b, 48, 96, 74, 1, 10, False) for b in (1, 3)],
    ("L1", 2, 0, 192, 42, 6, 0, False),
    ("L2", 2, 0, 96, 64, 2, 5, False),
    ("L1", 4, 0, 192, 45, 6, 3, False),
    ("L1", 5, 0, 48, 47, 5, 0, False), ("L1", 5, 48, 48, 49, 5, 2, False),
    ("L2", 5, 0, 16, 57, 3, 2, False), ("L2", 5, 16, 16, 59, 3, 4, False), ("L2", 5, 32, 16, 60, 3, 5, False),
    ("L1", 6, 0, 48, 69, 1, 3, True), ("L1", 6, 48, 48, 71, 1, 5, True), ("L1", 6, 96, 96, 73, 1, 7, True),
    ("L2", 6, 0, 48, 69, 1, 5, True), ("L2", 6, 48, 48, 71, 1, 7, True), ("L2", 6, 96, 96, 73, 1, 9, True),
]
EXPECTED_BARS = [(0, 1, 4, "quarter"), (1, 4, 4, "quarter"), (2, 4, 4, "quarter"), (3, 4, 4, "quarter"),
                 (4, 4, 4, "quarter"), (5, 6, 8, "dotted_quarter"), (6, 4, 4, "quarter")]
PLAY_ORDER = [1, 2, 3, 2, 4, 5, 6]


def build_song(track1_name: str = TRACK1_NAME):
    import guitarpro as gp

    song = gp.Song(title="gtab fixture", artist="gtab", tempo=TEMPO)
    song.measureHeaders = []
    for i, (num, den, ropen, rclose, alt) in enumerate(HEADERS, 1):
        h = gp.MeasureHeader(number=i, timeSignature=gp.TimeSignature(num, gp.Duration(den)),
                             isRepeatOpen=ropen, repeatClose=rclose, repeatAlternative=alt)
        song.addMeasureHeader(h)
    song.tracks = []
    for ti, (name, capo, bars, program) in enumerate(((track1_name, CAPO1, T1_BARS, 29),
                                                     (TRACK2_NAME, 0, T2_BARS, 30)), 1):
        track = gp.Track(song, number=ti, name=name, offset=capo,
                         strings=[gp.GuitarString(n, v) for n, v in enumerate(STANDARD, 1)],
                         channel=gp.MidiChannel(channel=ti - 1, effectChannel=ti + 7, instrument=program))
        for measure, beats in zip(track.measures, bars, strict=True):
            voice = measure.voices[0]
            for value, dotted, tup, notes in beats:
                tuplet = gp.Tuplet(*tup) if tup else gp.Tuplet()
                beat = gp.Beat(voice, duration=gp.Duration(value, dotted, tuplet),
                               status=gp.BeatStatus.rest if notes is None else gp.BeatStatus.normal)
                for string, fret, tie in notes or ():
                    beat.notes.append(gp.Note(beat, value=fret, string=string,
                                              type=gp.NoteType.tie if tie else gp.NoteType.normal))
                voice.beats.append(beat)
        song.tracks.append(track)
    return song


def make_fixture_gp5(path: Path, *, track1_name: str = TRACK1_NAME, encoding: str = "cp949") -> Path:
    """Write the fixture as GP5 (strings encoded cp949 by default so the Korean track name survives)."""
    import guitarpro as gp

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gp.write(build_song(track1_name), str(path), version=(5, 1, 0), encoding=encoding)
    return path


if __name__ == "__main__":  # manual use: python tests/fixtures/gp_make.py out.gp5
    import sys

    print(make_fixture_gp5(Path(sys.argv[1] if len(sys.argv) > 1 else "fixture.gp5")))
