"""alphaTex writer (alphaTab 1.8 syntax, checked against alphaTab 1.8.4's own parser and AlphaTexExporter).

Tracks share the master bars (meter, tempo, triplet feel, sections), written in the first track; every bar of
every track holds its spelled beats (``bandscribe.tab.notation``). Fretted tracks are ``\\staff {tabs}`` with
``fret.string`` notes (string 1 = highest; ``\\tuning`` lists open strings highest first); unfretted tracks are
``\\staff {score}`` with pitched notes (``C#4``). Ties are the note property ``{t}``, dots and tuplets beat
properties ``{d}`` / ``{tu (3 2)}``. Every metadata argument is parenthesised: an unwrapped ``\\section "A"``
takes the next note as its second argument (AIZO, 2026-10-05). One ``\\sync (bar 0 ms)`` per bar puts every bar
line on the recording's time (the grid's bar start), so a player following external media stays on the audio
(DESIGN 6.2 "두 시간축"; measured in the viewer: cursor within 31 ms of the grid over a whole song).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from bandscribe.tab.notation import WBar, WBeat

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(midi: int) -> str:
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def _q(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', "'") + '"'


@dataclass
class WTrack:
    name: str
    short: str
    instrument: str               # alphaTab / GM instrument name, e.g. "electricbassfinger"
    fretted: bool
    tuning: list[int] | None      # open pitches, lowest string first (fretted tracks)
    capo: int
    beats: list[list[WBeat]]      # per bar of the score
    clef: str | None = None       # "f4" for a bass staff in score view


@dataclass
class WScore:
    title: str
    bars: list[WBar]
    tempo_qpm: list[int]          # written tempo per bar (quarter notes per minute)
    tracks: list[WTrack] = field(default_factory=list)
    artist: str | None = None


def _note(n, fretted: bool) -> str:
    base = f"{n.fret}.{n.string}" if fretted else pitch_name(n.pitch)
    return base + ("{t}" if n.tie else "")


def _beat(be: WBeat, fretted: bool) -> str:
    if not be.notes:
        body = "r"
    elif len(be.notes) == 1:
        body = _note(be.notes[0], fretted)
    else:
        body = "(" + " ".join(_note(n, fretted) for n in be.notes) + ")"
    props = []
    if be.dots:
        props.append("d" if be.dots == 1 else "dd")
    if be.tuplet:
        props.append(f"tu ({be.tuplet[0]} {be.tuplet[1]})")
    return f"{body}.{be.value}" + ("{" + " ".join(props) + "}" if props else "")


def write(score: WScore) -> str:
    lines = [f"\\title {_q(score.title)}"]
    if score.artist:
        lines.append(f"\\artist {_q(score.artist)}")
    lines.append(f"\\tempo ({score.tempo_qpm[0] if score.tempo_qpm else 120})")
    for ti, tr in enumerate(score.tracks):
        lines.append(f"\\track ({_q(tr.name)} {_q(tr.short)})")
        lines.append("\\staff {tabs}" if tr.fretted else "\\staff {score}")
        lines.append(f"\\instrument ({tr.instrument})")
        if tr.fretted and tr.tuning:
            lines.append("\\tuning (" + " ".join(pitch_name(p) for p in reversed(tr.tuning)) + ")")
            if tr.capo:
                lines.append(f"\\capo ({tr.capo})")
        elif tr.clef:
            lines.append(f"\\clef ({tr.clef})")
        prev_sig, prev_tempo, prev_feel = None, None, None
        for bi, (bar, beats) in enumerate(zip(score.bars, tr.beats)):
            meta = []
            if ti == 0:
                sig = (bar.numerator, bar.denominator)
                if sig != prev_sig:
                    meta.append(f"\\ts ({sig[0]} {sig[1]})")
                    prev_sig = sig
                tq = score.tempo_qpm[bi] if bi < len(score.tempo_qpm) else prev_tempo
                if tq != prev_tempo and bi > 0:
                    meta.append(f"\\tempo ({tq})")
                prev_tempo = tq
                feel = bar.feel or "none"
                if feel != (prev_feel or "none"):
                    meta.append(f"\\tf ({feel})")
                prev_feel = feel
                if bar.section:
                    meta.append(f"\\section ({_q(bar.section)})")
            body = " ".join(_beat(be, tr.fretted) for be in beats)
            lines.append((" ".join(meta) + " " if meta else "") + body + " |")
    last = -1.0
    for bi, bar in enumerate(score.bars):
        ms = round(bar.start_s * 1000.0)
        if ms > last:
            lines.append(f"\\sync ({bi} 0 {ms})")
            last = ms
    return "\n".join(lines) + "\n"


def written_tempi(bars: Sequence[WBar], *, section_starts: Sequence[int] = (), change_ratio: float = 0.04) -> list[int]:
    """A readable tempo per bar (quarter notes per minute): the song's median, changed only at a section start
    whose median differs by more than ``change_ratio`` (the bar lines follow the audio through sync points)."""
    factor = {"quarter": 1.0, "dotted_quarter": 1.5, "eighth": 0.5}
    qpm = []
    for i, b in enumerate(bars):
        nxt = bars[i + 1].start_s if i + 1 < len(bars) else None
        dur = (nxt - b.start_s) if nxt is not None else None
        qpm.append(60.0 * b.n_beats * factor[b.beat_unit] / dur if dur and dur > 0 else float("nan"))
    import numpy as np

    vals = np.array(qpm, dtype=float)
    ok = np.isfinite(vals) & (vals > 20) & (vals < 400)
    song = float(np.median(vals[ok])) if ok.any() else 120.0
    starts = set(section_starts)
    bounds = sorted({0} | {i for i, b in enumerate(bars) if b.bar in starts}) + [len(bars)]
    out: list[int] = []
    cur = int(round(song))
    for a, e in zip(bounds, bounds[1:]):
        seg = vals[a:e]
        seg = seg[np.isfinite(seg) & (seg > 20) & (seg < 400)]
        if seg.size >= 2 and abs(float(np.median(seg)) - cur) / cur > change_ratio:
            cur = int(round(float(np.median(seg))))
        out.extend([cur] * (e - a))
    return out
