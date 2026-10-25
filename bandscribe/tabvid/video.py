"""A tab video -> the tab's bars in play order (docs/TAB_VIDEO.md).

Tab videos show a strip or a page of the score and turn it as the music plays: frames that show the same bar lines are
one *page*. A page's bars are read in every one of its frames and voted; bar numbers (small digits above each bar) are
voted per page, and a page's first number is the start near the previous page's end that fits its readings best (a
read '24' fits 124: a leading digit is often lost). Bars on two pages vote together. A frame with several tab staves
(Guitar 1 over Guitar 2) gives one *track* per staff height, each read on its own.

Measured on JINZO-TABS' 青春コンプレックス rhythm guitar (1080p, 2 frames/s): bars 2-158 in order with none missing,
98.9 % of a bar's frames agreeing; bar-bag F1 0.86 against the owner's band score of the same part (a different
transcriber: the bars checked by eye were read right and differ in the arrangement). On Rika's あのバンド video against
the same transcriber's band score: F1 0.979 / 0.861 / 0.922 for guitar 1 / guitar 2 / bass.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from bandscribe.tabvid import staff as S

FORMAT = "bandscribe.tabvid/1"


def extract_frames(video: Path, out_dir: Path, *, fps: float = 2.0) -> list[Path]:
    """Grey PNG frames of ``video`` at ``fps`` (ffmpeg); reuses frames already there."""
    out_dir = Path(out_dir)
    if not any(out_dir.glob("f*.png")):
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps},format=gray",
                        str(out_dir / "f%05d.png")], check=True, timeout=3600)
    return sorted(out_dir.glob("f*.png"))


def tab_staves(gray: np.ndarray, n: int = 6) -> list[tuple[list[float], list[int], float]]:
    """Staves that hold a tab: bars wider than 4 line gaps (a notation staff misread as lines has stems for bar lines),
    gaps in the lines (fret numbers), a light background."""
    out = []
    for st in S.find_staves(gray, n):
        xs, sp = S.barlines(gray, st)
        if len(xs) < 2 or float(np.median(np.diff(xs))) < 4 * sp:
            continue
        y0 = max(0, int(st[0] - sp))
        sub = gray[y0:int(st[-1] + sp)].astype(float)
        if sum(len(S.line_gaps(sub, ly, y0, sp)) for ly in st) < 2:
            continue
        if float(np.median(gray[int(st[0]):int(st[-1]) + 1, xs[0]:xs[-1]])) < 170:
            continue
        out.append((st, xs, sp))
    return out


def _notes(bars: list[dict[str, Any]]) -> int:
    return sum(len(c) for b in bars for c in b["cols"])


def _same_lines(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    return len(a) == len(b) and all(abs(p["x"][0] - q["x"][0]) <= 3 and abs(p["x"][1] - q["x"][1]) <= 3
                                    for p, q in zip(a, b))


def frame_staves(gray: np.ndarray, n: int = 6) -> list[dict[str, Any]]:
    """Every tab staff of the frame, top to bottom: {y (top line), sp, bars}. Tabs are dark-on-light or light-on-dark in
    any part of the frame (a dark strip over a light video), so both polarities are searched and the one with more fret
    numbers kept. Staves with the same bar lines are one system: the ones below the first take its bar numbers (they
    are printed once, above the system)."""
    best: list[dict[str, Any]] = []
    for g in (gray, 255 - gray):
        out: list[dict[str, Any]] = []
        prev = None
        for st, _xs, sp in sorted(tab_staves(g, n), key=lambda t: t[0][0]):
            bars = S.read_staff(g, st, prev)
            prev = st[-1]
            for up in out:
                if _same_lines(up["bars"], bars):
                    for b, u in zip(bars, up["bars"]):
                        if b["number"] is None:
                            b["number"] = u["number"]
                    break
            out.append({"y": float(st[0]), "sp": float(sp), "bars": bars})
        if sum(_notes(t["bars"]) for t in out) > sum(_notes(t["bars"]) for t in best):
            best = out
    return best


def read_frame(gray: np.ndarray, n: int = 6) -> list[dict[str, Any]]:
    """The bars of the frame's tab staff with the most fret numbers."""
    return max((t["bars"] for t in frame_staves(gray, n)), key=_notes, default=[])


def pages(frames: Iterable[np.ndarray], n: int = 6) -> list[dict[str, Any]]:
    """``pages_of`` the staff with the most fret numbers in each frame."""
    return pages_of(read_frame(g, n) for g in frames)


def pages_of(readings: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Consecutive frames' readings (bars of one staff) with the same bar lines (within 3 px) grouped: {bars: [(x0,
    x1)], frames, readings: [Counter of column tuples per bar], numbers: [Counter of read numbers per bar], phases}."""
    out: list[dict[str, Any]] = []
    for bars in readings:
        if not bars:
            continue
        sig = [b["x"] for b in bars]
        same = bool(out) and len(out[-1]["bars"]) == len(sig) and all(
            abs(a[0] - b[0]) <= 3 and abs(a[1] - b[1]) <= 3 for a, b in zip(out[-1]["bars"], sig))
        if same:  # the same layout can show the next bars: the content must agree too (numbers misread too often)
            top = [c.most_common(1)[0][0] if c else None for c in out[-1]["readings"]]
            now = [tuple(tuple(c) for c in b["cols"]) for b in bars]
            same = sum(1 for t, w in zip(top, now) if t == w) >= 0.5 * len(now)
        if not same:
            out.append({"bars": sig, "frames": 0, "readings": [Counter() for _ in sig],
                        "numbers": [Counter() for _ in sig], "phases": [b["phases"] for b in bars]})
        pg = out[-1]
        pg["frames"] += 1
        for i, b in enumerate(bars):
            pg["readings"][i][tuple(tuple(c) for c in b["cols"])] += 1
            if b["number"] is not None:
                pg["numbers"][i][b["number"]] += 1
    return out


def fits(read: int, n: int) -> float:
    """1 when the read number is n, 0.5 when it is n's last one or two digits (a leading digit lost), else 0."""
    if read == n:
        return 1.0
    if n >= 10 and (read == n % 100 or read == n % 10):
        return 0.5
    return 0.0


def number_pages(pgs: list[dict[str, Any]], *, min_frames: int = 2, back: int = 6, ahead: int = 15
                 ) -> list[dict[str, Any]]:
    """[{number, readings, phases}] in play order (module doc). A page shown for fewer than ``min_frames`` frames is a
    turn in progress and is skipped; a page's start is searched from ``back`` bars before to ``ahead`` after the end
    of the previous page."""
    seq: list[dict[str, Any]] = []
    last = 0
    for pg in pgs:
        if pg["frames"] < min_frames:
            continue
        best = None
        starts = range(max(1, last - back), last + ahead + 1)
        if last == 0:  # the first page may start anywhere: its own readings propose the starts
            starts = sorted({rd - i for i, c in enumerate(pg["numbers"]) for rd in c if rd - i >= 1} | {1})
        for s0 in starts:
            sc = sum(k * fits(rd, s0 + i) for i, c in enumerate(pg["numbers"]) for rd, k in c.items())
            if last:  # pages follow on (overlapping by a bar or two): a jump must be paid for by exact readings
                sc -= 0.25 * pg["frames"] * max(0, abs(s0 - (last + 1)) - 2)
            key = (sc, -abs(s0 - (last + 1)))
            if best is None or key > best[0]:
                best = (key, s0)
        assert best is not None
        first = best[1] if best[0][0] > 0 or last else last + 1
        for i, rd in enumerate(pg["readings"]):
            seq.append({"number": first + i, "readings": rd, "phases": pg["phases"][i]})
        last = max(last, first + len(pg["readings"]) - 1)
    return seq


def assemble(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One reading per bar number: the most voted over all its frames; {number, cols, agree, frames, phases}."""
    acc: dict[int, Counter] = {}
    phases: dict[int, list[float]] = {}
    for b in seq:
        acc.setdefault(b["number"], Counter()).update(b["readings"])
        phases.setdefault(b["number"], b["phases"])
    out = []
    for num in sorted(acc):
        (cols, k), = acc[num].most_common(1)
        tot = sum(acc[num].values())
        ph = phases[num] if len(phases[num]) == len(cols) else [round((i + 0.5) / max(1, len(cols)), 4)
                                                                 for i in range(len(cols))]
        out.append({"number": num, "cols": [list(c) for c in cols], "agree": round(k / tot, 3), "frames": tot,
                    "phases": ph})
    return out


def iter_frames(paths: Iterable[Path]) -> Iterator[np.ndarray]:
    from PIL import Image

    for p in paths:
        yield np.asarray(Image.open(p).convert("L"))


def tracks(per_frame: list[list[dict[str, Any]]]) -> list[tuple[float, list[list[dict[str, Any]]]]]:
    """The staves of every frame sorted into tracks by height (a staff within 1.5 line gaps of a track's height is that
    track; a video keeps each part's staff at one height): [(height, its bars in every frame, [] where absent)]."""
    ys: list[float] = []
    for staves in per_frame:
        for t in staves:
            if not any(abs(t["y"] - y) <= 1.5 * t["sp"] for y in ys):
                ys.append(t["y"])
    ys.sort()
    out: list[tuple[float, list[list[dict[str, Any]]]]] = [(y, [[] for _ in per_frame]) for y in ys]
    for i, staves in enumerate(per_frame):
        for t in staves:
            k = min(range(len(ys)), key=lambda j: abs(ys[j] - t["y"]))
            out[k][1][i] = t["bars"]
    return out


def read_track(readings: list[list[dict[str, Any]]]) -> dict[str, Any]:
    pgs = pages_of(readings)
    bars = assemble(number_pages(pgs))
    return {"pages": len(pgs), "stable_pages": sum(1 for p in pgs if p["frames"] >= 2), "bars": bars,
            "notes": sum(len(c) for b in bars for c in b["cols"]),
            "agree": round(float(np.mean([b["agree"] for b in bars])), 4) if bars else None}


def read_video(frames: list[Path], n: int = 6) -> dict[str, Any]:
    """Every tab track of the video (each staff height: Guitar 1, Guitar 2 ... on a multi-part video), top to bottom;
    ``bars`` / ``agree`` / ``pages`` repeat the track with the most fret numbers."""
    per_frame = [frame_staves(g, n) for g in iter_frames(frames)]
    tr = [dict(read_track(r), y=round(y, 1)) for y, r in tracks(per_frame)]
    main = max(tr, key=lambda t: t["notes"], default={"pages": 0, "stable_pages": 0, "bars": [], "agree": None})
    return {"format": FORMAT, "strings": n, "frames": len(frames), "tracks": tr, "pages": main["pages"],
            "stable_pages": main["stable_pages"], "bars": main["bars"], "agree": main["agree"]}
