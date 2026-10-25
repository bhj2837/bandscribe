"""A tab video -> the tab's bars in play order (docs/TAB_VIDEO.md).

Tab videos show a strip or a page of the score and turn it as the music plays: frames that show the same bar lines are
one *page*. A page's bars are read in every one of its frames and voted; bar numbers (small digits above each bar) are
voted per page, and a page's first number is the start near the previous page's end that fits its readings best (a
read '24' fits 124: a leading digit is often lost). Bars on two pages vote together.

Measured on JINZO-TABS' 青春コンプレックス rhythm guitar (1080p, 2 frames/s): 160 bars in order with none missing,
98.8 % of a bar's frames agreeing; bar-bag F1 0.86 against the owner's band score of the same part (a different
transcriber: the bars checked by eye were read right and differ in the arrangement).
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


def read_frame(gray: np.ndarray, n: int = 6) -> list[dict[str, Any]]:
    """The bars of the frame's tab staff with the most fret numbers. Tabs are dark-on-light or light-on-dark in any part
    of the frame (a dark strip over a light video), so both polarities are searched."""
    best: list[dict[str, Any]] = []
    for g in (gray, 255 - gray):
        for st, _xs, _sp in tab_staves(g, n):
            bars = S.read_staff(g, st)
            if sum(len(c) for b in bars for c in b["cols"]) > sum(len(c) for b in best for c in b["cols"]):
                best = bars
    return best


def pages(frames: Iterable[np.ndarray], n: int = 6) -> list[dict[str, Any]]:
    """Consecutive frames with the same bar lines (within 3 px) grouped: {bars: [(x0, x1)], frames, readings: [Counter
    of column tuples per bar], numbers: [Counter of read numbers per bar], phases}."""
    out: list[dict[str, Any]] = []
    for g in frames:
        bars = read_frame(g, n)
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


def read_video(frames: list[Path], n: int = 6) -> dict[str, Any]:
    pgs = pages(iter_frames(frames), n)
    bars = assemble(number_pages(pgs))
    return {"format": FORMAT, "strings": n, "frames": len(frames), "pages": len(pgs),
            "stable_pages": sum(1 for p in pgs if p["frames"] >= 2), "bars": bars,
            "agree": round(float(np.mean([b["agree"] for b in bars])), 4) if bars else None}
