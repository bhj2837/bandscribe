"""A tab staff in one image: its lines, its bar lines, the fret numbers in each bar, and the bar numbers
(docs/TAB_VIDEO.md).

Reading is box-first: Guitar Pro, alphaTab, TuxGuitar and Songsterr draw each fret number on a small white box that
cuts its staff line, while stems, ties and slashes cross the line without a gap. So every gap in a staff line is a
candidate number, and the ink inside the gap's box is the number, cut into digits by empty columns. This is what made a
compressed 1080p video readable (a component-based reading broke on blurred strokes, stems and tie arcs).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy import ndimage

from bandscribe.tabvid import glyphs as G


# ------------------------------------------------------------------------------------------------- staff lines


def _groups(ys: list[float], n: int, tol: float | None = None) -> list[list[float]]:
    out, i = [], 0
    while i + n <= len(ys):
        grp = ys[i:i + n]
        d = np.diff(grp)
        lim = tol if tol is not None else max(2.0, 0.1 * float(d.mean()))
        if d.min() > 4 and d.max() - d.min() <= lim:
            out.append(grp)
            i += n
            continue
        i += 1
    return out


def staves_strict(gray: np.ndarray, n: int = 6) -> list[list[float]]:
    """Rows darker than 200 across half the width, merged; n equally spaced lines (within 2 px). Its line positions
    (half-pixel for a two-row line) read best on the formats it finds."""
    dark = gray < 200
    rows = np.flatnonzero(dark.sum(1) > 0.5 * gray.shape[1])
    lines: list[list[int]] = []
    for r in rows:
        if lines and r - lines[-1][-1] <= 1:
            lines[-1].append(int(r))
        else:
            lines.append([int(r)])
    return _groups([float(np.mean(x)) for x in lines], n, tol=2.0)


def line_rows(gray: np.ndarray) -> list[float]:
    """Thin rows (1-4 px) of ink across 30 % of the width, judged against the bright background; for pale lines and
    frames where a dark video sits next to the tab."""
    g = gray.astype(int)
    h, w = g.shape
    bg = float(np.percentile(g, 90))
    cnt = (g < bg - 15).sum(1)
    cand = cnt > 0.3 * w
    lines, y = [], 0
    while y < h:
        if not cand[y]:
            y += 1
            continue
        y1 = y
        while y1 + 1 < h and cand[y1 + 1]:
            y1 += 1
        peak = cnt[y:y1 + 1].max()
        above = cnt[y - 2] if y >= 2 else 0
        below = cnt[y1 + 2] if y1 + 2 < h else 0
        if y1 - y + 1 <= 4 and above < 0.5 * peak and below < 0.5 * peak:
            lines.append((y + y1) / 2)
        y = y1 + 1
    return lines


def find_staves(gray: np.ndarray, n: int = 6) -> list[list[float]]:
    """The strict finder's staves plus the robust finder's staves it missed."""
    a = staves_strict(gray, n)
    return a + [s for s in _groups(line_rows(gray), n) if not any(abs(s[0] - t[0]) < 5 for t in a)]


def barlines(gray: np.ndarray, st: list[float]) -> tuple[list[int], float]:
    """(x of each bar line, line spacing). A bar line spans the staff exactly; a stem or a slash goes on beyond it."""
    top, bot = int(round(st[0])), int(round(st[-1]))
    sp = (st[-1] - st[0]) / (len(st) - 1)
    frac = (gray[top:bot + 1] < 160).mean(0)
    below = gray[bot + 3:bot + int(sp)] < 160
    above = gray[max(0, top - int(sp)):max(0, top - 2)] < 160
    xs: list[int] = []
    for x in np.flatnonzero(frac > 0.97):
        if (below.size and below[:, x].mean() > 0.5) or (above.size and above[:, x].mean() > 0.5):
            continue
        if xs and x - xs[-1] <= 2:
            continue
        xs.append(int(x))
    return xs, sp


# ------------------------------------------------------------------------------------------------- numbers


def line_gaps(sub: np.ndarray, ly: float, y0: int, sp: float) -> list[tuple[int, int]]:
    """Runs of a staff line (rows around ``ly``) where the line's own grey is missing: the white boxes behind
    numbers."""
    on = np.zeros(sub.shape[1], bool)
    for rr in {int(np.floor(ly)) - y0, int(np.ceil(ly)) - y0}:
        if not 0 <= rr < sub.shape[0]:
            continue
        row = sub[rr]
        dark = row < 235
        if not dark.any():
            continue
        lg = float(np.median(row[dark]))
        on |= np.abs(row - lg) <= max(25.0, 0.25 * (255 - lg))
    on = ndimage.binary_opening(on, structure=np.ones(3))  # specks of line grey inside a box
    k = max(3, int(0.3 * sp))  # digit pixels as grey as the line split one box: rejoin
    gap = ndimage.binary_closing(np.r_[np.zeros(k, bool), ~on, np.zeros(k, bool)], structure=np.ones(k))[k:-k]
    edges = np.flatnonzero(np.diff(np.r_[0, gap.astype(int), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])
            if 0.3 * sp <= b - a <= 2.8 * sp and a > 0 and b < len(gap)]


def _digits(piece: np.ndarray, h: int, *, digits_only: bool = False) -> list[tuple[str | None, float]]:
    """A piece wider than one digit (digits joined by antialiasing, '00') split evenly by its width; one entry per
    digit with its (label, score)."""
    w = piece.shape[1]
    k = max(1, int(round(w / (0.62 * h)))) if w > 1.25 * h else 1  # one glyph ("X", "4") is at most ~1.1 h wide
    if k == 1:
        return [G.classify(piece, digits_only=digits_only)]
    edges = np.linspace(0, w, k + 1).round().astype(int)
    return [G.classify(piece[:, a:b], digits_only=True) for a, b in zip(edges, edges[1:])]


def _pieces(ink: np.ndarray) -> list[tuple[int, int, np.ndarray, int]]:
    cols = (ink > 0.3).any(0)
    edges = np.flatnonzero(np.diff(np.r_[0, cols.astype(int), 0]))
    out = []
    for c0, c1 in zip(edges[::2], edges[1::2]):
        crop = ink[:, c0:c1]
        rows = np.flatnonzero((crop > 0.3).any(1))
        if len(rows):
            out.append((int(c0), int(c1), crop[rows.min():rows.max() + 1], int(rows.max() - rows.min() + 1)))
    return out


def _strip_stems(ink: np.ndarray, ya: int, yb: int, g0: int, g1: int) -> np.ndarray:
    """The box's ink without the vertical runs that go on past its top or bottom edge (a stem or a slash that starts at
    the number)."""
    box = ink[ya:yb, g0:g1].copy()
    on = box > 0.3
    for c in range(box.shape[1]):
        if yb < ink.shape[0] and ink[yb:yb + 2, g0 + c].max() > 0.3:
            j = box.shape[0] - 1
            while j >= 0 and on[j, c]:
                box[j, c] = 0.0
                j -= 1
        if ya > 0 and ink[max(0, ya - 2):ya, g0 + c].max() > 0.3:
            j = 0
            while j < box.shape[0] and on[j, c]:
                box[j, c] = 0.0
                j += 1
    return box


def _trim_edges(ink: np.ndarray, box: np.ndarray, ya: int, g0: int, r: int) -> np.ndarray:
    """The box without a tie arc or a slur that starts next to the number and shares its columns: when ink at the box's
    left or right edge goes on outside it (off the staff line's own rows, ``r``), the thin columns from that edge
    inward are cleared (a digit's own columns hold more ink), then the arc's tail left under the digit."""
    box = box.copy()
    hb, wb = box.shape
    off_line = np.abs(np.arange(hb) + ya - r) > 1
    thin = max(3, int(0.2 * hb))
    for edge, step, outside in ((0, 1, g0 - 1), (wb - 1, -1, g0 + wb)):
        if not 0 <= outside < ink.shape[1]:
            continue
        rows = np.flatnonzero((box[:, edge] > 0.3) & off_line)
        if not len(rows):
            continue
        near = np.unique(np.clip(np.r_[rows - 1, rows, rows + 1], 0, hb - 1))
        near = near[off_line[near]] + ya
        if not len(near) or ink[near[near < ink.shape[0]], outside].max(initial=0.0) <= 0.3:
            continue
        c = edge
        while 0 <= c < wb and (box[:, c] > 0.3).sum() <= thin:
            box[:, c] = 0.0
            c += step
        if c != edge:
            lab, n = ndimage.label(box > 0.3, structure=np.ones((3, 3)))
            objs = ndimage.find_objects(lab)
            if n > 1:
                hmax = max(sl[0].stop - sl[0].start for sl in objs)
                for k, sl in enumerate(objs, 1):
                    if sl[0].stop - sl[0].start < 0.25 * hmax:
                        box[lab == k] = 0.0
    return box


def read_numbers(gray: np.ndarray, st: list[float], sp: float) -> list[dict[str, Any]]:
    """Every fret number on the staff: {line (0 = top), x0, x1, cx, text ('0'-'24' or 'x'), tied, conf}."""
    y0, y1 = max(0, int(st[0] - sp)), int(st[-1] + sp)
    sub = gray[y0:y1].astype(float)
    bg = float(np.median(sub))
    ink_all = np.clip((bg - 10 - sub) / max(1.0, bg - 10), 0, 1)
    nums = []
    for li, ly in enumerate(st):
        r = int(round(ly)) - y0
        ya, yb = max(0, int(r - 0.5 * sp)), min(sub.shape[0], int(r + 0.5 * sp) + 1)
        for g0, g1 in line_gaps(sub, ly, y0, sp):
            raw = ink_all[ya:yb, g0:g1]
            on = np.flatnonzero((raw > 0.3).any(0))
            if g1 - g0 < 0.45 * sp and len(on) and on[0] == 0 and on[-1] == g1 - g0 - 1:
                continue  # no white box around it: a stroke over the line (the 'A' of the TAB clef), not a number
            box = _strip_stems(ink_all, ya, yb, g0, g1)
            if box.sum() < 0.5 * raw.sum():  # the stroke itself ran on into the stem (a '1' with its stem): keep it
                box = raw
            box = _trim_edges(ink_all, box, ya, g0, r)
            pieces = _pieces(box)
            if not pieces:
                continue
            hmax = max(p[3] for p in pieces)
            if hmax < 0.4 * sp:
                continue
            text, tied, conf = "", False, []
            for c0, c1, crop, h in pieces:
                if h < 0.55 * hmax:  # a speck or a stem stub
                    continue
                if c1 - c0 >= 3 and (crop > 0.5).mean() > 0.85:  # a solid block: a whole or half rest
                    continue
                if G.bracket(crop, h, c1 - c0):
                    tied = True
                    continue
                for d, sc in _digits(crop, h):
                    if d in ("(", ")"):
                        tied = True
                        continue
                    if d == "rest" or sc < 0.5:
                        continue
                    text += str(d)
                    conf.append(sc)
            if not text:
                continue
            nums.append({"line": li, "x0": g0, "x1": g1, "cx": (g0 + g1) / 2, "text": "x" if "x" in text else text,
                         "tied": tied, "conf": round(min(conf), 3)})
    return nums


def bar_numbers(gray: np.ndarray, st: list[float], xs: list[int], sp: float, ceiling: float | None = None
                ) -> dict[int, int]:
    """{bar line x: number}: the small digits right of each bar line anywhere above the staff (Guitar Pro prints them
    above the top staff of the system, e.g. the notation staff), up to ``ceiling`` (the tab staff above, on a page
    with several systems). They are faint grey in videos."""
    top = st[0]
    y0, y1 = max(0, int(top - 12 * sp)), max(0, int(top - 0.3 * sp))
    if ceiling is not None:
        y0 = max(y0, int(ceiling + 0.6 * sp))
    if y1 <= y0:
        return {}
    sub = gray[y0:y1].astype(float)
    bg = float(np.median(sub))
    lab, _ = ndimage.label(sub < bg - 25, structure=np.ones((3, 3)))
    found = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (0.3 * sp <= h <= 0.75 * sp) or w > 2.2 * sp:
            continue
        crop = np.where(lab[sl] == k, np.clip((bg - sub[sl]) / bg, 0, 1), 0.0)
        for c0, c1, piece, ph in _pieces(crop):  # faint antialiasing can join two small digits
            if ph < 0.6 * h:
                continue
            ds = _digits(piece, ph)
            if all(sc >= (0.7 if len(ds) == 1 else 0.6) and str(d).isdigit() for d, sc in ds):
                step = (c1 - c0) / len(ds)
                for i, (d, _sc) in enumerate(ds):
                    found.append((sl[0].start, sl[1].start + c0 + int(i * step), sl[1].start + c0 + int((i + 1) * step),
                                  str(d)))
    unit = max(1, int(0.3 * sp))
    windows = []
    for x, nxt in zip(xs, list(xs[1:]) + [None]):
        right = x + 3.5 * sp if nxt is None else min(x + 3.5 * sp, nxt - 0.4 * sp)
        windows.append((x, [f for f in found if x - 0.4 * sp <= f[1] <= right]))
    # the bar numbers of a staff share one row: the row with digits at the most bar lines (then the nearest the staff)
    cover: Counter = Counter()
    for _x, cand in windows:
        for r in {f[0] // unit for f in cand}:
            cover[r] += 1
    if not cover:
        return {}
    best = max(cover, key=lambda r: (cover[r], r))
    out = {}
    for x, cand in windows:
        digs = sorted((f for f in cand if abs(f[0] // unit - best) <= 1), key=lambda f: f[1])
        if not digs:
            continue
        text, last = digs[0][3], digs[0][2]
        for f in digs[1:]:
            if f[1] - last <= 0.4 * sp:
                text += f[3]
                last = f[2]
        out[x] = int(text)
    return out


def read_staff(gray: np.ndarray, st: list[float], ceiling: float | None = None) -> list[dict[str, Any]]:
    """The staff's bars: {x: (left, right bar line), number (or None), cols: [[(string, text), ...] per column]}.
    String 1 is the top line. Tied notes (in brackets) are left out, as the sheet evaluation does. ``ceiling``: the
    bottom line of the tab staff above (bar numbers are not searched beyond it)."""
    xs, sp = barlines(gray, st)
    nums = read_numbers(gray, st, sp)
    numbers = bar_numbers(gray, st, xs, sp, ceiling)
    bars = []
    for a, b in zip(xs, xs[1:]):
        cols: list[list[dict[str, Any]]] = []
        cur = None
        for nm in sorted((n for n in nums if a < n["cx"] < b), key=lambda n: n["cx"]):
            if cur is not None and abs(nm["cx"] - cur) <= 0.6 * sp:
                cols[-1].append(nm)
            else:
                cols.append([nm])
                cur = nm["cx"]
        bars.append({"x": (a, b), "number": numbers.get(a),
                     "cols": [sorted((n["line"] + 1, n["text"]) for n in c if not n["tied"]) for c in cols],
                     "phases": [round((float(np.mean([n["cx"] for n in c])) - a) / max(1, b - a), 4) for c in cols]})
    return bars


def read_page(gray: np.ndarray, n: int = 6) -> list[dict[str, Any]]:
    """Every tab staff of a page image in reading order (top to bottom), each limited to the space below the one
    above."""
    out: list[dict[str, Any]] = []
    prev = None
    for st in sorted(find_staves(gray, n), key=lambda s: s[0]):
        out += read_staff(gray, st, prev)
        prev = st[-1]
    return out
