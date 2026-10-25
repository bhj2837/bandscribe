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
    """Runs of n equally spaced lines with no further line at that spacing above or below (four lines of a six-line
    staff are not a bass staff)."""
    out, i = [], 0
    while i + n <= len(ys):
        grp = ys[i:i + n]
        d = np.diff(grp)
        lim = tol if tol is not None else max(2.0, 0.1 * float(d.mean()))
        if d.min() > 4 and d.max() - d.min() <= lim:
            dm = float(d.mean())
            if not any(abs(y - (grp[0] - dm)) <= lim or abs(y - (grp[-1] + dm)) <= lim for y in ys):
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
    """The strict finder's staves, or the robust finder's when it finds none. Measured on nine formats: the robust one
    was needed only where the strict one found nothing, and next to a strict staff it took a notation staff with a row
    of ledger lines for a sixth line (JINZO)."""
    return staves_strict(gray, n) or _groups(line_rows(gray), n)


def barlines(gray: np.ndarray, st: list[float]) -> tuple[list[int], float]:
    """(x of each bar line, line spacing). A bar line spans the staff exactly; a stem or a slash goes on beyond it. The
    one exception is the line where the staff starts: on a score page it joins the staves of a system."""
    top, bot = int(round(st[0])), int(round(st[-1]))
    sp = (st[-1] - st[0]) / (len(st) - 1)
    frac = (gray[top:bot + 1] < 160).mean(0)
    below = gray[bot + 3:bot + int(sp)] < 160
    above = gray[max(0, top - int(sp)):max(0, top - 2)] < 160
    mid = np.flatnonzero(gray[int(round(st[len(st) // 2]))] < 200)
    start = int(mid[0]) if len(mid) else -1  # where the staff's lines begin
    xs: list[int] = []
    for x in np.flatnonzero(frac > 0.97):
        if (below.size and below[:, x].mean() > 0.5) or (above.size and above[:, x].mean() > 0.5):
            if xs or x > start + sp:
                continue
        if xs and x - xs[-1] <= 2:
            continue
        if xs and x - xs[-1] < 0.8 * sp:  # a double bar line (a section or a repeat): one bar line, the later one
            xs[-1] = int(x)
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
            if 0.3 * sp <= b - a <= 30 * sp and a > 0 and b < len(gap)]  # wide: runs of close boxes merge


def _digits(piece: np.ndarray, h: int, *, digits_only: bool = False) -> list[tuple[str | None, float]]:
    """A piece wider than one digit (digits joined by antialiasing, '00') split evenly by its width; one entry per
    digit with its (label, score)."""
    w = piece.shape[1]
    k = max(1, int(round(w / (0.62 * h)))) if w > 1.25 * h else 1  # one glyph ("X", "4") is at most ~1.1 h wide
    if k == 1:
        return [G.classify(piece, digits_only=digits_only)]
    edges = np.linspace(0, w, k + 1).round().astype(int)
    return [G.classify(piece[:, a:b], digits_only=True) for a, b in zip(edges, edges[1:])]


def _pieces(ink: np.ndarray, thr: float = 0.3) -> list[tuple[int, int, np.ndarray, int]]:
    """(left, right, crop, height) of each run of columns holding ink above ``thr``."""
    cols = (ink > thr).any(0)
    edges = np.flatnonzero(np.diff(np.r_[0, cols.astype(int), 0]))
    out = []
    for c0, c1 in zip(edges[::2], edges[1::2]):
        crop = ink[:, c0:c1]
        rows = np.flatnonzero((crop > thr).any(1))
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
            digs, tied = [], False
            for c0, c1, crop, h in pieces:
                if h < 0.7 * hmax:  # a speck, a stem stub or a bend arrow's head (a number's digits share a height)
                    continue
                if c1 - c0 < max(2, 0.12 * sp):  # a hairline left of an arc or a stem; a '1' is a stroke and a flag
                    continue
                if c1 - c0 >= 3 and (crop > 0.5).mean() > 0.85:  # a solid block: a whole or half rest
                    continue
                if G.bracket(crop, h, c1 - c0):
                    tied = True
                    continue
                ds = _digits(crop, h)
                for i, (d, sc) in enumerate(ds):
                    if d in ("(", ")"):
                        tied = True
                        continue
                    if d == "rest" or sc < 0.5:
                        continue
                    w = (c1 - c0) / len(ds)
                    digs.append((str(d), sc, c0 + i * w, c0 + (i + 1) * w))
            if not digs:
                continue
            if any(d[0] == "x" for d in digs):
                nums.append({"line": li, "x0": g0, "x1": g1, "cx": (g0 + g1) / 2, "text": "x", "tied": tied,
                             "conf": round(min(d[1] for d in digs), 3)})
                continue
            parts = _frets(digs)
            for part in parts:  # numbers sharing a box (close notes, a slide between) keep their own extent
                x0, x1 = (g0, g1) if len(parts) == 1 else (g0 + int(part[0][2]) - 1, g0 + int(np.ceil(part[-1][3])) + 1)
                nums.append({"line": li, "x0": x0, "x1": x1, "cx": g0 + (part[0][2] + part[-1][3]) / 2,
                             "text": "".join(d[0] for d in part), "tied": tied,
                             "conf": round(min(d[1] for d in part), 3)})
    return nums


def _frets(digs: list[tuple[str, float, float, float]]) -> list[list[tuple[str, float, float, float]]]:
    """Digits (label, score, left, right) of one box as fret numbers (0-24): boxes of close numbers merge ('13 13' read
    '1313'), so a run that is no fret splits where both sides are frets, else at its widest gap between digits; two
    digits that are no fret keep the surer one."""
    def fret(part: list[tuple[str, float, float, float]]) -> bool:
        return len(part) <= 2 and int("".join(d[0] for d in part)) <= 24

    if len(digs) == 1 or (len(digs) == 2 and fret(digs)):
        return [digs]
    if len(digs) == 2:
        return [[max(digs, key=lambda d: d[1])]]
    k = max(range(1, len(digs)), key=lambda k: (fret(digs[:k]) + fret(digs[k:]), digs[k][2] - digs[k - 1][3]))
    return _frets(digs[:k]) + _frets(digs[k:])


def bar_numbers(gray: np.ndarray, st: list[float], xs: list[int], sp: float, ceiling: float | None = None
                ) -> dict[int, int]:
    """{bar line x: number}: the small digits at each bar line (right of it, or centred on it) anywhere above the staff
    (Guitar Pro prints them above the top staff of the system, e.g. the notation staff), up to ``ceiling`` (the tab
    staff above, on a page with several systems) or the edge of the staff's panel (a tab strip over a video). They are
    faint grey in videos."""
    top = st[0]
    y0, y1 = max(0, int(top - 16 * sp)), max(0, int(top - 0.3 * sp))
    if ceiling is not None:
        y0 = max(y0, int(ceiling + 0.6 * sp))
    if y1 <= y0 or len(xs) < 2:
        return {}
    bg = float(np.median(gray[int(st[0]):int(st[-1]) + 1, xs[0]:xs[-1]]))
    rows = np.median(gray[y0:y1, xs[0]:xs[-1]], axis=1)
    off = np.r_[0, (np.abs(rows - bg) > 40).astype(int), 0]
    edges = np.flatnonzero(np.diff(off))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= max(3, 0.5 * sp)]  # not a staff line
    if runs:
        y0 += int(runs[-1][1])
    if y1 <= y0:
        return {}
    # down to just inside the staff, its top line rows wiped: a number set close on the line keeps its full height, and
    # a fret number on string 1 (it reaches below the line) is no candidate
    y1 = min(gray.shape[0], int(top + 0.6 * sp))
    sub = gray[y0:y1].astype(float)
    mask = sub < bg - 25
    # long straight strokes are no digits: a cursor box around the playing bar, a play cursor, notation staff lines
    # (taken out before the top line is, which would cut a box's edge into short pieces)
    mask &= ~ndimage.binary_opening(mask, structure=np.ones((int(1.1 * sp) + 1, 1)))
    mask &= ~ndimage.binary_opening(mask, structure=np.ones((1, int(2 * sp) + 1)))
    mask[max(0, int(np.floor(top)) - 1 - y0):int(np.ceil(top)) + 2 - y0] = False
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    found = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (0.3 * sp <= h <= 0.75 * sp) or w > 2.2 * sp or sl[0].stop + y0 > top - 1:
            continue
        crop = np.where(lab[sl] == k, np.clip((bg - sub[sl]) / bg, 0, 1), 0.0)
        # faint antialiasing can join two small digits; a lowish threshold keeps a thin serif stroke (Times '5') whole
        for c0, c1, piece, ph in _pieces(crop, 0.2):
            if ph < 0.6 * h:
                continue
            ds = _digits(piece, ph, digits_only=True)  # bar numbers are digits (a 7 is not a rest)
            if all(sc >= 0.45 and str(d).isdigit() for d, sc in ds):
                sure = all(sc >= (0.7 if len(ds) == 1 else 0.6) for _d, sc in ds)
                step = (c1 - c0) / len(ds)
                for i, (d, _sc) in enumerate(ds):
                    found.append((sl[0].start, sl[1].start + c0 + int(i * step), sl[1].start + c0 + int((i + 1) * step),
                                  str(d), sure))
    unit = max(1, int(0.3 * sp))
    windows = []
    for x, nxt in zip(xs, list(xs[1:]) + [None]):
        right = x + 3.5 * sp if nxt is None else min(x + 3.5 * sp, nxt - 0.4 * sp)
        windows.append((x, [f for f in found if x - 1.2 * sp <= f[1] <= right]))
    # the bar numbers of a staff share one row: the row with sure digits at the most bar lines (then the nearest the
    # staff); in that row a less sure digit counts too (a thin serif '5' scores about 0.55)
    cover: Counter = Counter()
    for _x, cand in windows:
        for r in {f[0] // unit for f in cand if f[4]}:
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


def slashes(gray: np.ndarray, st: list[float], sp: float, nums: list[dict[str, Any]]) -> list[float]:
    """x of every rhythm slash: a stem that rises from below the staff into it (a note's stem starts under its number
    or under the staff) with no number at it and a short stroke rising right from its top. A slash strikes the chord
    shape before it again."""
    y0, y1 = max(0, int(st[0] - sp)), min(gray.shape[0], int(st[-1] + 2.5 * sp))
    sub = gray[y0:y1].astype(float)
    bg = float(np.median(sub))
    stem = (bg - sub) / max(1.0, bg) > 0.6
    bottom = int(round(st[-1])) - y0
    band = stem[bottom + int(0.4 * sp):bottom + int(1.5 * sp)]
    if not band.size:
        return []
    cols = np.flatnonzero(band.mean(0) > 0.9)
    out: list[float] = []
    for grp in np.split(cols, np.flatnonzero(np.diff(cols) > 2) + 1) if len(cols) else []:
        x = int(round(float(grp.mean())))
        if any(abs(n["cx"] - x) <= 0.6 * sp or n["x0"] - 2 <= x <= n["x1"] + 2 for n in nums):
            continue
        y = bottom + int(0.4 * sp)
        while y > 0 and stem[y - 1, max(0, x - 1):x + 2].any():
            y -= 1
        if y + y0 > st[-1] - 0.5 * sp:
            continue
        if _rising(stem[max(0, y - int(1.3 * sp)):y + 1, x + 2:x + int(1.2 * sp)], sp):
            out.append(float(x))
    return out


def _rising(win: np.ndarray, sp: float) -> bool:
    """A thin stroke rising to the right across most of the window's rows (a slash), not a digit. Only each row's
    first run counts (sixteenth slashes stand close: the next one can reach into the window)."""
    rows = []
    for i, r in enumerate(win):
        c = np.flatnonzero(r)
        if len(c):
            rows.append((i, c[:int(np.argmax(np.diff(np.r_[c, c[-1] + 2]) > 1)) + 1]))
    if len(rows) < 0.6 * sp or any(len(c) > max(4, 0.3 * sp) for _i, c in rows):
        return False
    ys = np.array([i for i, _c in rows], float)
    xs = np.array([c.mean() for _i, c in rows])
    return bool(np.ptp(xs) >= 0.4 * sp and np.corrcoef(ys, xs)[0, 1] < -0.8)


def bends(gray: np.ndarray, st: list[float], sp: float, nums: list[dict[str, Any]]) -> set[int]:
    """``id``s of the fret numbers with a bend arrow: a thin stroke that starts just right of the number (along its line
    or above it) and rises more than 1.5 line gaps above the staff (Guitar Pro, Rika: a curved arrow up to 'full' or
    '1/2'; a tie arc over string 1 rises about 0.65). The staff lines are wiped first and the cuts they leave in the
    stroke bridged; a release arrow joined to it changes nothing."""
    top, bot = st[0], st[-1]
    y0, y1 = max(0, int(top - 4 * sp)), min(gray.shape[0], int(bot + 0.5 * sp))
    sub = gray[y0:y1].astype(float)
    bg = float(np.median(gray[int(top):int(bot) + 1]))
    mask = sub < bg - 60
    for ly in st:
        r = int(round(ly)) - y0
        mask[max(0, r - 1):r + 2] = False
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 1)))
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    out = set()
    for n in nums:
        ly = st[n["line"]] - y0
        ya, yb = max(0, int(ly - 1.0 * sp)), min(lab.shape[0], int(ly + 0.6 * sp) + 1)
        xa, xb = int(n["x1"]), min(lab.shape[1], int(n["x1"] + 1.6 * sp))
        for k in set(np.unique(lab[ya:yb, xa:xb])) - {0}:
            sl = objs[k - 1]
            if sl[0].start + y0 < top - 1.5 * sp and (lab[sl] == k).sum() < 15 * sp * sp:
                out.add(id(n))
                break
    return out


def _wipe_lines(mask: np.ndarray, st: list[float], y0: int) -> np.ndarray:
    """The staff lines out of an ink mask, except where a stroke crosses them (ink both just above and just below)."""
    mask = mask.copy()
    keep = np.zeros_like(mask)
    for ly in st:
        r = int(round(ly)) - y0
        a, b = max(0, r - 1), min(mask.shape[0], r + 2)
        up = mask[max(0, a - 3):a].any(0) if a > 0 else np.zeros(mask.shape[1], bool)
        dn = mask[b:b + 3].any(0) if b < mask.shape[0] else np.zeros(mask.shape[1], bool)
        cross = ndimage.binary_dilation(up, np.ones(7)) & ndimage.binary_dilation(dn, np.ones(7))
        keep[a:b] = mask[a:b] & cross
        mask[a:b] = False
    return mask | keep


def glides(gray: np.ndarray, st: list[float], sp: float, nums: list[dict[str, Any]]) -> set[int]:
    """``id``s of the fret numbers with a slide line next to them: a thin straight stroke, slanting up or down, within
    1.2 line gaps left or right of the number at its own line, off the line's rows (a slide between two numbers, or into
    or out of one). Strokes with a number's box in them, and anything taller than a line gap, are left out."""
    top, bot = st[0], st[-1]
    y0, y1 = max(0, int(top - sp)), min(gray.shape[0], int(bot + sp))
    sub = gray[y0:y1].astype(float)
    bg = float(np.median(gray[int(top):int(bot) + 1]))
    mask = _wipe_lines(sub < bg - 60, st, y0)
    for n in nums:  # the numbers themselves out
        r = int(round(st[n["line"]])) - y0
        mask[max(0, int(r - 0.5 * sp)):int(r + 0.5 * sp) + 1, max(0, int(n["x0"]) + 1):int(n["x1"])] = False
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(lab)
    good = set()
    for k, sl in enumerate(objs, 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if not (0.3 * sp <= h <= 1.0 * sp and 0.25 * sp <= w <= 1.5 * sp):
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        if len(ys) > 0.5 * h * w or len(set(ys)) < 3 or np.std(xs) == 0:
            continue
        if abs(np.corrcoef(ys, xs)[0, 1]) >= 0.8:
            good.add(k)
    out = set()
    for n in nums:
        r = int(round(st[n["line"]])) - y0
        ya, yb = max(0, int(r - 0.5 * sp)), min(lab.shape[0], int(r + 0.5 * sp) + 1)
        for xa, xb in ((int(n["x0"] - 1.2 * sp), int(n["x0"]) + 2), (int(n["x1"]) - 2, int(n["x1"] + 1.2 * sp))):
            seen = set(np.unique(lab[ya:yb, max(0, xa):max(0, xb)])) & good
            if seen:
                out.add(id(n))
                break
    return out


def read_staff(gray: np.ndarray, st: list[float], ceiling: float | None = None) -> list[dict[str, Any]]:
    """The staff's bars: {x: (left, right bar line), number (or None), cols: [[(string, text), ...] per column],
    slash: [bool per column], bend / slide: [bool per column: a bend arrow / a slide line at one of its numbers]}.
    String 1 is the top line. Tied notes (in brackets) are left out, as the sheet evaluation does. A rhythm slash is a
    column of the chord shape before it (its dead strings left out, as a band score writes the repeated strums), marked
    in ``slash``. ``ceiling``: the bottom line of the tab staff above (bar numbers are
    not searched beyond it)."""
    xs, sp = barlines(gray, st)
    nums = read_numbers(gray, st, sp)
    numbers = bar_numbers(gray, st, xs, sp, ceiling)
    marks = slashes(gray, st, sp, nums)
    bent = bends(gray, st, sp, nums)
    slid = glides(gray, st, sp, nums)
    bars = []
    shape: list[tuple[int, str]] | None = None
    for a, b in zip(xs, xs[1:]):
        cols: list[list[dict[str, Any]]] = []
        cur = None
        for nm in sorted((n for n in nums if a < n["cx"] < b), key=lambda n: n["cx"]):
            if cur is not None and abs(nm["cx"] - cur) <= 0.6 * sp:
                cols[-1].append(nm)
            else:
                cols.append([nm])
                cur = nm["cx"]
        events = [(float(np.mean([n["cx"] for n in c])), sorted((n["line"] + 1, n["text"]) for n in c if not n["tied"]),
                   False, any(id(n) in bent for n in c), any(id(n) in slid for n in c)) for c in cols]
        events += [(x, [], True, False, False) for x in marks if a < x < b]
        out_cols, flags, bend, slide, phases = [], [], [], [], []
        for x, notes, is_slash, is_bent, is_slid in sorted(events, key=lambda e: e[0]):
            if is_slash:
                if not shape:
                    continue
                notes = [n for n in shape if n[1] != "x"]
            elif notes:
                shape = notes
            out_cols.append(notes)
            flags.append(is_slash)
            bend.append(is_bent)
            slide.append(is_slid)
            phases.append(round((x - a) / max(1, b - a), 4))
        bars.append({"x": (a, b), "number": numbers.get(a), "cols": out_cols, "slash": flags, "bend": bend,
                     "slide": slide, "phases": phases, "y": (round(float(st[0]), 1), round(float(st[-1]), 1))})
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
