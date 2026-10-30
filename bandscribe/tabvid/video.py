"""A tab video -> the tab's bars in play order (docs/TAB_VIDEO.md).

Tab videos show a strip or a page of the score and turn it as the music plays: frames that show the same bar lines are
one *page*. A page's bars are read in every one of its frames and voted; bar numbers (small digits above each bar) are
voted per page, and a page's first number is the start near the previous page's end that fits its readings best (a
read '24' fits 124: a leading digit is often lost). Bars on two pages vote together. A frame with several tab staves
(Guitar 1 over Guitar 2) gives one *track* per place in the system, each read on its own; the systems of a score
page follow one another.

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
    gaps in the lines (fret numbers), a light background, and line gaps at least 0.8 of the widest in the frame (every
    format seen draws its tab lines wider apart than a notation staff: JINZO 20 / 13.5 px, Rika 17 / 12, a GP page
    13.2 / 8.8). A staff without fret numbers (a part resting) counts when another tab of the frame has its bar lines:
    leaving it out would move the parts below it up a place in their system."""
    cand = []
    for st in S.find_staves(gray, n):
        xs, sp = S.barlines(gray, st)
        if len(xs) < 2 or float(np.median(np.diff(xs))) < 4 * sp:
            continue
        if float(np.median(gray[int(st[0]):int(st[-1]) + 1, xs[0]:xs[-1]])) < 170:
            continue
        y0 = max(0, int(st[0] - sp))
        sub = gray[y0:int(st[-1] + sp)].astype(float)
        cand.append((st, xs, sp, sum(len(S.line_gaps(sub, ly, y0, sp)) for ly in st) >= 2))
    numbered = [c for c in cand if c[3]]
    out = [(st, xs, sp) for st, xs, sp, has in cand
           if has or any(len(xs) == len(o[1]) and all(abs(a - b) <= 3 for a, b in zip(xs, o[1])) for o in numbered)]
    widest = max((sp for _st, _xs, sp in out), default=0.0)
    return [t for t in out if t[2] >= 0.8 * widest]


def _notes(bars: list[dict[str, Any]]) -> int:
    return sum(len(c) for b in bars for c in b["cols"])


def _same_lines(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    return len(a) == len(b) and all(abs(p["x"][0] - q["x"][0]) <= 3 and abs(p["x"][1] - q["x"][1]) <= 3
                                    for p, q in zip(a, b))


def _goes_on(above: list[dict[str, Any]], bars: list[dict[str, Any]]) -> bool:
    """Most of the staff's own bar numbers come after those of the staff above: the next system of a page, even with
    the same bar lines (an even layout); another part of the same system shows no numbers or the same ones."""
    up = [b["number"] for b in above if b["number"] is not None]
    mine = [b["number"] for b in bars if b["number"] is not None]
    return bool(up and mine) and sum(1 for m in mine if m > max(up)) > len(mine) / 2


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
                if _same_lines(up["bars"], bars) and not _goes_on(up["bars"], bars):
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


def _alike(a: tuple | None, b: tuple) -> bool:
    """Two readings of one bar are the same bar when their (string, fret) bags share 60 %: a frame's reading wavers (a
    play cursor crossing a number, a thin font), a new bar differs in most notes."""
    if a is None:
        return not b
    x, y = Counter(n for c in a for n in c), Counter(n for c in b for n in c)
    return sum((x & y).values()) >= 0.6 * max(sum(x.values()), sum(y.values()), 1) or (not x and not y)


def pages_of(readings: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Consecutive frames' readings (bars of one staff) with the same bar lines (within 3 px) grouped: {bars: [(x0,
    x1)], frames, first / last (frame index), readings: [Counter of column tuples per bar], numbers: [Counter of read
    numbers per bar], phases}."""
    out: list[dict[str, Any]] = []
    for fi, bars in enumerate(readings):
        if not bars:
            continue
        sig = [b["x"] for b in bars]
        same = bool(out) and len(out[-1]["bars"]) == len(sig) and all(
            abs(a[0] - b[0]) <= 3 and abs(a[1] - b[1]) <= 3 for a, b in zip(out[-1]["bars"], sig))
        if same:  # the same layout can show the next bars: the content must agree too (numbers misread too often)
            top = [c.most_common(1)[0][0] if c else None for c in out[-1]["readings"]]
            now = [tuple(tuple(c) for c in b["cols"]) for b in bars]
            same = sum(1 for t, w in zip(top, now) if _alike(t, w)) >= 0.5 * len(now)
        if same:  # a riff repeated on the next page looks alike: bar numbers all moved by one step tell it apart
            was = [c.most_common(1)[0][0] if c else None for c in out[-1]["numbers"]]
            moves = Counter(b["number"] - w for b, w in zip(bars, was) if b["number"] is not None and w is not None)
            step, k = max(moves.items(), key=lambda kv: kv[1], default=(0, 0))
            same = not (step != 0 and k >= 2 and moves[0] == 0)
        if not same:
            out.append({"bars": sig, "frames": 0, "first": fi, "readings": [Counter() for _ in sig],
                        "numbers": [Counter() for _ in sig], "bends": [Counter() for _ in sig],
                        "phases": [b["phases"] for b in bars], "ys": [b.get("y") for b in bars]})
        pg = out[-1]
        pg["frames"] += 1
        pg["last"] = fi
        for i, b in enumerate(bars):
            pg["readings"][i][tuple(tuple(c) for c in b["cols"])] += 1
            pg["bends"][i][tuple(zip(b.get("bend") or [False] * len(b["cols"]),
                                     b.get("slide") or [False] * len(b["cols"])))] += 1
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


def _ys(pg: dict[str, Any], i: int) -> tuple[float, float]:
    ys = pg.get("ys") or []
    return tuple(ys[i]) if i < len(ys) and ys[i] else (0.0, 0.0)  # type: ignore[return-value]


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
        read = sum(k * fits(rd, first + i) for i, c in enumerate(pg["numbers"]) for rd, k in c.items()) > 0
        for i, rd in enumerate(pg["readings"]):
            seq.append({"number": first + i, "readings": rd, "phases": pg["phases"][i], "by_number": read,
                        "bends": pg["bends"][i] if "bends" in pg else Counter(),
                        "shown": (pg.get("first", 0), pg.get("last", 0), i, len(pg["readings"]), pg["frames"]),
                        "view": (pg.get("first", 0), *pg["bars"][i], *_ys(pg, i)) if "bars" in pg else None})
        last = max(last, first + len(pg["readings"]) - 1)
    return seq


def consensus(readings: Counter) -> tuple[tuple, ...]:
    """A bar's reading voted note by note: among the frames that read the most common number of columns, each column
    keeps the (string, fret) notes more than half of them read (one frame drops a note, another misreads one)."""
    by_len: Counter = Counter()
    for r, k in readings.items():
        by_len[len(r)] += k
    n = by_len.most_common(1)[0][0]
    same = {r: k for r, k in readings.items() if len(r) == n}
    tot = sum(same.values())
    out = []
    for i in range(n):
        votes: Counter = Counter()
        for r, k in same.items():
            for note in r[i]:
                votes[note] += k
        out.append(tuple(sorted(note for note, v in votes.items() if v > tot / 2)))
    return tuple(out)


def assemble(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One reading per bar number, voted note by note over all its frames (``consensus``); ``agree`` is the share of
    frames whose whole reading is the most common one. {number, cols, agree, frames, phases}."""
    acc: dict[int, Counter] = {}
    views: dict[int, tuple] = {}
    bends: dict[int, Counter] = {}
    phases: dict[int, list[float]] = {}
    for b in seq:
        acc.setdefault(b["number"], Counter()).update(b["readings"])
        bends.setdefault(b["number"], Counter()).update(b.get("bends") or {})
        phases.setdefault(b["number"], b["phases"])
        if b.get("view") and b["shown"][4] > views.get(b["number"], (0, None))[0]:
            views[b["number"]] = (b["shown"][4], b["view"])
    out = []
    for num in sorted(acc):
        (_top, k), = acc[num].most_common(1)
        cols = consensus(acc[num])
        tot = sum(acc[num].values())
        ph = phases[num] if len(phases[num]) == len(cols) else [round((i + 0.5) / max(1, len(cols)), 4)
                                                                 for i in range(len(cols))]
        fl = [f for f, _k in bends.get(num, Counter()).most_common() if len(f) == len(cols)]
        marks = list(fl[0]) if fl else [(False, False)] * len(cols)
        out.append({"number": num, "cols": [list(c) for c in cols], "agree": round(k / tot, 3), "frames": tot,
                    "phases": ph, "bend": [m[0] for m in marks], "slide": [m[1] for m in marks],
                    "view": list(views[num][1]) if num in views else None})
    return out


def iter_frames(paths: Iterable[Path]) -> Iterator[np.ndarray]:
    from PIL import Image

    for p in paths:
        yield np.asarray(Image.open(p).convert("L"))


def keep_tab_spacing(per_frame: list[list[dict[str, Any]]], share: float = 0.2) -> list[list[dict[str, Any]]]:
    """The staves of the video's tab line gap. Line gaps fall into groups (within 20 % of their neighbours); the tab is
    the group with the widest gap among those holding at least ``share`` of the staves - every format seen draws its
    tab lines wider apart than notation. A frame whose tab staff was missed would otherwise keep its notation staff,
    the widest within that frame (BassScore: notation 14-16 px, tab 21-24 px; noteheads read as digits outnumber its
    fret numbers)."""
    sps = sorted(t["sp"] for staves in per_frame for t in staves if t["bars"])
    if not sps:
        return per_frame
    groups: list[list[float]] = [[sps[0]]]
    for x in sps[1:]:
        if x > 1.2 * groups[-1][-1]:
            groups.append([x])
        else:
            groups[-1].append(x)
    big = [g for g in groups if len(g) >= share * len(sps)] or groups
    lo, hi = 0.95 * big[-1][0], 1.05 * big[-1][-1]
    return [[t for t in staves if lo <= t["sp"] <= hi] for staves in per_frame]


def systems(staves: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """A frame's staves (top to bottom) grouped into systems: a staff with the same bar lines as the system above it is
    another part of that system (Guitar 1 over Guitar 2); a staff with other bar lines starts the next system (a score
    page shows the next bars below) unless most of its bar numbers are ones the system above shows too (another part
    laid out on its own, Rika's arpeggio guitar under the bridge). Bar numbers that go on after the system above make
    a staff the next system even with the same bar lines (``_goes_on``: a page laid out evenly)."""
    out: list[list[dict[str, Any]]] = []
    for t in staves:
        if out and _same_lines(out[-1][0]["bars"], t["bars"]) and not _goes_on(out[-1][0]["bars"], t["bars"]):
            out[-1].append(t)
            continue
        above = [b["number"] for s in out[-1] for b in s["bars"] if b["number"] is not None] if out else []
        mine = [b["number"] for b in t["bars"] if b["number"] is not None]
        if above and mine and sum(1 for m in mine if m <= max(above)) > len(mine) / 2:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def tracks(per_frame: list[list[dict[str, Any]]]) -> list[list[list[dict[str, Any]]]]:
    """Per part (the staff's place in its system: first, second ...), its bars in every frame, the frame's systems one
    after another ([] where the frame has no such part)."""
    k = max((len(sy) for staves in per_frame for sy in systems(staves)), default=0)
    out: list[list[list[dict[str, Any]]]] = [[[] for _ in per_frame] for _ in range(k)]
    for i, staves in enumerate(per_frame):
        for sy in systems(staves):
            for j, t in enumerate(sy):
                out[j][i] = out[j][i] + t["bars"]
    return out


def bar_times(seq: list[dict[str, Any]], fps: float) -> dict[int, float]:
    """{bar number: the video second it starts playing}. A page appears when its first bar is about to play: each page
    gives one anchor (its first bar, the second it appears; a bar that opens several pages keeps the page shown
    longest), the anchors are made to rise with the bar numbers (pool adjacent violators) and the bars between two
    anchors share the time between them evenly. How long a page stays says little (pages turn early or late), and a
    constant lead of the pages over the sound is left to ``sheet.score_time``."""
    best: dict[int, tuple[int, float]] = {}
    nums = set()
    for e in seq:
        nums.add(e["number"])
        if "shown" not in e:
            continue
        a, _b, i, _k, n = e["shown"]
        if i == 0 and (e["number"] not in best or n > best[e["number"]][0]):
            best[e["number"]] = (n, a / fps)
    if not best:
        return {}
    pts: list[list] = []
    for x in sorted(best):  # pool adjacent violators: a non-decreasing fit of the anchors
        pts.append([best[x][1], 1.0, [x]])
        while len(pts) > 1 and pts[-2][0] > pts[-1][0]:
            t2, w2, n2 = pts.pop()
            t1, w1, n1 = pts.pop()
            pts.append([(t1 * w1 + t2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    xs = np.array([float(np.mean(ns)) for _t, _w, ns in pts])
    ts = np.array([t for t, _w, _ns in pts])
    slope = float(np.median(np.diff(ts) / np.diff(xs))) if len(xs) > 1 else 2.0
    out = {}
    for x in sorted(nums):
        if x < xs[0]:
            t = ts[0] - slope * (xs[0] - x)
        elif x > xs[-1]:
            t = ts[-1] + slope * (x - xs[-1])
        else:
            t = float(np.interp(x, xs, ts))
        out[x] = round(float(t), 2)
    return out


USABLE = {"suspicious": 0.35, "by_number": 0.6, "even": 0.6}
UNUSABLE = {"suspicious": 0.6, "by_number": 0.2, "even": 0.3, "bars": 8}


def quality(bars: list[dict[str, Any]], by_number: float) -> dict[str, Any]:
    """How far a track's reading can be trusted: {verdict: usable | doubtful | unusable, reasons, suspicious (bars whose
    frames disagree), by_number (bars whose page was placed by its read bar numbers, not by following on), even (bars
    whose time on screen is within 35 % of the median bar: torn pages and wrong numbers make it uneven)}. Thresholds
    from the measured videos (docs/TAB_VIDEO.md)."""
    if len(bars) < UNUSABLE["bars"]:
        return {"verdict": "unusable", "reasons": [f"마디가 {len(bars)}개뿐"], "suspicious": None, "by_number": None,
                "even": None}
    sus = sum(1 for b in bars if b["agree"] < 0.8) / len(bars)
    ts = [(b["number"], b.get("t")) for b in bars if b.get("t") is not None]
    d = [t2 - t1 for (n1, t1), (n2, t2) in zip(ts, ts[1:]) if n2 == n1 + 1]
    med = float(np.median(d)) if d else 0.0
    even = sum(1 for x in d if med > 0 and abs(x - med) <= 0.35 * med) / len(d) if d else 0.0
    reasons, bad, weak = [], False, False
    for key, val, name in (("suspicious", sus, "프레임끼리 판독이 갈리는 마디"), ("by_number", by_number, "마디 번호로 자리를 잡은 마디"),
                           ("even", even, "마디 시간 간격이 고른 마디")):
        worse = val > UNUSABLE[key] if key == "suspicious" else val < UNUSABLE[key]
        short = val > USABLE[key] if key == "suspicious" else val < USABLE[key]
        if worse or short:
            reasons.append(f"{name} {100 * val:.0f} %")
        bad, weak = bad or worse, weak or short
    verdict = "unusable" if bad else "doubtful" if weak else "usable"
    return {"verdict": verdict, "reasons": reasons, "suspicious": round(sus, 3), "by_number": round(by_number, 3),
            "even": round(even, 3)}


def read_track(readings: list[list[dict[str, Any]]], fps: float = 2.0) -> dict[str, Any]:
    pgs = pages_of(readings)
    seq = number_pages(pgs)
    bars = assemble(seq)
    times = bar_times(seq, fps)
    for b in bars:
        b["t"] = times.get(b["number"])
    shown = [i for i, r in enumerate(readings) if r]
    numbered = {e["number"] for e in seq if e.get("by_number")}
    return {"pages": len(pgs), "stable_pages": sum(1 for p in pgs if p["frames"] >= 2), "bars": bars,
            "notes": sum(len(c) for b in bars for c in b["cols"]),
            "agree": round(float(np.mean([b["agree"] for b in bars])), 4) if bars else None,
            "span_s": [round(shown[0] / fps, 1), round((shown[-1] + 1) / fps, 1)] if shown else None,
            "quality": quality(bars, len(numbered & {b["number"] for b in bars}) / max(1, len(bars)))}


def read_video(frames: list[Path], n: int = 6, fps: float = 2.0) -> dict[str, Any]:
    """Every tab track of the video (Guitar 1, Guitar 2 ... by the staff's place in its system) in that order;
    ``bars`` / ``agree`` / ``pages`` repeat the track with the most fret numbers. ``fps``: the frames' rate (bar
    times, ``bar_times``)."""
    per_frame = keep_tab_spacing([frame_staves(g, n) for g in iter_frames(frames)])
    tr = [dict(read_track(r, fps), part=j + 1) for j, r in enumerate(tracks(per_frame))]
    main = max(tr, key=lambda t: t["notes"], default={"pages": 0, "stable_pages": 0, "bars": [], "agree": None})
    return {"format": FORMAT, "strings": n, "frames": len(frames), "fps": fps, "tracks": tr,
            "frames_dir": str(Path(frames[0]).parent) if frames else None, "pages": main["pages"],
            "stable_pages": main["stable_pages"], "bars": main["bars"], "agree": main["agree"]}
