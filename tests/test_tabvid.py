"""Tab videos -> bars (docs/TAB_VIDEO.md) on synthetic tab images drawn like Guitar Pro: grey staff lines, fret
numbers on white boxes that cut the lines, stems that cross them, small grey bar numbers above."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from bandscribe.tabvid import glyphs as G
from bandscribe.tabvid import sheet as SH
from bandscribe.tabvid import staff as S
from bandscribe.tabvid import video as V

TOP, SP, X0, X1 = 200, 20, 40, 1160
FONT = G.font_dir() / "arial.ttf"
pytestmark = pytest.mark.skipif(not FONT.is_file(), reason="needs the Windows fonts")


def draw(bars, *, first_number=1, invert=False, width=1200):
    """bars: [[(string, text, column), ...] per bar]; four bars over the staff; a stem under every column."""
    im = Image.new("L", (width, 420), 255)
    d = ImageDraw.Draw(im)
    big = ImageFont.truetype(str(FONT), 18)
    small = ImageFont.truetype(str(FONT), 14)
    for i in range(6):
        d.line([(X0, TOP + i * SP), (X1, TOP + i * SP)], fill=170, width=1)
    edges = np.linspace(X0, X1, len(bars) + 1).astype(int)
    for k, x in enumerate(edges):
        d.line([(x, TOP), (x, TOP + 5 * SP)], fill=0, width=2)
        if k < len(bars):
            d.text((x + 3, TOP - 40), str(first_number + k), fill=140, font=small)
    for k, notes in enumerate(bars):
        for s, text, col in notes:
            cx = edges[k] + 40 + col * 60
            cy = TOP + (s - 1) * SP
            w = 7 * len(text) + 6
            d.rectangle([cx - w // 2, cy - 9, cx + w // 2, cy + 9], fill=255)
            d.text((cx, cy), text, fill=0, font=big, anchor="mm")
        for col in {c for _, _, c in notes}:
            cx = edges[k] + 40 + col * 60
            d.line([(cx, TOP + 5 * SP + 4), (cx, TOP + 5 * SP + 40)], fill=0, width=1)
    a = np.asarray(im)
    return 255 - a if invert else a


BARS = [[(2, "5", 0), (3, "3", 0), (4, "5", 0), (5, "4", 0), (2, "5", 1), (3, "3", 1)],
        [(1, "0", 0), (2, "12", 0), (6, "X", 1), (5, "7", 1)],
        [(3, "9", 0), (4, "10", 0), (3, "9", 2)],
        [(6, "3", 0), (5, "2", 1), (4, "0", 2)]]


def test_glyphs_read_digits_and_brackets():
    from PIL import ImageFont

    font = ImageFont.truetype(str(FONT), 18)
    for ch in "0123456789":
        img = Image.new("L", (40, 40), 0)
        ImageDraw.Draw(img).text((10, 6), ch, fill=255, font=font)
        assert G.classify(np.asarray(img) / 255.0)[0] == ch
    paren = Image.new("L", (30, 40), 0)
    ImageDraw.Draw(paren).text((10, 6), "(", fill=255, font=font)
    a = np.asarray(paren) / 255.0
    rows, cols = np.nonzero(a > 0.3)
    crop = a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    assert G.bracket(crop, *crop.shape) == "("
    one = Image.new("L", (30, 40), 0)
    ImageDraw.Draw(one).text((10, 6), "1", fill=255, font=font)
    b = np.asarray(one) / 255.0
    rows, cols = np.nonzero(b > 0.3)
    crop = b[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    assert G.bracket(crop, *crop.shape) is None


def test_a_staff_is_read_box_first_with_its_bar_numbers():
    g = draw(BARS, first_number=37)
    (st,) = S.find_staves(g)
    assert [round(y) for y in st] == [TOP + i * SP for i in range(6)]
    bars = S.read_staff(g, st)
    assert [b["number"] for b in bars] == [37, 38, 39, 40]
    assert bars[0]["cols"] == [[(2, "5"), (3, "3"), (4, "5"), (5, "4")], [(2, "5"), (3, "3")]]
    assert bars[1]["cols"] == [[(1, "0"), (2, "12")], [(5, "7"), (6, "x")]]
    assert bars[2]["cols"] == [[(3, "9"), (4, "10")], [(3, "9")]]
    assert bars[3]["cols"] == [[(6, "3")], [(5, "2")], [(4, "0")]]


def test_the_tab_clef_rests_and_tie_arcs_are_not_numbers():
    """alphaTab and Guitar Pro draw these over the lines too: the 'TAB' clef's A cuts string 3, a whole rest is a black
    block on it, and a tie arc starts at the number's box and runs on out of it."""
    im = Image.fromarray(draw(BARS))
    d = ImageDraw.Draw(im)
    clef = ImageFont.truetype(str(FONT), 24)
    for ch, y in (("T", TOP + 14), ("A", TOP + 2 * SP + 4), ("B", TOP + 4 * SP - 6)):  # string 3 under A's apex
        d.text((X0 + 14, y), ch, fill=0, font=clef, anchor="mm")
    edges = np.linspace(X0, X1, len(BARS) + 1).astype(int)
    cx = edges[2] + 40 + 60  # bar 3, the free column: a whole rest on string 3
    d.rectangle([cx - 7, TOP + 2 * SP - 4, cx + 7, TOP + 2 * SP + 4], fill=0)
    a = edges[0] + 40  # bar 1: the 5 on string 2 tied on to the next column, the arc joined to its shoulder
    d.arc([a + 3, TOP + SP - 14, a + 56, TOP + SP + 6], 190, 350, fill=0, width=2)
    b = edges[2] + 40  # bar 3: the 9s on string 3 tied under
    d.arc([b + 6, TOP + 2 * SP + 2, b + 116, TOP + 2 * SP + 16], 10, 170, fill=0, width=2)
    bars = S.read_staff(np.asarray(im), S.find_staves(np.asarray(im))[0])
    assert bars[0]["cols"] == [[(2, "5"), (3, "3"), (4, "5"), (5, "4")], [(2, "5"), (3, "3")]]
    assert bars[2]["cols"] == [[(3, "9"), (4, "10")], [(3, "9")]]


def test_both_polarities_are_read():
    assert [b["number"] for b in V.read_frame(draw(BARS, first_number=5, invert=True))] == [5, 6, 7, 8]


def test_pages_are_numbered_in_order_and_overlaps_vote_together():
    p1 = [draw(BARS, first_number=8)] * 3
    p2 = [draw(BARS[2:] + BARS[:2], first_number=10)] * 3  # overlaps p1 by two bars (10, 11)
    doc_pages = V.pages(p1 + p2)
    assert len(doc_pages) == 2
    bars = V.assemble(V.number_pages(doc_pages))
    assert [b["number"] for b in bars] == [8, 9, 10, 11, 12, 13]
    assert all(b["agree"] == 1.0 for b in bars) and bars[2]["frames"] == 6


def test_fits_and_number_fitting_recover_lost_leading_digits():
    assert V.fits(124, 124) == 1.0 and V.fits(24, 124) == 0.5 and V.fits(4, 124) == 0.5 and V.fits(3, 124) == 0.0
    from collections import Counter

    pg = {"frames": 5, "readings": [Counter({(): 5})] * 3, "numbers": [Counter({21: 5}), Counter(), Counter({23: 5})],
          "phases": [[]] * 3}
    prev = {"frames": 5, "readings": [Counter({(): 5})] * 4, "numbers": [Counter({117 + i: 5}) for i in range(4)],
            "phases": [[]] * 4}
    seq = V.number_pages([prev, pg])
    assert [b["number"] for b in seq] == [117, 118, 119, 120, 121, 122, 123]


def test_a_stray_exact_reading_does_not_make_a_page_jump_ahead():
    """'2 3 4' fit 102.. and 112.. alike (a lost leading digit); one misread '115' must not move the page by ten."""
    from collections import Counter

    prev = {"frames": 5, "readings": [Counter({(): 5})] * 4, "numbers": [Counter({98 + i: 5}) for i in range(4)],
            "phases": [[]] * 4}
    pg = {"frames": 5, "readings": [Counter({(): 5})] * 4,
          "numbers": [Counter({2: 5}), Counter({3: 5}), Counter({4: 5}), Counter({115: 1})], "phases": [[]] * 4}
    seq = V.number_pages([prev, pg])
    assert [b["number"] for b in seq][4:] == [102, 103, 104, 105]


def test_sheet_part_and_comparison():
    bars = [{"number": 2, "cols": [[(2, "5"), (5, "x")]], "agree": 1.0, "phases": [0.2]},
            {"number": 4, "cols": [[(1, "0")]], "agree": 0.9, "phases": [0.5]}]
    part = SH.to_sheet_part(bars)
    assert [b["bar"] for b in part["bars"]] == [2, 3, 4] and part["bars"][1]["rest_only"]
    note = part["bars"][0]["columns"][0]["notes"]
    assert note[1] == {"string": 5, "fret": None, "dead": True, "tied": False}
    shifted = {"bars": [dict(b, bar=b["bar"] + 1) for b in part["bars"]]}
    res = SH.compare(part, shifted)
    assert res["offset"] == 1 and res["F1"] == 1.0 and res["bars_same"] == 3
