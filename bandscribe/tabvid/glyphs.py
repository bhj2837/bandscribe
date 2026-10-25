"""Glyph recognition for tab images (docs/TAB_VIDEO.md): fret digits, the dead-note 'x', brackets of tied notes and
rests, by normalised correlation with templates drawn from the PC's own fonts (no OCR engine, no download).

A crop's ink is cut to its bounding box, scaled into ``TW`` x ``TH`` keeping the aspect ratio, centred, made zero-mean
and unit-norm; the label is the template with the highest dot product (a score near 1 is a near-perfect match)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

TW, TH = 10, 14
FONTS = ("arial.ttf", "arialbd.ttf", "calibri.ttf", "segoeui.ttf", "tahoma.ttf", "verdana.ttf", "times.ttf",
         "cour.ttf", "consola.ttf", "malgun.ttf", "trebuc.ttf", "georgia.ttf")
SIZES = (12, 16, 24)
REST_CODEPOINTS = (0xE4E3, 0xE4E4, 0xE4E5, 0xE4E6, 0xE4E7, 0xE4E8)  # SMuFL rests (Bravura), whole to 32nd


def font_dir() -> Path:
    return Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def bravura() -> Path:
    from bandscribe import paths

    return Path(paths.NODE_DIR) / "node_modules" / "@coderline" / "alphatab" / "dist" / "font" / "Bravura.otf"


def norm(ink: np.ndarray) -> np.ndarray | None:
    """The comparison vector of a crop (ink in 0..1), or None when it holds no ink."""
    from PIL import Image

    a = np.asarray(ink, dtype=float)
    ys, xs = np.nonzero(a > 0.25)
    if not len(ys):
        return None
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    scale = min(TH / h, TW / w) if w / h > TW / TH else TH / h
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                                            Image.BILINEAR)
    b = np.asarray(im, dtype=float) / 255.0
    out = np.zeros((TH, TW))
    bh, bw = min(TH, b.shape[0]), min(TW, b.shape[1])
    y0, x0 = (TH - bh) // 2, (TW - bw) // 2
    out[y0:y0 + bh, x0:x0 + bw] = b[:bh, :bw]
    out -= out.mean()
    n = np.linalg.norm(out)
    return out / n if n else None


def _draw(font, ch: str, size: int) -> np.ndarray | None:
    from PIL import Image, ImageDraw

    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size, size // 2), ch, fill=255, font=font)
    return norm(np.asarray(img) / 255.0)


@lru_cache(maxsize=1)
def templates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, vectors, is_digit) of every template."""
    from PIL import ImageFont

    rows: list[tuple[str, np.ndarray]] = []
    chars = [(d, d) for d in "0123456789"] + [("x", "x"), ("X", "x"), ("(", "("), (")", ")")]
    for f in FONTS:
        p = font_dir() / f
        if not p.is_file():
            continue
        for size in SIZES:
            font = ImageFont.truetype(str(p), size)
            for ch, lab in chars:
                v = _draw(font, ch, size)
                if v is not None:
                    rows.append((lab, v))
    if bravura().is_file():
        for size in (16, 24, 32):
            font = ImageFont.truetype(str(bravura()), size)
            for cp in REST_CODEPOINTS:
                v = _draw(font, chr(cp), size)
                if v is not None:
                    rows.append(("rest", v))
    if not rows:
        raise RuntimeError(f"글꼴을 찾지 못했습니다: {font_dir()}")
    labels = np.array([r[0] for r in rows])
    return labels, np.stack([r[1].ravel() for r in rows]), np.array([lab.isdigit() for lab in labels])


def classify(ink: np.ndarray, *, digits_only: bool = False) -> tuple[str | None, float]:
    """(label, score) of a crop: a digit, 'x', '(', ')' or 'rest'."""
    v = norm(ink)
    if v is None:
        return None, 0.0
    labels, mats, is_digit = templates()
    s = mats @ v.ravel()
    if digits_only:
        s = np.where(is_digit, s, -1.0)
    k = int(np.argmax(s))
    return str(labels[k]), float(s[k])


def bracket(ink: np.ndarray, h: int, w: int) -> str | None:
    """'(' or ')' when a narrow, tall stroke bulges sideways in its middle (a tied note's bracket); a '1' is
    straight."""
    if w > 0.45 * h or h < 4:
        return None
    m = ink > 0.3
    rows = np.flatnonzero(m.any(1))
    if len(rows) < 6:
        return None
    q = max(1, len(rows) // 4)
    xs = np.arange(m.shape[1])

    def cx(rs: np.ndarray) -> float:
        sel = m[rs]
        tot = sel.sum()
        return float((sel * xs).sum() / tot) if tot else 0.0

    top, mid, bot = cx(rows[:q]), cx(rows[len(rows) // 2 - q // 2:len(rows) // 2 + q // 2 + 1]), cx(rows[-q:])
    lim = max(0.8, 0.18 * w)
    if top - mid > lim and bot - mid > lim:
        return "("
    if mid - top > lim and mid - bot > lim:
        return ")"
    return None
