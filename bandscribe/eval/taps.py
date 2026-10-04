"""Tapped downbeats from an Audacity label track (``bandscribe gt taps``; DESIGN 8.3 step 3).

Audacity exports labels as ``start<TAB>end<TAB>label`` per line (UTF-8; files saved by other tools may be
cp949). A label ``b17`` or ``17`` is the expanded bar number of that downbeat; any other / empty label is assigned
to the nearest GT downbeat of the current tempo map (nominal tempo if the song is not aligned yet), in order.
Frequency-band lines (``\\`` continuation lines of spectral labels) are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_BAR_LABEL = re.compile(r"^\s*[bB]?\s*(-?\d+)\s*$")


def read_label_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def parse_audacity_labels(text: str) -> list[dict[str, Any]]:
    """-> [{"t_s", "bar" | None, "label"}] sorted by time."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("\\"):
            continue
        parts = line.split("\t")
        try:
            t = float(parts[0].strip().replace(",", "."))
        except ValueError:
            continue
        label = parts[2].strip() if len(parts) > 2 else ""
        m = _BAR_LABEL.match(label)
        out.append({"t_s": round(t, 6), "bar": int(m.group(1)) if m else None, "label": label})
    out.sort(key=lambda r: r["t_s"])
    return out


def assign_bars(taps: Sequence[dict[str, Any]], tempo_map: Any, bars: Sequence[int]) -> list[dict[str, Any]]:
    """Fill ``bar`` of unlabeled taps with the nearest GT downbeat, never reusing or going back in bar order."""
    gb = np.array(sorted(set(int(b) for b in bars)))
    t_down = np.asarray(tempo_map.bar_tick_to_time(gb, np.zeros(len(gb))), dtype=float)
    out = []
    last = -10 ** 9
    for tp in taps:
        tp = dict(tp)
        if tp.get("bar") is None:
            order = np.argsort(np.abs(t_down - tp["t_s"]), kind="stable")
            pick = next((int(gb[i]) for i in order if int(gb[i]) > last), None)
            tp["bar"] = pick
        if tp["bar"] is not None:
            last = int(tp["bar"])
        out.append(tp)
    return out


def taps_json(source_file: str, taps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"format": "bandscribe.taps/1", "source_file": source_file,
            "taps": [{"t_s": t["t_s"], "bar": t["bar"], "label": t["label"]} for t in taps]}


def load_taps(path: Path) -> list[tuple[float, int]]:
    from bandscribe import atomic

    d = atomic.read_json(path)
    return [(float(t["t_s"]), int(t["bar"])) for t in d.get("taps", []) if t.get("bar") is not None]
