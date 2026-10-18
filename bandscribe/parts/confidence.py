"""Part confidence (M10; E36 in docs/decisions.md): the probability that a guitar note's part is right.

The parts stage's posterior ``p`` is close to 1 for almost every note (E36's registration: mean 0.998, right 0.838 on
the whole-song Tier B remixes), so it cannot rank or be calibrated alone. ``part_features`` describes each assigned
note by the spatial evidence behind its part: how far and on which side its ILD sits against its part's notes, its
coherence and M/S level, the router label of its window, the posterior, the part count and the polyphony.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FEATURES = ("abs_ild", "ild_same_side", "ild_z", "coh", "ms_db", "lab_R0", "lab_R1", "lab_R2", "lab_R3", "lab_none",
            "post", "k", "poly")


def part_features(notes: Sequence[Mapping[str, Any]], sigs: Sequence[Mapping[str, Any]],
                  lines: Sequence[str | int | None], posts: Sequence[float], labels: Sequence[str | None],
                  k: int) -> tuple[np.ndarray, np.ndarray]:
    """(X for the notes with a part, their indices into ``notes``)."""
    col = {f: i for i, f in enumerate(FEATURES)}
    ild = np.array([float((s or {}).get("ild_db") or 0.0) for s in sigs])
    on = np.array([float(n["onset_s"]) for n in notes])
    order = np.argsort(on, kind="stable")
    so = on[order]
    stats: dict[Any, tuple[float, float]] = {}
    for ln in {x for x in lines if x is not None}:
        v = ild[[i for i, x in enumerate(lines) if x == ln]]
        stats[ln] = (float(v.mean()), float(v.std()))
    idx = [i for i, x in enumerate(lines) if x is not None]
    X = np.zeros((len(idx), len(FEATURES)))
    for r, i in enumerate(idx):
        mu, sd = stats[lines[i]]
        s = sigs[i] or {}
        X[r, col["abs_ild"]] = abs(ild[i])
        X[r, col["ild_same_side"]] = float(np.sign(ild[i]) == np.sign(mu) or abs(mu) < 1.0)
        X[r, col["ild_z"]] = abs(ild[i] - mu) / (sd + 1.0)
        X[r, col["coh"]] = float(s.get("coh") or 0.0)
        X[r, col["ms_db"]] = float(s.get("ms_db") or 0.0)
        lab = labels[i] if labels[i] in ("R0", "R1", "R2", "R3") else None
        X[r, col[f"lab_{lab}" if lab else "lab_none"]] = 1.0
        X[r, col["post"]] = float(posts[i])
        X[r, col["k"]] = float(k)
        X[r, col["poly"]] = float(np.searchsorted(so, on[i] + 0.03, side="right") - np.searchsorted(so, on[i] - 0.03) - 1)
    return X, np.array(idx, dtype=int)
