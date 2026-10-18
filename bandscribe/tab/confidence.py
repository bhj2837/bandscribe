"""Note confidence (M10; E35 in docs/decisions.md): the probability that a merged guitar note is one the sheet has.

``note_features`` turns a job's quantised guitar notes (with their part and fretting) and its Basic Pitch guitar
transcription into a fixed feature matrix; ``NoteModel`` is a standardised logistic regression (coefficients in a
small JSON, ``model.json`` next to this module once trained) that maps it to P(correct).
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = ("log_dur", "poly", "rel_pitch", "octave_partner", "bp_same", "bp_octave", "cls_clean", "cls_dist",
            "cls_acoustic", "part_p", "unassigned", "fret_conf", "fret_dropped", "abs_resid_ms", "density_1s",
            "below_e2")
CLASSES = {"electric_guitar_clean": "cls_clean", "distorted_electric_guitar": "cls_dist",
           "acoustic_guitar": "cls_acoustic"}
MODEL_FILE = Path(__file__).with_name("note_confidence.json")


def note_features(notes: Sequence[Mapping[str, Any]], tab: Mapping[str, Any] | None,
                  bp_notes: Sequence[Mapping[str, Any]] | None) -> np.ndarray:
    """(notes, len(FEATURES)) for the quantised ``guitar_all`` notes (``i`` = index into them, as the tab's uids)."""
    n = len(notes)
    X = np.zeros((n, len(FEATURES)))
    if not n:
        return X
    col = {f: k for k, f in enumerate(FEATURES)}
    on = np.array([float(x["onset_s"]) for x in notes])
    pitch = np.array([int(x["pitch"]) for x in notes])
    med = float(np.median(pitch))
    order = np.argsort(on)
    so = on[order]
    fret: dict[int, Mapping[str, Any]] = {}
    for tid, tr in ((tab or {}).get("tracks") or {}).items():
        if tid.startswith("gtr_") or tid == "guitar_all":
            for f in tr.get("notes") or []:
                uid = str(f.get("uid") or "")
                if uid.startswith("guitar_all:"):
                    fret[int(uid.split(":", 1)[1])] = f
    bp = sorted((float(b["onset_s"]), int(b["pitch"])) for b in bp_notes or [])
    bp_on = [t for t, _ in bp]
    for k, x in enumerate(notes):
        t, p = on[k], pitch[k]
        lo, hi = np.searchsorted(so, t - 0.03), np.searchsorted(so, t + 0.03, side="right")
        same = order[lo:hi]
        X[k, col["log_dur"]] = math.log(max(1e-3, float(x.get("offset_s") or t) - t))
        X[k, col["poly"]] = len(same) - 1
        X[k, col["rel_pitch"]] = p - med
        X[k, col["octave_partner"]] = float(any(abs(int(pitch[j]) - p) == 12 for j in same))
        a, b = bisect.bisect_left(bp_on, t - 0.05), bisect.bisect_right(bp_on, t + 0.05)
        near = [q for _t, q in bp[a:b]]
        X[k, col["bp_same"]] = float(p in near)
        X[k, col["bp_octave"]] = float(any(abs(q - p) in (12, 24) for q in near) and p not in near)
        c = CLASSES.get(str(x.get("instrument")))
        if c:
            X[k, col[c]] = 1.0
        line = x.get("line")
        X[k, col["part_p"]] = float(x.get("p") if x.get("p") is not None else 1.0)
        X[k, col["unassigned"]] = float(line in (None, "u"))
        f = fret.get(int(x.get("i", k)))
        placed = f is not None and f.get("string") is not None
        X[k, col["fret_conf"]] = float(f.get("conf") or 0.0) if placed else 0.0
        X[k, col["fret_dropped"]] = float(not placed)
        X[k, col["abs_resid_ms"]] = abs(float(x.get("resid_ms") or 0.0))
        X[k, col["density_1s"]] = float(np.searchsorted(so, t + 1.0) - np.searchsorted(so, t - 1.0)) / 2.0
        X[k, col["below_e2"]] = float(p < 40)
    return X


class NoteModel:
    """Standardised logistic regression: P(correct) = sigmoid(b + w . (x - mean) / scale)."""

    def __init__(self, mean: Sequence[float], scale: Sequence[float], coef: Sequence[float], intercept: float,
                 meta: Mapping[str, Any] | None = None) -> None:
        self.mean, self.scale = np.asarray(mean, float), np.asarray(scale, float)
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.meta = dict(meta or {})

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, *, C: float = 1.0, meta: Mapping[str, Any] | None = None) -> "NoteModel":
        from sklearn.linear_model import LogisticRegression

        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        lr = LogisticRegression(C=C, max_iter=2000).fit((X - mean) / scale, y)
        return cls(mean, scale, lr.coef_[0], float(lr.intercept_[0]), meta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.intercept + ((np.asarray(X, float) - self.mean) / self.scale) @ self.coef
        return 1.0 / (1.0 + np.exp(-z))

    def to_json(self) -> dict[str, Any]:
        return {"format": "bandscribe.note_confidence/1", "features": list(FEATURES), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist(), "intercept": self.intercept, "meta": self.meta}

    @classmethod
    def load(cls, path: Path = MODEL_FILE) -> "NoteModel | None":
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if list(d.get("features") or []) != list(FEATURES):
            return None
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("meta"))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins weighted by their share of notes."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def capture(score: np.ndarray, wrong: np.ndarray, share: float = 0.2) -> float:
    """Share of the wrong notes among the ``share`` lowest-scored notes (ties broken by position)."""
    wrong = np.asarray(wrong, bool)
    if not wrong.any():
        return float("nan")
    k = max(1, int(round(share * len(score))))
    top = np.argsort(np.asarray(score, float), kind="stable")[:k]
    return float(wrong[top].sum() / wrong.sum())
