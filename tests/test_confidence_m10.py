"""M10 confidence helpers (E35, E36): features, the logistic model and its JSON, ECE and the review-queue capture."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.parts import confidence as PC
from bandscribe.tab import confidence as CF


def test_ece_and_capture():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    assert CF.ece(p, np.array([1, 1, 0, 0])) == pytest.approx(0.1)
    assert CF.ece(np.full(4, 0.5), np.array([1, 0, 1, 0])) == pytest.approx(0.0)  # calibrated, no resolution
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.95, 0.85, 0.6, 0.65, 0.75])
    wrong = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    assert CF.capture(score, wrong) == 1.0 and np.isnan(CF.capture(score, np.zeros(10, bool)))


def test_note_features_and_model_round_trip(tmp_path):
    notes = [{"i": 0, "onset_s": 1.0, "offset_s": 1.5, "pitch": 52, "instrument": "distorted_electric_guitar",
              "line": "1", "p": 0.9, "resid_ms": -12.0},
             {"i": 1, "onset_s": 1.01, "offset_s": 1.4, "pitch": 64, "instrument": "electric_guitar_clean",
              "line": "u", "p": None, "resid_ms": 30.0},
             {"i": 2, "onset_s": 3.0, "offset_s": 3.2, "pitch": 38, "instrument": "acoustic_guitar", "line": "2", "p": 0.6}]
    tab = {"tracks": {"gtr_1": {"notes": [{"uid": "guitar_all:0", "string": 5, "fret": 7, "conf": 0.8}]}}}
    bp = [{"onset_s": 1.02, "pitch": 52}, {"onset_s": 1.0, "pitch": 76}]
    X = CF.note_features(notes, tab, bp)
    c = {f: i for i, f in enumerate(CF.FEATURES)}
    assert X.shape == (3, len(CF.FEATURES))
    assert X[0, c["poly"]] == 1 and X[0, c["octave_partner"]] == 1 and X[0, c["bp_same"]] == 1
    assert X[1, c["bp_octave"]] == 1 and X[1, c["unassigned"]] == 1 and X[1, c["cls_clean"]] == 1
    assert X[0, c["fret_conf"]] == 0.8 and X[1, c["fret_dropped"]] == 1 and X[2, c["below_e2"]] == 1
    assert X[0, c["abs_resid_ms"]] == 12.0
    rng = np.random.default_rng(1)
    Xr = rng.normal(size=(200, len(CF.FEATURES)))
    y = (Xr[:, 0] > 0).astype(int)
    m = CF.NoteModel.fit(Xr, y)
    assert m.predict(Xr[y == 1]).mean() > 0.7 > 0.3 > m.predict(Xr[y == 0]).mean()
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps(m.to_json()), encoding="utf-8")
    m2 = CF.NoteModel.load(path)
    assert np.allclose(m2.predict(Xr[:5]), m.predict(Xr[:5]))


def test_part_features():
    notes = [{"onset_s": t, "pitch": 60} for t in (0.0, 0.0, 1.0, 2.0)]
    sigs = [{"ild_db": 8.0, "coh": 0.6, "ms_db": 3.0}, {"ild_db": -8.0}, {"ild_db": 7.0}, {"ild_db": -1.0}]
    X, idx = PC.part_features(notes, sigs, ["1", "2", "1", None], [1.0, 0.9, 0.99, 0.5], ["R3", "R3", None, "R0"], 2)
    c = {f: i for i, f in enumerate(PC.FEATURES)}
    assert idx.tolist() == [0, 1, 2] and X.shape == (3, len(PC.FEATURES))
    assert X[0, c["abs_ild"]] == 8.0 and X[0, c["ild_same_side"]] == 1 and X[0, c["lab_R3"]] == 1
    assert X[2, c["lab_none"]] == 1 and X[0, c["poly"]] == 1 and X[0, c["k"]] == 2
