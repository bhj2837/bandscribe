"""Repeat residual pass 1 (M1_M2_SPEC 9.3 item 6)."""
from __future__ import annotations

import numpy as np
import pytest

from gtab.analysis import repetition
from grid_testlib import riff_song


@pytest.fixture(scope="module")
def riff():
    return riff_song(n_occ=3, lead_in=(1, 2))


def test_lead_bars_stand_out(riff):
    y, grid, secs, rep, lead_bars = riff
    out, doc = repetition.residual(y, 44100, grid, secs, rep, min_occurrences=3)
    assert doc["groups_used"] == ["A"] and doc["skipped"] == []
    vals = {b["bar"]: b["resid_rel_db"] for b in doc["per_bar"]}
    assert all(v is not None for v in vals.values())
    lead = np.mean([vals[b] for b in lead_bars])
    plain = np.mean([vals[b] for b in vals if b not in lead_bars])
    assert lead - plain >= 10.0, (lead, plain)
    assert out.dtype == np.float32 and out.shape == y.shape
    assert not np.any(out[: int(0.9 * 44100)])  # nothing before the first occurrence
    assert doc["method"] == "beat-aligned median background, repet3 soft mask (W=min(median,V))"


def test_too_few_occurrences_are_skipped(riff):
    y, grid, secs, rep, _ = riff
    out, doc = repetition.residual(y, 44100, grid, secs, rep, min_occurrences=4)
    assert doc["groups_used"] == []
    assert doc["skipped"][0]["label"] == "A"
    assert all(b["resid_rel_db"] is None and b["group"] is None for b in doc["per_bar"])
    assert not np.any(out)


def test_offsets_are_applied():
    # the third occurrence is played 40 ms late; the repeat map says so -> still a clean residual
    y, grid, secs, rep, lead_bars = riff_song(n_occ=3, lead_in=(1,))
    sr = 44100
    bar_s = 2.0
    shift = int(0.040 * sr)
    start = int(round((1.0 + 2 * 4 * bar_s) * sr))
    seg = y[start:start + int(4 * bar_s * sr)].copy()
    y2 = y.copy()
    y2[start:start + seg.size] = 0
    y2[start + shift:start + shift + seg.size] = seg[: y2.size - start - shift]
    rep_ok = {**rep, "groups": [{**rep["groups"][0], "occurrences": [
        {**o, "offset_s": 0.040 if o["section"] == "S3" else 0.0} for o in rep["groups"][0]["occurrences"]]}]}
    _, doc_ok = repetition.residual(y2, sr, grid, secs, rep_ok, 3)
    _, doc_bad = repetition.residual(y2, sr, grid, secs, rep, 3)
    plain_ok = np.mean([b["resid_rel_db"] for b in doc_ok["per_bar"] if b["bar"] in range(9, 13)])
    plain_bad = np.mean([b["resid_rel_db"] for b in doc_bad["per_bar"] if b["bar"] in range(9, 13)])
    assert plain_ok < plain_bad - 1.0  # applying the offset leaves less of the repeating riff in the residual


def test_background_mask_matches_repet3_formula():
    rng = np.random.default_rng(0)
    V = np.abs(rng.normal(size=(3, 5, 7)))
    Mb = repetition.background_mask(V)
    med = np.median(V, axis=0)
    W = np.minimum(med, V)
    ref = W ** 2 / (W ** 2 + np.maximum(V - W, 0) ** 2 + 1e-12)
    assert np.array_equal(Mb, ref)
    assert Mb.shape == V.shape and np.all((0 <= Mb) & (Mb <= 1))
