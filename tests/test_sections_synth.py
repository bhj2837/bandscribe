"""Sections + repeat map on synthetic songs (M1_M2_SPEC 9.3 item 4)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from bandscribe import atomic
from bandscribe.analysis import sections
from grid_testlib import onsets_from, synth_song

SHIFT_S = 0.030


@pytest.fixture(scope="module")
def ababca():
    # the second A (section index 2) is played 30 ms late relative to the grid
    y, grid, hits = synth_song("ABABCA", shift={2: SHIFT_S})
    ons = onsets_from(hits, duration=grid["duration_s"] + 1)
    sec, rep = sections.analyze_signal(y, grid, ons, {})
    return y, grid, ons, sec, rep


def test_labels_and_boundaries(ababca):
    _y, grid, _ons, sec, _rep = ababca
    got = [(s["label"], s["start_bar"], s["end_bar"]) for s in sec["sections"]]
    assert got == [("A", 1, 9), ("B", 9, 17), ("A", 17, 25), ("B", 25, 33), ("C", 33, 41), ("A", 41, 49)]
    downbeats = {b["start_s"] for b in grid["bars"]}
    assert all(s["start_s"] in downbeats for s in sec["sections"])  # boundaries on downbeats
    assert [s["id"] for s in sec["sections"]] == ["S1", "S2", "S3", "S4", "S5", "S6"]
    assert all(0.0 <= s["confidence"] <= 1.0 for s in sec["sections"])


def test_repeat_groups_and_offset(ababca):
    _y, _grid, _ons, sec, rep = ababca
    groups = {g["label"]: g for g in rep["groups"]}
    assert set(groups) == {"A", "B"}  # C occurs once -> no group
    occ = {o["section"]: o for o in groups["A"]["occurrences"]}
    assert set(occ) == {"S1", "S3", "S6"}
    # relative timing of the shifted repeat vs the other A occurrences
    assert occ["S3"]["offset_s"] - occ["S1"]["offset_s"] == pytest.approx(SHIFT_S, abs=0.005)
    assert occ["S6"]["offset_s"] - occ["S1"]["offset_s"] == pytest.approx(0.0, abs=0.005)
    ref = groups["A"]["reference"]
    assert occ[ref]["offset_s"] == 0.0
    b_occ = {o["section"]: o["offset_s"] for o in groups["B"]["occurrences"]}
    assert abs(b_occ["S2"] - b_occ["S4"]) < 0.005


def test_bar_matches(ababca):
    _y, _grid, _ons, _sec, rep = ababca
    by_bar = {b["bar"]: b["matches"] for b in rep["bars"]}
    assert len(by_bar) == 48
    m1 = by_bar[1]
    assert m1, "bar 1 should repeat"
    assert all(m["score"] >= 0.6 for m in m1)
    assert [m["score"] for m in m1] == sorted((m["score"] for m in m1), reverse=True)
    # bar 1 (first A) matches bar 17 (second A, played 30 ms late) with that offset
    m17 = [m for m in m1 if m["bar"] == 17]
    assert m17 and m17[0]["offset_s"] == pytest.approx(SHIFT_S, abs=0.005)


def test_short_and_single_section():
    y, grid, hits = synth_song("A", n_bars=4)
    sec, rep = sections.analyze_signal(y, grid, onsets_from(hits, duration=grid["duration_s"] + 1), {})
    assert [(s["label"], s["start_bar"], s["end_bar"]) for s in sec["sections"]] == [("A", 1, 5)]
    assert rep["groups"] == []


def test_empty_grid():
    sec, rep = sections.analyze_signal(np.zeros(22050, dtype=np.float32),
                                       {"bars": [], "beats": []}, None, {})
    assert sec["sections"] == [] and rep["bars"] == []


def _dump_hash(tmp_path, name, doc) -> str:
    p = tmp_path / name
    atomic.write_json(p, doc)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_determinism_across_thread_settings(tmp_path, ababca):
    from threadpoolctl import threadpool_limits

    y, grid, ons, sec1, rep1 = ababca
    with threadpool_limits(limits=4):  # BLAS allowed 4 threads outside the stage
        sec2, rep2 = sections.analyze_signal(y, grid, ons, {})
    assert _dump_hash(tmp_path, "a.json", sec1) == _dump_hash(tmp_path, "b.json", sec2)
    assert _dump_hash(tmp_path, "c.json", rep1) == _dump_hash(tmp_path, "d.json", rep2)
    assert json.dumps(sec1) == json.dumps(sec2)


def test_schema(ababca):
    from bandscribe.schema.sections import RepeatMap, Sections

    _y, _grid, _ons, sec, rep = ababca
    Sections.model_validate(sec)
    RepeatMap.model_validate(rep)


def test_laplacian_is_symmetric_psd(ababca):
    """The lag-domain median filter breaks symmetry; eigh would then read one triangle (negative eigenvalues
    down to -0.04 on real songs)."""
    y, grid, _ons, _sec, _rep = ababca
    bt = np.asarray([b["t_s"] for b in grid["beats"]])
    C, M = sections._beat_features(y, bt, float(grid["bars"][-1]["end_s"]))
    F = sections._stacked(C, M, 8)
    _evecs, evals, k = sections._laplacian_embedding(F, M, 16)
    assert evals.min() > -1e-9 and 2 <= k <= 16
