"""Loader checks on the real datasets under data/datasets (skipped for any dataset not on this PC).

Marked ``data(name)`` (P0's conftest skips when the dataset is not in datasets.lock.json). The extra
``skipif`` uses ``bandscribe.datasets`` availability, which also skips a *partial* fetch (e.g. an interrupted
Google Drive download is recorded in the lock but misses required folders).
"""

from __future__ import annotations

import collections

import pytest

from bandscribe import datasets
from bandscribe.datasets import store


def _need(name: str):
    return pytest.mark.skipif(not store.is_available(name), reason=f"{name} not fetched on this PC")


def _conventions(t) -> None:
    assert t.families_present
    styles = t.meta.get("styles")  # IDMT: expression "HA" = natural harmonic (sounding pitch != tuning + fret)
    for lid, ns in t.notes.items():
        if t.string_fret and t.tuning and lid in t.string_fret:
            for i, ((s, f), n) in enumerate(zip(t.string_fret[lid], ns.notes)):
                if styles and styles[i]["expression"] == "HA":
                    continue
                assert t.tuning[s - 1] + f == n.pitch


@pytest.mark.data("guitarset")
@_need("guitarset")
def test_guitarset_real():
    ids = ["00_BN1-129-Eb_comp", "03_Rock1-90-C#_solo", "05_SS3-98-C_comp"]
    for tid in ids:
        t = datasets.load_track("guitarset", tid)
        _conventions(t)
        assert t.string_fret is not None and (datasets.dataset_root("guitarset") / t.mix).is_file()
        # JAMS source 0 = low E -> the lowest notes sit on string 6
        low = min(t.notes["gtr"].notes, key=lambda n: n.pitch)
        idx = t.notes["gtr"].notes.index(low)
        assert t.string_fret["gtr"][idx][0] >= 5
    root = datasets.dataset_root("guitarset")
    from bandscribe.datasets.loaders import guitarset

    idx = guitarset.index(root)
    assert len(idx) == 360 and all("mic" in v for v in idx.values())
    assert len(datasets.track_ids("guitarset", split="dev")) == len(datasets.track_ids("guitarset", split="test")) == 180
    players = collections.Counter(k[:2] for k in idx)
    assert players == {p: 60 for p in ("00", "01", "02", "03", "04", "05")}


@pytest.mark.data("idmt_bass_st")
@_need("idmt_bass_st")
def test_idmt_real():
    tracks = list(datasets.iter_tracks("idmt_bass_st"))
    assert len(tracks) == 17 and datasets.track_ids("idmt_bass_st") == [t.track_id for t in tracks]
    for t in tracks:
        _conventions(t)
        assert t.string_fret is not None and len(t.tuning) in (4, 5)


@pytest.mark.data("filobass")
@_need("filobass")
def test_filobass_real():
    tracks = list(datasets.iter_tracks("filobass"))
    assert len(tracks) == 48 and datasets.track_ids("filobass") == [t.track_id for t in tracks]
    t = tracks[0]
    assert t.notes["bass"].notes and t.downbeats_s and t.beats_s
    assert all(b2 > b1 for b1, b2 in zip(t.downbeats_s, t.downbeats_s[1:]))


@pytest.mark.data("egdb")
@_need("egdb")
def test_egdb_real():
    root = datasets.dataset_root("egdb")
    from bandscribe.datasets.loaders import egdb

    idx = egdb.index(root)
    assert len(idx["labels"]) == 240
    t = egdb.load_track(root, "di_001")
    _conventions(t)
    test_clips = [c for c in idx["labels"] if egdb.clip_split(c) == "test"]
    assert len(test_clips) == 25  # README: 216-241, and 241 does not exist
    # every amp render of every clip is there (the gate-m2 distortion subset needs Marshall/Mesa/Plexi)
    assert {t: sum(1 for (tt, _) in idx["audio"] if tt == t) for t in egdb.TIMBRES} == {t: 240 for t in egdb.TIMBRES}
    di = [egdb.load_track(root, f"di_{c:03d}") for c in sorted(idx["labels"])]
    for t in di:
        _conventions(t)
        assert t.tuning == [64, 59, 55, 50, 45, 40]
    # measured 2026-10-03: 84 notes below their labelled string's open pitch (label noise), 31 without a string
    assert sum(t.meta["string_label_errors"] for t in di) == 84
    assert sum(t.meta["notes_without_string"] for t in di) == 31


@pytest.mark.data("cambridge_mt")
@_need("cambridge_mt")
def test_cambridge_real():
    pytest.importorskip("yaml")
    tracks = {t.track_id: t for t in datasets.iter_tracks("cambridge_mt")}
    assert datasets.track_ids("cambridge_mt") == list(tracks)
    assert len(tracks) >= 4
    for t in tracks.values():
        assert sum(1 for ln in t.lines if ln.kind == "guitar") >= 2
        assert "guitar" in t.families_present
