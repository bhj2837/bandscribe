"""GP3-5 import via PyGuitarPro -> RefScore -> expanded GtNotes (repeats, endings, pickup, 6/8, capo, unison,
ties, triplets, cp949 track names) and the gt.yaml skeleton."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("guitarpro")
sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
import gp_make  # noqa: E402

from bandscribe.eval import gp_import, gtstore, refscore  # noqa: E402


@pytest.fixture(scope="module")
def gp5(tmp_path_factory):
    return gp_make.make_fixture_gp5(tmp_path_factory.mktemp("gp") / "fixture.gp5")


def _key(n):
    return (n.line, n.bar, round(n.tick, 3), round(n.dur_ticks, 3), n.pitch, n.string, n.fret, n.shared)


def test_refscore_tracks_and_conventions(gp5):
    rs = gp_import.read_gp345(gp5)
    assert rs.source["importer"] == "pyguitarpro-0.11" and rs.source["encoding"] == "cp949"
    assert rs.tracks[0].name == gp_make.TRACK1_NAME and rs.tracks[1].name == "Lead"
    assert rs.tracks[0].tuning == gp_make.STANDARD  # string 1 = highest
    assert rs.tracks[0].capo == gp_make.CAPO1 and rs.tracks[1].capo == 0
    for n in rs.notes:
        t = rs.tracks[n.track - 1]
        assert n.pitch == t.tuning[n.string - 1] + t.capo + n.fret
    mb = rs.masterbars
    assert (mb[1].repeat_open, mb[2].repeat_count, mb[2].alternate, mb[3].alternate) == (True, 2, 1, 2)
    assert (mb[4].numerator, mb[4].denominator) == (6, 8)


def test_expansion_matches_expected(gp5):
    rs = gp_import.read_gp345(gp5)
    order = refscore.play_order(rs.masterbars)
    assert order == gp_make.PLAY_ORDER
    assert refscore.detect_pickup(rs.masterbars, order)
    gtn = refscore.expand(rs, song_id="fx", track_lines={1: "L1", 2: "L2"})
    assert sorted(_key(n) for n in gtn.notes) == sorted(gp_make.EXPECTED)
    assert [(b["bar"], b["numerator"], b["denominator"], b["beat_unit"]) for b in gtn.bars] == gp_make.EXPECTED_BARS
    assert gtn.bars[5]["nominal_bpm"] == pytest.approx(80.0)  # 6/8 at quarter = 120 -> dotted quarter = 80
    tied = [n for n in gtn.notes if n.tied]
    assert len(tied) == 2 and all(n.dur_ticks == 96 for n in tied)  # the tie merged two quarters (bars 1 and 3)
    assert all(n.shared == (n.bar == 6) for n in gtn.notes)


def test_bar_order_override_and_no_pickup(gp5):
    rs = gp_import.read_gp345(gp5)
    gtn = refscore.expand(rs, song_id="fx", track_lines={1: "L1"}, bar_order=[2, 3, 4], pickup=False)
    assert [b["bar"] for b in gtn.bars] == [1, 2, 3]
    assert {n.line for n in gtn.notes} == {"L1"}
    with pytest.raises(ValueError):
        refscore.play_order(rs.masterbars, [1, 99])


def test_cp1252_fallback_and_override(tmp_path, gp5):
    latin = gp_make.make_fixture_gp5(tmp_path / "latin.gp5", track1_name="Café", encoding="cp1252")
    rs = gp_import.read_gp345(latin)
    assert rs.tracks[0].name == "Café" and rs.source["encoding"] == "cp1252"
    forced = gp_import.read_gp345(gp5, encoding="cp1252")  # gt.yaml override wins, even if it garbles
    assert forced.source["encoding"] == "cp1252" and forced.tracks[0].name != gp_make.TRACK1_NAME


def test_nominal_tempo_map_and_fill_times(gp5):
    rs = gp_import.read_gp345(gp5)
    gtn = refscore.expand(rs, song_id="fx", track_lines={1: "L1", 2: "L2"})
    tm = refscore.nominal_tempo_map(gtn)
    timed = refscore.fill_times(gtn, tm)
    first = next(n for n in timed.notes if n.bar == 0)
    assert first.onset_s == pytest.approx(0.0)
    b1 = next(n for n in timed.notes if n.bar == 1 and n.line == "L1" and n.tick == 0)
    assert b1.onset_s == pytest.approx(0.5)  # 1/4 pickup at 120 BPM
    b6 = next(n for n in timed.notes if n.bar == 6 and n.tick == 0)
    assert b6.onset_s == pytest.approx(0.5 + 4 * 2.0 + 1.5)


def test_skeleton_yaml_is_valid_and_rebuilds(tmp_path, gp5):
    import yaml

    from bandscribe.schema.gt import GtSpec

    rs = gp_import.read_gp345(gp5)
    order = refscore.play_order(rs.masterbars)
    text = gtstore.skeleton_yaml("fx_song", rs, song_key="f-0123456789abcdef", ref_file="ref.gp5",
                                 expanded_bars=len(order), encoding="cp949", pickup=True)
    assert "end_bar: 25" in text and "families_present" in text
    spec = GtSpec.model_validate(yaml.safe_load(text))
    assert [ln.id for ln in spec.lines] == ["L1", "L2"] and spec.families_present == ["guitar"]
    assert spec.reference["expanded_bars"] == 7 and spec.reference["pickup"] is True
    d = tmp_path / "fx_song"
    d.mkdir()
    (d / "gt.yaml").write_text(text, encoding="utf-8")
    rs.save(d / "refscore.json")
    gtn = gtstore.rebuild_gt_notes(gtstore.load_song(d))
    assert sorted(_key(n) for n in gtn.notes) == sorted(gp_make.EXPECTED)
    gtstore.save_spec_fields(d / "gt.yaml", {"alignment": {"method": "dtw", "approved": True,
                                                          "approved_utc": "2026-09-30T00:00:00Z", "minutes_spent": 30}})
    spec2 = gtstore.load_spec(d / "gt.yaml")
    assert spec2.alignment["approved"] is True and spec2.lines == spec.lines
    assert "end_bar: 25" in (d / "gt.yaml").read_text(encoding="utf-8")  # comments elsewhere survive


def test_song_id_validation():
    with pytest.raises(gtstore.GtError):
        gtstore.check_song_id("My Song")
    assert gtstore.check_song_id("my_song-01") == "my_song-01"
