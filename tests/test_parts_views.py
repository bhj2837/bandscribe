"""M7c S9 views (bandscribe.parts.views): section roles, the Role view and the Player view with its policies."""

from __future__ import annotations

import pytest

from bandscribe.parts import views as V

BAR_S = 2.0


def _note(bar: int, beat: float, pitch: int, line: str, dur_beats: float = 0.5, shared: bool = False) -> dict:
    t = bar * BAR_S + beat * BAR_S / 4
    return {"bar": bar, "gtick": int((bar * 4 + beat) * 480), "dur": int(dur_beats * 480), "pitch": pitch,
            "onset_s": t, "offset_s": t + dur_beats * BAR_S / 4 * 0.9, "line": line, "shared": shared,
            "instrument": "distorted_electric_guitar"}


def chords(line: str, bars: range, root: int = 40) -> list[dict]:
    return [_note(b, beat, root + iv, line) for b in bars for beat in (0, 1, 2, 3) for iv in (0, 7, 12)]


def melody(line: str, bars: range, base: int = 67) -> list[dict]:
    return [_note(b, beat, base + (b + int(beat * 2)) % 5, line) for b in bars for beat in (0, 0.5, 1, 2, 2.5, 3)]


SECS = [{"id": "S1", "start_bar": 0, "end_bar": 4}, {"id": "S2", "start_bar": 4, "end_bar": 8},
        {"id": "S3", "start_bar": 8, "end_bar": 12}]


def test_section_roles():
    g = chords("1", range(0, 12)) + melody("2", range(4, 8)) + melody("3", range(8, 9))
    roles = {(r["line"], r["section"]): r["role"] for r in V.section_roles(g, SECS)}
    assert roles[("1", "S1")] == "Rhythm" and roles[("2", "S2")] == "Lead"
    assert roles[("2", "S1")] == "Tacet" and roles[("3", "S3")] == "Fill"  # one bar of a four-bar section
    arp = [_note(b, beat, 52 + (int(beat * 2) % 4) * 3, "1", dur_beats=1.0) for b in range(4) for beat in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5)]
    assert V.section_roles(arp, SECS[:1])[0]["role"] == "Arpeggio"


def test_role_view_regroups_parts_by_role():
    g = chords("1", range(0, 12)) + melody("2", range(4, 8)) + chords("2", range(8, 12), root=45)
    tracks = {t["name"]: set(t["notes"]) for t in V.role_view(g, SECS)}
    assert list(tracks) == ["Rhythm", "Rhythm 2", "Lead"]
    lead2 = {k for k, n in enumerate(g) if n["line"] == "2" and 4 <= n["bar"] < 8}
    assert tracks["Lead"] == lead2  # part 2 is Lead in S2, Rhythm 2 in S3


def test_two_players_two_parts_and_no_simultaneous_parts_for_one_player():
    g = chords("1", range(0, 12)) + melody("2", range(4, 12))
    v = V.player_view(g, SECS, 2)
    assert v["violations"] == 0 and [t["name"] for t in v["tracks"]] == ["Player 1", "Player 2"]
    assert v["tracks"][0]["lines"] == ["1"] and v["tracks"][1]["lines"] == ["2"]
    assert any(m["text"] == "rest" and m["track"] == "Player 2" and m["bar"] == 0 for m in v["markers"])


def test_more_parts_than_players_send_the_least_important_to_gtr_extra():
    g = chords("1", range(0, 12)) + melody("2", range(0, 12)) + chords("3", range(0, 12), root=45)
    v = V.player_view(g, SECS, 2)
    assert v["violations"] == 0 and v["tracks"][-1]["name"] == V.EXTRA_NAME
    extra_lines = {g[k]["line"] for k in v["tracks"][-1]["notes"]}
    assert extra_lines == {"3"}  # a second rhythm part goes before the lead and the main rhythm (most notes first)
    assert all(m["track"] == V.EXTRA_NAME for m in v["markers"] if "no free player" in m["text"])


def test_a_part_that_never_overlaps_another_shares_its_player():
    g = chords("1", range(0, 12)) + melody("2", range(0, 4)) + melody("3", range(8, 12), base=72)
    v = V.player_view(g, SECS, 2)
    assert v["violations"] == 0 and len(v["tracks"]) == 2  # no extra track
    p2 = v["tracks"][1]
    assert p2["lines"] == ["2", "3"] and any(m["text"].startswith("plays Gtr 3") for m in v["markers"])


def test_idle_policy_double_copies_the_main_rhythm():
    g = chords("1", range(0, 12)) + melody("2", range(4, 12))
    rest = V.player_view(g, SECS, 2, idle_policy="rest")
    dbl = V.player_view(g, SECS, 2, idle_policy="double")
    s1_rhythm = {k for k, n in enumerate(g) if n["line"] == "1" and n["bar"] < 4}
    assert not (s1_rhythm & set(rest["tracks"][1]["notes"]))
    assert s1_rhythm <= set(dbl["tracks"][1]["notes"]) and dbl["copies"] == len(s1_rhythm)
    assert dbl["violations"] == 0 and any("doubles Gtr 1" in m["text"] for m in dbl["markers"])
    with pytest.raises(ValueError):
        V.player_view(g, SECS, 2, idle_policy="loop")


def test_unison_policy():
    g = chords("1", range(0, 8)) + melody("2", range(0, 8))
    g += [_note(b, 3.5, 64, "1", shared=True) for b in range(0, 8)]  # a unison figure both play
    shared = {k for k, n in enumerate(g) if n["shared"]}
    dup = V.player_view(g, SECS[:2], 2, unison_policy="duplicate")
    assert shared <= set(dup["tracks"][0]["notes"]) and shared <= set(dup["tracks"][1]["notes"])
    prim = V.player_view(g, SECS[:2], 2, unison_policy="primary_only")
    assert shared <= set(prim["tracks"][0]["notes"]) and not shared & set(prim["tracks"][1]["notes"])
    assert any(m["track"] == "Player 2" and m["text"] == "unison with Player 1" for m in prim["markers"])
    assert dup["violations"] == prim["violations"] == 0


def test_no_part_note_is_lost_in_a_view():
    """Review: a part with fewer than TACET_MIN notes in a section (labelled Tacet) still has its notes played."""
    g = chords("1", range(0, 12)) + melody("2", range(4, 12)) + [_note(1, 0, 70, "2"), _note(2, 0, 71, "2")]
    parted = {k for k, n in enumerate(g) if n["line"] in V.PART_LINES}
    for n_p in (1, 2, 3):
        v = V.player_view(g, SECS, n_p)
        assert set().union(*(set(t["notes"]) for t in v["tracks"])) == parted and v["violations"] == 0
    assert set().union(*(set(t["notes"]) for t in V.role_view(g, SECS))) == parted


def test_notes_before_the_first_section_are_played():
    g = chords("1", range(0, 12)) + [_note(-1, 3, 52, "1")]  # a pickup bar before S1
    v = V.player_view(g, SECS, 2)
    assert len(g) - 1 in v["tracks"][0]["notes"]


def test_extra_priority_stray_notes_first_then_fewer_notes():
    """Review 2026-10-17: a section's few stray notes go to Gtr extra before a real part, and of two equal parts the
    one with fewer notes."""
    g = chords("1", range(0, 4)) + melody("2", range(0, 4)) + [_note(0, 1, 70, "3"), _note(1, 1, 71, "3")]
    v = V.player_view(g, SECS[:1], 2)
    assert {g[k]["line"] for k in v["tracks"][-1]["notes"]} == {"3"}
    big, small = chords("1", range(0, 4)), chords("3", range(0, 1), root=45)
    g = big + melody("2", range(0, 4)) + small
    v = V.player_view(g, SECS[:1], 2)
    assert {g[k]["line"] for k in v["tracks"][-1]["notes"]} == {"3"}  # rhythm vs rhythm: the smaller one


def test_player_view_stays_fast_with_four_parts_and_players():
    import time

    secs = [{"id": f"S{i}", "start_bar": 2 * i, "end_bar": 2 * i + 2} for i in range(20)]
    g = (chords("1", range(0, 40)) + melody("2", range(0, 40)) + chords("3", range(0, 40), root=45)
         + melody("4", range(0, 40), base=74))
    for n_p in (2, 3, 4):
        t = time.perf_counter()
        v = V.player_view(g, secs, n_p)
        assert time.perf_counter() - t < 5.0 and v["violations"] == 0, n_p
