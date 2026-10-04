"""Synthetic scenes: line maps (R1–R9), note counts, shared flags, determinism, GT-only mode."""

from __future__ import annotations

import numpy as np
import pytest

from bandscribe.eval import remix

EXPECTED_LINES = {
    "A": ["rhythm", "lead"], "A_rhythm_only": ["rhythm"], "B": ["rhythm", "lead"], "F": ["rhythm", "lead"],
    "ADT": ["rhythm", "lead"], "quad": ["rhythm", "lead"], "P70": ["rhythm", "lead"], "P_hard": ["rhythm", "lead"],
    "twin": ["lead1", "lead2"], "twin_double": ["rhythm", "lead1", "lead2"], "same_part_double": ["lead"],
    "centre_unison": ["rhythm", "lead"],
}
RHYTHM_NOTES = 4 * 8 * 3   # 4 bars x 8 eighths x 3-note power chords
LEAD_NOTES = len(remix.LEAD_PHRASE)


@pytest.mark.parametrize("scenario", remix.SCENARIOS)
def test_line_maps_and_note_counts(scenario):
    sc = remix.make_scene(scenario, render=False)
    assert [ln.id for ln in sc.lines] == EXPECTED_LINES[scenario]
    counts = {ln: sum(1 for n in sc.gt.notes if n.line == ln) for ln in EXPECTED_LINES[scenario]}
    if "rhythm" in counts:
        assert counts["rhythm"] == RHYTHM_NOTES  # a doubled line lists its notes once (first take)
    if scenario in ("A", "B", "F", "ADT", "quad", "P70", "P_hard"):
        assert counts["lead"] == LEAD_NOTES
    if scenario.startswith("twin"):
        assert counts["lead1"] == counts["lead2"] == LEAD_NOTES
    # every GT note is on the 120 BPM grid within the jitter (+ strum offset)
    for n in sc.gt.notes:
        assert 0 <= n.onset_s < 8.0 and n.offset_s > n.onset_s
        assert abs(((n.bar - 1) * 2.0 + n.tick / 48 * 0.5) - n.onset_s) < 1e-3


def test_doubles_keep_takes_in_one_line():
    for sc_name, n_takes in (("A", 2), ("quad", 4), ("ADT", 2), ("same_part_double", 2), ("F", 2)):
        sc = remix.make_scene(sc_name, render=False)
        line = "lead" if sc_name == "same_part_double" else "rhythm"
        assert sum(1 for t in sc.takes if t.line == line) == n_takes
        gl = next(ln for ln in sc.lines if ln.id == line)
        assert len(gl.takes) == n_takes


def test_centre_unison_shared_flags():
    sc = remix.make_scene("centre_unison", render=False)
    shared = [n for n in sc.gt.notes if n.shared]
    lead_shared = [n for n in shared if n.line == "lead"]
    rhythm_shared = [n for n in shared if n.line == "rhythm"]
    assert len(lead_shared) == 16 == len(rhythm_shared)  # 2 bars x 8 eighths, roots only
    assert all(n.onset_s < 4.0 for n in shared)
    for sc_name in ("A", "twin", "B"):
        assert not any(n.shared for n in remix.make_scene(sc_name, render=False).gt.notes)


def test_render_false_gives_identical_notes():
    for sc_name in ("A", "twin_double", "centre_unison", "same_part_double"):
        a = remix.make_scene(sc_name, render=False)
        b = remix.make_scene(sc_name, render=True)
        assert a.gt.model_dump() == b.gt.model_dump()


def test_scene_determinism_and_mix_is_sum_of_takes():
    a = remix.make_scene("A", seed=0)
    b = remix.make_scene("A", seed=0)
    assert np.array_equal(a.mix, b.mix)
    assert np.allclose(a.mix, np.sum([t.stereo for t in a.takes], axis=0))
    c = remix.make_scene("A", seed=1)
    assert not np.array_equal(a.mix, c.mix)


def test_parity_rows_match_scene_builder():
    """make_scene('A', seed=0) is router_exp row 1 exactly (same seeds, same RNG order)."""
    rows = remix.parity_scenes()
    assert np.array_equal(remix.make_scene("A").mix, rows["A  doubles L/R + centre lead 0dB"])
    assert np.allclose(remix.make_scene("A", lead_gain_db=20 * np.log10(0.5)).mix, rows["A  doubles L/R + centre lead -6dB"])
    assert np.array_equal(remix.make_scene("twin").mix, rows["TRAP twin leads L/R, no rhythm"])


def test_full_band_adds_parts_and_keeps_guitars():
    base = remix.make_scene("A")
    band = remix.make_scene("A", full_band=True)
    assert np.array_equal(base.guitar_stereo(), band.guitar_stereo())
    kinds = {t.kind for t in band.takes}
    assert {"guitar", "bass", "drums", "vocals", "keys"} <= kinds
    assert "bass" in [ln.id for ln in band.lines]
    assert np.abs(band.stem("bass")).max() > 0


def test_write_scene(tmp_path):
    import json

    import soundfile as sf

    sc = remix.make_scene("P70")
    out = remix.write_scene(sc, tmp_path / "scene")
    x, sr = sf.read(str(out / "mix.wav"), dtype="float32")
    assert sr == 44100 and x.shape == (int(8.0 * 44100), 2)
    assert np.allclose(x, sc.mix.T, atol=1e-6)
    meta = json.loads((out / "scene.json").read_text(encoding="utf-8"))
    assert meta["item"] == "synth:P70_s0"
    assert all(":" not in p.name for p in out.rglob("*"))
    tm = json.loads((out / "tempo_map.json").read_text(encoding="utf-8"))
    assert tm["beats"][4] == {"t_s": 2.0, "bar": 2, "beat": 1}
