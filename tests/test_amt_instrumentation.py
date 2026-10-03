"""S3.5 v0 instrumentation estimator on synthetic evidence (gtab.amt.instrumentation)."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from gtab.amt import instrumentation as S
from gtab.amt import instruments as I

N_BARS = 20
BAR_S = 2.0
BARS = [(b + 1, b * BAR_S, (b + 1) * BAR_S) for b in range(N_BARS)]
FPS = 10.0
N_FR = int(N_BARS * BAR_S * FPS)


def energy(**rel_db: float) -> dict[str, np.ndarray]:
    """Mix at -20 dB; each named stem at mix + rel_db in every frame; unnamed stems silent."""
    e = {"mix": np.full(N_FR, -20.0, dtype=np.float32)}
    for stem in ("bass", "drums", "other", "vocals", "guitar", "piano"):
        e[stem] = np.full(N_FR, -20.0 + rel_db.get(stem, -100.0), dtype=np.float32)
    return e


def mix_notes(cls: str, bars: range, per_bar: int = 3) -> list[dict]:
    out = []
    for b in bars:
        for k in range(per_bar):
            t = b * BAR_S + 0.3 + 0.4 * k
            out.append({"onset_s": t, "offset_s": t + 0.2, "pitch": 60 + k, "instrument": cls})
    return out


def est(**kw):
    base = dict(profile="quality", bars=BARS, mix_notes=[], energy=energy(guitar=-6, bass=-6), sections=[],
                hint="auto", params={})
    base.update(kw)
    return S.estimate(**base)


def test_keys_present_via_stem_only():
    doc = est(energy=energy(guitar=-6, piano=-10), mix_notes=[])
    assert doc["classes"]["acoustic_piano"]["present"]  # 0.6 x 1.0 >= 0.5
    assert list(doc["classes"]["acoustic_piano"]["sources"]["stem_energy"]["stems"]) == ["piano"]
    assert set(doc["classes"]["organ"]["sources"]["stem_energy"]["stems"]) == {"piano", "other"}
    assert "keys" in S.estimated_families(doc)


def test_keys_present_via_mix_only():
    doc = est(energy=energy(guitar=-6), mix_notes=mix_notes("electric_piano", range(0, 4)))
    ev = doc["classes"]["electric_piano"]
    assert ev["present"] and ev["sources"]["mix_transcription"]["active_bars"] == 4
    assert not doc["classes"]["acoustic_piano"]["present"]


def test_keys_present_via_hint_and_explicit_list_suppresses():
    doc = est(hint="guitar,keys")
    assert all(doc["classes"][c]["present"] for c in I.FAMILIES["keys"])
    doc2 = est(energy=energy(guitar=-6, piano=-10), mix_notes=mix_notes("synth_lead", range(10)), hint="guitar,bass")
    assert not doc2["classes"]["acoustic_piano"]["present"]
    assert doc2["classes"]["synth_lead"]["p"] <= 0.05
    assert doc2["classes"]["acoustic_piano"]["sources"]["hint"]["effect"] == "not_listed"


def test_hint_never_suppresses_guitar():
    doc = est(energy=energy(guitar=-100), mix_notes=mix_notes("clean_electric_guitar", range(10)), hint="keys")
    assert doc["classes"]["clean_electric_guitar"]["present"]
    assert S.guitar_class_omissions(doc) == 0


def test_guitar_stem_rule_marks_all_three_guitar_classes():
    doc = est(energy=energy(guitar=-6), mix_notes=[])
    assert all(doc["classes"][c]["present"] for c in I.GUITAR_CLASSES)
    assert "guitar_stem_rule" in doc["classes"]["acoustic_guitar"]["sources"]


def test_min_notes_rule():
    few = mix_notes("organ", range(0, 3), per_bar=2)  # 6 notes < 8
    doc = est(energy=energy(guitar=-6), mix_notes=few)
    assert doc["classes"]["organ"]["sources"]["mix_transcription"]["e"] == 0.0
    assert not doc["classes"]["organ"]["present"]


@pytest.mark.parametrize("thr,present", [(0.54, True), (0.55, False)])
def test_threshold_edges(thr, present):
    # piano stem active in 18 of 20 bars -> e = 0.9 -> p = 0.6 * 0.9 = 0.54
    e = energy(guitar=-6, piano=-10)
    e["piano"][: int(2 * BAR_S * FPS)] = -200.0
    doc = est(energy=e, params={"threshold": thr})
    assert doc["classes"]["acoustic_piano"]["p"] == pytest.approx(0.54)
    assert doc["classes"]["acoustic_piano"]["present"] is present


def test_fast_profile_ignores_mix_transcription():
    doc = est(profile="fast", energy=energy(guitar=-6), mix_notes=mix_notes("organ", range(10)))
    assert "mix_transcription" not in doc["classes"]["organ"]["sources"]
    assert not doc["classes"]["organ"]["present"]


def test_masks_and_guitar_view_collision():
    doc = est(energy=energy(guitar=-6, piano=-3))
    assert doc["masks"]["mix_mono"] is None
    assert doc["masks"]["guitar_mono"] == doc["masks"]["guitar_pass"]
    assert "acoustic_piano" in doc["masks"]["guitar_pass"]
    doc2 = est(energy=energy(guitar=-6, piano=-3), guitar_view="mix_mono")
    assert doc2["masks"]["mix_mono"] is None  # the pass-1 all-class mask stays
    assert "acoustic_piano" in doc2["masks"]["guitar_pass"]


def test_per_section_uses_half_open_bar_ranges():
    e = energy(guitar=-6)
    e["piano"][:] = -200.0
    e["piano"][int(4 * BAR_S * FPS): int(8 * BAR_S * FPS)] = -30.0  # bars 5..8 active
    doc = est(energy=e, sections=[{"id": "s1", "start_bar": 1, "end_bar": 5}, {"id": "s2", "start_bar": 5, "end_bar": 9}])
    ps = {p["section"]: p["family_active"] for p in doc["per_section"]}
    assert ps["s1"]["keys"] == 0.0 and ps["s2"]["keys"] == 1.0
    assert ps["s1"]["guitar"] == 1.0


def test_guitar_omission_zero_property_randomized():
    rng = random.Random(7)
    classes = list(I.MT3_GROUPS)
    for _ in range(150):
        rel = {s: rng.choice([-100.0, -45.0, -30.0, -10.0]) for s in ("bass", "drums", "other", "vocals", "guitar",
                                                                       "piano")}
        notes = []
        for _k in range(rng.randint(0, 5)):
            notes += mix_notes(rng.choice(classes), range(rng.randint(0, 10)))
        hint = rng.choice(["auto", "keys", "guitar,keys", "bass,drums", "synth"])
        policy = rng.choice(I.GUITAR_MASK_POLICIES)
        doc = est(energy=energy(**rel), mix_notes=notes, hint=hint, params={"threshold": rng.random()},
                  guitar_mask=policy, profile=rng.choice(["quality", "fast"]))
        assert S.guitar_class_omissions(doc) == 0


def test_deterministic_and_schema_valid():
    kw = dict(energy=energy(guitar=-6, piano=-10, vocals=-3), mix_notes=mix_notes("organ", range(6)),
              sections=[{"id": "A", "start_bar": 1, "end_bar": 11}])
    a = json.dumps(est(**kw), ensure_ascii=False)
    b = json.dumps(est(**kw), ensure_ascii=False)
    assert a == b
    doc = json.loads(a)
    assert list(doc["classes"]) == list(I.MT3_GROUPS)
    assert list(doc["families"]) == list(I.FAMILIES)
    try:
        from gtab.schema.instrumentation import Instrumentation
    except ImportError:
        pytest.skip("P0 schema not present yet")
    Instrumentation.model_validate(doc)


def test_keys_warning_when_stem_but_no_mix_notes():
    doc = est(energy=energy(guitar=-6, piano=-10), mix_notes=[])
    assert any("건반" in w for w in doc["warnings"])
