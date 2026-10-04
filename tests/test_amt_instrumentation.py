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
    """The M2 estimator (experiments): the mix pass as evidence."""
    doc = est(energy=energy(guitar=-6), mix_notes=mix_notes("electric_piano", range(0, 4)), mix_as_evidence=True)
    ev = doc["classes"]["electric_piano"]
    assert ev["present"] and ev["sources"]["mix_transcription"]["active_bars"] == 4
    assert ev["sources"]["mix_transcription"]["w"] == 0.9
    assert not doc["classes"]["acoustic_piano"]["present"]


def test_keys_present_via_hint_and_explicit_list_suppresses():
    doc = est(hint="guitar,keys")
    assert all(doc["classes"][c]["present"] for c in I.FAMILIES["keys"])
    doc2 = est(energy=energy(guitar=-6, piano=-10), mix_notes=mix_notes("synth_lead", range(10)), hint="guitar,bass")
    assert not doc2["classes"]["acoustic_piano"]["present"]
    assert doc2["classes"]["synth_lead"]["p"] <= 0.05
    assert doc2["classes"]["acoustic_piano"]["sources"]["hint"]["effect"] == "not_listed"


def test_hint_never_suppresses_guitar():
    doc = est(energy=energy(guitar=-100), mix_notes=mix_notes("clean_electric_guitar", range(10)), hint="keys",
              mix_as_evidence=True)
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
    # no transcription covers keys: the stem weight renormalises to 0.9; piano stem active in 12 of 20 bars
    # -> e = 0.6 -> p = 0.9 * 0.6 = 0.54
    e = energy(guitar=-6, piano=-10)
    e["piano"][: int(8 * BAR_S * FPS)] = -200.0
    doc = est(energy=e, mix_notes=None, params={"threshold": thr})
    assert doc["classes"]["acoustic_piano"]["p"] == pytest.approx(0.54)
    assert doc["classes"]["acoustic_piano"]["present"] is present


@pytest.mark.parametrize("profile", ["fast", "quality", "eval"])
def test_mix_pass_is_reported_but_never_evidence(profile):
    """Review 2026-10-04: eval's B0 mix pass must not change the estimate (eval would test another system)."""
    notes = mix_notes("organ", range(10))
    doc = est(profile=profile, energy=energy(guitar=-6), mix_notes=notes)
    src = doc["classes"]["organ"]["sources"]["mix_transcription"]
    assert src["n_notes"] == 30 and src["w"] == 0.0 and src["used"] is False
    assert not doc["classes"]["organ"]["present"]
    base = est(profile=profile, energy=energy(guitar=-6), mix_notes=None)
    strip = {c: (v["p"], v["present"]) for c, v in doc["classes"].items()}
    assert strip == {c: (v["p"], v["present"]) for c, v in base["classes"].items()}
    assert doc["masks"] == base["masks"] and doc["thresholds"]["mix_as_evidence"] is False


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


def test_keys_warning_when_stem_but_no_transcribed_keys_notes():
    doc = est(energy=energy(guitar=-6, piano=-10), mix_notes=None,
              presence={"mode": "sampled", "notes": [], "spans": [[0.0, 10.0]]})
    assert any(w.startswith("건반(piano) 스템 에너지는 있는데") for w in doc["warnings"])


# ------------------------------------------------------------- presence source (2026-10-04, KH / aotonat)


def window_notes(cls: str, spans, per_s: float, pitch0: int = 60) -> list[dict]:
    """``per_s`` notes per second of class ``cls`` inside each span (dense, like a pad decoded by MuScriptor)."""
    out = []
    for a, z in spans:
        n = int(round((z - a) * per_s))
        for k in range(n):
            t = round(a + k / per_s, 6)
            out.append({"onset_s": t, "offset_s": round(t + 0.5, 6), "pitch": pitch0 + k % 7, "instrument": cls})
    return out


KH_SPANS = [[2.0, 12.0], [20.0, 30.0]]  # 2 windows x 10 s = 10 of the 20 bars


def kh_energy():
    """怪獣の花唄-like stems: guitar stem active, piano stem in 16 % of bars, other stem in 71 % (pads)."""
    e = energy(guitar=-6, bass=-6, vocals=-3, drums=-6, other=-30, piano=-30)
    e["piano"][int(3.2 * BAR_S * FPS):] = -200.0  # piano active in about 3 of 20 bars
    e["other"][int(14.2 * BAR_S * FPS):] = -200.0  # other active in about 14 of 20 bars
    return e


def test_pad_case_synth_pad_absent_without_presence_and_present_with_it():
    """Synthetic pad case shaped like the first diagnosis of job f-e064a271b0834683 (怪獣の花唄): without the
    piano+other notes a pad stays below the threshold (other stem alone, x0.5) and the mask is guitar-only. The
    numbers are synthetic: on the real song, with strings in the piano+other mask, those notes came out as
    strings, not synth_pad, and the guitar view labelled no note non-guitar."""
    old = est(profile="quality", mix_notes=None, energy=kh_energy())
    assert not old["classes"]["synth_pad"]["present"] and old["classes"]["synth_pad"]["p"] < 0.5
    assert old["masks"]["guitar_pass"] == I.mask_guitar_only()
    notes = window_notes("synth_pad", KH_SPANS, per_s=70.0) + window_notes("organ", KH_SPANS, per_s=7.0, pitch0=48)
    notes.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))
    doc = est(profile="quality", mix_notes=None, energy=kh_energy(),
              presence={"mode": "sampled", "notes": notes, "spans": KH_SPANS})
    pad = doc["classes"]["synth_pad"]
    assert pad["present"] and pad["p"] >= 0.9
    src = pad["sources"]["presence_transcription"]
    assert src["n_notes"] == 1400 and src["covered_bars"] == 10 and src["active_bars"] == 10
    assert src["windows"] == 2 and src["active_windows"] == 2 and src["active_window_ratio"] == 1.0
    assert src["mode"] == "sampled" and src["saturation"] == 0.25 and src["w"] == 0.9
    assert doc["classes"]["organ"]["present"]
    gp = doc["masks"]["guitar_pass"]
    assert {"synth_pad", "organ"} <= set(gp) and set(I.GUITAR_CLASSES) <= set(gp)
    assert "synth_lead" not in gp and "string_ensemble" not in gp  # absent classes stay out
    assert doc["presence"]["class_notes"] == {"organ": 140, "synth_pad": 1400}
    assert "mix_transcription" not in pad["sources"] and S.guitar_class_omissions(doc) == 0


def test_strings_are_reported_but_join_only_the_strings_policy_mask():
    notes = window_notes("string_ensemble", KH_SPANS, per_s=3.0)
    pres = {"mode": "sampled", "notes": notes, "spans": KH_SPANS}
    doc = est(mix_notes=None, energy=kh_energy(), presence=pres)
    assert doc["classes"]["string_ensemble"]["present"] and "strings" in S.estimated_families(doc)
    assert doc["masks"]["guitar_pass"] == I.mask_guitar_only()  # default guitar+present: keys/synth only
    doc2 = est(mix_notes=None, energy=kh_energy(), presence=pres, guitar_mask="guitar+present+strings")
    assert "string_ensemble" in doc2["masks"]["guitar_pass"]


def test_presence_evidence_only_counts_notes_inside_the_windows_and_needs_min_notes():
    inside = window_notes("synth_lead", [[2.0, 4.0]], per_s=3.0)  # 6 notes < 8
    outside = window_notes("synth_lead", [[30.0, 38.0]], per_s=3.0)
    doc = est(mix_notes=None, presence={"mode": "sampled", "notes": inside + outside, "spans": [[2.0, 12.0]]})
    src = doc["classes"]["synth_lead"]["sources"]["presence_transcription"]
    assert src["n_notes"] == 6 and src["e"] == 0.0 and not doc["classes"]["synth_lead"]["present"]


def test_presence_without_grid_bars_uses_the_windows_as_units():
    notes = window_notes("synth_pad", KH_SPANS, per_s=5.0)
    ev = S.presence_evidence(notes, [], spans=KH_SPANS, classes=["synth_pad", "organ"], min_notes=8,
                             saturation=0.25)
    assert ev["synth_pad"]["covered_bars"] == 0 and ev["synth_pad"]["active_ratio"] == 1.0
    assert ev["synth_pad"]["e"] == 1.0 and ev["organ"]["e"] == 0.0


def test_full_presence_pass_uses_the_song_wide_saturation():
    notes = window_notes("electric_piano", [[0.0, 4.0]], per_s=2.0)  # 2 of 20 bars active = 0.1 -> e = 1
    doc = est(mix_notes=None, energy=energy(guitar=-6, piano=-10), presence={"mode": "full", "notes": notes,
                                                                            "spans": None})
    src = doc["classes"]["electric_piano"]["sources"]["presence_transcription"]
    assert src["covered_bars"] == N_BARS and src["active_bars"] == 2 and src["saturation"] == 0.1
    assert src["e"] == 1.0 and src["active_window_ratio"] is None and src["sections_needed"] is None


def test_mix_and_presence_form_one_transcription_term():
    """Same model, overlapping audio: max of the two, not a second noisy-OR term (experiments' M2 estimator)."""
    mix = mix_notes("organ", range(1)) + mix_notes("organ", range(10, 15), per_bar=1)  # 1 active bar of 20
    pres = mix_notes("organ", range(2)) + mix_notes("organ", range(10, 13), per_bar=1)  # 2 active bars of 20
    e = energy(guitar=-6, piano=-10)
    e["piano"][:] = -200.0
    e["piano"][: int(2 * BAR_S * FPS)] = -10.0  # piano stem active in 2 bars: agrees (10 %), adds 0.6 x 0.1
    doc = est(energy=e, mix_notes=mix, presence={"mode": "full", "notes": pres, "spans": None}, mix_as_evidence=True)
    ev = doc["classes"]["organ"]
    e_mix = ev["sources"]["mix_transcription"]["e"]
    e_pres = ev["sources"]["presence_transcription"]["e"]
    assert (e_mix, e_pres) == (0.5, 1.0)
    assert ev["p"] == pytest.approx(1 - (1 - 0.9 * max(e_mix, e_pres)) * (1 - 0.6 * 0.1), abs=1e-4)


def test_weights_renormalise_only_for_classes_no_pass_covers():
    # no transcription at all (no presence, no bass pass): the stem weight scales 0.6 -> 0.9 (cap)
    doc = est(energy=energy(guitar=-6, piano=-10), mix_notes=None)
    ev = doc["classes"]["acoustic_piano"]
    assert ev["sources"]["stem_energy"]["w"] == 0.9 and ev["p"] == pytest.approx(0.9)
    assert any("가중치 재정규화" in w for w in doc["warnings"])
    # a presence pass that covered the class keeps the nominal weights; it saw the piano stem active in its 5
    # covered bars and found no keys note, so the stem term is not attributed to the keys classes (p 0)
    doc2 = est(energy=energy(guitar=-6, piano=-10), mix_notes=None,
               presence={"mode": "sampled", "notes": [], "spans": [[0.0, 10.0]]})
    ev2 = doc2["classes"]["acoustic_piano"]
    st = ev2["sources"]["stem_energy"]
    assert st["w"] == 0.6 and st["stems"]["piano"]["attributed"] is False and st["stems"]["piano"]["e"] == 0.0
    assert st["stems"]["piano"]["covered_active_bars"] == 5 and st["stems"]["piano"]["uncovered_active_ratio"] == 0.75
    assert ev2["p"] == 0.0 and not ev2["present"]
    # classes outside the presence mask (brass) keep stem-only evidence, renormalised
    assert doc2["classes"]["trumpet"]["sources"]["stem_energy"]["w"] == 0.9
    assert "presence_transcription" not in doc2["classes"]["trumpet"]["sources"]
    assert S.renormalised_weights(["transcription", "stem_energy"]) == {"transcription": 0.9, "stem_energy": 0.6}
    assert S.renormalised_weights(["transcription"]) == {"transcription": 0.9}
    assert S.renormalised_weights([]) == {}


def test_skipped_presence_pass_covers_with_zero_evidence_and_no_renormalisation():
    """Review 2026-10-04 (3b): a pass that ran and found no window is coverage with e = 0, not a missing pass.
    A staccato piano stem (active in 70 % of the bars) used to reach p 0.63 through the renormalised stem weight."""
    e = energy(guitar=-6, bass=-6)
    e["piano"][: int(14 * BAR_S * FPS)] = -30.0  # active in 14 of 20 bars
    skipped = {"mode": "sampled", "skipped": "piano_other_inactive", "notes": [], "spans": None,
               "summary": {"view": "piano_other_presence"}}
    doc = est(mix_notes=None, energy=e, presence=skipped)
    ev = doc["classes"]["acoustic_piano"]
    assert ev["sources"]["presence_transcription"]["e"] == 0.0
    assert ev["sources"]["presence_transcription"]["skipped"] == "piano_other_inactive"
    assert ev["sources"]["stem_energy"]["w"] == 0.6 and ev["sources"]["stem_energy"]["stems"]["piano"]["attributed"]
    assert ev["p"] == pytest.approx(0.42) and not ev["present"]
    assert doc["presence"]["skipped"] == "piano_other_inactive" and doc["presence"]["view"] == "piano_other_presence"
    assert not any("재정규화" in w for w in doc["warnings"])  # a quiet piano/other is not a missing pass
    assert any("고르지 못했습니다" in w for w in doc["warnings"])  # but stems with sound and no window: say so
    assert doc["masks"]["guitar_pass"] == I.mask_guitar_only()


def test_presence_classes_follow_the_piano_other_mode():
    assert set(I.presence_evidence_classes("keys+guitar")) == set(I.PRESENCE_CLASSES)
    assert I.presence_evidence_classes("keys") == I.presence_evidence_classes("keys+guitar")
    assert not set(I.presence_evidence_classes("keys+guitar")) & set(I.GUITAR_CLASSES)


def test_guitar_omission_zero_with_presence_randomized():
    rng = random.Random(11)
    classes = list(I.PRESENCE_CLASSES) + list(I.GUITAR_CLASSES)
    for _ in range(80):
        spans = [[0.0, 10.0], [20.0, 30.0]][: rng.randint(1, 2)]
        notes = []
        for _k in range(rng.randint(0, 4)):
            notes += window_notes(rng.choice(classes), spans, per_s=rng.choice([0.5, 2.0, 20.0]))
        notes.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))
        doc = est(mix_notes=None, presence={"mode": "sampled", "notes": notes, "spans": spans},
                  energy=energy(guitar=rng.choice([-100.0, -6.0]), other=-30), hint=rng.choice(["auto", "keys"]),
                  guitar_mask=rng.choice(I.GUITAR_MASK_POLICIES))
        assert S.guitar_class_omissions(doc) == 0


# ------------------------------------------------- guards and attribution (review 2026-10-04: SC, 青と夏, KH)


SC_SPANS = [[0.0, 10.0], [10.0, 20.0], [26.0, 36.0]]


def test_sections_guard_rejects_a_burst_in_one_section():
    """SC's false electric_piano: 37 notes in two adjacent windows of one section."""
    notes = window_notes("electric_piano", SC_SPANS[:2], per_s=2.0)  # 40 notes, both windows in section S4
    e = energy(guitar=-6, piano=-10)
    pres = {"mode": "sampled", "notes": notes, "spans": SC_SPANS, "window_sections": ["S4", "S4", "S7"]}
    doc = est(mix_notes=None, energy=e, presence=pres)
    src = doc["classes"]["electric_piano"]["sources"]["presence_transcription"]
    assert src["n_notes"] == 40
    assert (src["active_sections"], src["sampled_sections"], src["sections_needed"]) == (1, 2, 2)
    assert src["e"] == 0.0 and not doc["classes"]["electric_piano"]["present"]
    assert doc["presence"]["rejected"] == {"electric_piano": "sections"}
    assert doc["masks"]["guitar_pass"] == I.mask_guitar_only()
    # the same notes over two sections count
    doc2 = est(mix_notes=None, energy=e, presence=dict(pres, window_sections=["S4", "S5", "S7"]))
    assert doc2["classes"]["electric_piano"]["present"] and "electric_piano" in doc2["masks"]["guitar_pass"]
    # one sampled section only: one section is enough
    doc3 = est(mix_notes=None, energy=e, presence=dict(pres, window_sections=["S4", "S4", "S4"]))
    assert doc3["classes"]["electric_piano"]["sources"]["presence_transcription"]["sections_needed"] == 1
    assert doc3["classes"]["electric_piano"]["present"]


def test_stem_agreement_guard_needs_the_class_stem_active():
    """SC: piano stem active in 3 % of the bars; keys notes from the piano+other pass are not enough alone."""
    notes = window_notes("electric_piano", SC_SPANS, per_s=2.0)
    e = energy(guitar=-6, other=-10)
    e["piano"][: int(0.5 * BAR_S * FPS)] = -10.0  # piano active in < 10 % of the bars (1 of 20 = 5 %)
    doc = est(mix_notes=None, energy=e, presence={"mode": "sampled", "notes": notes, "spans": SC_SPANS})
    src = doc["classes"]["electric_piano"]["sources"]["presence_transcription"]
    assert src["stem_agree"] == 0.05 and src["e"] == 0.0 and not doc["classes"]["electric_piano"]["present"]
    assert doc["presence"]["rejected"] == {"electric_piano": "stem"}
    # synth classes also agree with the piano stem (SW often puts pads there); strings only with the other stem
    pads = window_notes("synth_pad", SC_SPANS, per_s=2.0) + window_notes("violin", SC_SPANS, per_s=2.0, pitch0=70)
    pads.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))
    doc2 = est(mix_notes=None, energy=energy(guitar=-6, piano=-10),
               presence={"mode": "sampled", "notes": pads, "spans": SC_SPANS})
    assert doc2["classes"]["synth_pad"]["present"] and not doc2["classes"]["violin"]["present"]
    assert S.agreement_stems("synth_pad") == ["piano", "other"] and S.agreement_stems("violin") == ["other"]


def test_stem_term_goes_to_the_classes_the_pass_found():
    """青と夏: piano stem active in 85 % of bars; the presence pass found acoustic/electric piano, no organ or
    chromatic percussion. The class-agnostic stem term used to keep those two present (p 0.63 / 0.51)."""
    spans = [[2.0, 12.0], [20.0, 30.0]]
    notes = window_notes("acoustic_piano", spans, per_s=2.0) + window_notes("electric_piano", spans, per_s=2.0,
                                                                            pitch0=70)
    notes.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))
    doc = est(mix_notes=None, energy=energy(guitar=-6, piano=-10, other=-10),
              presence={"mode": "sampled", "notes": notes, "spans": spans})
    for cls in ("acoustic_piano", "electric_piano"):
        assert doc["classes"][cls]["present"]
        assert doc["classes"][cls]["sources"]["stem_energy"]["stems"]["piano"]["attributed"] is True
    for cls in ("organ", "chromatic_percussion"):
        ev = doc["classes"][cls]
        assert not ev["present"] and ev["p"] == 0.0
        assert all(st["attributed"] is False for st in ev["sources"]["stem_energy"]["stems"].values())
    assert doc["masks"]["guitar_pass"] == I.sort_mt3(["acoustic_piano", "electric_piano", *I.GUITAR_CLASSES])


def test_stem_term_stays_when_the_windows_barely_saw_the_stem():
    """A keys part that plays only outside the windows: the pass saw the piano stem in < 4 covered bars, so its
    silence says nothing and the stem term stays attributed (unknown, not absent)."""
    e = energy(guitar=-6, other=-10)
    e["piano"][int(10 * BAR_S * FPS):] = -10.0  # piano active in bars 11..20 only
    doc = est(mix_notes=None, energy=e, presence={"mode": "sampled", "notes": [], "spans": [[0.0, 10.0]]})
    st = doc["classes"]["organ"]["sources"]["stem_energy"]["stems"]
    assert st["piano"]["covered_active_bars"] == 0 and st["piano"]["attributed"] is True
    assert st["piano"]["uncovered_active_ratio"] == 0.5
    assert st["other"]["attributed"] is False  # the other stem was active inside the windows: attributed elsewhere
    assert doc["classes"]["acoustic_piano"]["p"] == pytest.approx(0.6 * 0.5)


def test_bass_pass_decides_the_bass_classes():
    """Without the bass pass both bass classes came from the bass stem alone and were always both present."""
    bass = mix_notes("electric_bass", range(N_BARS))
    doc = est(mix_notes=None, energy=energy(guitar=-6, bass=-6), bass_notes=bass)
    assert doc["classes"]["electric_bass"]["present"]
    ab = doc["classes"]["acoustic_bass"]
    assert not ab["present"] and ab["sources"]["bass_transcription"]["n_notes"] == 0
    assert ab["sources"]["stem_energy"]["stems"]["bass"]["attributed"] is False
    old = est(mix_notes=None, energy=energy(guitar=-6, bass=-6))
    assert old["classes"]["acoustic_bass"]["present"] and old["classes"]["electric_bass"]["present"]


def test_guards_never_drop_a_guitar_class_randomized():
    rng = random.Random(23)
    classes = list(I.PRESENCE_CLASSES) + list(I.GUITAR_CLASSES)
    for _ in range(60):
        spans = SC_SPANS[: rng.randint(1, 3)]
        secs = [rng.choice(["S1", "S2"]) for _s in spans]
        notes = []
        for _k in range(rng.randint(0, 4)):
            notes += window_notes(rng.choice(classes), spans[: rng.randint(1, len(spans))], per_s=2.0)
        notes.sort(key=lambda n: (n["onset_s"], n["pitch"], n["offset_s"], n["instrument"]))
        rel = {st: rng.choice([-100.0, -45.0, -10.0]) for st in ("guitar", "piano", "other", "bass")}
        doc = est(mix_notes=None, energy=energy(**rel), bass_notes=rng.choice([None, []]),
                  presence={"mode": "sampled", "notes": notes, "spans": spans, "window_sections": secs},
                  guitar_mask=rng.choice(I.GUITAR_MASK_POLICIES))
        assert S.guitar_class_omissions(doc) == 0
        from gtab.schema.instrumentation import Instrumentation

        Instrumentation.model_validate(doc)
