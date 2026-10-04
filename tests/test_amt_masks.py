"""MT3 masks, hint parsing, sort order and E3 threshold edges (gtab.amt.instruments)."""
from __future__ import annotations

import random

import pytest

from gtab.amt import instruments as I
from gtab.config import ConfigError


def _inst(present: set[str]) -> dict:
    return {"classes": {c: {"present": c in present, "p": 1.0 if c in present else 0.0} for c in I.MT3_GROUPS}}


def test_sort_mt3_order_dedup_and_unknown():
    assert I.sort_mt3(["synth_pad", "acoustic_piano", "drums", "acoustic_piano"]) == ["acoustic_piano", "synth_pad",
                                                                                     "drums"]
    with pytest.raises(ValueError):
        I.sort_mt3(["kazoo"])


@pytest.mark.parametrize("policy", I.GUITAR_MASK_POLICIES)
def test_guitar_classes_never_dropped_randomized(policy):
    rng = random.Random(20260930)
    names = list(I.MT3_GROUPS)
    for _ in range(300):
        present = {c for c in names if rng.random() < 0.4}
        mask = I.mask_guitar_view(_inst(present), policy)
        assert I.guitar_omissions(mask) == 0
        if mask is not None:
            assert mask == I.sort_mt3(mask)
            assert set(I.GUITAR_CLASSES) <= set(mask)


def test_guitar_plus_present_adds_only_present_keys_synth():
    """Review 2026-10-04: the default policy is back to the M2 keys/synth set; strings only in the E3b variant."""
    present = {"electric_piano", "synth_pad", "violin", "contrabass", "orchestral_harp", "voice", "trumpet", "flutes"}
    assert I.mask_guitar_view(_inst(present), "guitar+present") == [
        "electric_piano", "acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar", "synth_pad"]
    assert I.mask_guitar_view(_inst(set()), "guitar+present") == I.mask_guitar_only()
    assert I.mask_guitar_view(_inst({"organ"}), "guitar_only") == I.mask_guitar_only()
    assert I.mask_guitar_view(_inst({"organ"}), "all") is None


def test_strings_policy_adds_bowed_strings_but_never_harp_or_contrabass():
    every_string = set(I.FAMILIES["strings"]) | {"synth_strings"}
    mask = I.mask_guitar_view(_inst(every_string), "guitar+present+strings")
    assert {"violin", "viola", "cello", "string_ensemble", "synth_strings"} <= set(mask)
    assert "orchestral_harp" not in mask and "contrabass" not in mask  # guitar-like timbre / register
    assert set(I.GUITAR_CLASSES) <= set(mask) and mask == I.sort_mt3(mask)
    assert "violin" not in I.mask_guitar_view(_inst(every_string), "guitar+present")


def test_guitar_mask_accepts_pydantic_like_objects():
    class Ev:
        def __init__(self, present):
            self.present = present

    class Inst:
        classes = {"organ": Ev(True), "synth_lead": Ev(False)}

    assert "organ" in I.mask_guitar_view(Inst(), "guitar+present")


def test_unknown_policy_is_config_error():
    with pytest.raises(ConfigError):
        I.mask_guitar_view(_inst(set()), "guitar")
    with pytest.raises(ConfigError):
        I.mask_piano_other("piano")


def test_piano_other_modes():
    keys = I.mask_piano_other("keys")
    both = I.mask_piano_other("keys+guitar")
    assert set(keys) == set(I.PRESENCE_CLASSES)
    assert set(I.PRESENCE_CLASSES) == set(I.KEYS_SYNTH_CLASSES) | set(I.FAMILIES["strings"])
    assert set(both) == set(I.PRESENCE_CLASSES) | set(I.GUITAR_CLASSES)
    assert keys == I.sort_mt3(keys)
    assert both == I.sort_mt3(both)
    assert I.mask_bass() == ["acoustic_bass", "electric_bass"]


@pytest.mark.parametrize("hint,expected", [
    ("auto", None), ("", None), (None, None), ("AUTO", None),
    ("guitar,keys", ["guitar", "keys"]), ("keys, Guitar ; bass", ["guitar", "bass", "keys"]),
    (["synth", "guitar"], ["guitar", "synth"]), ("keys,keys", ["keys"]),
])
def test_parse_hint(hint, expected):
    assert I.parse_hint(hint) == expected


def test_parse_hint_unknown_family_is_korean_config_error():
    with pytest.raises(ConfigError) as ei:
        I.parse_hint("guitar,piano")
    assert "알 수 없는" in str(ei.value) and "piano" in str(ei.value)


def test_pass1_masks():
    m = I.pass1_masks("keys+guitar")
    assert m["mix_mono"] is None
    assert m["bass_mono"] == I.mask_bass()
    assert I.guitar_omissions(m["piano_other_mono"]) == 0
    assert m["piano_other_presence"] == m["piano_other_mono"]  # the sampled pass uses the full pass's mask


def test_family_tables_consistent():
    covered = [c for fam in I.FAMILIES.values() for c in fam]
    assert sorted(covered) == sorted(I.MT3_GROUPS)  # every class in exactly one family
    assert len(covered) == len(set(covered))
    assert I.family_of("organ") == "keys"
