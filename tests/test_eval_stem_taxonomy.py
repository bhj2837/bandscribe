"""Stem file name -> sound category (bandscribe.eval.stem_taxonomy): fixed cases + every name on disk."""
from __future__ import annotations

from pathlib import Path

import pytest

from bandscribe.eval import stem_taxonomy as T

CASES = {
    "08_ElecGtr1.wav": "guitar_electric",
    "11_ElecGtr1a_VoxM160.wav": "guitar_electric",   # a Vox amp mic, not a vocal
    "21_MIDIElecGtr1.wav": "guitar_electric",
    "23_ElecGtr1DI.wav": "guitar_electric",
    "27_AcGtr1.wav": "guitar_acoustic",
    "35_AcousticGtr1.wav": "guitar_acoustic",
    "26_GtrFX.wav": "guitar_electric",               # guitar first: an FX part still played on the guitar
    "08_BassDI.wav": "bass",
    "21_BassSynth.wav": "bass",
    "14_BassFX.wav": "bass",
    "01_BassDrum.wav": "drums",
    "01_KickIn.wav": "drums",
    "05_OverheadsStereo.wav": "drums",
    "07_Shaker.wav": "drums",
    "10_Clap1.wav": "drums",
    "16_LeadVox.wav": "vocals",                      # "Lead" + "Vox" is a vocal, never a synth lead
    "17_BackingVox1Wet.wav": "vocals",
    "23_Choir.wav": "vocals",
    "11_Piano.wav": "keys_piano",
    "19_Rhodes.wav": "keys_epiano",
    "21_ElecPiano.wav": "keys_epiano",
    "16_Wurlitzer.wav": "keys_epiano",
    "39_Clavinet.wav": "keys_epiano",
    "08_Hammond.wav": "organ",
    "26_Organ.wav": "organ",
    "09_SynthPad1.wav": "synth_pad",
    "12_SynthLead.wav": "synth_lead",
    "Lead Synth Dbl.L.wav": "synth_lead",
    "13_Stylophone.wav": "synth_lead",
    "11_SynthFuzz.wav": "synth_other",
    "21_Synth1.wav": "synth_other",
    "19_Strings.wav": "strings",
    "25_StringPad.wav": "strings",                   # strings before pads
    "24_Cello.wav": "strings",
    "19_Horns.wav": "brass_winds",
    "18_Flute.wav": "brass_winds",
    "Bassoon.wav": "brass_winds",
    "14_SFX1.wav": "sfx",
    "13_SynthFX1.wav": "sfx",                        # sfx before synth
    "20_RhodesSFX.wav": "sfx",                       # sfx before keys
    "12_SubDrop.wav": "sfx",
    "10_RevCymbal.wav": "sfx",
    "Rising synth pan L AMS reverb_1.wav": "sfx",
    "23_SFXWolf1.wav": "sfx",
    "11_Loop.wav": "other",
    "99_Thingamajig.wav": "other",
}


@pytest.mark.parametrize("name,cat", sorted(CASES.items()))
def test_categorize_cases(name, cat):
    assert T.categorize(name) == cat


def test_categories_partition():
    assert set(T.BASE) | set(T.DISTRACTORS) == set(T.CATEGORIES)
    assert not set(T.BASE) & set(T.DISTRACTORS)
    assert set(T.LABEL_KO) == set(T.CATEGORIES)


def test_tokens_split_camel_case_and_acronyms():
    assert T.tokens("12_ElecGtr2DI.wav") == ["12", "elec", "gtr", "2", "di"]
    assert T.tokens("23_SFXWolf1.wav") == ["23", "sfx", "wolf", "1"]
    assert T.tokens("21_MIDIElecGtr1.wav") == ["21", "midi", "elec", "gtr", "1"]


def test_unknown_names_are_listed_and_fall_back_to_other():
    names = ["01_Kick.wav", "99_Thingamajig.wav", "11_Loop.wav", "13_Click.wav", "05_Zorblax.wav"]
    assert T.unknown_names(names) == ["05_Zorblax.wav", "99_Thingamajig.wav"]  # loops are known, clicks excluded
    assert T.categorize("05_Zorblax.wav") == "other"
    assert T.is_excluded("13_Click.wav") and not T.is_excluded("01_Kick.wav")


def test_every_stem_name_on_disk_is_classified():
    root = Path(__file__).resolve().parents[1] / "data" / "datasets" / "cambridge_mt" / "extracted"
    names = sorted({p.name for p in root.rglob("*") if p.suffix.lower() in (".wav", ".aif", ".aiff", ".flac")}) \
        if root.is_dir() else []
    if not names:
        pytest.skip("Cambridge-MT 멀티트랙이 이 PC 에 없음")
    assert T.unknown_names(names) == []
    cats = {T.categorize(n) for n in names}
    # the distractor set covers these categories (registry.toml, 2026-10-04)
    assert {"synth_pad", "synth_lead", "keys_epiano", "keys_piano", "organ", "strings", "sfx"} <= cats
