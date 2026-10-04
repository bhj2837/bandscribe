"""Dataset registry (bandscribe/datasets/registry.toml) and model sources (bandscribe/model_sources.toml): schema checks."""

from __future__ import annotations

import importlib.util
import tomllib

import pytest

from bandscribe import model_fetch
from bandscribe.datasets import registry


def test_registry_loads_and_every_entry_is_complete():
    reg = registry.load_registry()
    assert {"guitarset", "idmt_bass_st", "egdb", "filobass", "cambridge_mt", "medleydb"} <= set(reg)
    for name, spec in reg.items():
        assert spec.license.strip(), name
        assert spec.tier in ("B", "C"), name
        assert spec.access in registry.ACCESS_KINDS, name
        assert spec.page.startswith("https://"), name
        assert spec.loader and importlib.util.find_spec(f"bandscribe.datasets.loaders.{spec.loader}"), name
        assert spec.families, name
        if spec.access != "manual":
            assert spec.approx_bytes > 0, name
            assert spec.required_parts(), name
        assert spec.manual_ko, f"{name}: Korean manual steps are shown when a download is refused"
        for f in spec.files:
            assert f.name.isascii() and "/" not in f.name
            assert f.url.startswith("https://")
            if spec.access == "zenodo":
                assert f.bytes > 0 and len(f.md5) == 32, (name, f.name)


def test_zenodo_facts_match_spec():
    reg = registry.load_registry()
    gs = {f.name: f for f in reg["guitarset"].files}
    assert gs["annotation.zip"].bytes == 39132574 and gs["annotation.zip"].md5 == "b39b78e63d3446f2e54ddb7a54df9b10"
    assert gs["audio_mono-mic.zip"].bytes == 656927981 and gs["audio_mono-mic.zip"].required
    assert not gs["audio_hex-pickup_original.zip"].required
    assert reg["filobass"].files[0].md5 == "ea1f52ffd492bf3654d720529f15bd9b"
    assert reg["idmt_bass_st"].license == "CC BY-NC-ND 4.0"


def test_cambridge_songs_have_ids_and_plain_host():
    spec = registry.get("cambridge_mt")
    required = [f for f in spec.files if f.required]
    assert 4 <= len(required) <= 6
    # the distractor set (2026-10-04): optional parts, one per song, under ~3 GB in all
    optional = [f for f in spec.files if not f.required]
    assert len(optional) == 7 and all(f.part == f.song for f in optional)
    assert sum(f.bytes for f in optional) < 3_000_000_000
    for f in spec.files:
        assert f.song and f.song.isascii() and f.song == f.song.lower()
        # only the MTK archive host answered plain requests; the Cloudflare-protected hosts must not appear
        assert f.url.startswith("https://mtkdata.cambridgemusictechnology.co.uk/")
        assert f.bytes > 0
    assert sum(f.bytes for f in required) == spec.approx_bytes


def test_egdb_folders_and_split_notes():
    spec = registry.get("egdb")
    names = {d.name for d in spec.folders}
    assert {"audio_label", "audio_DI", "audio_JCjazz", "audio_Plexi", "audio_Marshall", "audio_Ftwin", "audio_Mesa"} <= names
    assert "216" in spec.notes


def test_unknown_name_suggests(tmp_path):
    with pytest.raises(registry.RegistryError, match="guitarset"):
        registry.get("guitarsett")


def _write(tmp_path, text):
    p = tmp_path / "reg.toml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.mark.parametrize("body, msg", [
    ('[x]\ntitle="t"\ntier="A"\naccess="manual"\nlicense="l"\npage="https://a"\napprox_bytes=0\nloader="guitarset"\nfamilies=[]\n', "tier"),
    ('[x]\ntitle="t"\ntier="C"\naccess="ftp"\nlicense="l"\npage="https://a"\napprox_bytes=0\nloader="guitarset"\nfamilies=[]\n', "access"),
    ('[x]\ntitle="t"\ntier="C"\naccess="manual"\nlicense=" "\npage="https://a"\napprox_bytes=0\nloader="guitarset"\nfamilies=[]\n', "license"),
    ('[x]\ntitle="t"\ntier="C"\naccess="manual"\nlicense="l"\npage="https://a"\nloader="guitarset"\nfamilies=[]\n', "approx_bytes"),
    ('[x]\ntitle="t"\ntier="C"\naccess="zenodo"\nlicense="l"\npage="https://a"\napprox_bytes=1\nloader="guitarset"\nfamilies=[]\n', "files"),
    ('[x]\ntitle="t"\ntier="C"\naccess="manual"\nlicense="l"\npage="https://a"\napprox_bytes=0\nloader="guitarset"\nfamilies=["kazoo"]\n', "families"),
    ('[x]\ntitle="t"\ntier="C"\naccess="direct"\nlicense="l"\npage="https://a"\napprox_bytes=1\nloader="guitarset"\nfamilies=[]\n'
     '[[x.files]]\nname="a.zip"\nurl="ftp://a"\n', "https"),
    ('[x]\ntitle="t"\ntier="C"\naccess="direct"\nlicense="l"\npage="https://a"\napprox_bytes=1\nloader="guitarset"\nfamilies=[]\n'
     '[[x.files]]\nname="../a.zip"\nurl="https://a"\n', "ASCII"),
    ('[x]\ntitle="t"\ntier="C"\naccess="manual"\nlicense="l"\npage="https://a"\napprox_bytes=0\nloader="guitarset"\nfamilies=[]\ntypo=1\n', "typo"),
])
def test_malformed_registry_rejected(tmp_path, body, msg):
    p = _write(tmp_path, 'format = "bandscribe.dataset_registry/1"\n' + body)
    with pytest.raises(registry.RegistryError, match=msg):
        registry.load_registry(p)


def test_family_keys_match_schema_when_available():
    try:
        from bandscribe.schema.instrumentation import FAMILIES
    except ImportError:
        pytest.skip("P0 schema (bandscribe.schema.instrumentation) not present yet")
    assert tuple(FAMILIES) == registry.FAMILY_KEYS


def test_model_sources_toml():
    src = model_fetch.load_sources()
    bt = src["beat_this.final0"]
    assert bt.url == "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
    assert bt.dest == "data/models/beat_this/final0.ckpt"
    assert "MIT" in bt.license
    with open(model_fetch.SOURCES_PATH, "rb") as f:
        assert tomllib.load(f)["format"] == model_fetch.SOURCES_FORMAT
