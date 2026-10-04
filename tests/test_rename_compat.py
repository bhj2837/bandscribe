"""gtab -> bandscribe rename (2026-10-04, docs/RENAME.md): files written before it must keep loading, and the
key material that kept the old name must never change.

Existing job outputs, caches and registries are never rewritten (stage manifests record their sha256), so every
reader accepts the legacy ``"gtab.<name>/N"`` format ids and the legacy extraction marker name.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from bandscribe import atomic, formats, paths
from bandscribe.amt import cache as amt_cache
from bandscribe.ingest.filekey import content_key


def _write_raw(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


# --------------------------------------------------------------------------------------- format ids


@pytest.mark.parametrize("fmt, want", [
    ("gtab.notes/1", "bandscribe.notes/1"),
    ("gtab.dataset_registry/1", "bandscribe.dataset_registry/1"),
    ("bandscribe.notes/1", "bandscribe.notes/1"),
    ("gp5", "gp5"),
    ("xgtab.notes/1", "xgtab.notes/1"),
    (None, None),
    (3, 3),
])
def test_normalize(fmt, want):
    assert formats.normalize(fmt) == want


def test_normalize_doc_copies_only_when_needed():
    new = {"format": "bandscribe.grid/1", "a": 1}
    assert formats.normalize_doc(new) is new
    old = {"format": "gtab.grid/1", "a": 1}
    out = formats.normalize_doc(old)
    assert out == {"format": "bandscribe.grid/1", "a": 1} and old["format"] == "gtab.grid/1"
    assert formats.normalize_doc([1]) == [1]


def test_read_json_normalises_top_level_only(tmp_path):
    p = tmp_path / "d.json"
    _write_raw(p, {"format": "gtab.notes/1", "backend": {"format": "gtab.keep/1"}, "source": {"format": "gp5"}})
    d = atomic.read_json(p)
    assert d["format"] == "bandscribe.notes/1"
    assert d["backend"]["format"] == "gtab.keep/1" and d["source"]["format"] == "gp5"
    assert json.loads(p.read_text(encoding="utf-8"))["format"] == "gtab.notes/1"  # the file is not rewritten


def test_pydantic_models_accept_legacy_ids():
    from bandscribe.eval.refscore import RefScore
    from bandscribe.schema.notes import NoteSet
    from bandscribe.schema.sections import Sections

    raw = json.dumps({"format": "gtab.notes/1", "view": "guitar_mono", "backend": {"name": "muscriptor"}, "notes": []})
    assert NoteSet.model_validate_json(raw).format == "bandscribe.notes/1"
    sec = Sections.model_validate({"format": "gtab.sections/1", "method": "m", "sections": [], "params": {}})
    assert sec.format == "bandscribe.sections/1"
    with pytest.raises(ValueError):
        Sections.model_validate({"format": "other.sections/1", "method": "m", "sections": [], "params": {}})
    rs = RefScore.model_validate_json(json.dumps({"format": "gtab.refscore/1", "source": {"format": "gp5"},
                                                  "tracks": [], "masterbars": [], "notes": []}))
    assert rs.format == "bandscribe.refscore/1" and rs.source == {"format": "gp5"}


def test_tempomap_from_legacy_dict():
    from bandscribe.schema.grid import TempoMap

    tm = TempoMap.from_dict({"format": "gtab.tempomap/1", "tpb": 48, "beats_per_bar": {"1": 4},
                             "beats": [{"t_s": 0.0, "bar": 1, "beat": 1}, {"t_s": 0.5, "bar": 1, "beat": 2}]})
    assert tm is not None


def test_score_legacy_version_field():
    from bandscribe.schema.score import Score

    doc = {"song": {"song_key": "f-0123456789abcdef", "duration_s": 1.0, "source_kind": "file", "source_ref": "x.wav"},
           "gtab_version": "0.0.9"}
    assert Score.model_validate(doc).bandscribe_version == "0.0.9"


def test_amt_cache_reads_legacy_entry(tmp_path):
    c = amt_cache.AmtCache(tmp_path)
    key = "ab" * 32
    _write_raw(c.path(key), {"format": "gtab.notes/1", "notes": []})
    assert c.get(key)["format"] == "bandscribe.notes/1"


def test_gpu_run_legacy(tmp_path):
    from bandscribe import vram

    _write_raw(tmp_path / vram.GPU_RUN_FILE, {"format": "gtab.gpu_run/1", "degraded": False})
    assert vram.read_gpu_run(tmp_path)["format"] == vram.GPU_RUN_FORMAT


def test_legacy_export_json_is_up_to_date(tmp_path):
    from bandscribe.pipeline import publish

    export = tmp_path / "export"
    entries: list[dict] = []
    doc = publish._doc(entries)
    _write_raw(export / publish.EXPORT_JSON, {**doc, "format": "gtab.export/1"})
    assert publish._up_to_date(export, doc, entries)


def test_model_lock_and_sha_cache_legacy(tmp_path, monkeypatch):
    from bandscribe import models

    monkeypatch.setattr(paths, "MODELS", tmp_path)
    monkeypatch.setattr(paths, "MODELS_LOCK", tmp_path / "models.lock.json")
    _write_raw(tmp_path / "models.lock.json", {"format": "gtab.models/1", "models": {}})
    assert models._read_lock(tmp_path / "models.lock.json")["models"] == {}
    _write_raw(tmp_path / "sha.json", {"format": "gtab.sha_cache/1", "entries": {"k": "v"}})
    assert models._read_cache(tmp_path / "sha.json") == {"k": "v"}  # not discarded -> no 1.2 GB rehash


def test_registry_and_model_sources_legacy(tmp_path):
    import tomllib

    from bandscribe import model_fetch
    from bandscribe.datasets import registry

    pkg = Path(registry.__file__).resolve().parent
    data = tomllib.loads((pkg / "registry.toml").read_text(encoding="utf-8"))
    data["format"] = "gtab.dataset_registry/1"
    assert registry.parse_registry(data)
    src = Path(model_fetch.SOURCES_PATH).read_text(encoding="utf-8").replace(
        'format = "bandscribe.model_sources/1"', 'format = "gtab.model_sources/1"')
    assert 'format = "gtab.model_sources/1"' in src
    (tmp_path / "model_sources.toml").write_text(src, encoding="utf-8")
    assert model_fetch.load_sources(tmp_path / "model_sources.toml")


def test_cmt_lines_and_tiera_songs_legacy(tmp_path):
    from bandscribe.datasets import store, tiera
    from bandscribe.datasets.loaders import cambridge_mt

    text = store.lines_skeleton_text(["Gtr1.wav", "Bass.wav"], song="s")
    assert "bandscribe.cmt_lines/1" in text
    (tmp_path / "lines.yaml").write_text(text.replace("bandscribe.cmt_lines/1", "gtab.cmt_lines/1"), encoding="utf-8")
    assert cambridge_mt.read_lines(tmp_path / "lines.yaml")["format"] == "bandscribe.cmt_lines/1"
    (tmp_path / "songs.yaml").write_text("format: gtab.tiera_songs/1\nsongs: []\n", encoding="utf-8")
    assert tiera.load_songs(tmp_path / "songs.yaml") == {}


def test_legacy_extract_marker_is_honoured(tmp_path):
    from bandscribe.datasets import fetch, store

    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("top/a.txt", b"a")
    out = fetch.extract_archive(z, tmp_path / "extracted" / "d", sha256="s1")
    assert (out / store.EXTRACT_MARKER).is_file()
    # simulate an extraction made before the rename
    (out / store.EXTRACT_MARKER).rename(out / ".gtab_extracted.json")
    (out / "top" / "a.txt").write_bytes(b"user edit")
    assert store.find_marker(out).name == ".gtab_extracted.json"
    fetch.extract_archive(z, out, sha256="s1")  # same archive -> not redone
    assert (out / "top" / "a.txt").read_bytes() == b"user edit"
    assert store.count_files(out) == 1
    fetch.extract_archive(z, out, sha256="s2")  # a new archive re-extracts and writes the new marker only
    assert (out / store.EXTRACT_MARKER).is_file() and not (out / ".gtab_extracted.json").exists()


# --------------------------------------------------------------------------------------- frozen key material


def test_content_key_domain_is_frozen(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"bandscribe" * 1000)
    # computed with the pre-rename code; a different value means every jobs/index.json entry became a miss
    assert content_key(p) == "a6a388e5497da60710cef1ba9122692372ce009fc149a423c11e6f5f4368b283"


def test_amt_cache_key_is_frozen():
    assert amt_cache.KEY_FORMAT == "gtab.amt_cache_key/1"
    key = amt_cache.amt_key("muscriptor", "0.3.0", "ab" * 32, "cd" * 32,
                            ["distorted_electric_guitar", "acoustic_piano"], {"beam_size": 1}, "2", "medium-lean")
    # computed with the pre-rename code; a different value orphans every cached transcription
    assert key == "6b313936290f95229c834f11ccb3c9a704350feb46c7aa1549b63f5fabbdd95a"

