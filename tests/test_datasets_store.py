"""datasets.lock.json handling, status/verify, import-local and the Cambridge-MT filename guesses."""

from __future__ import annotations

import os
import threading

import pytest

from fixtures.datasets_fakes import write_wav
from gtab import atomic
from gtab.datasets import store


def _rec(sha="0" * 64, n=3):
    return {"url": "https://x", "bytes": n, "sha256": sha, "md5": None, "fetched_utc": "2026-09-30T00:00:00Z",
            "verified_against": None}


def test_lock_roundtrip_and_format(tmp_path):
    assert store.read_lock(tmp_path) == {"format": "gtab.datasets/1", "datasets": {}}
    store.record_files("guitarset", {"raw/b.zip": _rec(), "raw/a.zip": _rec()}, status="fetched", root=tmp_path)
    lock = store.read_lock(tmp_path)
    ds = lock["datasets"]["guitarset"]
    assert list(ds["files"]) == ["raw/a.zip", "raw/b.zip"] and ds["status"] == "fetched"
    assert ds["extracted"] == "extracted"


def test_corrupt_lock_is_an_error_not_a_reset(tmp_path):
    (tmp_path / store.LOCK_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(store.StoreError):
        store.read_lock(tmp_path)
    with pytest.raises(store.StoreError):
        store.record_files("x", {"raw/a": _rec()}, root=tmp_path)
    assert (tmp_path / store.LOCK_NAME).read_text(encoding="utf-8") == "{not json"


def test_concurrent_writers_keep_every_record(tmp_path):
    def work(i):
        for j in range(10):
            store.record_files(f"d{i % 3}", {f"raw/{i}_{j}": _rec()}, root=tmp_path)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lock = store.read_lock(tmp_path)
    assert sum(len(d["files"]) for d in lock["datasets"].values()) == 60


def test_status_partial_and_verify(tmp_path):
    base = tmp_path / "idmt_bass_st" / "raw"
    base.mkdir(parents=True)
    f = base / "IDMT-SMT-BASS-SINGLE-TRACKS.zip"
    f.write_bytes(b"abc")
    sha, _ = store.hash_file(f)
    assert store.status("idmt_bass_st", tmp_path) == "absent"
    store.record_files("idmt_bass_st", {"raw/IDMT-SMT-BASS-SINGLE-TRACKS.zip": _rec(sha)}, status="fetched",
                       root=tmp_path)
    assert store.status("idmt_bass_st", tmp_path) == "fetched" and store.is_available("idmt_bass_st", tmp_path)
    r = store.verify("idmt_bass_st", tmp_path)  # the archive is recorded but was never unpacked
    assert not r["ok"] and r["mismatch"] == [] and r["missing"] == [] and "없습니다" in r["extracted"][0]
    ext = tmp_path / "idmt_bass_st" / "extracted" / "IDMT-SMT-BASS-SINGLE-TRACKS"
    ext.mkdir(parents=True)
    (ext / "a.xml").write_text("x", encoding="utf-8")
    atomic.write_json(ext / store.EXTRACT_MARKER, {"archive": f.name, "sha256": sha, "files": 1, "members": ["a.xml"]})
    r = store.verify("idmt_bass_st", tmp_path)
    assert r["ok"] and r["status"] == "verified" and r["extracted"] == []
    f.write_bytes(b"abd")  # modified on disk
    r = store.verify("idmt_bass_st", tmp_path)
    assert not r["ok"] and r["mismatch"] == ["raw/IDMT-SMT-BASS-SINGLE-TRACKS.zip"] and r["status"] == "fetched"
    f.unlink()
    assert store.verify("idmt_bass_st", tmp_path)["missing"] == ["raw/IDMT-SMT-BASS-SINGLE-TRACKS.zip"]
    # guitarset needs two required archives -> one of them is "partial"
    store.record_files("guitarset", {"raw/annotation.zip": _rec()}, status="fetched", root=tmp_path)
    assert store.status("guitarset", tmp_path) == "partial" and not store.is_available("guitarset", tmp_path)


def test_import_local_links_records_and_keeps_labels(tmp_path):
    pytest.importorskip("yaml")
    src = tmp_path / "download" / "Band_Song"
    for n in ("01_Kick.wav", "02_ElecGtr1.wav", "03_ElecGtr2.wav", "04_Bass.wav"):
        write_wav(src / n, 0.1)
    (src / ".DS_Store").write_bytes(b"x")
    root = tmp_path / "datasets"
    res = store.import_local("cambridge_mt", src, song="Band - Song!", root=root)
    dest = root / "cambridge_mt" / "local" / "band_-_song"
    assert res["dest"] == dest and res["files"] == 4 and res["linked"] + res["copied"] == 4
    assert not (dest / ".DS_Store").exists()
    assert (dest / "lines.yaml").is_file()
    lock = store.read_lock(root)["datasets"]["cambridge_mt"]
    assert lock["status"] == "manual" and "local/band_-_song/02_ElecGtr1.wav" in lock["files"]
    assert lock["files"]["local/band_-_song/02_ElecGtr1.wav"]["local"] == str(src / "02_ElecGtr1.wav")
    assert store.status("cambridge_mt", root) == "manual"
    if res["linked"]:
        assert os.path.samefile(src / "01_Kick.wav", dest / "01_Kick.wav")
    # re-import keeps the user's lines.yaml
    (dest / "lines.yaml").write_text("user labels", encoding="utf-8")
    store.import_local("cambridge_mt", src, song="Band - Song!", root=root, link=False)
    assert (dest / "lines.yaml").read_text(encoding="utf-8") == "user labels"
    assert store.verify("cambridge_mt", root)["ok"]


def test_import_local_errors(tmp_path):
    with pytest.raises(store.StoreError, match="--song"):
        store.import_local("cambridge_mt", tmp_path, root=tmp_path / "d")
    with pytest.raises(store.StoreError, match="폴더"):
        store.import_local("egdb", tmp_path / "nope", root=tmp_path / "d")
    (tmp_path / "empty").mkdir()
    with pytest.raises(store.StoreError, match="파일이 없습니다"):
        store.import_local("egdb", tmp_path / "empty", root=tmp_path / "d")


@pytest.mark.parametrize("name, kind, family, number, mic", [
    ("08_ElecGtr1a.wav", "guitar", "guitar", 1, False),
    ("21_ElecGtr3DI.wav", "di", "guitar", 3, True),
    ("11_ElecGtr1a_VoxM160.wav", "guitar", "guitar", 1, True),   # a Vox amp, not a voice
    ("09_ElecGtrMic1.wav", "guitar", "guitar", None, True),
    ("16_AcousticGtr.wav", "guitar", "guitar", None, False),
    ("01_BassDrum.wav", "drums", "drums", None, False),
    ("09_BassAmpMic1.wav", "bass", "bass", None, True),
    ("07_BassDI.wav", "di", "bass", None, True),
    ("14_LeadVoxDT.wav", "vocals", "vocals", None, False),
    ("08_Hammond.wav", "keys", "keys", None, False),
    ("13_Rhodes.wav", "keys", "keys", None, False),
    ("13_Stylophone.wav", "other", "synth", None, False),
    ("05_Overheads.wav", "drums", "drums", None, False),
    ("22_Bassoon.wav", "other", "winds", None, False),
    ("14_SFX1.wav", "other", None, None, False),
])
def test_guess_part(name, kind, family, number, mic):
    g = store.guess_part(name)
    assert (g["kind"], g["family"], g["number"], g["mic"]) == (kind, family, number, mic)


def test_guess_acoustic_guitar_line():
    text = store.lines_skeleton_text(["01_AcousticGtr.wav", "02_ElecGtr1.wav"], song="s")
    assert 'line: "AG1"' in text and 'line: "G1"' in text


@pytest.mark.parametrize("s, out", [("job:amt_gtr", "job-amt_gtr"), ("Woodfire - Haunted House", "woodfire_-_haunted_house"),
                                    ("guitarset:00_BN1", "guitarset-00_bn1"), ("청춘 컴플렉스 SC", "sc")])
def test_slug(s, out):
    assert store.slug(s) == out


def test_slug_empty():
    with pytest.raises(ValueError):
        store.slug("괴수")
