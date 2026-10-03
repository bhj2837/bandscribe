"""gtab.datasets.fetch against a local http.server: resume, checksums, refusals, extraction, lock records."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from fixtures.datasets_fakes import HttpFixture
from gtab.datasets import fetch as F
from gtab.datasets import registry, store


@pytest.fixture
def http():
    h = HttpFixture()
    yield h
    h.close()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


BLOB = bytes(range(256)) * 4096  # 1 MiB


# ---------------------------------------------------------------------------------------- download_file


def test_download_verifies_and_renames(http, tmp_path):
    http.files["/a.bin"] = BLOB
    res = F.download_file(http.url("/a.bin"), tmp_path / "a.bin", expected_bytes=len(BLOB), md5=_md5(BLOB),
                          progress=False)
    assert (tmp_path / "a.bin").read_bytes() == BLOB
    assert not (tmp_path / "a.bin.part").exists()
    assert res["sha256"] == hashlib.sha256(BLOB).hexdigest() and res["bytes"] == len(BLOB)
    assert res["verified_against"] == "md5"


def test_resume_after_cut(http, tmp_path):
    http.files["/a.bin"] = BLOB
    http.mode["/a.bin"] = "cut:300000"
    with pytest.raises(F.FetchError, match="이어 받"):
        F.download_file(http.url("/a.bin"), tmp_path / "a.bin", expected_bytes=len(BLOB), md5=_md5(BLOB),
                        progress=False)
    part = tmp_path / "a.bin.part"
    assert part.exists() and 0 < part.stat().st_size < len(BLOB)
    assert not (tmp_path / "a.bin").exists()
    res = F.download_file(http.url("/a.bin"), tmp_path / "a.bin", expected_bytes=len(BLOB), md5=_md5(BLOB),
                          progress=False)
    assert (tmp_path / "a.bin").read_bytes() == BLOB
    assert res["sha256"] == hashlib.sha256(BLOB).hexdigest()  # hash covers the resumed prefix too
    ranges = [r for (_, p, r) in http.requests if p == "/a.bin"]
    assert ranges[0] is None
    assert ranges[-1] is not None and ranges[-1].startswith("bytes=") and ranges[-1] != "bytes=0-"


def test_complete_part_file_is_finished_via_416(http, tmp_path):
    http.files["/a.bin"] = BLOB
    (tmp_path / "a.bin.part").write_bytes(BLOB)
    res = F.download_file(http.url("/a.bin"), tmp_path / "a.bin", expected_bytes=len(BLOB), md5=_md5(BLOB),
                          progress=False)
    assert res["bytes"] == len(BLOB) and (tmp_path / "a.bin").read_bytes() == BLOB


def test_server_ignoring_range_restarts(http, tmp_path):
    http.files["/a.bin"] = BLOB
    http.mode["/a.bin"] = "norange"
    (tmp_path / "a.bin.part").write_bytes(b"garbage" * 10)
    F.download_file(http.url("/a.bin"), tmp_path / "a.bin", md5=_md5(BLOB), progress=False)
    assert (tmp_path / "a.bin").read_bytes() == BLOB


def test_md5_mismatch_leaves_nothing(http, tmp_path):
    http.files["/a.bin"] = BLOB
    with pytest.raises(F.ChecksumMismatch, match="MD5"):
        F.download_file(http.url("/a.bin"), tmp_path / "a.bin", md5="0" * 32, progress=False)
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()


def test_server_sha1_is_checked(http, tmp_path):
    http.files["/m.ckpt"] = BLOB
    http.mode["/m.ckpt"] = "sha1"
    res = F.download_file(http.url("/m.ckpt"), tmp_path / "m.ckpt", progress=False)
    assert res["verified_against"] == "sha1"


@pytest.mark.parametrize("mode", ["403", "cf", "html"])
def test_refusals_become_manual_download(http, tmp_path, mode):
    http.files["/x.zip"] = BLOB
    http.mode["/x.zip"] = mode
    with pytest.raises(F.ManualDownloadRequired) as ei:
        F.download_file(http.url("/x.zip"), tmp_path / "x.zip", progress=False)
    assert "브라우저" in str(ei.value)
    assert not (tmp_path / "x.zip").exists()
    # exactly one plain request: no retry dance against bot protection
    assert len([r for r in http.requests if r[1] == "/x.zip"]) == 1


def test_404_is_fetch_error(http, tmp_path):
    with pytest.raises(F.FetchError, match="404"):
        F.download_file(http.url("/nope"), tmp_path / "nope", progress=False)


def test_progress_callback(http, tmp_path):
    http.files["/a.bin"] = BLOB
    seen = []
    F.download_file(http.url("/a.bin"), tmp_path / "a.bin", progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == len(BLOB)


# ------------------------------------------------------------------------------------------ extraction


def test_extract_atomic_and_idempotent(tmp_path):
    z = tmp_path / "d.zip"
    z.write_bytes(_zip_bytes({"top/a.txt": b"a", "top/sub/b.txt": b"b", "__MACOSX/._a.txt": b"x"}))
    out = F.extract_archive(z, tmp_path / "extracted" / "d", sha256="s1")
    assert (out / "top" / "sub" / "b.txt").read_bytes() == b"b"
    assert not (out / "__MACOSX").exists()
    assert (out / F.EXTRACT_MARKER).is_file()
    (out / "top" / "a.txt").write_bytes(b"changed")
    F.extract_archive(z, out, sha256="s1")  # same archive -> untouched
    assert (out / "top" / "a.txt").read_bytes() == b"changed"
    F.extract_archive(z, out, sha256="s2")  # new archive content -> re-extracted
    assert (out / "top" / "a.txt").read_bytes() == b"a"
    assert not [p for p in (tmp_path / "extracted").iterdir() if p.name.startswith(".tmp-")]


def test_extract_redone_after_deletion_keeps_added_files(tmp_path):
    z = tmp_path / "song.zip"
    z.write_bytes(_zip_bytes({"Song/Gtr1.wav": b"g", "Song/Bass.wav": b"b"}))
    out = F.extract_archive(z, tmp_path / "extracted" / "song", sha256="s1")
    (out / "Song" / "lines.yaml").write_text("reviewed: true", encoding="utf-8")  # the user's labels
    F.extract_archive(z, out, sha256="s1")  # an added file does not trigger a re-extraction
    (out / "Song" / "Bass.wav").unlink()
    F.extract_archive(z, out, sha256="s1")  # a deleted archive file does
    assert (out / "Song" / "Bass.wav").read_bytes() == b"b"
    assert (out / "Song" / "lines.yaml").read_text(encoding="utf-8") == "reviewed: true"
    F.extract_archive(z, out, sha256="s2")  # a new archive keeps user files too
    assert (out / "Song" / "lines.yaml").is_file()
    assert not [p for p in (tmp_path / "extracted").iterdir() if p.name.startswith(".tmp-")]


def test_verify_reports_extraction_problems(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch)
    F.fetch("toyset", root=root, yes=True, progress=False)
    assert store.verify("toyset", root)["ok"]
    ext = root / "toyset" / "extracted" / "annotation"
    (ext / "annotation" / "x.jams").unlink()
    r = store.verify("toyset", root)
    assert not r["ok"] and r["missing"] == [] and r["mismatch"] == [] and "모자랍니다" in r["extracted"][0]
    F.fetch("toyset", root=root, yes=True, progress=False)  # re-run re-extracts without downloading
    assert store.verify("toyset", root)["ok"] and (ext / "annotation" / "x.jams").is_file()
    store.remove_tree(ext)
    r = store.verify("toyset", root)
    assert not r["ok"] and "없습니다" in r["extracted"][0]


def test_extract_rejects_unsafe_paths(tmp_path):
    z = tmp_path / "evil.zip"
    z.write_bytes(_zip_bytes({"../evil.txt": b"x"}))
    with pytest.raises(F.FetchError, match="안전"):
        F.extract_archive(z, tmp_path / "extracted" / "evil")
    assert not (tmp_path / "extracted" / "evil").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_corrupt_archive_leaves_no_dir(tmp_path):
    z = tmp_path / "bad.zip"
    z.write_bytes(b"PK\x03\x04 not really a zip")
    with pytest.raises(F.FetchError):
        F.extract_archive(z, tmp_path / "extracted" / "bad")
    assert not (tmp_path / "extracted" / "bad").exists()


# ------------------------------------------------------------------------------------ dataset fetches


def _registry(tmp_path, http, monkeypatch, *, md5_ok=True) -> Path:
    zdata = _zip_bytes({"annotation/x.jams": b"{}"})
    http.files["/annotation.zip"] = zdata
    http.files["/opt.zip"] = _zip_bytes({"o.txt": b"o"})
    md5 = _md5(zdata) if md5_ok else "f" * 32
    reg = tmp_path / "registry.toml"
    reg.write_text(f'''format = "gtab.dataset_registry/1"
[toyset]
title = "Toy"
tier = "C"
access = "zenodo"
license = "CC BY 4.0"
page = "https://example.org/toy"
approx_bytes = {len(zdata)}
loader = "guitarset"
families = ["guitar"]
manual_ko = "브라우저로 받아 import-local 하세요"

[[toyset.files]]
name = "annotation.zip"
url = "{http.url('/annotation.zip')}"
bytes = {len(zdata)}
md5 = "{md5}"
required = true
extract = true

[[toyset.files]]
name = "opt.zip"
url = "{http.url('/opt.zip')}"
bytes = 0
md5 = ""
required = false
extract = true

[closed]
title = "Closed"
tier = "B"
access = "manual"
license = "restricted"
page = "https://example.org/closed"
approx_bytes = 0
loader = "medleydb"
families = ["guitar"]
manual_ko = "접근 요청 후 import-local"
''', encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg)
    return tmp_path / "datasets"


def test_fetch_dataset_records_lock_and_extracts(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch)
    prompts = []
    res = F.fetch("toyset", root=root, confirm=lambda t: prompts.append(t) or True, progress=False)
    assert "CC BY 4.0" in prompts[0] and "https://example.org/toy" in prompts[0] and str(root / "toyset") in prompts[0]
    assert res["downloaded"] == 1 and res["status"] == "fetched"
    assert (root / "toyset" / "extracted" / "annotation" / "annotation" / "x.jams").is_file()
    lock = store.read_lock(root)
    rec = lock["datasets"]["toyset"]["files"]["raw/annotation.zip"]
    assert rec["verified_against"] == "md5" and len(rec["sha256"]) == 64 and rec["fetched_utc"]
    assert not (root / "toyset" / "raw" / "opt.zip").exists()  # optional part not fetched by default
    # second run: nothing downloaded again
    n_req = len(http.requests)
    res2 = F.fetch("toyset", root=root, yes=True, progress=False)
    assert res2["downloaded"] == 0 and res2["skipped"] == 1 and len(http.requests) == n_req
    # optional part on request
    res3 = F.fetch("toyset", root=root, yes=True, parts=["opt.zip"], progress=False)
    assert res3["downloaded"] == 1 and "raw/opt.zip" in store.read_lock(root)["datasets"]["toyset"]["files"]


def test_fetch_cancel_and_unknown_part(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch)
    with pytest.raises(F.FetchCancelled):
        F.fetch("toyset", root=root, confirm=lambda t: False, progress=False)
    assert http.requests == []
    with pytest.raises(F.FetchError, match="part"):
        F.plan_fetch("toyset", parts=["nope.zip"], root=root)


def test_fetch_md5_mismatch_no_extracted_dir(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch, md5_ok=False)
    with pytest.raises(F.ChecksumMismatch):
        F.fetch("toyset", root=root, yes=True, progress=False)
    assert not (root / "toyset" / "extracted").exists()
    assert not (root / "toyset" / "raw" / "annotation.zip").exists()
    assert store.status("toyset", root) == "absent"


def test_manual_dataset_raises_with_korean_steps(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch)
    with pytest.raises(F.ManualDownloadRequired) as ei:
        F.plan_fetch("closed", root=root)
    assert "접근 요청 후 import-local" in str(ei.value)


def test_refused_dataset_file_carries_manual_steps(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch)
    http.mode["/annotation.zip"] = "cf"
    with pytest.raises(F.ManualDownloadRequired) as ei:
        F.fetch("toyset", root=root, yes=True, progress=False)
    assert "브라우저로 받아 import-local" in str(ei.value)


def test_present_file_without_lock_record_is_adopted(http, tmp_path, monkeypatch):
    root = _registry(tmp_path, http, monkeypatch)
    zdata = http.files["/annotation.zip"]
    (root / "toyset" / "raw").mkdir(parents=True)
    (root / "toyset" / "raw" / "annotation.zip").write_bytes(zdata)
    res = F.fetch("toyset", root=root, yes=True, progress=False)
    assert res["downloaded"] == 0 and http.requests == []
    assert store.read_lock(root)["datasets"]["toyset"]["files"]["raw/annotation.zip"]["verified_against"] == "md5"


# ------------------------------------------------------------------------------------------ Google Drive

_DRIVE_PAGE = (
    '<div class="flip-entries"><div class="flip-entry" id="entry-FOLDERSUB" tabindex="0" role="link">'
    '<div class="flip-entry-info"><a href="https://drive.google.com/drive/folders/FOLDERSUB" target="_blank">'
    '<div class="flip-entry-title">Sub</div></a></div></div>'
    '<div class="flip-entry" id="entry-FILE1" tabindex="0" role="link"><div class="flip-entry-info">'
    '<a href="https://drive.google.com/file/d/FILE1/view?usp=drive_web" target="_blank">'
    '<div class="flip-entry-title">1.wav</div></a></div></div>'
    '<div class="flip-entry" id="entry-DS" tabindex="0" role="link"><div class="flip-entry-info">'
    '<a href="https://drive.google.com/file/d/DS/view?usp=drive_web" target="_blank">'
    '<div class="flip-entry-title">.DS_Store</div></a></div></div></div>'
)
_DRIVE_SUB = (
    '<div class="flip-entry" id="entry-FILE2" tabindex="0" role="link"><div class="flip-entry-info">'
    '<a href="https://drive.google.com/file/d/FILE2/view?usp=drive_web" target="_blank">'
    '<div class="flip-entry-title">A &amp; B.midi</div></a></div></div>'
)


def test_parse_gdrive_listing():
    items = F.parse_gdrive_listing(_DRIVE_PAGE)
    assert items[0] == {"id": "FOLDERSUB", "name": "Sub", "type": "folder"}
    assert items[1] == {"id": "FILE1", "name": "1.wav", "type": "file"}


def test_list_gdrive_folder_recursive_skips_hidden():
    pages = {"ROOT": _DRIVE_PAGE, "FOLDERSUB": _DRIVE_SUB}
    files = F.list_gdrive_folder("ROOT", prefix="audio_x/",
                                 fetch_text=lambda url: pages[url.split("id=")[1]])
    assert files == [{"id": "FILE1", "path": "audio_x/1.wav"}, {"id": "FILE2", "path": "audio_x/Sub/A & B.midi"}]


def test_gdrive_folder_fetch_and_resume(http, tmp_path, monkeypatch):
    http.files["/list/ROOT"] = _DRIVE_PAGE.encode()
    http.files["/list/FOLDERSUB"] = _DRIVE_SUB.encode()
    http.files["/dl/FILE1"] = b"wav-bytes"
    http.files["/dl/FILE2"] = b"midi-bytes"
    for p in ("/list/ROOT", "/list/FOLDERSUB"):
        http.mode[p] = "html"
    monkeypatch.setattr(F, "GDRIVE_LIST_URL", http.url("/list/{id}"))
    monkeypatch.setattr(F, "GDRIVE_FILE_URL", http.url("/dl/{id}"))
    monkeypatch.setattr(F, "GDRIVE_PACE_S", 0.0)
    reg = tmp_path / "registry.toml"
    reg.write_text('''format = "gtab.dataset_registry/1"
[drv]
title = "Drive"
tier = "C"
access = "gdrive_folder"
license = "not stated"
page = "https://drive.google.com/x"
approx_bytes = 100
loader = "egdb"
families = ["guitar"]
manual_ko = "브라우저로 폴더를 받으세요"
[[drv.folders]]
name = "audio_x"
folder_id = "ROOT"
required = true
expected_files = 2
''', encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg)
    root = tmp_path / "datasets"
    res = F.fetch("drv", root=root, yes=True, progress=False)
    assert res["downloaded"] == 2 and res["status"] == "fetched"
    assert (root / "drv" / "raw" / "audio_x" / "Sub" / "A & B.midi").read_bytes() == b"midi-bytes"
    files = store.read_lock(root)["datasets"]["drv"]["files"]
    assert set(files) == {"raw/audio_x/1.wav", "raw/audio_x/Sub/A & B.midi"}
    assert files["raw/audio_x/1.wav"]["url"].endswith("/dl/FILE1")
    # throttled mid-way: what was fetched stays recorded, the error explains how to continue
    (root / "drv" / "raw" / "audio_x" / "1.wav").unlink()
    store.forget_files("drv", ["raw/audio_x/1.wav"], root)
    http.mode["/dl/FILE1"] = "html"
    with pytest.raises(F.ManualDownloadRequired, match="다시 실행"):
        F.fetch("drv", root=root, yes=True, progress=False)
    assert "raw/audio_x/Sub/A & B.midi" in store.read_lock(root)["datasets"]["drv"]["files"]
    assert store.status("drv", root) == "partial"
    # a transient server error (HTTP 500) is retried a bounded number of times, refusals are not
    http.mode["/dl/FILE1"] = "500x2"
    monkeypatch.setattr(F, "GDRIVE_RETRY_S", (0.0, 0.0))
    res = F.fetch("drv", root=root, yes=True, progress=False)
    assert res["downloaded"] == 1 and store.status("drv", root) == "fetched"
    (root / "drv" / "raw" / "audio_x" / "1.wav").unlink()
    store.forget_files("drv", ["raw/audio_x/1.wav"], root)
    http.mode["/dl/FILE1"] = "500x3"
    with pytest.raises(F.ServerError, match="HTTP 500"):
        F.fetch("drv", root=root, yes=True, progress=False)
