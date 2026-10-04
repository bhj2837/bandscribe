"""bandscribe models fetch/verify/list against a local http.server (M1_M2_SPEC §9.5 item 7).

The real ``bandscribe.models`` records into a throw-away models.lock.json (``bandscribe.paths`` repointed at a temp
BANDSCRIBE_ROOT); one more test runs the fetch in a subprocess with BANDSCRIBE_ROOT set, as the CLI would.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from fixtures.datasets_fakes import HttpFixture, models_root
from bandscribe import models as M
from bandscribe import model_fetch as MF
from bandscribe.datasets import fetch as F

CKPT = b"checkpoint-bytes" * 5000


@pytest.fixture
def env(tmp_path, monkeypatch):
    http = HttpFixture()
    http.files["/final0.ckpt"] = CKPT
    http.mode["/final0.ckpt"] = "sha1"
    root = tmp_path / "bandscribe_root"
    lock = models_root(monkeypatch, root)
    src = tmp_path / "sources.toml"

    def write_sources(sha: str = "") -> None:
        src.write_text(f'''format = "bandscribe.model_sources/1"
["beat_this.final0"]
url = "{http.url('/final0.ckpt')}"
dest = "data/models/beat_this/final0.ckpt"
bytes = {len(CKPT)}
sha256 = "{sha}"
license = "MIT"
source = "https://github.com/CPJKU/beat_this"
notes = "test"
''', encoding="utf-8")

    write_sources()
    monkeypatch.setattr(MF, "SOURCES_PATH", src)
    yield {"http": http, "root": root, "lock": lock, "write_sources": write_sources}
    http.close()


def test_fetch_records_entry(env):
    prompts = []
    e = MF.fetch("beat_this.final0", confirm=lambda t: prompts.append(t) or True, progress=False)
    assert "MIT" in prompts[0] and "final0.ckpt" in prompts[0] and "TOFU" in prompts[0]
    assert f"{len(CKPT) / 1e6:,.1f} MB" in prompts[0]  # size from HEAD
    lock = json.loads(env["lock"].read_text(encoding="utf-8"))["models"]["beat_this.final0"]
    assert lock["url"].endswith("/final0.ckpt") and lock["bytes"] == len(CKPT)
    assert lock["sha256"] == hashlib.sha256(CKPT).hexdigest() and lock["license"] == "MIT"
    assert lock["path"] == "data/models/beat_this/final0.ckpt" and "SHA1" in lock["notes"]
    assert (env["root"] / "data" / "models" / "beat_this" / "final0.ckpt").read_bytes() == CKPT
    # already present and recorded: no network, no prompt
    n = len(env["http"].requests)
    e2 = MF.fetch("beat_this.final0", confirm=lambda t: pytest.fail("no prompt expected"), progress=False)
    assert e2.sha256 == e.sha256 and len(env["http"].requests) == n


def test_pinned_sha_mismatch_records_nothing(env):
    env["write_sources"]("0" * 64)
    with pytest.raises(F.ChecksumMismatch, match="SHA256"):
        MF.fetch("beat_this.final0", yes=True, progress=False)
    assert not (env["root"] / "data" / "models" / "beat_this" / "final0.ckpt").exists()
    assert M.get("beat_this.final0") is None


def test_pinned_sha_match(env):
    env["write_sources"](hashlib.sha256(CKPT).hexdigest())
    e = MF.fetch("beat_this.final0", yes=True, progress=False)
    assert e.sha256 == hashlib.sha256(CKPT).hexdigest()


def test_cancel_downloads_nothing(env):
    with pytest.raises(F.FetchCancelled):
        MF.fetch("beat_this.final0", confirm=lambda t: False, progress=False)
    assert [r for r in env["http"].requests if r[0] == "GET"] == []


def test_existing_file_is_adopted_without_download(env):
    dest = env["root"] / "data" / "models" / "beat_this" / "final0.ckpt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(CKPT)
    e = MF.fetch("beat_this.final0", yes=True, progress=False)
    assert e.sha256 == hashlib.sha256(CKPT).hexdigest()
    # no pin: one HEAD asks the server for its SHA1 (OC-Checksum), no GET
    assert [r[0] for r in env["http"].requests] == ["HEAD"]
    assert "adopted" in e.notes and "server-announced SHA1" in e.notes
    assert M.get("beat_this.final0").sha256 == e.sha256


def test_existing_file_not_matching_server_sha1_is_replaced(env):
    dest = env["root"] / "data" / "models" / "beat_this" / "final0.ckpt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * len(CKPT))  # right size, wrong content
    e = MF.fetch("beat_this.final0", yes=True, progress=False)
    assert dest.read_bytes() == CKPT and e.sha256 == hashlib.sha256(CKPT).hexdigest()
    assert any(r[0] == "GET" for r in env["http"].requests)


def test_existing_file_adopted_against_pinned_sha_without_network(env):
    env["write_sources"](hashlib.sha256(CKPT).hexdigest())
    dest = env["root"] / "data" / "models" / "beat_this" / "final0.ckpt"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(CKPT)
    e = MF.fetch("beat_this.final0", yes=True, progress=False)
    assert env["http"].requests == [] and "pinned sha256" in e.notes


def test_corrupt_lock_is_an_error_not_empty(env):
    env["lock"].write_text("{not json", encoding="utf-8")
    with pytest.raises(M.ModelRegistryError):
        MF.list_models()
    with pytest.raises(M.ModelRegistryError):
        MF.fetch("beat_this.final0", yes=True, progress=False)
    assert env["lock"].read_text(encoding="utf-8") == "{not json"  # never overwritten


def test_verify_detects_modified_file(env):
    MF.fetch("beat_this.final0", yes=True, progress=False)
    assert MF.verify() == [("beat_this.final0", True)]
    (env["root"] / "data" / "models" / "beat_this" / "final0.ckpt").write_bytes(b"tampered")
    assert MF.verify("beat_this.final0") == [("beat_this.final0", False)]
    with pytest.raises(MF.ModelFetchError):
        MF.verify("nope")


def test_list_merges_sources_and_lock(env):
    M.record(M.ModelEntry(name="bs_roformer_sw.ckpt", path="data/models/bs_roformer_sw/x.ckpt", url="unknown",
                          bytes=3, sha256="0" * 64, license="unknown - personal use only", source="local copy"))
    rows = {r["name"]: r for r in MF.list_models()}
    assert rows["beat_this.final0"]["status"] == "absent" and not rows["beat_this.final0"]["local_only"]
    assert rows["bs_roformer_sw.ckpt"]["local_only"] and rows["bs_roformer_sw.ckpt"]["status"] == "missing_file"
    with pytest.raises(MF.ModelFetchError, match="로컬 전용"):
        MF.fetch("bs_roformer_sw.ckpt", yes=True)
    with pytest.raises(MF.ModelFetchError, match="beat_this.final0"):
        MF.fetch("beat_this.final", yes=True)


def test_sources_validation(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('format = "bandscribe.model_sources/1"\n["x"]\nurl = "https://a/b"\ndest = "../outside"\nlicense = "MIT"\n')
    with pytest.raises(MF.ModelFetchError, match="상대 경로"):
        MF.load_sources(p)
    p.write_text('format = "bandscribe.model_sources/1"\n["x"]\nurl = "ftp://a/b"\ndest = "d"\nlicense = "MIT"\n')
    with pytest.raises(MF.ModelFetchError):
        MF.load_sources(p)


def test_real_models_module_in_subprocess(tmp_path):
    http = HttpFixture()
    try:
        http.files["/final0.ckpt"] = CKPT
        src = tmp_path / "sources.toml"
        src.write_text(f'format = "bandscribe.model_sources/1"\n["beat_this.final0"]\nurl = "{http.url("/final0.ckpt")}"\n'
                       f'dest = "data/models/beat_this/final0.ckpt"\nbytes = {len(CKPT)}\nsha256 = ""\n'
                       'license = "MIT"\nsource = "s"\nnotes = ""\n', encoding="utf-8")
        root = tmp_path / "root"
        code = ("import sys; from pathlib import Path; from bandscribe import model_fetch as MF; "
                f"MF.SOURCES_PATH = Path(r'{src}'); e = MF.fetch('beat_this.final0', yes=True, progress=False); "
                "print(e.sha256); assert MF.verify('beat_this.final0') == [('beat_this.final0', True)]")
        envv = dict(os.environ, BANDSCRIBE_ROOT=str(root), PYTHONUTF8="1")
        run = subprocess.run([sys.executable, "-c", code], env=envv, capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stderr
        lock = json.loads((root / "data" / "models" / "models.lock.json").read_text(encoding="utf-8"))
        assert lock["models"]["beat_this.final0"]["sha256"] == hashlib.sha256(CKPT).hexdigest()
    finally:
        http.close()
