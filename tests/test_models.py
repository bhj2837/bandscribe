"""bandscribe.models: models.lock.json registry, sha256 memo, ModelMissing (M1_M2_SPEC 1.5)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bandscribe import models, paths


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    r = tmp_path / "root"
    (r / "data" / "models").mkdir(parents=True)
    monkeypatch.setattr(paths, "ROOT", r)
    monkeypatch.setattr(paths, "MODELS", r / "data" / "models")
    monkeypatch.setattr(paths, "MODELS_LOCK", r / "data" / "models" / "models.lock.json")
    return r


def _entry(name: str, rel: str, data: bytes, **kw) -> models.ModelEntry:
    return models.ModelEntry(name=name, path=rel, url=kw.get("url", "https://example.invalid/m"), bytes=len(data),
                             sha256=hashlib.sha256(data).hexdigest(), license="MIT", source="test",
                             notes=kw.get("notes", ""), recorded_utc="2026-10-03T00:00:00Z")


def _put(root: Path, rel: str, data: bytes) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_record_get_replace_and_format(root: Path) -> None:
    assert models.get("x") is None and models.entries() == {}
    _put(root, "data/models/x/w.bin", b"abc")
    e = _entry("x", "data/models/x/w.bin", b"abc")
    models.record(e)
    assert models.get("x") == e
    models.record(_entry("a.first", "data/models/a.bin", b"zz"))
    models.record(_entry("x", "data/models/x/w.bin", b"abc", notes="replaced"))
    doc = json.loads(paths.MODELS_LOCK.read_text(encoding="utf-8"))
    assert doc["format"] == "bandscribe.models/1"
    assert list(doc["models"]) == ["a.first", "x"]  # sorted, replaced in place (no duplicate)
    assert doc["models"]["x"]["notes"] == "replaced"
    assert set(doc["models"]["x"]) == {"name", "path", "url", "bytes", "sha256", "license", "source", "notes",
                                       "recorded_utc"}
    assert models.MODELS_LOCK == paths.MODELS_LOCK  # live attribute for bandscribe.model_fetch


def test_local_path_and_model_missing(root: Path) -> None:
    with pytest.raises(models.ModelMissing) as ei:
        models.local_path("beat_this.final0")
    e = ei.value
    assert isinstance(e, RuntimeError) and e.name == "beat_this.final0"
    assert e.hint_ko == str(e)
    assert "Beat This! 체크포인트가 없습니다" in str(e) and "bandscribe models fetch beat_this.final0" in str(e)

    models.record(_entry("beat_this.final0", "data/models/beat_this/final0.ckpt", b"ckpt"))
    with pytest.raises(models.ModelMissing, match="파일이 없습니다"):
        models.local_path("beat_this.final0")  # recorded but absent
    p = _put(root, "data/models/beat_this/final0.ckpt", b"ckpt")
    assert models.local_path("beat_this.final0") == p


def test_model_missing_hints_for_local_only_models() -> None:
    assert "받을 곳이 없습니다" in str(models.ModelMissing("bs_roformer_sw.ckpt"))
    assert "hf-login" in str(models.ModelMissing("muscriptor-small"))
    custom = models.ModelMissing("n", hint_ko="직접 쓴 안내")
    assert str(custom) == "직접 쓴 안내" and custom.name == "n"


def test_absolute_entry_path_is_kept(root: Path, tmp_path: Path) -> None:
    p = tmp_path / "elsewhere" / "m.bin"
    p.parent.mkdir()
    p.write_bytes(b"q")
    models.record(_entry("abs", str(p), b"q"))
    assert models.local_path("abs") == p


def test_verify_detects_modification(root: Path) -> None:
    p = _put(root, "data/models/m.bin", b"original")
    models.record(_entry("m", "data/models/m.bin", b"original"))
    assert models.verify("m") is True
    p.write_bytes(b"tampered")
    assert models.verify("m") is False
    p.unlink()
    with pytest.raises(FileNotFoundError):
        models.verify("m")
    with pytest.raises(models.ModelMissing):
        models.verify("nope")


def test_sha256_cached_memo(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _put(root, "data/models/big.bin", b"x" * 100_000)
    want = hashlib.sha256(b"x" * 100_000).hexdigest()
    calls = []
    real = models.atomic.sha256_file
    monkeypatch.setattr(models.atomic, "sha256_file", lambda q, *a, **k: calls.append(q) or real(q, *a, **k))
    assert models.sha256_cached(p) == want
    assert models.sha256_cached(p) == want
    assert len(calls) == 1  # second call is a memo hit
    cache = json.loads(models.sha_cache_path().read_text(encoding="utf-8"))
    assert cache["format"] == "bandscribe.sha_cache/1" and len(cache["entries"]) == 1
    # a changed file (size / mtime) is re-hashed
    p.write_bytes(b"y" * 100_001)
    assert models.sha256_cached(p) == hashlib.sha256(b"y" * 100_001).hexdigest()
    assert len(calls) == 2


def test_corrupt_sha_cache_is_discarded(root: Path) -> None:
    p = _put(root, "data/models/m.bin", b"abc")
    models.sha_cache_path().write_text("{not json", encoding="utf-8")
    assert models.sha256_cached(p) == hashlib.sha256(b"abc").hexdigest()
    assert json.loads(models.sha_cache_path().read_text(encoding="utf-8"))["format"] == "bandscribe.sha_cache/1"
    with pytest.raises(FileNotFoundError):
        models.sha256_cached(root / "missing.bin")


def test_corrupt_lock_is_a_clear_error_and_never_overwritten(root: Path) -> None:
    paths.MODELS_LOCK.write_text("[1, 2", encoding="utf-8")
    with pytest.raises(models.ModelRegistryError, match="읽지 못했습니다"):
        models.get("x")
    with pytest.raises(models.ModelRegistryError):
        models.record(_entry("x", "data/models/x.bin", b"1"))
    assert paths.MODELS_LOCK.read_text(encoding="utf-8") == "[1, 2"


def test_lenient_entry_fields(root: Path) -> None:
    # doctor/test fixtures write minimal entries; get() must still work
    paths.MODELS_LOCK.write_text(json.dumps({"format": "bandscribe.models/1", "models": {"m": {"path": "a/b.bin"}}}),
                                 encoding="utf-8")
    e = models.get("m")
    assert e is not None and e.name == "m" and e.path == "a/b.bin" and e.bytes == 0


def test_concurrent_records_in_processes_keep_every_entry(root: Path) -> None:
    code = (
        "import sys; from pathlib import Path; from bandscribe import models, paths\n"
        "paths.MODELS_LOCK = Path(sys.argv[1]); paths.MODELS = paths.MODELS_LOCK.parent\n"
        "for i in range(10):\n"
        "    models.record(models.ModelEntry(name=f'{sys.argv[2]}{i}', path='p', url='u', bytes=1, sha256='0'*64,\n"
        "                                    license='l', source='s'))\n"
    )
    env = dict(os.environ, PYTHONUTF8="1")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(paths.MODELS_LOCK), tag], env=env)
             for tag in ("a", "b", "c")]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    assert len(models.entries()) == 30


def test_seeded_registry_matches_config() -> None:
    """The real models.lock.json (seeded by P0) records the SW checkpoint with the configured sha."""
    from bandscribe.config import Config

    lock = Path(__file__).resolve().parents[1] / "data" / "models" / "models.lock.json"
    if not lock.is_file():
        pytest.skip("data/models/models.lock.json not seeded on this machine")
    doc = json.loads(lock.read_text(encoding="utf-8"))
    sw = doc["models"].get("bs_roformer_sw.ckpt")
    assert sw is not None and sw["sha256"] == Config().sep.ckpt_sha256
    assert "personal use only" in sw["license"]
    ms = doc["models"].get("muscriptor-medium")
    assert ms is not None and Config().amt.muscriptor_revision in ms["url"] and "CC BY-NC" in ms["license"]
