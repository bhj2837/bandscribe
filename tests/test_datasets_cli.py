"""`bandscribe data ...` and `bandscribe models ...` commands (Korean output, exit codes 0/1/2)."""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest
import soundfile as sf
import typer
from typer.testing import CliRunner

from fixtures.datasets_fakes import HttpFixture, models_root, write_wav
from bandscribe import model_fetch as MF
from bandscribe import paths
from bandscribe.commands import data as data_cmd
from bandscribe.commands import models as models_cmd
from bandscribe.datasets import registry


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def _root() -> None:
        pass

    data_cmd.register(app)
    models_cmd.register(app)
    return app


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATASETS", tmp_path / "datasets", raising=False)
    monkeypatch.setattr(paths, "EVAL", tmp_path / "eval", raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    runner = CliRunner()
    app = _app()
    return lambda *args, input=None: runner.invoke(app, list(args), input=input)


@pytest.fixture
def toy(tmp_path, monkeypatch):
    http = HttpFixture()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("annotation/a.jams", "{}")
    data = buf.getvalue()
    http.files["/a.zip"] = data
    reg = tmp_path / "registry.toml"
    reg.write_text(f'''format = "bandscribe.dataset_registry/1"
[toyset]
title = "Toy"
tier = "C"
access = "zenodo"
license = "CC BY 4.0"
page = "https://example.org/toy"
approx_bytes = {len(data)}
loader = "guitarset"
families = ["guitar"]
manual_ko = "브라우저로 받아 import-local"
[[toyset.files]]
name = "a.zip"
url = "{http.url('/a.zip')}"
bytes = {len(data)}
md5 = "{hashlib.md5(data).hexdigest()}"
required = true
extract = true
[closed]
title = "Closed"
tier = "B"
access = "manual"
license = "restricted"
page = "https://example.org/c"
approx_bytes = 0
loader = "medleydb"
families = ["guitar"]
manual_ko = "접근 요청 후 import-local"
''', encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg)
    yield http
    http.close()


def test_data_list_real_registry(cli):
    r = cli("data", "list")
    assert r.exit_code == 0, r.output
    for word in ("guitarset", "cambridge_mt", "없음", "수동 필요", "CC BY 4.0", "라이선스"):
        assert word in r.output


def test_data_fetch_verify_flow(cli, toy):
    r = cli("data", "fetch", "toyset", input="n\n")
    assert r.exit_code == 1 and "취소" in r.output and "https://example.org/toy" in r.output
    r = cli("data", "fetch", "toyset", "--yes")
    assert r.exit_code == 0, r.output
    assert "새로 받음 1개" in r.output
    r = cli("data", "verify", "toyset")
    assert r.exit_code == 0 and "OK" in r.output and "검증됨" in r.output
    r = cli("data", "list")
    assert "검증됨" in r.output


def test_data_fetch_errors(cli, toy):
    r = cli("data", "fetch", "nosuch")
    assert r.exit_code == 2 and "등록되지 않은" in r.output
    r = cli("data", "fetch", "closed")
    assert r.exit_code == 1 and "접근 요청 후 import-local" in r.output
    toy.mode["/a.zip"] = "cf"
    r = cli("data", "fetch", "toyset", "--yes")
    assert r.exit_code == 1 and "import-local" in r.output


def test_import_local_cambridge(cli, tmp_path):
    pytest.importorskip("yaml")
    src = tmp_path / "dl"
    for n in ("01_Kick.wav", "02_ElecGtr1.wav", "03_ElecGtr2.wav"):
        write_wav(src / n, 0.1)
    r = cli("data", "import-local", "cambridge_mt", str(src))
    assert r.exit_code == 1 and "--song" in r.output
    r = cli("data", "import-local", "cambridge_mt", str(src), "--song", "My Song", "--copy")
    assert r.exit_code == 0, r.output
    assert "lines.yaml" in r.output and (tmp_path / "datasets" / "cambridge_mt" / "local" / "my_song" / "lines.yaml").is_file()
    r = cli("data", "list")
    assert "받음(직접)" in r.output


def test_tier_a_commands(cli, tmp_path):
    pytest.importorskip("yaml")
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal((5 * 44100, 2))).astype(np.float32)
    sf.write(str(tmp_path / "o.wav"), x, 44100, subtype="FLOAT")
    sf.write(str(tmp_path / "b.wav"), np.concatenate([np.zeros((300, 2), np.float32), 0.5 * x]), 44100, subtype="FLOAT")
    r = cli("data", "tier-a", "list")
    assert r.exit_code == 0 and "등록된 곡이 없습니다" in r.output
    r = cli("data", "tier-a", "add", "t1", str(tmp_path / "o.wav"), "--title", "제목", "--artist", "가수",
            "--backing", str(tmp_path / "b.wav"))
    assert r.exit_code == 0, r.output
    r = cli("data", "tier-a", "list")
    assert "t1" in r.output and "제목" in r.output
    r = cli("data", "tier-a", "align", "t1", "--max-offset-s", "1")
    assert r.exit_code == 0, r.output
    assert "정답이 아닙니다" in r.output and "+0.0068" in r.output  # 300 samples
    out = tmp_path / "eval" / "tierA" / "t1" / "backing"
    assert (out / "residual_not_gt.wav").is_file() and (out / "README_NOT_GROUND_TRUTH.txt").is_file()
    r = cli("data", "tier-a", "align", "nope")
    assert r.exit_code == 1


def test_models_commands(cli, tmp_path, monkeypatch):
    http = HttpFixture()
    try:
        http.files["/m.ckpt"] = b"x" * 1000
        src = tmp_path / "sources.toml"
        src.write_text(f'format = "bandscribe.model_sources/1"\n["toy.model"]\nurl = "{http.url("/m.ckpt")}"\n'
                       'dest = "data/models/toy/m.ckpt"\nbytes = 1000\nsha256 = ""\nlicense = "MIT"\n',
                       encoding="utf-8")
        monkeypatch.setattr(MF, "SOURCES_PATH", src)
        models_root(monkeypatch, tmp_path / "root")
        r = cli("models", "list")
        assert r.exit_code == 0 and "toy.model" in r.output and "없음" in r.output
        r = cli("models", "fetch", "toy.model", input="n\n")
        assert r.exit_code == 1 and "취소" in r.output and "MIT" in r.output
        r = cli("models", "fetch", "toy.model", "--yes")
        assert r.exit_code == 0 and "기록" in r.output, r.output
        r = cli("models", "verify")
        assert r.exit_code == 0 and "OK" in r.output
        (tmp_path / "root" / "data" / "models" / "toy" / "m.ckpt").write_bytes(b"y" * 1000)
        r = cli("models", "verify", "toy.model")
        assert r.exit_code == 1
        r = cli("models", "fetch", "nope", "--yes")
        assert r.exit_code == 1 and "모르는 모델" in r.output
    finally:
        http.close()


def test_command_modules_import_light():
    import subprocess
    import sys

    code = ("import sys, bandscribe.commands.data, bandscribe.commands.models; "
            "heavy = [m for m in ('torch','librosa','numpy','scipy','bandscribe.datasets','yaml') if m in sys.modules]; "
            "print(','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
