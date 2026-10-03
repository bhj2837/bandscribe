"""Node / npm invocation: npm is run as `node.exe …\\npm-cli.js ci` with the cache on D:, never through npm.cmd."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gtab import paths
from gtab.eval import node

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")


def test_npm_ci_command_uses_node_and_npm_cli():
    cmd = node.npm_ci_command()
    assert Path(cmd[0]).name.lower() in ("node.exe", "node")
    assert cmd[1].replace("\\", "/").endswith("node_modules/npm/bin/npm-cli.js")
    assert cmd[2:] == ["ci"]
    assert not any(c.lower().endswith((".cmd", ".bat", ".ps1")) for c in cmd)


def test_npm_env_keeps_cache_off_appdata():
    env = node.npm_env()
    assert env["npm_config_cache"] == str(paths.NPM_CACHE)
    assert Path(env["npm_config_cache"]).resolve().is_relative_to(paths.GTAB_ROOT.resolve())
    assert env["npm_config_update_notifier"] == "false"
    assert env["npm_config_fund"] == "false" and env["npm_config_audit"] == "false"
    assert env["PYTHONUTF8"] == "1"  # the managed env is the base


def test_npm_ci_runs_through_run_captured(monkeypatch):
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "ok", ""

    def fake(args, **kw):
        seen.update(args=args, **kw)
        return R()

    monkeypatch.setattr(node, "run_captured", fake)
    node.npm_ci(timeout_s=5)
    assert seen["args"] == node.npm_ci_command()
    assert seen["cwd"] == str(paths.NODE_DIR) and seen["env"]["npm_config_cache"] == str(paths.NPM_CACHE)


def test_package_json_pins_alphatab():
    pkg = json.loads((paths.GTAB_ROOT / "node" / "package.json").read_text(encoding="utf-8"))
    assert pkg["dependencies"]["@coderline/alphatab"] == node.ALPHATAB_VERSION
    lock = json.loads((paths.GTAB_ROOT / "node" / "package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"]["node_modules/@coderline/alphatab"]["version"] == node.ALPHATAB_VERSION


def test_missing_alphatab_is_a_clean_korean_error(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "NODE_DIR", tmp_path)
    assert not node.alphatab_available()
    with pytest.raises(node.NodeError, match="npm-cli.js ci"):
        node.require_alphatab()
