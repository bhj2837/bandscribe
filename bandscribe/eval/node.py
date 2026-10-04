"""The one place that finds Node / npm and runs ``node/bandscribe_score.mjs`` (alphaTab).

Windows notes (spec §0): ``npm`` is ``npm.cmd``, a batch file that needs a shell and mangles arguments, so it
is never launched; we run ``node.exe <nodejs dir>\\node_modules\\npm\\bin\\npm-cli.js ci`` instead. npm's default
cache is under %LOCALAPPDATA%, which the Claude app sandbox redirects, so the cache goes to ``paths.NPM_CACHE``
(and update notifier / fund / audit are off: no network chatter besides the install itself). Every external
program runs through ``winjob.run_captured`` (bounded, tree-killing).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from bandscribe import paths
from bandscribe.winjob import CapturedRun, run_captured

log = logging.getLogger(__name__)

SCRIPT_NAME = "bandscribe_score.mjs"
ALPHATAB_VERSION = "1.8.4"


class NodeError(RuntimeError):
    """Node / alphaTab problem; the message is Korean and user-facing."""


def node_dir() -> Path:
    return paths.NODE_DIR


def score_script() -> Path:
    return node_dir() / SCRIPT_NAME


def alphatab_dir() -> Path:
    return node_dir() / "node_modules" / "@coderline" / "alphatab"


def node_exe() -> Path:
    found = shutil.which("node")
    if not found:
        raise NodeError("Node.js(node.exe)를 찾지 못했습니다. Node 22 이상을 설치하고 PATH 에 넣어 주세요.")
    return Path(found)


def npm_cli() -> Path:
    """npm's JS entry point next to node.exe (never npm.cmd)."""
    cli = node_exe().parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    if not cli.is_file():
        raise NodeError(f"npm-cli.js 를 찾지 못했습니다: {cli}")
    return cli


def npm_env() -> dict[str, str]:
    return paths.child_env({
        "npm_config_cache": str(paths.NPM_CACHE),
        "npm_config_update_notifier": "false",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
    })


def npm_ci_command() -> list[str]:
    return [str(node_exe()), str(npm_cli()), "ci"]


def npm_ci(*, timeout_s: float = 900) -> CapturedRun:
    """Install the pinned node packages from ``node/package-lock.json`` into ``node/node_modules``."""
    paths.NPM_CACHE.mkdir(parents=True, exist_ok=True)
    r = run_captured(npm_ci_command(), timeout_s=timeout_s, env=npm_env(), cwd=str(node_dir()))
    if r.returncode != 0:
        raise NodeError(f"npm ci 실패 (코드 {r.returncode}): {r.stderr.strip()[-800:]}")
    return r


def alphatab_available() -> bool:
    pkg = alphatab_dir() / "package.json"
    if not pkg.is_file() or not score_script().is_file() or shutil.which("node") is None:
        return False
    try:
        return json.loads(pkg.read_text(encoding="utf-8")).get("version") == ALPHATAB_VERSION
    except (OSError, ValueError):
        return False


def require_alphatab() -> None:
    if not alphatab_available():
        raise NodeError(f"alphaTab {ALPHATAB_VERSION} 이 설치되어 있지 않습니다. {node_dir()} 에서 "
                        f"`node <nodejs>\\node_modules\\npm\\bin\\npm-cli.js ci` 를 실행하세요 "
                        "(bandscribe.eval.node.npm_ci()).")


def run_score(args: list[str], *, timeout_s: float = 600) -> CapturedRun:
    require_alphatab()
    r = run_captured([str(node_exe()), str(score_script()), *[str(a) for a in args]], timeout_s=timeout_s,
                     env=paths.child_env(), cwd=str(node_dir()))
    if r.timed_out:
        raise NodeError(f"alphaTab 스크립트가 {timeout_s:.0f}초 안에 끝나지 않았습니다: {args[0]}")
    if r.returncode == 3 and "SOUNDFONT_MISSING" in r.stderr:
        raise NodeError("alphaTab 사운드폰트(dist/soundfont/sonivox.sf2)를 찾지 못했습니다. npm ci 를 다시 실행하세요.")
    if r.returncode != 0:
        raise NodeError(f"alphaTab 스크립트 실패 (코드 {r.returncode}): {r.stderr.strip()[-800:]}")
    return r


def _scratch() -> tempfile.TemporaryDirectory:
    # Under data/, not %TEMP%: the app sandbox may redirect %LOCALAPPDATA% writes of child processes.
    base = paths.DATA / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="node-", dir=base)


def alphatab_import(path: Path, *, encoding: str | None = None, timeout_s: float = 300) -> Any:
    """Import a score with alphaTab -> ``RefScore``."""
    from bandscribe.eval.refscore import RefScore

    with _scratch() as tmp:
        out = Path(tmp) / "refscore.json"
        args = ["import", str(Path(path).resolve()), str(out)]
        if encoding:
            args += ["--encoding", encoding]
        run_score(args, timeout_s=timeout_s)
        return RefScore.model_validate_json(out.read_bytes())


def alphatab_render(path: Path, out_wav: Path, *, timeout_s: float = 900, sample_rate: int = 44100) -> dict:
    """Render a score to a float32 stereo WAV with alphaTab's synthesizer (sonivox soundfont)."""
    r = run_score(["render", str(Path(path).resolve()), str(Path(out_wav).resolve()), "--sample-rate", str(sample_rate)],
                  timeout_s=timeout_s)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {}
