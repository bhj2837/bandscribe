"""Local score viewer (DESIGN 10 M3 "원곡에 싱크된 정적 alphaTab 뷰어"): ``bandscribe view <song>``.

A threaded HTTP server bound to 127.0.0.1 only (DESIGN 7.2) serves one job:
- ``/``                  the page (``viewer/index.html``; alphaTab in external-media mode follows the recording),
- ``/score.alphatex``    ``export/tab/score.alphatex`` (bar lines synced to the recording by ``\\sync`` points),
- ``/score.gp5``, ``/tab/<track>.txt``   downloads,
- ``/audio``             the job's original input (with HTTP Range, so the browser can seek),
- ``/alphatab/...``      alphaTab 1.8.4's ``dist`` from ``node/node_modules`` (fonts, worker script).
Nothing else of the disk is reachable (paths are resolved and must stay inside their root). Workers and fonts
need http(s), which is why this is a server and not a file:// page.
"""

from __future__ import annotations

import html
import mimetypes
import re
import threading
from collections.abc import Mapping
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from bandscribe import atomic

TEMPLATE = Path(__file__).with_name("viewer") / "index.html"
AUDIO_EXTS = (".mp3", ".m4a", ".mp4", ".wav", ".flac", ".ogg", ".opus", ".webm", ".aac")
MIME = {".alphatex": "text/plain; charset=utf-8", ".gp5": "application/octet-stream", ".mjs": "text/javascript",
        ".js": "text/javascript", ".woff2": "font/woff2", ".woff": "font/woff", ".sf2": "application/octet-stream",
        ".txt": "text/plain; charset=utf-8", ".mp4": "audio/mp4", ".m4a": "audio/mp4", ".flac": "audio/flac",
        ".wav": "audio/wav", ".mp3": "audio/mpeg"}
_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class ViewerError(RuntimeError):
    """User-facing (Korean) problem of the viewer."""


def job_files(job_dir: Path) -> dict[str, Any]:
    """The files a job's viewer serves (raises ViewerError when the score was never exported)."""
    job_dir = Path(job_dir)
    export = job_dir / "export"
    tex = export / "tab" / "score.alphatex"
    if not tex.is_file():
        raise ViewerError(f"악보가 아직 없습니다: {tex}. 먼저 `bandscribe run {job_dir.name}` 을 실행하세요.")
    inputs = sorted(p for p in (job_dir / "input").glob("source.*") if p.suffix.lower() in AUDIO_EXTS)
    audio = inputs[0] if inputs else job_dir / "input" / "mix_44k_f32.wav"
    meta: dict[str, Any] = {}
    try:
        meta = atomic.read_json(job_dir / "input" / "meta.json")
    except (OSError, ValueError):
        pass
    title = Path(str((meta.get("source") or {}).get("filename") or job_dir.name)).stem
    info = ""
    try:
        tab = atomic.read_json(export / "tab" / "tab.json") if (export / "tab" / "tab.json").is_file() else {}
        tun = tab.get("tuning") or {}
        bits = [f"{ko} {tun[k]['label']}" for k, ko in (("guitar", "기타"), ("bass", "베이스"))
                if isinstance(tun.get(k), dict) and tun[k].get("label")]
        info = " · ".join(bits)
    except (OSError, ValueError):
        pass
    return {"tex": tex, "gp5": export / "tab" / "score.gp5", "tab_dir": export / "tab", "audio": audio,
            "title": title, "info": info}


def _inside(root: Path, rel: str) -> Path | None:
    p = (root / rel).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError:
        return None
    return p if p.is_file() else None


class _Handler(BaseHTTPRequestHandler):
    server_version = "bandscribe-viewer"

    def __init__(self, *args: Any, files: Mapping[str, Any], alphatab: Path, **kw: Any) -> None:
        self.files = files
        self.alphatab = alphatab
        super().__init__(*args, **kw)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, ctype: str | None = None) -> None:
        ctype = ctype or MIME.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        start, end = 0, size - 1
        m = _RANGE.match(self.headers.get("Range", "") or "")
        partial_req = bool(m and (m.group(1) or m.group(2)))
        if partial_req:
            a, b = m.group(1), m.group(2)
            if a:
                start, end = int(a), int(b) if b else size - 1
            else:  # suffix range: the last N bytes
                start, end = max(0, size - int(b)), size - 1
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
        self.send_response(206 if partial_req else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial_req:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return  # the browser cancelled (seek)
                left -= len(chunk)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = unquote(urlsplit(self.path).path)
        f = self.files
        if route in ("/", "/index.html"):
            page = TEMPLATE.read_text(encoding="utf-8").replace("{{TITLE}}", html.escape(f["title"]))
            page = page.replace("{{INFO}}", html.escape(f["info"]))
            return self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
        if route == "/score.alphatex":
            return self._send_file(f["tex"])
        if route == "/score.gp5" and Path(f["gp5"]).is_file():
            return self._send_file(f["gp5"])
        if route == "/audio" and Path(f["audio"]).is_file():
            return self._send_file(f["audio"])
        if route.startswith("/tab/") and route.endswith(".txt"):
            p = _inside(Path(f["tab_dir"]), route[len("/tab/"):])
            if p:
                return self._send_file(p)
        if route.startswith("/alphatab/"):
            p = _inside(self.alphatab, route[len("/alphatab/"):])
            if p:
                return self._send_file(p)
        self.send_error(404)


def serve(job_dir: Path, *, port: int = 0, alphatab: Path | None = None) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the viewer server on 127.0.0.1 in a daemon thread; returns (server, thread)."""
    from bandscribe.eval import node

    files = job_files(job_dir)
    at = Path(alphatab or node.alphatab_dir() / "dist")
    if not (at / "alphaTab.min.js").is_file():
        raise ViewerError(f"alphaTab 이 없습니다: {at} (node 폴더에서 npm ci)")
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), partial(_Handler, files=files, alphatab=at))
    t = threading.Thread(target=httpd.serve_forever, name="bandscribe-viewer", daemon=True)
    t.start()
    return httpd, t
