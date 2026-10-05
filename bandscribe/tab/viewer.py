"""Local score viewer (DESIGN 10 M3 "원곡에 싱크된 정적 alphaTab 뷰어"): ``bandscribe view <song>``.

A threaded HTTP server bound to 127.0.0.1 only (DESIGN 7.2) serves one job:
- ``/``                  the page (``viewer/index.html``; alphaTab in external-media mode follows the recording),
- ``/score.alphatex``    ``export/tab/score.alphatex`` (bar lines synced to the recording by ``\\sync`` points),
- ``/score.gp5``, ``/tab/<track>.txt``   downloads,
- ``/audio``             the job's original input (with HTTP Range, so the browser can seek),
- ``/alphatab/...``      alphaTab 1.8.4's ``dist`` from ``node/node_modules`` (fonts, worker script),
- ``/api/score``         M4a: the (edited) alphaTex, the note map and the edit version in one JSON,
- ``POST /api/cmd``      M4a: one edit (``bandscribe.tab.edits.Session.command``); needs the page's token header.
With an edit session ``/score.alphatex`` is the edited score; every edit lands in the job's ``edits.jsonl`` and
the score files of ``export/`` follow it. Every request must carry this server's own Host (DNS rebinding).
Nothing else of the disk is reachable (paths are resolved and must stay inside their root). Workers and fonts
need http(s), which is why this is a server and not a file:// page.
"""

from __future__ import annotations

import html
import json
import logging
import mimetypes
import re
import secrets
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
MAX_BODY = 64 * 1024  # an edit command is a few hundred bytes
log = logging.getLogger(__name__)


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
    title = ""
    try:  # the score's own title (hints.title or the file name), so the page and the score agree
        if (export / "score.json").is_file():
            title = str((atomic.read_json(export / "score.json").get("song") or {}).get("title") or "")
    except (OSError, ValueError):
        pass
    title = title or Path(str((meta.get("source") or {}).get("filename") or job_dir.name)).stem
    info = ""
    try:
        tab = atomic.read_json(export / "tab" / "tab.json") if (export / "tab" / "tab.json").is_file() else {}
        tun = tab.get("tuning") or {}
        bits = [f"{ko} {tun[k]['label']}" + (" (지정)" if tun[k].get("user") else "")
                for k, ko in (("guitar", "기타"), ("bass", "베이스")) if isinstance(tun.get(k), dict) and tun[k].get("label")]
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


class _Ctx:
    """What every request of one viewer server shares."""

    def __init__(self, files: Mapping[str, Any], alphatab: Path, session: Any, readonly: str) -> None:
        self.files = files
        self.alphatab = alphatab
        self.session = session          # bandscribe.tab.edits.Session, or None (read-only)
        self.readonly = readonly        # why there is no session ("" when editing works)
        self.token = secrets.token_urlsafe(18)


class _Handler(BaseHTTPRequestHandler):
    server_version = "bandscribe-viewer"

    def __init__(self, *args: Any, ctx: _Ctx, **kw: Any) -> None:
        self.ctx = ctx
        self.files = ctx.files
        self.alphatab = ctx.alphatab
        super().__init__(*args, **kw)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _host_ok(self) -> bool:
        """Only this server's own address (a page elsewhere that re-points a host name at 127.0.0.1 is refused)."""
        port = self.server.server_address[1]
        return (self.headers.get("Host") or "").strip().lower() in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send_bytes(self, data: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if ctype.startswith("text/html"):  # no other site may frame the editor and steer its keys
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

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
        if not self._host_ok():
            return self.send_error(403)
        route = unquote(urlsplit(self.path).path)
        f = self.files
        s = self.ctx.session
        if route in ("/", "/index.html"):
            page = TEMPLATE.read_text(encoding="utf-8").replace("{{TITLE}}", html.escape(f["title"]))
            page = page.replace("{{INFO}}", html.escape(f["info"])).replace("{{TOKEN}}", self.ctx.token)
            return self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
        if route == "/api/score":
            if s is None:
                return self._json({"readonly": self.ctx.readonly, "version": 0, "map": None,
                                   "tex": Path(f["tex"]).read_text(encoding="utf-8")})
            return self._json({"readonly": "", **s.snapshot()})
        if route == "/score.alphatex":
            if s is not None:
                return self._send_bytes(s.snapshot()["tex"].encode("utf-8"), MIME[".alphatex"])
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

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        """``/api/cmd``: one edit command from the page (JSON). Needs this server's token in ``X-Bandscribe-Token``
        (a page from another origin cannot send that header) and the server's own Host."""
        from bandscribe.tab import edits as E

        if not self._host_ok():
            return self._json({"error": "허용되지 않은 주소입니다"}, 403)
        if unquote(urlsplit(self.path).path) != "/api/cmd":
            return self._json({"error": "없는 경로입니다"}, 404)
        if not secrets.compare_digest(self.headers.get("X-Bandscribe-Token") or "", self.ctx.token):
            return self._json({"error": "편집 권한이 없습니다(페이지를 새로 고치세요)"}, 403)
        s = self.ctx.session
        if s is None:
            return self._json({"error": f"이 악보는 편집할 수 없습니다: {self.ctx.readonly}"}, 409)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            return self._json({"error": "요청 크기가 맞지 않습니다"}, 413)
        try:
            cmd = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(cmd, dict):
                raise ValueError("not an object")
        except ValueError:
            return self._json({"error": "요청을 읽지 못했습니다"}, 400)
        try:
            return self._json(s.command(cmd))
        except E.StaleVersion as e:
            return self._json({"error": str(e), "stale": True}, 409)
        except E.EditError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001 - the page shows it; the server keeps running
            log.exception("viewer: command %s failed", cmd.get("cmd"))
            return self._json({"error": f"편집을 처리하지 못했습니다: {e}"}, 500)


def serve(job_dir: Path, *, port: int = 0, alphatab: Path | None = None, edit: bool = True
          ) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start the viewer server on 127.0.0.1 in a daemon thread; returns (server, thread). With ``edit`` the page
    can edit the score (``bandscribe.tab.edits``); ``httpd.session`` is that session (None: read-only, and
    ``httpd.readonly`` says why)."""
    from bandscribe.eval import node

    files = job_files(job_dir)
    at = Path(alphatab or node.alphatab_dir() / "dist")
    if not (at / "alphaTab.min.js").is_file():
        raise ViewerError(f"alphaTab 이 없습니다: {at} (node 폴더에서 npm ci)")
    session, readonly = None, "편집 끔"
    if edit:
        from bandscribe.tab import edits as E

        try:
            session, readonly = E.Session(Path(job_dir)), ""
        except E.EditError as e:
            readonly = str(e)
        except Exception as e:  # noqa: BLE001 - the score can still be viewed
            log.exception("viewer: the edit session of %s did not open", Path(job_dir).name)
            readonly = f"편집 기록을 열지 못했습니다: {e}"
    ctx = _Ctx(files, at, session, readonly)
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), partial(_Handler, ctx=ctx))
    httpd.session = session  # type: ignore[attr-defined]
    httpd.readonly = readonly  # type: ignore[attr-defined]
    t = threading.Thread(target=httpd.serve_forever, name="bandscribe-viewer", daemon=True)
    t.start()
    return httpd, t
