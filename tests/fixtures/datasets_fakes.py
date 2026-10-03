"""Test helpers for the DATA owner's tests (tests/test_datasets_*.py, tests/test_model_fetch.py).

- ``models_root(monkeypatch, root)``: repoint P0's ``gtab.models`` (via ``gtab.paths``) at a throw-away
  GTAB_ROOT, so tests record into a temp ``models.lock.json`` through the real module.
- ``write_midi``: a tiny Standard MIDI File writer (format 1) for label fixtures.
- ``HttpFixture``: a local ``http.server`` in a thread with Range support and switchable failure modes.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------------------------- models


def models_root(monkeypatch: Any, root: Path) -> Path:
    """Point ``gtab.paths`` (GTAB_ROOT, MODELS, MODELS_LOCK) at ``root``; ``gtab.models`` reads them at call time."""
    from gtab import paths

    root = Path(root)
    (root / "data" / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(paths, "GTAB_ROOT", root)
    monkeypatch.setattr(paths, "MODELS", root / "data" / "models")
    monkeypatch.setattr(paths, "MODELS_LOCK", root / "data" / "models" / "models.lock.json")
    return root / "data" / "models" / "models.lock.json"


# ------------------------------------------------------------------------------------------------ MIDI


def _vlq(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def write_midi(path: Path, tracks: list[tuple[str, list[tuple[float, float, int, int]]]], *, bpm: float = 120.0,
               tpb: int = 480, name_tracks: bool = True) -> None:
    """Format-1 SMF: track 0 = tempo; then one track per (name, [(on_s, off_s, pitch, vel)]), channel = index."""
    us = int(round(60e6 / bpm))
    t0 = _vlq(0) + b"\xff\x51\x03" + us.to_bytes(3, "big") + _vlq(0) + b"\xff\x58\x04\x04\x02\x18\x08"
    t0 += _vlq(0) + b"\xff\x2f\x00"
    chunks = [t0]
    for ch, (name, notes) in enumerate(tracks):
        evs = []
        for on, off, pitch, vel in notes:
            evs.append((round(on * bpm / 60 * tpb), 1, bytes([0x90 | ch, pitch, vel])))
            evs.append((round(off * bpm / 60 * tpb), 0, bytes([0x80 | ch, pitch, 0])))
        evs.sort(key=lambda e: (e[0], e[1]))
        body = b""
        if name_tracks:
            nb = name.encode("latin-1")
            body += _vlq(0) + b"\xff\x03" + _vlq(len(nb)) + nb
        last = 0
        for tick, _, data in evs:
            body += _vlq(tick - last) + data
            last = tick
        body += _vlq(0) + b"\xff\x2f\x00"
        chunks.append(body)
    out = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), tpb)
    for c in chunks:
        out += b"MTrk" + struct.pack(">I", len(c)) + c
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(out)


def write_wav(path: Path, seconds: float = 1.0, *, sr: int = 44100, channels: int = 1, seed: int = 0) -> None:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    x = (0.1 * rng.standard_normal((int(seconds * sr), channels))).astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")


# ------------------------------------------------------------------------------------------------ HTTP


class HttpFixture:
    """Serve ``files`` ({"/path": bytes}) on 127.0.0.1. Knobs per path via ``mode``:
    ``"cut:<n>"`` = send only n bytes then drop the connection (once), ``"403"``, ``"cf"`` (403 + Cloudflare
    challenge header), ``"html"`` (200 text/html), ``"norange"`` (ignore Range), ``"sha1"`` (OC-Checksum),
    ``"500x<n>"`` (HTTP 500 for the next n requests)."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.mode: dict[str, str] = {}
        self.requests: list[tuple[str, str, str | None]] = []
        fixture = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:
                pass

            def _serve(self, head: bool) -> None:
                path = self.path.split("?")[0]
                rng = self.headers.get("Range")
                fixture.requests.append((self.command, path, rng))
                mode = fixture.mode.get(path, "")
                if path not in fixture.files:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if mode.startswith("500x"):  # "500x<n>": answer HTTP 500 n times, then serve normally
                    left = int(mode[4:])
                    fixture.mode[path] = f"500x{left - 1}" if left > 1 else ""
                    self.send_response(500)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if mode in ("403", "cf"):
                    self.send_response(403)
                    if mode == "cf":
                        self.send_header("cf-mitigated", "challenge")
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = fixture.files[path]
                start = 0
                if rng and mode != "norange":
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Type", "text/html" if mode == "html" else "application/octet-stream")
                if mode == "sha1":
                    self.send_header("OC-Checksum", "SHA1:" + hashlib.sha1(data).hexdigest())
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if head:
                    return
                if mode.startswith("cut:"):
                    n = int(mode.split(":")[1])
                    fixture.mode[path] = ""  # only once
                    self.wfile.write(body[:n])
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(2)
                    except OSError:
                        pass
                    return
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._serve(False)

            def do_HEAD(self) -> None:
                self._serve(True)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}{path}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
