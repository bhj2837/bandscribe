"""Logging setup: rich console handler (stderr) + plain UTF-8 file handler (data/logs/bandscribe.log).

Idempotent: calling setup_logging() again replaces bandscribe's own handlers instead of stacking them, so the
CLI can call it once at start-up and again when ``--verbose`` is given.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

from bandscribe import paths

FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(process)d] %(name)s: %(message)s"
_TAG = "_bandscribe_handler"
_logfile: Path | None = None


class _OwnDebugFilter(logging.Filter):
    """Full DEBUG detail from bandscribe's own loggers, INFO+ from libraries (filelock alone would flood the file)."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO or record.name == "bandscribe" or record.name.startswith("bandscribe.")


class _StderrHandler(logging.StreamHandler):
    """Fallback console handler that looks up sys.stderr at emit time (it may be swapped, e.g. by test runners)."""

    @property  # type: ignore[override]
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, value: object) -> None:
        pass  # StreamHandler.__init__ assigns it; ignored on purpose


def _to_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    value = logging.getLevelName(level.upper())
    if not isinstance(value, int):
        raise ValueError(f"unknown log level: {level!r}")
    return value


def _console_handler() -> logging.Handler:
    try:
        from rich.console import Console
        from rich.logging import RichHandler
    except ImportError:
        handler: logging.Handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        return handler
    # Console(stderr=True) without a file resolves sys.stderr on every write, so it follows re-wrapping.
    # markup=False: log messages carry paths and titles with [brackets] that must print literally.
    return RichHandler(
        console=Console(stderr=True),
        show_path=False,
        markup=False,
        rich_tracebacks=False,
        log_time_format="[%H:%M:%S]",
    )


def remove_handlers() -> None:
    """Detach and close the handlers installed by setup_logging (tests use this to release the log file)."""
    global _logfile
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _TAG, False):
            root.removeHandler(handler)
            handler.close()
    _logfile = None


def setup_logging(level: str | int = "INFO", logfile: Path | None = None) -> None:
    """Console at ``level``; file at DEBUG for bandscribe.* (INFO+ for libraries). ``logfile`` defaults to LOGS/bandscribe.log."""
    global _logfile
    console_level = _to_level(level)  # validate before tearing down the current handlers
    remove_handlers()
    root = logging.getLogger()

    console = _console_handler()
    console.setLevel(console_level)
    setattr(console, _TAG, True)
    root.addHandler(console)

    path = Path(paths.LOGS / "bandscribe.log" if logfile is None else logfile)
    file_error: OSError | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A plain FileHandler (no rotation): rotating renames fail on Windows while the UI and the CLI
        # both hold the file open.
        fh = logging.FileHandler(path, encoding="utf-8")
    except OSError as e:
        file_error = e
    else:
        fh.setLevel(logging.DEBUG)
        fh.addFilter(_OwnDebugFilter())
        fh.setFormatter(logging.Formatter(FILE_FORMAT))
        setattr(fh, _TAG, True)
        root.addHandler(fh)
        _logfile = path

    # Handlers do the filtering; the root must let DEBUG through for the file handler to see it.
    root.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    if file_error is not None:
        logging.getLogger(__name__).warning("로그 파일을 열 수 없어 콘솔에만 기록합니다: %s (%s)", path, file_error)


def get_logfile() -> Path | None:
    """Path of the active log file, or None when file logging is off."""
    return _logfile
