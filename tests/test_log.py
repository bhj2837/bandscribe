from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from gtab import log, paths


def _ours() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if getattr(h, "_gtab_handler", False)]


@pytest.fixture(autouse=True)
def _restore_root() -> Iterator[None]:
    root = logging.getLogger()
    level = root.level
    yield
    log.remove_handlers()
    root.setLevel(level)


def test_setup_is_idempotent_and_writes_utf8(tmp_path: Path) -> None:
    logfile = tmp_path / "logs" / "gtab.log"
    log.setup_logging("INFO", logfile)
    log.setup_logging("INFO", logfile)
    handlers = _ours()
    assert len(handlers) == 2
    assert sum(isinstance(h, logging.FileHandler) for h in handlers) == 1
    assert log.get_logfile() == logfile

    logging.getLogger("gtab.test").debug("디버그 [괄호] 한글")
    logging.getLogger("thirdparty").debug("library debug noise")
    logging.getLogger("thirdparty").info("library info")
    for h in handlers:
        h.flush()
    text = logfile.read_text(encoding="utf-8")
    assert "디버그 [괄호] 한글" in text  # gtab.* DEBUG reaches the file
    assert "library debug noise" not in text  # third-party DEBUG does not
    assert "library info" in text


def test_console_level_and_default_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "LOGS", tmp_path / "L")
    log.setup_logging("warning")
    console = [h for h in _ours() if not isinstance(h, logging.FileHandler)]
    assert [h.level for h in console] == [logging.WARNING]
    assert log.get_logfile() == tmp_path / "L" / "gtab.log"


def test_remove_handlers_releases_file(tmp_path: Path) -> None:
    logfile = tmp_path / "gtab.log"
    log.setup_logging(logfile=logfile)
    logging.getLogger("gtab.x").info("hello")
    log.remove_handlers()
    assert _ours() == [] and log.get_logfile() is None
    logfile.unlink()  # would fail on Windows if the handler still held the file open


def test_bad_level() -> None:
    with pytest.raises(ValueError):
        log.setup_logging("LOUD")
