"""JobStore: stage keys, atomic stage_writer, crash safety, gc_temp, index (M0_SPEC §6, §15)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from gtab import atomic
from gtab.jobs import store as store_mod
from gtab.jobs.store import JobStore, StoreError, make_temp_dir, process_start_time

SONG = "f-0123456789abcdef"
KEY = JobStore.stage_key("sep", "1", {"model": "x"}, {"input": "a" * 64})


def _tmp_dirs(root: Path) -> list[Path]:
    return [p for p in root.rglob(".tmp-*") if p.is_dir()]


def _wait_for(pred, timeout: float = 30.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        time.sleep(0.02)


def _child_env() -> dict[str, str]:
    return dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs")


# ---------------------------------------------------------------------------- stage_key / layout


def test_stage_key_is_deterministic_and_order_independent():
    a = JobStore.stage_key("s", "1", {"x": 1, "y": {"b": 2, "a": [1, 2]}}, {"i": "h1", "j": "h2"}, {"m": "sha"})
    b = JobStore.stage_key("s", "1", {"y": {"a": [1, 2], "b": 2}, "x": 1}, {"j": "h2", "i": "h1"}, {"m": "sha"})
    assert a == b
    assert len(a) == 64 and int(a, 16) >= 0
    # models=None and {} describe the same recipe
    assert JobStore.stage_key("s", "1", {}, {}) == JobStore.stage_key("s", "1", {}, {}, {})


@pytest.mark.parametrize(
    "change",
    [
        dict(stage="t"),
        dict(code_version="2"),
        dict(params={"x": 2}),
        dict(params={"x": 1.0000001}),
        dict(params={"x": 1, "extra": None}),
        dict(inputs={"i": "h9"}),
        dict(inputs={"i": "h1", "k": "h2"}),
        dict(models={"m": "other"}),
    ],
)
def test_stage_key_is_sensitive_to_every_argument(change):
    base = dict(stage="s", code_version="1", params={"x": 1}, inputs={"i": "h1"}, models={"m": "sha"})
    assert JobStore.stage_key(**base) != JobStore.stage_key(**(base | change))


def test_stage_key_rejects_non_canonical_params():
    with pytest.raises(TypeError):
        JobStore.stage_key("s", "1", {"p": Path("C:/x")}, {})
    with pytest.raises(ValueError):
        JobStore.stage_key("s", "1", {"p": float("nan")}, {})


def test_layout(store: JobStore):
    jd = store.job_dir(SONG)
    assert jd == store.root / SONG and jd.is_dir()
    assert store.input_dir(SONG) == jd / "input"
    assert not store.input_dir(SONG).exists()  # ingest renames its temp dir onto it
    assert store.stage_dir(SONG, "sep", KEY) == jd / "stages" / "sep" / KEY[:12]
    assert store.list_jobs() == [SONG]


@pytest.mark.parametrize("bad", ["", "..", "a/b", "a\\b", "CON", "nul.txt", "x.", ".hidden", "한글", "a b"])
def test_invalid_names_rejected(store: JobStore, bad: str):
    with pytest.raises(ValueError):
        store.job_dir(bad)
    with pytest.raises(ValueError):
        store.stage_dir(SONG, bad, KEY)


@pytest.mark.parametrize("bad", ["", "abc", "ABCDEF0123456789", "../../0123456789ab", "g" * 64])
def test_invalid_keys_rejected(store: JobStore, bad: str):
    with pytest.raises(ValueError):
        store.stage_dir(SONG, "sep", bad)


# ---------------------------------------------------------------------------- stage_writer


def test_stage_writer_publishes_complete_dir_with_manifest(store: JobStore):
    assert not store.is_complete(SONG, "sep", KEY)
    with store.stage_writer(SONG, "sep", KEY, {"params": {"model": "x"}, "note": "한글 ok"}) as tmp:
        assert tmp.name.startswith(f".tmp-{KEY[:12]}-{os.getpid()}-")
        (tmp / "a.bin").write_bytes(b"\x00\x01" * 1000)
        (tmp / "stems").mkdir()
        (tmp / "stems" / "guitar.txt").write_text("gtr", encoding="utf-8")
        assert not store.stage_dir(SONG, "sep", KEY).exists()  # nothing visible before commit

    final = store.stage_dir(SONG, "sep", KEY)
    assert store.is_complete(SONG, "sep", KEY)
    m = store.load_manifest(SONG, "sep", KEY)
    assert m["stage"] == "sep" and m["key"] == KEY and m["status"] == "complete" and m["song_key"] == SONG
    assert m["note"] == "한글 ok" and m["params"] == {"model": "x"}
    assert m["created_utc"].endswith("Z") and m["duration_s"] >= 0
    assert m["outputs"] == {
        "a.bin": {"bytes": 2000, "sha256": hashlib.sha256(b"\x00\x01" * 1000).hexdigest()},
        "stems/guitar.txt": {"bytes": 3, "sha256": hashlib.sha256(b"gtr").hexdigest()},
    }
    assert (final / "stems" / "guitar.txt").read_text(encoding="utf-8") == "gtr"
    assert _tmp_dirs(store.root) == []


def test_is_complete_requires_matching_key_and_status(store: JobStore):
    with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
        (tmp / "x").write_bytes(b"x")
    other = KEY[:12] + "f" * 52  # same key12 prefix, different full key
    assert other != KEY and not store.is_complete(SONG, "sep", other)

    mpath = store.stage_dir(SONG, "sep", KEY) / "manifest.json"
    m = atomic.read_json(mpath)
    atomic.write_json(mpath, m | {"status": "partial"})
    assert not store.is_complete(SONG, "sep", KEY)
    mpath.write_text("{not json", encoding="utf-8")
    assert not store.is_complete(SONG, "sep", KEY)


def test_exception_inside_writer_leaves_nothing(store: JobStore):
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
            (tmp / "half.wav").write_bytes(b"\x00" * 4096)
            raise Boom("simulated failure mid-write")
    assert not store.stage_dir(SONG, "sep", KEY).exists()
    assert not store.is_complete(SONG, "sep", KEY)
    assert _tmp_dirs(store.root) == []


def test_keyboard_interrupt_inside_writer_leaves_nothing(store: JobStore):
    with pytest.raises(KeyboardInterrupt):
        with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
            (tmp / "x").write_bytes(b"x")
            raise KeyboardInterrupt
    assert not store.stage_dir(SONG, "sep", KEY).exists()
    assert _tmp_dirs(store.root) == []


def test_reserved_meta_and_manifest_name_rejected(store: JobStore):
    with pytest.raises(ValueError, match="reserved"):
        with store.stage_writer(SONG, "sep", KEY, {"status": "complete"}):
            pass
    with pytest.raises(ValueError, match="manifest.json"):
        with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
            (tmp / "manifest.json").write_text("{}", encoding="utf-8")
    assert not store.stage_dir(SONG, "sep", KEY).exists()
    assert _tmp_dirs(store.root) == []


def test_second_writer_for_existing_key_is_discarded(store: JobStore):
    with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
        (tmp / "v").write_text("first", encoding="utf-8")
    with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
        (tmp / "v").write_text("second", encoding="utf-8")
    assert (store.stage_dir(SONG, "sep", KEY) / "v").read_text(encoding="utf-8") == "first"
    assert _tmp_dirs(store.root) == []


def test_replace_swaps_existing_final(store: JobStore):
    with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
        (tmp / "v").write_text("first", encoding="utf-8")
    with store.stage_writer(SONG, "sep", KEY, {}, replace=True) as tmp:
        (tmp / "v").write_text("second", encoding="utf-8")
    final = store.stage_dir(SONG, "sep", KEY)
    assert (final / "v").read_text(encoding="utf-8") == "second"
    assert store.load_manifest(SONG, "sep", KEY)["outputs"]["v"]["sha256"] == hashlib.sha256(b"second").hexdigest()
    assert _tmp_dirs(store.root) == []


def test_broken_final_dir_is_replaced(store: JobStore):
    final = store.stage_dir(SONG, "sep", KEY)
    final.mkdir(parents=True)
    (final / "junk").write_text("no manifest here", encoding="utf-8")
    with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
        (tmp / "v").write_text("good", encoding="utf-8")
    assert store.is_complete(SONG, "sep", KEY)
    assert not (final / "junk").exists()
    assert _tmp_dirs(store.root) == []


def test_key12_collision_is_an_error_not_an_overwrite(store: JobStore):
    with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
        (tmp / "v").write_text("original", encoding="utf-8")
    other = KEY[:12] + "0" * 52
    with pytest.raises(StoreError, match="collision"):
        with store.stage_writer(SONG, "sep", other, {}) as tmp:
            (tmp / "v").write_text("impostor", encoding="utf-8")
    assert (store.stage_dir(SONG, "sep", KEY) / "v").read_text(encoding="utf-8") == "original"


def test_open_handle_at_commit_fails_cleanly(store: JobStore, monkeypatch):
    # Windows cannot rename a directory with an open file inside; that must surface as a clear
    # error, never as a published-but-incomplete stage dir.
    monkeypatch.setattr(store_mod, "_RENAME_RETRIES", 2)
    f = None
    try:
        with pytest.raises(StoreError, match="still open"):
            with store.stage_writer(SONG, "sep", KEY, {}) as tmp:
                f = open(tmp / "leaked.bin", "wb")
                f.write(b"x")
                f.flush()
    finally:
        if f is not None:
            f.close()
    assert not store.stage_dir(SONG, "sep", KEY).exists()
    for p in _tmp_dirs(store.root):  # rmtree could not run while the handle was open
        store_mod.rmtree_quiet(p)


# ---------------------------------------------------------------------------- concurrency

_RACE_CHILD = textwrap.dedent(
    """
    import logging, os, sys, time
    from pathlib import Path
    from gtab.jobs.store import JobStore

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    root, sync, n, i, key = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]

    def wait(pred, what):
        deadline = time.monotonic() + 60
        while not pred():
            if time.monotonic() > deadline:
                sys.exit(f"timeout waiting for {what}")
            time.sleep(0.005)

    store = JobStore(root)
    wait(lambda: (sync / "go").exists(), "go")
    with store.stage_writer("f-race", "sep", key, {"writer": i}) as tmp:
        (tmp / "who.txt").write_text(i, encoding="utf-8")
        (tmp / "payload.bin").write_bytes(os.urandom(256 * 1024))
        (sync / f"ready-{i}").touch()
        # every writer holds a finished temp dir before anyone commits -> a real rename race
        wait(lambda: len(list(sync.glob("ready-*"))) == n, "all writers ready")
    print("ok", i)
    """
)


def test_concurrent_processes_single_final_dir(tmp_path: Path):
    root, sync = tmp_path / "jobs", tmp_path / "sync"
    sync.mkdir()
    script = tmp_path / "race_child.py"
    script.write_text(_RACE_CHILD, encoding="utf-8")
    n = 4
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(root), str(sync), str(n), str(i), KEY],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            env=_child_env(),
        )
        for i in range(n)
    ]
    (sync / "go").touch()
    outs = [p.communicate(timeout=120) for p in procs]
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0, err  # every caller succeeds, winners and losers alike
        assert out.startswith("ok")
    # the race was real: all but one writer hit the "already completed" path
    assert sum("discarding ours" in err for _, err in outs) == n - 1

    store = JobStore(root)
    stage_parent = root / "f-race" / "stages" / "sep"
    assert [p.name for p in stage_parent.iterdir()] == [KEY[:12]]  # exactly one final, no temps
    m = store.load_manifest("f-race", "sep", KEY)
    final = store.stage_dir("f-race", "sep", KEY)
    winner = (final / "who.txt").read_text(encoding="utf-8")
    assert str(m["writer"]) == winner  # manifest and files come from the same writer
    assert m["outputs"]["payload.bin"]["sha256"] == atomic.sha256_file(final / "payload.bin")


def test_concurrent_threads_single_final_dir(store: JobStore, caplog):
    caplog.set_level(logging.INFO, logger="gtab.jobs.store")
    n = 8
    barrier = threading.Barrier(n, timeout=30)
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            with store.stage_writer(SONG, "sep", KEY, {"writer": i}) as tmp:
                (tmp / "who.txt").write_text(str(i), encoding="utf-8")
                barrier.wait()
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert errors == []
    assert sum("discarding ours" in r.getMessage() for r in caplog.records) == n - 1
    assert [p.name for p in (store.root / SONG / "stages" / "sep").iterdir()] == [KEY[:12]]
    m = store.load_manifest(SONG, "sep", KEY)
    assert (store.stage_dir(SONG, "sep", KEY) / "who.txt").read_text(encoding="utf-8") == str(m["writer"])


# ---------------------------------------------------------------------------- crash + gc_temp

_CRASH_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    from gtab.jobs.store import JobStore

    root, ready, key = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    with JobStore(root).stage_writer("f-crash", "sep", key, {}) as tmp:
        with open(tmp / "half.wav", "wb") as f:
            f.write(b"\\0" * 65536)
        ready.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(600)  # killed here, mid-stage
    """
)


def test_killed_writer_leaves_only_temp_and_gc_removes_it(tmp_path: Path):
    root, ready = tmp_path / "jobs", tmp_path / "ready.pid"
    script = tmp_path / "crash_child.py"
    script.write_text(_CRASH_CHILD, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(script), str(root), str(ready), KEY],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        env=_child_env(),
    )
    try:
        _wait_for(lambda: ready.exists() and ready.read_text(encoding="utf-8").strip(), 60, "child ready")
        # The venv python.exe is a launcher; kill the interpreter that actually holds the temp dir.
        pid = int(ready.read_text(encoding="utf-8"))
        os.kill(pid, signal.SIGTERM)  # TerminateProcess: no finally/except blocks run
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    _wait_for(lambda: process_start_time(pid) is None, 10, "writer pid to die")

    store = JobStore(root)
    assert not store.stage_dir("f-crash", "sep", KEY).exists()
    assert not store.is_complete("f-crash", "sep", KEY)
    temps = _tmp_dirs(root)
    assert len(temps) == 1 and temps[0].name.startswith(f".tmp-{KEY[:12]}-{pid}-")
    assert (temps[0] / "half.wav").exists()

    assert store.gc_temp() == 1
    assert _tmp_dirs(root) == []
    assert store.gc_temp() == 0


def test_gc_keeps_live_owner_and_unknown_names(store: JobStore):
    stage_parent = store.job_dir(SONG) / "stages" / "sep"
    live = make_temp_dir(stage_parent, KEY[:12])  # our own pid: alive
    odd = stage_parent / ".tmp-not-a-pid"
    odd.mkdir()
    input_tmp = make_temp_dir(store.job_dir(SONG), "input")
    assert store.gc_temp() == 0
    assert live.is_dir() and odd.is_dir() and input_tmp.is_dir()


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "import os; print(os.getpid())"], stdout=subprocess.PIPE, text=True)
    out, _ = p.communicate(timeout=60)
    pid = int(out.strip())
    _wait_for(lambda: process_start_time(pid) is None, 10, "pid to die")
    return pid


def test_gc_removes_dead_owner_at_every_level(store: JobStore):
    dead = _dead_pid()
    jd = store.job_dir(SONG)
    made = [
        store.root / f".tmp-job-{dead}-aa11",
        jd / f".tmp-input-{dead}-bb22",
        jd / "stages" / "sep" / f".tmp-{KEY[:12]}-{dead}-cc33",
        jd / "stages" / "sep" / f".tmp-old-{KEY[:12]}-{dead}-dd44",
    ]
    for p in made:
        (p / "sub").mkdir(parents=True)
        (p / "sub" / "f.bin").write_bytes(b"x")
    ro = made[2] / "readonly.bin"
    ro.write_bytes(b"r")
    os.chmod(ro, 0o444)  # read-only files must not block cleanup on Windows
    assert store.gc_temp() == len(made)
    assert _tmp_dirs(store.root) == []


def test_gc_detects_recycled_pid(store: JobStore, monkeypatch):
    # Our own pid is alive, but pretend the process holding it started after the temp dir was
    # made: then it cannot be the writer (Windows recycles pids aggressively).
    stage_parent = store.job_dir(SONG) / "stages" / "sep"
    tmp = make_temp_dir(stage_parent, KEY[:12])
    monkeypatch.setattr(store_mod, "process_start_time", lambda pid: time.time() + 3600)
    assert store.gc_temp() == 1
    assert not tmp.exists()


def test_gc_treats_unqueryable_process_as_alive(store: JobStore, monkeypatch):
    tmp = make_temp_dir(store.job_dir(SONG) / "stages" / "sep", KEY[:12])

    def denied(pid):
        raise PermissionError(5, "access denied")

    monkeypatch.setattr(store_mod, "process_start_time", denied)
    assert store.gc_temp() == 0
    assert tmp.exists()


def test_process_start_time():
    t = process_start_time(os.getpid())
    assert t is not None and time.time() - 7 * 86400 < t <= time.time() + 1
    assert process_start_time(_dead_pid()) is None


def test_gc_on_missing_root(tmp_path: Path):
    assert JobStore(tmp_path / "nope").gc_temp() == 0
    assert JobStore(tmp_path / "nope").list_jobs() == []


# ---------------------------------------------------------------------------- index


def test_index_roundtrip(store: JobStore):
    assert store.index_get("file:" + "a" * 64) is None
    store.index_put("file:" + "a" * 64, SONG)
    store.index_put("yt:dQw4w9WgXcQ", "yt-dQw4w9WgXcQ")
    assert store.index_get("file:" + "a" * 64) == SONG
    assert store.index_get("yt:dQw4w9WgXcQ") == "yt-dQw4w9WgXcQ"
    store.index_put("yt:dQw4w9WgXcQ", "yt-other")
    assert store.index_get("yt:dQw4w9WgXcQ") == "yt-other"
    assert json.loads((store.root / "index.json").read_text(encoding="utf-8")) == store.index_all()
    with pytest.raises(ValueError):
        store.index_put("file:x", "../escape")


def test_index_survives_corruption(store: JobStore):
    store.root.mkdir(parents=True)
    (store.root / "index.json").write_text("{truncated", encoding="utf-8")
    assert store.index_get("yt:x") is None
    store.index_put("yt:x", "yt-x")
    assert store.index_get("yt:x") == "yt-x"
    assert len(list(store.root.glob("index.corrupt-*.json"))) == 1


_INDEX_CHILD = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from gtab.jobs.store import JobStore

    root, go, i, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    while not go.exists():
        time.sleep(0.005)
    s = JobStore(root)
    for j in range(n):
        s.index_put(f"file:{i:02d}{j:062d}", f"f-{i:02d}{j:014d}")
        s.index_get(f"file:{(i + 1) % 4:02d}{j:062d}")  # interleave reads with other writers
    """
)


def test_index_concurrent_writers_lose_nothing(tmp_path: Path):
    root, go = tmp_path / "jobs", tmp_path / "go"
    script = tmp_path / "index_child.py"
    script.write_text(_INDEX_CHILD, encoding="utf-8")
    writers, per = 4, 25
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(root), str(go), str(i), str(per)],
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", env=_child_env(),
        )
        for i in range(writers)
    ]
    go.touch()
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    idx = JobStore(root).index_all()
    assert len(idx) == writers * per
    assert idx["file:" + "03" + "0" * 60 + "24"] == "f-03" + "0" * 12 + "24"
