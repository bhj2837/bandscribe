# M0 implementation spec (contracts between modules)

Source of truth for *what* M0 must achieve: `docs/DESIGN.md` §9 (code structure), §10 M0, §3 S0, §12.
This file fixes the *interfaces* so modules can be implemented in parallel. English for code; user-facing CLI text in Korean.

## 0. Ground rules (all modules)

- Python 3.12, `from __future__ import annotations`, full type hints, `logging.getLogger(__name__)`, no `print` outside `cli.py`.
- Repo root `D:\bandscribe` (= `paths.ROOT`). Package `bandscribe/` (flat layout). Tests in `tests/`.
- **Two environments.**
  - core: `D:\bandscribe\.venv` — orchestrator/CLI. **Never import torch** anywhere under `bandscribe/` except `bandscribe/workers/*`.
  - gpu: `D:\bandscribe\envs\gpu\.venv` — torch 2.11.0+cu128; has `bandscribe` installed editable (light deps only: numpy, soundfile, filelock, pydantic) plus nvidia-ml-py.
  - Modules imported by workers (`bandscribe/workers/*`, `bandscribe/sysmon.py`, `bandscribe/paths.py`, `bandscribe/atomic.py`) may only import stdlib + numpy/soundfile/filelock/pydantic (+ torch/pynvml inside workers, lazily).
- Commands: run tests with `D:\bandscribe\.venv\Scripts\python.exe -m pytest -q`. Sync envs with `D:\bandscribe\tools\uvw.cmd sync --extra core --group dev` run *from* `D:\bandscribe` (the wrapper keeps uv's python/cache on D:). Do **not** use the global Python 3.10 or plain `pip`.
- Windows pitfalls (from DESIGN §9.8 + user memory): console is cp949 → every subprocess gets `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`; open text files with `encoding="utf-8"`; wrap `sys.stdout` for UTF-8 **only once**, in `cli.main()`; all paths we create are ASCII; no `shell=True`; subprocess text I/O uses `encoding="utf-8", errors="replace"`; std `wave` cannot read float32 WAV → use `soundfile`.
- The Claude desktop app sandbox redirects writes under `%APPDATA%`/`%LOCALAPPDATA%` for processes it launches. Never place durable state there; everything lives under `D:\bandscribe`.
- Atomicity: any file others may read is written via `bandscribe.atomic.write_bytes/write_text/write_json` (temp file in same dir + `os.replace`). Directories via `JobStore.stage_writer` (temp dir + rename).
- Every module gets unit tests. Tests must not need network or GPU unless marked `@pytest.mark.network` / `@pytest.mark.gpu` (they are skipped automatically when unavailable — see `tests/conftest.py`).

## 1. `bandscribe/paths.py` (owner: A)

```python
BANDSCRIBE_ROOT: Path          # env BANDSCRIBE_ROOT or Path(__file__).resolve().parents[1]
DATA, TOOLS, MODELS, JOBS, HF_HOME, TORCH_HOME, UV_CACHE, DOWNLOADS, LOGS: Path  # DATA=BANDSCRIBE_ROOT/"data", DOWNLOADS=DATA/"downloads", LOGS=DATA/"logs"
GPU_PYTHON: Path         # BANDSCRIBE_ROOT/"envs/gpu/.venv/Scripts/python.exe"
CORE_PYTHON: Path        # BANDSCRIBE_ROOT/".venv/Scripts/python.exe"
YTDLP_EXE: Path          # TOOLS/"yt-dlp.exe" ; YTDLP_CONF = TOOLS/"yt-dlp.conf"
GPU_LOCK: Path           # DATA/"gpu.lock"
def child_env(extra: dict[str,str] | None = None) -> dict[str,str]
    # os.environ copy + HF_HOME, TORCH_HOME, UV_CACHE_DIR (all under DATA), PYTHONUTF8=1, PYTHONIOENCODING=utf-8,
    # PYTHONNOUSERSITE=1, BANDSCRIBE_ROOT ; then `extra`.
def apply_process_env() -> None   # sets the same vars in os.environ (called once by cli.main)
def ensure_dirs() -> None         # mkdir -p for DATA subdirs
```

## 2. `bandscribe/atomic.py` (owner: B)

`write_bytes(path, data)`, `write_text(path, text)`, `write_json(path, obj)` (utf-8, `ensure_ascii=False`, indent=1, sorted keys False), `read_json(path)`, `sha256_file(path, chunk=1<<20) -> str`, `sha256_bytes(b) -> str`.

## 3. `bandscribe/config.py` + `bandscribe/defaults.toml` (owner: A)

- Layers (later wins): package `defaults.toml` → `DATA/config.toml` (user) → job `hints.toml` (optional path) → `--set a.b=value` overrides (value parsed as TOML literal, fallback string).
- `--set` with an unknown top-level section is a `ConfigError` (CLI exit 2) with a difflib suggestion: on the command line it is a typo (`--set youtub.enabled=false` must not leave YouTube on). Unknown sections in config *files* stay lenient (kept in `Config.extra`).
- `ingest.sample_rate` is fixed at 44100; any other value is a `ConfigError` (the mix is `mix_44k_f32.wav` and the `f-<PCM sha256>` key is defined on a 44.1 kHz decode).
- `load_config(job_hints: Path|None=None, overrides: list[str]=()) -> Config` (pydantic v2 model, `extra="forbid"` on known sections; unknown top-level sections allowed and kept in `Config.extra`).
- `Config.get(dotted: str)` helper; `dump_effective(cfg) -> str` (TOML via tomli_w).
- defaults.toml must include at least:

```toml
[youtube]
enabled = true            # user request 2026-09-30: YouTube input ON by default (risk notice acknowledged, see data/config.toml consent)
format = "ba[format_id!*=drc][acodec=opus]/ba[format_id!*=drc]/ba"
cookies = false
po_token = false
timeout_s = 600

[ingest]
sample_rate = 44100
true_peak_ceiling_dbfs = -1.0
quasi_mono_side_mid_db = -30.0
lowpass_detect_hz = 16000

[gpu]
lock_timeout_s = 7200
worker_timeout_s = 7200
vram_margin_gb = 0.7
target_reserved_gb = 4.5
insufficient_vram_policy = "wait"   # wait | error | cpu

[doctor]
min_free_disk_gb = 60
min_node_major = 22
```

- `DATA/config.toml` is created by `ensure_user_config()` if missing, containing a commented header and:
```toml
[consent]
youtube = "2026-09-30 사용자 동의: YouTube 약관(다운로드 금지)과 JS 챌린지 해결의 기술적 보호조치 우회 위험(해외 판례, 한국법 미검증)을 고지받고 기본 켜기를 요청함. 개인 비공개 사용."
```

## 4. `bandscribe/log.py` (owner: A)

`setup_logging(level="INFO", logfile: Path|None=None)` — rich handler on console when available, plain file handler (utf-8) to `LOGS/bandscribe.log`. Idempotent.

## 5. `bandscribe/schema/` (owner: A) — score.json v0

pydantic models (JSON-schema exportable via `export_json_schema(path)`):
`Confidence{pitch,onset,part,fret: float|None}`,
`Note{onset_s: float, offset_s: float, pitch: int (MIDI), velocity: int|None, string: int|None, fret: int|None, techniques: list[str], confidence: Confidence, sources: list[str], part_posterior: dict[str,float], flags: dict[str,bool]}` (flags keys used later: shared,double,octave,variant,low_conf,unison_undecided),
`Line{id: str, name: str, kind: Literal["guitar","bass","reference","unassigned"], role_segments: list[{start_s,end_s,role}], tuning: list[int] (MIDI, low→high) | None, capo: int|None, notes: list[Note]}`,
`SongMeta{song_key, title, artist, duration_s, source_kind: Literal["file","youtube"], source_ref: str}`,
`Score{schema_version: Literal[0] = 0, song: SongMeta, lines: list[Line], created_utc: str, bandscribe_version: str}`.
Round-trip test: model → json → model equal.

## 6. `bandscribe/jobs/store.py` (owner: B)

```python
class JobStore:
    def __init__(self, root: Path = paths.JOBS)
    def job_dir(self, song_key: str) -> Path                      # root/song_key (create)
    def input_dir(self, song_key: str) -> Path                    # job_dir/"input"
    @staticmethod
    def stage_key(stage: str, code_version: str, params: dict, inputs: dict[str,str], models: dict[str,str] | None = None) -> str
        # sha256 hex of canonical JSON (sort_keys, separators=(",",":")) of all args
    def stage_dir(self, song_key, stage, key) -> Path             # job_dir/"stages"/stage/key[:12]
    def is_complete(self, song_key, stage, key) -> bool           # manifest.json exists and status=="complete" and manifest.key==key
    @contextmanager
    def stage_writer(self, song_key, stage, key, meta: dict) -> Iterator[Path]
        # yields a temp dir (sibling ".tmp-<key12>-<pid>-<rand>"); caller writes outputs;
        # on normal exit: write manifest.json {stage,key,status:"complete",created_utc,duration_s,outputs:{relpath:{bytes,sha256}}, **meta}
        #   then os.replace(temp, final). If final already exists & complete → discard temp (someone else won).
        # on exception: remove temp dir, re-raise. Never leaves a half-written final dir.
    def load_manifest(self, song_key, stage, key) -> dict
    def gc_temp(self) -> int   # remove stale .tmp-* dirs whose pid is not alive
```
- **Index** `root/"index.json"` guarded by `filelock` (`root/"index.lock"`): maps `"file:<content key>" → song_key`, `"yt:<videoId>" → song_key`. API: `index_get(key) -> str|None`, `index_put(key, song_key)`. The content key is `bandscribe.ingest.filekey.content_key(path)`: sha256 over the size and the sha256 of every 8 MiB chunk, hashed on a thread pool (every byte is hashed; plain sha256 at ~400 MiB/s broke the 1 s cache-hit goal for 165 MiB WAVs).

## 7. `bandscribe/jobs/dag.py` (owner: B)

```python
@dataclass
class Stage:
    name: str
    deps: list[str]
    run: Callable[["StageContext"], None]   # writes outputs into ctx.out_dir
    code_version: str = "0"
    params: dict = field(default_factory=dict)
    device: Literal["cpu","gpu"] = "cpu"
class CycleError(ValueError): cycle: list[str]
def topo_order(stages: list[Stage]) -> list[str]         # raises CycleError (with the cycle path) or KeyError for unknown dep
@dataclass
class StageContext: song_key, stage, key, out_dir: Path, dep_dirs: dict[str, Path], config, store
def run_dag(stages, song_key, store, config, until: str|None=None, force: set[str]=frozenset(), input_hashes: dict[str,str]|None=None) -> dict[str, Path]
    # stage key = JobStore.stage_key(name, code_version, params, {dep: dep_key...} | input_hashes)
    # skips complete stages (cache), runs others via store.stage_writer, returns {stage: final_dir}
```

## 8. `bandscribe/winjob.py` (owner: C) — ctypes only, no pywin32

```python
class JobObject:
    def __init__(self, kill_on_close: bool = True)   # CreateJobObjectW + SetInformationJobObject(JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    def assign(self, process_handle: int) -> None
    def active_processes(self) -> int                # QueryInformationJobObject(JobObjectBasicAccountingInformation).ActiveProcesses
    def process_ids(self) -> list[int]               # JobObjectBasicProcessIdList
    def terminate(self, exit_code: int = 1) -> None
    def track(self) -> list[int]                     # sample pids + open SYNCHRONIZE handles to new ones
    def wait_empty(self, timeout_s: float) -> bool   # poll active_processes()==0, THEN wait until every tracked
                                                     # process object is signaled (ActiveProcesses hits 0 up to
                                                     # ~130 ms before teardown/VRAM release finishes; measured 2026-09-30)
    def close(self) -> None                          # CloseHandle (kills remaining processes due to KILL_ON_JOB_CLOSE)
def run_captured(args, *, timeout_s: float | None, env=None, cwd=None, encoding="utf-8", errors="replace") -> CapturedRun
    # CapturedRun(args, returncode, stdout, stderr, timed_out, killed, duration_s, pids). The replacement for
    # subprocess.run(capture_output=True, timeout=) for anything that may be a launcher (yt-dlp.exe = PyInstaller
    # bootloader + child, py.exe, venv python.exe): runs in a kill-on-close job, drains pipes on threads, and on
    # timeout / Ctrl+C terminates the whole tree, so it returns within ~timeout_s + 10 s. Stragglers left after a
    # normal exit get 2 s, then are killed.
def spawn_in_job(args: list[str], job: JobObject, env: dict, cwd: Path, stdout, stderr) -> subprocess.Popen
    # Start with CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP, assign to job, then resume via ntdll.NtResumeProcess(handle).
    # Rationale: the venv python.exe is a *launcher* that spawns the base interpreter as a child (verified on this PC, DESIGN §9.5);
    # assigning before resume guarantees the child is born inside the job.
```

## 9. `bandscribe/sysmon.py` (owner: C) — importable from both envs, lazy imports

```python
def nvml_info() -> dict | None       # {driver, devices:[{index,name,total_mb,used_mb,free_mb,capability:(maj,min)}], processes:[{pid,name,used_mb|None}]}; None if NVML unavailable
def pdh_gpu_process_memory() -> dict[int, dict[str,int]] | None   # pid -> {"dedicated": bytes, "shared": bytes}
def pdh_gpu_adapter_memory() -> dict[str,int] | None              # {"dedicated": bytes, "shared": bytes} summed over adapters
```
- PDH via ctypes `pdh.dll`: `PdhOpenQueryW`, **`PdhAddEnglishCounterW`** (Windows is Korean-localized; English names still work), `PdhCollectQueryData`, `PdhGetFormattedCounterArrayW` with `PDH_FMT_LARGE`. Counters: `\GPU Process Memory(*)\Dedicated Usage`, `\GPU Process Memory(*)\Shared Usage`, `\GPU Adapter Memory(*)\Dedicated Usage`, `\GPU Adapter Memory(*)\Shared Usage`. Instance names look like `pid_1234_luid_0x..._phys_0` → parse pid, sum per pid.
- NVML via `pynvml` (package `nvidia-ml-py`); under WDDM per-process `usedGpuMemory` is often unavailable → `None`.

## 10. `bandscribe/gpu.py` (owner: C) — worker runner

```python
@dataclass
class WorkerResult:
    ok: bool; returncode: int | None; result: dict; work_dir: Path; stdout_path: Path; stderr_path: Path
    duration_s: float; timed_out: bool; killed: bool; lock_wait_s: float
    pid: int | None; job_pids: list[int]; lock_released_dirty: bool = False
def run_worker(name: str, request: dict, work_dir: Path, *, timeout_s: float | None = None,
               python: Path = paths.GPU_PYTHON, use_lock: bool = True, lock_timeout_s: float | None = None) -> WorkerResult
```
- Writes `work_dir/request.json`; runs `[python, "-m", f"bandscribe.workers.{name}", str(request.json), str(result.json)]` via `spawn_in_job`, env=`paths.child_env()`, stdout/stderr to `work_dir/{stdout,stderr}.log`.
- `use_lock`: acquire `filelock.FileLock(paths.GPU_LOCK)` **before** spawning; release only **after** `job.wait_empty()` returned True (after normal exit, timeout-terminate, or KeyboardInterrupt → terminate). This is the "one model on the GPU" guarantee (DESIGN §9.5).
- `timeout_s`: on expiry `job.terminate()` then `wait_empty(10)`; `timed_out=True, killed=True`.
- **Hard-killed orchestrator (added 2026-09-30):** Windows frees the lock file the instant its holder dies, while kill-on-close needs a few more ms to end the worker. The holder therefore keeps `data/gpu.lock.pids` (job pids, updated as they appear); a clean release deletes it. The next holder, right after acquiring the lock, waits (≤ 60 s) for any pids listed in a leftover sidecar to exit before spawning.
- `ok` = returncode==0 and result.json exists and result["ok"] is True and the release was clean.
- Trade-off (documented, deliberate): if processes survive `TerminateJobObject` for `DRAIN_TIMEOUT_S + DRAIN_EXTRA_S` (10 + 60 s; e.g. stuck in the driver during a TDR), the runner gives up and releases the lock anyway, so one hung kernel call cannot block every later GPU run until a reboot. This is never silent: an error is logged, `lock_released_dirty=True`, `ok=False`, and `bandscribe doctor` reports a failed `gpu.cleanup` check. Callers that start another GPU worker must stop on `lock_released_dirty`.

## 11. `bandscribe/workers/` (owner: C)

- `bandscribe/workers/base.py`: `worker_main(handler: Callable[[dict, WorkerContext], dict]) -> NoReturn`. Reads argv[1] request.json, runs handler, writes argv[2] result.json atomically:
  `{ok, worker, version, started_utc, duration_s, output: <handler dict>, error: str|None, traceback: str|None, python: sys.executable, base_executable: sys._base_executable, pid, ppid, cuda: {max_allocated_mb, max_reserved_mb} | None, sysmon: {nvml_used_peak_mb, pdh_self_dedicated_peak_mb, pdh_self_shared_peak_mb, pdh_self_shared_start_mb}}`.
  A background sampler thread (every 0.25 s) records NVML device used-memory peak and PDH dedicated/shared usage **for this process's pid**. torch stats are read only if torch was imported by the handler. Exit code 0 when ok else 1.
- `bandscribe/workers/selftest.py` handler modes (`request["mode"]`):
  - `"cuda"` (default): report `torch.__version__`, `torch.version.cuda`, `cuda_available`, device name, capability, total VRAM; run fp16 matmul (1024², check finite & close to fp32 within tolerance) and `torch.autocast("cuda", dtype=torch.float16)` conv2d; report bf16 support flag (expected False/emulated on Turing — do not use bf16).
  - `"hold"`: sleep `request["seconds"]` (for kill tests; no torch import).
  - `"spawn_grandchild"`: start a grandchild `python -c "import time; time.sleep(600)"` (same interpreter), write its pid to output file `request["pid_file"]`, then sleep `seconds` (no torch).
  - `"echo"`: return request (no torch).

## 12. `bandscribe/ingest/` (owner: D)

- `decode.py`
  - `probe(path) -> dict` via `ffprobe -v error -show_format -show_streams -of json` (codec_name, sample_rate, channels, bit_rate, duration, tags).
  - `decode_to_f32(src: Path, dst: Path, sr: int = 44100) -> None`: `ffmpeg -nostdin -v error -i SRC -map 0:a:0 -vn -af aresample=resampler=soxr:osr=SR -ac 2 -c:a pcm_f32le -f wav DST`. Never int16. Samples > 1.0 must survive (test with a loud synthetic master encoded to Opus 128k/AAC 128k by ffmpeg in the test).
  - `pcm_sha256(wav: Path) -> str`: sha256 of the float32 sample bytes (read with soundfile, dtype float32, always_2d) — independent of container/filename.
- `loudness.py`: `measure(wav) -> {integrated_lufs, true_peak_dbfs, sample_peak_dbfs}` — integrated via `pyloudnorm` (BS.1770-4) and true peak via ffmpeg `ebur128=peak=true` parsed from stderr summary (also parse its integrated value as `ffmpeg_integrated_lufs` for cross-check). `apply_ceiling(wav_in, wav_out, ceiling_dbfs) -> gain_db` (float gain so true peak ≤ ceiling; 0.0 if already below; writes float32).
- `flags.py`: `compute_flags(wav, probe_info, cfg) -> dict`: `quasi_mono` (10*log10(E_side/E_mid) < cfg), `side_mid_db`, `lowpass_hz_est` + `lossy_cutoff` (spectral energy above cutoff vs below), `nonstandard_sr` (source sr not in {44100,48000,88200,96000}), `drc` (youtube format_id contains "drc"), `mono_source` (source channels==1).
- `youtube.py`:
  - `parse_video_id(url) -> str|None` (watch?v=, youtu.be/, shorts/, music.youtube.com, embed/, with extra params).
  - `download(url, out_dir, cfg) -> (audio_path, info: dict)`: `[YTDLP_EXE, "--ignore-config", "--encoding", "utf-8", "--config-locations", YTDLP_CONF, "--no-playlist", "-f", cfg.youtube.format, "--write-info-json", "--no-write-comments", "--no-progress", "--no-cookies", "--no-cookies-from-browser", "-o", f"{out_dir}/%(id)s.%(ext)s", url]`, env=`paths.child_env()`, run via `winjob.run_captured` so `youtube.timeout_s` kills the whole yt-dlp tree (`--encoding utf-8`: the frozen exe ignores PYTHONUTF8 and would print cp949). Downloads are paced across processes: the end time of the last download is kept in `DOWNLOADS/.last_download` and the next one waits for the rest of `MIN_INTERVAL_S` (5 s). No cookies unless cfg says so (it doesn't in M0). Read `<id>.info.json`; return path of the downloaded audio and a trimmed info dict: id, title, track, artist(s), album, release_year, uploader, channel, duration, chapters, live_status, was_live, format_id, acodec, abr, asr, ext, description_tail (last 200 chars), is_art_track (description endswith "Auto-generated by YouTube.").
  - Errors → `YoutubeError` with the stderr tail and a Korean hint ("yt-dlp -U 또는 --update-to nightly, 실패하면 로컬 파일을 넣어 주세요").
- `__init__.py`: `ingest(source: str, *, store: JobStore, cfg) -> IngestResult(song_key, job_dir, meta: dict, cache_hit: bool)`:
  1. URL? (`http(s)://`): require `cfg.youtube.enabled` else raise with Korean message; video id → `index_get("yt:<id>")` and existing complete input → cache hit **without network**. Else download to `DOWNLOADS/<id>/`, song_key = `yt-<id>`.
  2. File: `content_key(path)` → `index_get("file:<key>")` and input complete with equal params → cache hit (**must return in < 1 s**, CLI start-up included, for a copy of the same file under another name; checked with a 165 MiB WAV in `tests/test_ingest.py`). Else decode to temp, `pcm_sha256` → song_key `f-<first16>`; if that job's input is already complete (same audio, other container) → cache hit too.
  3. Build `input/` atomically (temp dir + rename): `source.<ext>` (byte copy, never re-encoded), `mix_44k_f32.wav` (after `apply_ceiling`), `meta.json` {bandscribe_version, created_utc, song_key, params, source:{kind, ref, filename, stored_as, sha256, content_key, probe}, youtube: info|None, decode:{sr, pcm_sha256}, loudness: {..., gain_db}, flags}. Input is "complete" iff `input/meta.json` exists.
  3a. `params` = `ingest_params(cfg)` = {version (INGEST_VERSION), sample_rate, true_peak_ceiling_dbfs, quasi_mono_side_mid_db, lowpass_detect_hz}. Every cache hit (file index, PCM hash, yt index) requires `meta["params"] == ingest_params(cfg)`; otherwise the input is rebuilt from the job's own stored `source.<ext>` (no network; the job keeps its kind/ref/filename/youtube info) and the old `input/` is moved aside (`.tmp-old-input-*`) and replaced. Inputs without `params` (older builds) are rebuilt once.
  4. Update index for `file:<sha>` and/or `yt:<id>`.

## 13. `bandscribe/doctor.py` (owner: E)

`run_doctor(cfg) -> DoctorReport(checks: list[Check(name, ok: bool, detail: str, required: bool)])`; `DoctorReport.ok` = all required ok. Checks (DESIGN §10 M0):
1. gpu env python exists; `run_worker("selftest", {"mode":"cuda"})` → CUDA True, capability == (7,5) (warn-only if different GPU), fp16 matmul OK, autocast conv OK; report torch/cuda versions, VRAM, max_reserved.
2. gpu venv `python.exe` is a launcher: selftest result `python` vs `base_executable` → report the base interpreter path (this is where the NVIDIA per-program profile must be set).
3. ffmpeg present + `-buildconf` has `--enable-libsoxr` and `--enable-chromaprint`; `-filters` has `ebur128`.
4. Node ≥ `doctor.min_node_major` (`node --version`).
5. D: free ≥ `doctor.min_free_disk_gb`; `child_env()` HF_HOME/TORCH_HOME/UV_CACHE_DIR resolve under BANDSCRIBE_ROOT's drive.
6. NVML readable: driver, total/free VRAM, GPU processes list (names); PDH Shared Usage counter readable (adapter + per-process).
7. yt-dlp.exe exists, `--version`, config has `--js-runtimes node` (required only if youtube.enabled).
8. Global Python untouched (informational): `py -3.10 -c "import sys,torch;print(sys.version, torch.__version__)"` if `py` exists — never modify it.
Probes run external programs through `winjob.run_captured` (several are launchers), so their timeouts hold. If the selftest's `WorkerResult.lock_released_dirty` is True, a required `gpu.cleanup` check fails.
Non-required checks are warnings. The CLI exits non-zero if `report.ok` is False.

## 14. `bandscribe/cli.py` (owner: A, integrates others)

typer app `main()`:
- `bandscribe doctor [--json]` → table (rich), exit code 1 on failure.
- `bandscribe ingest <file|url> [--set k=v ...]` → prints song_key, cache hit, duration, LUFS/TP/gain, flags, job dir.
- `bandscribe status [song_key]` → list jobs (index + dirs) or show one job's meta and stage dirs.
- `bandscribe config show [--set ...]` → effective config TOML.
- `bandscribe gc` → `JobStore.gc_temp()`.
`main()` does: UTF-8 stdout wrap (once), `paths.apply_process_env()`, `paths.ensure_dirs()`, `config.ensure_user_config()`, `log.setup_logging()`.
Launcher `D:\bandscribe\bandscribe.cmd`: `@"%~dp0.venv\Scripts\python.exe" -m bandscribe.cli %*`.

## 15. Tests (each owner writes their own; integrator runs all)

- config layering & `--set` parsing; user config creation; schema round-trip + JSON schema export.
- store: stage_key determinism/sensitivity; stage_writer atomic (exception → no final dir; simulated crash leaves only `.tmp-*` which `gc_temp` removes); concurrent writers → single final; index concurrency.
- dag: topo order, CycleError path, cache skip on rerun, `force`, `until`.
- winjob/gpu (use `paths.CORE_PYTHON` as `python=` to avoid needing torch): hold + timeout → killed; spawn_grandchild → after terminate both pids dead (check via `OpenProcess`/`psutil`-free ctypes `GetExitCodeProcess` or `tasklist`); lock not released until job empty (second runner with `lock_timeout_s` small fails while first holds); normal echo round-trip; KeyboardInterrupt path if feasible.
- sysmon: returns dict or None without raising; PDH parses pid instance names (unit-test the parser).
- ingest: synthetic stereo signal → WAV/FLAC/MP3/M4A(AAC)/Opus via ffmpeg → decode → float32 stereo 44.1k; loud master (peaks ~+1 dBFS after lossy) keeps samples > 1.0 in `decode_to_f32`; `apply_ceiling` → TP ≤ −1.0 dBFS; pyloudnorm LUFS within ±0.1 LU of ffmpeg ebur128 integrated on a ≥ 10 s signal; same file copied under another name → cache hit < 1 s; interrupted ingest leaves no complete input; mono source flagged; quasi-mono flag; youtube id parser cases; a `@pytest.mark.network` test downloads a short Creative-Commons clip (e.g. Blender "Big Buck Bunny" trailer id `YE7VzlLtp-4`) — skipped offline.
- doctor: runs and returns a report; with GPU env present the selftest must pass (`@pytest.mark.gpu`).
