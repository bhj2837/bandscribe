# M1a + M2 implementation spec (contracts between owners)

Source of truth for *what* must be achieved: `docs/DESIGN.md` — S2 (전역 분석), S3 (스템 분리), S3.5 (스템 1차 분석),
S6 (전사), §8 (평가 체계), §9 (코드 구조), §10 M1a / M2. This file fixes the *interfaces, file ownership and
evidence* so that five owners can work in parallel. It extends `docs/M0_SPEC.md`; every M0 rule still holds.
English for code/docs; every user-facing CLI string (help, progress, errors) is Korean.

Tags used below: **[검증 필요]** = the owner must confirm against the installed package / real file before
relying on it (and record the finding in the module docstring). **[잠정]** = starting value from DESIGN or this
spec, to be replaced by a measured/decided value. **[사용자]** = needs an action only the user can take.

Facts checked while writing this spec (2026-09-30, web/PyPI/this PC; items marked ✔ were re-checked against
the installed wheels / real files / GPU probes by the critique pass, scripts in `data/scratch/critic/`):
- ✔ PyPI `muscriptor` 0.3.0 (2026-08-05) requires `beat-this>=1.1`, `einops`, `fastapi`, `httpx`, `huggingface-hub`
  (2.0.0 imports fine), `mido`, `numpy>=1.24`, `packaging`, `python-multipart`, `safetensors`, `soundfile>=0.14.0`,
  `torch>=2.0`, `typer`, `uvicorn[standard]`.
  `TranscriptionModel.load_model(weights_path=None, device=None, dtype=None)`;
  `transcribe(audio: str|Path|tuple[Tensor,int], use_sampling=False, temperature=1.0, cfg_coef=1.0,
  instruments: list[str]|None=None, batch_size=None, no_eos_is_ok=True, beam_size=1, prelude_forcing=True)`
  yields `NoteStartEvent(pitch, start_time, index, instrument: str)` / `NoteEndEvent(end_time, start_event)` /
  `ProgressEvent(completed, total)` — **`ProgressEvent` is imported from `muscriptor.events`**. Tensor input is
  `[C, T]` float32; `.wav` paths go through stdlib `wave` (cannot read float32). Group table = §6.5 `MT3_GROUPS`
  (35 entries, ids 0–33 + drums 36).
- ✔ **MuScriptor dtype/device (closes old open question 7):** the model enables fp16 autocast **only if it is built
  with a CUDA device**; weights stay fp32, conditioners stay fp32. The model remembers its build device and moves
  audio, masks and prompts there, so `load_model(device="cpu")` + `.to("cuda")` gives device-mismatch errors (and
  would run plain fp32 even if patched). The upstream loader holds the model and a second copy of the weights on the
  GPU at once: measured load 10.3 s, peak 2343 MB allocated / 2354 MB reserved; transcribing a 30 s synthetic guitar
  clip: max allocated 1689 MB, max reserved 2360 MB, 3.1× realtime.
- ✔ PyPI `beat-this` 1.1.0: `Audio2Frames`, `Audio2Beats(checkpoint_path="final0", device="cpu", float16=False,
  dbn=False)` and `Postprocessor` exist; `__call__(signal, sr)`; signal `(samples,)` or `(samples, channels)`
  (mean-downmixed), resampled with soxr to 22050. Its minimal postprocessing already snaps every downbeat to the
  nearest beat; output times are quantised to 20 ms (50 fps). Named checkpoints go through
  `torch.hub.load_state_dict_from_url` (→ `TORCH_HOME`), which loads with `weights_only=False` (torch 2.11 default
  on that path); a **local file path** is loaded with `weights_only=True`. final0 URL:
  `https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt`. Depends on torchaudio, einops,
  rotary-embedding-torch, soxr.
- ✔ PyPI `synctoolbox` 1.4.2 requires `librosa>=0.11,<1`, `numba>=0.63,<0.66`, **`numpy>=2.0,<2.5`**,
  `music21>=9.9,<10`, `pandas>=2.3,<3`, `pretty_midi`, `libfmp>=1.3`, `matplotlib`, `scipy`, `soundfile`. Every
  function name used in §9.4 exists, including `sync_via_mrmsdtw_with_anchors(..., anchor_pairs=[(t_score,
  t_audio), ...])`. It imports `matplotlib.pyplot` at import time and imports sklearn without declaring it;
  `libfmp` 1.3 pulls IPython 8 into the core env → keep all of these imports lazy (§0).
  The core env currently has numpy 2.5.3 → it will be downgraded to 2.4.x by the lock (fine: gtab needs `>=2.0`).
- ✔ PyPI `pyguitarpro` 0.11 (2026-05-03), Python >=3.10, dep `attrs`, LGPL-3.0-only. String 1 = highest-pitched
  string; `Note.realValue` = tuning + fret **without** the capo (`Track.offset`); `guitarpro.parse` decodes strings
  as cp1252 by default.
- ✔ npm `@coderline/alphatab` 1.8.4 is the latest release (MPL-2.0); `importer.ScoreLoader.loadScoreFromBytes`
  exists. alphaTab numbers **string 1 = lowest-pitched** string and computes pitch = tuning + capo + fret. Headless
  audio in Node works through `AlphaSynth` (`MidiFileGenerator` → `new AlphaSynth(dummyOutput, ms)
  .exportAudio(options, midi, [], transpositions)`, which returns the exporter synchronously; soundfont from
  `dist/soundfont/`).
- ✔ PyPI `basic-pitch` 0.4.0: on Windows + Python <3.11 pulls `onnxruntime` (not TF); also librosa, mir-eval,
  pretty-midi, `resampy<0.4.3`, scikit-learn. Apache-2.0. onnxruntime 1.30 needs Python ≥ 3.11, so on 3.10 the lock
  resolves `onnxruntime<=1.23.2`, `numpy<=2.2.6`, `scipy<=1.15.3` (works; noted in the bp310 lock).
  **Minimum note length:** `predict()` converts ms → frames with `round(ms/1000 × 22050/256)` and
  `note_creation` keeps a note only if its length is **strictly greater** than that many frames (`<=` is dropped).
  So 46.4 ms (4 frames) keeps notes of ≥ 5 frames (≈ 58 ms); the upstream default 127.7 ms (11 frames) keeps ≥ 12
  frames (≈ 139 ms). `note_creation.model_output_to_notes(output, onset_thresh, frame_thresh, infer_onsets,
  min_note_len (frames), min_freq, max_freq, include_pitch_bends, multiple_pitch_bends, melodia_trick, ...)` works
  on a cached `inference.run_inference` output, so thresholds can be swept without re-running the model.
  Basic Pitch training data (paper + Spotify post): GuitarSet, iKala, MAESTRO, MedleyDB-Pitch, Slakh → **EGDB is
  clean for Basic Pitch**; MedleyDB(-Pitch) is **not** (matters for Tier B MedleyDB items).
- ✔ MSST (`ZFTurbo/Music-Source-Separation-Training`) HEAD = `84b1eac0887756b4f1a9d7a1ff49105939749ed2`
  (2026-09-26; that commit moved demix overlap-add buffers onto the GPU — we do **not** copy that, see §9.1).
  At that commit `models/bs_roformer/__init__.py` also imports the conformer and mel-band models, and the mel-band
  ones import librosa (not in the gpu env) → a trimmed `__init__.py` is a listed local patch (load verified with
  it). `attend.py` already uses `torch.nn.attention.sdpa_kernel` on the CUDA inference path; the CPU path still calls
  the deprecated `torch.backends.cuda.sdp_kernel` (warns on every call). `BSRoformer` defaults `zero_dc=True`
  (zeroes the DC bin of every stem).
- ✔ **SW checkpoint** loads with `torch.load(..., weights_only=True)` and loads **strictly** into the pinned
  `BSRoformer` (0 missing, 0 unexpected keys; 174.7 M fp32 parameters; the checkpoint stores `rotary_embed.freqs`
  keys → pin `rotary-embedding-torch==0.9.1`).
- ✔ **SW VRAM** (autocast fp16, batch 1, this PC): reserved peak 1200 / 1354 / 1796 MB for chunk 262144 / 352256 /
  588800; a 13.35 s chunk takes 1.07 s (≈ 6× realtime with overlap 2). CPU: a 5.94 s chunk takes 17.1 s (≈ 0.17×
  realtime → ≈ 23 min for a 4-min song). NVML free (4764 MB) and the worker's `mem_get_info` free (5112 MB)
  disagreed by ≈ 350 MB at about the same moment → never compare one source's need with the other's free (§8).
- **VRAM is not the binding limit on this PC unless a game is running:** with the realistic ~3.4 GB budget, SW's
  top rung (≈ 2.3 GB incl. context) and MuScriptor-medium (≈ 2.9 GB incl. context, upstream loader) both fit; b1 and
  b9 (≤ 4.5 GB reserved) are expected to pass with margin.
- SW yaml (`BS-Rofo-SW-Fixed.yaml`): `audio.chunk_size 588800`, `model.stft_hop_length 512`
  (588800 = 512×1150, 352256 = 512×688, 262144 = 512×512), `training.instruments =
  ['bass','drums','other','vocals','guitar','piano']` (**this is the output stem order**), `training.use_amp true`,
  `inference: batch_size 1, num_overlap 2, normalize false`, `model.flash_attn: true`, `!!python/tuple` tags.
  The `.audiosep.yaml` differs only in `inference.dim_t 801` (audio-separator derives 441×(801−1)=352,800 from it).
- MuScriptor medium is in `D:\gtab\data\hf\hub\models--MuScriptor--muscriptor-medium\snapshots\
  f32236969308476e01fd3aae67357de5feb05a2d\model.safetensors` (config: dim 1024, 24 layers).
- ✔ Zenodo (sizes, md5 and licences re-checked via the API): GuitarSet `3371780` (CC BY 4.0; `annotation.zip` 39,132,574 B md5 `b39b78e63d3446f2e54ddb7a54df9b10`,
  `audio_mono-mic.zip` 656,927,981 B md5 `275966d6610ac34999b58426beb119c3`, `audio_mono-pickup_mix.zip`
  683,145,360 B md5 `aecce79f425a44e2055e46f680e10f6a`, hex zips 3.2/3.6 GB); IDMT-SMT-Bass-Single-Track `7544099`
  (CC BY-NC-ND 4.0, `IDMT-SMT-BASS-SINGLE-TRACKS.zip` 20.5 MB, open); FiloBass `10069709` (CC BY 4.0,
  `FiloBass_v1.0.0.zip` 434,088,235 B md5 `ea1f52ffd492bf3654d720529f15bd9b`). EGDB: Google Drive folder
  `https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_` (license not stated).
  `www.cambridge-mt.com/ms/mtk/` answered **HTTP 403** to a plain fetch (Cloudflare). EGDB-PG (Zenodo `19789500`,
  CC BY 4.0) exists but is one 135 GB zip — not a practical replacement for EGDB. The EGDB paper's 80/5/15 split
  does not publish track ids.
- **libsndfile float WAVs are not byte-deterministic**: `soundfile.write(..., subtype="FLOAT")` adds a `PEAK`
  chunk with a timestamp (two writes of the same array 1.5 s apart differ; verified on this PC). CPU stages must
  write WAVs with `gtab.audio.write_f32` (§1.4), never `soundfile.write`.

---

## 0. Ground rules (additions to M0_SPEC §0)

- **Ownership is exclusive.** Each file below has exactly one owner. Do not edit a file you do not own; if you
  need a change there, put it under "integration requests" in your final report. New tests go in new files
  named after your area (`tests/test_<area>_*.py`).
- **torch boundary.** torch may be imported only (lazily) in `gtab/workers/*` and `gtab/third_party/msst/*`.
  Nothing in core (`gtab/**` otherwise) imports torch, beat_this, muscriptor, basic_pitch or onnxruntime.
  `tests/test_no_torch_in_core.py` (P0) enforces this by AST scan.
- **Stages depend only on `ctx.params`, dep dirs and the job input.** `ctx.config` may be read only for
  non-semantic knobs (timeouts, VRAM wait policy, lock timeouts). Everything that changes a stage's output must be
  in its params (so it is in the cache key, DESIGN §9.3).
- **CPU stage outputs are byte-deterministic** (DESIGN §8.7): no wall-clock timestamps, pids, absolute temp paths
  or dict-order accidents in any **data output** file; floats rounded where noted; WAVs via `gtab.audio.write_f32`;
  JSON via `gtab.atomic.write_json` (stable key order = insertion order → build dicts deterministically);
  MIDI via mido with explicit ordering. **Data outputs = every file in the stage dir except `manifest.json` and
  `_worker/**`** (provenance: `_worker/request.json` holds the random temp `out_dir`, `result.json` timestamps and
  pids, and M0's manifest hashes both). Determinism checks (tests, a9) compare data outputs only; helper
  `gtab.pipeline.runner.data_outputs(stage_dir) -> dict[rel, sha256]` (GRID) is the one implementation.
  Multithreaded numerics in CPU stages run with a pinned thread count (BLAS via `threadpoolctl`, onnxruntime via
  `SessionOptions`) so results do not depend on the machine's core count.
- **Workers never import `gtab.schema`** (keeps the bp310 worker Python-3.10-safe and decouples envs). They
  write plain JSON that matches the schema; the core-side stage function validates it with pydantic.
- **Nothing downloads during `gtab run`, and nothing downloads while the GPU lock is held.** Every model goes
  through `gtab models fetch` (§5, DESIGN §9.4) — including the Beat This! checkpoint, which the worker receives as
  a local path (never a hub name). All other downloads go through explicit commands (`gtab data fetch`, uv/npm
  installs). Workers set `HF_HUB_OFFLINE=1`. A stage whose model file is missing fails **before anything runs**
  (at plan time, §3.1) with a Korean message naming the fetch command; this also keeps stage keys stable (a model
  sha that appears after a first-run download would change the key of that stage and everything downstream).
- **Heavy imports stay lazy.** `gtab.cli` and every `gtab.commands.*` module import only typer/rich/stdlib at module
  level; librosa, synctoolbox, libfmp (pulls IPython 8), music21, pandas, sklearn, matplotlib, numba, mir_eval,
  pretty_midi and pyguitarpro are imported inside functions. Enforced by P0's `tests/test_import_weight.py`.
- **Never circumvent bot protection** (no UA spoofing, no Cloudflare solvers) when a dataset site refuses a plain
  download; fall back to the manual `import-local` path.
- **Licenses:** everything stays local; nothing is redistributed. Every downloaded model/dataset is recorded
  (models → `data/models/models.lock.json` via `gtab.models`; datasets → `data/datasets/datasets.lock.json` via
  `gtab.datasets`). MuScriptor weights CC BY-NC 4.0 + rights warranty (user accepted on HF); SW weights: unknown
  provenance, personal use only.
- **Printing:** `gtab/commands/*` are part of the CLI and may print (rich); everything else logs only (M0 rule).
  Long-running stage functions report progress through the runner's `progress` callback, not print.
- **Windows:** all M0 pitfalls apply (cp949, launcher venv python, ASCII paths, `winjob.run_captured` for every
  external program incl. `node`, no `shell=True`).
  - `npm` is `npm.cmd` (a batch file: needs a shell, broken argument escaping) → never launch it. Run
    `node.exe <nodejs dir>\node_modules\npm\bin\npm-cli.js ci` (nodejs dir = parent of `shutil.which("node")`) with
    env `npm_config_cache=<paths.NPM_CACHE>`, `npm_config_update_notifier=false`, `npm_config_fund=false`,
    `npm_config_audit=false` (npm's default cache is under `%LOCALAPPDATA%`, which the app sandbox redirects).
  - **Every generated file/folder name is ASCII `[a-z0-9_-]`** via `gtab.eval.runs.slug()` (EVAL; colon → `-`,
    other characters → `_`, empty → error). System ids like `job:amt_gtr`, `external:klangio` and item ids like
    `guitarset:00_BN1-129-Eb_comp` keep their original spelling *inside* JSON, never in paths (`a:b` is illegal on
    NTFS and even creates an alternate data stream). User-facing ids are ASCII (`--system other`, not `기타`).
  - `MPLBACKEND=Agg` is in every child env and the CLI process (P0, `paths._managed_env`): synctoolbox imports
    `matplotlib.pyplot` at import time and reports use matplotlib.
- Markers (P0 adds to `pyproject.toml` and auto-skips in `tests/conftest.py`):
  `gpu` (CUDA + gpu env, M0), `gpuenv` (gpu venv exists; CPU torch is enough), `bp310` (bp310 venv exists),
  `node` (node + `node/node_modules/@coderline/alphatab` exist), `data(name)` (dataset present in
  `datasets.lock.json`; e.g. `@pytest.mark.data("guitarset")`), `model(name)` (model file present and recorded in
  `models.lock.json`; e.g. `@pytest.mark.model("beat_this.final0")`), `network`, `slow`.
  Default `pytest -q` must stay green without GPU, network or datasets. The existing 370 tests stay green.

---

## 1. Phase P0 — integrator, before fan-out (serial, ~1–2 h)

P0 creates the shared contract layer. Owners start after P0 is committed. P0 owns these files for the whole
of M1a/M2 (later changes = integration requests to the integrator).

### 1.1 Dependencies

`pyproject.toml` `[project.optional-dependencies] core` add:
`librosa==0.11.*`, `mir_eval>=0.8,<0.9`, `synctoolbox==1.4.2`, `pyguitarpro==0.11`, `pretty_midi>=0.2.10`,
`mido>=1.3`, `music21>=9.9,<10`, `jinja2>=3.1`, `matplotlib>=3.10`, `pyyaml>=6`, `scikit-learn>=1.5`
(synctoolbox imports it without declaring it), `threadpoolctl>=3`, `pandas>=2.3,<3`. Then
`D:\gtab\tools\uvw.cmd lock` + `sync --extra core --group dev` from `D:\gtab`.
Expect numpy → 2.4.x (synctoolbox cap) and IPython 8 (via libfmp; harmless if imported lazily, §0). If numba has
no wheel for py3.12 + that numpy, pin `numpy<2.4` and note it.

`envs/gpu/pyproject.toml` add: `muscriptor==0.3.0`, `beat-this==1.1.0`, `einops`, `rotary-embedding-torch==0.9.1`
(SW checkpoint stores `rotary_embed.freqs`; strict load), `beartype`, `packaging`, `pyyaml`, `soundfile>=0.14`,
`soxr`, `safetensors`, `huggingface-hub`, `filelock`. **Not** `omegaconf` / `ml_collections` (only MSST's settings
loader needs them; we load the yaml ourselves, §9.1). muscriptor brings fastapi/uvicorn/python-multipart/typer —
unused but harmless. torch/torchaudio **must stay `2.11.0+cu128`** (check `uv.lock`: no CPU torch sneaks in via
muscriptor/beat-this; keep the `pytorch-cu128` index pins). `uvw.cmd lock` + `sync` from `D:\gtab\envs\gpu`.
Smoke: `envs\gpu\.venv\Scripts\python.exe -c "import torch,muscriptor,beat_this,einops,rotary_embedding_torch,
beartype;print(torch.__version__, torch.cuda.is_available())"` → `2.11.0+cu128 True`.

### 1.2 `gtab/paths.py` additions

```python
BP310_PYTHON: Path = GTAB_ROOT / "envs" / "bp310" / ".venv" / "Scripts" / "python.exe"
NODE_DIR: Path     = GTAB_ROOT / "node"
EVAL: Path         = DATA / "eval"          # tierA/, tierB/, external/, reftx_cache/
RUNS: Path         = DATA / "runs"
DATASETS: Path     = DATA / "datasets"
BENCH: Path        = DATA / "bench"
MODELS_LOCK: Path  = MODELS / "models.lock.json"
VRAM_TABLE: Path   = BENCH / "vram_table.json"
NPM_CACHE: Path    = DATA / "npm-cache"
```
Add EVAL, RUNS, DATASETS, BENCH, NPM_CACHE to `_DIRS`. `_managed_env()` gains `"MPLBACKEND": "Agg"` (reaches
workers, node children and — via `apply_process_env` — the CLI itself). `tests/test_cli.py::MANAGED_ENV` lists
the managed keys; P0 adds `MPLBACKEND` there (test_cli.py is already P0's file).

### 1.3 Config sections (`gtab/config.py` models + `gtab/defaults.toml`; keep `tests/test_config.py` sync test)

All new sections use `extra="forbid"`. Values are defaults **[잠정]** unless stated.

```toml
[run]
profile = "quality"            # fast | quality   (max = later milestones)

[hints]                        # user hints; CLI flags write these via --set
instruments = "auto"           # "auto" or comma list of families: guitar,keys,synth,bass,strings,brass,winds,vocals
parts = "auto"                 # used from M6 on; stored now

[sep]
backend = "sw_msst"
model_dir = "data/models/bs_roformer_sw"          # relative to GTAB_ROOT
ckpt = "BS-Rofo-SW-Fixed.ckpt"
config = "BS-Rofo-SW-Fixed.yaml"
ckpt_sha256 = "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
msst_commit = "84b1eac0887756b4f1a9d7a1ff49105939749ed2"
chunk_ladder = [588800, 352256, 262144]           # multiples of stft_hop_length 512 (DESIGN S3)
num_overlap = 2
batch_size = 1
dtype = "upstream"                                 # upstream (fp32 weights + AMP) | fp16 (E24-sep only)
input_lufs = "off"                                 # E1: "off" | -14.0 | -16.0 (gain before SW, inverse gain after)
leftover_warn_db = -20.0                           # used by the `stems` stage only
# (no allow_cpu here: CPU rungs are governed by gpu.insufficient_vram_policy + gpu.cpu_backends, §8.2)

[amt]
muscriptor_model = "medium"
muscriptor_revision = "f32236969308476e01fd3aae67357de5feb05a2d"
muscriptor_fallback = ["small"]                    # used only if model.safetensors is already in the HF cache
muscriptor_loader = "lean"                         # lean since 2026-10-03 (parity gate passed: identical notes, 1740 vs
                                                   # 2392 MB reserved, bench) | upstream (load_model on CUDA) (§7.3)
dtype = "upstream"                                 # upstream (fp32 weights + built-in fp16 autocast) | fp16 (E24-amt); never bf16
prelude_forcing = true
beam_size = 1
cfg_coef = 1.0
batch_size = 1
latency_s = -0.003                                 # latency experiment 2026-10-03 (docs/decisions.md); subtracted from MuScriptor times
piano_other_instruments = "keys+guitar"            # keys | keys+guitar  (see §12 A5)
guitar_view = "guitar_mono"                        # E2: guitar_mono | mix_mono | nonvox_mono | guitar_other_mono
guitar_mask = "guitar+present"                     # E3: guitar_only | guitar+present (3 guitar ∪ present keys/synth) | all
bp_onset_threshold = 0.6                           # E23 (2026-10-03, EGDB dev argmax among frames 3-7) tunes these three
bp_frame_threshold = 0.4
bp_min_note_frames = 7                             # shortest KEPT note in frames (1 frame = 256/22050 s = 11.61 ms);
                                                   # 4 = 46.4 ms, inside DESIGN's 3-5 frame start range. The worker
                                                   # passes min_note_len = frames - 1 (Basic Pitch drops len <= it).
                                                   # 12 frames (= upstream 127.7 ms default) is refused (DESIGN S6).
bp_min_freq_hz = 0.0                               # 0 = backend default
bp_max_freq_hz = 0.0
bp_melodia_trick = true
bp_threads = 4                                     # onnxruntime intra-op threads (pinned: output must not depend on core count)

[instr]
threshold = 0.5                                    # E3
guitar_threshold = 0.15                            # low: omitting a guitar class deletes its notes
guitar_stem_active_ratio = 0.05                    # guitar stem active in >=5 % of bars -> all 3 guitar classes
active_rel_db = -40.0                              # a stem/bar is "active" above this RMS relative to the mix
mix_class_min_notes = 8

[grid]
checkpoint = "final0"                              # model name "beat_this.final0" in models.lock.json (gtab models fetch)
dbn = false                                        # true is rejected at config load (needs madmom, not installed)
refine_window_ms = 35.0                            # Beat This! times are quantised to 20 ms, so refinement matters
half_double_check = true
compound_triple_ratio = 0.6                        # E14
compound_margin = 0.1
downbeat_snap_ms = 70.0
meter_smooth_bars = 8

[sections]
min_section_bars = 4
max_sections = 16
repeat_min_score = 0.6
align_max_offset_ms = 50.0

[s35]
vocal_active_rel_db = -30.0
resid_min_occurrences = 3

[eval]
tpb = 48
onset_tol_s = 0.05
bootstrap_iterations = 10000
seed = 20260930
mde_points = 1.0                                   # default MDE (F1 points); costly elements use 1.5 (DESIGN 8.1)

[datasets]
root = "data/datasets"
```

`[gpu]` gains: `wait_poll_s = 30`, `wait_max_s = 3600`, `max_worker_retries = 3` (worker answered
`insufficient_vram` or every GPU rung failed → wait and retry at most this often), `worker_headroom_mb = 256`
(worker-side check, §8.1), `ctx_mb_default = 150` (CUDA context estimate until the bench measures it; was 500
**[잠정]**, lowered 2026-10-03 after workers measured 71–101 MB),
`shared_growth_fail_mb = 64` (**[잠정]**; calibrated from fallback-off bench runs, §9.1), `rtf_fail_ratio = 0.5`,
`cpu_backends = ["beats_beatthis"]` (backends that may use a CPU rung, and only under policy `"cpu"`; DESIGN §12
"작은 모델만 CPU"). Existing `vram_margin_gb` (0.7) is redefined as headroom for *other apps' fluctuation and
fragmentation only* — the CUDA context is counted separately (§8.1); `target_reserved_gb`,
`insufficient_vram_policy` stay. All `[gpu]` keys are non-semantic (not in any stage key); the rung actually used
is recorded per stage instead (§3.1 `gpu_run.json`).
Config validation (P0): `grid.dbn = true` → ConfigError `"grid.dbn=true 는 madmom 이 필요해 지원하지 않습니다"`;
`amt.dtype`/`sep.dtype` accept only `upstream|fp16`; `sep.input_lufs` is `"off"` or a float in [−30, −6];
`amt.bp_min_note_frames` int in [1, 30].

### 1.4 `gtab/audio.py` (importable in all three envs: stdlib + numpy + soundfile, Python-3.10-safe)

```python
SR = 44100
def info(path) -> tuple[int, int, int]                         # (frames, samplerate, channels) via soundfile.info
def iter_blocks(path, block_frames: int = 1 << 18) -> Iterator[np.ndarray]   # float32, always 2-D (n, ch)
def read_f32(path) -> tuple[np.ndarray, int]                   # whole file, float32 (n, ch) — only for small files
def write_f32(path, data: np.ndarray, sr: int = SR) -> None
    # Deterministic RIFF/WAVE IEEE-float writer: 'fmt ' (tag 3, 18-byte), 'fact', 'data'. No PEAK chunk.
    # Writes to a temp file in the same dir, then os.replace. data (n,) or (n, ch) float32.
class F32Writer:  # streaming variant, same format; header patched on close; atomic on close
    def __init__(self, path, sr: int, channels: int); def write(self, block: np.ndarray); def close(self)
def to_mono(x: np.ndarray) -> np.ndarray                        # (L+R)/2 float32 (n,)
def pcm_sha256(path) -> str                                     # sha256 over float32 samples (same as ingest.decode)
def rms_db(x) -> float                                          # 10*log10(mean(x^2)+1e-20)
```
Test: two `write_f32` of the same array (1 s apart) are byte-identical; soundfile reads back bit-exact;
values > 1.0 survive; `F32Writer` equals `write_f32`.

### 1.5 `gtab/models.py` (model registry; core + workers)

```python
@dataclass class ModelEntry: name: str; path: str; url: str; bytes: int; sha256: str; license: str;
                             source: str; notes: str = ""; recorded_utc: str = ""
def get(name: str) -> ModelEntry | None
def record(entry: ModelEntry) -> None           # filelock(MODELS_LOCK + ".lock") + atomic write; replaces same name
def sha256_cached(path: Path) -> str            # memo keyed by (abs path, size, mtime_ns) in data/models/.sha_cache.json;
                                                # read-modify-write under filelock(".sha_cache.json.lock") — core AND
                                                # workers write it; a corrupt cache file is discarded, never fatal
def verify(name: str) -> bool                   # recompute sha256, compare
def local_path(name: str) -> Path               # GTAB_ROOT / entry.path; raises ModelMissing if unrecorded or absent
class ModelMissing(RuntimeError)                # .name, .hint_ko e.g. "Beat This! 체크포인트가 없습니다. 먼저
                                                # `gtab models fetch beat_this.final0` 를 실행하세요."
```
Stage `models(cfg)` functions (§3.1) call `local_path` / `sha256_cached`; `ModelMissing` raised there surfaces at
plan time (runner, §4), before any stage runs. `gtab.models` imports stdlib + filelock only (Python-3.10-safe,
importable in all three envs).
`models.lock.json` = `{"format": "gtab.models/1", "models": {name: ModelEntry-as-dict}}`. P0 seeds:
`bs_roformer_sw.ckpt` (sha from config, url `"unknown (original uploader account deleted; local copy)"`,
license `"unknown - personal use only"`), `bs_roformer_sw.yaml`, `bs_roformer_sw.audiosep.yaml`,
`muscriptor-medium` (`hf://MuScriptor/muscriptor-medium@f322369…`, sha of model.safetensors,
`"CC BY-NC 4.0 + HF gated conditions (rights warranty / indemnity), accepted by the user's HF account"`).
`beat_this.final0` is **not** seeded (not on disk yet); DATA's `gtab models fetch` records it (§9.5).

### 1.6 Schemas (`gtab/schema/*.py`; pydantic v2, `extra="forbid"`, `allow_inf_nan=False`; Python-3.10-safe)

Formats are defined in §6; P0 writes the models exactly as listed there plus round-trip tests:
`notes.py` (§6.1), `grid.py` (§6.2 incl. `TempoMap`), `sections.py` (§6.3), `instrumentation.py` (§6.5 incl.
`MT3_GROUPS`, `FAMILIES`), `gt.py` (§6.7), `dataset.py` (§6.8), `evalio.py` (§6.9). Export them from
`gtab/schema/__init__.py`. Every file-level model carries `format: Literal["gtab.<name>/1"]` (a field named
`schema` would shadow `BaseModel.schema`).

### 1.7 Packages, CLI wiring, test infra

- Empty (docstring-only) `__init__.py` for: `gtab/commands`, `gtab/analysis`, `gtab/sep`, `gtab/amt`, `gtab/eval`,
  `gtab/datasets`, `gtab/datasets/loaders`, `gtab/pipeline`, `gtab/third_party`.
- `gtab/commands/{run,bench,eval,gt,data,models}.py` stubs: each exports `register(app: typer.Typer) -> None` that
  adds its command(s) with the Korean help from §5 and a body `raise typer.Exit(_not_ready())`. Owners replace the
  whole file. Command modules import only typer/rich/stdlib at module level (M0 lazy-backend pattern).
- **Stub stage modules** `gtab/analysis/stage.py`, `gtab/sep/stage.py`, `gtab/amt/stage.py`, each
  `STAGES: dict[str, dict] = {}` plus a docstring pointing to §3.2. GRID/SEP/AMT replace their file wholesale.
  GRID's `graph.py` imports these lazily and turns a missing stage name into a Korean error
  (`"단계 '<name>' 가 아직 구현되지 않았습니다 (<module>)"`), so the graph lands before its providers.
- `gtab/cli.py`: after the existing commands, `for mod in ("run","bench","eval","gt","data","models"):
  _register_optional(app, f"gtab.commands.{mod}")` — import failure registers a placeholder group/command that
  prints `"<name> 명령을 불러오지 못했습니다: <error>"` and exits 1, so one broken module never breaks `gtab`.
- `tests/conftest.py` markers of §0; `tests/test_no_torch_in_core.py`.
- `tests/test_py310_compat.py`: `ast.parse(src, feature_version=(3, 10))` for every module a bp310 worker can
  import — `gtab/__init__.py`, `gtab/workers/__init__.py`, `gtab/workers/base.py`, `gtab/atomic.py`,
  `gtab/sysmon.py`, `gtab/paths.py`, `gtab/audio.py`, `gtab/models.py` — plus every `gtab/**/*.py` whose first
  5 lines contain `# gtab: py310` (AMT marks `amt_basicpitch.py`; the M0 versions pass today). Parse only — the
  `bp310`-marked worker test covers real imports on 3.10. SEP's additive `base.py` change must keep this green.
- `tests/test_import_weight.py`: in a fresh subprocess, `import gtab.cli` and every `gtab.commands.*` module, then
  assert none of `torch, librosa, synctoolbox, libfmp, IPython, music21, pandas, sklearn, matplotlib, numba,
  mir_eval, pretty_midi, guitarpro` is in `sys.modules` (CLI start-up stays fast; §0).
- `.gitignore`: add `node/node_modules/` (already present) — nothing else; `data/` stays ignored.

---

## 2. Owners and exclusive files

| Owner | Scope (DESIGN) | Files (exclusive) |
|---|---|---|
| **P0** integrator | contracts, deps, wiring | §1: `pyproject.toml`, `uv.lock`, `envs/gpu/pyproject.toml`, `envs/gpu/uv.lock`, `gtab/config.py`, `gtab/defaults.toml`, `gtab/paths.py`, `gtab/cli.py`, `gtab/audio.py`, `gtab/models.py`, `gtab/schema/{notes,grid,sections,instrumentation,gt,dataset,evalio}.py`, `gtab/schema/__init__.py`, all new empty `__init__.py`, **initial stubs** of `gtab/commands/{run,bench,eval,gt,data,models}.py` and `gtab/{analysis,sep,amt}/stage.py` (ownership passes to the owner named below on their first commit), `tests/conftest.py`, `tests/test_config.py`, `tests/test_audio.py`, `tests/test_models.py`, `tests/test_schema_m2.py`, `tests/test_no_torch_in_core.py`, `tests/test_py310_compat.py`, `tests/test_import_weight.py`, `tests/test_cli.py` (registration + `MANAGED_ENV`) |
| **SEP** | S3, VRAM, bench (M2) | `gtab/third_party/msst/**`, `gtab/workers/sep_msst.py`, `gtab/workers/gpu_common.py`, `gtab/workers/vram_ballast.py`, `gtab/workers/base.py` (additive, stays py3.10-safe), `gtab/vram.py`, `gtab/sep/{stage,run,derive,archive,checks,experiments}.py`, `gtab/bench.py`, `gtab/commands/bench.py`, `gtab/doctor.py` (two additive checks), `tests/test_sep_*.py`, `tests/test_vram.py`, `tests/test_gpu_common.py`, `tests/test_bench.py` |
| **AMT** | S6 backends, S3.5 v0 transcription + instrumentation, MIDI, **all transcription-scored experiments** (E1, E2, E3, E23, E24-sep, E24-amt, latency, gate-m2) | `envs/bp310/{pyproject.toml,uv.lock}`, `gtab/workers/amt_muscriptor.py`, `gtab/workers/amt_basicpitch.py`, `gtab/amt/{stage,cache,instruments,instrumentation,latency,midi,reftx,experiments}.py`, `tests/test_amt_*.py` |
| **GRID** | S2 (beats→grid, sections, repeat map), S3.5 vocal activity + repeat residual pass 1, pipeline graph + `gtab run` | `gtab/workers/beats_beatthis.py`, `gtab/analysis/{onsets,grid,sections,repetition,vocal,stage}.py`, `gtab/pipeline/{graph,publish,runner}.py`, `gtab/commands/run.py`, `tests/test_grid_*.py`, `tests/test_sections_*.py`, `tests/test_s35_*.py`, `tests/test_pipeline_*.py` |
| **EVAL** | §8 harness (M1a), GT tools, synthetic remix, decisions/contamination | `gtab/eval/**` (incl. `templates/`, `runs.py` with `slug()`), `gtab/analysis/stereo.py`, `gtab/commands/{eval,gt}.py`, `node/{package.json,package-lock.json,gtab_score.mjs}`, `docs/decisions.md`, `docs/eval/contamination.md`, `tests/test_eval_*.py`, `tests/fixtures/gp_make.py` |
| **DATA** | Tier B/C registry, fetch, loaders, **model fetch** | `gtab/datasets/{registry.toml,registry.py,fetch.py,store.py}`, `gtab/datasets/loaders/{guitarset,idmt_bass,egdb,filobass,cambridge_mt,medleydb}.py`, `gtab/datasets/__init__.py` (public API, replaces P0 stub), `gtab/model_fetch.py`, `gtab/model_sources.toml`, `gtab/commands/{data,models}.py`, `tests/test_datasets_*.py`, `tests/test_model_fetch.py` |

Cross-owner dependencies are **only through the APIs in this spec** (arrow = "imports / calls"):

| From → To | API |
|---|---|
| AMT, GRID → SEP | `gtab.workers.gpu_common`, `gtab.vram` (incl. `write_gpu_run`/`read_gpu_run`) |
| AMT → SEP | `gtab.sep.run.separate(...)` (E1, E24-sep; §9.1) |
| AMT, SEP → EVAL | `gtab.eval.metrics`, `gtab.eval.stats`, `gtab.eval.remix` (`tierb_mix`, `make_scene(full_band=True)`) |
| SEP → EVAL | `gtab.analysis.stereo` |
| EVAL → AMT | `gtab.amt.reftx`, `gtab.amt.experiments.EXPERIMENTS` |
| EVAL → SEP | `gtab.sep.experiments.EXPERIMENTS` (`stereo-preservation` only) |
| EVAL, AMT → DATA | `gtab.datasets` public API |
| DATA → P0 | `gtab.models.record/verify` (model fetch) |
| GRID → SEP, AMT, GRID | `STAGES` dicts (lazy import, §1.7) |
| GRID → P0 | `gtab.models.local_path` (Beat This! checkpoint) |

**Merge order** (each step lands on `main` before the next step may depend on it; owners work in parallel inside
a step and against stubs/fakes for later steps):
1. **P0** (contracts, stubs, deps).
2. **Foundations** (first commit of each owner, within its first hours): SEP `vram.py` + `gpu_common.py` + base.py
   additions; EVAL `metrics.py` + `stats.py` + `runs.py`; DATA `registry/fetch/store` + `gtab.datasets` API +
   `gtab models fetch`. Until then consumers stub locally in tests.
3. **Components**: SEP (vendored MSST, worker, stages, `sep.run`), AMT (workers, stages, reftx), GRID (beats worker,
   analysis, graph/runner/publish), EVAL (remix, GT tools, suites, report), DATA (loaders).
4. **Experiments & gates** (E1–E3, E23, E24-*, latency, gate-m2, stereo-preservation, bench runs): need steps 2–3
   of EVAL and DATA plus the backend they measure.

---

## 3. Stage graph for `gtab run <file|url|job> --until notes`

### 3.1 Stage contract

Each owner module exports
```python
STAGES: dict[str, dict] = {
  "<stage>": {"run": Callable[[StageContext], None], "code_version": "1",
              "params": Callable[[Config], dict], "device": "cpu"|"gpu",
              "models": Callable[[Config], dict[str, str]]},   # model name -> sha256 (gtab.models.sha256_cached)
}
```
`gtab/pipeline/graph.py` (GRID) owns **names and deps** (table below) and builds `list[Stage]` by importing
`gtab.analysis.stage`, `gtab.sep.stage`, `gtab.amt.stage`. `input_hashes = {"input.mix": meta["decode"]["pcm_sha256"],
"input.ingest": sha256(canonical_json(meta["params"]))}`. Stage key = M0 `JobStore.stage_key` (name + code
version + params + dep keys + input hashes + model shas). The mix is `store.input_dir(song_key)/"mix_44k_f32.wav"`.
Profile and hints enter only through the params of the stages that use them.

`models(cfg)` may raise `gtab.models.ModelMissing`; the runner calls every selected stage's `models` at plan time
and reports all missing models at once (exit 1) before running anything (§0).

GPU stages call `gtab.vram.run_gpu_stage(...)` (§8.1), which does `wait_for_budget` and then
`gtab.gpu.run_worker(name, request, work_dir=ctx.out_dir/"_worker", timeout_s=cfg.gpu.worker_timeout_s,
lock_timeout_s=cfg.gpu.lock_timeout_s)`, retrying on `insufficient_vram`. The worker writes its data
files into `ctx.out_dir` (paths in the request); `_worker/{request,result}.json` and logs stay in the stage dir
as provenance. A failed worker (`ok=False` or `lock_released_dirty=True`) raises `StageError` with a Korean
message naming `_worker/stderr.log`; the DAG publishes nothing for that stage (M0).

**Rung provenance (the rung is not in the key).** Every GPU stage writes `gpu_run.json` at its stage root via
`gtab.vram.write_gpu_run(out_dir, backend, worker_output)`:
`{"format":"gtab.gpu_run/1","backend","ladder":[labels],"rung":{"index","label","device"},"degraded": bool,
"waited_s"}` with `degraded = rung.index > 0 or device == "cpu"` (relative to the configured ladder). Output
bytes differ by chunk size, medium vs small and GPU vs CPU, but the key holds only the ladder — so a result made
while a game was running would otherwise be reused forever. The runner reads `gpu_run.json` of every **cached**
GPU stage and, if `degraded`, prints `"<stage>: 낮은 칸(<label>)으로 만든 캐시입니다. 여유가 있을 때
'gtab run <곡> --force <stage>' 로 다시 만드세요."` (warning, not failure). Experiments never use
`gtab run` caches for measured backends: they pass `force_rung` and write the rung into their metric rows (§6.10).

### 3.2 Stages (list order = DAG order; GPU order is beats → sep → amt_ms1 → amt_gtr, DESIGN §9.5)

| # | Stage | Owner | Deps | Device / env / worker | Params (keys) | Outputs in stage dir |
|---|---|---|---|---|---|---|
| 1 | `beats` | GRID | – | gpu / gpu env / `beats_beatthis` | `grid.checkpoint`, `grid.dbn`, float16=false; models: `beat_this.final0` sha (fetched beforehand) | `beats.json` (§6.2 BeatsRaw), `activations.npz` (beat/downbeat logits, fps 50), `gpu_run.json` |
| 2 | `grid` | GRID | beats | cpu / core | `grid.*` | `grid.json` (§6.2), `onsets.npz` (percussive onset envelope + onset times) |
| 3 | `sections` | GRID | grid | cpu / core | `sections.*` | `sections.json`, `repeat_map.json` (§6.3) |
| 4 | `sep` | SEP | – | gpu / gpu env / `sep_msst` | `ckpt_sha256`, `config_sha256`, `msst_commit`, `chunk_ladder`, `num_overlap`, `batch_size`, `dtype`, `input_lufs` (nothing else — `leftover_warn_db`, `model_dir`, VRAM knobs do not change the stems); models: ckpt + yaml sha | `stems/{bass,drums,other,vocals,guitar,piano}.wav` (stereo f32 44.1k), `sep.json` (§6.4), `gpu_run.json` |
| 5 | `stems` | SEP | sep | cpu / core | `sep.leftover_warn_db` | `nonvox.wav`, `leftover.wav` (stereo), `practice/mix_minus_guitar.wav`, `practice/mix_minus_bass.wav` (stereo), `views/{mix,guitar,bass,piano_other,nonvox,guitar_other}_mono.wav`, `views.json`, `stem_stats.json`, `energy.npz` (§6.4) |
| 6 | `vocal` | GRID | stems, grid, sep (reads `stems/vocals.wav`; integration 2026-10-03) | cpu / core | `s35.vocal_active_rel_db` | `vocal_activity.json` (§6.6) |
| 7 | `resid1` | GRID | stems, grid, sections | cpu / core | `s35.resid_min_occurrences` | `repeat_resid_pass1.wav` (mono f32), `resid1.json` (§6.6) |
| 8 | `amt_ms1` | AMT | stems | gpu / gpu env / `amt_muscriptor` | `amt.*` MuScriptor keys (`muscriptor_model`, `revision`, `fallback`, `loader`, `dtype`, decode keys, `piano_other_instruments`), `run.profile`, view list; models: medium sha | `raw/{mix_mono,bass_mono,piano_other_mono}__muscriptor.json` (NoteSet; `mix_mono` only in `quality`), `gpu_run.json` |
| 9 | `amt_bp` | AMT | stems | cpu / bp310 / `amt_basicpitch` (no GPU lock) | `amt.bp_*` incl. `bp_threads`; models: bundled ONNX sha, or `{"basic_pitch": "absent"}` when the bp310 env is missing (key changes once it is installed) | `raw/{guitar_mono,bass_mono}__basicpitch.json` (NoteSet with bends); **optional**: without bp310 it writes `skipped.json` (`{"reason":"bp310 env missing"}`) and a Korean warning, and downstream stages treat Basic Pitch outputs as absent (DESIGN: auxiliary) |
| 10 | `instr` | AMT | amt_ms1, stems, sections, grid | cpu / core | `instr.*`, `hints.instruments`, `run.profile` | `instrumentation.json` (§6.5) |
| 11 | `amt_gtr` | AMT | stems, instr | gpu / gpu env / `amt_muscriptor` | MuScriptor keys + `amt.guitar_view` + `amt.guitar_mask` (the policy only; the mask *content* comes from `instr`'s output and is covered by the `instr` dep key, not by params) | `raw/<amt.guitar_view>__muscriptor.json` (default `guitar_mono`), `gpu_run.json` |
| 12 | `s35` | AMT | instr, vocal, resid1, amt_ms1, amt_gtr, amt_bp | cpu / core | – | `s35.json`: index of the S3.5 artifacts (paths relative to the **job dir**, e.g. `stages/amt_gtr/<key12>/raw/...`; deterministic because keys are; Basic Pitch entries `null` when skipped); later stages (S4+) depend on this one stage |
| 13 | `notes` | AMT | amt_gtr, amt_ms1, amt_bp, grid, s35 | cpu / core | – | `guitar_all.json` (NoteSet: guitar classes of the `amt.guitar_view` transcription), `midi/guitar_all.mid`, `midi/bass_raw.mid`, `midi/b0_guitar.mid` (quality only), `midi/guitar_basicpitch.mid` (only if Basic Pitch ran), `notes_summary.json` (incl. `midi_bar_offset`, §9.2 item 7) |

Notes:
- `vocal`/`resid1` are not needed for MIDI but are S3.5 deliverables; routing them through `s35` makes
  `--until notes` produce them (DESIGN M2 "S3.5 v0").
- `fast` profile: `amt_ms1` drops the `mix_mono` view (no B0); `instr` then uses stem energy + hints only
  (DESIGN §2 fast). `quality` is the default.
- **Transcription cache (shared S3.5 ↔ S6, DESIGN §9.3):** `gtab/amt/cache.py`:
  `amt_key(backend, backend_version, model_sha256, view_pcm_sha256, instruments: list[str]|None, params: dict,
  code_version, rung_label) -> str` (sha256 of canonical JSON). `instruments` is **sorted by MT3 id**
  (`gtab.amt.instruments.sort_mt3`) before it enters the key *and* before it is sent to MuScriptor — MuScriptor
  turns the list into ids in the given order and its conditioning depends on that order. `rung_label` (e.g.
  `medium`, `small`) is part of the entry key, so a degraded transcription never answers for the top rung. Entries: `jobs/<song_key>/cache/amt/<key[:2]>/<key>.json` =
  raw NoteSet **before** latency correction, written atomically. `amt_ms1`/`amt_gtr`/`amt_bp` send only the
  views without a cache entry to the worker; then write the stage outputs (latency applied) from the cache.
  Outside jobs (reftx, experiments) the cache root is `data/eval/reftx_cache/`.
- MuScriptor latency: raw notes keep backend times; the stage writes `onset_s = raw − amt.latency_s` (clamped ≥ 0)
  and records `time_offset_applied_s = −latency_s`. Basic Pitch: 0 (its own latency is measured in the same M2
  experiment and only reported).

### 3.3 Publishing (`gtab/pipeline/publish.py`, GRID)

After a run whose stage set includes `notes`, publish into `jobs/<song_key>/export/` (DESIGN §7.1):
`midi/guitar_all.mid`, `midi/bass_raw.mid`, `midi/b0_guitar.mid` (if present), `practice/mix_minus_guitar.wav`,
`practice/mix_minus_bass.wav` (from `stems`), `practice/bass_stem.wav` (from **`sep`**'s `stems/bass.wav`),
`instrumentation.json`, `export.json` (`{"format":"gtab.export/1","files":{rel:{"stage","key12","sha256"}}}`).
Build in `store.make_temp_dir(job_dir, "export")`:
- **Small files (MIDI, JSON) are copied**; only WAVs are hardlinked (`os.link`, fallback copy), and every hardlinked
  WAV is marked read-only (`os.chmod(S_IREAD)`) — the attribute is shared with the stage file, which protects the
  cache from in-place edits (M0 `rmtree_quiet` already clears read-only on delete). Editing an exported MIDI in a
  DAW therefore never alters a cached stage file.
- Swap like ingest's rebuild: rename old `export/` aside to `make_temp_dir(job_dir, "export-old")`, rename new in,
  remove old. **If the old dir is locked** (a DAW/player has a file open), the aside rename or the removal fails →
  leave the `.tmp-export-old-*` dir for `gtab gc` (M0 `gc_temp` scans job dirs) and log a Korean note; if even the
  aside rename fails, keep the old export, leave the new build as `.tmp-export-*`, and fail publish with
  `"export 폴더의 파일이 다른 프로그램에서 열려 있습니다: <path>. 닫고 다시 실행하세요."` (exit 1; stages stay cached).
- Idempotent (same stage keys → same `export.json`, no rewrite).

---

## 4. Runner (`gtab/pipeline/runner.py`, GRID)

```python
@dataclass class RunPlanItem: stage: str; key12: str; cached: bool; device: str
def resolve_job(source: str, store: JobStore, cfg) -> IngestResult   # existing job key (prefix ok) or ingest(source)
def plan(song_key, cfg, *, until="notes", force=()) -> list[RunPlanItem]
def run(song_key, cfg, *, until="notes", force=(), progress: Callable[[str, dict], None] | None = None) -> dict[str, Path]
def data_outputs(stage_dir: Path) -> dict[str, str]   # rel path -> sha256, excluding manifest.json and _worker/** (§0)
```
`--from X` = `force={X}` (M0 `run_dag` already cascades force downstream). `progress` receives
`("stage_start"|"stage_done"|"stage_cached"|"stage_degraded_cache"|"vram_wait"|"model_missing", info)` for Korean
console output. `plan()` evaluates every selected stage's `models(cfg)` first and raises one `ModelMissing` listing
all absent models (`--dry-run` shows them as `모델 없음` rows instead). Wall time per stage is taken from manifests.

---

## 5. CLI (Korean help; exit codes: 0 ok, 1 failure, 2 usage/config error)

```
gtab run <파일|URL|작업키> [--until 단계] [--from 단계] [--force 단계,단계] [--profile fast|quality]
         [--instruments auto|"guitar,keys"] [--dry-run] [--set 키=값 ...]
  "파이프라인을 실행한다. 기본은 --until notes: 6 스템, nonvox, leftover, 편성(instrumentation.json),
   guitar_all.mid, bass_raw.mid, 연습용 반주(기타 뺀 믹스, 베이스 뺀 믹스). 끝난 단계는 캐시를 쓴다."
  --dry-run: "실행하지 않고 단계별 캐시 여부만 보여 준다."
  --instruments → --set hints.instruments=...; --profile → --set run.profile=...
  Output: 단계 표(단계, 키, 캐시/실행, 소요, 장치), then 결과 파일 목록 (export/ 경로), 경고(leftover, VRAM 대기).

gtab bench gpu [--clip 파일|작업키] [--seconds 240] [--backend sep|amt|beats|all] [--rungs all|auto]
               [--cap-mb N] [--ballast-mb N|auto] [--fallback on|off|unknown] [--repeat 1]
  "GPU 벤치마크: 백엔드와 사다리 칸마다 VRAM(reserved·NVML·Shared Usage)과 실시간 배수를 잰다.
   결과는 data/bench/ 아래에 저장되고 VRAM 사전 점검 표(vram_table.json)를 갱신한다.
   --fallback 은 NVIDIA 'Sysmem Fallback' 설정 상태를 기록용으로 적는다(프로그램이 바꾸지 않는다)."
  --cap-mb:     "PyTorch 할당기 한도(OOM 사다리 검증용). 드라이버 폴백은 일어나지 않는다."
  --ballast-mb: "별도 프로세스가 실제 VRAM을 N MB 붙잡아 여유를 줄인다(Sysmem Fallback 켬/끔 비교용).
                 auto = 첫 칸이 들어가지 않을 만큼."
  --rungs all = each rung separately with force_rung; auto = normal ladder from rung 0 (bench never prechecks,
  so the worker really attempts rung 0). Capped or ballasted runs are reported but never merged into vram_table.

gtab eval run --suite quick|bp-smoke|synth|tierA|tierB|components [--system 이름] [--out 폴더] [--set ...]
gtab eval compare <실행A> <실행B>                      "두 평가 실행을 곡 블록 부트스트랩으로 비교한다."
gtab eval import-external <곡ID> <파일(.mid/.gp5/.gp)> --system songsterr_ai|klangio|mvsep101|other [--map "1=L1,2=L2"]
gtab eval exp <E1|E2|E3|E23|E24-sep|E24-amt|latency|gate-m2|stereo-preservation> [--dry-run] [--data 세트]
  "사전 등록된 실험을 돌린다."  (ids = the `## <id>` headings of docs/decisions.md, exactly)
gtab eval reftx <데이터셋> [--track ID]              "Tier B 참 단독 트랙을 MuScriptor로 참조 전사한다."

gtab gt import <곡ID> <참조탭(.gp3/.gp4/.gp5/.gp/.gpx)> --audio <파일|작업키>
  "참조 탭을 가져와 반복·다른 엔딩을 펼치고 gt.yaml 초안을 만든다(구간·Line 지도는 직접 채운다)."
gtab gt render <곡ID> [--engine internal|alphatab]   "참조 탭을 오디오로 렌더한다(청취·A/B용)."
gtab gt align <곡ID> [--taps 라벨파일]                "DTW로 정답 템포맵을 만들고 align_check.wav 를 쓴다(Beat This! 안 씀)."
gtab gt taps <곡ID> <Audacity 라벨 파일>              "Audacity 라벨 트랙의 다운비트 탭을 가져와 정렬을 보정한다."
gtab gt approve <곡ID>                               "align_check.wav 를 듣고 승인한 것으로 기록한다."

gtab data list                                       "데이터셋 목록: 계층, 라이선스, 크기, 상태, 위치"
gtab data fetch <이름> [--yes] [--part 파일]          "URL·크기·라이선스·저장 위치를 보여 주고 y/N 을 받은 뒤 받는다."
gtab data verify [<이름>]                            "받은 파일의 SHA256 을 다시 계산해 기록과 비교한다."
gtab data import-local <이름> <폴더> [--song 이름]     "직접 받은 데이터(수동 다운로드)를 등록한다."

gtab models list                                     "모델 목록: 이름, 크기, 라이선스, 상태(없음/받음/검증됨), 위치"
gtab models fetch <이름> [--yes]                      "URL·크기·라이선스·저장 위치를 보여 주고 y/N 을 받은 뒤 받는다."
gtab models verify [<이름>]                          "모델 파일의 SHA256 을 다시 계산해 models.lock.json 과 비교한다."
```

---

## 6. Data formats (P0 writes the pydantic models; owners produce/consume them)

Conventions: times in seconds (float, rounded to 1e-6 in files); MIDI pitch ints; **bar numbers are musical:
1 = first full bar, 0 = pickup bar if present**; **bar ranges are half-open everywhere** (`start_bar` inclusive,
`end_bar` exclusive — `Section`, `GtSection`, `TempoSegment`, meter runs); ticks = 48 per beat of the bar's beat
unit (DESIGN 6.2); **string numbers are 1-based from the highest-pitched string** (tab top line = Guitar Pro /
PyGuitarPro convention; alphaTab and GuitarSet/IDMT indices are converted on import); `tuning[i]` is the open
pitch of string `i+1`; pitch = tuning + capo + fret; all relative paths are POSIX-style relative to the file that
contains them.

### 6.1 Notes — `gtab/schema/notes.py`

```python
class PitchBend(_Model):  t_s: list[float]; cents: list[float]         # same length; cents relative to `pitch`
class NoteEvent(_Model):
    onset_s: Seconds; offset_s: Seconds; pitch: Midi
    instrument: str | None = None          # MT3_FULL_PLUS group name (§6.5); None = class-agnostic backend
    velocity: int | None = None            # MuScriptor never sets it
    amplitude: float | None = None         # Basic Pitch note amplitude
    confidence: Prob | None = None
    bend: PitchBend | None = None
class BackendInfo(_Model):
    name: str                              # "muscriptor" | "basicpitch" | "gt" | "external:<system>" | "synthetic"
    version: str | None = None; model: str | None = None; model_sha256: str | None = None
    params: dict[str, Any] = {}; instruments: list[str] | None = None   # the hard mask actually used (None = all)
    device: str | None = None; dtype: str | None = None
class NoteSet(_Model):
    format: Literal["gtab.notes/1"] = "gtab.notes/1"
    view: str                              # view id: mix_mono | guitar_mono | bass_mono | piano_other_mono | nonvox_mono | ...
    view_sha256: str | None = None         # gtab.audio.pcm_sha256 of the input view
    backend: BackendInfo
    time_offset_applied_s: float = 0.0     # already added to onset/offset (MuScriptor: -latency)
    audio_duration_s: Seconds | None = None
    notes: list[NoteEvent]                 # sorted by (onset_s, pitch, offset_s, instrument or "")
def notes_to_arrays(ns: NoteSet, *, instruments: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]  # (n,2) s, (n,) midi
```
File naming: `<view>__<backend>.json`. No timestamps.

### 6.2 Beats and grid — `gtab/schema/grid.py`

```python
class BeatsRaw(_Model):   # beats.json
    format: Literal["gtab.beats/1"]; tracker: str            # "beat_this/final0"
    checkpoint_sha256: str | None; dbn: bool; fps: float     # 50
    beats_s: list[float]; downbeats_s: list[float]; duration_s: Seconds
class GridBeat(_Model):
    t_s: Seconds; t_raw_s: Seconds; bar: int; beat: int      # beat = 1-based position in bar
    is_downbeat: bool; shift_ms: float; activation: Prob | None
class GridBar(_Model):
    bar: int; start_s: Seconds; end_s: Seconds; n_beats: int
    numerator: int; denominator: int; beat_unit: Literal["quarter","dotted_quarter","eighth"]; pickup: bool = False
class TempoSegment(_Model): start_bar: int; end_bar: int; bpm: float        # end exclusive; bpm of the beat unit
class Grid(_Model):       # grid.json
    format: Literal["gtab.grid/1"]; tpb: int = 48; duration_s: Seconds
    beats: list[GridBeat]; bars: list[GridBar]; tempo: list[TempoSegment]
    meter: list[dict]                   # [{"start_bar","end_bar","numerator","denominator","beat_unit"}]
    beat_unit: Literal["quarter","dotted_quarter","eighth"]      # dominant
    pickup: dict                        # {"present": bool, "beats": int}
    hypotheses: dict                    # {"tempo_octave": {"decision":"keep|half|double","scores":{...},"onset_rate_hz":..},
                                        #  "compound": {"triple_ratio":..,"decision":"simple|compound|ambiguous",
                                        #               "alternatives":[{"numerator","denominator","beat_unit","score"}]}}
    confidence: dict                    # {"beats": p, "downbeats": p, "meter": p}
    params: dict
class TempoMap:                          # plain class, numpy; used by grid AND ground truth
    def __init__(self, beat_times: Sequence[float], bars: Sequence[int], beats: Sequence[int],
                 beats_per_bar: Mapping[int, int], tpb: int = 48)
    @classmethod
    def from_grid(cls, grid: Grid) -> TempoMap
    @classmethod
    def from_file(cls, path) -> TempoMap          # tempo_map.json (§6.7) or grid.json
    def time_to_beat(self, t) -> np.ndarray       # fractional global beat index; linear between beats,
                                                  # extrapolated with first/last interval outside
    def beat_to_time(self, b) -> np.ndarray
    def time_to_bar_tick(self, t) -> tuple[np.ndarray, np.ndarray]   # bar (int), tick (float) within bar
    def bar_tick_to_time(self, bar, tick) -> np.ndarray
    def ticks_per_second(self, t) -> np.ndarray   # local tpb / beat duration (for ±50 ms → tick tolerance)
```
TempoMap tests (P0): round trip time→(bar,tick)→time < 1e-9 s; tempo change; pickup bar 0; extrapolation.

### 6.3 Sections and repeat map — `gtab/schema/sections.py`

```python
class Section(_Model): id: str; label: str; start_bar: int; end_bar: int      # end exclusive
                       start_s: Seconds; end_s: Seconds; confidence: Prob
class Sections(_Model): format: Literal["gtab.sections/1"]; method: str; sections: list[Section]; params: dict
class Occurrence(_Model): section: str; start_bar: int; n_bars: int; offset_s: float     # offset vs group reference
class RepeatGroup(_Model): label: str; occurrences: list[Occurrence]; reference: str      # reference section id
class BarMatch(_Model): bar: int; matches: list[dict]     # [{"bar": int, "score": Prob, "offset_s": float}] best first
class RepeatMap(_Model): format: Literal["gtab.repeat_map/1"]; groups: list[RepeatGroup]; bars: list[BarMatch]; params: dict
```

### 6.4 Separation outputs (SEP)

`sep.json`: `{"format":"gtab.sep/1","backend":"sw_msst","msst_commit","ckpt_sha256","config_sha256",
"stem_order":["bass","drums","other","vocals","guitar","piano"],"chunk_size","num_overlap","dtype":"fp32+amp",
"zero_dc":true,"input_lufs","input_gain_db","device_used","rung":{"index","label"},"audio_s","process_s","rtf",
"vram":{...§7.1...}}` (GPU stage; not byte-deterministic).
`views.json`: `{"format":"gtab.views/1","views":{"guitar_mono":{"path":"views/guitar_mono.wav","pcm_sha256":..,
"source":"stems/guitar.wav","mono":"(L+R)/2"}, ...}}`; views: `mix`, `guitar`, `bass`, `piano_other` (piano +
other), `nonvox`, `guitar_other` (guitar + other; E2 option), each `<id>_mono`.
`stem_stats.json`: `{"format":"gtab.stem_stats/1","mix_rms_db","stems":{name:{"rms_db","rel_db","peak"}},
"leftover":{"rms_db_rel_mix","flag": rel > leftover_warn_db,"worst_windows":[{"start_s","rel_db"}] (10 s windows,
top 5)},"nonvox":{"rel_db"}}`. `energy.npz`: per stem (incl. mix, nonvox, leftover) RMS in dB at 10 Hz (float32).

### 6.5 Instrumentation — `gtab/schema/instrumentation.py`

```python
MT3_GROUPS: dict[str, int] = {  # = installed muscriptor 0.3.0 MT3_FULL_PLUS group table (verified: 35 entries, 0-33 + drums 36)
 "acoustic_piano":0,"electric_piano":1,"chromatic_percussion":2,"organ":3,"acoustic_guitar":4,
 "clean_electric_guitar":5,"distorted_electric_guitar":6,"acoustic_bass":7,"electric_bass":8,"violin":9,
 "viola":10,"cello":11,"contrabass":12,"orchestral_harp":13,"timpani":14,"string_ensemble":15,"synth_strings":16,
 "voice":17,"orchestra_hit":18,"trumpet":19,"trombone":20,"tuba":21,"french_horn":22,"brass_section":23,
 "soprano_and_alto_sax":24,"tenor_sax":25,"baritone_sax":26,"oboe":27,"english_horn":28,"bassoon":29,
 "clarinet":30,"flutes":31,"synth_lead":32,"synth_pad":33,"drums":36}
FAMILIES: dict[str, tuple[str, ...]] = {
 "guitar": ("acoustic_guitar","clean_electric_guitar","distorted_electric_guitar"),
 "bass": ("acoustic_bass","electric_bass"),
 "keys": ("acoustic_piano","electric_piano","organ","chromatic_percussion"),
 "synth": ("synth_lead","synth_pad","synth_strings"),
 "strings": ("violin","viola","cello","contrabass","orchestral_harp","string_ensemble"),
 "brass": ("trumpet","trombone","tuba","french_horn","brass_section","orchestra_hit"),
 "winds": ("soprano_and_alto_sax","tenor_sax","baritone_sax","oboe","english_horn","bassoon","clarinet","flutes"),
 "vocals": ("voice",), "drums": ("drums","timpani")}
GUITAR_CLASSES = FAMILIES["guitar"]; BASS_CLASSES = FAMILIES["bass"]
class ClassEvidence(_Model): p: Prob; present: bool; sources: dict[str, dict]   # "mix_transcription" | "stem_energy" | "hint"
class Instrumentation(_Model):
    format: Literal["gtab.instrumentation/1"]; profile: str
    classes: dict[str, ClassEvidence]          # every MT3 group, deterministic order = MT3_GROUPS order
    families: dict[str, Prob]
    masks: dict[str, list[str] | None]         # {"guitar_mono": [...], "mix_mono": None, "bass_mono": [...], "piano_other_mono": [...]}
    per_section: list[dict]                    # [{"section","family_active": {family: ratio}}]
    thresholds: dict; hints: dict; warnings: list[str]
```

### 6.6 S3.5 CPU artifacts (GRID)

`vocal_activity.json`: `{"format":"gtab.vocal/1","fps":10,"rel_db":[...float32 rounded 0.1...],"active":[0/1...],
"per_bar":[{"bar","active_ratio"}],"threshold_rel_db"}`.
`resid1.json`: `{"format":"gtab.resid1/1","per_bar":[{"bar","resid_rel_db"|null,"group"|null}],
"groups_used":[label...],"skipped":[{"label","reason"}],"method":"beat-aligned median background, repet3 soft mask (W=min(median,V))"}`.

### 6.7 Ground truth — `gtab/schema/gt.py`

Per song directory `data/eval/tierA/<song_id>/` (song_id ASCII `[a-z0-9_-]+`):
`gt.yaml`, `ref.<ext>` (byte copy), `refscore.json` (neutral import, unexpanded), `gt_notes.json` (expanded),
`tempo_map.json`, `render.wav`, `taps.json`, `align_report.json`, `align_check.wav`.

```python
class GtTrack(_Model): ref_track: int; name: str; kind: Literal["guitar","bass","other"]; line: str | None
                       tuning: list[Midi] | None; capo: int = 0; player: str | None = None
class GtLine(_Model): id: str; name: str; kind: Literal["guitar","bass"]; takes: list[int]      # ref_track numbers
                      relation: dict = {}   # {"double": bool, "unison_with": [line ids], "twin_with": id|None, "overdub": bool}
class GtSection(_Model): id: str; start_bar: int; end_bar: int     # expanded musical bars, end EXCLUSIVE (like Section)
                         scenario: str; roles: dict[str, Literal["lead","rhythm","both","silent"]]; split: Literal["dev","test"]
class GtSpec(_Model):     # gt.yaml
    format: Literal["gtab.gt/1"]; song_id: str; title: str | None; artist: str | None
    audio: dict                    # {"song_key": "f-...", "note": "CD rip"}
    version_flags: dict            # {"version": "album|mv|live", "trimmed": bool, "note": str}
    reference: dict                # {"file": "ref.gp5", "format": "gp5", "expanded_bars": int, "bar_order": list[int] | None,
                                   #  "encoding": str | None}   # GP3-5 string encoding override (default: try cp949, then cp1252)
    families_present: list[str]    # ground truth for b13: FAMILIES keys actually playing (e.g. ["guitar","bass","keys","vocals","drums"])
    tracks: list[GtTrack]; lines: list[GtLine]; players: list[dict]; sections: list[GtSection]
    alignment: dict                # {"method","approved": bool,"approved_utc","minutes_spent"}
class GtNote(_Model): line: str; ref_track: int; bar: int; tick: float; dur_ticks: float; pitch: Midi
                      string: int | None; fret: int | None; tied: bool = False; shared: bool = False
                      onset_s: Seconds | None = None; offset_s: Seconds | None = None     # filled after align
class GtNotes(_Model): format: Literal["gtab.gtnotes/1"]; song_id: str; tpb: int; bars: list[dict]   # [{"bar","numerator","denominator","beat_unit","nominal_bpm"}]
                       notes: list[GtNote]
# tempo_map.json: {"format":"gtab.tempomap/1","source":"dtw|dtw+taps|synthetic","tpb":48,
#                  "beats":[{"t_s","bar","beat"}],"beats_per_bar":{"<bar>": n}}
# taps.json: {"format":"gtab.taps/1","source_file","taps":[{"t_s","bar"|null,"label"}]}
```
`shared`: set by import for notes whose (pitch, bar, tick) also occur in another line (unison, DESIGN R3).
The `gt.yaml` skeleton explains in Korean comments that `end_bar` is exclusive ("17~24마디면 start_bar: 17,
end_bar: 25").
`bar_order` handles D.S./coda manually (list of written bar numbers in play order); `None` = automatic
repeat/alternate-ending expansion.

### 6.8 Dataset tracks — `gtab/schema/dataset.py`

```python
class Part(_Model): id: str; kind: Literal["guitar","bass","vocals","drums","keys","other","mix","di"]
                    audio: str                           # path relative to the dataset root
                    channels: int; line: str | None = None; take: int | None = None
class DatasetTrack(_Model):
    format: Literal["gtab.dataset_track/1"]; dataset: str; track_id: str; split: str | None
    group: str | None                        # song/performer id for block bootstrap and comp/solo pairing
    tier: Literal["B","C"]; license: str
    mix: str | None; parts: list[Part]; lines: list[GtLine]
    families_present: list[str]              # FAMILIES keys playing in this track (GuitarSet: ["guitar"]; Cambridge-MT: from lines.yaml)
    notes: dict[str, NoteSet]                # line id -> GT notes (seconds); NoteEvent.instrument may be None
    string_fret: dict[str, list[tuple[int,int]]] | None   # line id -> per-note (string, fret), same order as notes;
                                             # string numbering per §6 conventions (1 = highest-pitched)
    beats_s: list[float] | None; downbeats_s: list[float] | None; tuning: list[int] | None
    meta: dict
```

### 6.9 Predictions for evaluation — `gtab/schema/evalio.py`

```python
class PredNote(_Model): onset_s: Seconds; offset_s: Seconds; pitch: Midi
                        line: str                          # line id or "unassigned"
                        posterior: Prob | None = None      # max part posterior (coverage–accuracy curve)
                        shared: bool = False; bar: int | None = None; tick: float | None = None   # bar/tick only if quantized
class Prediction(_Model): format: Literal["gtab.prediction/1"]; system: str; item: str   # song_id / dataset:track / scene id
                          lines: list[dict]; notes: list[PredNote]; meta: dict = {}
```

### 6.10 Evaluation run layout (EVAL; DESIGN §8.7)

`data/runs/<YYYYMMDD-HHMMSS>_<gitsha7>_<slug(system)>/` (`slug` §0: `job:amt_gtr` → `job-amt_gtr`); external
predictions live in `data/eval/external/<slug(system)>/<slug(item)>.json`:
- `run.json` — suite, system, config, model shas, dataset lock shas, git sha, timestamps (the only file with them)
- `metrics.csv` — long format, UTF-8, `\n`, sorted by (suite, dataset, item, scenario, section, line, metric, rung):
  `suite,system,dataset,item,group,scenario,section,line,rung,metric,value,tp,fp,fn,n,ci_low,ci_high,note`
  (per-item rows have empty CI; aggregate rows have `item="*"` and CI from the block bootstrap; `rung` = GPU rung
  label that produced the prediction, e.g. `chunk=588800|medium`, empty for CPU-only systems — experiments always
  fill it, and rows with different rungs are never pooled).
- `summary.json` — aggregate metrics with CIs, baselines, MDE checks (deterministic).
- `report.html` — jinja2: per item/line/scenario tables, confusion matrix, coverage–accuracy curve (inline SVG
  via matplotlib), worst-bars list (bar links `gtab://job/<song_key>?bar=N` placeholder for the M4 UI).
- `worst_bars.csv`.
`metrics.csv` and `summary.json` must be byte-identical between two runs with the same inputs.

### 6.11 Registries

- `data/datasets/datasets.lock.json`: `{"format":"gtab.datasets/1","datasets":{name:{"files":{fname:{"url"|"local",
  "bytes","sha256","md5"|null,"fetched_utc"}},"extracted":"<dir>","status":"fetched|verified|manual"}}}`.
- `data/bench/vram_table.json`: `{"format":"gtab.vram_table/1","gpu","driver","entries":{backend:{rung_label:
  {"reserved_peak_mb","ctx_mb","nvml_delta_peak_mb","rtf","fallback":"on|off|unknown","measured_utc","clip_s"}}},
  "calibration":{"shared_growth_noise_mb","runs"}}`.
  **Two quantities, two sources, never mixed** (NVML and `mem_get_info` free disagreed by ≈ 350 MB on this PC):
  `reserved_peak_mb` = torch `max_memory_reserved` for the rung (model + activations); `ctx_mb` = NVML used after
  CUDA-context creation minus NVML used just before it (worker, before model load; default
  `gpu.ctx_mb_default` 500 **[잠정]**). `nvml_delta_peak_mb` is informational. Only uncapped, ballast-free runs
  with status `ok` are merged (median over repeats); capped/ballasted runs are evidence for b3/b4 only.

---

## 7. Worker contracts

### 7.1 Common envelope (all new workers)

Request:
```json
{"format": "gtab.worker/1", "job": {"song_key": "...", "stage": "...", "stage_key": "..."} ,
 "out_dir": "<abs path; the worker writes data files only here>",
 "device": "auto|cuda|cpu",
 "vram": {"policy": "wait|error|cpu", "cpu_allowed": false, "precheck": true,
          "worker_headroom_mb": 256,
          "need_table": {"<rung label>": {"reserved_peak_mb": 1800, "ctx_mb": 500}},
          "shared_growth_fail_mb": 64, "rtf_ref": {"<rung label>": 6.0} , "rtf_fail_ratio": 0.5,
          "cap_mb": null, "force_rung": null},
 "determinism": true}
```
`job` may be null (bench, experiments). `cpu_allowed` = policy `"cpu"` **and** backend in `gpu.cpu_backends`
(computed by the orchestrator; the worker never decides it). `precheck: false` (bench only) makes the worker attempt
rungs without its budget check. `cap_mb` → `torch.cuda.set_per_process_memory_fraction(cap/total)` (OOM-ladder
test b4: the PyTorch allocator raises OOM before real memory runs out, so the driver's Sysmem Fallback never
engages — b3 uses the ballast instead, §9.1). `force_rung` → start at that rung and do not step down (bench,
experiments). `rtf_ref` comes from `vram_table.json` (`null` for a rung the bench has not measured → no slowness
check). The orchestrator's budget (`margin_mb`, NVML free) is not sent: the worker checks only its own source
(§8.1).

`result["output"]` (inside the M0 envelope):
```json
{"status": "ok" | "insufficient_vram",
 "backend": {"name", "version", "model", "model_sha256", "dtype_policy", "device_used"},
 "vram": {"device": "cuda:0", "total_mb", "start_free_mb" (mem_get_info, after context), "ctx_mb" (measured,
          §6.11), "start_nvml_used_mb", "rung_index", "rung_label", "cap_mb",
          "attempts": [{"rung_index", "rung_label", "device", "status": "ok|oom|shared_growth|unavailable|skipped_budget",
                        "max_allocated_mb", "max_reserved_mb", "nvml_used_peak_mb", "pdh_self_shared_growth_mb",
                        "pdh_adapter_shared_delta_mb", "rtf", "slow": bool, "error"}]},
 "timing": {"load_s", "process_s", "audio_s", "rtf"},          // rtf = audio_s / process_s (실시간 배수)
 "files": {"<rel path>": {"bytes": n}}, "warnings": ["..."]}
```
**Required in every GPU result** (M0 base + SEP's additive base.py change): top-level `cuda.max_allocated_mb`,
`cuda.max_reserved_mb`, `sysmon.nvml_used_peak_mb`, `sysmon.nvml_used_start_mb`, `sysmon.pdh_self_shared_start_mb`,
`sysmon.pdh_self_shared_peak_mb`, **new** `sysmon.marks` (`{"model_loaded": {"t_s","pdh_self_shared_mb",
"pdh_adapter_shared_mb","nvml_used_mb"}}`, set by the worker via `ctx.sampler.mark("model_loaded")`),
`sysmon.pdh_self_shared_growth_mb` (= this pid's shared peak **after** the `model_loaded` mark − its value at the
mark; `null` if PDH was unavailable), `sysmon.pdh_self_shared_delta_mb` (= peak − start; info: includes legitimate
staging during upload), `sysmon.pdh_adapter_shared_delta_mb` (adapter-wide; **info only** — Chrome, Discord,
Slack and the Claude app move it on their own) and `sysmon.shared_growth` (bool: `pdh_self_shared_growth_mb >
shared_growth_fail_mb`, default 64 **[잠정]**, calibrated from fallback-off bench runs). Per-rung values in
`vram.attempts` come from `torch.cuda.reset_peak_memory_stats()` + `ResourceSampler.snapshot()` at rung start/end
(§8.2); each rung's growth is measured from the snapshot taken after that rung's model is resident.
`status: "insufficient_vram"` = the worker's own check (`mem_get_info` free < `reserved_peak_mb` +
`worker_headroom_mb` for every GPU rung, and CPU not allowed) or every GPU rung failed and CPU is not allowed; it
exits ok without outputs and the stage function applies the wait policy and retries (§8.1).

### 7.2 `sep_msst` (SEP, gpu env)

Request (+ envelope): `{"mix_wav", "model": {"ckpt", "ckpt_sha256", "config_yaml", "arch": "bs_roformer"},
"chunk_ladder": [588800, 352256, 262144], "num_overlap": 2, "batch_size": 1, "dtype": "upstream|fp16",
"input_gain_db": 0.0, "outputs": {"stems_dir": "stems"}}` (stems_dir relative to out_dir). Rung labels
`chunk=<n>`; CPU rung `chunk=<smallest>@cpu` only if `vram.cpu_allowed`. `input_gain_db` (E1 / `sep.input_lufs`)
is computed core-side by the `sep` stage (`input_lufs − integrated LUFS of the mix` via
`gtab.ingest.loudness.measure`, rounded to 0.01 dB; 0 when `"off"`); the worker multiplies the mix by the gain and
every stem by its inverse.
Output extra: `{"stem_order": [...from yaml training.instruments...], "stems": {"guitar": "stems/guitar.wav", ...},
"chunk_size": 588800, "config_sha256", "zero_dc": true}`. Writes stereo float32 WAVs via `gtab.audio.write_f32`.
Modes: `"mode": "separate"` (default) | `"selftest_identity"` (CPU, builds an identity "model" and runs the
vendored demix loop on the given wav; reports max abs reconstruction error; used by `gpuenv` tests).

### 7.3 `amt_muscriptor` (AMT, gpu env)

Request: `{"model": {"size": "medium", "weights_path": "<snapshot>/model.safetensors", "sha256", "revision",
"loader": "upstream|lean", "fallback": [{"size": "small", "repo": "MuScriptor/muscriptor-small", "weights_path": null}]},
"decode": {"prelude_forcing": true, "beam_size": 1, "cfg_coef": 1.0, "use_sampling": false, "batch_size": 1},
"dtype": "upstream|fp16",
"views": [{"id": "guitar_mono", "wav": "<abs>", "instruments": ["acoustic_guitar", ...] | null,
           "out": "<abs path of raw NoteSet json>", "view_sha256": "..."}]}`.
The core side computes the medium `weights_path` from `paths.HF_HOME` + `amt.muscriptor_revision` (no HF library
in core). `instruments` arrive sorted by MT3 id (§3.2); the worker re-sorts defensively.
**Loading (the model cannot be moved after construction, see facts):**
- `loader: "upstream"` (default): `TranscriptionModel.load_model(<local weights path>, device="cuda")` — never a
  size keyword, never a hub id. Reserved ≈ 2.4 GB (model + a transient second copy of the weights).
- `loader: "lean"` (optional, saves ≈ 0.6 GB): for the pinned 0.3.0 only (worker checks
  `muscriptor.__version__ == "0.3.0"`, else falls back to upstream with a warning): build on CUDA with
  `_build_model(torch.device("cuda"), _resolve_config(p, p))`, load weights on CPU with
  `safetensors.torch.load_file(p, device="cpu")`, apply `_remap_single_codebook_keys`, `load_state_dict(strict=True)`,
  then `TranscriptionModel(model, MT3Tokenizer("MT3_FULL_PLUS", max_shift_steps=1001), device)`. Uses private
  functions → allowed as default only after the `gpu` parity test (§9.2 tests) passes; AMT then files an
  integration request to flip `amt.muscriptor_loader`.
- CPU rung (`medium@cpu`) only if `vram.cpu_allowed` (not by default: hours for four views): build with
  `device="cpu"` (plain fp32, no autocast) — never CPU-then-move.
- dtype: `upstream` = what the library does when built on CUDA (fp32 weights, built-in fp16 autocast,
  conditioners fp32). `fp16` (E24-amt only) = fp16 weights except the conditioners, recorded in
  `backend.dtype_policy`. **Any request or environment implying bfloat16 is refused** (sm_75 has no native bf16).
- Fallback `small`: path = `huggingface_hub.try_to_load_from_cache("MuScriptor/muscriptor-small",
  "model.safetensors")` (works under `HF_HUB_OFFLINE=1`); not a path → rung `unavailable` (never downloads).
Then `torch.inference_mode()`, batch 1; for each view read the WAV with soundfile → `torch.from_numpy(x.T)` `[C,T]`
float32 → `model.transcribe((tensor, sr), instruments=...)`; assemble notes from Start/End events (unterminated
notes end at audio end; `ProgressEvent` from `muscriptor.events` drives the Watchdog); write NoteSet JSON (raw
times, `time_offset_applied_s: 0`). `instruments=None` means all classes. Rung labels: `medium`, `medium-lean`,
`small`, `medium@cpu`. Output extra: `{"views": [{"id", "out", "n_notes", "audio_s", "process_s"}], "api":
{"muscriptor_version", "transcribe_signature", "input_kind": "tensor"}, "load_s", "loader"}`. Mode `"probe"`: report
the installed API (signature via `inspect`, group names) without loading weights.

### 7.4 `amt_basicpitch` (AMT, bp310 env, Python 3.10, CPU/ONNX, `use_lock=False`)

Request: `{"mode": "transcribe|sweep", "threads": 4, "model_output_cache": "<abs dir>|null",
"views": [{"id", "wav", "out", "view_sha256"}], "params": {"onset_threshold": 0.5, "frame_threshold": 0.3,
"min_note_frames": 4, "minimum_frequency": null, "maximum_frequency": null, "melodia_trick": true,
"multiple_pitch_bends": false}, "settings": [<params dict>, ...] (sweep only), "allow_default_min_len": false}`.
**Two-step path in both modes** (so tuned E23 settings reproduce exactly in the pipeline): (1)
`basic_pitch.inference.run_inference(wav, model)` once per view with the bundled ONNX model, the model loaded
**once per worker**, the ORT session built with `SessionOptions(intra_op_num_threads=threads,
inter_op_num_threads=1, execution_mode=ORT_SEQUENTIAL)` (replace the library's session; pinned threads → stable
NoteSets); model output (`note`, `onset`, `contour` arrays) optionally cached as `<cache>/<view_sha256>__<model
sha12>.npz`. (2) `note_creation.model_output_to_notes(output, onset_thresh, frame_thresh, infer_onsets=True,
min_note_len=min_note_frames − 1, min_freq, max_freq, include_pitch_bends=True, multiple_pitch_bends,
melodia_trick)` — **frames are passed directly** (Basic Pitch drops notes with length ≤ `min_note_len`, so
`min_note_frames` is the shortest note kept; §1.3). The worker **rejects** `min_note_frames == 12` (the upstream
127.7 ms default) unless `"allow_default_min_len": true` (DESIGN §9.7 test; E23's 12-frame arm sets it).
`sweep` (E23): step 1 per view (cached), then step 2 for every entry of `settings`, writing
`<out stem>__s<k>.json` per setting — 72 settings cost one model pass per track.
Note events → NoteSet with `amplitude` and `bend` (Basic Pitch bends are per-frame bin offsets of 1/3 semitone at
86.13 fps → cents = 100/3 × bin; `t_s` = note onset + frame index / (22050/256)). Output extra: `{"model_path",
"model_sha256", "onnxruntime_version", "basic_pitch_version", "frames_per_second": 86.1328, "min_note_len_frames"
(= threshold passed), "min_note_frames_kept", "threads"}`.
Imports only stdlib, numpy, soundfile, basic_pitch, onnxruntime and `gtab.workers.base` (3.10-compatible); the file
carries `# gtab: py310` (P0's compat test). The bp310 lock notes the 3.10 resolution (`onnxruntime<=1.23.2`,
`numpy<=2.2.6`, `scipy<=1.15.3`).

### 7.5 `beats_beatthis` (GRID, gpu env)

Request: `{"wav": "<mix>", "checkpoint_path": "<abs local path from gtab.models.local_path('beat_this.final0')>",
"checkpoint_sha256", "dbn": false, "float16": false, "out_beats": "beats.json", "out_activations":
"activations.npz"}`. The worker **never** receives a hub name (a name would download inside the GPU lock and load
with `weights_only=False`); it verifies the sha256, then `Audio2Frames(checkpoint_path=<local path>, ...)`, which
loads local files with `weights_only=True`. If the stored hyper-parameters fail to unpickle under
`weights_only=True`, allow exactly those types with `torch.serialization.add_safe_globals([...])` (listed in the
module docstring) — never turn safe loading off. Read the audio with soundfile (float32, (n, 2)), pass **in
memory** (`(signal, sr)`, never a path — torchaudio.load needs TorchCodec). One model load gives both logits and
beats: `Audio2Frames` → logits → `beat_this.model.postprocessor.Postprocessor(type="minimal", fps=50)` (its minimal postprocessing
already snaps each downbeat to the nearest beat). Output extra: `{"checkpoint_path", "checkpoint_sha256",
"n_beats", "n_downbeats"}`. Rung labels `final0`, `final0@cpu`; the CPU rung needs `vram.cpu_allowed` (default
`gpu.cpu_backends` includes `beats_beatthis`, so it is used under policy `"cpu"` only).

---

## 8. VRAM budget, wait policy, OOM ladder (SEP; used by all GPU workers)

### 8.1 `gtab/vram.py` (torch-free, core + workers)

```python
@dataclass class GpuSnapshot: total_mb: float; used_mb: float; free_mb: float; processes: list[dict]
def snapshot() -> GpuSnapshot | None                       # NVML (gtab.sysmon); None if unavailable
def budget_mb(snap: GpuSnapshot, margin_mb: float) -> float  # NVML free_mb - margin_mb (margin = other apps' fluctuation)
@dataclass class RungNeed: reserved_peak_mb: float; ctx_mb: float
def load_table(path=paths.VRAM_TABLE, *, ctx_default_mb: float) -> dict[str, dict[str, RungNeed]]  # falls back per entry to DEFAULT_NEEDS
DEFAULT_NEEDS = {   # reserved_peak_mb seeded from the 2026-09-30 critique measurements (§ facts); ctx from gpu.ctx_mb_default
  "sep_msst":       {"chunk=588800": 1800, "chunk=352256": 1400, "chunk=262144": 1200},
  "amt_muscriptor": {"medium": 2400, "medium-lean": 1800, "small": 1400},   # small: estimate, never measured
  "beats_beatthis": {"final0": 500}}                                        # bench overwrites all of these
def orchestrator_need_mb(n: RungNeed) -> float              # reserved_peak_mb + ctx_mb (compared with NVML free - margin)
def worker_need_mb(n: RungNeed, headroom_mb: float) -> float  # reserved_peak_mb + headroom (compared with mem_get_info free)
def choose_rung(labels: list[str], budget: float, needs: dict[str, RungNeed]) -> int | None   # first (largest) that fits
class VramInsufficient(RuntimeError)                        # Korean message: need / free / other GPU processes
def wait_for_budget(backend: str, labels: list[str], cfg, *, clock=time.monotonic, sleep=time.sleep,
                    notify: Callable[[str], None] | None = None) -> tuple[int | None, float]
    # (first GPU rung that fits | None = use the CPU rung, waited_s).
    # policy "wait": poll every gpu.wait_poll_s up to gpu.wait_max_s, then VramInsufficient;
    # "error": raise at once; "cpu": return None iff backend in gpu.cpu_backends, else behave like "wait".
    # notify gets e.g. "VRAM 부족: 필요 약 2.3 GB, 여유 1.6 GB (다른 GPU 사용 프로그램: ...). 앱을 닫으면 이어서 진행합니다. 대기 중..."
def run_gpu_stage(backend, labels, request, cfg, work_dir, *, notify) -> WorkerResult
    # wait_for_budget → run_worker → on output.status == "insufficient_vram": wait again and retry, at most
    # gpu.max_worker_retries times, then VramInsufficient. Used by every GPU stage (SEP, AMT, GRID).
def write_gpu_run(out_dir: Path, backend: str, labels: list[str], worker_output: dict, waited_s: float) -> None  # §3.1
def read_gpu_run(stage_dir: Path) -> dict | None
```
**Never compare one source's need with the other source's free** (NVML and `mem_get_info` disagreed by ≈ 350 MB):
- orchestrator (no CUDA context): `reserved_peak_mb + ctx_mb + margin_mb ≤ NVML free`;
- worker (context exists, before model load): `reserved_peak_mb + worker_headroom_mb (256) ≤ mem_get_info free`.

With a typical desktop (≈ 4.1 GB NVML free, budget ≈ 3.4 GB) every top rung fits: SW 1800 + 500 = 2.3 GB,
MuScriptor-medium upstream 2400 + 500 = 2.9 GB. VRAM only binds when a game or another CUDA app is running.

### 8.2 `gtab/workers/gpu_common.py` (torch, lazy; workers only)

```python
def pick_device(req: dict) -> "torch.device"
def apply_determinism(on: bool) -> None      # CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before CUDA init),
                                             # torch.use_deterministic_algorithms(True, warn_only=True), cudnn flags
def apply_cap(req: dict) -> None             # set_per_process_memory_fraction from vram.cap_mb
def init_cuda_measure_ctx(sampler) -> float  # NVML used before/after torch.cuda.init() + 1-element alloc -> ctx_mb
def mem_get_info_mb() -> tuple[float, float]
def refuse_bf16(req: dict) -> None           # raises on any bf16 request/autocast dtype (sm_75)
@dataclass class Attempt: rung_index; rung_label; device; status; max_allocated_mb; max_reserved_mb;
                          nvml_used_peak_mb; pdh_self_shared_growth_mb; pdh_adapter_shared_delta_mb; rtf; slow; error
class SharedGrowth(RuntimeError)
class Watchdog:                              # called by the worker between chunks / on ProgressEvent
    def __init__(self, sampler, *, shared_growth_fail_mb, rtf_ref: float | None, rtf_fail_ratio: float,
                 audio_s_total: float)
    def arm(self) -> None                    # call once the rung's model is resident: baseline = sampler.snapshot()
    def check(self, audio_s_done: float) -> None
        # SharedGrowth iff this pid's PDH shared usage grew > shared_growth_fail_mb since arm() — the ONLY hard
        #   trigger. Adapter-wide shared usage is recorded, never decisive (other apps move it).
        # slow (rtf < rtf_fail_ratio × rtf_ref, evaluated after >= 10 % done) is a WARNING flag on the attempt,
        #   unless PDH self counters are unavailable in this worker — then slow alone raises SharedGrowth
        #   ("degraded detection", recorded), because the spill cannot be seen directly.
def run_ladder(rungs: list[dict], attempt: Callable[[dict, Watchdog], T], *, sampler, req) -> tuple[T | None, list[Attempt]]
    # rung dict: {"label","device","params"}. GPU rungs over the worker-side need are recorded skipped_budget
    # (unless req.vram.precheck is false). On torch.OutOfMemoryError / SharedGrowth: record, free (del + gc +
    # torch.cuda.empty_cache()), next GPU rung. The CPU rung is appended only if req.vram.cpu_allowed; it is
    # never reached because of slowness. If no rung succeeded and CPU is not allowed → returns (None, attempts)
    # and the worker answers status "insufficient_vram" (the stage waits and retries, §8.1); other exceptions
    # propagate (a real bug, not a VRAM problem).
```
`ResourceSampler` (M0 base.py, SEP additive change, stays Python-3.10-safe): `snapshot() -> dict` returning current
and peak values thread-safely; `mark(label)` stores a snapshot under `marks[label]`; `stats()` gains `marks`,
`pdh_self_shared_growth_mb` (vs the `model_loaded` mark), `pdh_self_shared_delta_mb`,
`pdh_adapter_shared_delta_mb`, `shared_growth` (§7.1). Sysmem Fallback turns OOM into a silent ~10× slowdown
(DESIGN §9.5): the direct signal is this process's shared-memory growth after its model is resident. Slowness alone
is usually GPU *contention* (a game rendering), which a smaller rung does not fix — so it only warns.

Ladders (CPU rung present only when `cpu_allowed` = policy `"cpu"` ∧ backend ∈ `gpu.cpu_backends`):
- `sep_msst`: `chunk=588800` → `chunk=352256` → `chunk=262144` (→ `chunk=262144@cpu`, ≈ 23 min per 4-min song,
  not in `cpu_backends` by default).
- `amt_muscriptor`: `medium` (or `medium-lean`) → `small` (only if cached) (→ `medium@cpu`, not by default).
- `beats_beatthis`: `final0` → `final0@cpu` (in `cpu_backends` by default; small model).
Under the default policy `"wait"`, "every GPU rung failed" means waiting (bounded by `wait_max_s` and
`max_worker_retries`), never a silent CPU run.

---

## 9. Owner work packages

### 9.1 SEP — separation, VRAM, bench (M2)

1. **First:** `gtab/vram.py` + `gtab/workers/gpu_common.py` signatures (§8) with working basic logic; base.py
   additive changes (`snapshot()`, `mark()`, growth/delta fields; Python-3.10-safe — P0's compat test). Commit.
2. **Vendor MSST** at `msst_commit` into `gtab/third_party/msst/`: exactly `models/bs_roformer/{__init__,
   bs_roformer,attend}.py`, upstream `LICENSE` (MIT), `VENDOR.md` (repo URL, commit, file list with upstream and
   local sha256, every local patch listed). Local patches (all verified necessary by the critique probe):
   - `__init__.py` **trimmed** to `from .bs_roformer import BSRoformer` (upstream also imports the conformer and
     mel-band models; the mel-band ones import librosa, absent from the gpu env);
   - imports made package-relative;
   - `attend.py`: the CUDA inference path already uses `torch.nn.attention.sdpa_kernel` (keep); the **CPU path**
     calls the deprecated `torch.backends.cuda.sdp_kernel` and warns on every call → replace with plain
     `F.scaled_dot_product_attention` (flash kernels do not exist on sm_75; SDPA picks mem-efficient/math);
   - no module-level matplotlib/librosa.
   `zero_dc=True` (HEAD default: zeroes the DC bin of every stem) is kept and documented in `VENDOR.md` and
   `sep.json`. `gtab/third_party/msst/demix.py`: port of MSST `demix` (generic, non-htdemucs branch: step = chunk //
   num_overlap, fade = chunk // 10, border = chunk − step, reflect pad when length > 2·border, windowed
   overlap-add) **with CPU float32 accumulators** (upstream HEAD keeps them on GPU; we keep VRAM for the model) and
   a per-chunk callback for the Watchdog.
3. **Config loading:** yaml with a `SafeLoader` subclass that maps `tag:yaml.org,2002:python/tuple` → tuple (never
   `yaml.load` unsafe; no omegaconf/ml_collections); model = `BSRoformer(**config["model"])`; weights
   `torch.load(ckpt, map_location="cpu", weights_only=True)` and **`load_state_dict(..., strict=True)`** (verified
   0 missing / 0 unexpected with `rotary-embedding-torch==0.9.1`); verify the file sha256 against
   `sep.ckpt_sha256` before loading. dtype upstream = fp32 weights + `torch.autocast("cuda", dtype=torch.float16)`
   (yaml `use_amp: true`); `refuse_bf16`.
4. **Worker `sep_msst`** (§7.2) with the §8 ladder; stems in yaml `training.instruments` order; `input_gain_db`
   applied before / inverted after. After the model is on the GPU: `ctx.sampler.mark("model_loaded")`,
   `Watchdog.arm()`.
5. **Stages** `sep`, `stems` in `gtab/sep/stage.py` (`sep` params exactly as §3.2 row 4; writes `gpu_run.json`);
   `gtab/sep/derive.py` streams blocks (M0 RAM rule: never hold six stems at once): `nonvox = mix − vocals − drums
   − bass`, `leftover = mix − Σ6 stems`, `mix_minus_guitar = mix − guitar`, `mix_minus_bass = mix − bass`, mono
   views `(L+R)/2` of mix, guitar, bass, `piano + other`, nonvox, `guitar + other`; `stem_stats.json`,
   `energy.npz`. Leftover rel RMS > `leftover_warn_db` → warning in stats and in the run output (not a failure).
6. **`gtab/sep/run.py`** — the one entry point for separation outside jobs (AMT's E1/E24-sep, EVAL scenes):
   `separate(mix_wav: Path, out_dir: Path, cfg, *, force_rung: str | None = None, input_lufs: float | str | None =
   None, dtype: str | None = None) -> SepResult` (`SepResult`: `stems: dict[str, Path]`, `rung`, `sep_json: dict`).
   Same request builder and `run_gpu_stage` as the stage; overrides default to the config values.
7. `gtab/sep/archive.py`: `archive(wav, flac) -> gain` (peak-normalise to −0.1 dBFS, 24-bit FLAC, gain in a
   sidecar `.gain.json`), `restore(flac) -> np.ndarray`; `roundtrip_error_db(wav, *, ref_rms_db: float) -> float`
   = error RMS after un-gaining **relative to `ref_rms_db` = the mix's RMS** (not the stem's own RMS: 24-bit
   quantisation noise after peak normalisation is ≈ −149 dBFS RMS, so a sparse stem with crest factor > ≈ 49 dB
   would fail a self-relative −100 dB test). Stems whose RMS is below −60 dB relative to the mix are reported as
   `skipped_sparse` rather than failed. (CLI `gtab clean --archive` is later; M2 needs the function + measurement.)
8. `gtab/sep/checks.py`: `stereo_preservation(true_stereo, est_stereo, sr) -> dict` — per-bin coherence and ILD
   (via `gtab.analysis.stereo.coherence_ild_maps`) of the true guitar-only stem vs the SW guitar stem, energy-
   weighted mean |Δcoh|, |ΔILD| dB, and `window_features` deltas.
9. `gtab/sep/experiments.py` `EXPERIMENTS = {"stereo-preservation": ...}` (b7 real-stem archive error + b8; §10,
   §11). **E1 and E24-sep are AMT's** (they are scored by MuScriptor vs reftx); SEP supplies `separate()`.
10. **Bench** `gtab/bench.py` + `gtab/commands/bench.py` (§5): for each backend × rung run the worker with
    `force_rung` and `precheck: false`, record `vram.attempts`, top-level cuda/sysmon, `ctx_mb`, rtf, load time;
    write `data/bench/<YYYYMMDD-HHMMSS>_<gitsha7>/bench.json` + `bench.csv`; merge uncapped, ballast-free `ok`
    rows into `vram_table.json` (median over `--repeat`). `bench.json` records at start the NVML process list with
    per-process used MB (`null` where WDDM reports N/A) — the Claude app itself (claude.exe, msedgewebview2) holds
    VRAM and cannot be closed while it drives gtab, so b1/b9 evidence states what was running.
    - `--cap-mb` = OOM-ladder validation (b4: expects `oom` on large rungs, success lower down). It cannot show
      Sysmem Fallback (the PyTorch allocator raises before real memory runs out).
    - `--ballast-mb N|auto` (b3): `gtab/workers/vram_ballast.py` (gpu env) is started with
      `winjob.spawn_in_job` in **its own Job Object, outside the GPU lock**, before the measured worker. It
      allocates N MB with `torch.empty(..., dtype=torch.uint8, device="cuda").fill_(1)` in 256 MB blocks
      (touching commits real VRAM), writes `ready.json` (`{"held_mb"}`), then sleeps until a `stop` file appears
      in its work dir, its parent pid dies, or `max_s` passes. Bench checks NVML used rose by ≈ N (±10 %) before
      continuing (else aborts the run with a Korean message) and closes the job afterwards (kill-on-close
      guarantees cleanup). `auto` = NVML free − (orchestrator need of rung 0 − 400 MB), i.e. rung 0 does not fit
      but lower rungs may. b3 runs `--rungs auto --ballast-mb auto` once with the NVIDIA setting off and once on
      **[사용자]**.
    - Calibration: over all `--fallback off` runs, the maximum `pdh_self_shared_growth_mb` is stored as
      `calibration.shared_growth_noise_mb`; bench prints the recommended `gpu.shared_growth_fail_mb =
      max(64, 2 × noise)` (an integration request if it differs from 64).
    - Backends `amt` and `beats` call AMT's/GRID's workers with their documented requests (no import of their code).
11. `gtab/doctor.py` two additive non-required checks: `gpu.vram_table` (present / age / entries) and
    `models.present` (every `models.lock.json` entry exists; `beat_this.final0` missing → hint
    `gtab models fetch beat_this.final0`).

Tests: `test_vram.py` (choose_rung with `RungNeed`, orchestrator vs worker need never mixed, wait policy with fake
clock/snapshots, `cpu` policy honours `cpu_backends`, `run_gpu_stage` retry bound with a fake `run_worker`,
`gpu_run.json` round trip + `degraded` — no GPU), `test_gpu_common.py` (`gpuenv`: ladder with a fake attempt that
raises OOM; all GPU rungs fail + CPU not allowed → `(None, attempts)`; Watchdog with a fake sampler: growth
measured from `arm()`, adapter growth ignored, slow → warning only, slow + PDH unavailable → SharedGrowth),
`test_sep_demix.py` (`gpuenv`: `selftest_identity` reconstruction error < 1e-5 on a 30 s random stereo file, chunk
262144 and 588800), `test_sep_vendor.py` (`gpuenv`: vendored files match `VENDOR.md` sha256; strict load of a
tiny random-weights BSRoformer state dict; importing `gtab.third_party.msst` does not import librosa and emits no
`sdp_kernel` FutureWarning on CPU), `test_sep_upstream_parity.py` (`gpu`, `slow`: vendored model + our demix vs
upstream MSST `demix` from a pristine checkout of `msst_commit` at `data/scratch/msst_ref/` (skipped if absent) on
a 20 s clip → SNR > 60 dB per stem), `test_sep_derive.py` (fake stems on disk: nonvox/leftover/practice identities
exact, mono views incl. `guitar_other`, byte-determinism of two runs (data outputs per §0; the test hashes files
itself rather than importing GRID's runner), block streaming on a
file longer than 2 blocks), `test_sep_archive.py` (roundtrip < −100 dB rel. mix, peaks > 1.0 handled, **a sparse
synthetic stem at −55 dB rel. mix with crest factor > 50 dB passes** and one below −60 dB is `skipped_sparse`),
`test_sep_run.py` (fake `run_gpu_stage`: overrides reach the request; `input_lufs` → gain math),
`test_sep_worker.py` (`gpu`, `slow`: 20 s clip → 6 stems, result has every §7.1 field, `gpu_run.json` written),
`test_bench.py` (table merge: capped/ballasted/failed rows never merged, median over repeats, calibration value;
ballast launcher with a fake worker that writes `ready.json`).

### 9.2 AMT — transcription backends, S3.5 v0, MIDI (M1a bp310; M2)

1. **First:** after P0's env sync, run `amt_muscriptor` in `"probe"` mode and write the (already verified, see
   facts) API findings into the worker's module docstring: §7.3 call path, `ProgressEvent` from
   `muscriptor.events`, tensor `[C,T]` float32, hard-mask semantics and **order-dependence** of `instruments`,
   device-at-build + built-in fp16 autocast (conditioners fp32), and `MT3_GROUPS` (§6.5) equal to the installed
   table (test with `gpuenv`).
2. **bp310 env** (M1a): `envs/bp310/pyproject.toml` (`requires-python = ">=3.10,<3.11"`, `[tool.uv] package =
   false`, deps `basic-pitch==0.4.0`, `onnxruntime`, `soundfile`, `numpy`, `filelock`, `pydantic`; a comment notes
   the 3.10 resolution `onnxruntime<=1.23.2`, `numpy<=2.2.6`, `scipy<=1.15.3`); install Python 3.10 with
   `tools\uvw.cmd python install 3.10` (goes to `D:\gtab\tools\python`, the global 3.10 is never touched); lock +
   sync from `D:\gtab\envs\bp310`. gtab is **not** installed there (requires 3.12); workers import it through the
   `PYTHONPATH` that `gpu.run_worker` sets. Record the bundled ONNX model (`basic_pitch/saved_models/icassp_2022/
   nmp.onnx` in the bp310 site-packages) as `basic_pitch.icassp2022` in `models.lock.json`; `amt_bp`'s `models()`
   hashes that file, or returns `{"basic_pitch": "absent"}` if `paths.BP310_PYTHON` does not exist (§3.2 row 9).
3. Workers `amt_muscriptor` (§7.3; default loader `upstream`, optional `lean`; `ctx.sampler.mark("model_loaded")`
   + `Watchdog.arm()` after load; `refuse_bf16`) and `amt_basicpitch` (§7.4; `transcribe` and `sweep` modes).
4. `gtab/amt/instruments.py`: family ↔ class maps (from the schema), hint parsing (`"guitar,keys"` → classes;
   unknown family → Korean ConfigError), `sort_mt3(classes) -> list[str]` (MT3 id order; used for every request and
   cache key), mask builders: `mask_bass()`, `mask_piano_other(mode)`, `mask_guitar_view(instrumentation, policy)`
   with policy `amt.guitar_mask`: `guitar_only` = 3 guitar classes; `guitar+present` (default) = 3 guitar classes ∪
   present keys/synth classes; `all` = `None`. A guitar class is never dropped.
5. `gtab/amt/instrumentation.py` — S3.5 v0 estimator (hand rules **[잠정]**, E3 calibrates):
   - mix transcription evidence per class: `active_ratio` = fraction of bars (from sections/grid bars via
     `sections.json` bar times) with ≥2 notes of that class; `e = clip(active_ratio / 0.1, 0, 1)` if ≥
     `mix_class_min_notes` notes else 0.
   - stem energy evidence: fraction of bars whose stem RMS is above `active_rel_db` relative to the mix: piano stem
     → keys classes; other stem → synth/organ/strings/brass/winds (weak, ×0.5); guitar stem → guitar classes.
   - noisy-OR `p = 1 − Π(1 − w_s e_s)` with `w_mix = 0.9`, `w_stem = 0.6` **[잠정]**; hints: listed families
     `p = max(p, 0.99)`; with an explicit list, unlisted families `p = min(p, 0.05)` (guitar excepted).
   - present = `p ≥ instr.threshold` (`guitar_threshold` for guitar classes). **If the guitar stem is active in
     ≥ `guitar_stem_active_ratio` of bars, all three guitar classes are present** (DESIGN S3.5 start value).
   - `per_section` family activity; `warnings` (e.g. "건반 에너지는 있는데 믹스 전사에 건반 음이 없음").
   - `fast` profile: mix evidence absent.
6. Stages `amt_ms1`, `amt_bp`, `instr`, `amt_gtr`, `s35`, `notes` (`gtab/amt/stage.py`) with the cache (§3.2); GPU
   stages via `gtab.vram.run_gpu_stage` and `write_gpu_run`. `amt_bp` is optional (§3.2 row 9); `s35`/`notes`
   treat its outputs as absent when `skipped.json` exists.
7. `gtab/amt/midi.py`: `write_performance_midi(path, tracks: list[(name, program, NoteSet)], grid: Grid | None)`
   via mido, ticks_per_beat 480 (per **quarter note**, as MIDI tempo is always µs per quarter), time signatures from
   `grid.bars`, deterministic event order, velocity 90 when absent, one track per name. Time 0 of the MIDI = time 0
   of the audio. Tempo: one `set_tempo` per grid beat interval so DAW bar lines follow the song; a beat of unit
   *u* and duration *d* s spans `480 × {quarter: 1, dotted_quarter: 1.5, eighth: 0.5}[u]` ticks (6/8: 720 ticks per
   dotted quarter) and its tempo is `round(d / factor[u] × 1e6)` µs per quarter.
   **Lead-in:** if the first grid beat `t0` is > 0, insert lead-in bar(s) before it so that the pickup bar (if any)
   and then bar 1 start exactly on their beats: one bar of `1/4` with tempo `t0` s per quarter (= 60 / t0 BPM) when
   `t0 ≤ 2 ×` the first beat interval; otherwise `k/4` with `k = ceil(t0 / first interval)` and tempo `t0 / k` s per
   quarter. The pickup bar (bar 0) gets its own `time_signature` (`n_pickup_beats/<denominator>`). An arbitrary gap
   (e.g. 0.37 s = 0.74 beat) cannot be expressed as a partial bar via `time_signature` alone — hence the lead-in.
   `notes_summary.json` records `midi_bar_offset` (number of MIDI bars before musical bar 1, i.e. lead-in +
   pickup). Without a grid: 120 BPM constant, no lead-in. `guitar_all.mid` = guitar classes of the `amt.guitar_view`
   MuScriptor transcription, one track per class present, GM programs (0-based) acoustic 25, clean 27, distorted 30;
   `bass_raw.mid` = bass classes of `bass_mono` (electric 33, acoustic 32); `b0_guitar.mid` = guitar classes of the
   mix transcription merged into one track (baseline B0, DESIGN 8.5).
8. `gtab/amt/latency.py`: `spectral_flux_onsets(wav) -> np.ndarray`; `estimate_latency(notes, ref_onsets, max_ms=100)
   -> {"median_ms","mad_ms","n"}`; experiment `latency` (pre-registered split, so the criterion does not grade
   itself): **estimate** the constant on a dev set = GuitarSet players `00`–`02` (+ the EGDB dev split if
   available); **report** the median absolute onset residual after correction on held-out GuitarSet players
   `03`–`05` (block bootstrap CI over tracks); flux onsets on the same clips as a cross-check. Set `amt.latency_s`
   (in `defaults.toml` via an integration request) to the dev GT-based median. Cross-check only: muscriptor's own
   `transcribe_and_postprocess` estimate (constant-tempo grid, DESIGN S6).
9. `gtab/amt/reftx.py`: `reference_transcribe(track: DatasetTrack, out_dir, cfg) -> dict[line, NoteSet]` — mono
   downmix of each true solo guitar/bass part → MuScriptor (guitar parts: 3 guitar classes; bass: bass classes)
   with the reftx cache (rung label in the key). Used by `gtab eval reftx` (EVAL CLI) — "Tier B 참 단독 트랙의
   MuScriptor 참조 전사".
10. `gtab/amt/experiments.py` `EXPERIMENTS = {"E1", "E2", "E3", "E23", "E24-sep", "E24-amt", "latency", "gate-m2"}`
    (§10). E1/E24-sep call `gtab.sep.run.separate(..., force_rung=<top rung>, input_lufs=/dtype=)` on
    `gtab.eval.remix.tierb_mix(...)` inputs and score MuScriptor (via AMT's own worker, `force_rung="medium"`) on the
    SW stems against reftx. E23 uses `amt_basicpitch` mode `sweep` (one model pass per track). Every metric row
    carries `rung` (§6.10).

Tests: `test_amt_events.py` (event stream fixture → NoteSet; unterminated notes; sorting), `test_amt_cache.py`
(key sensitivity incl. rung label and instrument order — `["b","a"]` and `["a","b"]` give the same key because both
are sorted; only missing views sent — fake `run_worker`), `test_amt_masks.py` (guitar classes never dropped under
all three `guitar_mask` policies; hint parsing; `sort_mt3`; E3 threshold edges), `test_amt_instrumentation.py`
(synthetic inputs: keys present via stem only, via mix only, via hint; explicit list suppresses; guitar-class
omission count 0 over a randomized property test), `test_amt_midi.py` (write → mido read: tempo map, pickup bar,
**lead-in bar for t0 = 0.37 s at 120 BPM puts bar 1 exactly on the first downbeat (±1 tick)**, long-silence
`k/4` lead-in, **6/8: 720 ticks per dotted-quarter beat with tempo per quarter**, deterministic bytes, notes at the
right seconds ±1 ms, `midi_bar_offset`), `test_amt_latency.py` (synthetic onsets shifted by 25 ms → estimate
25 ± 1 ms; dev/test split never overlaps), `test_amt_bp_params.py` (config frames → `min_note_len = frames − 1`;
`min_note_frames == 12` rejected unless allowed), `test_amt_worker_bp.py` (`bp310`, `slow`: echo import on 3.10; 5 s
synthetic melody → notes with bends present; **a synthetic model output with 4- and 5-frame notes: `min_note_frames
= 5` keeps the 5-frame note and drops the 4-frame one**; `sweep` over 3 settings loads the model once and matches 3
separate `transcribe` runs; **two runs → byte-identical NoteSet files**), `test_amt_worker_ms.py` (`gpu`, `slow`:
10 s synthetic guitar-ish clip → ≥1 note, tensor path used — `output.api.input_kind == "tensor"`; `probe` mode lists
35 groups; bf16 request refused; small fallback absent → attempt `unavailable`, no network access;
`test_lean_loader_parity`: lean and upstream loaders give identical notes on the clip and lean's
`max_reserved_mb` is lower — the gate for switching the default).

### 9.3 GRID — S2, S3.5 CPU parts, pipeline, `gtab run`

1. Worker `beats_beatthis` (§7.5); stage `beats`: `models(cfg)` = `{"beat_this.final0":
   sha256_cached(gtab.models.local_path("beat_this.final0"))}` (raises `ModelMissing` → plan-time Korean error
   naming `gtab models fetch beat_this.final0`); the stage passes the local path and never records or downloads
   models itself; writes `gpu_run.json`.
2. `gtab/analysis/onsets.py`: mix → mono 22050 Hz (soxr) → `librosa.effects.hpss` on the STFT (n_fft 2048,
   hop 256; margin 2.0 **[잠정]**) → percussive onset strength (`librosa.onset.onset_strength`) + peak picking →
   `onsets.npz` (env, times, fps). **Full mix only** — drum stems are not used (DESIGN S2).
3. `gtab/analysis/grid.py` (`derive_grid(beats_raw, activations, onsets, params) -> Grid`), in this order:
   1. downbeat check: Beat This!'s minimal postprocessing already snaps downbeats to beats, so this step only
      verifies every downbeat coincides with a beat (±1 ms); violations (e.g. after a future DBN option) are
      snapped within `downbeat_snap_ms` or dropped, and counted in confidence;
   1b. **one metrical level per song** (added 2026-10-03 after the first real songs; see the `grid.py`
      docstring): Beat This! switches between full and half tempo *within* a song (AIZO, 絶対零度, 青と夏), which
      a song-wide half/double decision cannot repair. Intervals near half the duration-weighted median interval
      drop every other beat (runs of >= 2), intervals near twice it get their midpoint (±19 % tolerance, so real
      tempo changes and swing stay); missed beats are filled;
   2. **tempo octave** (half/double, DESIGN S2): candidates {T/2, T, 2T}; score = fraction of percussive onsets
      within ±70 ms of candidate beats × log-normal tempo prior (centre 120 BPM, σ = 0.5 oct) and onset-density
      ratio; switch only if an alternative beats "keep" by > 0.1; rebuild beats (midpoints / every other,
      downbeats preserved);
   3. **refinement** (matters: Beat This! times are quantised to 20 ms): each beat → strongest percussive onset
      within ±`refine_window_ms` if its strength ≥ local median ×1.5, else unchanged; `shift_ms` recorded;
   4. bars from downbeats; beats before the first downbeat = pickup bar 0;
   5. **meter**: bar lines are a minimum-cost (dynamic-programming) segmentation of the beats over Beat This!'s
      downbeat probabilities; an odd bar costs about three bars of contrary evidence, so it is kept only when the
      following bars confirm the new phase; meter = mode of tracked bar lengths, overridden by a running mode
      over `meter_smooth_bars` only for a consistent non-divisor run (3/4 in 4/4); a song-wide 2 or 8 is read as
      half/double bars of 4 (recorded with a Korean warning) — not per section, since sections are computed from
      the grid (see §12 A7). Implemented 2026-10-03 (replaces the plain running mode);
   6. **compound check**: per beat, onset-envelope energy at 1/3, 2/3 vs 1/2 of the interval →
      `triple_ratio`; ≥ `compound_triple_ratio` → compound (6/8 or 12/8, beat unit dotted quarter; if Beat This!
      tracked eighths, i.e. 6 or 12 beats per bar grouped in threes, regroup 3 tracked beats per dotted-quarter
      beat); within ±`compound_margin` → `ambiguous`, both hypotheses kept in `alternatives` (UI toggle later,
      E14); triplet *feel* is decided later in M3;
   7. tempo per bar (median beat interval, 0.1 BPM), confidences from activations.
4. `gtab/analysis/sections.py`: beat-synchronous CENS (librosa `chroma_cens`) + MFCC (13, standardized) →
   recurrence/self-similarity with `librosa.segment.recurrence_matrix` (affinity) → Laplacian segmentation
   (McFee–Ellis; number of segments by eigengap in [2, `max_sections`]) → boundaries snapped to downbeats,
   segments shorter than `min_section_bars` merged → labels by clustering segment feature means → `sections.json`;
   lag matrix (`librosa.segment.recurrence_to_lag`) → per-bar best matches (score ≥ `repeat_min_score`) and group
   occurrences; `offset_s` per occurrence by cross-correlating percussive onset envelopes over ±`align_max_offset_ms`
   → `repeat_map.json`. **Determinism** (eigen-decomposition and KMeans can differ in the last bit across runs,
   especially with multithreaded OpenBLAS): the whole stage runs under
   `threadpoolctl.threadpool_limits(1)`; eigenvectors are sign-normalised (largest-|x| component positive) and
   ordered by (eigenvalue, index); KMeans uses `random_state=0`, `n_init=10`; labels are renamed A, B, C… in order
   of first appearance; scores rounded to 1e-4 and times to 1e-6 before writing.
5. `gtab/analysis/vocal.py`: vocals-stem RMS at 10 Hz relative to mix RMS, median-smoothed 0.5 s, active if >
   `vocal_active_rel_db`; per-bar active ratio.
6. `gtab/analysis/repetition.py` (**repeat residual pass 1, no solo gate**, DESIGN S3.5 step 5; **exact port of
   `docs/research/experiments/repet3.py`'s mask**, which is DESIGN's evidence for the 0 → 17 dB lead SIR gain): for
   each repeat group with ≥ `resid_min_occurrences` occurrences, guitar-stem mono STFT (n_fft 4096, hop 1024) of
   every occurrence warped onto the reference occurrence's beat grid (per-beat linear frame resampling, offset
   applied) → `V = |X|` stacked per occurrence → background `W = min(median over occurrences of V, V)` →
   `Mb = W² / (W² + max(V − W, 0)² + 1e-12)` → foreground (residual) `= X · (1 − Mb)` → ISTFT back into the
   original timeline (overlap-add per occurrence) → `repeat_resid_pass1.wav` + per-bar residual energy rel. to the
   guitar stem. The mask core is a pure function `background_mask(V_stack: np.ndarray) -> np.ndarray` (shape
   `(occ, freq, frames)`) so it can be parity-tested. Groups with fewer occurrences: `null` + `skipped` reason.
   `resid1.json` `method` = `"beat-aligned median background, repet3 soft mask (W=min(median,V))"`.
7. `gtab/analysis/stereo.py` is **EVAL's** file (router features); GRID does not need it in M2.
8. Pipeline: `gtab/pipeline/graph.py` (§3; lazy imports of the three `STAGES` modules with the Korean
   not-implemented error of §1.7), `runner.py` (§4 incl. `data_outputs`, plan-time `ModelMissing`, degraded-cache
   warning from `gpu_run.json`), `publish.py` (§3.3), `gtab/commands/run.py` (§5).

Tests: `test_grid_derive.py` (fake beats/onsets: constant tempo, tempo ramp, pickup 1–3 beats, half-time tracker
output → doubled, double-time → halved, 6/8 synthetic onsets at thirds → compound, 4/4 duple → simple, ambiguous
band, beat refinement pulls a 20 ms-late beat onto the onset), `test_grid_onsets.py` (synthetic click track +
sustained chords: onsets found within 10 ms), `test_sections_synth.py` (synthetic A B A B C A from distinct chord
progressions/timbres: labels A/B/C recovered, boundaries on downbeats, repeat offsets of a 30 ms-shifted repeat
recovered ± 5 ms; **two runs → byte-identical `sections.json`/`repeat_map.json`, also with BLAS allowed 4 threads
outside the stage**), `test_s35_vocal.py`, `test_s35_resid.py` (three occurrences of a riff + lead in two of them:
residual energy higher in lead bars by ≥ 10 dB), `test_s35_repet3_parity.py` (a **subprocess** harness imports
`repet_exp`/`repet3` generators from `docs/research/experiments/` — they rewrap stdout at import — rebuilds
repet3's signals, applies `background_mask` in place of repet3's inline mask and prints SIR JSON; the test also
runs original `repet3.py` and compares the `median` rows at 0/20/50 ms misalignment: **|ΔSIR| ≤ 0.5 dB**),
`test_pipeline_graph.py` (acyclic; names/deps match §3.2; GPU stage order beats → sep → amt_ms1 → amt_gtr; `fast`
drops the mix view; `--until notes` includes vocal/resid1; a missing stage module → Korean error),
`test_pipeline_run.py` (fake STAGES impls: cache hit on rerun, `--from` cascades, publish idempotent, **publish with
a locked old export** (file held open) leaves `.tmp-export-old-*` or fails cleanly per §3.3, exported MIDI is a
copy (editing it leaves the stage file unchanged), exported WAVs read-only, CPU byte determinism of two forced
runs compared with `data_outputs` (ignores `_worker/**` and the manifest), plan-time `ModelMissing` lists every
missing model before any stage runs, degraded cached GPU stage → warning event), `test_grid_worker.py` (`gpu`,
`slow`, `model("beat_this.final0")`: 30 s 120 BPM click+bass synthetic → beats within 20 ms, downbeats correct;
checkpoint loaded from the local path with `weights_only=True`).

### 9.4 EVAL — measurement harness (M1a) and experiment framework (M2)

1. `gtab/eval/matching.py` + `metrics.py` (DESIGN 8.4 items 1, 2, 3, 4, 7, 10, and 11 which M1a names).
   Note inputs are `(intervals (n,2) seconds, pitches (n,) MIDI)` tuples (`gtab.schema.notes.notes_to_arrays`);
   line-aware metrics take events built by `events_from_gt(GtNotes | DatasetTrack)` and
   `events_from_prediction(Prediction)`. mir_eval is imported lazily (§0).
   - `note_prf(ref, est, *, onset_tol=0.05, pitch_tol_cents=50, offset_ratio=None, chroma=False) -> PRF`:
     **pitches are converted MIDI → Hz with `mir_eval.util.midi_to_hz` before any mir_eval call** (mir_eval expects
     Hz; with raw MIDI numbers 61 vs 60 comes out as ≈ 29 "cents", so semitone errors would count as hits — and the
     a3 octave test would still pass, hiding the bug). Counts come from
     `mir_eval.transcription.match_notes(ref_int, ref_hz, est_int, est_hz, onset_tolerance, pitch_tolerance,
     offset_ratio, offset_min_tolerance)`: `tp = len(matching)`, `fp = n_est − tp`, `fn = n_ref − tp`; P/R/F1 are
     computed from the counts (the pooled block bootstrap needs counts; `precision_recall_f1_overlap` returns only
     ratios). Chroma variant: pitches mod 12 mapped into one octave (MIDI 60–71) before the Hz conversion.
     `PRF = (tp, fp, fn, precision, recall, f1)`. Variants reported: 50 ms onset-only, 100 ms onset-only, 50 ms
     with offsets (`offset_ratio=0.2, offset_min_tolerance=0.05`).
   - `tick_prf(gt: GtNotes, est, tempo_map: TempoMap, *, sections, tol_s=0.05)`: predictions mapped with the
     **GT** tempo map into (bar, tick); a match needs equal pitch and |Δ| ≤ the tick range equivalent to ±50 ms at
     that position (`ticks_per_second`); only notes inside the GT sections (half-open bar ranges) count; bipartite
     maximum matching. The system's own grid is never used (DESIGN 8.4 #1).
   - `notation_match(gt, est_bar_tick, bar_map) -> {"exact_ratio","n"}`: exact (bar, tick) equality after mapping
     system bars to GT bars by nearest downbeat time (reported separately, DESIGN 8.4 #2).
   - `octave_error_rate = chroma_f1 − pitch_f1` (#4).
   - `assignment(gt_events, est_events) -> AssignResult` (#7, DESIGN 4.11): dedupe GT notes with equal (pitch,
     onset ±1 tick) into events with `valid_lines`; dedupe predicted notes the same way into events with
     `pred_lines`; match events line-agnostically (50 ms, pitch); confusion C[pred_line, gt_line] over matched
     events; Hungarian mapping π (Unassigned never mapped); per GT line r: accuracy over matched events with
     r ∈ valid_lines, correct iff some mapped pred line ∈ valid_lines; **Unassigned = error**; macro = mean over GT
     lines; note-weighted accuracy as reference; `unassigned_ratio` (#10); `label_swap_raw` (identity mapping,
     for the swap test).
   - `coverage_accuracy(gt_events, est_events, taus) -> list[(tau, coverage, accuracy)]` (posterior < τ →
     Unassigned; accuracy conditional on assigned; single point when no posteriors).
   - `shared_prf(gt_events, est_events, sections)` in unison-tagged sections (#11).
2. `gtab/eval/stats.py`: `block_bootstrap(items: list[ItemCounts], stat: Callable, n=10000, seed) ->
   (point, lo, hi)` resampling items (songs / tracks; `group` field) with replacement and recomputing the statistic
   from pooled counts; `paired_bootstrap(a, b, stat, ...) -> (delta, lo, hi)`; `decide(delta, lo, hi, mde,
   subsets) -> "adopt"|"reject"|"hold"` implementing DESIGN 8.1 (CI excludes 0 and point ≥ MDE and no scenario
   subset with CI upper < −2 points). numpy `Generator(PCG64(seed))`, deterministic.
3. `gtab/eval/baselines.py`: `majority(pred)` (all assigned notes → one line), `no_split` (guitar stem
   transcription as one line), `b0` (mix transcription guitar classes as one line). Symbolic-split baseline is M6.
4. `gtab/eval/remix.py` (DESIGN 8.2 synthetic; port of `docs/research/experiments/synth_exp.py` +
   `router_exp.py`): generators `tone`, `amp_cab`, `rhythm_take`, `lead_line`, `twin_lead`, `pan`,
   `stereo_reverb` ported **with identical RNG call order**, extended to also return their note lists
   (MIDI pitch, onset incl. jitter/strum offsets, duration) as GT.
   ```python
   SCENARIOS = ("A","A_rhythm_only","B","F","ADT","quad","P70","P_hard","twin","twin_double",
                "same_part_double","centre_unison")
   @dataclass class Take: id: str; line: str; stereo: np.ndarray; notes: list[NoteEvent]
   @dataclass class Scene: scenario: str; seed: int; sr: int; mix: np.ndarray; takes: list[Take];
                           lines: list[GtLine]; gt: GtNotes; meta: dict
   def make_scene(scenario, seed=0, *, dur_s=8.0, lead_gain_db=0.0, reverb=False, full_band=False) -> Scene
   def parity_scenes() -> dict[str, Scene]      # the 14 rows of router_exp.main(), same seeds and RNG order
   def write_scene(scene, out_dir) -> Path       # mix.wav, takes/*.wav (write_f32), gt_notes.json, scene.json
   def tierb_mix(track: DatasetTrack, scenario="original", master_lufs=-8.0) -> np.ndarray   # M2 needs "original"
   def master(x: np.ndarray, sr: int, *, lufs: float = -8.0, ceiling_dbtp: float = -1.0) -> np.ndarray
   ```
   Line maps follow DESIGN R1–R9: doubles/ADT/quad = one line; P = two lines; twin = Lead 1 / Lead 2;
   twin_double = rhythm + Lead 1 + Lead 2; same_part_double = one line with two takes (self-agreement data);
   centre_unison = two lines with `shared` notes. `full_band=True` adds synthetic bass/kick/hat/vocal-like/pad so a
   scene can go through SW (M2 stereo-preservation check, later E6).
   `master` (used by `tierb_mix`; makes E1's inputs reproducible) is a **deterministic look-ahead brickwall
   limiter**: float64 throughout, no threads, no randomness. (1) static gain to `lufs` (pyloudnorm with the
   "DeMan" filter, as M0 `ingest/loudness.py`); (2) true-peak estimate per input sample = max |x| over channels and
   the 4 phases of a 4× oversampled signal (`scipy.signal.resample_poly(x, 4, 1)`); required gain `g = min(1,
   10^(ceiling/20) / peak)`; sliding minimum over a 5 ms look-ahead window; release = one-pole towards 1 with
   τ = 80 ms (attack instantaneous); gain applied to the signal delayed by the look-ahead, delay then trimmed so
   output is time-aligned with input; (3) re-measure and repeat (1)–(2) at most 3 times until within ±0.2 LU;
   (4) final sample-domain clip at the ceiling as a safety net. Returns float32.
5. `gtab/analysis/stereo.py`: `window_features(stereo, sr) -> dict` = exact port of `router_exp.features`
   (keys `SM, r0, xcoff, lag, coh, cen, hard, fcoh, ildcoh`); `coherence_ild_maps(stereo, sr, n_fft=4096,
   hop=1024, tau_s=0.15) -> (coh, ild_db, weight)`.
6. **GT tools** (`gtab/eval/{refscore,gp_import,render_ref,align,taps}.py`, `gtab/commands/gt.py`):
   - Neutral `RefScore` JSON (tracks, tuning, capo, masterbars with time signature, repeat open/close counts,
     alternate endings, notes with string/fret/duration in ticks@960 per quarter, ties, tuplets). **RefScore fixes
     the §6 conventions:** string 1 = highest-pitched string, `tuning[i]` = open pitch of string `i+1`, `pitch =
     tuning + capo + fret` stored explicitly per note. Two importers produce it:
     - PyGuitarPro 0.11 for `.gp3/.gp4/.gp5`: its string numbering already matches; **`Note.realValue` omits the
       capo, so the importer adds `Track.offset`**; `guitarpro.parse` decodes strings as cp1252 by default, which
       garbles Korean track names → use `reference.encoding` from gt.yaml if set, else try `cp949` (strict), then
       `cp1252`, and record the encoding used in `refscore.json` (names only; notes are unaffected).
     - Node for `.gp/.gpx` (and any format as a cross-check): `node/gtab_score.mjs import <in> <out.json>` using
       `@coderline/alphatab` 1.8.4 (`importer.ScoreLoader.loadScoreFromBytes`), run via `winjob.run_captured`.
       alphaTab numbers string 1 = **lowest** string → the script converts (`n_strings − s + 1`) and reverses the
       tuning list; alphaTab's own pitch (tuning + capo + fret) is emitted and used as the **reference** for capo
       handling.
     `node/package.json` pins `@coderline/alphatab` 1.8.4; install via `node.exe <nodejs>\node_modules\npm\bin\
     npm-cli.js ci` with `npm_config_cache=paths.NPM_CACHE` and notifier/fund/audit off (§0), through
     `run_captured`; `gtab/eval/node.py` is the one place that locates node/npm-cli and builds this env.
   - Expansion in Python (one implementation for both importers): repeats, alternate endings, or `bar_order` from
     gt.yaml; → expanded bars (1-based, pickup 0) and `GtNotes` at `tpb` 48 per beat unit; pitch from RefScore;
     tied notes merged; `shared` flags for unison notes across lines.
   - `gt import` writes `gt.yaml` skeleton (tracks from the file, one line per guitar/bass track,
     `families_present` guessed from track instruments, Korean comments telling the user to fill `sections`
     (half-open bar ranges), `families_present`, doubles, roles), ingests `--audio` with `gtab.ingest.ingest` to get
     the `song_key`.
   - `render_ref`: internal deterministic numpy synth (harmonic tones per line, `tempo_map` if present else nominal
     tempo) → `render.wav` (default). `--engine alphatab`: `gtab_score.mjs render <in> <out.wav>` headless in Node:
     `MidiFileGenerator` builds the MIDI, then `new AlphaSynth(<dummy output>, <ms>).exportAudio(options, midi, [],
     transpositions)` returns the exporter synchronously; the soundfont is `node_modules/@coderline/alphatab/dist/
     soundfont/sonivox.sf2` (path resolved at run time; missing → Korean message). Float samples are written as a
     float32 WAV by the script; `render_ref` rewrites it with `write_f32`.
   - `align` (independent of Beat This!, DESIGN 8.3; `align.py` must not import `gtab.analysis.grid` or any beat
     worker — test; synctoolbox/libfmp imported inside functions): audio: `audio_to_pitch_features` +
     `audio_to_pitch_onset_features` of the mix (mono 22050 Hz, feature rate 50); score: `df_to_pitch_features` +
     `df_to_pitch_onset_features` from the GT notes at nominal tempo; chroma → `quantize_chroma` / CENS, onsets →
     `pitch_onset_features_to_DLNCO`; without taps `sync_via_mrmsdtw(..., input_feature_rate=50,
     step_weights=[1.5,1.5,2.0], threshold_rec=10**6)`; **with taps `sync_via_mrmsdtw_with_anchors(...,
     anchor_pairs=[(t_score, t_audio), ...])`** using the tapped downbeats (score time = nominal time of that
     expanded bar's downbeat) — no piecewise re-anchoring after the fact; `make_path_strictly_monotonic`; map every
     nominal beat time through the path (linear between path points) → `tempo_map.json` (`source: dtw` or
     `dtw+taps`); `align_report.json` (per tap: residual after alignment, max drift, expanded bar count check vs
     `reference.expanded_bars`); `align_check.wav` = mix (−6 dB) + clicks at every GT note onset (accented at
     downbeats), float32 stereo.
   - `taps`: Audacity label file (`start\tend\tlabel`, UTF-8 or cp949 tolerated); label `b17`/`17` = expanded bar
     number, else sequential assignment to the nearest GT downbeat → `taps.json`, then re-align.
   - `approve`: sets `alignment.approved = true`, `approved_utc`, asks for minutes spent (optional flag).
7. `gtab/eval/external.py` + `import-external`: MIDI (per track) / GP (via RefScore) → `Prediction` stored in
   `data/eval/external/<slug(system)>/<slug(item)>.json`; `--system` ∈ `songsterr_ai|klangio|mvsep101|other`;
   `--map` maps source tracks to line ids.
8. `gtab/eval/suites.py`, `runs.py`, `report.py`, `experiments.py`, `gtab/commands/eval.py`:
   `runs.slug(s) -> str` (§0: lower-case, `[a-z0-9_-]` kept, `:` → `-`, anything else → `_`, collapse repeats,
   empty → ValueError) is used for every generated path. Systems: `oracle` (GT as prediction, sanity = 1.0),
   `oracle-noisy` (seeded deletions/octave/jitter/label swaps for harness checks), `job:<stage>` (NoteSets from a
   job's stage dirs; the stage's `gpu_run.json` rung goes into the `rung` column), `external:<name>`, `b0`,
   `no_split`, `majority`. Suites: `quick` (synthetic scenes + `oracle-noisy`, runs in < 60 s, the pre-merge gate of
   DESIGN 9.7), `synth` (all scenes), `bp-smoke` (GuitarSet: 20 tracks drawn with `seed`, Basic Pitch via AMT's
   worker, onset-only F1 at 50 ms; flagged "학습 데이터 포함(Basic Pitch 학습에 GuitarSet 사용), 판정용 아님"),
   `tierA`, `tierB`, `components` (runs registered experiments). `gtab eval exp <id>` imports
   `gtab.sep.experiments` / `gtab.amt.experiments` `EXPERIMENTS[id](out_dir: Path, cfg, args: dict) -> list[dict]`
   (metrics.csv rows, `rung` filled) and writes a run dir with the decision computed by `stats.decide` against the
   pre-registration in `docs/decisions.md`. **Runner states:** runs entries marked `사전 등록` (then flips the entry
   to `실행 중`) or `실행 중` (reruns allowed while undecided); refuses missing entries and `결정: …` entries (Korean
   message: a decided experiment needs a new pre-registered id to be re-examined). `gtab eval reftx <dataset>` wraps
   `gtab.amt.reftx.reference_transcribe` for every track.
9. `docs/decisions.md` — template + pre-registered entries whose `## <id>` headings are **exactly** the CLI ids:
   `E1`, `E2`, `E3`, `E23`, `E24-sep`, `E24-amt`, `gate-m2`, `latency`, `stereo-preservation` (the last two are
   descriptive: 결정 규칙 "서술(판정 없음)"; `latency` still pre-registers its dev/test split, §9.2 item 8). The
   `gate-m2` entry pre-registers the contamination rule of §10. Every later E# gets the same template.
   `docs/eval/contamination.md`: table datasets GuitarSet, EGDB, IDMT-SMT-Bass-ST, FiloBass, Cambridge-MT,
   MedleyDB, GOAT, musdb18(HQ), MoisesDB × models MuScriptor-medium, Basic Pitch 0.4.0, BS-RoFormer SW, Beat This!
   final0; cells `학습 포함` / `학습 제외` / `불명` with source link and date; filled from model cards and papers;
   unknown = `불명`; DESIGN 8.1 rule "불명 세트로 절대 성능 주장 금지" quoted at the top. Pre-verified row
   (2026-09-30, Basic Pitch paper arXiv 2203.09893 + Spotify engineering post): Basic Pitch trained on GuitarSet,
   iKala, MAESTRO, MedleyDB-Pitch, Slakh → GuitarSet `학습 포함`, MedleyDB `학습 포함` (MedleyDB-Pitch), EGDB
   `학습 제외`. MuScriptor × EGDB is expected to be `학습 포함` or `불명` (EGDB split track ids unpublished).

Tests (all CPU unless noted): `test_eval_metrics.py` — the M1a identities: identity F1 = 1.0; random 20 %
deletion → recall 0.80 ± 0.02 (seeded, ≥ 2,000 notes); +12 semitones → pitch F1 0, chroma F1 1.0; **every note
+1 semitone → pitch F1 = 0 (guards the Hz conversion)**; tp/fp/fn counts consistent with P/R/F1; label swap → 0
before Hungarian, 1.0 after; adding Unassigned never raises macro accuracy (property test); all notes in one line =
majority baseline; **tempo-warped synthetic performance: predictions at true times, system grid shifted by half a
beat and one downbeat → tick F1 unchanged (= 1.0)**. `test_eval_stats.py` (bootstrap deterministic with seed; CI
covers known mean in a simulation; `decide` truth table), `test_eval_determinism.py` (`gtab eval run --suite quick
--system oracle-noisy` twice → identical `metrics.csv`/`summary.json` sha256), `test_eval_remix_parity.py` (runs
the original `router_exp.py` generators in a **subprocess** with the core python — `synth_exp.py` rewraps stdout
at import — printing raw feature floats as JSON; compares with `window_features(parity_scenes())`: every feature
within ±0.02), `test_eval_remix_scenes.py` (line maps, note counts, shared flags, determinism), `test_eval_master.py`
(two `master` calls bit-identical; loudness within ±0.2 LU of target; 4×-oversampled peak ≤ ceiling + 0.1 dB;
output aligned with input), `test_eval_gp_import.py` (fixtures generated with PyGuitarPro in
`tests/fixtures/gp_make.py`: repeats ×2, alternate endings 1./2., pickup, 6/8, capo, two guitar tracks with a
unison bar, ties, triplets, a cp949 Korean track name → expected expanded GtNotes and decoded name; string numbers
follow §6), `test_eval_alphatab.py` (`node`: the same gp5 through the Node importer equals the PyGuitarPro path;
**capo pitches are checked against alphaTab's own pitch**, not against PyGuitarPro — a file PyGuitarPro wrote cannot
prove its own capo semantics; `render --engine alphatab` produces non-silent audio of the expected length),
`test_eval_node.py` (npm command line is `node.exe …npm-cli.js ci` with the cache env; never `npm.cmd`),
`test_eval_align.py` (`slow`: 32-bar synthetic score rendered with a known warp — tempo ramp 100→130 BPM plus ±3 %
sinusoidal drift — **≥ 95 % of downbeats within ±70 ms**; with 4 taps the anchored path passes through every tap
within one feature frame; static check that `align.py` does not import beat code), `test_eval_taps.py` (Audacity
parsing), `test_eval_runs.py` (`slug`: `job:amt_gtr` → `job-amt_gtr`, `guitarset:00_BN1` → `guitarset-00_bn1`, no
`:` in any generated path; runner state machine: `사전 등록`/`실행 중` run, `결정` refused, missing refused),
`test_eval_report.py` (report renders, no timestamps in metrics files, `rung` column present), `test_eval_cli.py`.

### 9.5 DATA — dataset registry, fetchers, loaders (M1a)

1. `gtab/datasets/registry.toml` (committed; the source of truth for URLs, sizes, checksums, licenses):
   | name | tier | access | files (required) | license | notes |
   |---|---|---|---|---|---|
   | `guitarset` | C | zenodo 3371780 | `annotation.zip`, `audio_mono-mic.zip` (optional: `audio_mono-pickup_mix.zip`, hex zips) | CC BY 4.0 | sizes/md5 in §"facts"; used for structure, string GT, BP smoke, latency (players 00–02 dev / 03–05 test) — not transcription judgment (in Basic Pitch's training data) |
   | `idmt_bass_st` | C | zenodo 7544099 | `IDMT-SMT-BASS-SINGLE-TRACKS.zip` (20.5 MB) | CC BY-NC-ND 4.0 | string/fret GT (M5) |
   | `egdb` | C | automatic (public Google Drive folder listing, resumable; ~9.1 GB) — updated 2026-10-03 | – | 불명 (not stated) | `gtab data fetch egdb`; E23 dev set + distortion gate |
   | `filobass` | C | zenodo 10069709 | `FiloBass_v1.0.0.zip` (434 MB) | CC BY 4.0 | M5; jazz upright |
   | `cambridge_mt` | B | automatic (plain-HTTP MTK archive host; the www site's 403 is avoided, no bot circumvention) — updated 2026-10-03; `import-local --song` still works | – | Cambridge-MT terms (educational) | `gtab data fetch cambridge_mt`; `lines.yaml` by ear **[사용자]**; 2+ rock songs for M1a, 4–6 by M6 |
   | `medleydb` | B | manual (Zenodo access request) **[사용자]** | – | CC BY-NC-SA 4.0 **[검증 필요]** | loader for a local copy only; MedleyDB-Pitch is in Basic Pitch's training data → flag in contamination |
   md5 from the Zenodo API is verified at download; sha256 is computed and stored in `datasets.lock.json`.
2. `gtab/datasets/fetch.py`: public `download_file(url, dest, *, expected_bytes=None, md5=None, progress=True) ->
   dict` (stdlib `urllib` to `<dest>.part` with HTTP Range resume, progress to stderr, size/md5 verification, sha256
   computed while streaming, then rename; returns `{"url","bytes","sha256","md5"}`) — reused by model fetch (item 7).
   Dataset fetch: `data/datasets/<name>/raw/<file>`; one plain attempt per URL (403/429/Cloudflare challenge →
   `ManualDownloadRequired` with the Korean manual steps: page URL, which files, where to put them, the
   `import-local` command); extraction (`zipfile` / `tarfile` with `filter="data"`) into
   `data/datasets/<name>/extracted/` via temp dir + rename; lock update under filelock. Confirmation prompt shows
   URL, size, license and destination (`--yes` skips).
3. `gtab/datasets/store.py`: lock file I/O, `status(name)`, `verify(name)` (sha256 recomputation), `import_local`
   (copy or hardlink into `data/datasets/<name>/local/`, record sha256; for `cambridge_mt` `--song` writes
   `lines.yaml` skeleton listing every WAV with a guessed kind from filename keywords — gtr/guitar, bass, vox, kick,
   snare, keys, synth, … — plus a `families_present:` list guessed from the same keywords; the user edits kinds,
   lines, takes, roles and families).
4. Loaders (`gtab/datasets/loaders/*.py`, each `iter_tracks(root, *, split=None) -> Iterator[DatasetTrack]`,
   `load_track(root, track_id)`); formats verified against the real files **[검증 필요]** except GuitarSet. Every
   loader fills `families_present` and converts string indices to §6's convention (1 = highest-pitched string):
   - GuitarSet: JAMS JSON parsed directly (no `jams` dependency): `note_midi` annotations per string (6; JAMS
     index 0 = low E → string 6) → notes + string numbers, `beat_position` → beats/downbeats; `group` = player id
     (`00`–`05`); comp/solo pairs via the track-id stem; `families_present = ["guitar"]`.
   - IDMT-SMT-Bass-Single-Track: per-track XML with onset/offset/pitch/string/fret (+ plucking/expression style
     in meta).
   - EGDB: DI + amp renders per track, MIDI/annotation labels; official split if provided, else by track id
     hash (seeded) — note in meta.
   - FiloBass: bass stem, performance-aligned MIDI, beats/downbeats.
   - Cambridge-MT: WAV stems + `lines.yaml` (by-ear line/role labels) → parts/lines; mix = sum of stems
     (or the provided mix if present).
   - MedleyDB: metadata YAML instrument labels (distorted/clean electric, acoustic guitar, electric bass), stems,
     melody f0 annotations (meta).
5. `gtab/datasets/__init__.py` public API: `list_datasets()`, `dataset_root(name)`, `is_available(name)`,
   `iter_tracks(name, split=None)`, `load_track(name, track_id)`.
6. `gtab/commands/data.py` (§5). `data list` columns: 이름, 계층, 라이선스, 크기, 상태(없음/받음/검증됨/수동 필요),
   위치.
7. **Model fetch** (DESIGN §9.4 `gtab models fetch|verify`): `gtab/model_sources.toml` (committed; name, url,
   dest relative to `GTAB_ROOT`, expected sha256 or empty, license, notes) with at least
   `beat_this.final0` = `https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt` →
   `data/models/beat_this/final0.ckpt`, license: Beat This! code MIT, checkpoint licence **[검증 필요 — record as
   found]**. `gtab/model_fetch.py`: `fetch(name, *, yes) -> ModelEntry` (prompt shows URL, size (HEAD), license,
   destination; `download_file`; if the source lists a sha256 it must match; else trust-on-first-use and the
   recorded sha becomes the pin) → `gtab.models.record(...)`; `verify(name|None)` → `gtab.models.verify`; `list()`
   merges `model_sources.toml` with `models.lock.json` (entries seeded by P0 show as `로컬 전용`, no URL).
   `gtab/commands/models.py` (§5) replaces P0's stub. Never called by `gtab run` or any worker.

Tests: `test_datasets_registry.py` (TOML schema, every entry has license/size/access), `test_datasets_fetch.py`
(local `http.server` in a thread: resume after a cut, md5 mismatch → error and no extracted dir, 403 →
`ManualDownloadRequired` with Korean text, atomic extraction), `test_datasets_loaders.py` (tiny synthetic trees
generated in tmp mimicking each layout; GuitarSet JAMS fixture with 2 strings → string numbers per §6;
`families_present` filled), `test_datasets_cli.py`, `test_model_fetch.py` (local `http.server`: fetch records
url/bytes/sha256/license in a temp `models.lock.json`; sha mismatch against a pinned source → error, nothing
recorded; `verify` detects a modified file), `test_datasets_network.py` (`network`, `slow`: fetch `idmt_bass_st`
(20 MB) + verify).

---

## 10. Pre-registered experiments (EVAL writes them into `docs/decisions.md` before any run)

Template per entry: `## E# 제목` / 상태 (`사전 등록` → `실행 중` → `결정: 채택|기각|판단 보류`) / 질문 / 선택지 /
주 지표 / MDE / 데이터(세트, 분할, 창 정의) / 분석(곡 블록 부트스트랩 10k, 쌍 비교, 시나리오 하위 세트) /
결정 규칙 / 비용 / 실행 명령 / 결과(run dir, 날짜). Metrics may not be changed after the first run.

Every experiment passes `force_rung` for each GPU backend it measures (top rung unless the experiment is about
rungs) and writes `rung` into its metric rows (§6.10); it never reuses `gtab run` stage caches for the measured
backend. The ids below are the `## <id>` headings in `docs/decisions.md` and the `gtab eval exp` ids, exactly.

| id | Choice | Primary metric | MDE | Data | Rule / notes | Runner |
|---|---|---|---|---|---|---|
| **E1** | SW input: original vs −14 vs −16 LUFS (`sep.input_lufs`; gain before SW, inverse after) | onset F1 (50 ms) of MuScriptor on the SW guitar and bass stems vs the **reference transcription of the true tracks** (reftx) | +1.0 | Tier B songs remixed to a commercial master (`tierb_mix(master_lufs=-8)`, deterministic limiter §9.4); Tier A GP songs descriptive | DESIGN 8.1 adopt rule; SDR diagnostic only | AMT `E1` (calls `gtab.sep.run.separate`) |
| **E2** | MuScriptor input (`amt.guitar_view`): mix / guitar stem / nonvox / guitar+other | guitar note F1 (50 ms onset) vs reftx (Tier B) and vs GT (Tier A, descriptive) | +1.0 (+1.5 if the option adds a pass) | Tier B, Tier A 2–3 | reftx is MuScriptor-derived: valid for comparing inputs, not for absolute claims | AMT `E2` |
| **E3** | `instruments` mask (`amt.guitar_mask`): guitar-only / +present keys-synth / all; S3.5 threshold | guitar-class F1 on keys-containing items; constraint: guitar-class omission = 0; b13 table: estimated families vs `families_present` | +1.0 | Tier B with keys, synthetic keys-bleed scenes, Tier A | pick the threshold maximizing F1 under the constraint on dev, confirm on test | AMT `E3` |
| **E23** | Basic Pitch shortest kept note `bp_min_note_frames` {3, 4, 5, 6, 7, 12} (= keep notes ≥ 35/46/58/70/81/139 ms; 12 = upstream 127.7 ms default) × onset {0.3,0.4,0.5,0.6} × frame {0.2,0.3,0.4} | onset-only F1 (50 ms) | – (tuning); confirm tuned ≥ default on test | EGDB (not in Basic Pitch training — verified, §facts) dev/test split by track group | `amt_basicpitch` mode `sweep`: one model pass per track, 72 postprocessing settings; choose argmax on dev; record params in manifest and `defaults.toml`; test result reported with CI | AMT `E23` |
| **E24-sep** | SW dtype: upstream (fp32 + AMP) vs fp16 weights | downstream MuScriptor stem transcription F1 vs reftx | non-inferiority margin −0.5; adopt fp16 only if CI lower bound > −0.5 **and** reserved peak −20 % or RTF +20 % | Tier B/C subset | default stays upstream | AMT `E24-sep` (calls `separate(dtype=...)`) |
| **E24-amt** | MuScriptor dtype: upstream (fp32 + built-in fp16 autocast) vs fp16 weights (conditioners fp32) | note F1 (50 ms onset) | as E24-sep | Tier B/C subset | default stays upstream; bf16 never | AMT `E24-amt` |
| **gate-m2** | primary transcriber check | onset F1, distortion subset, both tuned | MuScriptor − Basic Pitch ≥ +5 points, else "주 전사기 재검토" recorded | EGDB amp-rendered test tracks (official test split if one exists; the EGDB paper's 80/5/15 split publishes no track ids). EGDB is clean for Basic Pitch but may be in MuScriptor's training data. Tier B is **excluded**: its reference is MuScriptor-derived (reftx) and would favour MuScriptor | **pre-registered contamination rule:** if `contamination.md` has MuScriptor × EGDB = `학습 포함` or `불명`, the entry is recorded as `판단 보류 (오염 불명)` (or `(오염)`), never as a pass; the numbers are still reported with CI | AMT `gate-m2` |
| **latency** | MuScriptor latency constant | median absolute onset residual after correction (ms) | – (descriptive; b10 threshold 10 ms) | estimate on GuitarSet players 00–02 (+ EGDB dev); report on held-out players 03–05 | flux onsets = cross-check | AMT `latency` |
| **stereo-preservation** | – (M6 early look) | energy-weighted |Δcoh|, |ΔILD| per scene; real-stem archive round-trip error | – (descriptive) | synthetic `full_band` scenes through SW (top rung); real stems of ≥ 1 job | – | SEP `stereo-preservation` |

---

## 11. Completion criteria (copied from DESIGN §10) and the evidence that proves each

### M1a (Tier A 없이 끝낸다)

| # | Criterion (DESIGN) | Command / evidence |
|---|---|---|
| a1 | 지표 단위 테스트: 항등 F1 = 1.0 | `pytest tests/test_eval_metrics.py -q -k identity` |
| a2 | 무작위 20 % 삭제면 재현율 0.80±0.02 | `... -k deletion` |
| a3 | +12반음이면 pitch F1 = 0, chroma F1 = 1.0 | `... -k octave` |
| a4 | 라벨 교환이면 헝가리안 전 0, 후 1.0 | `... -k label_swap` |
| a5 | Unassigned만 늘리면 macro 정확도가 오르지 않는다 | `... -k unassigned_monotone` |
| a6 | 모든 음을 한 Line에 넣으면 다수 클래스 기준선과 같다 | `... -k majority` |
| a7 | 템포를 흔든 합성 연주에서 격자 오류가 음표 지표에 새지 않는다 | `... -k grid_leak` |
| a8 | 정렬 도구: 템포를 알려진 방식으로 흔든 합성 렌더에서 DTW가 다운비트의 95 % 이상을 ±70 ms 안으로 복원 | `pytest -m slow tests/test_eval_align.py -q` (prints ratio) |
| a9 | 부트스트랩 CI가 나오고, CPU 단계를 두 번 실행한 결과가 byte 단위로 같다 | `gtab eval run --suite quick --system oracle-noisy` ×2 → `ci_low/ci_high` filled; `Get-FileHash metrics.csv` equal (and `tests/test_eval_determinism.py`); pipeline CPU stages: `tests/test_pipeline_run.py -k determinism` and `tests/test_sections_synth.py -k determinism` (data outputs only, §0) |
| a10 | 리믹스가 router_exp 특징을 ±0.02 안에서 재현한다 | `pytest tests/test_eval_remix_parity.py -q` |
| a11 | Basic Pitch 스모크: GuitarSet 임의 20트랙에서 onset-only F1 > 0.6 (판정용 아님) | `gtab data fetch guitarset` → `gtab eval run --suite bp-smoke` → `summary.json` value (flagged: GuitarSet is in Basic Pitch's training data) |
| a12 | contamination 표에 MuScriptor·Basic Pitch·SW의 학습 목록을 모델 카드와 논문에서 채운다(모르면 '불명') | `docs/eval/contamination.md` reviewed, sources linked |
| a13 | 산출물: `gtab eval run|compare|import-external`, `gtab gt import|render|align|taps`, `eval/remix.py`, Tier B 로더(Cambridge-MT 2곡 이상 **[사용자]**, MedleyDB는 승인 시), Tier C 로더, bp310 venv, `decisions.md` | `gtab eval --help`, `gtab gt --help`; `gtab data list` shows guitarset/idmt/filobass fetched and ≥2 `cambridge_mt` songs imported; `pytest -m bp310`; `docs/decisions.md` has `E1`, `E2`, `E3`, `E23`, `E24-sep`, `E24-amt`, `gate-m2`, `latency`, `stereo-preservation` as `사전 등록` |

### M2 (첫 수직 슬라이스)

| # | Criterion (DESIGN) | Command / evidence |
|---|---|---|
| b1 | VRAM [판정]: 다른 앱을 닫은 상태에서 SW가 4분 곡을 처리할 때 reserved 피크 4.5 GB 이하, 실행 중 Shared Usage가 늘지 않는다(잠정) **[사용자: 앱 닫기]** | `gtab bench gpu --clip <4분 곡> --backend sep --rungs all --fallback off` → bench.json: rung 0 `max_reserved_mb ≤ 4608` (expected ≈ 1.8 GB), `pdh_self_shared_growth_mb ≤ 64`, `shared_growth=false`; bench.json's start process list documents what could not be closed (the Claude app itself) |
| b2 | 평소 앱을 켠 상태에서는 사전 점검이 예산에 맞는 칸을 고르거나, 이유를 밝히고 기다린다 | normal desktop: `gtab run <곡> --until notes` → log shows chosen rung + budget, or the Korean wait message; `tests/test_vram.py` covers the logic |
| b3 | Sysmem Fallback을 끈 상태와 켠 상태 둘 다에서 사다리 시험; 켠 상태에서 Shared Usage가 늘거나 실시간 배수가 끈 상태의 50 % 미만이면 실패로 보고 사다리를 내려간다 **[사용자: NVIDIA 설정 전환]** | **no `--cap-mb`** (the allocator cap raises OOM before the driver can fall back, so on/off would look identical): `gtab bench gpu --backend sep --rungs auto --ballast-mb auto --fallback off`, then the user flips the setting and runs the same with `--fallback on`. Pass: *off* → rung 0 `oom`, a lower rung `ok`; *on* → rung 0 `shared_growth` (this pid's shared growth after model load > threshold) → lower rung `ok`, and rung 0's `rtf` / `slow` flag reported against the off run's rung 0 (slowness alone warns; with PDH unavailable it counts as failure, §8.2) |
| b4 | `set_per_process_memory_fraction`으로 OOM 사다리를 검증한다 | `gtab bench gpu --backend sep --cap-mb 2500` → attempts `oom` then `ok` on a lower rung; `tests/test_gpu_common.py` |
| b5 | 실행한 칸마다 reserved·NVML 피크와 실시간 배수를 기록한다 | `data/bench/*/bench.csv` columns; `vram_table.json` |
| b6 | 스템 점검: leftover RMS가 믹스 대비 −20 dB 미만(넘으면 곡과 구간 기록) | `stems/stem_stats.json` per song; `gtab run` warning lists windows |
| b7 | float 스템 24-bit FLAC 보관 후 복원 오차 −100 dB 미만(게인 되돌린 뒤) | error measured **relative to the mix RMS** (stems < −60 dB rel. mix reported `skipped_sparse`): `pytest tests/test_sep_archive.py` (incl. a sparse stem); `gtab eval exp stereo-preservation` also reports it for real stems |
| b8 | 합성 A 믹스를 SW에 통과시켜 bin별 coherence·ILD 보존 편차 보고(M6 조기판) | `gtab eval exp stereo-preservation` → run dir with Δcoh/ΔILD per scene |
| b9 | MuScriptor-medium 전곡 reserved 피크 4.5 GB 이하, 실시간 배수와 모델 로드 시간 기록 | `gtab bench gpu --backend amt --clip <4분 곡>` → `max_reserved_mb` (expected ≈ 2.4 GB with the upstream loader), `rtf`, `load_s`, loader recorded; start process list as in b1 |
| b10 | 지연 보정 후 onset 잔차 중앙값이 10 ms 미만 | `gtab eval exp latency` → constant estimated on the dev split (GuitarSet players 00–02), `summary.json` `median_abs_residual_ms < 10` measured on **held-out** players 03–05 (with CI); `amt.latency_s` updated |
| b11 | E23 [판정]: Basic Pitch 최소 음 길이·문턱을 학습 밖 dev 세트에서 튜닝하고 manifest에 남긴다 | `gtab eval exp E23` → decisions.md 결정; `amt_bp` manifest params show the tuned values |
| b12 | 게이트 [판정]: 디스토션 부분집합에서 MuScriptor가 Basic Pitch보다 onset F1이 5점 이상 높지 않으면 재검토 | `gtab eval exp gate-m2` → decisions.md entry; if MuScriptor × EGDB is `학습 포함`/`불명` in contamination.md the recorded status is `판단 보류 (오염 불명)` — never a pass (pre-registered, §10) |
| b13 | S3.5 [판정/서술]: 편성 추정이 Tier B·Tier A 곡의 실제 편성(건반·신스 유무)을 맞히는지 곡별 보고; 기타 클래스를 마스크에서 빠뜨린 경우 0건 | `gtab eval exp E3` report table per song: estimated `families` vs ground-truth `families_present` (gt.yaml / DatasetTrack / lines.yaml); `tests/test_amt_instrumentation.py` property test |
| b14 | 캐시로 재실행하면 2분 안에 끝난다 | models fetched beforehand (`gtab models fetch beat_this.final0`), then the second `gtab run <곡> --until notes` → all stages `캐시` (no key changed: nothing downloads during a run), wall time < 120 s (`Measure-Command`) |
| b15 | 산출물: `sep_msst`, `amt_muscriptor`; S3.5 v0(편성, 원시 전사, 보컬 활동, 반복 잔차 1차, B0); `gtab run --until notes` → 6 스템, nonvox, leftover, instrumentation.json, guitar_all.mid, bass_raw.mid; 연습용 반주; `gtab bench gpu`, VRAM 사전 점검·대기, Shared Usage 감시; Tier B 참조 전사; 지연 상수; E1–E3, E23, E24 결정 (E24 = `E24-sep` + `E24-amt`) | `gtab run <곡> --until notes` on a real song → `export/` listing (incl. `practice/bass_stem.wav` from `sep`); decisions for `E1`, `E2`, `E3`, `E23`, `E24-sep`, `E24-amt`; `gtab eval reftx cambridge_mt`; decisions.md entries (E1/E2/E3 need Tier B data **[사용자]**; otherwise `판단 보류 (데이터 대기)` is the honest status) |

Plus (both milestones): full `pytest -q` green (M0's 370 + new), and `gtab doctor` still passes.

---

## 12. Ambiguities resolved here

- **A1 Phase P0.** DESIGN has no notion of parallel owners; shared files (deps, config, CLI, schemas) get one
  writer up front so owners never touch the same file.
- **A2 MSST vendoring location** `gtab/third_party/msst/` (task allowed this or `gtab/workers/_msst`); the torch
  boundary rule is extended to it.
- **A3 Own demix loop keeps overlap-add buffers on CPU**, unlike MSST HEAD (GPU buffers) — VRAM is the scarce
  resource on this PC; RAM cost ≈ 0.6 GB for a 4-min song.
- **A4 `sep_audiosep` deferred.** DESIGN lists it as "+"; it becomes the fallback only if the vendored path fails
  (then SEP raises it as an integration request).
- **A5 piano+other mask = keys/synth ∪ guitar classes** (`amt.piano_other_instruments = "keys+guitar"`): with a keys-
  only hard mask, guitar bleed in the other stem would be forced into keys classes and later penalize true guitar
  notes in S7. Only non-guitar notes of that view are used as bleed evidence. `"keys"` (DESIGN-literal) stays
  selectable; E3 can revisit.
- **A6 MuScriptor runs twice in pass 1** (`amt_ms1`: mix/bass/piano+other; `amt_gtr`: guitar view after
  instrumentation) because the guitar mask depends on instrumentation (DESIGN S6/E3). The extra load is measured
  in M2 (bench `load_s`).
- **A7 Meter smoothing without sections.** DESIGN derives the time signature as "구간별 최빈값", but sections are
  computed from the grid; the grid uses a running mode over 8 bars instead, and `sections` may annotate meter
  per section later. Avoids a cycle.
- **A8 Bar numbering** is musical everywhere (1 = first full bar, 0 = pickup), ticks 48 per beat unit; bar
  ranges are half-open everywhere (GtSection used to be inclusive); strings are numbered from the highest-pitched
  string (Guitar Pro convention; alphaTab and dataset indices are converted on import).
- **A9 Alignment uses symbolic score features** (synctoolbox `df_to_*`) against the mix, with tapped downbeats as
  DTW anchors (`sync_via_mrmsdtw_with_anchors`); rendering (`gt render`) is for listening/A-B, with an internal
  deterministic synth by default. alphaTab is mandatory for `.gp/.gpx` import and is the reference for capo
  semantics; its headless `AlphaSynth` export route is the optional `--engine alphatab` render.
- **A10 Latency constant** is estimated against GT onsets on a dev split (GuitarSet players 00–02, + EGDB dev) and
  reported on held-out players 03–05, cross-checked with spectral-flux onsets; the M2 check is the median
  **absolute** residual after correction (a signed median is ~0 by construction; estimating and grading on the
  same clips would grade itself).
- **A11 Basic Pitch minimum note length** is configured as the shortest **kept** note in frames
  (`amt.bp_min_note_frames = 4` ≈ 46.4 ms), because Basic Pitch drops notes whose length is ≤ its threshold: the
  worker passes `min_note_len = frames − 1` to `model_output_to_notes`. DESIGN's "30–58 ms (3–5 frames)" start
  range is read as kept lengths of 3–5 frames. The old "46.4 ms" value actually kept ≥ 58 ms.
- **A12 Deterministic WAV writer** (§1.4) because libsndfile's float WAVs are timestamped.
- **A13 Workers never import `gtab.schema`**, so the Python-3.10 bp310 worker cannot break on 3.11+ syntax in
  schema modules; validation happens core-side.
- **A14 `eval/contamination.md`** lives at `docs/eval/contamination.md` next to `docs/decisions.md`.
- **A15 `vocal`/`resid1` are pulled into `--until notes`** through the `s35` bundle stage.
- **A16 The GPU rung is provenance, not key.** Chunk size / model size / device are recorded per stage
  (`gpu_run.json`, §3.1) and in cache entries (`rung_label`), not in stage keys — keys stay stable across VRAM
  conditions, the runner warns about degraded caches, and experiments pin rungs with `force_rung`.
- **A17 CPU rungs only under policy `cpu`** and only for `gpu.cpu_backends` (default: Beat This!). Under the default
  `wait` policy nothing silently falls back to a 23-minute CPU separation; slowness alone never steps down the
  ladder (it is usually contention, not a memory spill).
- **A18 Two VRAM measurements, two checks.** Orchestrator: reserved + context + margin vs NVML free. Worker:
  reserved + 256 MB vs `mem_get_info` free. The context is no longer hidden in the margin or double-counted.
- **A19 Models are fetched explicitly** (`gtab models fetch`, DATA) — never inside `gtab run` or under the GPU lock.
  Keeps stage keys stable (b14) and keeps unsafe hub loading out.
- **A20 Transcription-scored experiments belong to AMT** (E1, E24-sep included); SEP exposes `separate()` and keeps
  only `stereo-preservation`. Experiment ids are the `decisions.md` headings (`E24-sep`, `E24-amt`, `gate-m2`).
- **A21 `amt_bp` is optional** (DESIGN: Basic Pitch is auxiliary): without the bp310 env the pipeline still reaches
  `notes`; the stage key changes once the env appears.
- **A22 Ground truth for b13** comes from `families_present` (GtSpec, DatasetTrack, `lines.yaml`) rather than from
  `Part.kind`/`GtTrack.kind`, which cannot express keys vs synth.

## 13. Open questions and user actions

1. ~~[사용자] Cambridge-MT manual download~~ — **resolved 2026-10-03 (DATA)**: `gtab data fetch cambridge_mt`
   downloads from the plain-HTTP MTK archive host (6 songs on disk). **[사용자]** remains: check the
   filename-guessed `lines.yaml` drafts (`reviewed: false`, incl. `families_present`) by ear. M2 E1/E2/E3 use them.
2. ~~[사용자] EGDB manual download~~ — **resolved 2026-10-03 (DATA)**: `gtab data fetch egdb --yes` downloads the
   public Drive folder (resumable, ~9.1 GB; license not stated → personal research use only, never redistributed).
3. **[사용자] MedleyDB** Zenodo access request (optional for M1a). Note: MedleyDB-Pitch is in Basic Pitch's
   training data.
4. **[사용자] Tier A**: 2–3 songs with a machine-readable GP file for M2 decisions (descriptive only).
5. **[사용자] VRAM tests**: close apps for b1/b9 (the Claude app itself cannot be closed; bench records what ran);
   toggle the NVIDIA "CUDA - Sysmem Fallback Policy" between the two b3 runs (per program on the base interpreter
   path shown by `gtab doctor`).
6. ~~SW checkpoint `weights_only=True`~~ — **resolved**: loads safely and strictly (critique probe).
7. ~~MuScriptor dtype~~ — **resolved**: fp16 autocast is built in when the model is built on CUDA; conditioners fp32;
   "upstream" = that; E24-amt tests fp16 weights.
8. **EGDB in MuScriptor training** (contamination) — still `불명` unless the model card/paper says otherwise; the
   gate-m2 rule (§10) handles it. Basic Pitch × EGDB is resolved: `학습 제외`.
9. **numpy downgrade** in core (2.5.3 → 2.4.x) for synctoolbox; if numba/py3.12 cannot resolve, pin lower.
10. Beat This! checkpoint license is not stated on PyPI **[검증 필요]** — DATA records what it finds in
    `model_sources.toml` / `models.lock.json`.
11. **MuScriptor lean loader**: switch `amt.muscriptor_loader` to `lean` only after `test_lean_loader_parity` passes
    (integration request from AMT). Not needed for the ~3.4 GB budget.
12. **DESIGN errata for the integrator** (DESIGN.md is not an owner file): S6 "minimum_note_length 약 30–58 ms
    (3–5프레임)" should say "keep notes ≥ 3–5 frames (Basic Pitch drops length ≤ threshold, so pass threshold =
    frames − 1)"; E23's grid should be stated in kept frames {3,4,5,6,7,12}.

---

## 14. Critique resolution (2026-09-30)

The critique pass re-checked the doubtful facts against the installed wheels (muscriptor 0.3.0, beat-this 1.1.0,
basic-pitch 0.4.0, synctoolbox 1.4.2, pyguitarpro 0.11, alphaTab 1.8.4), MSST at `84b1eac`, the PyPI/Zenodo/npm
APIs and three probes on this PC (two on the GPU, under `data\gpu.lock`; scripts in `data\scratch\critic\`). The
confirmed facts are folded into the header (✔ items) and their **[검증 필요]** tags were removed. Every numbered
point was accepted; none was rejected.

| # | Problem (short) | Resolution | Where | Owner |
|---|---|---|---|---|
| 1 | MuScriptor CPU→CUDA load breaks (device fixed at build; autocast only if built on CUDA) | `load_model(<local path>, device="cuda")` default; optional `lean` loader (private API, pinned version, parity-gated); CPU rung builds on CPU; bf16 refused | §7.3, §1.3 `amt.muscriptor_loader`, §9.2 tests | AMT |
| 2 | `DEFAULT_NEEDS` far too pessimistic | reseeded from measurements as `reserved_peak_mb` (SW 1800/1400/1200, medium 2400 / lean 1800, beats 500) + `ctx_mb` 500; note that b1/b9 pass easily and VRAM rarely binds | §8.1, facts | SEP |
| 3 | Context double-counted; NVML vs `mem_get_info` mixed | table stores `reserved_peak_mb` and `ctx_mb` separately; orchestrator: reserved+ctx+margin vs NVML free; worker: reserved+256 vs `mem_get_info` free; never cross-compared | §6.11, §7.1, §8.1, §1.3 `[gpu]` | SEP (P0 config keys) |
| 4 | b3 impossible with `--cap-mb` | `--ballast-mb N` or `auto` via `gtab/workers/vram_ballast.py` in its own Job Object outside the GPU lock; b3 = no cap + ballast, fallback off then on | §5, §9.1 item 10, §11 b3 | SEP |
| 5 | Shared-growth false alarms | decide on this pid's shared growth measured after the `model_loaded` mark only; adapter delta informational; threshold calibrated from fallback-off runs | §7.1, §8.2, §9.1 | SEP |
| 6 | CPU rung vs "wait"; TooSlow harmful | `sep.allow_cpu` removed; CPU only under policy `cpu` ∧ `gpu.cpu_backends` (default Beat This! only); all GPU rungs failing under `wait` → `insufficient_vram` + bounded wait/retry; slowness warns only (fails only when PDH is unavailable); never steps to CPU | §1.3, §7.1, §8.1–8.2 | SEP, P0 |
| 7 | Rung not in cache key | `gpu_run.json` per GPU stage + runner warning suggesting `--force`; `rung_label` in AMT cache keys; experiments use `force_rung` and a `rung` column | §3.1, §3.2, §4, §6.10, §10 | SEP (helpers), GRID (runner), AMT, EVAL |
| 8 | Beat This! download breaks keys / runs under GPU lock / unsafe load | `gtab models fetch/verify/list` (DATA) with the final0 URL; worker gets a local path (`weights_only=True`, `add_safe_globals` if needed); missing model → plan-time Korean error | §0, §1.5, §3.1, §5, §7.5, §9.3, §9.5 item 7 | DATA, GRID, P0 |
| 9 | MSST vendoring self-contradictory | exactly three files with a trimmed `__init__.py` patch; strict load; `rotary-embedding-torch==0.9.1`; no omegaconf/ml_collections; CPU `sdp_kernel` patch; `zero_dc` documented; upstream-parity gpu test (SNR > 60 dB) | §1.1, §9.1 items 2–3, tests | SEP, P0 |
| 10 | `amt_bp` breaks byte-determinism | determinism = data outputs only (excl. `_worker/**`, manifest) via `runner.data_outputs`; ORT threads pinned (`amt.bp_threads`) | §0, §4, §7.4 | GRID, AMT |
| 11 | mir_eval given MIDI, no counts | `midi_to_hz` before every call; counts from `match_notes`; +1-semitone test | §9.4 item 1, tests | EVAL |
| 12 | Basic Pitch min length off by one; E23 wasteful | config in kept frames (`bp_min_note_frames = 4`, pass `frames − 1`); E23 grid {3,4,5,6,7,12} frames; worker `sweep` mode: one `run_inference` per track, 72 `model_output_to_notes` passes | §1.3, §7.4, §10 E23, A11, §13 item 12 | AMT |
| 13 | Missing cross-owner deps / merge order | E1 + E24-sep moved to AMT; SEP exposes `gtab.sep.run.separate`; dependency table; 4-step merge order; P0 stub `STAGES` modules + lazy graph imports | §2, §1.7, §9.1 item 6 | all |
| 14 | Residual mask ≠ validated repet3 mask | exact repet3 mask (`W=min(median,V)`, `Mb=W²/(W²+max(V−W,0)²)`, fg = X·(1−Mb)) as pure `background_mask`; subprocess parity test ±0.5 dB SIR | §9.3 item 6, tests | GRID |
| 15 | Illegal / non-ASCII generated names | `gtab.eval.runs.slug()` for every generated path; `--system other` | §0, §5, §6.10, §9.4 items 7–8 | EVAL |
| 16 | npm.cmd and matplotlib on Windows | `node.exe …\npm-cli.js ci` with `npm_config_cache=paths.NPM_CACHE`; `MPLBACKEND=Agg` in the managed env | §0, §1.2, §9.4 item 6 | P0, EVAL |
| 17 | GP import details | alphaTab `AlphaSynth` render route; one string convention (1 = highest); capo added to PyGuitarPro `realValue`, checked against alphaTab; cp949/cp1252 + `reference.encoding` | §6 conventions, §6.7, §9.4 item 6, tests | EVAL, P0 (schema) |
| 18 | Taps: use synctoolbox anchors; heavy imports | `sync_via_mrmsdtw_with_anchors`; lazy imports enforced by `test_import_weight.py` | §0, §9.4 item 6, §1.7 | EVAL, P0 |
| 19 | MIDI lead-in not expressible as a pickup | lead-in bar(s) `1/4` or `k/4` with tempo from the gap; 720 ticks per dotted quarter in 6/8; `midi_bar_offset`; tests | §9.2 item 7 | AMT |
| 20 | No ground truth for b13 | `families_present` on `GtSpec`, `DatasetTrack`, `lines.yaml` | §6.7, §6.8, §9.5, §11 b13 | P0, DATA, EVAL |
| 21 | b7 fails on sparse stems | error relative to mix RMS; stems < −60 dB rel. mix `skipped_sparse`; sparse-stem test | §9.1 item 7, §11 b7 | SEP |
| 22 | `sep` key includes non-semantic keys | sep params = {ckpt_sha256, config_sha256, msst_commit, chunk_ladder, num_overlap, batch_size, dtype, input_lufs}; `leftover_warn_db` only in `stems` | §3.2 row 4 | SEP |
| 23 | E1/E2/E3 adoption would need P0 edits | knobs added now: `sep.input_lufs`, `amt.guitar_view`, `amt.guitar_mask` (+ `guitar_other` view); deterministic `master()` limiter specified | §1.3, §3.2, §7.2, §9.4 item 4 | P0, SEP, AMT, EVAL |
| 24 | `small` fallback offline; instrument order | `try_to_load_from_cache` path or `unavailable`, never size keywords; `sort_mt3` before request and key | §3.2, §7.3, §9.2 item 4 | AMT |
| 25 | `sections` not byte-deterministic | `threadpool_limits(1)`, sign-normalised eigenvectors, seeded KMeans, rounded outputs, determinism test | §9.3 item 4, tests | GRID |
| 26 | Hardlinked export problems | small files copied, WAVs hardlinked read-only; locked old export renamed aside for `gtab gc`, clean failure otherwise | §3.3, tests | GRID |
| 27 | Latency criterion grades itself | estimate on GuitarSet players 00–02 (+ EGDB dev), report on held-out 03–05 | §9.2 item 8, §10, §11 b10, A10 | AMT |
| 28 | Gate may compare a contaminated model | pre-registered: MuScriptor × EGDB `학습 포함`/`불명` → `판단 보류 (오염 불명)`; Basic Pitch training sets recorded (EGDB clean for BP; MedleyDB not) | §10 gate-m2, §9.4 item 9, §11 b12, facts | EVAL, AMT |
| 29 | Minor inconsistencies | bass stem from `sep`; half-open bar ranges everywhere; `amt_gtr` mask via dep key; ids aligned (`E24-sep`/`E24-amt`/`gate-m2`); runner accepts `실행 중`; `grid.dbn=true` rejected; grid step 3.1 reduced to a check, refinement kept (20 ms quantisation) | §3.2, §3.3, §6, §9.3, §9.4 items 8–9, §1.3 | GRID, P0, EVAL |
| 30 | Env/test notes | muscriptor deps completed; bp310 3.10 resolution noted; `amt_bp` optional; `test_py310_compat.py`; `sha_cache.json` under filelock; b1/b9 record other GPU processes (Claude app) | facts, §1.5, §1.7, §3.2 row 9, §7.4, §9.1 item 10 | P0, AMT, SEP |

**New or re-assigned files** (all included in §2): P0 — `tests/test_py310_compat.py`, `tests/test_import_weight.py`,
initial stubs `gtab/commands/models.py` and `gtab/{analysis,sep,amt}/stage.py`; SEP — `gtab/workers/vram_ballast.py`,
`gtab/sep/run.py` (SEP's `experiments.py` shrinks to `stereo-preservation`); AMT — E1 and E24-sep runners; DATA —
`gtab/model_fetch.py`, `gtab/model_sources.toml`, `gtab/commands/models.py`, `tests/test_model_fetch.py`;
EVAL — `gtab/eval/node.py` (inside `gtab/eval/**`). No file gained a second owner; M0's `gtab/workers/selftest.py`
is **not** edited (the ballast is its own worker).
