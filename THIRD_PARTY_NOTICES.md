# Third-party notices

bandscribe's own code is licensed under the MIT License ([`LICENSE`](LICENSE), Copyright (c) 2026 bhj2837). That
licence covers only what was written for this project. This file lists everything else that bandscribe includes,
installs, downloads or reads, and the terms that apply to each.

Checked on 2026-10-04: Python licences from the installed packages' metadata in all three environments, Node
licences from `node/package-lock.json` and the installed package, and model and dataset terms from their upstream
repositories, model cards and dataset records. This is a record of what was checked, **not legal advice**.

> **한국어 요약.** 이 저장소의 우리 코드는 MIT다. 저장소 안에 들어 있는 제3자 코드는
> `bandscribe/third_party/msst/` 하나뿐이고, MIT 저작권 고지 두 개(ZFTurbo, Phil Wang)를 함께 둔다. 파이썬·Node
> 의존성, 모델 가중치, 데이터셋, yt-dlp·ffmpeg 같은 도구는 저장소에 없고 각 사용자가 직접 받는다. 그중 MuScriptor
> 가중치(CC BY-NC 4.0 + 게이트 조건), BS-RoFormer SW(라이선스 불명), EGDB(라이선스 미기재), IDMT-SMT-Bass(CC BY-NC-ND),
> MedleyDB(CC BY-NC-SA), Cambridge-MT(라이브러리 약관)는 비상업·개인용으로만 쓰고 재배포하지 않는다. 곡 오디오,
> 분리 스템, 전사 결과는 `data/` 아래에만 있고 git 에 올리지 않는다.

## At a glance

| What | In this repository? | Licence / terms | Section |
|---|---|---|---|
| bandscribe's own code, docs and tests | yes | MIT | [`LICENSE`](LICENSE) |
| MSST BS-RoFormer model code (vendored) | yes, `bandscribe/third_party/msst/` | MIT (ZFTurbo) + MIT (Phil Wang) | 1 |
| Python packages (core, gpu, bp310 environments) | no, only the lock files | mostly permissive; LGPL / MPL ones listed | 2 |
| alphaTab (Node) | no, only `node/package-lock.json` | MPL-2.0 | 2.3 |
| Model weights | no | per model; one is non-commercial, one unknown | 3 |
| Datasets | no | per dataset; several non-commercial | 4 |
| yt-dlp, ffmpeg, Node.js, uv, Python | no | Unlicense / GPL / MIT / ... | 5 |
| Your songs, stems and transcriptions | no (`data/` is git-ignored) | the rightsholders' copyright | 6 |

## 1. Third-party code in this repository

### 1.1 Music-Source-Separation-Training (MSST)

- Path: `bandscribe/third_party/msst/`.
- Upstream: <https://github.com/ZFTurbo/Music-Source-Separation-Training>, commit
  `84b1eac0887756b4f1a9d7a1ff49105939749ed2` (2026-09-26).
- Licence: MIT, Copyright (c) 2024 Roman Solovyev (ZFTurbo). The full text is kept verbatim in
  `bandscribe/third_party/msst/LICENSE` (its sha256 equals upstream's; `tests/test_sep_vendor.py` checks it).
- What is vendored:
  - `models/bs_roformer/__init__.py`, `bs_roformer.py`, `attend.py`, with small patches;
  - `demix.py`, a port of MSST's `utils/model_utils.py::demix`. It is derived from MSST, so the same notice applies.
  - `loader.py`, `__init__.py` and `models/__init__.py` were written for bandscribe (MIT, ours).
- Patches, hashes and how to update: `bandscribe/third_party/msst/VENDOR.md`.

### 1.2 lucidrains/BS-RoFormer (inside the MSST files)

- MSST's `bs_roformer.py` and `attend.py` are derived from <https://github.com/lucidrains/BS-RoFormer>
  (MIT, Copyright (c) 2023 Phil Wang).
  - Measured 2026-10-04 against its commit `84f2b25297424e93f721d5fb9d42291c6d934284`: 343 of its 584 lines of
    `bs_roformer.py` and 116 of its 120 lines of `attend.py` appear unchanged, in order, in the vendored files.
- MSST's README credits that repository, but MSST's `LICENSE` carries only ZFTurbo's notice. MIT requires the
  notice in every copy of a substantial portion, so Phil Wang's notice is kept verbatim in
  `bandscribe/third_party/msst/LICENSE.BS-RoFormer`.
- The BS-RoFormer architecture itself comes from Lu et al., "Music Source Separation with Band-Split RoPE
  Transformer" (arXiv:2309.02612). The paper's ideas are not code; the code above is what the licences cover.

### 1.3 Other code checked: no third-party code found

All 255 files tracked before this notice was added were searched for "port", "adapted", "copied", "vendored",
upstream URLs and licence headers. Every hit was checked:

- `bandscribe/eval/remix.py` ports generators from `docs/research/experiments/synth_exp.py` / `router_exp.py`.
  Those experiments were written for this project (MIT).
- `docs/research/experiments/*.py` are this project's own synthetic experiments. `repet*.py` implement the
  REPET / REPET-SIM idea (Rafii & Pardo) from the papers; no code was copied.
- The workers (`bandscribe/workers/amt_muscriptor.py`, `amt_basicpitch.py`, `beats_beatthis.py`) import MuScriptor,
  Basic Pitch and Beat This! from the installed packages at run time. That includes MuScriptor's private helpers
  `_build_model`, `_resolve_config` and `_remap_single_codebook_keys`, which the lean loader calls. No code is copied
  from those packages.
- `tests/test_amt_events.py` defines three stand-in classes with the field names of MuScriptor's event classes.
  That is the interface only, and MuScriptor's code is MIT in any case.
- `node/bandscribe_score.mjs` imports alphaTab from `node/node_modules` at run time. No alphaTab code is copied.
- The test fixtures (`tests/fixtures/`) generate their audio, MIDI and Guitar Pro files during the test run. No
  binary or dataset fixture is committed.

## 2. Dependencies: installed, not included

The repository holds only lock files: `uv.lock`, `envs/gpu/uv.lock`, `envs/bp310/uv.lock` and
`node/package-lock.json`. They list package names, versions, URLs and hashes. uv and npm download the packages
onto each user's PC, and the installed copies are git-ignored (`.venv/`, `envs/*/.venv/`, `node/node_modules/`).

### 2.1 Licences that need a note (⚠ in the table below)

| Package | Licence | How bandscribe uses it | Why our code can still be MIT |
|---|---|---|---|
| PyGuitarPro 0.11 (core) | LGPL-3.0-only | `import guitarpro`: reads reference Guitar Pro files and writes test fixtures. Unmodified. | The LGPL lets programs under any licence use the library. Its conditions (licence text, a replaceable library, no ban on reverse engineering to debug such changes) bind whoever **conveys** the library or a combined work. We convey neither: users install the unmodified wheel from PyPI. If bandscribe is ever shipped as one bundle (e.g. PyInstaller), include the LGPL text and keep PyGuitarPro a separate, replaceable file. |
| soxr 1.1.0 (all three envs) | LGPL-2.1-or-later (python-soxr and the bundled libsoxr); PFFFT is BSD | `import soxr` for resampling. Unmodified. | Same reasoning (LGPL-2.1 §5: a "work that uses the Library"). |
| soundfile 0.14.0 (all three envs) | BSD-3-Clause; the Windows wheel bundles libsndfile (LGPL-2.1-or-later), LAME (LGPL-2+) and mpg123 (LGPL-2.1), plus BSD codecs | `import soundfile`; libsndfile is loaded as a DLL. Unmodified. | Same reasoning. |
| alphaTab 1.8.4 (Node) | MPL-2.0 | Imported from `node/node_modules` for Guitar Pro import and audio rendering. Unmodified. | MPL-2.0 is file-level copyleft: only changes to alphaTab's own files must stay under the MPL. Separate files, such as ours, may use any licence (MPL-2.0 §3.3, "Larger Work"). |
| certifi (all three envs) | MPL-2.0 | Transitive (requests, httpx / httpcore): CA certificates. Unmodified. | Same as alphaTab. |
| tqdm (gpu) | MPL-2.0 AND MIT | Transitive (huggingface_hub). Unmodified. | Same as alphaTab. |
| numpy, scipy | BSD-3-Clause; the wheels bundle OpenBLAS (BSD-3-Clause) and the GCC runtime (libgfortran etc., GPL-3.0-or-later WITH GCC-exception-3.1) | Direct and transitive. | The GCC Runtime Library Exception allows the compiled code to be used and distributed under any licence. |
| torch 2.11.0+cu128 (gpu) | BSD-3-Clause; the Windows CUDA wheel bundles NVIDIA's CUDA runtime, cuBLAS and cuDNN DLLs | GPU workers. | NVIDIA's own terms govern redistributing those DLLs. We do not redistribute them: users install the wheel from download.pytorch.org. |

No Python package in any environment is under the GPL or AGPL. GPL text appears only inside numpy and scipy, for
the GCC runtime under its exception. Everything else is MIT, BSD, ISC, Apache-2.0, PSF-2.0, 0BSD, MIT-0, MIT-CMU or
the PSF-style Matplotlib licence.

The reasoning above follows the common reading of the LGPL and MPL in the FSF's and Mozilla's FAQs: using an
unmodified, separately installed library from code under another licence is allowed, and the copyleft conditions
apply when the library itself, or a modified version of it, is distributed. Not legal advice.

### 2.2 All Python packages

Read on 2026-10-04 with `importlib.metadata` in each venv (`License-Expression`, then `License`, then the licence
classifiers). Where those were empty or free text, the package's own licence file was read instead.

- **Bold** version: declared directly in a `pyproject.toml`. The others are transitive.
- Python 3.12 for core and gpu, 3.10 for bp310.
- Counts: core 85, gpu 60, bp310 47; 126 distinct packages (bandscribe itself excluded).

| Package | core | gpu | bp310 | Licence | Notes |
|---|---|---|---|---|---|
| annotated-doc | 0.0.5 | 0.0.5 |  | MIT |  |
| annotated-types | 0.8.0 | 0.8.0 | 0.8.0 | MIT |  |
| anyio |  | 4.15.1 |  | MIT |  |
| asttokens | 3.0.2 |  |  | Apache-2.0 |  |
| attrs | 26.1.0 |  |  | MIT |  |
| audioread | 3.1.0 |  | 3.1.0 | MIT |  |
| basic-pitch |  |  | **0.4.0** | Apache-2.0 | includes the Basic Pitch model files (section 3.4) |
| beartype |  | **0.22.9** |  | MIT |  |
| beat-this |  | **1.1.0** |  | MIT | code only; the checkpoint is separate (section 3.3) |
| certifi | 2026.7.22 | 2026.7.22 | 2026.7.22 | MPL-2.0 ⚠ | Mozilla CA bundle; see section 2.1 |
| cffi | 2.1.1 | 2.1.1 | 2.1.1 | MIT-0 |  |
| chardet | 7.6.0 |  |  | 0BSD | via music21; this version's licence file is 0BSD (older chardet releases were LGPL) |
| charset-normalizer | 3.5.2 |  | 3.5.2 | MIT |  |
| click |  | 8.5.0 |  | BSD-3-Clause |  |
| cloudpickle | 3.1.2 |  | 3.1.2 | BSD-3-Clause |  |
| colorama | 0.4.6 | 0.4.6 |  | BSD-3-Clause |  |
| coloredlogs |  |  | 15.0.1 | MIT |  |
| contourpy | 1.4.0 |  |  | BSD-3-Clause |  |
| cycler | 0.12.1 |  |  | BSD-3-Clause |  |
| decorator | 5.3.1 |  | 5.3.1 | BSD-2-Clause |  |
| einops |  | **0.8.2** |  | MIT |  |
| executing | 2.2.1 |  |  | MIT |  |
| fastapi |  | 0.142.1 |  | MIT |  |
| filelock | **4.0.7** | **4.0.7** | **4.0.7** | MIT |  |
| flatbuffers |  |  | 25.12.19 | Apache-2.0 |  |
| fonttools | 4.66.1 |  |  | MIT |  |
| fsspec |  | 2026.9.0 |  | BSD-3-Clause |  |
| h11 |  | 0.16.0 |  | MIT |  |
| hf-xet |  | 1.6.0 |  | Apache-2.0 |  |
| httpcore |  | 1.0.9 |  | BSD-3-Clause |  |
| httpcore2 |  | 2.13.1 |  | BSD-3-Clause |  |
| httptools |  | 0.8.0 |  | MIT |  |
| httpx |  | 0.28.1 |  | BSD-3-Clause |  |
| httpx2 |  | 2.13.1 |  | BSD-3-Clause |  |
| huggingface_hub |  | **2.0.0** |  | Apache-2.0 |  |
| humanfriendly |  |  | 10.0 | MIT |  |
| idna | 3.20 | 3.20 | 3.20 | BSD-3-Clause |  |
| importlib_resources | 7.1.0 |  | 7.1.0 | Apache-2.0 |  |
| iniconfig | 2.3.0 |  |  | MIT |  |
| ipython | 8.39.0 |  |  | BSD-3-Clause |  |
| jedi | 0.20.0 |  |  | MIT |  |
| Jinja2 | **3.1.6** | 3.1.6 |  | BSD-3-Clause |  |
| joblib | 1.6.0 |  | 1.6.0 | BSD-3-Clause |  |
| jsonpickle | 4.1.2 |  |  | BSD-3-Clause |  |
| kiwisolver | 1.5.1 |  |  | BSD-3-Clause |  |
| lazy-loader | 0.6 |  | 0.6 | BSD-3-Clause |  |
| libfmp | 1.3.0 |  |  | MIT |  |
| librosa | **0.11.0** |  | 0.11.0 | ISC |  |
| llvmlite | 0.47.0 |  | 0.49.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception | bundles LLVM |
| markdown-it-py | 4.2.0 | 4.2.0 |  | MIT |  |
| MarkupSafe | 3.0.3 | 3.0.3 |  | BSD-3-Clause |  |
| matplotlib | **3.11.2** |  |  | Matplotlib License (PSF-style, permissive) | bundles DejaVu / STIX fonts (permissive font licences) |
| matplotlib-inline | 0.2.2 |  |  | BSD-3-Clause |  |
| mdurl | 0.1.2 | 0.1.2 |  | MIT |  |
| mido | **1.3.3** | 1.3.3 | 1.3.3 | MIT |  |
| mir_eval | **0.8.2** |  | 0.8.2 | MIT |  |
| more-itertools | 11.1.0 |  |  | MIT |  |
| mpmath |  | 1.3.0 | 1.3.0 | BSD-3-Clause |  |
| msgpack | 1.2.3 |  | 1.2.3 | Apache-2.0 |  |
| muscriptor |  | **0.3.0** |  | MIT | code only; the weights are separate (section 3.2) |
| music21 | **9.9.2** |  |  | BSD-3-Clause | its bundled corpus has its own licence notes; bandscribe does not use the corpus |
| narwhals | 2.26.0 |  |  | MIT |  |
| networkx |  | 3.7 |  | BSD-3-Clause |  |
| numba | 0.65.1 |  | 0.67.0 | BSD-2-Clause |  |
| numpy | **2.4.6** | **2.5.3** | **2.2.6** | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 parts) | wheel bundles OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0-or-later WITH GCC-exception-3.1) |
| nvidia-ml-py | **13.615.71** | **13.615.71** |  | BSD (metadata; no licence file in the wheel) |  |
| onnxruntime |  |  | **1.23.2** | MIT |  |
| opentelemetry-api |  | 1.45.0 |  | Apache-2.0 |  |
| packaging | 26.3 | **26.3** | 26.3 | Apache-2.0 OR BSD-2-Clause |  |
| pandas | **2.3.3** |  |  | BSD-3-Clause |  |
| parso | 0.8.7 |  |  | MIT |  |
| pillow | 12.3.0 |  |  | MIT-CMU |  |
| platformdirs | 4.12.2 |  | 4.12.2 | MIT |  |
| pluggy | 1.6.0 |  |  | MIT |  |
| pooch | 1.9.0 |  | 1.9.0 | BSD-3-Clause |  |
| pretty_midi | **0.2.11.post0** |  | 0.2.11.post0 | MIT |  |
| prompt_toolkit | 3.0.53 |  |  | BSD-3-Clause |  |
| protobuf |  |  | 7.36.2 | BSD-3-Clause |  |
| pure_eval | 0.2.4 |  |  | MIT |  |
| pycparser | 3.0 | 3.0 | 3.0 | BSD-3-Clause |  |
| pydantic | **2.13.5** | **2.13.5** | **2.13.5** | MIT |  |
| pydantic_core | 2.46.5 | 2.46.5 | 2.46.5 | MIT |  |
| Pygments | 2.21.0 | 2.21.0 |  | BSD-2-Clause |  |
| PyGuitarPro | **0.11** |  |  | LGPL-3.0-only ⚠ | see section 2.1 |
| pyloudnorm | **0.2.0** |  |  | MIT |  |
| pyparsing | 3.3.3 |  |  | MIT |  |
| pyreadline3 |  |  | 3.5.6 | BSD-3-Clause |  |
| pytest | **9.1.1** |  |  | MIT |  |
| python-dateutil | 2.9.0.post0 |  |  | Apache-2.0 AND BSD-3-Clause |  |
| python-dotenv |  | 1.2.3 |  | BSD-3-Clause |  |
| python-multipart |  | 0.0.32 |  | Apache-2.0 |  |
| pytz | 2026.4 |  |  | MIT |  |
| PyYAML | **6.0.3** | **6.0.3** |  | MIT |  |
| requests | 2.34.2 |  | 2.34.2 | Apache-2.0 |  |
| resampy |  |  | 0.4.2 | ISC |  |
| rich | **15.0.0** | 15.0.0 |  | MIT |  |
| rotary-embedding-torch |  | **0.9.1** |  | MIT |  |
| safetensors |  | **0.8.0** |  | Apache-2.0 |  |
| scikit-learn | **1.9.1** |  | 1.7.2 | BSD-3-Clause |  |
| scipy | **1.18.1** |  | 1.15.3 | BSD-3-Clause | wheel bundles OpenBLAS and the GCC runtime, as numpy |
| setuptools |  | 81.0.0 | **80.10.2** | MIT |  |
| shellingham | 1.5.4 | 1.5.4 |  | ISC |  |
| six | 1.17.0 |  | 1.17.0 | MIT |  |
| soundfile | **0.14.0** | **0.14.0** | **0.14.0** | BSD-3-Clause ⚠ | see section 2.1; the Windows wheel bundles libsndfile (LGPL-2.1-or-later) with LAME / mpg123 (LGPL) and FLAC / Ogg / Vorbis / Opus (BSD) |
| soxr | **1.1.0** | **1.1.0** | 1.1.0 | LGPL-2.1-or-later ⚠ | see section 2.1; bundles libsoxr (LGPL-2.1-or-later) and PFFFT (BSD) |
| stack-data | 0.6.3 |  |  | MIT |  |
| starlette |  | 1.7.0 |  | BSD-3-Clause |  |
| sympy |  | 1.14.0 | 1.14.0 | BSD-3-Clause |  |
| synctoolbox | **1.4.2** |  |  | MIT |  |
| threadpoolctl | **3.7.0** |  | 3.7.0 | BSD-3-Clause |  |
| tomli_w | **1.2.0** |  |  | MIT |  |
| torch |  | **2.11.0+cu128** |  | BSD-3-Clause | Windows CUDA wheel bundles NVIDIA CUDA / cuBLAS / cuDNN DLLs under NVIDIA's licence terms |
| torchaudio |  | **2.11.0+cu128** |  | BSD-2-Clause |  |
| tqdm |  | 4.70.1 |  | MPL-2.0 AND MIT ⚠ |  |
| traitlets | 5.16.1 |  |  | BSD-3-Clause |  |
| truststore |  | 0.10.4 |  | MIT |  |
| typer | **0.27.2** | 0.27.2 |  | MIT |  |
| typing_extensions | 4.16.0 | 4.16.0 | 4.16.0 | PSF-2.0 |  |
| typing-inspection | 0.4.4 | 0.4.4 | 0.4.4 | MIT |  |
| tzdata | 2026.4 |  |  | Apache-2.0 |  |
| urllib3 | 2.8.0 |  | 2.8.0 | MIT |  |
| uvicorn |  | 0.54.0 |  | BSD-3-Clause |  |
| watchfiles |  | 1.3.0 |  | MIT |  |
| wcwidth | 0.9.1 |  |  | MIT |  |
| webcolors | 25.10.0 |  |  | BSD-3-Clause |  |
| websockets |  | 17.1 |  | BSD-3-Clause |  |

### 2.3 Node packages

- `node/package-lock.json` pins exactly one package, with no transitive dependencies: `@coderline/alphatab` 1.8.4,
  MPL-2.0, Copyright Daniel Kuschny and Contributors.
- Its `LICENSE.header` lists the libraries built into it: TinySoundFont, SFZero, the Haxe standard library,
  SharpZipLib and NVorbis (all MIT), and libvorbis (BSD-3-Clause).
- It ships `dist/soundfont/sonivox.sf2`, which `bandscribe_score.mjs render` uses. That soundfont is based on the
  SONiVOX EAS bank (Copyright (c) 2004-2006 Sonic Network Inc.) and is Apache-2.0 according to its own LICENSE file.
- `npm ci` installs it into `node/node_modules` (git-ignored).

## 3. Models: not included, each user obtains them

Model files live under `data/models/` and `data/hf/`, which are git-ignored. The local
`data/models/models.lock.json` (also not committed) records the URL, size, sha256 and licence of each file.

- bandscribe downloads only the Beat This! checkpoint itself (`bandscribe models fetch`, which asks first).
- MuScriptor comes through the user's own Hugging Face account.
- Basic Pitch comes with its wheel.
- The SW checkpoint must be supplied by the user.

No other model is installed or loaded. Other models named in `docs/` were design-research candidates;
bandscribe does not install or run them.

### 3.1 BS-RoFormer SW (six stems)

- Files: `BS-Rofo-SW-Fixed.ckpt` (699 MB) and `BS-Rofo-SW-Fixed.yaml` in `data/models/bs_roformer_sw/` (registered as
  `bs_roformer_sw.ckpt` / `.yaml`), plus an unused reference config `BS-Roformer-SW.audiosep.yaml`.
- Licence: **unknown.** It is a community-trained checkpoint whose original uploader's account was deleted, so its
  origin and terms cannot be traced. Some later Hugging Face re-uploads (ONNX and GGUF conversions) label it MIT,
  but that is the re-uploader's label and could not be traced to whoever trained it.
- Conditions: personal, local use only. Never commit, upload or redistribute the checkpoint or the stems it
  produces.
  - Each user must obtain the checkpoint themselves. bandscribe has no download entry for it
    (`bandscribe/model_sources.toml`) and checks its sha256 before loading it.
- The code that runs it is the MSST / lucidrains code in section 1 (MIT).

### 3.2 MuScriptor (medium; small as an optional fallback)

- Sources:
  - weights: Hugging Face `MuScriptor/muscriptor-medium`, pinned revision `f32236969308476e01fd3aae67357de5feb05a2d`,
    and `MuScriptor/muscriptor-small`;
  - code: package `muscriptor` 0.3.0 from <https://github.com/muscriptor/muscriptor> (Kyutai x Mirelo).
- Code licence: MIT, Copyright (c) 2026 Kyutai x Mirelo.
- Weights licence: **CC BY-NC 4.0** (non-commercial), behind a Hugging Face gate. Accepting the gate means
  agreeing to the following (summary of the model card, checked 2026-10-04; both sizes carry the same gate):
  - you hold all rights needed for the music you transcribe; transcribing music without those rights is prohibited;
  - use of MuScriptor and its output complies with applicable law;
  - you indemnify Mirelo and Kyutai against claims arising from use that breaks these conditions or the licence;
  - the model comes as is, with no warranty, including no warranty of non-infringement.
- Conditions for bandscribe:
  - Each user accepts the gate with their own Hugging Face account and logs in (`tools/hf-login.cmd`). bandscribe
    and Claude never accept it on anyone's behalf.
  - Results are not used commercially, and the weights are never redistributed.
- Cite: Rouard et al., "MuScriptor: An Open Model for Multi-Instrument Music Transcription", arXiv:2607.08168
  (2026).

### 3.3 Beat This! (`final0` checkpoint)

- Sources: <https://github.com/CPJKU/beat_this> (package `beat-this` 1.1.0). The `final0.ckpt` checkpoint comes from
  the JKU server; its URL and sha256 are in `bandscribe/model_sources.toml`.
- Licence: MIT for the code and the published weights, Copyright (c) 2024 Institute of Computational Perception,
  JKU Linz. The README says so explicitly.
  - The README also warns that some of the training files are copyrighted or under limited Creative Commons
    licences, and leaves it to the user to judge whether that matters for their use.
- Cite: Foscarin, Schlüter and Widmer, "Beat This! Accurate beat tracking without DBN postprocessing", ISMIR 2024.

### 3.4 Basic Pitch (ICASSP 2022 model)

- Sources: <https://github.com/spotify/basic-pitch>, package `basic-pitch` 0.4.0. The model
  (`basic_pitch/saved_models/icassp_2022/nmp.onnx`) ships inside the wheel.
- Licence: Apache-2.0 for the code and the model, Copyright 2022 Spotify AB. Anyone redistributing the wheel must
  include its NOTICE file; we do not redistribute it.
- Cite: Bittner et al., "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch
  Estimation", ICASSP 2022.

## 4. Datasets: not included

Each user downloads or imports dataset audio and labels into `data/datasets/` (git-ignored) and uses them only on
that PC, for evaluation. The committed `bandscribe/datasets/registry.toml` holds only titles, URLs, sizes, checksums
and licences. Nothing from these datasets is in the repository: no audio, annotations, derived labels or excerpts.

| Dataset | Licence / terms | Source | Conditions for bandscribe |
|---|---|---|---|
| GuitarSet 1.1.0 | CC BY 4.0 | Zenodo, DOI 10.5281/zenodo.3371780 | Attribution (Xi et al., ISMIR 2018). |
| IDMT-SMT-Bass-Single-Track 1.0.0 | CC BY-NC-ND 4.0 | Zenodo, DOI 10.5281/zenodo.7544099 (Jakob Abeßer, Fraunhofer IDMT) | Non-commercial; never share modified versions (re-renders, conversions). |
| FiloBass 1.0.0 | CC BY 4.0 | Zenodo, DOI 10.5281/zenodo.10069709 (Riley and Dixon, ISMIR 2023) | Attribution. Its bass stems were separated from the FiloSax backing tracks. |
| EGDB | No licence stated; its README only invites feedback | Google Drive folder of Chen et al., ICASSP 2022 | Without a licence the default is all rights reserved: personal research use only, never redistributed. |
| Cambridge-MT "Mixing Secrets" multitracks (6 songs) | Cambridge-MT library terms | <https://www.cambridge-mt.com/ms3/mtk/> | The library is meant for education. Its FAQ calls research "within the spirit" of that, but not agreed with the contributors, who should be contacted; no commercial or promotional use. Kept local; the per-line labels (`lines.yaml`) stay under `data/`. |
| MedleyDB 1.0 audio | CC BY-NC-SA 4.0, non-commercial research only, restricted access | Zenodo, DOI 10.5281/zenodo.1649325 (access on request) | Not downloaded yet. Non-commercial research only; derivatives share alike; never redistributed. |

The Cambridge-MT songs were picked with the annotation CSV of <https://github.com/felixCheungcheung/mixing_secrets_v2>.
Only the song names were taken from it.

## 5. Tools: not included

| Tool | Licence | How bandscribe uses it |
|---|---|---|
| yt-dlp (`tools/yt-dlp.exe`, downloaded by the user) | Unlicense for the source and the PyPI packages. The PyInstaller-built `.exe` that bandscribe runs bundles GPLv3+ code, so that executable as a whole is GPLv3+ (yt-dlp README). | Separate process. |
| ffmpeg / ffprobe (installed by the user; on this PC the gyan.dev 8.0.1 "full" build) | That build is GPL-3.0-or-later (`--enable-gpl --enable-version3`). | Separate process (decoding and probing). |
| Node.js | MIT | Runs `node/bandscribe_score.mjs`, and yt-dlp's JavaScript challenge solver (`--js-runtimes node`). |
| uv | MIT OR Apache-2.0 | Environment manager, through `tools/uvw.cmd`. |
| CPython (installed by uv) | PSF-2.0 | Runtime. |

None of these is committed (`tools/*.exe`, `tools/yt-dlp*` and `tools/python/` are git-ignored). bandscribe does not
link, bundle or modify them; it starts them as separate programs, which leaves our code's licence unaffected.

### 5.1 YouTube

- yt-dlp's licence grants nothing about the videos it downloads.
- YouTube's Terms of Service (effective 2022-01-05) forbid downloading content without permission from YouTube or the
  rightsholders, and forbid accessing the service by automated means.
- YouTube input is on by default because the developer chose that after reviewing these risks (README, "법적 고지와
  개인 사용"). That choice is not anyone else's consent: judge the risk yourself, or set `[youtube] enabled = false` in
  `data/config.toml`. The risk is the person's who runs it.
- Downloaded audio stays under `data/downloads/` and `data/jobs/` (git-ignored) and is never redistributed. Local
  files the user owns are the recommended input. Not legal advice.

## 6. Your music and the results

- The songs you transcribe (local files, YouTube downloads), the stems separated from them and the MIDI and tabs
  made from them are covered by the rightsholders' copyright. A transcription is a derivative of the composition.
- Guitar Pro files supplied as ground truth (Tier A) are the same.
- Outputs of external services imported for comparison (`bandscribe eval import-external`: Songsterr AI, Klangio,
  MVSep and others) are also subject to those services' terms. They are stored under `data/eval/external/`.
- All of these live under `data/`, which is git-ignored. Never commit or share `data/jobs/`, `data/scratch/`,
  `data/eval/` or `data/downloads/`.

## 7. Repository scan before publishing (2026-10-04)

- **Files:** all 255 tracked files are text. There is no audio, model, archive, image or other binary file; the
  largest is `docs/DESIGN.md` (179 KB).
- **History:** every commit up to this one (7 before it, on `main` and `wip/m1a-m2`) contains only these files and
  their earlier `gtab/...` names. No binary blob was ever committed; the largest blob is 179 KB.
- **Secrets:** no Hugging Face or GitHub token and no private key in the tree or the history.
- **Copyrighted works:**
  - Song titles and artist names appear only as identifiers in docs and tests, next to measured statistics (note
    counts, durations, bleed shares).
  - There are no lyrics, no score or tab excerpts, no note or beat data taken from the songs and no dataset
    annotations.
  - YouTube video IDs appear only as test inputs for the URL parser.
- **Research notes:** `docs/research/*.txt` and `design_review.json` are this project's own summaries of papers and web
  pages, with links and short quotations.
