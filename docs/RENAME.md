# Rename: gtab → bandscribe (2026-10-04)

The project is now **bandscribe** everywhere (GitHub: `bhj2837/bandscribe`). The new name covers the planned
drum and piano scores. The folder moved from `D:\gtab` to `D:\bandscribe` the same day (see the last section).

## New names

| What | Before | After |
|---|---|---|
| Python package | `gtab/` | `bandscribe/` (`git mv`, history kept) |
| Distribution, CLI | `gtab`, `gtab.cmd`, `python -m gtab` | `bandscribe`, `bandscribe.cmd`, `python -m bandscribe` |
| Env vars | `GTAB_ROOT`, `GTAB_OFFLINE`, `GTAB_RUN_SLOW`, `GTAB_WORKER_LOG_LEVEL`, `GTAB_AUTOCAST_DTYPE` | `BANDSCRIBE_*`. Only the new names are read. |
| Root constant | `paths.GTAB_ROOT` | `paths.ROOT` |
| Log file | `data\logs\gtab.log` | `data\logs\bandscribe.log` (the old file is left alone) |
| Node helper | `node\gtab_score.mjs`, package `gtab-node` | `node\bandscribe_score.mjs`, `bandscribe-node` |
| Env projects | `gtab-gpu-env` (path dependency `gtab`), `gtab-bp310-env` | `bandscribe-gpu-env` (path dependency `bandscribe`), `bandscribe-bp310-env`. All three `uv.lock` files were re-locked; only the project names changed (torch stays 2.11.0+cu128). |
| On-disk format ids | `"gtab.<name>/1"` | `"bandscribe.<name>/1"` |
| Version field (stage manifest, job `meta.json`, `score.json`) | `gtab_version` | `bandscribe_version` |
| Dataset extraction marker | `.gtab_extracted.json` | `.bandscribe_extracted.json` |
| Misc | `gtab://job/` report links, `gtab/<ver>` user agent, `# gtab: py310` marker, thread names | `bandscribe…` |

User-facing text stays Korean. Only the command name changed (`bandscribe run …`).

## Old files keep loading. Nothing existing is rewritten.

Stage manifests record the sha256 of every output, and the cached results took hours of GPU time. So the readers
accept the old names instead of migrating files:

- **Format ids:** `bandscribe/formats.py` maps `gtab.<name>/N` to `bandscribe.<name>/N` on load.
  - `atomic.read_json` applies it to a document's top-level `format`.
  - The pydantic file models (`schema._common._Model`, `eval.refscore._Model`) apply it in a before-validator.
  - The TOML/YAML readers apply it too: dataset registry, model sources, Cambridge-MT `lines.yaml`, Tier A
    `songs.yaml`.
  - This covers job outputs, `export.json` (an old one still counts as up to date), the AMT cache,
    `models.lock.json`, the model sha cache, `datasets.lock.json`, `vram_table.json`, `gpu_run.json`, eval files
    and ground truth.
- **Other old names:**
  - `score.json`: `gtab_version` is read as `bandscribe_version`.
  - A `.gtab_extracted.json` marker still counts as an extraction. A re-extraction writes only the new name.
- **New writes:** they use the new names. A registry file switches to the new id the next time the program
  updates it (`models fetch`, `data fetch`, `bench gpu`).
- `tests/test_rename_compat.py` covers all of this.

## Kept the old name on purpose: frozen key material

| Constant | Why it stays |
|---|---|
| `ingest/filekey.py` `_DOMAIN = b"gtab-content-key-v1\0"` | It is hashed into every file content key: `file:<key>` in `jobs\index.json` and `source.content_key` in `meta.json`. A new domain would make every known file an index miss, so each one would be decoded again. |
| `amt/cache.py` `KEY_FORMAT = "gtab.amt_cache_key/1"` | It is hashed into every transcription cache key (`jobs\*\cache\amt`, `data\eval\reftx_cache`, about 700 entries). Renaming it would orphan all of them and cost GPU re-transcriptions. |

**Stage keys do not depend on the package name.** `JobStore.stage_key` hashes only the stage name, code version,
params, dep keys plus input hashes, and model shas. No params contain `gtab`.

**Check:** `data\scratch\rename\keys_snapshot.py` ran before and after the rename. Result:

- 1,755 planned keys (15 jobs × 9 config variants × 13 stages): **0 changed**.
- 0 changes to the "cached" flag.
- All 176 existing manifests still recompute to their recorded key.
- File content keys of the two test songs and a golden AMT cache key are unchanged.

## Folder move: `D:\gtab` → `D:\bandscribe` (done 2026-10-04)

The folder was renamed in place (`Move-Item`, same volume: instant, hard links and file ids kept). The code finds
its root from the package location (`paths.ROOT`), and `tools\uvw.cmd` / `tools\hf-login.cmd` from their own
location, so no code changed. Evidence: `data\scratch\move\`.

| Artefact | Absolute path inside? | Handling |
|---|---|---|
| Venvs (`pyvenv.cfg` home, `Scripts\*.exe` launchers, `activate*`, editable `.pth`, `direct_url.json`) | yes | All three `.venv` folders deleted and re-created with `uvw.cmd sync --locked --offline` (packages from `data\uv-cache`, no downloads), then `--compile-bytecode` (see PROGRESS "Envs": without `.pyc` files the CLI start-up doubles). Package lists identical before/after (core 86, gpu 61, bp310 47); torch 2.11.0+cu128, CUDA available. |
| `tools\python\cpython-3.1x-windows-x86_64-none` | yes: uv's minor-version **junctions** store an absolute target | Both junctions re-created (`mklink /J`) to the new folder. The interpreters themselves are relocatable. |
| Windows registry `HKCU\Software\Python\Astral\CPython3.10.21` (uv registers the 3.10 install, PEP 514) | yes | `ExecutablePath` / `WindowedExecutablePath` pointed at the new folder (uv warned about the dead path otherwise). |
| HF cache `data\hf` (MuScriptor weights, login token) | no (plain files, no symlinks) | Nothing. `HF_HOME` is derived from `paths.ROOT`; `hf auth whoami` still answers with the same account. |
| `data\jobs\index.json`, stage keys | no | Index is keyed by content key, stage keys by name/version/params/deps/model shas: 1,755 planned keys and all 176 manifests unchanged. |
| `meta.json` `source.ref`, `_worker\request.json` / `result.json` / `stderr.log` in jobs, eval, bench and runs | yes (provenance only) | Left as is: never read back for loading or keys, and the manifests pin their sha256. |
| `export\` WAV hard links | – | Kept by the same-volume rename (27/27 still linked to their stage file). |
| `models.lock.json`, `datasets.lock.json`, Tier A `songs.yaml`, `vram_table.json` | no (root-relative) | Nothing. |
| `data\models\.sha_cache.json` | yes (keys are absolute paths) | Rehashed on first use (same shas); entries for paths that no longer exist were pruned. |
| `data\config.toml` | only the `위치` comment | Comment updated; values unchanged. |
| Docs, `envs\bp310\pyproject.toml` comment, fake paths in tests | yes | Updated to `D:\bandscribe`. |
| Logs, `data\scratch`, `data\runs` | yes (history) | Left as is. |

**Stage params read package versions from the worker envs** (`muscriptor_version` from the gpu env,
`basic_pitch_version` from bp310, via their dist-info). While an env is missing or half-installed those params
change and so do the keys: a snapshot taken during the gpu re-sync showed 5 changed keys on the first job. Never run
`bandscribe` while an env is being re-created; after the sync, keys were identical again.

**Checks after the move** (`data\scratch\move\`):

- `keys_snapshot.py` (copied from the rename): 1,755 planned keys, 0 changed; all manifests recompute to their key;
  file content keys of the 5 `D:\backing` originals unchanged.
- `bandscribe doctor`: 22 checks, 0 failed (1 warning: VRAM budget, other apps).
- Tests: 1198 passed, 1 skipped.
- Cached `bandscribe run`: SC, KH, aotonat through `notes` (13/13 cached, export up to date) and AIZO, AZ through
  `resid1` (their last cached stage at the current stage versions), 0.6–0.8 s each. Cached YouTube URLs are cache
  hits with no download.
- Uncached GPU step: AIZO `amt_ms1` (MuScriptor medium-lean from the moved HF cache, 68.6 s) and `amt_bp`
  (bp310 env, 8.2 s), at their planned keys.
