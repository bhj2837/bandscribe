# Rename: gtab → bandscribe (2026-10-04)

The project is now **bandscribe** everywhere (GitHub: `bhj2837/bandscribe`). The new name covers the planned
drum and piano scores. The folder is still `D:\gtab`; moving it is a separate step (see the last section).

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

## Left for the folder move (`D:\gtab` → new folder)

The code no longer hard-codes `D:\gtab`:

- `tools\uvw.cmd` and `tools\hf-login.cmd` find the root from their own location.
- Korean hints print `paths.ROOT`.

Still absolute and to redo after the move:

- **Venvs:** the editable `.pth` files point at `D:\gtab`, and the base interpreters live under
  `D:\gtab\tools\python`. Re-create or re-sync all three envs.
- **Model sha cache:** its entries are keyed by absolute path, so the first run after the move rehashes the models
  once (seconds). The sha values, and therefore the keys, stay the same.
- **Docs:** `D:\gtab` paths in the docs and the `위치` comment in `data\config.toml`.
