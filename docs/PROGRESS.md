# gtab progress / handoff

Last updated: 2026-10-04 (paused by the user during the experiments phase: "too slow, continue later").

## Where things stand

| Milestone / phase | State |
|---|---|
| M0 foundation | ✅ done, on `main` as `1e422dc` |
| M1a+M2 spec (`docs/M1_M2_SPEC.md`) | ✅ done (30-point critique applied) |
| P0 contracts + 5 owners (SEP, AMT, GRID, EVAL, DATA) | ✅ implemented |
| Integration | ✅ 1110 tests passed; all 5 `D:\backing` songs + 1 Cambridge-MT mix run end to end |
| Experiments | 🟡 mostly done (see table below) |
| Review → Fix → Acceptance | ⬜ not started |

All M1a/M2 work is in the WIP commit on branch **`wip/m1a-m2`**. `main` still points at M0. Merge into `main` after acceptance.

### Experiments (`docs/decisions.md`)

| E# | Status |
|---|---|
| E1 SW input loudness | Undecided. Default `off` kept: every CI includes 0. |
| E2 MuScriptor input view | Undecided. Default `guitar_mono` kept. |
| E3 instruments mask + presence threshold | Undecided. Default `guitar+present` / 0.5 kept: thin data. |
| E23 Basic Pitch params | **Adopted** onset 0.6, frame 0.4, min_note_frames 7. Test Δ +6.6 [+3.8, +9.9]. |
| Latency | **Adopted** `amt.latency_s = -0.003` (descriptive; held-out 10.9 ms). |
| Gate-m2 (MuScriptor vs Basic Pitch, distorted) | Undecided. Δ +8.1 [+2.1, +13.5], but contamination is unknown, so not counted as a pass. |
| E24-sep, E24-amt (fp16 cast) | Pre-registered only, not run. Optional; skip unless cheap. |
| stereo-preservation (last entry in decisions.md) | Pre-registered only, not run. |
| Tier A descriptive pass on the 5 songs | Started (`data/scratch/exp/tiera_desc.py`), killed at the pause. Partial output is in `data/scratch/exp/tierA/`. |

## Priority for the next session: speed (the user said it takes too long)

- The **pipeline itself** takes 9–25 min per 3.3–4.6 min song on the RTX 2060. MuScriptor is 85–95 % of that.
- **Cost driver:** the S3.5 `piano_other_mono` MuScriptor pass on pad-heavy songs (864 s for 274 s of audio on 青と夏), then the full-mix B0 pass.
- **Options:**
  1. A `fast` profile that skips the piano+other pass and the B0 mix pass (instrumentation from stem energy + hints only; DESIGN S3.5 already allows this for `fast`).
  2. Cap or narrow the piano+other mask.
  3. Use the upstream loader (1.46× vs 1.28× realtime) when VRAM allows.
  4. Run the CPU Basic Pitch pass in parallel with GPU stages.
- **Workflow time:** the build itself took ~2 days, mostly the serial GPU experiment queue (~7 h of GPU time). Next time:
  - keep experiments small (cap tracks, stop at the first clear answer);
  - run review/fix/accept without re-running experiments;
  - ask the user before any GPU job longer than ~30 min.

## Next steps (in order)

1. `git checkout wip/m1a-m2`. Quick check: `D:\gtab\gtab.cmd doctor` and `D:\gtab\.venv\Scripts\python.exe -m pytest -q`.
2. Speed work above (at least the `fast` profile), measured on SC.mp3.
3. Review (correctness + GPU/Windows), fix, then check acceptance against DESIGN §10 M1a/M2 (evidence list in M1_M2_SPEC §11). Tier A GP ground truth is BLOCKED on the user.
4. Merge `wip/m1a-m2` into `main` and commit the milestone.
5. Then M3 (first tabs: grid/quantization, Viterbi fretting, `.gp5`/alphaTex export).

## Facts worth keeping

- The Claude desktop app sandbox redirects `%APPDATA%`/`%LOCALAPPDATA%` writes, so everything lives under `D:\gtab`. Use uv only through `D:\gtab\tools\uvw.cmd`.
- **Envs:**
  - core `.venv`: no torch.
  - gpu `envs\gpu\.venv`: torch 2.11.0+cu128.
  - bp310 `envs\bp310\.venv`: Basic Pitch / onnxruntime.
- **VRAM per stage:**
  - SW: 1.2–1.7 GB reserved.
  - MuScriptor lean: 1.74 GB.
  - Beat This!: 0.37 GB.
  - Other desktop apps hold ~1.5–2 GB.
- **Backings:** SC, AIZO and AZ remove the guitar. KH (怪獣の花唄) and aotonat (青と夏) remove nothing measurable; someone needs to listen.
- **Separation check** (Cambridge-MT Secretariat vs true tracks): guitar 7.6 dB, bass 10.0, drums 7.9, vocals 9.0.
- **Known issue b13:** Hammond organ → SW `other` → MuScriptor calls it distorted guitar, so keys go undetected.
- **EGDB** is only partly downloaded (~5.3 GB left). `gtab data fetch egdb --yes` resumes it.
- At the pause, `data/gpu.lock.pids` was left by a hard-killed run. That is expected: the next GPU run waits on the listed (dead) pids and removes the file.
- **Evidence:** `data/scratch/{integ,exp,amt}` hold copyrighted-audio analysis. Never commit them (`data/` is gitignored).

## Pending on the user

- Optional: accept the Hugging Face gate for `MuScriptor/muscriptor-small` (VRAM fallback).
- Delete the HF token that was pasted into chat (https://huggingface.co/settings/tokens). The login uses a separate OAuth token.
- Guitar Pro files (.gp/.gp5) for any song in `D:\backing`, to use as scored ground truth (Tier A).
- Listen to KH/aotonat backings vs originals: do they really still contain guitar?
- NVIDIA Control Panel: check the "CUDA - Sysmem Fallback Policy" setting (needed for bench items b1/b3).
- Close VRAM-heavy apps (Chrome, games) before GPU runs.
