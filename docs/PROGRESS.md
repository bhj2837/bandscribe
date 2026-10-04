# bandscribe progress / handoff

Last updated: 2026-10-04 (renamed gtab → bandscribe and moved the folder `D:\gtab` → `D:\bandscribe`, see
`docs/RENAME.md`; speed fix + review fixes measured, see "Speed and presence fix" below).

## Where things stand

| Milestone / phase | State |
|---|---|
| M0 foundation | ✅ done, on `main` as `287aead` |
| M1a+M2 spec (`docs/M1_M2_SPEC.md`) | ✅ done (30-point critique applied) |
| P0 contracts + 5 owners (SEP, AMT, GRID, EVAL, DATA) | ✅ implemented |
| Integration | ✅ 1174 tests passed (1 skipped) after the 2026-10-04 speed/presence fix; all 5 `D:\backing` songs + 1 Cambridge-MT mix run end to end |
| Rename gtab → bandscribe | ✅ done (`docs/RENAME.md`): 0 stage keys changed, old files load as is. Folder moved to `D:\bandscribe`, all three venvs re-created: keys again unchanged, 1198 tests passed (1 skipped) |
| Experiments | 🟡 mostly done (see table below) |
| Review → Fix → Acceptance | ⬜ not started |

All M1a/M2 work is in the WIP commit on branch **`wip/m1a-m2`**. `main` still points at M0. Merge into `main` after acceptance.

### Experiments (`docs/decisions.md`)

| E# | Status |
|---|---|
| E1 SW input loudness | Undecided. Default `off` kept: every CI includes 0. |
| E2 MuScriptor input view | Undecided. Default `guitar_mono` kept. |
| E3 instruments mask + presence threshold | Undecided. Default `guitar+present` / 0.5 kept: thin data. |
| E3b (same question on the production presence pass, + `guitar+present+strings`) | Pre-registered 2026-10-04, not run. |
| E23 Basic Pitch params | **Adopted** onset 0.6, frame 0.4, min_note_frames 7. Test Δ +6.6 [+3.8, +9.9]. |
| Latency | **Adopted** `amt.latency_s = -0.003` (descriptive; held-out 10.9 ms). |
| Gate-m2 (MuScriptor vs Basic Pitch, distorted) | Undecided. Δ +8.1 [+2.1, +13.5], but contamination is unknown, so not counted as a pass. |
| E24-sep, E24-amt (fp16 cast) | Pre-registered only, not run. Optional; skip unless cheap. |
| stereo-preservation (last entry in decisions.md) | Pre-registered only, not run. |
| Tier A descriptive pass on the 5 songs | Started (`data/scratch/exp/tiera_desc.py`), killed at the pause. Partial output is in `data/scratch/exp/tierA/`. |

## Speed and presence fix (2026-10-04, reviewed and fixed, committed as `2857d6d`)

Evidence (local only, `data/` is not in the repo): `data/scratch/speed/` (first fix) and `data/scratch/speed2/`
(review fixes: `results.json`, `retention2.json`, `analyze2.py`, `retention2.py`, `logs/`).

### Speed (done)

- `quality` (default) and `fast` no longer run the full-mix B0 pass. The full-length piano+other pass is replaced by a
  **sampled presence pass** (`bandscribe/amt/presence.py`): up to 6 × 10 s windows starting on bars, at most 60 s and 20 %
  of the song (`fast`: half). `eval` adds the B0 mix pass and the full-length piano+other pass **as extra outputs
  only** (`bandscribe run <곡> --profile eval`); `--system b0` and the Tier A baselines plan with `eval`.
- Measured on the RTX 2060 (quality, sep cached; "before" = the 2026-10-03 integration run):

  | Song | amt_ms1 before → after | amt_gtr after | whole run, from scratch (sum of stages) |
  |---|---|---|---|
  | 怪獣の花唄 (KH) | 730 s → 39 s (4 windows, 30 s decode) | 141 s | 986 s → ~342 s (16.4 → 5.7 min) |
  | 青と夏 (aotonat) | 1,205 s → 46 s (5 windows, 36 s decode) | 197 s | 1,477 s → ~463 s (24.6 → 7.7 min) |
  | SC | 394 s → 10 s (4 windows, 2 s decode) | 182 s (cached, same mask as 10-03) | 745 s → ~413 s (12.4 → 6.9 min) |

  - The amt_ms1 numbers assume the bass pass is cached. The bass pass is still full length (+49 / +94 / +61 s);
    the from-scratch sums include it.
  - **amt_gtr is now the largest stage** (141–197 s). Its time grows with the notes it decodes, which depends on the
    mask (aotonat 134 s with 3,565 notes, 197 s with 4,796).

### Instrumentation and guitar mask (review fixes)

The first fix added strings to the guitar mask and used the presence pass as evidence with no guards. A review plus
a backing check showed this did not reduce bleed and caused a regression on SC. Changes:

- **Guitar mask:** `guitar+present` (default) adds the present **keys/synth** classes again, which is the M2 set.
  Strings are an opt-in policy, `guitar+present+strings` (never `orchestral_harp` or `contrabass`). E3b, now
  pre-registered in decisions.md, will decide it on ground truth.
- **Presence guards:** a sampled class needs ≥ 8 notes, active windows in ≥ 2 distinct sections, and its own stem
  active in ≥ 10 % of bars (keys → piano, strings → other, organ/synth → either). Rejected classes are listed in
  `instrumentation.json` → `presence.rejected`.
- **Stem attribution:** a stem's energy counts only for the classes the transcription found, once the pass saw that
  stem in ≥ 4 covered bars. This removes 青と夏's organ / chromatic_percussion "present" from the piano stem alone.
  A skipped pass counts as coverage with e = 0, with no weight renormalisation. The bass pass now decides the bass
  classes.
- **`eval` estimates exactly like `quality`:** its mix pass is reported with weight 0, so eval no longer changes the
  system under test.
- **Window selection:** every active stem (piano, other) is sampled at least once. Windows keep a minimum gap of
  `max(window, song / 2n)` before section labels are considered. A window must lie inside its section. A window
  qualifies on frame share **or** power-mean level, and window length is clamped to the audio cap.
- **Other fixes:** an empty `segments` list now reads as a sampled pass. `s35.json` records `guitar_pass_mask` /
  `guitar_pass_non_guitar`. The notes stage prints a Korean warning when ≥ 10 % of guitar-view notes are dropped as
  non-guitar.
- Stage versions: amt_ms1 3, instr 4, s35 4, notes 3 (amt_gtr 2).

Measured after the review fixes. "Bleed share" = share of the original's guitar notes that also appear (same pitch,
onset ±50 ms) on the guitar-removed backing under **any** of the three masks tried, so it does not depend on which
mask the backing used:

| Song | guitar mask (non-guitar part) | guitar view → `guitar_all` | bleed share: 10-03 → strings mask → now |
|---|---|---|---|
| KH | 10-03: none; now: electric_piano | 3,543 → 2,307 (893 notes labelled electric_piano, dropped) | 17.7 % → 17.7 % → **11.9 %** (627 → 530 → 275 notes) |
| aotonat | 10-03: 4 keys; now: acoustic_piano, electric_piano, organ | 3,649 → 4,796 (0 dropped) | 17.0 % → 21.7 % → **17.7 %** (620 → 775 → 848 notes) |
| SC (control) | 10-03: none; now: none (electric_piano rejected: 1 section) | 4,759 → 4,759, **identical to 10-03** | – (backing guitar stem silent) |

- **SC regression fixed:** the sampled pass still finds 17 electric_piano notes in one section, but the sections
  guard rejects them. The guitar pass is a cache hit of the 10-03 transcription.
- **KH:** electric_piano passes every guard (194 notes in 2 sections; piano stem 16 % of bars), so it joins the mask.
  The guitar view then labels 893 notes (28 %) electric_piano.
  - Bleed share drops from 17.7 % to 11.9 %, but `guitar_all` shrinks by 35 %.
  - Only 366 of the 893 dropped notes (41 %) also occur on the backing, so **up to ~530 real guitar notes may have
    been relabelled away**. This is the same cost E3 saw on JetB (piano in the mask: F1 58.8 vs 65.8).
  - Whether KH really has electric piano is unverified. The old mix pass found none.
- **aotonat:** the guitar view labels nothing non-guitar, but dropping chromatic_percussion from the mask raised its
  note count by 31 % (3,649 → 4,796) and changed the class mix (clean 2,726 / distorted 1,155 / acoustic 915). The
  bleed share is unchanged (17.0 → 17.7 %).
- **Conclusion:** any change to the guitar-pass prompt reshuffles MuScriptor's output far more than it removes bleed.
  Measured as notes kept from 10-03:
  - KH kept 1,050 of 2,916 presumably-real notes and 239 of 627 bleed notes.
  - aotonat kept 1,372 of 3,029 and 476 of 620.
  - So keys in the mask is a trade-off with no ground truth yet: E3b, or a GP file for these songs, must decide it.
    Removing bleed after transcription (S7) is the better lever than widening the prompt.
- **Known limits** (documented in DESIGN S3.5 and in `instrumentation.py`):
  - A part that plays only outside the windows is missed (the safe failure for the mask).
  - Brass/winds can never be "present" from stems alone.
  - S7's bleed penalty (M6) needs the full-length piano+other pass (eval or on-demand).

## Priority for the next session

- **Decide keys in the guitar mask with evidence:** run E3b (ask before GPU > 30 min) or get a GP file for KH/aotonat. Until then the default stays `guitar+present` (keys/synth, as E3 left it). See the KH trade-off above.
- Remaining speed options (not done): `amt_gtr` (141–197 s) is now the largest stage; run the CPU Basic Pitch pass in parallel with GPU stages; upstream loader (1.46× vs 1.28× realtime) when VRAM allows; the bass pass is still full length (49–94 s).
- **Process:** M1a+M2 took ~2 days, mostly the serial GPU experiment queue (~7 h of GPU time). Next time:
  - keep experiments small (cap tracks, stop at the first clear answer);
  - run review/fix/accept without re-running experiments;
  - ask the user before any GPU job longer than ~30 min.

## Next steps (in order)

1. `git checkout wip/m1a-m2`. Quick check from the project folder: `bandscribe.cmd doctor` and
   `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
2. ~~Speed work~~ done 2026-10-04 (profiles + presence windows, reviewed and fixed, above). Optional: time `--profile fast` on SC.mp3.
3. Review (correctness + GPU/Windows), fix, then check acceptance against DESIGN §10 M1a/M2 (evidence list in M1_M2_SPEC §11). Tier A GP ground truth is BLOCKED on the user.
4. Merge `wip/m1a-m2` into `main` and commit the milestone.
5. Then M3 (first tabs: grid/quantization, Viterbi fretting, `.gp5`/alphaTex export).

## Facts worth keeping

- The Claude desktop app sandbox redirects `%APPDATA%`/`%LOCALAPPDATA%` writes, so everything lives under the project
  folder (`D:\bandscribe` on the dev PC). Use uv only through `tools\uvw.cmd`.
- **Envs:**
  - core `.venv`: no torch.
  - gpu `envs\gpu\.venv`: torch 2.11.0+cu128.
  - bp310 `envs\bp310\.venv`: Basic Pitch / onnxruntime.
  - Re-create an env with `uvw.cmd sync --locked --offline --compile-bytecode` from its folder (all packages are
    in `data\uv-cache`). `--compile-bytecode` matters: the Claude app sets `PYTHONDONTWRITEBYTECODE=1`, so a fresh
    venv used only from Claude never gets `.pyc` files, and CLI start-up doubles (cache hit 0.58 s → 1.1 s, which
    fails `test_cli_cache_hit_on_a_large_wav_is_under_1s`). On this USB disk a full re-create took core 24 min,
    gpu 6 min, bp310 3 min.
  - Stage params read `muscriptor` / `basic-pitch` versions from the gpu / bp310 envs' metadata. Never run
    `bandscribe` while an env is being re-created, or the keys change and stages recompute.
- **Cache state at current stage versions (2026-10-04, `quality`):** SC, KH, aotonat and the KH / aotonat backings
  are cached through `notes`. AIZO and AZ are cached only up to `resid1` (AIZO also `amt_ms1` + `amt_bp`, run
  10-04 as the post-move GPU check), SC_backing 8 of 13 stages, AIZO/AZ backings 2 of 13. A full `bandscribe run`
  on AIZO or AZ still costs roughly 5–7 min of GPU time.
- **VRAM per stage:**
  - SW: 1.2–1.7 GB reserved.
  - MuScriptor lean: 1.74 GB.
  - Beat This!: 0.37 GB.
  - Other desktop apps hold ~1.5–2 GB.
- **Backings (user-confirmed 2026-10-04):** the `*_backing.mp4` files are the user's earlier **AI guitar removals, which are imperfect**. The files without "backing" are the originals.
  - SC, AIZO and AZ: the guitar was removed (stem −19 to −36 dB).
  - KH (怪獣の花唄) and aotonat (青と夏): barely anything was removed. Their originals already show a weak guitar stem (−21 / −25 dB rel. mix), so treat both as **hard cases (guitar buried in the mix)** to inspect in later milestones.
  - **Diagnosis (2026-10-04):** by ear the user hears the guitar fully removed in both backings, and the original-minus-backing residual is −19 / −21 dB, so a real part was removed.
    - Yet only 22 % / 20 % of SW's "guitar"-stem active frames drop by more than 10 dB; for SC, AIZO and AZ it is 95–99 %.
    - So ~80 % of SW's guitar stem on these songs is **non-guitar content, very likely strings** (the user's hypothesis), and it would leak string lines into the guitar tab.
    - **Revised the same day at note level:** only ~17–18 % of the guitar notes also appear on the backing (bleed share, see "Speed and presence fix"), so most of that non-guitar energy never becomes guitar notes. A strings-widened guitar mask did not lower that share.
    - To do:
      - drop notes whose MuScriptor class is strings, or anything non-guitar, from guitar Lines;
      - keep KH and aotonat as regression cases for "strings misrouted to the guitar stem";
      - consider a backing-difference check in eval.
- **Separation check** (Cambridge-MT Secretariat vs true tracks): guitar 7.6 dB, bass 10.0, drums 7.9, vocals 9.0.
- **Known issue b13:** Hammond organ → SW `other` → MuScriptor calls it distorted guitar, so keys go undetected.
- **EGDB** is only partly downloaded (~5.3 GB left). `bandscribe data fetch egdb --yes` resumes it.
- At the pause, `data/gpu.lock.pids` was left by a hard-killed run. That is expected: the next GPU run waits on the listed (dead) pids and removes the file.
- **Evidence:** `data/scratch/{integ,exp,amt}` hold copyrighted-audio analysis. Never commit them (`data/` is gitignored).

## Pending on the user

- Optional: accept the Hugging Face gate for `MuScriptor/muscriptor-small` (VRAM fallback).
- Guitar Pro files (.gp/.gp5) for any song in `D:\backing`, to use as scored ground truth (Tier A).
- Listen to KH/aotonat backings vs originals: do they really still contain guitar?
- NVIDIA Control Panel: check the "CUDA - Sysmem Fallback Policy" setting (needed for bench items b1/b3). A
  per-program setting must target `D:\bandscribe\tools\python\cpython-3.12-windows-x86_64-none\python.exe`; one made
  on the path from before the folder move no longer applies.
- Close VRAM-heavy apps (Chrome, games) before GPU runs.
