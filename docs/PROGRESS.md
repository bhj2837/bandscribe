# bandscribe progress / handoff

Last updated: 2026-10-05 (**grid fix after M4a**: beats that Beat This! tracked twice are joined, see "Grid: beats
tracked twice" below. **M4a done**: editing in the viewer, see "M4a" below and `docs/acceptance/M4a.md`.
After M3: viewer scroll, silence gate, `hints.title`, see "After M3" below.
**M3 done with known gaps**: first tabs, see "M3 (2026-10-05)" below and
`docs/acceptance/M3.md`. Before that: M1a accepted and tagged `m1a`; M2 tagged `m2` with known gaps, see
`docs/acceptance/M1a_M2.md`; fixes for the two code reviews, see "Review fixes (2026-10-05)"; E3c run and rejected. 2026-10-04: distractor measurement, rename gtab → bandscribe, folder `D:\bandscribe`, speed fix + review
fixes).

## Where things stand

| Milestone / phase | State |
|---|---|
| M0 foundation | ✅ done, tag `m0` |
| M1a+M2 spec (`docs/M1_M2_SPEC.md`) | ✅ done (30-point critique applied) |
| P0 contracts + 5 owners (SEP, AMT, GRID, EVAL, DATA) | ✅ implemented |
| Integration | ✅ 1174 tests passed (1 skipped) after the 2026-10-04 speed/presence fix; all 5 `D:\backing` songs + 1 Cambridge-MT mix run end to end |
| Rename gtab → bandscribe | ✅ done (`docs/RENAME.md`): 0 stage keys changed, old files load as is. Folder moved to `D:\bandscribe`, all three venvs re-created: keys again unchanged, 1198 tests passed (1 skipped) |
| Experiments | ✅ run or closed for M2 (see table below); E3b still pre-registered only |
| Review → Fix → Acceptance | ✅ two reviews (0 blockers), 14 fixes, acceptance report `docs/acceptance/M1a_M2.md`: **M1a 13/13 pass** (tag `m1a`); **M2 19 pass / 3 fail / 3 blocked** (tag `m2`, accepted with known gaps by the user's decision) |
| M3 first tabs | ✅ quantisation, A4, tuning, Viterbi fretting (E12 → E12b), alphaTex/GP5/MIDI export, synced viewer; acceptance `docs/acceptance/M3.md`: **9 pass / 1 fail / 3 blocked** (tag `m3`) |
| M4a editing | ✅ keyboard edits in `bandscribe view`, `edits.jsonl`, replay + re-export, re-applied after `run`; acceptance `docs/acceptance/M4a.md`: **4 pass / 1 blocked** (human baseline) (tag `m4a`) |

All work is on **`main`**; milestones are tags (`m0`, `m1a`, `m2`, `m3`).

## M4a (2026-10-05): editing in the viewer

**User decision:** extend the existing HTML viewer (no FastAPI/Vite build), then M4a. `bandscribe view <job>` is the
editor (`--read-only` to only look). Code: `bandscribe/tab/edits.py` (log, anchors, replay, re-export, `Session`),
`bandscribe/tab/viewer.py` (`/api/score`, `POST /api/cmd` with a per-server token header and a Host check),
`bandscribe/tab/viewer/index.html` (selection box, keys, help), `fretting.assign(pins=)`, and `stage.py` split into
reusable `fret_tracks` / `write_score` / `score_doc` / `title_for` (stage outputs unchanged).

- Keys after clicking a note (each one keystroke): ←/→ next note, ↑/↓ semitone, Shift+↑/↓ octave, Delete, S / Shift+S
  other string (pinned; neighbours re-fretted around it), 1/2 to guitar/bass, A add at the playback position, [ / ]
  bar line one beat earlier/later (later bar lines move too), D tap "this is a downbeat" (taps during playback are
  applied on pause), Ctrl+Z / Ctrl+Y.
- `edits.jsonl` (job dir) is append-only; every edit is stored absolute; undo/redo are entries that switch an earlier
  edit off/on. Anchors: (track, raw onset, pitch); added notes by edit id; bar lines by beat time (within a third
  of a beat). After a re-run an anchor takes the nearest note of its pitch within 30 ms, else it is reported
  unmatched.
- Each edit re-quantises (bar lines only) and re-frets the changed track, renders, and the export is rewritten in
  the background: 0.3–0.6 s per note edit, 1.4 s per bar-line edit on seisyun complex. `bandscribe run` re-applies
  the log after publishing (`edits.reapply`) and warns about unmatched edits.
- With every edit undone the export is byte-identical to the stage outputs (score.json differs only in its time).
- One review (8 defects, all fixed with tests): pickup bar kept, edits logged only after they replay and render,
  torn log lines repaired, a viewer never writes an older run over a newer export (it reloads), export written
  aside and swapped GP5 first, bar-line anchors by beat time, page focus/busy fixes, no framing. One viewer per job
  (`edits.lock`).
- Acceptance: `docs/acceptance/M4a.md`.
- **Found while measuring drift (not M4a): the beat grids of aotonat, AIZO and AZ contain spurious beats 50–100 ms
  after real ones** (8.4 / 10.0 / 3.5 % of intervals shorter than half the median; SC and KH 0 %). The pairs are
  already in `t_raw_s`. Inside such bars the cursor is up to ~300 ms off and the quantisation is bent. Fixed in a
  separate session (grid stage); its first finding: the downbeat snap adds no beat, Beat This!'s activation is
  two-peaked there and the peak picking turns one beat into two beats 80–120 ms apart. **Fixed** (grid step 1a,
  "Grid: beats tracked twice" below): short intervals 8.4 → 0.5 % on aotonat (bars with a beat > 50 ms off the
  cursor 22 → 2), AIZO 10.0 → 3.5 %, AZ 3.5 → 2.7 %; what is left there is the tracker following double time.

## Grid: beats tracked twice (2026-10-05)

Follow-up of the drift measurement in "M4a" (commit `ba04242`). Code: `bandscribe/analysis/grid.py` step 1a
(`_merge_doubles`), grid stage code version 2 → 3; 5 new tests in `tests/test_grid_derive.py` (the three that build
doubles fail on the old code, which made bars of 6 and 5 beats out of the 青と夏-shaped one).

- **Cause (checked on the stored `beats` outputs).** Not `downbeat_snap_ms`: the grid's downbeat check never adds a
  beat, and every downbeat of aotonat / AIZO / AZ (139 / 113 / 128) sits exactly on a tracked beat (0 snapped,
  0 dropped). The pairs come from Beat This! itself: its minimal postprocessing keeps every frame that is the maximum
  of its ±3 frames (60 ms; its comment says ±70 ms) and above 0.5, so a beat whose activation has two humps gives two
  beats 80–120 ms apart (aotonat 122.44 / 122.52 s: the activation stays ≥ 0.72 between the two maxima). When the
  downbeat activation has the same double hump, both beats are downbeats (6 of aotonat's 22 pairs), which looked like
  a downbeat added next to a beat. On aotonat the later beat of a pair has the higher activation in all 22 pairs and
  the stronger percussive onset in 19 (80 ms is a sixteenth at the real 185 BPM, so the early peak is probably a
  push before the beat).
- **Fix (step 1a, before the one-level step 1b).** An interval shorter than 0.42 × *both* neighbouring intervals
  (below 1b's half-level band, so never a level switch or a phase jump) joins its two beats: the one with the higher
  beat activation stays (then the one leaving the outer intervals more even, then the earlier) and is a downbeat if
  either was. Runs of short intervals (fills tracked in sixteenths: AZ 105–120 s, SC 68 s) are left alone. New
  warning `비트 추적기가 한 박을 두 번 잡은 곳 N군데를 한 비트로 합쳤습니다`, new field
  `hypotheses.downbeat_check.doubles_merged`. Joined: aotonat 22, AIZO 10, AZ 3, SC 0, KH 0.
- **Results** (grid stage output before → after; "off > 50 ms" = a beat more than 50 ms from the viewer cursor's
  linear position inside its bar, computed from `grid.json` at every beat):

  | Song | intervals < ½ median | median IBI (BPM) | bars | bars with a beat off > 50 ms | in-bar offset p95 / max |
  |---|---|---|---|---|---|
  | aotonat (青と夏) | 8.4 % (37) → **0.5 % (2)** | 644.5 → 649.8 ms (93.1 → 92.3) | 111 → 103 | 22 → **2** | 241 / 425 → 31 / 334 ms |
  | AIZO | 10.0 % (36) → 3.5 % (12) | 630.1 → 634.0 ms (95.2 → 94.6) | 92 → 87 | 14 → 10 | 147 / 334 → 167 / 350 ms |
  | AZ | 3.5 % (13) → 2.7 % (10) | 540.0 → 540.0 ms (111.1) | 92 → 90 | 22 → 18 | 150 / 421 → 151 / 281 ms |
  | SC | 0 → 0 | 316.6 ms, unchanged | 158 | 0 → 0 | 26 / 49 ms |
  | KH | 0 → 0 | 400.0 ms, unchanged | 138 | 0 → 0 | 27 / 45 ms |

  - SC and KH: `grid.json` identical apart from the new `doubles_merged: 0`; sections and presence windows identical;
    every export file byte-identical (`score.json` apart from its creation time); the MuScriptor views came from
    the transcription cache.
  - **What is left is not doubles** but the tempo-level question (step 1b / octave, not touched here): aotonat 2
    single half-level beats at 121–127 s where the tracker followed 185 BPM (the real tempo, per the user); AIZO
    29–44 s, where the tracker follows double time with 240–380 ms jitter (outside 1b's ±19 % band) and puts four
    weak peaks at 29.8–30.3 s; AZ, tracked in eighths and halved, where downbeats half a beat apart restart the
    halving. These are the bars still off the cursor. AIZO's two bars at 54–59 s are fixed; its 29–44 s passage is
    wrong before and after (10 → 8 bars off, worst beats a little further off, so the p95 rose).
  - AZ odd bars 3 → 6: the intro (0–12 s) and 105–120 s have contradicting downbeats half a beat apart; joining the
    doubled downbeats at 5.12 / 5.22 s and 122.40 / 122.52 s changed which of them the halving and the meter DP
    follow. Neither version can be checked without ground truth.
- **Re-run** (`bandscribe run <job>`, the five jobs): grid 24–35 s and sections 2–3 s per song on the CPU.
  - SC and KH: everything else from the caches (MuScriptor views from the transcription cache, 0.2–0.7 s).
  - aotonat, AIZO, AZ: the moved bar lines moved 1–2 presence windows (they start on bar lines), and the window list
    is part of the MuScriptor cache key, so `amt_ms1` re-decoded their windows on the GPU (59 / 52 / 25 s). AIZO and
    AZ kept their guitar mask (`amt_gtr` from the cache, note counts unchanged; MIDI and tabs change with the grid).
  - **aotonat side effect:** the new windows found `synth_strings`, which joined the guitar mask, so `amt_gtr` ran
    again (4:08 on the GPU) and MuScriptor's guitar output changed far more than the grid did: 4,796 → 6,083 view
    notes (acoustic 691 → 2,310, distorted 1,105 → 496), `guitar_all` 3,996 → 4,978 after the silence gate. Without
    ground truth neither version can be called better; it is the prompt sensitivity seen in "Speed and presence
    fix". Presence windows on a fixed time grid instead of bar lines would keep grid fixes from reaching the guitar
    mask (not done).
  - GPU in total ~6.4 min.

## After M3 (2026-10-05, user feedback on seisyun complex)

The user put Guitar Pro PDF exports of SC (**seisyun complex**, 青春コンプレックス; "SC" was only a file name) for
every part in their sheet folder. A throwaway extractor and comparison live in `data/tmp/sheets_sc/` (gitignored;
`pdf_tab.py` needs `--with pypdfium2`; `eval_sc.py`, `bass_rest_sc.py`, `gate_energy_check.py`, `barmap_sc.py`). GT bar
n = our bar n for bars 2–116; GT 117 (2/4) sits at the end of our 6-beat bar 116, so later bars are ours = GT − 1.

- **Viewer:** following the playback scrolled the page and put the current line under the sticky header (86 px
  hidden at 1280×800). Only the score area scrolls now (alphaTab `scrollElement`), the line 12 px below the header.
- **Silence gate (`notes`, `amt.silence_gate_db = -50`, `bandscribe/amt/silence.py`):** MuScriptor writes notes over
  silent stems, often one note or chord again and again. SC: 81 G1 in bars 5–17 where the score's bass rests; the
  bass stem there is −109 dBFS and the onsets do not follow the kick (12 % within 40 ms, chance 13 %). Notes that
  start where their stem is > 50 dB under the mix's p95 frame level are dropped from `guitar_all` and the bass
  (`notes/bass_raw.json`, which `quant` now reads; `notes` depends on `stems` for `energy.npz`). Dropped: SC bass 78,
  KH guitar 685 + bass 93, aotonat guitar 800 + bass 33, AIZO guitar 16 (after the song ends), AZ 0. SC bass bar-bag
  F1 86.0 → 90.1 (verse 23.3 → 100; all 735 matched notes kept); SC guitar unchanged (nothing dropped). No bass note
  of any song lies between −50 and −40 dB; the matched SC bass notes are all within 12 dB of the reference. The
  guitar levels are not bimodal on KH/aotonat (604 aotonat notes at −50…−40 dB stay): no guitar GT there yet.
  Left in SC bars 4–6: a loud falling line in the bass stem (F2 → G1, −4 → −30 dB) where the score has dead notes
  then rests. **User (2026-10-05): it is one long slide down from fret 3**, written by us as separate notes
  (3 1 0 4 3 … on the tab). Slide detection belongs to M5 (bass techniques).
- **`hints.title`:** `[hints] title = "..."` in a job's `hints.toml` names the song in the score, GP5, viewer and
  `status` (input files stay read-only). SC's job has `title = "seisyun complex"`.
- The eval `job:notes` system reads `guitar_all.json`, so guitar suites now score the gated output; bass suites read
  the raw `amt_ms1` file as before.

## M3 (2026-10-05)

**What a run produces now** (`bandscribe run <song>`, default `--until score`): `export/tab/score.alphatex` (Guitar
(all parts) as a staff, Bass (raw) as tab, one `\sync` per bar at the recording's bar time), `export/tab/score.gp5`
(fretted tracks only), `export/tab/*.txt`, `export/tab/tab.json` (tuning + fingering), `export/midi/*_quantized.mid`,
`export/score.json` (schema v0). `--parts 1` frets the merged guitar too. `bandscribe view <song>` serves the score
with alphaTab in external-media mode on 127.0.0.1 (cursor within 31 ms of the grid over a whole song).

**New stages (CPU, ~13 s per song together):** `a4` (A4 offset from the pitched stem views), `quant` (per-beat
binary/ternary/swung Viterbi, 12/8 vs shuffle re-read, triplet feel per section), `tab` (joint guitar + bass tuning
search with fake-low-note guard; window-model fretting; out-of-range notes written an octave inside, flagged),
`score` (rhythm spelling → alphaTex, GP5, quantised MIDI, score.json). Code in `bandscribe/tab/`.

**Decisions:** E12 hold (v1 Viterbi 57.4 vs tuttut 46.7 per track, short of 75), E12b adopted (window model 69.3 per
track / 74.6 % per note). E14 (quantisation settings) and E16 (A4 varispeed) pre-registered and waiting (Tier A GT;
GPU). σ 30 ms is provisional.

**M3 known gaps** (`docs/acceptance/M3.md`):
- FAIL c7: fretting ≥ 75 % on GuitarSet test (69.3 per track, 74.6 % per note); ≥ tuttut passes.
- BLOCKED c1/c3/c8: Tier A ground truth (downbeat F1 / bar count, notation match, oracle-line fretting).
- BLOCKED c10 (TuxGuitar half): opening the GP5 in TuxGuitar is a manual check (alphaTab and PyGuitarPro read it).
- Not done: A4 varispeed (E16); the 5 test songs are +2–4 cents, so it does not apply to them.
- User answers (2026-10-05): 青と夏 is **185 BPM** (our grid: 93, half tempo → 8ths written as 16ths; a tempo hint
  or a better tempo-octave rule is open), AIZO 95 BPM (grid right). KH: the user plays it in **standard** tuning
  (unsure whether the original is half-step down; the recording's guitar plays D2 112 times, so the search says
  Drop D). Added `tab.guitar_tuning` / `tab.bass_tuning` (DESIGN 6.1 (f), aliases "standard", "하프다운", "drop d";
  unplayable notes octave-shifted and warned; the automatic pick stays in `tab.json` `auto_label`). KH's job has
  `hints.toml`: `guitar_tuning = "E standard"`, `guitar_parts = 1`.

**M2 known gaps** (accepted by the user on 2026-10-05 so that M3 can start; none blocks M3):
- FAIL b10: MuScriptor latency residual 10.9 ms vs < 10 ms (provisional threshold).
- FAIL b13b: per-song instrumentation table of the current estimator needs E3b (pre-registered, not run).
- FAIL b15i: E24-sep / E24-amt closed as 판단 보류 without running (GPU cost 40 min+).
- BLOCKED b3: Sysmem Fallback on/off bench needs the user to switch the NVIDIA setting.
- BLOCKED b12: gate-m2 passes numerically (Δ +8.1) but contamination is unknown; needs uncontaminated distorted GT.
- BLOCKED b13c: Tier A instrumentation GT (and M1b in general) needs the user's Guitar Pro files.

### Experiments (`docs/decisions.md`)

| E# | Status |
|---|---|
| E1 SW input loudness | Undecided. Default `off` kept: every CI includes 0. (Numbers used unmerged references: biased, see "Review fixes".) |
| E2 MuScriptor input view | Undecided. Default `guitar_mono` kept. (Unmerged references: biased.) |
| E3 instruments mask + presence threshold | Undecided. Default `guitar+present` / 0.5 kept: thin data. (Unmerged references: biased.) |
| E3b (same question on the production presence pass, + `guitar+present+strings`) | Pre-registered 2026-10-04, not run. Amended 2026-10-05 before its first run: references merged across guitar lines. |
| E3c (`guitar_only` vs production mask, held-out windows) | **Rejected** 2026-10-05: Δ −2.35 [−8.22, +0.47] (non-inferiority margin −1.0). JetB: the keys mask removed 191 piano-bleed notes (F1 61.9 → 76.3); Zeno: −1 point. Default `guitar+present` kept. |
| E23 Basic Pitch params | **Adopted** onset 0.6, frame 0.4, min_note_frames 7. Test Δ +6.6 [+3.8, +9.9]. |
| Latency | **Adopted** `amt.latency_s = -0.003` (descriptive; held-out 10.9 ms). |
| Gate-m2 (MuScriptor vs Basic Pitch, distorted) | Undecided. Δ +8.1 [+2.1, +13.5], but contamination is unknown, so not counted as a pass. |
| distractors (which sounds cause guitar FP/FN; descriptive) | **Run** 2026-10-04 (`20261004-192625_e3f8e5f_distractors`), recorded 판단 보류 (descriptive; its "mask filters nothing" hypothesis was rejected by E3c). Results below. |
| E24-sep, E24-amt (fp16 cast) | Closed as 판단 보류 without a run (GPU cost > 40 min). E24-sep amended 2026-10-05 (merged references) in case it is re-registered. |
| stereo-preservation (last entry in decisions.md) | Pre-registered only, not run. |
| Tier A descriptive pass on the 5 songs | Started (`data/scratch/exp/tiera_desc.py`), killed at the pause. Partial output is in `data/scratch/exp/tierA/`. |
| E12 fretting v1 vs tuttut (GuitarSet, dev = progression 1, test = 2–3) | **Hold (target missed)** 2026-10-05: dev-best v1 57.4 vs tuttut 46.7 points per track, Δ +10.65 [+8.91, +12.47]; registered target ≥ 75 not met. |
| E12b fretting window model vs E12's best v1 | **Adopted** 2026-10-05: `window`, shift 0.5, height 0, open 0 → 69.3 points per track (74.6 % per note) vs 57.4, Δ +11.96 [+8.41, +15.34]; solos +21.2. M3 fretting criterion (≥ 75 per track) still missed. |
| E14 quantisation settings (σ, levels, 24 vs 48 tick) | Pre-registered, waits for Tier A GT. σ 30 ms provisional (synthetic 0.990, fewer spurious 32nds on real songs). |
| E16 A4 varispeed threshold | Pre-registered, not run (GPU ~20–30 min). M3 measures and warns only; the 5 songs are +2–4 cents. |

## Review fixes (2026-10-05)

Two code reviews of M1a/M2 at `b50ea18` (correctness; GPU/Windows). Every fix below is in the working tree with
tests; nothing is committed. Test and SC-run results are at the end of this section.

**Correctness**

1. **Experiments merge references and estimates** (`amt/experiments.py` `merged`, `_score_merged`). The reftx of
   several guitar lines were concatenated, so a note two lines play together counted twice (~11 % of the 22,382 Tier B
   reftx guitar notes, Mistrusted 23 %), and one onset labelled with two guitar classes counted twice (another 11 %).
   That capped recall and could bias arm deltas. Rule = `eval.suites._merged_arrays` with a 50 ms tolerance (the
   metric's onset tolerance; the suites' 1-tick 10.5 ms merges only ~5 % of the cross-line copies, because double
   tracks are offset by human timing). Used by E1–E3b and E24-sep (incl. the synthetic GT of E3/E3b); single-part GT
   experiments (E23, gate-m2, latency, E24-amt) are unchanged. decisions.md: pre-run amendments to E3b and E24-sep;
   bias notes on E1/E2/E3 (all 판단 보류, so no default changed).
2. **`job:<stage>`, `b0`, `no_split` never score a stage dir made with other settings.** When the planned (current
   settings) dir is missing, the song is skipped with a Korean warning (console, `summary.json` `warnings`); `--arg
   allow_stale=true` opts in (note says `(stale)`). Every metrics row's `note` has `stage=<key12>`, run.json has
   `stage_keys`. An unplannable job (no config, foreign layout) still uses the newest dir.
3. **Tuning rule `exclude=<prefix>[|...]`** (also honoured by select-then-confirm). E23's 판정 설정 now has
   `exclude=f12_`; re-evaluating the 2026-10-03 E23 rows picks `f7_o0.6_fr0.4` (test Δ +6.56) = the recorded
   decision (the runner's raw pick was `f12_o0.6_fr0.3`). Amendment note in E23; the decision is unchanged.
4. **Distractor report: FP cause × loudest sound per condition and arm** (`summary.json` `fp_cause_vs_dominant` =
   {arm: {condition: {cause: {sound: n}}}}; summaries from before still render as one pooled table). The
   `guitar_timing` label is given before the loudest sound is looked at (attribution rule 1), so "pads hurt through
   the guitar's own segmentation" was not established (softened here, in DESIGN §8.8, README and decisions.md).
   Statements that the `guitar+present` mask "filters nothing / only reshuffles the output" were corrected (E3c:
   JetB +14 points). Rerun from the caches (0 GPU; local evidence `data/scratch/review1005/`): on the three pad
   windows (3.75 min) the pad adds 69 `guitar_timing` FPs (+18.4/min); in 19 of them (5.1/min) the pad is the
   loudest sound in the note's band, in 53 more (14.1/min) the guitar is. So most of the increase sits in
   guitar-dominated bands, but about a quarter is pad-dominated, and whether the pad makes the guitar re-trigger
   or the guitar splits on its own cannot be told apart by this attribution.
5. **MIDI lead-in tempo within 30–300 BPM** (`amt/midi.py`): the old rule is kept where its tempo is in range;
   otherwise another `k/4` (slow songs), else eighths or sixteenths (`k/8`, `k/16`); a lead-in < 50 ms is absorbed
   into the first beat, one > 255 units becomes several 4/4 bars. SC's 0.16 s lead-in was a 1/4 bar at 375 BPM, now 1/8 at 187.5 BPM. Note times
   unchanged. `notes` stage code version 3 → 4.
6. **Cache keys.** Basic Pitch: `threads` joins the transcription-cache key (the `amt_bp` stage key already had it;
   1 or 2 vs 4 ONNX threads change ~1,700 values of a 30 s clip's output by 1.2e-7, measured), and the model-output
   npz name gets `__t<threads>` except for 4 threads (the name every existing entry has). `beats` params now include
   the installed beat-this version (1.1.0), like MuScriptor / Basic Pitch: **every existing job's `beats` key changes
   once**, so beats re-runs on the GPU (seconds) and grid → sections → … → notes recompute on the CPU; the MuScriptor
   views come from the transcription cache as long as the grid comes out the same.
7. **a7 test through `suites.evaluate_item`**: predictions with their performed onsets and the (bar, tick) of a
   wrong system grid score tick F1 = 1.0 and the same rows as with GT or no bar/tick.
8. `latency`: `egdb_dev` defaults to False (the registration amendment excludes EGDB).
9. Basic Pitch: config and worker refuse `min_note_frames >= 12` (was only `== 12`) unless `allow_default_min_len`.

**GPU / Windows**

10. **Load-time spill** (`workers/gpu_common.py`): shared usage at `model_loaded` minus the `pre_load` mark (taken
    after the CUDA context, before the first upload) above `gpu.load_spill_fail_mb` (48 MB, [잠정]) fails the rung.
    Bench 2026-09-30 17:58 / 18:01 had 130–136 MB vs 66 MB and accepted `chunk=352256` at 1.4x / 0.87x realtime
    (benched 5.3x); normal loads add 0–4 MB, MuScriptor's upstream loader 32 MB. A rung below one that failed with
    SharedGrowth may not be slow (slow → failure → the stage waits and retries).
11. **Worker warnings reach the console**: `sep`, `amt_ms1`/`amt_gtr` and `beats` forward the worker's warnings
    (slow rung; MuScriptor no-EOS, now one Korean line per view) through the stage notifier; a freshly run lower
    rung warns live; `gpu_run.json` keeps `slow` and `warnings`; `run_warnings` shows cached degraded / slow / worker
    warnings (older `sep` dirs: from `sep.json`); `bandscribe run` prints each text once.
12. **MuScriptor: slowness never fails a rung** (`run_ladder(slow_can_fail=False)`: decode time depends on note
    density); only shared-usage growth (after or during the load) does.
13. **Separation RAM**: `demix` shares the mix (`torch.from_numpy`), drops the padded copy before the division and
    replaces NaN/inf in place (`torch.nan_to_num_`, same values) instead of numpy's mask-heavy `nan_to_num`. 10-min
    synthetic stereo input, identity model, CPU: peak working set 3.83 → 2.32 GB, peak commit 4.60 → 3.08 GB,
    outputs bitwise identical (VENDOR.md note; `demix.py` is ours, no sha change). `bandscribe run` / `ingest` warn
    in Korean above 15 min with an estimate (1.5 GB + 9.5 × the float32 mix: 4 min ≈ 2.3 GB, 15 min ≈ 4.3 GB).
14. **`gpu.wait_total_max_s`** (3600 s) bounds all the waiting of one GPU stage (pre-checks + retry pauses; before,
    each pre-check could wait `wait_max_s` again: ~4 h with 3 retries); a "아직 … 기다리는 중" line every 5 minutes.

New config keys: `gpu.load_spill_fail_mb = 48`, `gpu.wait_total_max_s = 3600`. Deferred items: see "Next steps".

**Tests:** full default suite 1346 passed, 2 skipped (6 min 57 s on the final state; incl. the GPU tests: Beat This!
worker load check 0 MB, SW upstream parity unchanged). **SC check** (`bandscribe run D:\backing\SC.mp3 --until notes`): first run after
the change ~39 s (beats 7.5 s on the GPU, grid 23.6 s, the rest < 3 s each; `amt_ms1` / `amt_gtr` from the
transcription cache, no MuScriptor worker), new MIDI lead-in 1/8 at 187.5 BPM; the next run, fully cached, 0.96 s.
Warnings printed by both: the cached `sep` slow flag (`chunk=588800: 실시간 배수 3.506x`, recorded by the 2026-10-03 run
before the watchdog's warm-up fix; now surfaced by fix 11, clears with `--force sep`) and the grid repair note.

## Distractor measurement (2026-10-04)

`bandscribe eval run --suite distractors` (DESIGN §8.8, pre-registered as `distractors` in decisions.md). Code:
`bandscribe/eval/{distractors,attribution,stem_taxonomy,distractor_report,backing}.py`. Evidence (local only):
`data/runs/20261004-192625_e3f8e5f_distractors/` (report.md / report.html / summary.json / metrics.csv), caches in
`data/eval/distractors/` and the MuScriptor eval cache, logs in `data/scratch/distr/`.

- **Data:** 7 Cambridge-MT songs added (optional `required = false` parts of `cambridge_mt`, 2.48 GB, sha256 in
  datasets.lock.json): JamesElder (synth pad/lead/fuzz/FX), Jokers (synth pad/lead), Zeno (synth pad, piano),
  RememberDecember (strings, sub drop), SigneJakobsen (string pad, organ, piano), Mu (Rhodes, synths, Rhodes FX),
  Moosmusic (Rhodes). With the old JetB (piano, SFX), Mistrusted (stylophone), Secretariat (Hammond): 10 songs x one
  75 s window (no song met the second-window rule; ButterflyEffect's SFX never overlaps the guitar enough).
  Stem taxonomy: 293 file names on disk, 0 unknown.
- **Method:** conditions base (guitars+drums+bass+vocals) / +one category / full, same mastering gain curve for all;
  reference = MuScriptor on the true summed guitars; system = production guitar path (SW -> presence pass +
  instrumentation mask -> MuScriptor), plus the guitar-only-mask arm and a decoder null (base view + -90 dBFS noise).
- **Results** (production arm, delta vs base, song-block 95 % CI; the reference is MuScriptor-derived, so no
  absolute claims):

  | Added sound (songs) | dFP/min | dFN/min | d recall (pts) | dF1 (pts) | dF1, guitar-only mask | leak into SW guitar | top FP cause |
  |---|---|---|---|---|---|---|---|
  | synth pad (3) | +18.7 [+12.0, +24.0] | +24.8 [+13.6, +44.0] | −3.7 [−6.1, −2.2] | −3.1 [−4.9, −1.9] | −2.4 [−4.9, +0.1] | −12.5 dB | labelled `guitar_timing` +18.4/min, labelled pad +1.9 (the timing label is given before the loudest sound is checked: see Review fixes 4) |
  | strings (2) | +55.6 [+50.4, +60.8] | +50.0 [+8.8, +91.2] | −5.5 [−10.8, −0.9] | −5.9 [−9.9, −2.7] | **+0.5 [0.0, +0.9]** | −11.5 dB | the mask (synth_strings / organ added), not the sound |
  | piano (3) | +21.9 [+4.0, +40.0] | −0.8 [−8.0, +4.8] | +0.1 [−0.7, +2.0] | −1.4 [−2.8, +0.7] | −1.5 [−2.8, −0.7] | −12.9 dB | guitar timing +16, octave +8 |
  | organ (2) | +17.6 [0.0, +35.2] | +4.0 [0.0, +8.0] | −0.6 [−1.5, 0.0] | −1.5 [−3.8, 0.0] | −1.5 [−3.8, 0.0] | −14.4 dB | organ-dominated +14.0/min |
  | FX / risers (3) | +9.1 [−4.8, +27.2] | 0.0 [−16.0, +10.4] | 0.0 [−1.9, +4.3] | −0.6 [−1.3, +0.6] | same | −19.0 dB | vocals +5.9, octave +4.8, FX +4.3 |
  | synth lead (3) | −82.4 [−247, +0.8] | +2.1 [−0.8, +6.4] | −0.3 [−1.2, +0.1] | +4.6 [−0.1, +12.7] | same | −16.3 dB | (FP went **down**: see the null) |
  | synth, unnamed (2) | −81.6 [−92.8, −70.4] | +16.4 [+3.2, +29.6] | −3.6 [−8.0, −0.6] | +2.7 [+2.1, +3.0] | same | −16.4 dB | synth-dominated +30.4, drums +21.6 |
  | electric piano (2) | −48.4 [−102, +5.6] | +1.6 [−19.2, +22.4] | −0.3 [−3.7, +5.2] | +3.4 [−2.5, +11.1] | +0.2 [−2.5, +2.9] | −23.6 dB | (FP went down: Mu, see the null) |
  | full mix (10) | +0.5 [−67.8, +53.2] | +36.0 [+10.6, +68.8] | −5.6 [−10.1, −1.7] | −3.3 [−8.0, +1.7] | −0.7 [−4.0, +3.1] | – | octave +3.0, organ +2.8 |
  | decoder null (10) | −13.0 [−44.0, +5.6] | +8.1 [+0.4, +20.1] | −1.3 [−3.0, −0.1] | +0.1 [−1.7, +2.3] | same | – | median abs. change 0.4 FP / 0.8 FN per min, but Mu −145.6 FP/min and Signe +56 FN/min |

  - base itself (drums+bass+vocals only): F1 82.9, FP 130/min, FN 95/min vs the clean-guitar reference. Biggest
    causes in the full mix: the guitar's own timing/splitting (FP 67.7/min, FN 70.6/min: same pitch, 50–200 ms off)
    and octave (FP 27.7, FN 40.9/min). All FPs *labelled* with a distractor (its sound loudest in the note's band)
    add up to ~7/min; FPs with a same-pitch reference note within 200 ms are labelled `guitar_timing` first, whatever
    is loudest, so they are not in that count (Review fixes 4).
  - **Mask:** in the 13 conditions where the production mask added keys/synth classes, MuScriptor labelled **0**
    notes non-guitar; pooled F1 85.3 (guitar-only mask) -> 82.3 (production), mean −2.1 points, 9 of 13 worse.
    **This did not generalize** (E3c, 2026-10-05, windows this run did not use): on JetB the keys mask removed 191
    piano-bleed notes (F1 61.9 guitar-only → 76.3 production); on Zeno it cost 1 point. The mask's effect varies by
    song; `guitar_only` as the default was rejected.
  - SW guitar stem: guitar retention −0.5 dB; 81–85 % of its energy lies in guitar-dominated bins, each distractor
    category ≤ 1.1 %.
- **Surprises:** (1) adding a synth lead / unnamed synth / Rhodes made FPs go *down* by 50–80/min; (2) an inaudible
  −90 dBFS change moved FP by −146/min on Mu and FN by +56/min on Signe: MuScriptor's greedy decoding is unstable on
  some excerpts, so 2–3-song category deltas can sit inside decoder noise; (3) the strings "damage" is entirely the
  mask; (4) the FPs a pad adds are mostly labelled `guitar_timing` (same pitch as a reference note within 200 ms),
  few as pad-pitched notes. That label is given before the loudest sound in the note's band is checked; per
  condition, 19 of the 69 added timing FPs have the pad as the loudest sound, 53 the guitar. So whether the pad made
  the guitar re-trigger or the guitar's own segmentation did is not settled (Review fixes 4).
- **What to fix first** (report ranking): guitar timing/segmentation (FN 70.6 + FP 67.7 /min) -> octave (FN 40.9 +
  FP 27.7) -> decoder instability (an inaudible change moves 25.4 errors/min on average) -> strings (mask-driven,
  21.1 weighted) -> synth pad (13.0) -> piano (6.3).
- **Limitations:** one 75 s window per song (2–3 songs per category, CIs from very few blocks); excerpts are separated
  alone (edge context differs from a whole-song run: 26 dB SNR in the 7 s edges, 51 dB inside); instrumentation from
  one 10 s presence window per excerpt; the reference is MuScriptor (its misses show up as `guitar_unref`); synths
  whose file names say nothing stay `synth_other`; the null covers MuScriptor only, not SW.
- **GPU time:** first full pass 51.6 min (SW 16.3 in 6 batch workers, MuScriptor 27.4 + 7.9), null 6.5 min, one-song
  smoke test 3.4 min, batching check 0.7 min: about 62 min. Reruns with the caches: 0 GPU, ~5 min CPU, and
  metrics.csv is byte-identical.
- **Tests:** new `tests/test_eval_{stem_taxonomy,attribution,distractors,backing}.py`; full default suite 1276 passed,
  1 skipped (8 min 12 s).
- **Auxiliary `backing` suite** (`bandscribe eval run --suite backing`, pairs in `data/eval/backing_pairs.toml`): KH
  10.3 % of guitar notes also on the current backing (11.9 % against every backing result), aotonat 5.4 % (17.7 %),
  SC 0 %; AIZO/AZ have no `notes` stage yet. The "every result" numbers reproduce the scratch-script values below.

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
- **Conclusion (for these two songs):** changing the guitar-pass prompt reshuffled MuScriptor's output far more than
  it removed bleed. Not a general rule: on JetB the keys mask did remove 191 piano-bleed notes (E3c, F1 +14 points);
  the effect varies by song. Measured as notes kept from 10-03:
  - KH kept 1,050 of 2,916 presumably-real notes and 239 of 627 bleed notes.
  - aotonat kept 1,372 of 3,029 and 476 of 620.
  - So keys in the mask is a trade-off with no ground truth yet: E3b, or a GP file for these songs, must decide it.
    Removing bleed after transcription (S7) is the better lever than widening the prompt.
- **Known limits** (documented in DESIGN S3.5 and in `instrumentation.py`):
  - A part that plays only outside the windows is missed (the safe failure for the mask).
  - Brass/winds can never be "present" from stems alone.
  - S7's bleed penalty (M6) needs the full-length piano+other pass (eval or on-demand).

## Priority for the next session

- **Guitar mask:** the distractor run found 0 notes relabelled non-guitar and −2.1 F1 points on average when the
  production mask added keys/synth classes (13 conditions), but E3c (held-out windows) rejected `guitar_only`
  (Δ −2.35 [−8.22, +0.47]; JetB: the keys mask removed 191 piano-bleed notes, +14 points). The default stays
  `guitar+present`; the policy question goes to E3b with more songs (its registration already notes the 13 Tier B
  songs and, since 2026-10-05, merged references).
- **Decoder stability:** an inaudible change moved up to 146 FP/min on one excerpt. Before trusting small deltas,
  measure overlapping-window re-decoding / voting or beam search (E21), or repeat the null on more excerpts.
- **S7 note clean-up** is the largest lever by error count (timing/splitting and octave), present even with no
  distractor at all.
- Earlier item: decide keys in the guitar mask with evidence (E3b, ask before GPU > 30 min) or get a GP file for KH/aotonat. See the KH trade-off above.
- Remaining speed options (not done): `amt_gtr` (141–197 s) is now the largest stage; run the CPU Basic Pitch pass in parallel with GPU stages; upstream loader (1.46× vs 1.28× realtime) when VRAM allows; the bass pass is still full length (49–94 s).
- **Process:** M1a+M2 took ~2 days, mostly the serial GPU experiment queue (~7 h of GPU time). Next time:
  - keep experiments small (cap tracks, stop at the first clear answer);
  - run review/fix/accept without re-running experiments;
  - ask the user before any GPU job longer than ~30 min.

## Next steps (in order)

1. Quick check from the project folder: `bandscribe.cmd doctor` and
   `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.
2. ~~Speed work~~ done 2026-10-04 (profiles + presence windows, reviewed and fixed, above). Optional: time `--profile fast` on SC.mp3.
3. Review (correctness + GPU/Windows): done; fixes applied 2026-10-05 ("Review fixes" above, not committed). Next:
   check acceptance against DESIGN §10 M1a/M2 (evidence list in M1_M2_SPEC §11). Tier A GP ground truth is BLOCKED on
   the user. The first `bandscribe run` per existing job recomputes `beats` (new key: beat-this version) and its CPU
   descendants once.
4. ~~Commit the milestone on `main` and tag it (`m2`).~~ Done 2026-10-05 (tags `m1a`, `m2`; M2 with known gaps, see the top of this file).
5. ~~M3~~ done 2026-10-05 (tag `m3`). ~~M4a~~ done 2026-10-05 (tag `m4a`, see "M4a" at the top; the NVML pre-check
   item 6 stays open: the editor never starts GPU runs). Next: **M5** (bass: octave anchors, slides — the user
   confirmed SC bars 4–6 are one slide we write as separate notes), the grid's tempo level (~~spurious beats~~:
   doubles joined 2026-10-05, see "Grid: beats tracked twice"; left: aotonat is 185 BPM but gridded at 93, AIZO
   29–44 s and AZ tracked at double time), or first the M3 gaps (fretting toward 75 %: same fingering
   for repeated sections, chord-shape templates; E16 if a detuned song shows up). Original M3 item: **First task of M3 (done
   2026-10-05: `tick_prf` and `evaluate_item` now match over the whole song and give a section the pairs whose GT
   note is inside, the unmatched GT notes inside and the unmatched estimates starting inside; partition counts add
   up to the song's, tests in `test_eval_metrics.py`):** note
   scoring drops matches at section boundaries (`eval/metrics.py` `tick_prf` ~201-206 filters estimates by the GT bar
   of their onset, `eval/suites.py` `evaluate_item` ~578-585 by onset inside the section windows): a note played up to
   50 ms before a section's first bar line is dropped from the estimate while its GT note stays (an FN), and a GT note
   just outside a section whose estimate lands inside becomes an FP; the same at the section's end. Deferred from
   the 2026-10-05 review because M2 has no Tier A section
   scoring yet (no GP ground truth) and tick F1 becomes the primary metric in M3, where the fix (match within a
   tolerance band around the section, then keep pairs whose GT note is inside) belongs with its tests.
6. **M4a: NVML budget pre-check outside the GPU lock** (`vram.py` `run_gpu_stage` ~408-424 checks NVML free, then
   `gpu.py` ~224-231 takes the lock): two concurrent bandscribe processes can both pass the pre-check and then run one
   after the other on a budget measured before the first one loaded. Deferred from the 2026-10-05 review: it only
   matters with two bandscribe processes at once, i.e. once the UI (M4a) can start runs next to the CLI; one CLI run
   is strictly sequential. Note (acceptance 2026-10-05): the worker's own `mem_get_info` is **not** a safeguard on this
   PC: under WDDM it reported 5,098 MB free whatever other processes held (0.8-5.0 GB). What actually catches a real
   shortage is the rung ladder with the shared-usage / load-spill checks (review fix 10).

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
  - 2026-10-05: the `beats` key now includes the beat-this version, so every job recomputes `beats` (GPU, seconds)
    and its CPU descendants (`grid` … `notes`, except `sep`, `stems`, `amt_bp`) once on its next run; MuScriptor
    views come from the transcription cache when the grid is unchanged. SC was re-run in the review check (below).
    Until a job is re-run, `bandscribe eval run --suite tierA` skips it (its planned `notes` dir does not exist
    yet) unless `--arg allow_stale=true`.
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
