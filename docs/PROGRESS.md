# bandscribe progress / handoff

**Mid-term review and plan (2026-10-24): `docs/MIDTERM_REVIEW.md`** - every area re-measured (CPU) and 86 limitations
checked by two skeptic lenses each (evidence and impact; none fully refuted, many narrowed). Biggest verified limits: only about 25-30 % of
4-bar part phrases are practisable with light edits (6 % by the strict test); rhythm strums match the sheet exactly 10 % of the time;
nothing measured what the player reads (string agreement 0.583, tuning+capo 13/18); no truth for the owner's J-pop songs; several silent
failures. The review also found two bugs in the E39/E40 wiring (the editor skipped the clean-up; the clean-up shifted quant's note index),
fixed in `2276733`. The plan: phase 0 backups and checks, phase 1 visible warnings / tempo question / library / sheet evaluation in git,
phase 2 section-first fixing tools, phase 3 E45/E44/E46 and the first J-pop measurement. Evidence: `data/tmp/midterm/`.
**Phase 0 done (2026-10-24):** backup of 1,825 files (2.13 GB, sha256 verified) in `C:\Users\user\bandscribe_backup\2026-10-24\`; the owner's
answers recorded (check lulu. / 狭心症 / 怪獣の花唄 first; the target is practising in bandscribe's own viewer, not Guitar Pro; E48
MuScriptor-large approved for phase 3); the 13 jobs re-run after `2276733` and checked (notes unchanged, repeat links 100 %, no-edit replay
= the run up to note order within a tick). Next: phase 1, viewer-first (see the appendix of `docs/MIDTERM_REVIEW.md`).
**Phase 1 so far (2026-10-24, local commits):** song health in `export/health.json`, printed after `run` and shown as a
banner in the viewer (1-4, 1-5, 1-8, 1-9; 0 input / burst / unplaced false alarms on the 13 valid jobs, all 3 guitar-removed
jobs caught); `run` asks the tempo after beats + grid, before the first GPU stage of a new song (1-6; Enter keeps, h / d /
a BPM; 60 s or no terminal = unconfirmed), and keeps `--tempo` / `--parts` / `--title` / `--instruments` / tuning in the
job's `hints.toml`; on the 8 local songs with a known tempo the automatic level is x2 off on 4, the right level is
always a candidate, the hint always lands on it (`tests/test_grid_known_tempo.py`). One viewer server for every song
(`bandscribe view` = the song list, `/s/<job>/`, `view.cmd`), a summary card at the end of `run`, per-stage time
estimates, `status --songs`, `refresh` (CPU stages only) (1-7). Repository safety (1-1): copyright guard test,
`-m smoke` (143 tests, 19 s), `.githooks/pre-push`, `--run-gpu` / `--run-network`, `bandscribe backup` + a doctor
warning (first run to `C:\Users\user\bandscribe_backup\mirror`: 1,851 files, 2.13 GB, 31 s; a rerun copies 0 in 1 s).
The metrics-and-split rules block heads `docs/decisions.md` (1-3; the split itself is due 11/15). **`bandscribe eval
sheets` (1-2)** puts the sheet-song measures in tracked code (`bandscribe/eval/sheets.py`; the extracted sheet JSON and
`data/eval/sheets/songs.yaml` stay local) and reproduces every number of the review exactly in about 3 s: merged 66.8,
parts 55.9 (song mean 55.4), bass 81.5, drifting 68.9 / 85.6, rhythm chords exact / pitch-class set / shape 10.1 /
22.8 / 36.2 %, lead 3+ notes 12.0 %, string / fret 0.583 / 0.397, downbeats 82.4 %, practisable phrases 26.6 %
(strict 6.2 %); a snapshot test (in `-m smoke`) fails on a move over 0.5 point. The test seal (1-3): a song's `split` in songs.yaml; test songs only through
`open_test(<registered experiment>, <view>)`, logged in `data/eval/test_access.log`, 4 distinct views. Later the same
day: the failure catalogue replay as a test (`tests/test_health_catalog.py`: 8 / 8 guitar-removed caught, 0 false alarms
on the 13 valid songs, lulu.'s bass burst window 150-270 s); part names write the guitar type only when 70 % of the
part's notes agree (parts code 5; SNN, HBT, 逆光 and others lost a type label that rested on 37-59 %); `weak_split`
health note (smallest part under 20 %: 怪獣の花唄, 狭心症, AIZO); GP5 titles no longer turn into '???' (cp949 kept,
`[hints] title_ascii`; score code 11); `bandscribe library` (13 songs in `data\library\<title>\`, 0 '?' titles).
**Phase 1 is done except the split, due 11/15** (1-3). Deferred, with reason: the Korean warning line inside the
alphaTex / GP header (1-4) - the owner practises in the viewer, which shows the banner, and computing the health
inside the score stage would tie it to the score cache. Waiting for the owner: the check of lulu. / 狭心症 / 怪獣の花唄.
Before it, **E41 rejected (2026-10-23)** - the piano+other transcription vote also removes more real guitar notes than
bleed (arrangements double the guitar); keys / strings bleed needs a better separation, not a note rule. Before it,
**E39 and E40 adopted (2026-10-21)** - note clean-up at the quant stage's input: same-pitch duplicates
within a part (`[tab] dedupe_s = 0.03`) and notes shorter than 50 ms, guitar and bass (`[tab] min_dur_s = 0.05`).
Unplayable notes in 怪獣の花唄's Gtr 1 fell from 2,080 to 88 and lulu.'s bass from 12,027 notes to 1,738. See "Note
clean-up (E39 / E40)" below. Before it, **E38 rejected** - a stem-energy bleed rule (piano+other stem louder than the guitar stem in a
note's band) removes more real guitar notes than fake ones at every threshold, because the separation puts much of
the guitar into the "other" stem. Before it **E37 (2026-10-19)**: strings and "other synth" hurt the guitar tab at
J-pop levels and the transcriber never labels such notes non-guitar. See "E37 / E38" just below.

Last updated: 2026-10-17 (**M7c done**: Role and Player views (exported to `tab/views/` and switchable in the
viewer, with N / idle / unison settings), part count switch and section reassignment in the viewer (a switch takes
about 1 s - M7a's k7 met through the viewer), see "M7c" below and `docs/acceptance/M7c.md`. Before it the
**same-position part count (E28)**: a "width" evidence - a centred, coherent part
next to a wide double in R2 windows - counts and splits DESIGN's scenario A; held (off by default: one of three test
songs did not change, so the CI lower bound is 0), opt-in per song with `[parts] use_width = true`; see "Same-position
part count" below. Before it: **M7b done with a known gap**: an identity tracker (off by default: on whole-song
Tier B remixes its dev choice changed no note - M7a's errors are single notes, not swapped windows), repeat links and a `part`
edit that spreads from verse 1 to verse 2, whole-song evaluation data - see "M7b" below and
`docs/acceptance/M7b.md`; before it the `--parts` switch 26.5 -> 4.0 s (a `tuning` stage). Earlier the same day:
**M4b done**: the viewer as a practice tool - the recordings and a tab synth per score
track played in step (A/B, mixer, 50-100 % slowdown at the same pitch, loop, count-in), a CQT piano roll, the part
band and a review queue - see "M4b" below and `docs/acceptance/M4b.md`; 怪獣の花唄's forced one part removed.
2026-10-11: **M7a done with known gaps**: the Part view - the guitar part count per song, each note's
part, one fretted tab per part next to the merged reference staff - see "M7a" below and `docs/acceptance/M7a.md`;
before it the M6 follow-up (E6b, E7, CenterWide models). 2026-10-10: **M6 done with known gaps**: the stereo router and
the first automatic part split - two guitars at different stereo positions become two fretted parts - see "M6"
below and `docs/acceptance/M6.md`; before
it the **tempo hint**, see "Grid: tempo hint". 2026-10-08: **M5 done with known gaps**: bass F0 trackers, the bass clean-up and E13, see "M5" below and
`docs/acceptance/M5.md`. 2026-10-05: **grid fix after M4a**: beats that Beat This! tracked twice are joined, see "Grid: beats
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
| M5 bass | ✅ `bass_f0` (CREPE + PESTO worker, pyin) and `bass` stages, E13 (only the merges adopted: one note + slide for a glide written as a staircase), techniques; acceptance `docs/acceptance/M5.md`: **4 pass / 1 fail / 3 blocked** (contamination) (tag `m5`) |
| M6 router + part split | ✅ `router` and `split` stages (CPU), E6 (router thresholds, GO), E22 (held) / E22b (P split adopted, A held), parts `Gtr 1 (L)` / `Gtr 2 (R)` fretted; acceptance `docs/acceptance/M6.md`: **7 pass / 2 fail / 1 blocked** after the 2026-10-11 follow-up (E6b, E7; tag `m6` = 5 / 3 / 2) |
| M4b A/B mixer + review tools | ✅ WebAudio engine in `bandscribe view` (recordings + per-track tab synth via alphaTab audio export, A/B hold key, mixer, sound position, 50-100 % slowdown with an own pitch shifter, loop, count-in), CQT piano roll with unkept candidates, part band, review queue; E26 (edit anchors after upstream changes) 0.94; acceptance `docs/acceptance/M4b.md`: **3 pass** |
| M10 calibration (interim) | ⏸ the measurable parts fail: note confidence (E35, AUC 0.56, queue capture 0.234 ≈ random) and part confidence (E36, ECE 0.108, AUC 0.48); edit learning and edit time need the user's edits; acceptance `docs/acceptance/M10.md`: **0 pass / 2 fail / 2 blocked / 2 n/a** |
| M9 chords, GP7, techniques | ✅ chord chart `export/chords/` (E31: 54 % on seisyun complex, under 70 %: chart only, names in the score by `tab.chords_in_score`), `tab/score.gp` (Gp7Exporter, backing MP3 + a sync point per beat; written notes vs detected onsets 9-17 ms median, 44-57 ms max with quantisation), bend rule (E32: 0 of 16 bends found - all inside chords; not wired); acceptance `docs/acceptance/M9.md`: **1 pass / 2 fail / 1 n/a** |
| M8 narrow mixes + S7 | ✅ closed without the GPU items (user, 2026-10-18): **bass bleed on** (E30b: 0.975 precision after the review found ghost notes in E30's truth; `[amt] bass_bleed`, notes -> `bleed.json`, viewer candidates); E10 rejected by its ceiling, E29 repeat consensus rejected (+0.11 with the review's duplicate fix), vocal bleed held; acceptance `docs/acceptance/M8.md`: **0 pass / 2 fail / 1 held / 1 blocked / 2 n/a** |
| M7c Role / Player views + part UI | ✅ `parts/views.py` (section roles, Role view, Player view: per-section DP, no two simultaneous parts per player, `Gtr extra (overdub)`, idle / unison policies, markers), `tab/views/` export + `score.json` roles, viewer: view switch, part count switch (`parts_k`, ~1 s), Ctrl+Shift+1-4/0 to the end of the section (`part_span`); acceptance `docs/acceptance/M7c.md`: **4 pass** |
| M7b identity tracking | ✅ `parts/track.py` (window layers, links, Viterbi over relabellings, repeat-map window pairs; off by default), repeat links `rep` + `part` edit (Shift+1-4/0, spreads to repeats), whole-song Tier B data (`parts/eval_track.py`, 24 remixes), E27 rejected (no note changed); acceptance `docs/acceptance/M7b.md`: **3 pass / 1 fail / 1 n/a** |
| M7a Part view v0 | ✅ `notesig` and `parts` stages (replace `split`), part count (spatial streams in 2+ windows), per-note part posteriors, `gtr_1`..`gtr_4` fretted with per-part tunings, `All guitars (reference)` / `Unassigned (review)` staffs, `--parts auto\|1-4`; E25 (settings chosen on dev); acceptance `docs/acceptance/M7a.md`: **3 pass / 4 fail** |

All work is on **`main`**; milestones are tags (`m0`, `m1a`, `m2`, `m3`, `m4a`, `m5`, `m6`, `m7a`, `m4b`, `m7b`).

## Note clean-up (E39 / E40, 2026-10-21): duplicates within a part, tiny-note bursts

Code: `bandscribe/tab/cleanup.py` (`dedupe_within_line`, `drop_short`), applied in `tab/stage.py` `quant_tracks`
(guitar and bass) when `[tab] dedupe_s` / `min_dur_s` are on (both on by default now; `quant_params` adds a
`cleanup` key only when on, so the quant key changes once). Evaluation `data/tmp/m10/e39_e40.py` (CPU), results in
`docs/decisions.md` E39 / E40.

- **Diagnosis first (dev items only, `data/tmp/m10/cleanup_diag*.py`):** on the whole-song Tier B remixes, notes
  under 50 ms were 100 % false but sat almost all in moosmusic (53-83 % of its notes: a MuScriptor degeneration
  like lulu.'s bass, not leakage as M8 thought); same-pitch restarts within 20 ms were 93 % false - one guitar note
  emitted under two classes (acoustic + clean electric). Octave errors had no simple symbolic tell.
- **E39 (dedupe within a part, 30 ms):** Tier B per-part F1 +1.90 [+0.14, +4.68], 7,270 notes removed costing 2
  part matches; sheet songs per-part bar F1 mean +0.29 (HBT +1.38), none worse than −0.04 → adopted.
- **E40 (drop < 50 ms, guitar and bass):** Tier B merged F1 +3.46 [−0.03, +10.38] (moosmusic 12.7 → 51.5), 5,918
  removed costing 17 matches; sheet songs within ±0.1 → adopted.
- **Effect on the J-pop jobs (rerun 2026-10-21, CPU, about a minute each):** unplayable notes dropped from the tab
  怪獣の花唄 2,080 / 46 → 88 / 3, 逆光 251 / 143 → 111 / 42, 青と夏 97 / 29 → 18 / 7, lulu. 19 / 60 → 4 / 29,
  狭心症 unchanged (131 / 5); lulu.'s bass 12,027 → 1,738 notes. The six sheet songs, AIZO and AZ were rerun on
  2026-10-23: merged guitar bar F1 against the sheets SC 73.9 → 74.0, GKA 60.6, ANB 64.1 → 64.2, SNN 71.5 → 71.7,
  HBT 68.6 → 71.5, HNN 59.1 → 59.2; bass unchanged (`data/tmp/m9/tiera_table.json`, before:
  `data/tmp/m10/tiera_table_before_cleanup.json`).
- **Octave errors, explored 2026-10-23 (not registered; dev items only, `data/tmp/m10/octave_diag.py`):** on the
  separated guitar stem, a sub-octave ghost's odd harmonics (q, q+19, q+28 against q+12, q+24, q+31) look like a
  real low note's (median −0.1 vs −0.8 dB; best precision 0.20 against a 0.23 base rate), and a lone wrong-octave
  note shows no tell either (precision 0.02-0.05). With E35's Basic Pitch agreement also uninformative, octave
  errors are not fixable by a note rule on these signals; they would need a better transcriber or a pitch tracker on
  a cleaner single-guitar view. Counts on dev: upper note of an octave pair false 2,105 (of 6,244), lower 1,376 (of
  5,908), lone wrong octave 599.
- Still open from the M10 diagnosis: splits the sheets write as one note.

## E37 / E38 (2026-10-18/21): keys, synths, strings and brass at J-pop levels

Why: the user's newer songs (Mrs. GREEN APPLE, Vaundy, King Gnu) carry keys / strings / synths about as loud as the
guitar (separated piano+other vs guitar −7 to +1 dB, `data/tmp/m10/arrangement.py`), while the six sheet songs
(Kessoku Band) have almost none (−24 to −79 dB). The 2026-10-04 distractor run used each multitrack's own balance:
piano −30 to −10 dB, strings −32 to −16 dB, no brass at all (`data/tmp/m10/distr_levels.py`). Licensed band scores
for those songs exist (Yamaha ぷりんと楽譜 / Fairy, with TAB and keyboard parts) but **are not sold outside Japan**
(site FAQ), so the measurement is done on multitracks instead.

- **Registration:** `docs/decisions.md` E37 (committed 47af085 before any GPU work). Levels −6 / 0 / +6 dB against
  the guitar sum per category; arms production / `strings` / `strings_winds` (the last two on base and the 0 dB
  strings / brass conditions only); "discrimination" = share of distractor-dominated output notes labelled
  non-guitar; decoder null. Rule: a category with ΔF1 ≤ −3 points at 0 dB (CI < 0, ≥ 2 songs) gets an E38 fix.
- **Code:** `bandscribe/eval/distractors.py` (`levels`, `level_cats`, `arms`, `guitar_only_arm`, `level_gains`,
  `level_conditions`, `extra_mask`, `discrimination`), report tables in `distractor_report.py`, runner
  `run_e37` / `e37_flags` in `bandscribe/amt/experiments.py`. `stem_taxonomy`: "Sample(s)" no longer forces drums
  (SampleBrass, TrumpetSamples, ViolinSamples) - no existing song changed.
- **Data:** 7 Cambridge-MT songs added (optional registry parts, 3.33 GB, user-approved, picked by the fixed rule in
  `data/tmp/m10/ms21_select.py` from the MS21 annotation CSV): Ben Carrigan ×2, Dunning Kruger, E J Rios, Flags,
  Tim Taler, Pablo Martin. Their `lines.yaml` are unreviewed drafts (fine for E37: it categorises stems by name).
- **Plan** (`data/tmp/m10/e37_plan.py`, before GPU): 17 windows, 122 conditions; windows per category: piano 7,
  electric piano 2, organ 3, synth pad 3, synth lead 3, other synth 5, strings 7, brass/winds 5.
- **Run:** 9 GPU chunks (about 3 h 6 min over two evenings, `data/tmp/m10/e37_chunks.log`; paused by the user after
  chunk 5 and resumed with `data/tmp/m10/e37_resume.sh`), then `bandscribe eval exp E37` from the caches:
  `data/runs/20261019-211741_fb7a7c2_exp-e37`. Table: `data/tmp/m10/e37_table.py`.
- **Results** (production arm, base F1 78.9; ΔF1 points at 0 dB against the guitar, song-block 95 % CI):
  - **strings −3.2 [−6.0, −1.1]** (6 songs; −0.6 / −3.2 / −6.6 at −6 / 0 / +6 dB; FP and FN both rise; null 0) and
    **other synth −14.1 [−24.4, −0.8]** (5 songs; Pablo Martin −32, Flags −11 are real, James Elder −16 sits on an
    unstable decoder) meet the registration's rule → E38 targets.
  - piano −4.3 [−12.9, +1.1]: not met, but three songs lose 11–33 points (Hey Carrie Anne, JetB, Tim Taler).
  - brass/winds −2.4 [−7.1, +0.1], organ +3.0, synth pad +1.9 (−8.7 at +6 dB), synth lead −0.2. Electric piano
    +12.2 is Mu's decoder instability (its null moves FP by −242/min), not a gain.
  - **Discrimination 0 % for every category:** MuScriptor never labelled a distractor-dominated note non-guitar in
    the production arm (piano 183, other synth 359, strings 119 notes kept as guitar). Only the separation stands
    between those sounds and the tab.
  - **A wider mask is not the fix:** `strings_winds` looks better only against its own base (+6.1); its absolute F1
    is 58.9 at base and 62.4 with strings (production 78.9 / 75.0) - it relabels real guitar notes as brass (296
    in the strings condition). `strings` is lower than production too. The registration's arm rule compared each
    arm with its own base; that flaw is written down in the record.
- **E38 (2026-10-21), rejected:** `amt.bleed.stem_ratio_db` / `stem_flags` (not wired) scored as `stem<T>` arms on
  E37's windows (CPU, `data/runs/20261021-193533_e5ba017_exp-e38`). Pooled F1 dev: production 74.3 vs 65.5-71.1 for
  T = −3..+10 dB; test 77.0 vs at best 76.0. At 0 dB, strings: 1,524 real notes lost vs 410 fakes removed; piano
  1,086 vs 433; even base 475 vs 253. The SW separation leaves much of the guitar in piano+other, so "other is louder
  in this band" is not evidence of bleed. Sheet songs unaffected (≤ 0.2 % removed); on the J-pop originals T = 0
  would have removed 38 % (青と夏), 27 % (lulu.), 19 % (狭心症) of the guitar notes - mostly real ones by the Tier B result.
- **E41 (2026-10-23), rejected:** the second opinion that is not energy - the piano+other stem transcribed over the
  whole window with the keys+guitar mask, a guitar note dropped when a non-guitar note of the same pitch starts
  within 30/50 ms (`amt.bleed.vote_flags`, not wired; `data/runs/20261023-205003_3b771bd_exp-e41`, GPU 27 min + 17
  min guard). Dev: production 74.3 vs 72.3-73.4; test 77.0 vs at best 75.2. At 0 dB, piano: 523 real notes lost vs
  141 fakes removed; strings 188 vs 37. Arrangements double the guitar: keys and strings play the same pitches at
  the same time, so a coincidence is not evidence of bleed.
- **Conclusion for keys / strings / synths:** two note-level rules (energy, transcription vote) failed for the same
  reason. What is left is upstream: a separation that keeps those sounds out of the guitar stem (another model,
  judged on E37's windows - the distractor machinery compares separators), or a transcriber that tells timbres
  apart. Both need downloads and their own registration.
- **Separator survey (2026-10-23):** the MVSEP guitar leaderboard still has BS-RoFormer SW (the one we use) first at
  9.01 dB SDR; the next entries (Logic Pro 11.2 9.00, unreleased test checkpoints 8.93-9.00, MLSLABS WCJ 8.89,
  xlance `bs_gtr` 8.60 - SW-derived) are within 0.1-0.4 dB, mostly not public. No public separator would keep keys /
  strings out of the guitar stem noticeably better, so this line stops here; see `docs/acceptance/M10.md`.

Also from this session (2026-10-18):

- **New originals processed** (all guitar stems −18 to −21 dB, so originals): 青と夏 f-250b3e07683d2d1a (185.5 BPM),
  怪獣の花唄 f-07f21d0ec2cb1bb4 (150), 逆光 f-abd70434e91c65ba (144.8, Eb standard - matches the band score's
  "half step down"), lulu. f-12befe96300880e4 (user: 88 BPM; the tracker had 175), 狭心症 f-5959ab5fa06d86ad (user:
  68 BPM; the tracker had 136). Tempo hints and real titles are in each job's `hints.toml`.
- **lulu.'s bass is broken after 2:30:** MuScriptor emitted 10,965 notes of 10 ms at MIDI 59-85 on the bass stem
  (83 % of its bass notes) while the F0 trackers see a normal bass near C2. Nothing in the pipeline catches a burst
  of tiny notes; a guard would be its own experiment.
- **Test suite:** the default run includes `network` and `gpu` marked tests (a YouTube clip download, the CUDA self
  test). Run `-m "not network and not gpu"` while GPU work is running or without network consent.

## M10 interim (2026-10-18): confidence and the review queue

Code (experimental, not wired): `bandscribe/tab/confidence.py` (note features, a standardised logistic model, ECE,
queue capture), `bandscribe/parts/confidence.py` (part features); runners `data/tmp/m10/e35.py`, `e36.py`.
Report: `docs/acceptance/M10.md`.

- **E35 note confidence** on the six Tier A songs (leave one song out): AUC 0.557, capture of wrong notes in the
  lowest 20 % 0.234 (random 0.20, the queue's part sort 0.235) → rejected. ECE 0.045 only because every note gets
  about the error rate.
- **Why:** of the 39 % "wrong" notes against the sheets, 16 points are a pitch the bar has but more times than the
  sheet (a split note, or strums the sheet writes fewer of), 13 points the right pitch class in another octave, 9
  points another pitch class. Octave/duplicate choice is the transcription's
  and the note clean-up's job, not a confidence model's.
- **E36 part confidence** on the whole-song Tier B remixes: the posterior is overconfident (mean 0.998 vs 0.838
  right, ECE 0.161); a spatial-feature model gets ECE 0.108 but AUC 0.48 across songs → rejected.
- **Blocked on the user:** silver data and edit-time comparisons need real edits in the viewer.

## M9 (2026-10-18): chord chart, Guitar Pro 7/8 with the backing, bends

Code: `bandscribe/tab/chords.py` (per-beat pitch-class templates + bass bonus + Viterbi change cost, bass cross-check,
text chart), `tab/stage.py` (`write_chords`, `write_gp7`, `gp7_sync_points`, `gp7_drift`, `backing_audio`; score code
8), `node/bandscribe_score.mjs gp7`, `bandscribe/tab/techniques.py` (bend rule, not wired). Config `[tab]
chords_in_score = false`, `gp7_backing = "original"`. Report: `docs/acceptance/M9.md`.

- **Ground truth from the user's PDFs:** `data/tmp/sheets_sc/pdf_marks.py` reads every word above the staves with
  its bar and position (pdfium's own reading order keeps "Bbmaj7" whole; split names are joined in the scorer):
  189 chord names (Ele. Guitar 1), 16 distinct bends, 24 slides, 50 H/P, 10 P.M. (Ele. Guitar 2).
- **E31 chords:** 0.546 root + class (root 0.639; 0.543 before the review fixed slash chords in the scorer) - under DESIGN's 0.70. The same recogniser on the PDF's own notes
  gives 0.601: the recogniser (guitar voicings without the root, tension chords) is the main limit, not the
  transcription. The chart is written anyway with a header saying how far to trust it.
- **GP7:** alphaTab 1.8.4 writes `BackingTrack` + `SyncPoint`s; the export embeds the recording as MP3 (libsndfile;
  cached in the job's `cache/gp7`) and replaces the alphaTex's per-bar sync points by one per grid beat. The beats
  are the sync points themselves (0 ms only proves the round trip); the written notes against their detected onsets
  are 9-17 ms median, 29-38 ms p95, 44-57 ms max (quantisation included; per bar the beats were up to 49 ms off).
  Not opened in Guitar Pro 8 (not installed); a player ignoring the MP3's encoder-delay header would add ~25 ms.
  The file holds the recording: personal use. The edits' re-export now moves every old file aside before
  installing (a .gp held open by a player refuses and the old export stays whole).
- **E32 bends:** none found - every marked bend sits inside two strumming guitars, where pyin on either stem channel
  is unvoiced (8 % / 1 % voiced over the song). Next: partial tracking around each note's pitch, judged against the f0
  of the clean Tier B lead takes.
- **Two more songs with the user's sheets (2026-10-18):** 「ギターと孤独と蒼い惑星」 (job f-56f32f7fc6873405) and
  「あのバンド」 (f-cb4a96577087aaf4), audio from the user, sheets extracted to `data/tmp/sheets_gka`, `sheets_anb`
  (`data/tmp/m9/songs.py`: sheet offset from note bags, E31's chord truth for any song).
  - The first song's grid was at half tempo (97 vs the sheet's 193): `hints.toml` `[grid] tempo_bpm = 193`.
  - E33 (chord recogniser v1 = bass in the histogram, chosen on seisyun complex 0.607): the new songs give v0 0.359,
    v1 0.343 → rejected, v0 kept. The chart header now says it is for reference (55 / 28 / 45 % on the three songs).
  - E32b: the bend rule finds 1 of 99 bends on the new songs (the lead shares each stem channel with a distorted
    rhythm, pyin sees no single line). With 115 marked bends on three songs, partial tracking now has dev and test
    data.
- **Six Tier A songs with sheets (2026-10-18):** three more (星座になれたら, ひとりぼっち東京, 光の中へ) run with their
  sheet tempos (`run --tempo`, kept in `hints.toml`); all five downloaded songs moved to `D:\backing` under their
  Japanese titles (same names as the sheet folders). Table in `docs/acceptance/M9.md`: guitar bar F1 59-74, bass
  71-91, chords 0.28-0.55.
  - Sheets carry tunings: a capo (光の中へ guitars 1 and 3), a half-step-down bass (星座になれたら, 光の中へ), a
    drop-D bass (あのバンド). The scorer now reads the PDF header; before, those songs looked misaligned or wrong.
    Capo sheets name the chord shapes.
  - Bends by partial tracking (explored on the three dev songs only): recall 0.28-0.44 but precision about 0.1, F1
    0.16 against DESIGN's 0.6 - the detections sit at chord onsets like the false ones. Not run on the test songs,
    not registered; the detector is kept as a research record (`docs/research/experiments/bends_cqt.py`).

## M8 (2026-10-17/18): narrow mixes, repeat consensus, bleed

Code: `bandscribe/amt/consensus.py` (repeat consensus, off), `bandscribe/amt/bleed.py` (vocal / bass bleed flags,
vocal f0 by pyin), `amt/stage.py` `bass_bleed` (notes stage, on), runners `bandscribe/parts/eval_e29.py`,
`eval_e30.py` (E30, E30b), `docs/research/experiments/e10_ceiling.py`. Report: `docs/acceptance/M8.md`.

- **E10 (repetition-subtraction lead view, R0 path) - rejected without a GPU run.**
  - Only 4 of the 9 Tier B songs have a single-note lead take next to a chord take.
  - A generous oracle lifts the lead Line F1 by +0.004 (+0.037 with the review's most generous definition), under
    +0.05: real leads repeat with the rhythm.
- **E29 (repeat consensus) - rejected.**
  - Test −0.02 points. The review found additions doubling notes already within 50 ms (fixed: `dup_s`). Without them
    it is +0.11. An oracle adding every missed note held by two repeats reaches +0.49.
  - On Tier A no bar voted: the repeat map's occurrences are out of phase (seisyun complex S5 fits 4 bars later).
    Realigning or pairing bars by guitar notes did not help either.
- **E30 / E30b (bleed flags) - bass on, vocal held.**
  - The bass rule (a note under E2 on the bass f0's pitch class) scored 0.70 in E30. Its truth (MuScriptor on the
    solo takes) had the same octave-below ghosts (A1 on a guitar).
  - E30b drops a take's low note when it has an octave partner and a fundamental 20 dB weaker. With that it scores
    0.975 → `[amt] bass_bleed = true`: the notes stage (now after `bass_f0`) moves the flagged notes to
    `notes/bleed.json`, and the viewer lists them as "베이스 블리드 의심" candidates.
  - Tier A: AZ 4, KH 5, the others 0 (corrected 2026-10-18: the two job ids were swapped at first).
  - The vocal rule stays off: 18 test notes, 0.56.
- **Explored, not registered:** octave ghosts looked 97 % false on dev, but 3,869 of the 4,026 came from moosmusic.
  Read precision per song.
- **Not built (user's call, 2026-10-18):** safety net and keys bleed (each needs another GPU transcription per song);
  cloud voters (accounts, upload consent) and E17 (downloads) stay optional.

## M7c (2026-10-17): Role and Player views, part UI

Code: `bandscribe/parts/views.py` (`section_roles`, `role_view`, `player_view`, `violations`), `tab/stage.py`
(`view_layout`, `build_score(layout=)`, `write_views`, score code 6, `score_params["views"]`), `[views]` config
(players 2, idle_policy rest, unison_policy duplicate), beat text in alphaTex / GP5, `tab/edits.py` (`parts_k`,
`part_span`, `Session.view`), viewer `/api/view` and the header controls. Tests: `tests/test_parts_views.py`,
`tests/test_parts_m7c.py`.

- **Views**: Role (per section: Rhythm, Rhythm 2, ..., Lead, Lead 2, ...) and Player (N guitarists; per-section
  dynamic programming: parts sounding in one bar never share a player, moving a part / role group / register /
  side cost, the least important part to `Gtr extra (overdub)`; `rest` or `double`; unison `duplicate` or
  `primary_only`; markers as beat text). Exported next to the Part view (`tab/views/`), and the edits re-write them.
  Every part note is in a view (a section's few notes and a pickup before the first section included).
- **Checks**: 0 violations on Tier A (5 songs x N 1-3 x rest/double) and the 8 two-part whole-song Tier B remixes;
  seisyun complex section roles vs the same rule on the user's GP guitars 22/26 (84.6 %).
- **Viewer**: 보기 (part / role / player, with 인원 / 쉴 때 / 유니즌), 파트 수 (auto / 1-4: an edit that runs the parts
  stage again from the run's own inputs and re-frets only the parts - one switch 0.89-1.06 s on seisyun complex),
  Ctrl+Shift+1-4/0 = this part from the selected note to the end of its section. The tab synth now renders only while
  it is heard (A/B or "with the recording"): alphaTab drew the next score only after a running audio export, which
  made switches take 20 s.
- **Pitfall found**: `bandscribe run <job> --until <stage before score>` publishes an export without the score
  (export = what the last run reached). Run without `--until` to get it back.

## Same-position part count (2026-10-17, after M7b): width evidence, E28

Code: `reid.width_split` and the `width` kind (R2 windows: notes with `ms_db >= width_ms_hi` and `coh >=
width_coh_hi` are centred, `ms_db <= width_ms_lo` wide; two streams with `width_n_min` of each), `arbitration`
(Lines seeded from centred / wide notes, a width term 6 x tanh((ms - 4 dB) / 3 dB) x the Line's mean width - only
when the width evidence made the Lines), `[parts] use_width` + thresholds (default off), `tierb` variant `D1` (A's
double alone: the one-guitar R2 control), `parts/eval_e28.py`. Tests: `tests/test_parts_width.py`.

- **Why**: E25/E27 - M7a counted every A1 / B2 remix as one part; forced to two it already split some. On dev the
  centred part of A1 is coherent and mid-heavy (coh 0.72-0.999, ms 6.6-33 dB), the double incoherent and wide
  (coh 0.27-0.56, ms -1-0 dB); an ADT (R1) looks wide too, so the evidence reads R2 windows only. B2 (both parts
  centred) has no stereo cue at all and stays one part.
- **E28** (`20261017-170948_0d5774a_exp-e28`, registered before the D1 data and any test scoring; D1 filled in 22 min
  of GPU): dev choice ms_hi 6, coh 0.9, ms_lo 0, n_min 6. Test: one guitar (ADT, N1 32 s, D1 whole song) 15/15 still
  one part; A1 counted right 0/6 -> 3/6, macro +17.0 [0.0, 26.6] points (zeno whole song 0.50 -> 0.99, signejakobsen
  32 s 0.50 -> 0.99, whole song 0.55; rememberdecember unchanged - its centre part never shows in an R2 window) -> the
  CI lower bound is 0, not above it -> rule 2 fails -> held, default off. No regression on P / B2; no new one-guitar
  false positive in 28 remixes. Tier A: unchanged (all five split on left/right already).
- **Use it**: per song, `bandscribe run <song> --set parts.use_width=true` or `[parts] use_width = true` in the job's
  `hints.toml` - worth it when a song has a wide rhythm double and a centred second guitar.

## M7b (2026-10-17): identity tracking

Code: `bandscribe/parts/track.py` (tracker, `repeat_pairs`, `repeat_links`), `bandscribe/parts/idmetrics.py`
(identity switches, block DER), `bandscribe/parts/eval_track.py` (whole-song E27 data), `bandscribe/parts/eval_e27.py`
(E27), `diarize(..., tracking=, repeat=)`, the `parts` stage (deps + `sections`; notes get `rep`; code 4),
`[parts] track` (false), the `part` edit in `bandscribe/tab/edits.py` and the viewer (Shift+1-4, Shift+0). Tests:
`tests/test_parts_track.py`, `tests/test_parts_e27.py`.

- **Tracker** (DESIGN 4.6, simplified): in each router window the notes v0 gave a Line form a layer with a signature
  (median pitch, chord / lead shares, density, R3 ILD, classes); layers of neighbouring windows are linked by a
  Hungarian match on signature distance; a Viterbi pass picks each window's relabelling of the Lines (emission = v0's
  log posteriors, transition = links that end in different Lines, cheaper at section boundaries); windows the repeat
  map pairs are pulled together (ICM). Off by default.
- **Repeat links and the `part` edit**: the parts stage maps each guitar note onto the repeat map's best matching bar
  (same place in the bar, same pitch, 60 ms) - seisyun complex 1,204 of 4,759 notes linked; a `part` edit sets a
  note's part (or Unassigned) and, on replay, the linked notes follow unless an edit of their own set them.
- **E27** (`20261017-160506_91ed416_exp-e27`, registered first; amended before scoring: repeat pairing by matched
  bars, a posterior floor in the grid): 9 Cambridge-MT songs x P/A1/B2 over the whole song = 24 remixes, separated,
  beat-tracked and transcribed in 1 h 52 min of GPU, one song at a time. The dev choice (`w_link` 1) changed no
  note anywhere (settings with `w_link` >= 2 relabelled 19 notes of jetb P, switches unchanged); test median identity
  switches v0 0 = tracker 0 -> FAIL (rule 1), macro difference 0.0 -> pass
  (rule 2); `[parts] track` stays off. Why: v0 counts one part on all 15 A1/B2 remixes (and jokers P), so nothing can
  switch; on the P remixes it does split, the switches are many (dev 27/20/18/10/4, test 23/2/0) but they are single
  notes whose ILD reads the wrong way inside R3 windows (posteriors ~1.0), not windows swapped as a whole.
  Diagnostic: with the true part count forced, v0 already splits centred parts on some songs (zeno A1 0.97, B2 1.00,
  signejakobsen A1 0.65; mistrusted / moosmusic A1 get two parts but no split) - the part count is the bottleneck for
  A and B, not identity.
- **Review** (independent agent; table in `docs/acceptance/M7b.md`): no blockers. Fixed: a bass note assigned to the
  guitar could take a repeat link meant for the guitar note with its index (`from_track`), part edits replayed onto a
  part the run no longer has (now unmatched), the spread dropped a repeat's pin, E27's windows lacked `end_bar` and the
  router's one-bar merge (no effect on the verdict), doc claims about the tracker's changes and the forced count.
- **Tier A**: seisyun complex with the tracker on: 0 notes changed, 11 repeat window pairs, 77.6 % as before.
- **Not done**: note-level smoothing, calibrated posteriors (M10), edits feeding back into the stages, cannot-link
  edits, the MERT solo classifier (E9, optional).

## M4b (2026-10-17): A/B mixer and review tools

Code: `bandscribe/tab/viewer/` (`engine.js` WebAudio engine, `synth.js` tab synth + `syncReport`, `roll.js` piano
roll and part band, `stretch_core.js` + `stretch.worklet.js` pitch shifter; the page `index.html` is an ES module
now), `bandscribe/tab/viewer_media.py` (stage dirs behind the export, recordings, CQT cache, unkept candidates,
router windows), `bandscribe/tab/viewer.py` (routes `/static/`, `/api/media`, `/media/<id>`, `/api/cqt/<id>`,
`/api/candidates`), `bandscribe/tab/edits.py` (note map: `t`, `d`, `on`, `line`, `p`, `amber`, `fc` and `timeline`;
session `not_applied`), `bandscribe/tab/reapply_eval.py` (E26). Tests: `tests/test_viewer_m4b.py`,
`tests/test_reapply_eval.py`, `tests/node/stretch_test.mjs`.

- **Playback**: every source is an AudioBuffer started together on one AudioContext: the original (on), the six sep
  stems and the two practice mixes (decoded when switched on), and one tab-synth buffer per score track. The synth is
  rendered in the background with alphaTab's `exportAudio` (sync points on, other tracks at volume 0) at 44.1 kHz -
  at 22.05 kHz alphaTab's synthesizer turns to NaN within a second - kept mono at 22.05 kHz, and only tracks whose
  notes changed are rendered again after an edit. On seisyun complex every written note time of Bass / Gtr 1 / Gtr 2
  finds its synth onset: median 0 ms, 5-95 % -19..+5 ms, no drift over the song.
- **Controls**: A/B mode (synth; hold B for the recordings), "원곡과 함께 듣기", per-source on / solo / volume, sound
  position (stereo / left / right / mid / side) on the recordings, speed 50-100 % in 5 % steps (`-` / `=`) through
  an AudioWorklet pitch shifter (two taps, 60 ms grains, SOLA alignment within 25 ms; ±0.25 cent at 50/60/75/90 %
  in Node and in the browser), loop (`L`: start -> end -> off; buffer loop points), count-in (`C`: one bar of clicks
  before the next beat). The cursor follows the heard position (output latency and the shifter's 42 ms taken off).
- **Lower panel**: part band over the whole song (sections, router labels, a lane per part + Unassigned + bass,
  click to seek); piano roll on the guitar stem's CQT (cached in the job's `cache/viewer`) with the score's notes
  (part colours, opacity = part probability or fretting confidence) and the transcribers' unkept notes dashed
  (seisyun complex 753); click a note to select it. Ctrl+wheel zooms.
- **Review queue**: edits that no longer land (red, with their time) -> Unassigned notes -> notes under the
  threshold (default 0.6) of the chosen confidence, lowest first; `N` / `P` step through it.
- **E26** (`20261017-130024_740d6fd_exp-e26`, registered first): synthetic edits anchored on the M4a-day result of
  the five Tier A songs (50 guitar + 50 bass notes + 20 bar lines each) landed 564 / 600 = 0.94 [0.82, 1.00] on
  today's result (guitar 0.86, bass 1.00, bar lines 0.99) -> pass. Weak evidence: only 青と夏's guitar notes changed
  since (tempo hint -> grid -> transcription, 4978 -> 3009 notes); there 15 / 50 landed, the 35 misses are notes that
  are gone (31) or moved an octave (4) - none a tolerance miss.
- **Review** (independent agent, 2026-10-17; table in `docs/acceptance/M4b.md`): no blockers, no security issue. Fixed:
  a source decoded during the count-in started early (now joins at the scheduled start), a part track removed by an
  edit left its synth playing and shifted the mixer rows (synth sources keyed by track id, `engine.remove`), editor
  keys fired while a slider had focus, rate changes rewound by the output latency, an empty loop near the end gave
  NaN, synth buffers shorter than the recording wrapped loops early (padded), the shifter's alignment search could
  overrun a block (coarse + fine: worst 1.9 ms of 2.9), the original is now `input/mix_44k_f32.wav` (the stems'
  source, no decoder delay), smaller races; E26's registration corrected (bar-line tolerance 0.3 beat, not ⅓).
  New `tests/node/engine_test.mjs` (fake AudioContext).
- **怪獣の花唄**: `guitar_parts = 1` removed from its `hints.toml` (user OK, 2026-10-17); re-run gives 2 parts
  (`Gtr 1 - Rhythm (acoustic, L)` 1047 notes / 70 bars, `Gtr 2 - Rhythm (acoustic, R)` 575 / 42).
- **Not done** (see `docs/acceptance/M4b.md`): dragging part boundaries (M7c), Line audio (M10), expected-error
  order (M10), colouring the score by confidence, alphaTab's Media Sync Editor (sync is fixed with the M4a grid
  edits).

## M5 (2026-10-08): bass

Two stages between `notes` and `quant` (code `bandscribe/bass/`, worker `bandscribe/workers/f0_bass.py`, gpu env gets
`torchcrepe==0.0.24` and `pesto-pitch==2.0.1`):

- **`bass_f0`** (GPU ~20 s per song + pyin on the CPU meanwhile; the stage ~1:15-1:40 per song): CREPE full and PESTO
  `mir-1k_g7` in the worker, pyin (librosa) in a thread, one 10 ms grid in `f0.npz`. CREPE is decoded here (one
  Viterbi over the whole song on the salience normalised to sum 1): **torchcrepe's own decoder softmaxes the sigmoid
  outputs, which is almost flat, and on bass stems its path sat on the lowest bin for whole phrases** (seisyun complex
  34-36 s: every frame at bin 0, confidence 0, while the activations peaked at 0.8-0.9 on the notes). fmin 32 Hz, the
  worker refuses < 31.71 (test). PESTO's checkpoint pickles OmegaConf containers: loaded with `weights_only=True` and an
  allow-list; one call over a whole song took > 5 GB, so 30 s chunks with 1 s of context (reserved 1.36 GB).
- **`bass`** (CPU, 1.4-2.4 s): `bandscribe.bass.clean` on the gated pass-1 bass. Steps: 2-of-3 octave/semitone anchor and
  harmonic duplicates, kick rejection (drums stem), fragment / decaying-tail merges (only without a new attack),
  **glide chains** (on the contour minus the song's tuning offset, the F0 leaves the first quarter of each interval and
  reaches the last over >= 40 ms of the middle half without a jump; inner notes without a 250 ms plateau go; the first
  note slides into the note the glide stops on; slide partners are fretted on one string via `fretting.assign(links=)`),
  F0 gap fill, the AND policy; vibrato / dead-note suggestions. `[bass]` switches each step.
- **E13** (`docs/decisions.md`, pre-registered at `55fe1df`, run `20261008-184108_55fe1df_exp-e13`, GPU ~14 min; Tier B
  MuScriptor came from E1's eval cache): IDMT 17, FiloBass 12 windows, the six E1 Tier B songs. Per step: anchor Δ −0.18
  [−0.42, +0.04] (hold), **merges Δ +1.29 [−0.06, +3.14] (non-inferior: adopted)**, kick (Tier B) +0.43 [−0.25, +1.04]
  (hold), gap fill −0.06 [−0.14, +0.01] (hold). Defaults now: fragments + slides on, the rest off. The anchor misreads
  pitches that are still moving (the ends of a slide) and the trackers share octave slips (Tier B octave error 1.12 →
  1.50 with it). Two voters vs AND: recall +20.9 [+17.2, +25.3], precision −15.4 [−36.4, −2.6] → the M5 criterion fails.
- **Techniques in the tab:** only slides are written (`notation.WRITTEN_TECH`); vibrato (IDMT 1/31 found, 1 of 4 marks
  right) and dead notes (17/28, 17 of 31) stay suggestions in `score.json` flags. AIZO gets 53 dead-note suggestions.
- **Trackers, 30-100 Hz** (raw pitch accuracy with confidence >= 0.5): IDMT CREPE 85.9 %, pyin 56.0 %, PESTO 0.9 % (its
  voicing confidence almost never passes 0.5 on bass; ignoring it 68.6 %); FiloBass 82.7 / 52.9 / 34.4 % (PESTO 7.1 %
  octave errors). pyin's pitch is good (rpa ignoring confidence 91 / 89 %), its voicing is strict.
- **Renamed:** the bass track id is `bass` ("Bass"; was `bass_raw` / "Bass (raw)"); export `tab/bass.txt`,
  `midi/bass_quantized.mid`, new `midi/bass.mid` (cleaned, audio timeline; `midi/bass_raw.mid` stays). Old edit logs
  naming `bass_raw` are read as `bass` (`edits.OLD_TRACK_IDS`).
- **seisyun complex** (the user's case): bars 4-6 were 41 39 38 37 36 35 31 31 31; now one note, E string fret 13
  sliding to fret 3 (then the 31s). Against the GP sheet (bar bags): bass F1 90.1 → 90.4 (fp 98 → 92), pitch-class
  91.8 → 92.1, bass string/fret on matched pitches 45.4 → 45.2 %, guitar unchanged (73.9).
- **Five songs re-run** with the adopted defaults (after the review fixes): guitar quantisation, beat families, guitar
  alphaTex and both tunings identical before / after; bass −2 to −6 notes per song; alphaTab parses all five (4-33
  slides per song). Tuning offsets the bass stage measures: SC +21, KH +13, AIZO +10, AZ +1, aotonat +0.5 cents.
- **Review** (one read-only pass, 12 probe scripts): 2 major + 3 minor, all fixed with tests: (1) the glide test
  counted contour frames anywhere in the interval's middle half, so a recording 25+ cents off A440 or a vibrato note
  read as a slide (+35 cents: a plucked 40-41-42-43-45 lost 41 and 42); now a real crossing on the contour minus the
  tuning offset. (2) same-pitch notes under 80 ms were joined even with their own attack (16ths at 170 BPM: 8 → 1).
  (3) a dropped duplicate kept a 50 ms sliver instead of the long note (anchor arms only). (4) the F0 eval cache key had
  no code version (`f0.CODE_VERSION`). (5) the worker's files sat in `_f0/_worker` and were hashed as stage outputs.
  Re-measured on E13's items (descriptive): the adopted merges +0.92 [−0.01, +2.16] over raw, still non-inferior.
- Acceptance (`docs/acceptance/M5.md`): PASS tracker report + fmin test, IDMT string/fret 82.1 % (>= 80), IDMT string
  accuracy and dead-note recall reported, ablation with CIs and no guitar regression. FAIL: two voters vs AND
  (precision). BLOCKED: octave error rate (IDMT 0.52, Tier B 1.10 points), tick F1 (IDMT 87.8, FiloBass 91.3) and
  FiloBass F1 (91.5) pass numerically, but MuScriptor's training data is private, so every public set is
  contamination-unknown and the rule forbids absolute claims; they need Tier A (the user's Guitar Pro files).
- Not done: SCNet ensemble and the 3-voter (YourMT3+) policy of E13 (not installed), HO/PO and slap (no ground truth),
  S14 root cross-check (needs rhythm Lines, M7a), a real F0 HMM segmenter (the gap fill is a simple stable-run rule).

## M7a (2026-10-11): Part view v0

Code: `bandscribe/parts/` (`layers.py` symbolic layers + the TabForge-style baseline, `reid.py` streams per window
and the robust part count, `arbitration.py` per-note Line posteriors and Unassigned, `roles.py` S9 order and
names, `diarize.py` end to end, `stage.py` the `notesig` and `parts` stages, `eval.py` E25), Tier B variants in
`bandscribe/router/tierb.py` (`VARIANTS`, `variant_plan`), part tracks in `bandscribe/tab/stage.py` (`DERIVED` =
`gtr_1`..`gtr_4` + `unassigned`, `track_name`, `line_tunings`). Tests: `tests/test_parts.py`,
`tests/test_parts_tracks.py` (replaces `test_split_tracks.py`).

- **What a run does now** (default `--until score`; both new stages CPU, seconds):
  - `notesig`: each guitar note's stereo signature (ILD / coherence / mid-side at its harmonic bins), once per
    notes result (another `--parts` does not redo it; seisyun complex 7.2 s).
  - `parts`: per router window the simultaneous streams - spatial only by default: an R3 window with 6+ notes at
    |ILD| >= 3 dB on each side; two parts when two streams hold in 2+ windows (over 2 sections, or all of them),
    else one. Lines are seeded by that evidence; each note's posterior from pan, role, class and share (strums
    outside R3 decided together); Unassigned under 0.4, amber under 0.6; names like `Gtr 1 - Rhythm (dist., L)`
    (ASCII: MIDI track names are Latin-1), Gtr 1 = most active bars. `--parts N` forces N; a forced part that gets
    no notes is dropped with a reason.
  - `tab` frets each part (`gtr_1`..), with its own tuning only when its 150+ notes clearly want another one (bass
    held at the song's tuning, 0.3 cost per note better); the merged guitar is the staff `All guitars (reference)`,
    Unassigned notes a staff `Unassigned (review)`; a one-part song is `Gtr` fretted. The M3-M6 path that fretted the
    merged guitar itself (`--parts 1`) is gone.
- **E25** (`20261011-182127_2436e55_exp-e25`; registered before running; settings chosen on dev songs, judged on
  test): E6's P and ADT remixes plus new variants A1 (wide double + one centred take), B2 (both P takes centred) and
  N1 (one take, mono), first 32 s. Dev choice `n_min` 6, 2 windows, side 3 dB, class and symbolic evidence off
  (class evidence split 3 of 10 dev one-guitar remixes - MuScriptor labels one take acoustic/clean; the symbolic
  lead layer never made two streams). Test: part count 0.72 (P 0.8, ADT 1.0, A1 0.0, B2 0.4, N1 1.0; A1+P 0.50),
  MAE 0.28; assignment over majority P +15.2 [0.0, 30.5] (the two P remixes whose count was right: +38; two P
  remixes have one active part, zeno's second take plays 7 s), A1 0, B2 0 (the symbolic-only baseline beat s8v0
  on A1 by 24.5 points: zeno, signejakobsen, rememberdecember have parts with different chord/single-note texture);
  one guitar 12/12; Unassigned 0 %; coverage-accuracy flat (posteriors over-confident).
  - The GPU cache fill ran one song at a time (`data/tmp/m7a/orchestrate.sh`, not tracked): it waited while a game
    held the GPU and redid a song a game interrupted.
- **Tier A** (descriptive, GPU stages cached): part counts seisyun complex 2 (37 of 41 windows), 青と夏 2 (19/52),
  AZ 2 (20/23), AIZO 2 (5/21; M6 left it merged; Gtr 2 = 205 notes in 19 bars), 怪獣の花唄 1 (its `hints.toml`
  `guitar_parts = 1` from M3; the estimate is 2). **seisyun complex vs the user's Guitar Pro tabs**: 77.6 % of the
  notes that sit in exactly one sheet guitar's bar went to that guitar's part (guitar 1 83.6 % of 1,992, guitar 2
  71.6 % of 898; M6 76.2 %); part bag F1 merged -> part 63.5 -> 65.7, 39.3 -> 58.1; every note is now in a part
  (M6 left 233 in merged windows). Script: `data/tmp/m7a/eval_sc_parts.py`.
- **`--parts` switch**: seisyun complex `--parts 3` rerun 26.5 s (parts 0.2, quant 0.7, tab 23.5 = song tuning 8.4
  + per-part tunings 13.4 + fretting 0.6, score 2.1) - far over the 2 s criterion; `--parts 3` there gives 2 parts
  (the spatial evidence wins every note; reason logged).
- **Review** (independent agent, 2026-10-11; table in `docs/acceptance/M7a.md`): an added note's edit record stored
  the part number of the moment, so a re-run that renumbered or recounted the parts moved it or made a one-note
  phantom part track - records now keep the neighbour's anchor (`near`) and follow its current part; editing a run
  from before M7a asks for a re-run; the removed `[router] split*` keys are ignored with a warning; guitar notes with
  no part (added or moved by an edit) go to the Unassigned staff; `--parts 3` no longer empties a side that is all
  lead layer (parts code version 3); the remix experiments read the pinned E6 run (`split_eval.E6_RUN`), fail on a
  missing E6 mix and use the config they are given; E6b's explanation of its high "different" F1 corrected (those
  windows have one take only; 72 % of the two-take windows fall under the veto).
- **Not done in M7a** (see `docs/acceptance/M7a.md`): same-position parts (A, B), automatic counts over 2, unison
  (R3) and twin (R5) rules and their fixtures, R4/R6/R9 rules, calibrated posteriors, a fast part-count switch.
- **Follow-up 2026-10-17 (the `--parts` switch):** a `tuning` stage (deps notes, bass, grid, sections; it quantises
  the merged guitar and the bass without the parts) holds the song's tuning, so another part count reuses it; a
  part's own tuning is searched among the song's best 5 only, with the bass held (`tuning.rank_guitar`); fretting
  works out each pitch tuple's candidates once per call; the score stage converts note times with a plain-float
  `TempoMap.beat_time` (same values). seisyun complex `--parts 4` (a new count): parts 0.2 + quant 0.8 + tab 1.5 +
  score 1.5 = 4.0 s (was 26.5 s); a count run before is all cache. The five Tier A songs keep their tunings and parts
  (the per-part search is narrower than before - the song's best 5, 7-string candidates from the song - so "same
  result" holds on these songs, not by construction; none of them has a part with its own tuning).
  Still over 2 s: the viewer switch (M7c) re-frets the changed parts with the stored tuning.

## M6 (2026-10-10): stereo router and the first automatic part split

Code: `bandscribe/router/` (`route.py` windows/features/labels/solo gate, `thresholds.py`, `notesig.py` per-note stereo
signature, `split.py` line assignment, `stage.py` the `router` and `split` stages, `tierb.py` Tier B remixes,
`experiments.py` E6, `split_eval.py` E22/E22b); part tracks in `bandscribe/tab/stage.py` (`DERIVED`, `track_notes`).
Tests: `tests/test_router.py` (42), `tests/test_split_tracks.py` (8).

- **What a run does now** (default `--until score`, both stages CPU, seconds):
  - `router`: per 4-bar window (cut at sections) the stereo image of SW's guitar stem -> R0 narrow / R1 ADT /
    R2 wide same-part double / R3 parts at different positions / mixed / silent (`router.json`), thresholds re-fit
    by E6; the solo gate v0 is computed and recorded but nothing uses it yet.
  - `split`: each guitar note's own ILD / coherence / mid-side at its harmonic bins (`notesig`); in R3 windows the
    sign of the ILD sends it to L or R (E22b). A / narrow / ADT / mixed windows stay merged. The song is split only
    when >= 50 % of its guitar notes are in R3 windows and each side has >= 30 notes; `tab.guitar_parts = 1`
    turns it off.
  - Then `tab` frets **Gtr 1 (L)** and **Gtr 2 (R)** (one player each) from the merged guitar's notes, the merged
    guitar stays a staff (`Guitar (all parts)`), plus `tab/gtr_l.txt`, `tab/gtr_r.txt` and their quantised MIDI.
    A part note is the guitar note (uid `guitar_all:<i>`): a viewer edit of a part note edits the guitar note and
    the part follows; a bar-line edit re-quantises from the split notes.
- **E6** (`20261010-180608_6bed725_exp-e6`, GPU 11.5 min): 13 Cambridge-MT songs remixed into B / A / P / ADT scenes
  (45 mixes, 284 windows). Machine take labels first (onset correlation + chroma): the lines.yaml drafts were wrong
  both ways (Woodfire "1a/1b" = one performance on two mics; Mu Gtr1/Gtr2 = a double under two line ids).
  Router features survive SW (true bus and SW stem score alike); the synthetic thresholds missed real doubles (dev
  R2 0.29-0.35). Re-fit on dev songs: test macro 0.994 [0.985, 1.000], R2 1.00 -> **GO**; re-fit thresholds are the
  defaults (`router/thresholds.py` ADOPTED). Pre-registration changes from a 2-song smoke run: ADT needs the delayed
  peak to beat the zero-lag correlation (centred mono parts looked like ADT), grouped threshold search.
- **E22** (as registered, synthetic split thresholds; `20261010-183611_fc4ac94_exp-e22`, GPU ~13 min): held. P +18.4
  [7.9, 29.6] points over majority (< 20); A lost the lead line (-35 points).
- **E22b** (registered before E22's results; thresholds chosen on dev songs, judged on test; no GPU): **P adopted**,
  `p_ild_db` 0 (the sign alone): test +33.6 [16.2, 44.6] points over majority, +32.7 [15.3, 44.5] over window-level
  pan. **A held**: the only dev settings within the leak/drop limits put nothing in the centre line.
- **Five songs** (all MuScriptor views from the cache; router + split + quant + tab + score on the CPU):
  - split on: seisyun complex (L 2,800 / R 1,726 notes, 233 merged), 青と夏 (2,514 / 457 / 38), AZ (2,542 / 2,259 /
    16); off: KH (one
    guitar by hint), AIZO (45 % of its notes in P windows). alphaTab parses all five scores (split songs: four
    tracks).
  - Router (re-fit thresholds): mostly R3 (seisyun complex 39/41 windows, 青と夏 48/52, AZ 23/23); AIZO R2 7, R3 6,
    R0 4.
  - **seisyun complex vs the user's Guitar Pro tabs** (descriptive, Tier A): part bag F1 per bar, merged -> part:
    Ele. Guitar 1 63.5 -> 65.1 (our L), Ele. Guitar 2 39.3 -> 58.1 (our R); of our notes that sit in exactly one
    sheet guitar's bar, 76.2 % went to that guitar's part (guitar 1 80.1 % of 1,992, guitar 2 72.3 % of 898;
    majority baseline 50 %). Script: `data/tmp/m6/eval_sc_split.py`.
- **Review** (independent agent, 2026-10-10; table in `docs/acceptance/M6.md`): the viewer's track list, the S key
  and the edit hash on part notes (fixed in f3823b7 before the review landed), shared notes missing from both
  parts when `split_p_ild_db` > 0, the tempo hint turning a compound meter simple (now skipped with a warning; grid
  code version 5), added notes without a part (an added note now joins the part of the note it was added next to),
  part-track warnings, an empty stem labelled `mixed`, E22b's unpaired A means and `tierb`'s onset/activity frame
  mapping (code version 2). The octave-overlap gap of the shared-bin rule is documented, not changed (the split was
  validated with it). Exports of the five songs byte-identical before/after; the parts now report the notes their tab
  drops as unplayable (seisyun complex 35 / 9, 青と夏 168 / 3, AZ 48 / 66).
- **Follow-up 2026-10-11** (user: "전부 진행시켜"): pushed `m6`; fetched the CenterWide BS-RoFormer v2 / v3beta centre
  models (`bandscribe/router/centre.py`, model card now CC BY-NC-SA 4.0, personal use; reserved 570-582 MB on the SW
  worker). **E6b** (R1 self-agreement, real doubles' L/R channel transcriptions): veto 0.61, variation 0.61 / 0.80
  (16 windows, provisional). **E7** (centre extractor + DESIGN's A rule "lead = centre notes the side view does not
  play"): held, every centre view below the merged baseline (best CenterWide v3beta -13.8 [-29.9, +0.5] lead F1
  points). Acceptance now 7 pass / 2 fail / 1 blocked.
- **Not done in M6** (see `docs/acceptance/M6.md`): views S5 and the centre extractor (E7 needs the CenterWide
  download, the user's call), S7 fusion + Basic Pitch voting (E19), R1 self-agreement calibration (E6b), E8, E20
  (xlance download), the A split (held by E22b), unison (shared) notes in P (with the sign rule a unison goes to
  the louder side only), human-checked Tier B labels.

## Grid: tempo hint (2026-10-10)

青と夏 is counted at 185 BPM (the user) but tracked and gridded at 93.75, so its eighths were written as sixteenths.
Code: `bandscribe/analysis/grid.py` (step 1b reference level and step 2 choice), `[grid] tempo_bpm` (`config.GridCfg`,
`defaults.toml`), `bandscribe run --tempo`, grid stage code version 3 → 4; 8 new tests (`test_grid_derive.py` 7,
`test_pipeline_cli.py` 1).

- **No automatic rule found.** On the stored outputs the onset rule cannot separate 青と夏 (counted 185) from AIZO (95):
  both are tracked at 93.75, onset F1 favours double time on both (0.759 vs 0.540; 0.569 vs 0.340) and no prior
  centre flips one without the other. A backbeat test (SW drum stem; kick 40-120 Hz vs three snare bands; is the
  kick/snare contrast alternating beat by beat at half / kept / double level?) found the alternation on SC and KH
  at their kept level (lag-1 autocorrelation -0.45 to -0.58) but none at any level on 青と夏, AIZO or AZ. So the
  level is the user's hint; the automatic decision is unchanged.
- **The hint** (`[grid] tempo_bpm = 185` in the job's `hints.toml`, or `--tempo 185` once) only picks the level: step
  1b's reference becomes the tracked level times 2^k (k = whole octaves to the hint, at most one), step 2 takes the
  candidate nearest the hint; the tempo itself stays the tracked one. A hint more than 19 % off every power-of-two
  multiple (e.g. a dotted-quarter tempo for tracked eighths) is ignored with a warning; one still more than half an
  octave away after both steps is warned about.
- **青と夏 with the hint** (job `hints.toml` written 2026-10-10): 339 beats inserted, the 8-beat tracked bars cut in two:
  408 → 811 beats, 103 → 204 bars, median 92.3 → 185.1 BPM; bars not in 4/4: a 5/4 and a 2/4 before, one 2/4
  and the song's cut-off last bar after.
  Guitar staff note values: sixteenths 61 % → 22 %, eighths 18 % → 46 %, 32nds 11 % → 0 %.
  - **Side effect:** the bar lines moved, so did the presence windows, so the guitar mask lost `organ` and
    `synth_strings` and `amt_gtr` ran again (2:57 GPU): `guitar_all` 4,978 → 3,009 notes (836 of the new pass's
    4,127 labelled `electric_piano` and left out). The same prompt sensitivity as in "Grid: beats tracked twice"
    (3,996 → 4,978 then); without ground truth neither is known to be better. Presence windows on a fixed time
    grid would keep grid changes out of the guitar mask (not done).
- **The other four songs** (no hint) re-ran grid → score on the CPU with the MuScriptor views from the cache: every
  export file byte-identical (alphaTex, GP5, text tab, tab.json, all MIDI, instrumentation.json).
- Not touched: AIZO 29-44 s (a double-time passage with jitter outside 1b's band) and AZ's contradicting
  downbeats half a beat apart (both in "Grid: beats tracked twice").

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

- **After E39 / E40 (2026-10-21):** duplicates and tiny-note bursts are cleaned (adopted). Left: octave errors (no
  symbolic tell found on dev; needs audio evidence), keys / strings bleed (a piano+other transcription vote, GPU per
  song - the energy route E38 failed), and re-running the sheet songs' M9 table with the clean-up on.

- **After M10 interim (2026-10-18):** the largest error class is octave/duplicate chord tones (30 % of the notes
  against the sheets); with six sheet songs a note clean-up experiment can now be judged. M10's edit learning waits
  for the user's edits; M11 (YouTube) is optional. Open from M9: chords (E31, 54 %: needs bass in the histogram + tension templates, judged on another song's
  chords), bends in chords (E32: partial tracking, judged on the clean Tier B lead takes' f0).
- **After M7c (2026-10-17):** by DESIGN's order the next milestone is M8 (narrow mixes: the B route, repetition
  subtraction, safety net; optional voters). Same-position count: the width evidence
  (E28) is held for want of test songs - a new experiment with more A-type songs could adopt it; B (two centred
  parts) still has no evidence. Note-level smoothing / calibrated posteriors are M10. Open M7a items: a `--parts` switch under 2 s (the tuning stage is done, 4.0 s now;
  the viewer switch in M7c re-frets with the stored tuning like M4a edits), same-position parts (E25: the
  symbolic-only split beat v0 on some A1/B2 songs but splits single guitars too - a gated symbolic evidence would
  need its own experiment), calibrated part posteriors (M10).

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
   item 6 stays open: the editor never starts GPU runs). ~~M5~~ done 2026-10-08 (tag `m5`, see "M5"). Next (DESIGN
   order): ~~**M6**~~ done 2026-10-10 (tag `m6`, see "M6"; tempo hint first, see "Grid: tempo hint"). Next (DESIGN order): **M7a**
   (Part view v0: K estimate, symbolic split, Unassigned) - before it E6b (R1 self-agreement calibration, needs L/R
   transcriptions); or the M6 gaps (A path: E7 centre extractor needs the CenterWide download, the user's call). Older
   note: the grid's tempo level (~~spurious beats~~:
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

- **Two of the user's D:\backing files are guitar-removed, not originals (found 2026-10-18):** `푸름과 여름.wav` (job
  f-8eafb8187d7fe235, "aotonat") and `괴수의 꽃노래.wav` (job f-e064a271b0834683, "KH") have a separated guitar
  stem at −38.5 / −35.5 dB - like the guitar-removed `*_backing.mp4` (−36.6 to −56.2 dB) and unlike the originals
  (SC.mp3, AIZO.mp3, AZ.mp3 and the five Kessoku Band songs: −18 to −25 dB). Every number from those two jobs
  (tempo hint, bleed shares, guitar note counts, the E3/presence notes) describes a backing's guitar residue. Their
  originals are needed. Job ids: f-cc5cb9e81a50d3a4 = AZ.mp3, f-e064a271b0834683 = 괴수의 꽃노래.wav (KH),
  f-93e11bb95a10b9fc = AIZO.mp3, f-bd0dc18775389993 = SC.mp3.

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
