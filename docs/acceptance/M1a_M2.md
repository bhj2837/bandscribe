# M1a·M2 인수 검증 (2026-10-05)

- **대상:** `main` `b50ea18` + 커밋하지 않은 작업 트리(2026-10-05 리뷰 수정 14건: 수정 43개 파일, 새 파일 `tests/test_eval_e3c.py`).
- **기준:** DESIGN §10 의 M1a·M1b·M2 완료 기준이 정본이다. 증거 명령은 M1_M2_SPEC §11 을 따랐다.
- **판정:** PASS / FAIL / BLOCKED / DEFERRED.
  - FAIL: 잠정(DESIGN §10: "수치 문턱은 잠정값") 문턱이라도 못 넘으면 FAIL 이다.
  - BLOCKED: 사용자만 줄 수 있는 것이 있어야 끝난다.
  - DEFERRED: PROGRESS 에서 이유와 함께 뒤 마일스톤으로 옮긴 것이다.
- **직접 실행한 것:** 전체 기본 테스트, `doctor`, `bandscribe run` 2곡 × 2회, a1–a13 증거 명령, M2 관련 테스트 파일.
- **GPU:** 캐시된 실행만 했다. 새로 쓴 GPU 는 Beat This! 7.6 s 뿐이다. 벤치와 실험은 다시 돌리지 않고 기존 기록(`data/bench`, `data/runs`)을 검토했다.
- **로그:** `data/scratch/acceptance_m2/` (로컬, 커밋하지 않음).

## 요약

| 마일스톤 | 항목 | PASS | FAIL | BLOCKED | DEFERRED |
|---|---|---|---|---|---|
| M1a | 13 | 13 | 0 | 0 | 0 |
| M1b (병행, 게이트 아님) | 3 | 0 | 0 | 3 | 0 |
| M2 | 25 | 19 | 3 | 3 | 0 |
| 공통 (테스트·doctor) | 2 | 2 | 0 | 0 | 0 |

- **M1a: 통과.**
- **M2: 아직 통과가 아니다.**
  - FAIL 3: b10 지연 잔차 10.9 ms(잠정 문턱 10 ms), b13b 지금 제품 편성 추정기의 곡별 표 없음(E3b 미실행), b15i E24 미실행.
  - BLOCKED 3: b3 NVIDIA 설정 전환, b12 오염 불명, b13c Tier A 편성 정답.
- **M1b: 0곡**(GP 정답 없음). 게이트는 아니지만 "M2 전 2–3곡" 목표는 맞추지 못했다.
- b13·b15 는 하위 항목으로 나눠 셌다. DEFERRED 로 옮겨진 완료 기준은 없다. 리뷰에서 연기된 항목 2개는 아래 「알려진 한계」에 있다.

## M1a 측정 도구

| id | 기준 (요약) | 판정 | 증거 |
|---|---|---|---|
| a1 | 항등이면 F1 = 1.0 | PASS | `pytest tests/test_eval_metrics.py -q -k identity` → 2 passed |
| a2 | 무작위 20 % 삭제 → 재현율 0.80±0.02 | PASS | `-k deletion` → 1 passed |
| a3 | +12반음 → pitch F1 0, chroma F1 1.0 | PASS | `-k octave` → 2 passed (±1반음도 0) |
| a4 | 라벨 교환 → 헝가리안 전 0, 후 1.0 | PASS | `-k label_swap` → 1 passed |
| a5 | Unassigned 만 늘리면 macro 정확도가 오르지 않음 | PASS | `-k unassigned_monotone` → 2 passed |
| a6 | 모든 음을 한 Line 에 → 다수 클래스 기준선과 같음 | PASS | `-k majority` → 3 passed |
| a7 | 템포를 흔든 합성 연주: 격자 오류가 음표 지표에 새지 않음 | PASS | `-k grid_leak` → 3 passed (`evaluate_item` 경유 포함, 리뷰 수정 7) |
| a8 | DTW 가 다운비트 ≥ 95 % 를 ±70 ms 안으로 복원 | PASS | `pytest -m slow tests/test_eval_align.py -q -s` → **0.969** (32개 중 31개, 중앙 13.4 ms, 최대 73.5 ms). 탭 4개 경로도 통과 |
| a9 | 부트스트랩 CI 출력 + CPU 단계 2회 byte 동일 | PASS | `bandscribe eval run --suite quick --system oracle-noisy` 2회: 집계 16행에 `ci_low/ci_high`, `metrics.csv`·`summary.json` SHA-256 동일(`08cf789c…`). 결정성 테스트 `test_eval_determinism.py` 2건, `-k "determinis or byte_identical"` 3건(파이프라인 CPU 단계, sections, amt) 통과 |
| a10 | 리믹스가 router_exp 특징을 ±0.02 안에서 재현 | PASS | `pytest tests/test_eval_remix_parity.py -q -s` → 14장면, max \|Δ\| = 0 |
| a11 | Basic Pitch 스모크: GuitarSet 20트랙 onset F1 > 0.6 (판정용 아님) | PASS | `bandscribe eval run --suite bp-smoke` → `note_f1_onset50` **0.781** [0.732, 0.851] (E23 값. 10-03 옛 기본값 실행은 0.726). GuitarSet 은 Basic Pitch 학습 데이터라 참고값이다 |
| a12 | contamination 표: MuScriptor·Basic Pitch·SW 학습 목록, 모르면 '불명' | PASS | `docs/eval/contamination.md`: MuScriptor 전 칸 불명(논문·모델 카드 근거), Basic Pitch 논문 기준으로 채움, SW 불명(학습 기록 없음), 각주 링크 |
| a13 | 산출물: eval run/compare/import-external, gt import/render/align/taps, remix, Tier B·C 로더, bp310, decisions.md | PASS | `bandscribe eval --help`(+exp, reftx), `bandscribe gt --help`(+approve). `eval/remix.py` 12개 시나리오. `bandscribe data list`: guitarset·idmt_bass_st·filobass 검증됨, cambridge_mt 13곡 받음, medleydb 수동 필요(승인 시 항목). `pytest -m bp310` → 5 passed (Python 3.10.21). decisions.md 에 9개 id 모두 있음(지금은 결정 상태) |

## M1b Tier A 점진 구축 (병행, 게이트 아님)

| id | 기준 (요약) | 판정 | 증거 |
|---|---|---|---|
| m1 | 곡마다 DTW 정렬 + 수동 다운비트 보정, `align_check.wav` 승인, 펼친 마디 수 일치 | BLOCKED | `data/eval/tierA/songs.yaml`: 후보 5곡 모두 `gt: null`이고 gt.yaml·GP·align_check 가 없다. 도구(`gt import/align/taps/approve`)는 M1a 에 있다. **필요: 사용자의 GP 파일** |
| m2 | 처음 2곡은 다운비트를 전부 수동 탭해 정렬 도구와 비교(≥ 95 % ±70 ms) | BLOCKED | 같은 이유. 탭할 정답 구간이 없다 |
| m3 | 목표 곡 수(M2 전 2–3곡) + 곡당 소요 시간 기록 | BLOCKED | 0곡 |

## M2 첫 수직 슬라이스

| id | 기준 (요약) | 판정 | 증거 |
|---|---|---|---|
| b1 | 앱을 닫고 SW 4분 곡: reserved 피크 ≤ 4.5 GB, Shared Usage 증가 없음 (잠정) | PASS | `data/bench/20261003-171435_1e422dc` (239 s, 칸 3개). chunk=588800: reserved 1688 MB, 자기 shared 증가 2 MB(문턱 64), `ok`.<br>조건 주의:<br>- 앱을 닫지 않았다(시작 시 GPU 프로세스 18개). 더 불리한 조건에서 충족한 것이다.<br>- 리뷰 수정 전 코드(`1e422dc`)로 쟀고, Sysmem 설정은 `unknown` 으로 기록됐다. |
| b2 | 평소 앱 상태: 사전 점검이 예산에 맞는 칸을 고르거나 이유를 밝히고 기다림 | PASS | 오늘 괴수의 꽃노래 실행: `VRAM 사전 점검(beats_beatthis): 예산 1.9 GB (NVML 여유 2.6 GB - 여유분 0.7 GB) -> final0 (필요 약 0.4 GB)`.<br>대기 사례: `data/logs/gtab.log` 2026-10-03 `VRAM 부족: 필요 약 2.8 GB, 여유 2.0 GB (다른 GPU 사용 프로그램: …). 앱을 닫으면 이어서 진행합니다. 대기 중...`.<br>`tests/test_vram.py` → 28 passed. |
| b3 | Sysmem Fallback 끔·켬 둘 다 사다리 시험, 켬에서 shared 증가·느림이면 실패 처리 (잠정) | BLOCKED | **필요: 사용자가 NVIDIA 제어판 "CUDA - Sysmem Fallback Policy" 를 바꿔 가며 두 번 실행.** 설정은 doctor 가 보여 주는 기본 인터프리터 경로에 건다. 명령: `bench gpu --backend sep --rungs auto --ballast-mb auto --fallback off`, 이어서 `on`.<br>지금 기록은 모두 `fallback=unknown` 이다.<br>참고: 09-30 ballast 실행(설정 미상)에서 칸 0 `shared_growth`(66–72 MB) → 칸 1 `ok` 로 내려갔다. 그러나 칸 1 은 느렸다(0.87–1.42×, 기준 약 5.4×). 오늘 수정 10·12 뒤에는 다시 재지 않았다. |
| b4 | `set_per_process_memory_fraction` 으로 OOM 사다리 검증 | PASS | `data/bench/20260930-181046_1e422dc` (`--cap-mb 1500`): 칸 0 `oom` → 칸 1 `ok` (reserved 1302 MB, 4.53×). `tests/test_gpu_common.py` → 36 passed |
| b5 | 실행한 칸마다 reserved·NVML 피크·실시간 배수 기록 | PASS | `data/bench/*/bench.csv` 11건(`max_reserved_mb`, `nvml_used_peak_mb`, `pdh_*`, `rtf`, `load_s` 열). `data/bench/vram_table.json` 6칸(sep 3, amt 2, beats 1) |
| b6 | leftover RMS < 믹스 −20 dB (잠정) | PASS | `stages/stems/*/stem_stats.json` 작업 11개: −27.6 ~ −33.7 dB, `flag` 0건, 10 s 창 최악값 −23.9 dB. SC −31.6, 괴수의 꽃노래 −32.4 dB |
| b7 | float 스템 → 24-bit FLAC 보관·복원 오차 < −100 dB (잠정) | PASS | `tests/test_sep_archive.py` → 6 passed(희박 스템 포함). `data/runs/20261005-143016_b50ea18_exp-stereo-preservation`: 실제 스템 −136.9 ~ −148.1 dB(piano 는 `skipped_sparse`) |
| b8 | 합성 A 믹스를 SW 에 통과: bin별 coherence·ILD 보존 편차 보고 | PASS (서술) | 같은 실행의 A 장면: bin Δcoh 0.26, ΔILD 5.3 dB. 창 특징은 Δcoh +0.005, Δhard +0.012 로 거의 보존.<br>B·P70·twin 은 SW 기타 스템이 −75 ~ −80 dB 로 비어 정보가 없다.<br>`summary.json` 의 4장면 평균(Δcoh 0.59, ΔILD 16.3 dB)은 빈 장면이 섞인 값이라 쓰면 안 된다. |
| b9 | MuScriptor-medium 전곡 reserved ≤ 4.5 GB, 실시간 배수·로드 시간 기록 | PASS | 239 s 클립 기준:<br>- upstream 로더(`data/bench/20261003-172121_1e422dc`): 2392 MB, 1.46×, 로드 0.9 s<br>- lean 로더(기본값, `…-171752_1e422dc`): 1740 MB, 1.28×, 로드 10.7 s<br>조건 주의는 b1 과 같다. small 칸은 가중치가 없어 `unavailable` 이다. |
| b10 | 지연 보정 후 onset 잔차 중앙값 < 10 ms | **FAIL** | `data/runs/20261003-190418_1e422dc_exp-latency`: held-out(GuitarSet 연주자 03–05, 180트랙) **10.87 ms [10.47, 11.28]**. CI 하한도 10 ms 위다.<br>문턱은 잠정값이다. 상수 −3 ms 는 채택됐다(b15g).<br>decisions.md 의 원인 추정: MuScriptor 출력의 10 ms 양자화와 GT 오차. |
| b11 | E23: Basic Pitch 최소 음 길이·문턱을 학습 밖 dev 에서 튜닝, manifest 기록 | PASS | decisions.md E23 채택 `f7_o0.6_fr0.4`, test Δ +6.6 [+3.8, +9.9]점(EGDB = Basic Pitch 학습 제외). SC·괴수의 꽃노래의 `amt_bp` manifest: onset 0.6 / frame 0.4 / min_note_frames 7 |
| b12 | 게이트: 디스토션에서 MuScriptor − Basic Pitch onset F1 ≥ +5점 | BLOCKED | `data/runs/20261004-010229_1e422dc_exp-gate-m2`: Δ +8.1 [+2.1, +13.5]. 그러나 MuScriptor × EGDB 가 `불명`이라 사전 등록 규칙상 `판단 보류 (오염 불명)` 이고, 통과로 쓰지 않는다.<br>**필요: MuScriptor 학습 밖이 확인된 디스토션 기타 정답(예: 사용자 GP 파일이 있는 곡).** |
| b13a | 기타 클래스를 마스크에서 빠뜨린 경우 0건 | PASS | `mask_guitar_view` 는 리스트 마스크에 기타 3클래스를 늘 넣는다(코드 구조). E3 의 모든 arm·문턱에서 누락 0. `tests/test_amt_instrumentation.py` → 32 passed. 오늘 두 곡의 마스크에도 기타 3클래스가 모두 있다 |
| b13b | 편성 추정이 Tier B 곡의 실제 편성을 맞히는지 곡별 보고 (지금 제품 추정기) | **FAIL** | SPEC §11 b13: 2026-10-04 이후 제품 경로(표본 존재 패스)를 기술하는 표는 **E3b** 인데, 사전 등록만 하고 실행하지 않았다.<br>있는 표는 E3 의 것이다(믹스 패스 추정기, 지금은 제품이 아님): JetB 건반 맞음, Mistrusted 신스 놓침, Secretariat 해먼드 놓침, 합성 패드 8장면 모두 놓침. |
| b13c | 같은 보고, Tier A 곡 | BLOCKED | Tier A 편성 정답이 없다. 추정값만 있다(예: 괴수의 꽃노래 keys 0.91, strings 0.92 → present). **필요: 사용자의 GP 파일 또는 편성 정답** |
| b14 | 캐시로 재실행하면 2분 안에 끝남 | PASS | `bandscribe.cmd run D:\backing\SC.mp3 --until notes`: 1회 0.87 s, 2회 0.84 s(13단계 모두 캐시).<br>`… D:\backing\괴수의 꽃노래.wav …`: 1회 40.5 s. beats 키가 바뀌어 Beat This! 7.6 s 를 GPU 로 다시 돌렸고, grid 24.7 s 등 CPU 단계를 다시 계산했다. MuScriptor 는 전사 캐시에서 왔다. 2회 0.89 s(13단계 캐시).<br>실행 중 다운로드 없음(doctor: 모델 6개 있음). |
| b15a | 워커 `sep_msst`, `amt_muscriptor` | PASS | 두 곡의 `gpu_run.json`: sep `chunk=588800`, amt_ms1·amt_gtr `medium-lean`. GPU 테스트(SW 업스트림 동등성 포함)는 전체 스위트에서 통과.<br>DESIGN 의 `+sep_audiosep` 은 SPEC A4 에서 "벤더링 경로가 실패할 때만 만든다"로 미뤘다. 벤더링 경로는 실패하지 않았고, SPEC §11 b15 에도 없어 세지 않았다. |
| b15b | S3.5 v0: 편성, 원시 전사, 보컬 활동, 반복 잔차 1차, B0 는 `--profile eval` 에서만 | PASS | `stages/s35/*/s35.json` 이 다음을 가리킨다: instrumentation, 원시 전사(MuScriptor bass·piano_other_presence·guitar, Basic Pitch guitar·bass), vocal_activity, resid1(`repeat_resid_pass1.wav`). quality 프로필에서는 `b0: null`.<br>eval 프로필의 `b0_guitar.mid` 는 `test_amt_stages.py` 로 확인했다. 실곡 B0 는 10-03 옛 코드 출력만 있다(오늘 재실행 안 함, GPU 비용). |
| b15c | `run --until notes`: 6 스템, nonvox, leftover, instrumentation.json, guitar_all.mid, bass_raw.mid | PASS | 두 곡 모두 스템 6개(float32 스테레오 44.1 kHz, 곡 길이), `nonvox.wav`, `leftover.wav`, `export/instrumentation.json` 이 있다.<br>`export/midi/guitar_all.mid`: SC 4,759음, 괴수의 꽃노래 2,307음.<br>`bass_raw.mid`: 911음 / 718음.<br>템포 맵은 133–225 BPM 이다. |
| b15d | 연습용 반주 (기타 뺀 믹스, 베이스 뺀 믹스) | PASS | 두 곡의 `export/practice/mix_minus_guitar.wav`, `mix_minus_bass.wav`, `bass_stem.wav` |
| b15e | `bench gpu`, VRAM 사전 점검·대기, Shared Usage 감시 | PASS | `bandscribe bench gpu --help`(`--rungs/--cap-mb/--ballast-mb/--fallback`), 벤치 기록 11건, b2 의 로그. doctor `pdh` OK. 오늘 beats 워커 기록: `pdh_self_shared_growth_mb` 2, `load_growth_mb` 0(수정 10) |
| b15f | Tier B 참 단독 트랙의 MuScriptor 참조 전사 | PASS | `bandscribe eval reftx` 명령. `data/eval/tierB/cambridge_mt/` 6곡, `*__reftx.json` 28개 |
| b15g | MuScriptor 지연 상수 | PASS | `amt.latency_s = -0.003` (defaults.toml, decisions.md `latency` 채택). 정확도 기준은 b10(FAIL) |
| b15h | E1–E3, E23 결정 | PASS | decisions.md: E1·E2·E3 판단 보류(Tier B 6곡, 기본값 유지), E23 채택.<br>주의: E1–E3 수치는 병합 전 참조로 쟀다(편향 가능, 2026-10-05 리뷰 주석). Tier A 서술은 GT 가 없어 빠졌다. |
| b15i | E24(SW·MuScriptor dtype) 결정 | **FAIL** | E24-sep·E24-amt 모두 실험 없이 `판단 보류 (M2 범위에서 미실행: GPU 40분 이상 비용)` 로 닫았다. 기록은 있지만 근거가 되는 측정이 없다. 실행에는 GPU 40분 이상이 들어 사용자 승인이 필요하다 |

## 공통

| id | 기준 | 판정 | 증거 |
|---|---|---|---|
| x1 | 전체 기본 `pytest` 통과 | PASS | `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` (건너뜀 사유를 보려고 `-rs` 추가) → **1346 passed, 2 skipped**, 7분 01초. 건너뛴 2건: 네트워크 slow 테스트, 이미 결정된 distractors |
| x2 | `bandscribe doctor` 통과 | PASS | `bandscribe.cmd doctor` → 22 checks, 0 fail, 2 warn, exit 0. 경고 2개: GPU 예산 2.0 GB(다른 앱이 VRAM 사용), 전역 Python 3.10 패키지 목록이 2026-09-30 기준과 다름(bandscribe 는 건드리지 않는다고 보고됨, 원인 미확인) |

## 알려진 한계와 다음 단계

1. **FAIL 해소 (M2):**
   - b10: 잠정 문턱 10 ms 를 다시 정할지, onset 을 더 다듬을지(M3 grid 의 onset 미세 보정) 사용자와 정한다.
   - b13b: E3b 를 실행한다(GPU 30분 이상, 먼저 승인).
   - b15i: E24 를 실행하거나(GPU 40분 이상, 승인), DESIGN·PROGRESS 에서 이유와 함께 뒤 마일스톤으로 옮긴다(그러면 DEFERRED).
2. **사용자에게 남은 것:**
   - b3: NVIDIA 설정을 바꿔 가며 두 번 실행한다.
   - b12·b13c·M1b: Tier A GP 파일을 준다.
   - Cambridge-MT `lines.yaml` 13개가 모두 `reviewed: false` 다. 귀로 확인한다(E1–E3 의 참조 라벨).
3. **측정 조건:** b1·b9 의 벤치는 앱을 켠 채, 리뷰 수정 전 코드로 쟀다. 앱을 닫고 지금 코드로 `bench gpu --backend sep,amt` 를 다시 재면 조건 주의를 지울 수 있다. 수정 10 의 로드 시 spill 검사도 함께 확인된다.
4. **리뷰에서 연기된 것 (PROGRESS 다음 단계 5·6):**
   - 섹션 경계 음표 채점: M3 첫 작업.
   - NVML 사전 점검이 GPU 락 밖에 있음: M4a.
   - 6번의 연기 근거 중 "워커 자신의 `mem_get_info` 검사"는 이 PC 에서 보호 장치가 못 된다. 벤치 기록의 워커 여유값은 다른 프로세스가 0.8–5.0 GB 를 써도 5,098 MB 로 같았다(WDDM, `gpu_common.py` 머리말에도 적혀 있다). 실제 보호는 사다리(OOM·shared 증가·로드 spill)다.
5. **검증 중 발견:**
   - SPEC §11 a9 의 `tests/test_pipeline_run.py -k determinism` 은 테스트를 0개 고른다. 테스트 이름이 `…_deterministic` 이라서다. `-k determinis` 로 고쳐야 한다.
   - stereo-preservation 의 `summary.json` 대표값은 빈 장면 3개를 섞은 평균이다(b8).
   - 캐시된 재실행은 sep 의 느림 경고와 격자 보정 메모는 다시 보여 준다. 그러나 괴수의 꽃노래의 notes 경고는 notes 를 계산한 실행에서만 나온다. 이 경고는 기타 뷰 음 28 %(893/3,200)를 electric_piano 로 보고 뺐다는 내용이다.
   - MuScriptor small 칸이 없다(HF 게이트 미수락). amt 사다리에는 medium-lean 아래 칸이 없다.
   - SC 의 `guitar_all.mid` 에는 144.4 s 에 6/4 마디가 하나 있다. 격자 보정("비트 추적기가 곡 중간에 박 단위를 바꿔…")의 흔적이다. M3 grid 반/배 검사에서 볼 것.
   - 괴수의 꽃노래 Beat This! 재실행의 `activations.npz` 는 이전 실행과 byte 단위로 같았다(GPU 결정성).
6. **다음:** 위 FAIL·BLOCKED 를 사용자와 정리한 뒤 커밋하고 `m2` 태그를 단다(PROGRESS 다음 단계 4). 인수 여부는 사용자가 판단한다.
