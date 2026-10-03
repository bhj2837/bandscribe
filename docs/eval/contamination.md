# 학습 데이터 오염 표 (DESIGN §8.1, §8.2)

> **규칙 (DESIGN 8.1):** 확인하지 못한 칸은 '불명'으로 두고, **불명인 세트로는 그 모델의 절대 성능을 주장하지 않는다.**
> '학습 포함' 세트의 수치는 참고용이다(판정 금지). 비교 실험은 두 모델이 모두 '학습 제외'인 세트에서만 판정한다.

- 칸 값은 `학습 포함` / `학습 제외` / `불명` 셋 중 하나로 시작한다(`gtab eval exp` 가 이 표를 읽는다. 괄호 안 id를 바꾸지 말 것).
- 근거는 각주 번호. 날짜는 확인한 날. 새 모델·세트를 넣으면 행·열을 추가하고 근거를 단다.
- 마지막 확인: 2026-09-30 (각주 7 추가 2026-10-03).

| 데이터셋 | MuScriptor-medium (`muscriptor-medium`) | Basic Pitch 0.4.0 (`basic-pitch`) | BS-RoFormer SW (`bs-roformer-sw`) | Beat This! final0 (`beat-this-final0`) |
|---|---|---|---|---|
| GuitarSet (`guitarset`) | 불명 [1] | 학습 포함 [2] | 불명 [3] | 학습 포함 [4] |
| EGDB (`egdb`) | 불명 [1] | 학습 제외 [2] | 불명 [3] | 학습 제외 [4] |
| IDMT-SMT-Bass-Single-Track (`idmt_bass_st`) | 불명 [1] | 학습 제외 [2] | 불명 [3] | 학습 제외 [4] |
| FiloBass (`filobass`) | 불명 [1] | 학습 제외 [2][5] | 불명 [3] | 학습 제외 [4] |
| Cambridge-MT (`cambridge_mt`) | 불명 [1] | 학습 제외 [2] | 불명 [3][7] | 학습 제외 [4] |
| MedleyDB (`medleydb`) | 불명 [1] | 학습 포함 [2] | 불명 [3] | 학습 제외 [4] |
| GOAT (`goat`) | 불명 [1] | 학습 제외 [2][5] | 불명 [3] | 학습 제외 [4] |
| musdb18 / musdb18-HQ (`musdb18`) | 불명 [1] | 불명 [6] | 불명 [3] | 학습 제외 [4] |
| MoisesDB (`moisesdb`) | 불명 [1] | 학습 제외 [2][5] | 불명 [3] | 학습 제외 [4] |

## 근거

1. **MuScriptor** — Rouard et al., "MuScriptor: An Open Model for Multi-Instrument Music Transcription", arXiv
   [2607.08168](https://arxiv.org/abs/2607.08168) (v2, 확인 2026-09-30) §3.1: 합성 사전학습(Lakh MIDI + 상용 MIDI 약 145만
   개를 사운드폰트로 합성), 실제 오디오 미세조정은 **내부 데이터셋 17만 곡(1.1만 시간, 대부분 오디오–악보 동기화)**,
   RL 사후학습은 그중 300곡. 곡 목록이 공개되지 않아 어느 공개 세트도 포함 여부를 확인할 수 없다 → 전부 `불명`.
   논문 Table 2의 벤치마크(Bach10, Dagstuhl ChoirSet, PHENICX-Anechoic, RWC)는 "두 모델의 학습 세트에 없는 세트"로
   골랐다고 적혀 있다(이 표에는 없는 세트). 모델 카드([huggingface.co/MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium))도
   세트 이름을 밝히지 않는다. DESIGN 8.2의 "GOAT·EGDB·GuitarSet in-distribution 가능성" 우려는 여전히 확인 불가.
   → **gate-m2(EGDB)는 사전 등록 규칙에 따라 `판단 보류 (오염 불명)`으로 기록된다.**
2. **Basic Pitch** — Bittner et al., "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and
   Multipitch Estimation", ICASSP 2022, arXiv [2203.09893](https://arxiv.org/abs/2203.09893), 그리고 Spotify 엔지니어링 글
   (확인 2026-09-30, 명세 검증 항목): 학습 세트 = GuitarSet, iKala, MAESTRO, MedleyDB-Pitch, Slakh. 이 목록에 없는 세트는
   `학습 제외`. MedleyDB는 MedleyDB-Pitch를 통해 `학습 포함`.
3. **BS-RoFormer SW** — 원 업로더 계정이 삭제되어 학습 데이터 기록이 없다(MVSEP 알고리즘 설명
   [mvsep.com/algorithms/77](https://mvsep.com/algorithms/77)에도 목록 없음, 확인 2026-09-30). 분리 모델은 흔히
   musdb18-HQ·MoisesDB·자체 멀티트랙으로 학습하지만 확인할 수 없다 → 전부 `불명`. 분리 품질(SDR)의 절대값은 주장하지 않는다.
4. **Beat This! final0** — README([github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)): final* 체크포인트는
   "GTZAN을 뺀 모든 데이터"로 학습. 데이터 목록([github.com/CPJKU/beat_this_annotations](https://github.com/CPJKU/beat_this_annotations),
   확인 2026-09-30): asap, ballroom, beatles, candombe, filosax, groove_midi, gtzan, **guitarset**, hainsworth, harmonix, hjdb,
   jaah, rwc, simac, smc, tapcorrect. → GuitarSet `학습 포함`, 표의 나머지 세트는 `학습 제외`. 코드와 가중치는 MIT.
   (GuitarSet 비트 정답으로 Beat This!를 평가하면 부풀려진다. M3 비트 지표는 Tier A·다른 세트로 본다.)
5. 세트 공개가 Basic Pitch 학습(2022) 뒤다: FiloBass 2023, GOAT 2024-25, MoisesDB 2023.
6. **musdb18** — 학습 목록에는 없지만 musdb18에는 MedleyDB 곡 46개가 들어 있고, 그중 일부가 MedleyDB-Pitch와 겹칠 수 있다.
   겹침을 곡 단위로 확인하기 전까지 `불명`.
7. **Cambridge-MT × 분리기** — musdb18의 100곡(DSD100, SiSEC 2016)은 Cambridge-MT("Mixing Secrets" 무료 멀티트랙 라이브러리)에서
   가져온 곡이다. musdb18-HQ로 학습한 분리기는 Cambridge-MT 곡 일부를 봤을 수 있으므로, 곡을 고를 때 musdb18 곡 목록과
   겹치는지 확인하고 겹치면 `lines.yaml` 의 메모에 적는다(확인 2026-10-03,
   [sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html): DSD100 100곡 + MedleyDB 46곡 + 기타 4곡). SW의 학습 목록이 없어
   칸은 계속 `불명` 이다. E1·E24-sep은 같은 분리기의 설정끼리 비교하므로 이 오염이 비교를 무효로 만들지는 않지만, 분리 품질의
   절대값은 주장하지 않는다.
