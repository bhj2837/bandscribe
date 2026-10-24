# 실험 등록부 (DESIGN §8.1, §8.6)

모든 선택지는 여기의 실험으로 관리한다. **실험 전에** 주 지표, 최소 검출 효과(MDE), 데이터, 분석 방법, 결정 규칙을 적는다.
**첫 실행 뒤에는 지표·MDE·데이터·규칙을 바꾸지 않는다.** 바꿔야 하면 새 id로 다시 사전 등록한다.

- 제목 줄 `## <id>` 의 id가 곧 `bandscribe eval exp <id>` 의 id다(정확히 같아야 한다).
- 상태: `사전 등록` → (첫 결과가 나오면 러너가 바꿈) `실행 중` → (사람이 확인 후) `결정: 채택|기각|판단 보류 (이유)`.
  러너는 `사전 등록`·`실행 중` 만 실행하고, `결정:` 항목은 거부한다(다시 보려면 새 id로 등록).
  데이터·모델이 없어 결과 없이 끝난 실행은 상태를 바꾸지 않는다. 데이터를 기다리기로 했으면
  `bandscribe eval exp <id> --record "판단 보류 (데이터 대기)"` 로 사람이 기록한다.
- `판정 설정` 의 `select=dev; split=test`: dev 에서 선택지 하나를 고르고 그 하나만 test 에서 확인한다(test 로 고르지 않음).
- `판정 설정` 의 `exclude=<접두어>[|<접두어>…]`(2026-10-05): 그 접두어로 시작하는 arm 은 점수를 보고만 하고 고르지 않는다
  (`rule=tuning` 과 `select=` 규칙; 기준 arm 은 늘 고를 수 있다). 선택지 정의가 "비교용으로만" 둔 arm 에 쓴다.
- `판정 설정` 줄은 러너가 읽는 기계 판독 설정이다(`키=값; …`). 단위는 점(F1 × 100).
- 분석 기본값: 곡(또는 연주자) 블록 부트스트랩 10,000회(seed 20260930), 후보 − 기준 쌍 비교, 95 % CI.
  채택 = CI가 0을 제외 **그리고** 점추정 ≥ MDE **그리고** 어느 시나리오 하위 세트도 CI 상한 < −2점이 아님(DESIGN 8.1).
  CI가 0을 포함하면 `판단 보류` 로 기록하고 기본값을 유지한다.
- 측정하는 GPU 백엔드는 실험마다 `force_rung` 으로 사다리 칸을 고정하고, 지표 행의 `rung` 열에 적는다. 칸이 다른 행은 합치지 않는다.
  `bandscribe run` 의 단계 캐시는 쓰지 않는다.
- 오염 규칙: `docs/eval/contamination.md` 에서 해당 모델 × 세트가 `학습 포함`·`불명` 이면 그 세트로 절대 성능을 주장하지 않는다.
- 지표 행(metrics.csv) 규약: `system` = 선택지(arm) 이름, `item` = 곡/트랙/창, `group` = 부트스트랩 블록(곡·연주자),
  `scenario` = 하위 세트 태그(예: `distortion`, `keys`), `section` = 분할(`dev`/`test`, 튜닝·지연 실험), `line` = 파트(`guitar`/`bass`).

## 지표와 분할 (중간 점검 5.2, 2026-10-24 — 이 블록은 실험이 아니다)

[`docs/MIDTERM_REVIEW.md`](MIDTERM_REVIEW.md) 5.2절의 규칙이다. 이 날짜 뒤에 등록하는 실험은 모두 이 규칙을 따른다.

- **판정 대상:** 품질 결정의 1차 대상은 악보곡(Tier A)이다. Tier B 는 해가 없는지 보는 확인용이고 팔 사이 차이로만 인용한다.
  모든 주장은 "Kessoku Band / 한 채보자" 범위로 적고, J-pop 은 늘 별도 표다.
- **집계:** 합친 기타·파트별·베이스 마디 F1 은 풀링(곡별 TP / FP / FN 합)이다. 계획 시작 값 66.8 / 55.9 / 81.5(곡 평균 66.9 / 55.4 /
  81.8 은 참고). 떠도는 오프셋 값도 풀링(합친 68.9, 베이스 85.6). 구절 비율은 모든 구절을 한데 센 비율(485개 중 26.6 %).
  "어느 곡도"로 시작하는 규칙만 곡별 값을 쓴다.
- **누적 바닥:** 시작 값에서 합친 −0.3, 파트별 −0.5, 베이스 −0.2, 구절 비율 −2점 → **66.5 / 55.4 / 81.3, 구절 24.6 %**.
  모든 채택이 실험별 허용 폭과 함께 이 바닥을 지킨다. 앞의 채택이 여유를 쓰면 뒤 실험의 허용 폭은 남은 여유로 줄어든다.
  3 / 3 분할이면 dev 와 test 의 각 시작 값에 같은 폭을 적용한다.
- **분할(마감 2026-11-15):** 그때까지 봉인 음원이 와서 판본 관문을 통과한 곡이 5곡 이상이면 dev = 지금 악보 6곡, test = 봉인 곡.
  아니면 6곡을 3 / 3 으로 잠근다: **test = SC, SNN, GKA / dev = ANB, HBT, HNN**("이미 본 곡" 단서). 3 / 3 이면 그 뒤의 개발(2-3,
  2-6, E43 특징 선택, E45 임계값, E46 팔 선택, 3-5)은 dev 3곡만 쓴다. 마감 뒤에 온 봉인 곡은 분할을 바꾸지 않고 추가 시험셋이 된다.
  (2026-10-24 현재: 분할 미커밋.)
- **이미 조정된 팔:** E45 임계값(0.3–0.7, 0.42)은 GKA·ANB·HBT·HNN 의 다운비트 4건에서, E46 최저음 팔은 HBT 에서, E42 는 18파트
  전체에서 왔다. 그 곡이 test 에 있으면 그 실험의 test 판정에서 빼거나 "표본 안, 잠정"으로 적고 봉인 곡이 오면 다시 판정한다.
- **봉인:** 분할을 커밋한 뒤에만 test 점수를 낸다. test 는 등록된 항목의 `bandscribe eval exp <id>` 로만 읽고, 읽을 때마다
  `data/eval/test_access.log` 에 남긴다(코드: `bandscribe.eval.sheets.open_test`, songs.yaml 의 `split`; 예산을 넘는
  새 보기는 거부한다). 보기 예산 4번: ① 분할 커밋 뒤 기준선, ② 2단계 묶음(E42 확인, E43), ③ 3단계 묶음(E45, E44,
  E46, E47, 준비됐으면 E21·E48), ④ 마지막 전체 확인. E21·E48 이 ③에 못 맞추면 예비 ⑤(미리 등록).
- **불확실성:** 곡 부트스트랩 CI. 6곡이면 합친 F1 ±4.5점이라 최소 검출 차이를 2–3점으로 잡고, 이득을 주장하는 채택 문턱을 그보다 낮게
  두지 않는다(E21·E48 은 +3). 3곡 test 의 CI 는 판정에 쓰지 않는다. 스트럼·구절 수를 n 으로 CI 를 내지 않는다.
- **등록 양식 추가 칸:** `이탈(deviation)` 과 `구절 비율 기대 효과(방향, 크기)`. 결과를 본 뒤 정답이나 규칙을 바꾸면 새 번호로, 새 곡에서
  판정한다.
- **실버 규칙:** 세션 1 은 dev 악보곡 중 사용자가 연습하지 않은 곡으로, 악보를 닫고 한다. 사용자 교정과 악보의 음 일치가 0.85 이상일 때만
  J-pop 실버 음 F1 을 헤드라인으로 쓴다. 아니면 평점과 분만 보고한다. 악보가 틀린 곳은 따로 표시한다.
- **임계값 검증:** "구절 F1 0.7 이상 = 연습 가능" 을 사용자 평점(1 그대로 / 2 두 군데 이하 수정 / 3 많이 손봄)과 비교한다. 악보가 있는 평점
  구절이 8개(32마디) 이상 모이면 판정한다(평점 1–2 와 일치 0.7 이상이면 유지). 세션 1(8마디)은 기록만 한다.

## 양식 (새 실험은 이 틀을 복사한다)

```
## E<번호>
- 제목:
- 상태: 사전 등록
- 질문:
- 선택지:
- 주 지표:
- MDE:
- 데이터(세트, 분할, 창 정의):
- 분석:
- 결정 규칙:
- 판정 설정: `rule=superiority; metric=onset_f1_50; baseline=<기준 arm>; mde=1.0`
- 비용:
- 실행 명령: `bandscribe eval exp E<번호>`
- 결과: (아직 없음)
```

---

## E1
- 제목: SW 입력 라우드니스 정규화 (원본 vs −14 LUFS vs −16 LUFS)
- 상태: 결정: 판단 보류 (기본값 off 유지; −14·−16 모두 CI 가 0 포함, 기타 Δ ≈ 0)
- 질문: 상용 마스터처럼 큰 믹스를 SW에 넣기 전에 라우드니스를 낮추면(분리 후 역게인) 스템 전사가 나아지는가?
- 선택지: `sep.input_lufs` = `off`(기준) | `-14` | `-16`. 게인은 SW 앞에서 곱하고 각 스템에 역수를 곱한다.
- 주 지표: SW 기타·베이스 스템에 MuScriptor(medium, `force_rung=medium`)를 돌린 onset F1(50 ms, pitch 50 cents),
  **참 단독 트랙의 참조 전사(reftx)** 대비. 기타·베이스 행을 풀링(`line` 열로 하위 보고).
- MDE: +1.0점
- 데이터(세트, 분할, 창 정의): Tier B(Cambridge-MT 2곡 이상, 승인 시 MedleyDB)를 `tierb_mix(master_lufs=-8)`(결정적 리미터)로
  상용 마스터처럼 다시 섞은 곡. 창 = 곡 전체(곡 = 블록). Tier A 2–3곡은 서술만. SDR은 진단 열.
- 분석: 곡 블록 부트스트랩 10k, 각 후보 − `off` 쌍 비교, 하위 세트 = `scenario`(곡 장르 태그가 있으면).
- 결정 규칙: DESIGN 8.1 채택 규칙. 두 후보가 모두 채택이면 점추정이 큰 쪽.
- 판정 설정: `rule=superiority; metric=onset_f1_50; baseline=off; mde=1.0`
- 비용: SW 곡당 3회 + MuScriptor 곡당 6뷰. 실행 시간 추가 없음(설정값 변경).
- 실행 명령: `bandscribe eval exp E1`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전(지표·MDE·결정 규칙은 그대로). (1) 지연 상수 `amt.latency_s` 는 추정 전사와 reftx 참조에 **똑같이** 뺀다. 둘 다 MuScriptor 원시 출력이라 같은 지연을 가지므로, 추정에만 빼면 모든 arm 이 상수만큼 참조와 어긋난다(실행 전에 `_refs_by_kind` 를 고침). (2) Tier B = Cambridge-MT 6곡(ButterflyEffect, JetB, Mistrusted, YoungGriffo, Secretariat, Woodfire). `lines.yaml` 은 사람이 검토하기 전(`reviewed: false`)이고 MedleyDB 는 없다.
  Tier A 는 GT 가 없어 이 실험에서 서술하지 않는다(E2/E3 의 Tier A 서술 표 참고).
- 결과:
  - 2026-10-04 `20261003-215706_1e422dc_exp-e1`: 계산된 판정 = 판단 보류 (기본값 off 유지)
  - 요약(실험자, 2026-10-04): Tier B 6곡, `tierb_mix(master_lufs=-8)`, rung `chunk=588800|medium-lean`. 기타+베이스 풀링
    onset F1(곡 블록 10k): off 50.9; −14 Δ −2.34 [−3.49, +0.04]; −16 Δ −0.76 [−1.20, +0.01]. 두 후보 모두 CI 가 0 을 포함하고
    점추정은 음수 → 채택 없음. 기타만: −14 Δ −0.09 [−0.19, +0.05], −16 Δ +0.01 [−0.00, +0.02](사실상 같음). 곡별 기타 F1 은
    세 arm 이 소수점 첫째 자리까지 거의 같다(예: Woodfire 78.1/78.1/78.1).
  - 베이스 풀링 값(off 31.6)은 **Mistrusted 한 곡의 퇴화 출력**이 지배한다. SW 베이스 스템에 MuScriptor 가 146 초 동안
    12,132–14,809개(초당 최대 330–370개)의 "electric_bass" 음을 냈고, 그중 9,758–12,014개가 MIDI 60 위다(참조 277개). 디코딩이
    반복 루프에 빠진 것이다. 이 곡을 빼면 베이스 F1 은 세 arm 이 같다(70.2/96.7/93.7/99.3/91.1, 곡마다 ±0.1). 베이스 Δ 가
    음수인 것도 이 곡의 오답 수가 arm 마다 달라서다(12,110 / 14,787 / 12,984).
  - 결론: 라우드니스 정규화는 SW→MuScriptor 전사에 측정 가능한 이득이 없다. 기본값 `sep.input_lufs = "off"` 를 유지한다.
    한계: 6곡뿐이고, 참조는 MuScriptor 파생(reftx)이며, SDR 진단 열은 이 러너가 내지 않는다. 퇴화 출력은 별도 문제로 보고한다
    (PROGRESS.md: 같은 현상이 Tier A 의 `piano_other_mono` 패스에서도 보인다. 青と夏 35,010개, 평균 초당 128개).
  - 주의(2026-10-05 리뷰, 수치는 고치지 않음): 위 수치는 **병합하지 않은 참조**로 쟀다. 기타 Line 들의 reftx 를 그냥 이어 붙여,
    두 Line 이 함께 친 음(같은 음높이·50 ms 안)이 두 번 세어졌고(6곡 22,382음 중 약 11 %, Mistrusted 23 %), 한 onset 에 기타 클래스
    두 개가 붙은 중복도 그대로였다. 그래서 재현율에 상한이 생기고 arm 사이 차이도 편향될 수 있다. 결정은 판단 보류라 바뀐 기본값은
    없다. 이후 실행은 참조·추정을 병합해 잰다(`bandscribe.amt.experiments.merged`, E3b·E24-sep 의 사전 등록 보완).

## E2
- 제목: MuScriptor 기타 입력 뷰
- 상태: 결정: 판단 보류 (기본값 guitar_mono 유지; 세 후보 모두 CI 가 0 포함, 점추정 음수)
- 질문: 기타 전사에 어떤 오디오를 넣어야 하는가?
- 선택지: `amt.guitar_view` = `guitar_mono`(기준) | `mix_mono` | `nonvox_mono` | `guitar_other_mono`
- 주 지표: 기타 클래스 onset F1(50 ms) — Tier B는 reftx 대비(판정), Tier A는 GT 대비(서술).
  reftx는 MuScriptor 파생이므로 **입력끼리의 비교**에만 유효하고 절대 성능 주장에는 쓰지 않는다.
- MDE: +1.0점 (선택지가 전사 패스를 하나 더 늘리면 +1.5점)
- 데이터(세트, 분할, 창 정의): Tier B 곡(E1과 같은 리믹스), Tier A 2–3곡. 창 = 곡 전체.
- 분석: 곡 블록 부트스트랩 10k, 후보 − `guitar_mono`.
- 결정 규칙: DESIGN 8.1 채택 규칙.
- 판정 설정: `rule=superiority; metric=onset_f1_50; baseline=guitar_mono; mde=1.0`
- 비용: MuScriptor 곡당 4뷰.
- 실행 명령: `bandscribe eval exp E2`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전(지표·MDE·결정 규칙은 그대로). (1) 지연 상수 `amt.latency_s` 는 추정 전사와 reftx 참조에 **똑같이** 뺀다. 둘 다 MuScriptor 원시 출력이라 같은 지연을 가지므로, 추정에만 빼면 모든 arm 이 상수만큼 참조와 어긋난다(실행 전에 `_refs_by_kind` 를 고침). (2) Tier B = Cambridge-MT 6곡(ButterflyEffect, JetB, Mistrusted, YoungGriffo, Secretariat, Woodfire). `lines.yaml` 은 사람이 검토하기 전(`reviewed: false`)이고 MedleyDB 는 없다.
  Tier A(`D:/backing` 의 5곡)는 GP 정답이 없어 판정 행에 넣지 않고, 변형별 음 수·음역·클래스 활동·변형 간 일치도만 서술한다.
- 결과:
  - 2026-10-03 `20261003-202931_1e422dc_exp-e2`: 계산된 판정 = 판단 보류 (기본값 guitar_mono 유지)
  - 요약(실험자, 2026-10-03): Tier B 6곡(Cambridge-MT, `tierb_mix(master_lufs=-8)`), rung `chunk=588800|medium-lean`, 모든 뷰
    guitar-only 마스크. 기타 onset F1(50 ms, reftx 대비, 곡 블록 10k): 기준 `guitar_mono` **60.6 [52.3, 72.6]**;
    `guitar_other_mono` 60.3, Δ −0.28 [−0.80, +0.43]; `nonvox_mono` 59.2, Δ −1.39 [−5.56, +1.60]; `mix_mono` 53.4,
    Δ −7.17 [−20.2, +1.21]. 세 후보 모두 CI 가 0 을 포함하고 점추정이 음수 → 채택 없음.
  - 곡별(guitar / guitar+other / nonvox / mix): ButterflyEffect 49.2/49.0/48.6/52.7, JetB 65.8/66.9/67.5/42.7,
    Mistrusted 48.0/47.9/51.5/48.6, Secretariat 65.6/63.9/64.5/61.3, Woodfire 78.1/78.1/67.3/52.8, YoungGriffo 75.6/75.6/77.4/66.9.
    mix 는 재현율이 오르지만(다른 악기 음을 기타로 받아 적음) 정밀도가 크게 떨어진다(JetB fp 3,385 vs 964). nonvox 는 곡마다
    엇갈린다(Woodfire −10.8, Mistrusted +3.5).
  - 한계: 6곡뿐이라 CI 가 넓다. reftx 는 MuScriptor 파생이라 입력끼리의 비교에만 유효하고 절대 성능은 주장하지 않는다.
    `lines.yaml` 은 사람이 검토하기 전이다. Tier A 서술은 `data/scratch/exp/tierA/` (커밋 금지) 와 PROGRESS.md 참고.
  - 주의(2026-10-05 리뷰, 수치는 고치지 않음): 위 수치는 **병합하지 않은 참조**로 쟀다(E1 결과의 주의와 같다: 여러 기타 Line 이
    함께 친 음이 두 번 세어져 재현율 상한·arm 간 편향). 결정은 판단 보류라 바뀐 기본값은 없다.

## E3
- 제목: `instruments` 하드 마스크 정책 + S3.5 존재 확률 문턱
- 상태: 결정: 판단 보류 (기본값 guitar+present·문턱 0.5 유지; test 정보 1곡, 합성 장면은 SW 기타 스템이 비어 정보 없음)
- 질문: 기타 뷰 전사에 건넬 클래스 마스크를 무엇으로 하고, 편성 추정 문턱을 얼마로 할 것인가?
- 선택지: `amt.guitar_mask` = `guitar+present`(기준) | `guitar_only` | `all`; `instr.threshold` ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
  (arm 이름: 정책, 문턱이 기본 0.5가 아니면 `정책|t=문턱`)
- 주 지표: 건반·신스가 있는 항목에서 기타 클래스 onset F1(50 ms). **제약: 기타 클래스 누락 = 0건**(누락 행이 하나라도 있는
  선택지는 채택 불가). 함께 보고: b13 표(추정 편성 vs `families_present`, 곡별).
- MDE: +1.0점
- 데이터(세트, 분할, 창 정의): 건반 포함 Tier B 곡, 합성 건반 블리드 장면(`full_band`), Tier A. 문턱은 dev에서 제약 아래 F1 최대,
  test에서 확인(`section` = dev/test).
- 분석: dev에서 제약(누락 0)을 지킨 선택지 중 풀링 F1이 가장 큰 하나를 고르고(같으면 기본값), **그 하나만** test에서
  `guitar+present` 와 쌍 비교한다(곡 블록 부트스트랩 10k). test 결과로 선택지를 고르지 않는다.
- 결정 규칙: DESIGN 8.1 채택 규칙 + 누락 제약(test 행에서도 확인). dev 최적이 기본값이면 기각(기본값 유지).
- 판정 설정: `rule=superiority; metric=onset_f1_50; baseline=guitar+present; mde=1.0; constraint=guitar_class_omissions==0; split_col=section; select=dev; split=test`
- 비용: 추가 패스 없음.
- 실행 명령: `bandscribe eval exp E3`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전(지표·MDE·결정 규칙은 그대로). (1) 지연 상수 `amt.latency_s` 는 추정 전사와 reftx 참조에 **똑같이** 뺀다. 둘 다 MuScriptor 원시 출력이라 같은 지연을 가지므로, 추정에만 빼면 모든 arm 이 상수만큼 참조와 어긋난다(실행 전에 `_refs_by_kind` 를 고침). (2) Tier B = Cambridge-MT 6곡(ButterflyEffect, JetB, Mistrusted, YoungGriffo, Secretariat, Woodfire). `lines.yaml` 은 사람이 검토하기 전(`reviewed: false`)이고 MedleyDB 는 없다.
  dev/test 는 등록대로 항목 id 해시(seed 20260930): test = Secretariat, Woodfire(건반 없음 → E3 항목 아님), synthetic A:s1·B:s1·twin:s0. Tier A 는 사람이 정한 편성 GT 가 없어 b13 표에 추정값만 서술한다.
- 결과:
  - 2026-10-04 `20261004-000848_1e422dc_exp-e3`: 계산된 판정 = 판단 보류 (기본값 guitar+present 유지)
  - 요약(실험자, 2026-10-04): 항목 = 건반·신스가 있는 Tier B 3곡(JetB 피아노, Mistrusted 스타일로폰, Secretariat 해먼드) +
    합성 `full_band` 장면 8개. rung `chunk=588800|medium-lean`. 기타 클래스 누락 = **0건**(모든 arm·문턱, 제약 충족).
    dev 선택: `all` 56.780 ≈ `guitar_only` = `t=0.6` = `t=0.7` 56.775 > 기준 `guitar+present` 53.21(0.005점 차라 사실상 동점).
    test 확인(`all` − 기준): Δ −3.60 [−3.70, 0.00] → 판단 보류. test 에서 정보가 있는 항목은 Secretariat 한 곡뿐이다
    (`all` 61.9 vs 65.6).
  - **합성 장면은 정보가 없다**: 8개 중 7개에서 SW 기타 스템 전사가 0음이다(같은 장면의 믹스 패스는 distorted guitar 91음을
    찾음). 합성 기타(배음 톤 + 앰프 모델)를 SW 가 기타 스템으로 보내지 않는다. 이 장면들은 모든 arm 에서 F1 0 이라 비교에
    기여하지 않는다(A:s0 77.7, P70:s1 16.4 만 값이 있음).
  - dev 관찰(판정 아님): JetB 에서 피아노가 '있음'으로 잡히면 `guitar+present` 마스크에 건반 클래스가 들어가고, 기타 F1 이
    58.8 로 떨어진다(`guitar_only` 65.8, `all` 66.7). MuScriptor 가 기타 음 일부를 건반 클래스로 보내기 때문이다. 문턱 0.6 이상이면
    피아노가 빠져 65.8 이 된다. test(Secretariat)는 해먼드를 못 잡아 `guitar+present` = `guitar_only` 라 이 차이를 확인할 수 없었다.
  - b13 표(추정 가족 vs `families_present`, 문턱 0.5): JetB keys 맞음 / Mistrusted synth **놓침**(스타일로폰) / Secretariat keys
    **놓침**(해먼드 → SW other 스템 → MuScriptor 가 distorted guitar 로 적음) / 합성 8장면 keys(패드) **모두 놓침**.
    Tier A 는 편성 GT 가 없어 추정값만 PROGRESS.md 에 서술한다.
  - 결론: 기본값 `amt.guitar_mask = "guitar+present"`, `instr.threshold = 0.5` 를 유지한다. 다시 보려면 새 id 로 사전 등록해야 한다.
    조건은 건반이 실제로 검출되는 Tier B 곡을 더 넣고, SW 가 기타로 분리하는 합성 장면을 쓰거나 합성 장면을 빼는 것이다.
    근거: `data/runs/20261004-000848_1e422dc_exp-e3/`.
  - 주의(2026-10-05 리뷰, 수치는 고치지 않음): Tier B 행은 **병합하지 않은 참조**로 쟀다(E1 결과의 주의와 같다: 여러 기타 Line 이
    함께 친 음이 두 번 세어져 재현율 상한·arm 간 편향). 결정은 판단 보류라 바뀐 기본값은 없다.

## E3b
- 제목: 표본 존재 패스 기준의 `instruments` 하드 마스크 정책 + 존재 문턱 (E3 재등록)
- 상태: 사전 등록
- 질문: 2026-10-04 속도·존재 수정 이후 `bandscribe run` 이 실제로 쓰는 편성 근거(표본 piano+other 존재 패스, 믹스 패스 없음)에서
  기타 뷰 마스크를 무엇으로 하고 문턱을 얼마로 할 것인가? 현악기를 마스크에 넣는 것(`guitar+present+strings`)이 도움이 되는가?
- 선택지: `amt.guitar_mask` = `guitar+present`(기준, 건반·신스만) | `guitar_only` | `all` | `guitar+present+strings`;
  `instr.threshold` ∈ {0.3, 0.4, 0.5, 0.6, 0.7} (`guitar+present` 만; arm 이름은 E3 와 같은 규칙)
- 주 지표: 건반·신스가 있는 항목에서 기타 클래스 onset F1(50 ms). **제약: 기타 클래스 누락 = 0건**. 함께 보고: b13 표
  (keys·synth·strings 가족 추정 vs `families_present`).
- MDE: +1.0점
- 데이터(세트, 분할, 창 정의): E3 와 같다(건반 포함 Tier B 곡, 합성 `full_band` 장면, 분할 = 항목 id 해시 seed 20260930).
  존재 패스 창은 `bandscribe run` quality 설정(최대 6 × 10 s, 60 s, 곡의 20 %)을 2 s 의사 마디에서 고른다(구간 정보 없음).
  E3 에서 합성 장면 7/8 이 SW 기타 스템 0음이라 정보가 없었다: 실행 전에 건반이 실제로 검출되는 Tier B 곡을 더 넣는다.
- 분석: E3 와 같다(dev 에서 제약을 지킨 최대 F1 선택지 하나를 test 에서 `guitar+present` 와 쌍 비교, 곡 블록 부트스트랩 10k).
- 결정 규칙: DESIGN 8.1 채택 규칙 + 누락 제약. dev 최적이 기본값이면 기각(기본값 유지).
- 판정 설정: `rule=superiority; metric=onset_f1_50; baseline=guitar+present; mde=1.0; constraint=guitar_class_omissions==0; split_col=section; select=dev; split=test`
- 비용: 항목마다 piano+other 존재 창 전사(최대 60 s 오디오) + 정책·문턱마다 기타 뷰 전사. GPU 30분을 넘으면 사용자에게 먼저 묻는다.
- 실행 명령: `bandscribe eval exp E3b`
- 등록 근거(2026-10-04 리뷰): E3 는 믹스 패스로 편성을 추정해 지금의 `bandscribe run` 을 재지 않는다. 실곡 3곡에서 현악기를 넣은
  마스크는 백킹 대조 오검출을 줄이지 못했고(怪獣の花唄 17.7 → 17.7 %, 青と夏 15.3 → 20.2 %) 기타 하위 클래스 라벨만 뒤섞었다.
  그래서 기본값은 `guitar+present`(건반·신스)로 되돌렸고, 현악기 변형은 이 실험이 정답 데이터로 판정할 때까지 켜지 않는다.
- 사전 등록 보완: 2026-10-05, 첫 실행 전(2026-10-05 리뷰). **참조를 기타 Line 들에 걸쳐 병합한다**(reference merged across guitar
  lines): 한 곡의 기타 Line reftx 를 합친 뒤 같은 음높이·onset 50 ms 안의 음은 하나로 센다(`eval.suites._merged_arrays` 규칙,
  허용폭 = 지표의 onset 허용폭 50 ms). 추정도 같은 규칙으로 병합한다(한 onset 에 기타 클래스 두 개가 붙은 중복 포함).
  합성 장면의 GT 도 같다. 이유: 이어 붙이기만 하면 두 Line 이 함께 친 음(Tier B 6곡 reftx 22,382음 중 약 11 %, Mistrusted 23 %)이
  두 번 세어져, 기타 스템 하나를 전사한 추정의 재현율에 상한이 생기고 arm 사이 차이가 편향된다(E1·E2·E3 의 기록 수치가 그 상태).
  주 지표 이름·MDE·데이터·결정 규칙은 그대로다.
- 결과: (아직 없음)

## E3c
- 제목: 기타 패스 마스크 기본값 단순화 — `guitar_only` vs `guitar+present` (보류 구간 확인)
- 상태: 결정: 기각 (guitar_only − production Δ −2.35 [−8.22, +0.47]점, 비열등 한계 −1.0 미달; 기본값 guitar+present 유지)
- 질문: `distractors` 측정(서술)에서 `guitar+present` 마스크는 비기타 클래스를 넣은 13조건에서 비기타로 빠진 음이 0개였고
  F1 만 낮췄다(평균 −2.1점). 그 측정에 **쓰지 않은** 구간에서도 더 단순한 `guitar_only` 가 `guitar+present` 보다 나쁘지 않은가?
- 선택지: `production`(= 현재 기본값 `amt.guitar_mask = "guitar+present"`, 표본 존재 패스 + `instrumentation.estimate`) |
  `guitar_only`(같은 기타 뷰, 기타 3클래스만).
- 주 지표: `full` 조건(모든 스템)에서 기타 onset F1(`note_f1_onset50`, 50 ms / 50 cents), 참 기타 스템 합의 MuScriptor 전사 대비
  (`distractors` 와 같은 참조·지연 처리). 참조가 MuScriptor 파생이므로 arm 끼리의 차이만 해석한다.
- MDE: 해당 없음(비열등). 비열등 한계 −1.0점.
- 데이터(세트, 분할, 창 정의): `distractors` 10곡 중 그 실행의 창과 **겹치지 않는** 75 s 창. 고정 규칙(결과 보기 전):
  기타 활성 ≥ 50 %, 건반·오르간·신스·현악 범주 중 하나 이상 ≥ 10 % 활성(`distractors` 의 활성 규칙), 곡마다 그 범주 점유 합이
  가장 큰 창 1개. 조건을 만족한 곡 5곡: JamesElder 35–110 s, JetB 5–80 s, Moosmusic 0–75 s, Secretariat 145–220 s, Zeno 0–75 s
  (`bandscribe.amt.experiments.E3C_WINDOWS`). 조건 = `base` + `full`(+범주 조건 없음, null 없음). 한계: 같은 곡의 다른 구간이라
  완전히 독립된 데이터는 아니다(곡 단위 특성 공유).
- 분석: 곡 블록 부트스트랩 10k(seed 20260930), `guitar_only` − `production` 쌍 비교(5곡). 함께 보고: 마스크가 실제로 달라진 창 수,
  `non_guitar_dropped`(제품 마스크로 비기타 라벨이 붙어 빠진 음 수), 창별 F1.
- 결정 규칙: 비열등 + 단순화. `guitar_only` − `production` 의 95 % CI 하한 > −1.0점 **그리고** `guitar_only` 가 마스크의 비기타
  클래스 수를 줄였으면(`mask_non_guitar` 평균 100 % 감소; 제품 마스크가 모든 창에서 비기타 클래스를 넣지 않았으면 정보 없음 →
  판단 보류) 채택. 하한 ≤ −1.0 이면 기각(기본값 유지).
- 판정 설정: `rule=noninferiority; metric=note_f1_onset50; baseline=production; margin=-1.0; resource=mask_non_guitar:-100%`
- 비용: 분리 10조건 × 75 s + MuScriptor(참조 5, base 5, full × 2 arm) ≈ GPU 20분.
- 실행 명령: `bandscribe eval exp E3c`
- 등록 메모(2026-10-05, 실행 전): `distractors` 결과를 보고 세운 가설을 그 측정에 쓰지 않은 구간으로 확인하는 실험이다.
  채택되면 `amt.guitar_mask` 기본값을 `guitar_only` 로 바꾸고, 정책 전반(현악 포함 변형·문턱)은 E3b 로 남긴다.
- 결과:
  - 2026-10-05 `20261005-130211_b50ea18_exp-e3c`: 계산된 판정 = 기각 (기본값 production 유지)
  - 요약(2026-10-05): 5개 창 중 제품 마스크가 비기타 클래스를 넣은 창은 2개뿐이고, 나머지 3개(JamesElder·Moosmusic·Secretariat)는
    두 arm 이 같은 마스크·같은 출력이다. **JetB 5–80 s**: 제품 마스크(건반 4클래스)가 191음을 비기타로 빼서 FP 357 → 145,
    F1 61.9(guitar_only) vs **76.3(production)**. **Zeno 0–75 s**: 마스크에 electric_piano 만 들어가 빠진 음 0개, FP 42 → 55,
    F1 91.0 vs 90.0. 풀링 `guitar_only` − `production` = −2.35 [−8.22, +0.47]점 → 하한이 −1.0 아래라 비열등 실패 → 기각.
  - 해석: `distractors` 의 "제품 마스크는 비기타 음을 0개 거른다"는 관찰은 **일반화되지 않았다.** 피아노가 뚜렷한 JetB 구간에서는
    건반 마스크가 피아노 블리드를 실제로 걸러 F1 을 14점 올렸다. 마스크의 효과는 곡·구간마다 다르다(걸러내기도, 출력만
    흔들기도 한다). 기본값 `guitar+present` 를 유지하고, 정책 전반은 곡 수를 늘린 E3b 로 판정한다. 한계: 5곡, 실제로 갈린 창 2개.

## distractors
- 제목: 방해 소리별 기타 전사 오류 측정 (서술, 원인 귀속)
- 상태: 결정: 판단 보류 (서술 실험이라 판정 없음. 마스크를 기타만으로 바꾸자는 가설은 E3c 에서 기각됨)
- 질문: 밴드 믹스에서 기타 전사의 오검출(FP)과 누락(FN)은 어떤 종류의 소리(신스 패드·리드, 피아노·일렉트릭 피아노·오르간,
  현악, 효과음 등) 때문에 얼마나 생기는가? 분리(SW)에서 각 소리가 기타 스템으로 얼마나 새는가? (악기 마스크를 바꾸면 블리드가
  줄기보다 출력이 뒤섞인다는 2026-10-04 발견 뒤, 이후 결정을 이 측정으로 하기 위해)
- 선택지: 없음(측정). 조건 = `base`(기타·드럼·베이스·보컬) / `+<범주>`(base + 방해 범주 하나) / `full`(모든 스템).
  시스템 arm = `production`(제품 경로: 표본 존재 패스 + `instrumentation.estimate` 마스크, 설정의 `amt.guitar_mask`),
  진단 arm = `guitar_only`(같은 기타 뷰, 기타만 마스크; 소리 효과와 마스크 효과를 가르려고).
- 주 지표: 범주마다 `+<범주>` − `base` 의 FP/분, FN/분, 재현율·정밀도·F1(점), `production` arm, 곡 블록 부트스트랩 95 % CI
  (10,000회, seed 20260930). 함께 보고: FP·FN 원인 라벨 분포(조건 × 원인), 원인 라벨 × 대역 에너지 1위 소리, 분리 누출(dB)과
  기타 스템 비중, 곡별 표, `full` − `base`, "먼저 고칠 것" 순위.
- MDE: 없음(서술). 리포트에서 "효과 있음"이라고 쓰는 기준은 CI 가 0 을 제외하는 것. 곡이 1개뿐인 범주는 CI 없이 서술만 한다.
- 데이터(세트, 분할, 창 정의): Cambridge-MT 멀티트랙 중 방해 범주 스템이 있는 곡: 기존 6곡 중 4곡(ButterflyEffect SFX, JetB
  피아노·SFX, Mistrusted 스타일로폰, Secretariat 해먼드) + 이 등록과 함께 받은 7곡(JamesElder 신스 패드·리드·FX, Jokers 신스
  패드·리드, Zeno 신스 패드·피아노, RememberDecember 현악·서브드롭, SigneJakobsen 스트링 패드·오르간·피아노, Mu 로즈·신스,
  Moosmusic 로즈; registry.toml). 스템 범주 = 파일 이름 규칙(`bandscribe/eval/stem_taxonomy.py`, 디스크의 293개 이름 모두 분류,
  미분류 0). 곡당 75 s 창 최대 2개(`bandscribe/eval/distractors.py`): 5 s 간격 후보 중 기타 활성 ≥ 50 % 이고 방해 범주가
  ≥ 20 % 활성(‑30 dB, 기타는 ‑20 dB, ‑60 dBFS 이상)인 창, 첫 창은 활성 범주가 가장 많은 창, 두 번째 창은 겹치지 않고 새 범주를
  더하는 창("있음" 기준은 아래 보완 참고). 조건 오디오 = 스템 단위 합(멀티트랙 자체 밸런스) × `full` 의 마스터링 게인 곡선(‑8 LUFS, ‑1 dBTP 리미터, 모든
  조건에 같은 곡선). 분할 없음(서술).
- 참조: 참 기타 스템 합(같은 게인)의 MuScriptor 전사(기타만 마스크, top rung). 참조가 MuScriptor 파생이므로 절대 성능을 주장하지
  않고 조건끼리의 차이만 해석한다.
- 분석: mir_eval onset 50 ms / pitch 50 cents, 지연 상수 `amt.latency_s` 를 추정·참조 모두에. 원인 규칙(`attribution.py`,
  첫 규칙 우선, 보완 2 참고): FP = 같은 음높이 참조 음이 겹치거나 200 ms 안(`guitar_timing`) → 그 음의 대역(±50 cents,
  STFT 8192/1024) 에너지 1위가 비기타 참 스템이면 그 범주 → ±12/19/24 반음 참조 음이 겹침(`guitar_octave`) → ±1/2 반음
  (`guitar_near`) → 기타가 1위면 `guitar_unref`, 바닥 아래면 `unknown`. FN = 추정 음과의 같은 관계(`est_*`) → 가장 센 비기타 범주가
  기타 에너지 ‑6 dB 이상이면 그 범주, 아니면 `guitar_clear`. 누출 = 범주가 80 % 이상 차지하는 시간-주파수 칸에서 SW 기타 스템
  에너지 / 참 에너지(dB). 분리는 조건들을 무음(≥ 1 청크, 청크 스텝 정렬)으로 띄워 한 워커에 넣는다(각 조건이 혼자 분리될 때와
  같은 청크 격자). 모든 GPU 작업은 워커·GPU 락·내용 주소 캐시를 거친다.
- 결정 규칙: 서술(판정 없음). 결과는 "먼저 고칠 것" 순위로 요약하고, 마스크·S7 블리드 감점 같은 이후 결정은 이 결과를 근거로 새
  E# 로 사전 등록한다.
- 판정 설정: `rule=descriptive; metric=fp_per_min; suite=distractors`
- 비용: GPU 60분 이내(`--arg gpu_budget_min`, 예산을 넘으면 두 번째 창부터 뺀다; 캐시된 작업은 0).
- 실행 명령: `bandscribe eval run --suite distractors`
- 등록 메모(2026-10-04, 첫 실행 전): 이 등록으로 Tier B `cambridge_mt` 가 13곡이 됐다(새 7곡은 `required = false` 부분).
  아직 실행 전인 E3b·E24-sep 의 등록 문구는 6곡을 전제로 하므로, 그 실험을 돌릴 때 곡 목록을 등록 문구에 맞춰 고르거나
  새 곡을 넣는다고 먼저 보완 기록할 것.
- 사전 등록 보완: 2026-10-04, 첫 실행 전(구간 계획만 보고, 전사·지표는 보기 전). 방해 범주 "있음" 기준을 창 프레임의
  ≥ 20 % → ≥ 10 % 로 낮춘다. 20 % 로는 띄엄띄엄 나오는 소리(Jokers 신스 리드 19 %, Signe 오르간 16 %, Mu 로즈 13 %,
  Mistrusted 스타일로폰)가 빠지고 두 번째 창이 하나도 생기지 않았다. 띄엄띄엄 나오는 소리의 효과가 구간 1분당으로 묽어지지
  않게, 범주 표에 "그 소리가 활성인 1분당" ΔFP·ΔFN 도 함께 보고한다(활성 = 참 스템 기준 위 규칙).
- 사전 등록 보완 2 (사후 보완, 2026-10-05 리뷰로 표시): **엄밀한 사전 등록이 아니다.** 등록부에 기록하지 않은 1곡 시험 실행의 FP
  원인 결과를 본 뒤 고친 규칙이다. 2026-10-04, 1곡 시험 실행(Secretariat, 등록부 미기록) 뒤·전체 실행 전. 주 지표(ΔFP/분·ΔFN/분·재현율·F1)와
  데이터는 그대로이고 **FP 원인 라벨 규칙만** 고친다. 시험 실행에서 해먼드 오르간을 더하자 늘어난 FP 의 대부분이 `guitar_octave`
  로 찍혔는데, 그중 70/104개는 그 음높이 대역을 가장 크게 울린 소리가 오르간이었다(건반은 기타와 같은 화성을 쳐서 옥타브·5도
  관계가 흔하다). 그래서 (1) `guitar_timing` 다음에 "대역 1위가 비기타 소리면 그 소리"를 두고 그 뒤에 옥타브·반음 규칙을 둔다,
  (2) 2배음 대역은 MIDI 52 미만 음에만 더한다(그 위에서는 한 옥타브 **위**에서 울리는 소리를 원인으로 잘못 짚는다),
  (3) 바닥에 "그 음 구간 믹스 전체 전력의 ‑40 dB" 국소 바닥을 더한다(큰 화음의 onset 번짐이 엉뚱한 음높이를 설명하지 않게).
  분리 묶음 처리 확인(같은 조건 오디오, 혼자 분리 vs 무음으로 띄워 묶어 분리, 기타 스템): 안쪽 51.5 dB, 가장자리 7 s 26.4 dB,
  전체 34.5 dB SNR(가장자리 차이는 reflect 대 0 패딩).
- 사후 진단(사전 등록 밖, 첫 전체 결과를 본 뒤 추가): `base_null` 조건. 첫 결과에서 신스 리드·`synth_other`·로즈를 더하자 FP 가
  오히려 분당 50–80개 **줄었고**, 기타만 마스크 arm 에서도 같았다(마스크 탓이 아님). 같은 SW 기타 뷰끼리 22 dB SNR 차이에서
  MuScriptor 음 수가 1,028 → 711 로 바뀐 경우가 있어, 디코딩이 입력의 작은 변화에 크게 흔들리는지 따로 재야 범주 효과를
  해석할 수 있다. 그래서 구간마다 기본 조건의 기타 뷰에 ‑90 dBFS RMS 백색 잡음(시드 고정, 들리지 않음)만 더해 같은 마스크로
  다시 전사하고 `base_null` − `base` 를 "잡음 기준"으로 보고한다(MuScriptor 만의 민감도, SW 의 민감도는 포함하지 않음).
  범주별 표와 판정 규칙은 그대로이고 해석에만 쓴다.
- 결과:
  - 2026-10-04 `20261004-190420_e3f8e5f_distractors`: 계산된 판정 = 서술(판정 없음)
  - 2026-10-04 `20261004-191316_e3f8e5f_distractors`: 계산된 판정 = 서술(판정 없음)
  - 2026-10-04 `20261004-192625_e3f8e5f_distractors`: 계산된 판정 = 서술(판정 없음)
  - 실행 기록: 첫 전체 실행 `20261004-180627_e3f8e5f_distractors` 는 GPU 작업(분리 16.3분, MuScriptor 27.4 + 7.9분)을 모두
    마친 뒤 요약 단계의 버그(`_leak_summary`)로 멈췄다(결과 행 없음, 캐시에 남음). 고친 뒤 `20261004-190420` 은 같은 GPU 출력의
    재계산, `20261004-191316` 은 사후 진단 `base_null` 추가(MuScriptor 6.5분), `20261004-192625` 가 최종판(모든 GPU 출력 캐시,
    metrics.csv 가 191316 과 바이트 단위로 같음). 시험 실행(1곡) 3.4분과 분리 묶음 확인 0.7분을 더해 GPU 약 62분.
  - 요약(실험자, 2026-10-04, `20261004-192625_e3f8e5f_distractors`): 10곡 × 75 s 창 1개(두 번째 창 조건을 만족한 곡 없음,
    ButterflyEffect 는 SFX 가 기타와 겹치는 구간 부족으로 빠짐), 조건 40개 + null 10개. 참조 대비(MuScriptor 파생, 절대 성능 아님)
    `base` F1 82.9, FP 130/분, FN 95/분. 제품 경로 Δ(+범주 − base, 곡 블록 95 % CI):
    - 신스 패드(3곡): FP +18.7 [+12.0, +24.0]/분, FN +24.8 [+13.6, +44.0]/분, 재현율 −3.7 [−6.1, −2.2]점, F1 −3.1 [−4.9, −1.9].
      늘어난 FP 는 대부분 `guitar_timing` 라벨(+18.4/분)이고 패드 라벨은 +1.9/분뿐. 누출 −12.5 dB. (2026-10-05 리뷰 보완:
      `guitar_timing` 은 대역 1위 소리를 보기 **전에** 붙는 라벨이라 "기타 자신의 분할 탓"은 확정되지 않는다. 캐시로 다시 낸
      조건·경로별 표에서 패드 3구간에 더해진 timing FP 69개 중 19개(5.1/분)는 그 음 대역 1위가 패드, 53개(14.1/분)는 기타였다.
      패드가 기타 음을 다시 잡히게 만든 것인지 기타 자신의 분할인지는 이 원인 규칙으로 가를 수 없다.)
    - 현악(2곡): FP +55.6 [+50.4, +60.8], FN +50.0 [+8.8, +91.2], F1 −5.9 [−9.9, −2.7] — 그러나 같은 오디오를 기타만 마스크로
      전사하면 F1 +0.5 [0.0, +0.9]. 손해는 현악 소리가 아니라 편성 추정이 넣은 마스크 클래스(synth_strings, organ) 탓.
    - 피아노(3곡): FP +21.9 [+4.0, +40.0], F1 −1.4 [−2.8, +0.7](기타만 마스크 −1.5 [−2.8, −0.7]). 오르간(2곡): FP +17.6
      [0.0, +35.2](Secretariat +35.2, Signe 0), 오르간이 대역 1위인 FP +14.0/분. 효과음(3곡): FP +9.1 [−4.8, +27.2], F1 −0.6
      [−1.3, +0.6](효과 없음). 신스 리드(3곡) FP −82.4 [−247, +0.8], 로즈(2곡) FP −48.4 [−102, +5.6], 이름 없는 신스(2곡) FP −81.6
      [−92.8, −70.4]·FN +16.4 [+3.2, +29.6]: 소리를 **더했는데** FP 가 줄었다 → 아래 null 참고, 해석 보류.
    - `full`(10곡): FN +36.0 [+10.6, +68.8], 재현율 −5.6 [−10.1, −1.7], FP +0.5 [−67.8, +53.2], F1 −3.3 [−8.0, +1.7];
      기타만 마스크로는 F1 −0.7 [−4.0, +3.1].
    - null(사후 진단, ‑90 dBFS 잡음만): |ΔFP| 중앙값 0.4/분, |ΔFN| 0.8/분이지만 Mu −145.6 FP/분(F1 +9.1), Signe +56 FN/분
      (F1 −5.0). 풀링 재현율 −1.3 [−3.0, −0.1]. **MuScriptor 디코딩은 일부 구간에서 들리지 않는 변화에도 크게 흔들린다.**
    - 마스크: 제품 마스크가 비기타 클래스를 넣은 13조건에서 비기타로 표시돼 빠진 음은 0개, F1(풀링) 기타만 85.3 → 제품 82.3,
      조건별 평균 −2.1점, 13개 중 9개 나빠짐. 이 구간들에서는 2026-10-04 의 "마스크는 블리드를 거르지 않고 출력만 뒤섞는다"와
      같았다. (2026-10-05 보완: **일반화되지 않았다.** 이 측정에 쓰지 않은 구간의 E3c 에서 JetB 의 건반 마스크가 피아노 블리드
      191음을 걸러 F1 이 61.9 → 76.3 으로 올랐다.)
    - 원인(전체 믹스, 분당): 기타 자신의 타이밍·분할 FP 67.7 / 추정 음 타이밍 어긋남 FN 70.6, 옥타브 FP 27.7 / FN 40.9 가 가장
      크고, 방해 소리가 대역 1위인 FP 는 모두 합쳐 약 7/분. 이 오류는 드럼·베이스·보컬만 있는 base 에도 이미 있다.
    - 누출(SW 기타 스템, 그 소리가 지배하는 칸): 현악 −11.5, 패드 −12.5, 피아노 −12.9, 오르간 −14.4, 신스 리드 −16.3, 이름 없는
      신스 −16.4, 효과음 −19.0, 로즈 −23.6 dB(베이스 −14.4, 드럼 −16.2, 보컬 −16.5; 기타 유지 −0.5 dB). 기타 스템 에너지의
      81–85 % 가 기타 지배 칸이고 방해 범주 칸은 각각 1.1 % 이하.
  - 결론(서술): (1) 기타 패스 마스크에 비기타 클래스를 넣는 `guitar+present` 는 이 10개 구간에서 블리드를 하나도 거르지 않고
    F1 만 떨어뜨렸다. **2026-10-05 정정:** 이것은 일반적인 성질이 아니었다. 이 관찰로 세운 E3c(보류 구간 5곡)에서 JetB 의 건반
    마스크는 피아노 블리드 191음을 걸러 F1 을 14점 올렸고(Zeno 는 −1점), `guitar_only` 로 바꾸자는 가설은 기각됐다. 마스크의
    효과는 곡·구간마다 다르며(걸러내기도, 출력만 흔들기도 한다), 정책은 곡 수를 늘린 E3b 로 판정한다. (2) 가장 큰 오류는 방해 소리가 아니라 분리된
    기타 전사의 음 분할·타이밍(50–200 ms)·옥타브 차이다(S7 정리 대상). (3) 확실히 해로운 소리는 신스 패드(재현율 −3.7점, FN 증가)와
    오르간·피아노(FP +18–22/분). (4) 곡 2–3개짜리 범주의 차이는 디코딩 흔들림(null) 안에 있을 수 있으니, 결정 전에 디코딩 안정화
    (겹치는 창 재디코딩, beam: E21) 또는 곡 수 확대가 필요하다. 상태(`결정:`)는 사람이 확인한 뒤 기록한다.

## E23
- 제목: Basic Pitch 최소 음 길이·문턱 튜닝
- 상태: 결정: 채택 (f7_o0.6_fr0.4: onset 0.6, frame 0.4, min_note_frames 7; test Δ +6.6점 [+3.8, +9.9]; 12 프레임은 등록대로 비교용)
- 질문: Basic Pitch의 `min_note_frames`(남기는 가장 짧은 음, 프레임), onset·frame 문턱을 어떻게 둘 것인가?
- 선택지: `bp_min_note_frames` ∈ {3, 4, 5, 6, 7, 12}(= 약 35/46/58/70/81/139 ms 이상 유지; 12 = 업스트림 127.7 ms 기본값,
  비교용으로만 허용) × onset ∈ {0.3, 0.4, 0.5, 0.6} × frame ∈ {0.2, 0.3, 0.4} = 72개. 기준 = 현재 기본값 `f4_o0.5_fr0.3`.
- 주 지표: onset-only F1(50 ms).
- MDE: 없음(튜닝). 확인 조건: test에서 튜닝값 ≥ 기본값(점추정), CI 함께 보고.
- 데이터(세트, 분할, 창 정의): EGDB(Basic Pitch 학습 제외 — contamination.md) 트랙 그룹 단위 dev/test 분할(공식 분할이 없으면
  트랙 id 해시, seed 고정). `section` = dev/test.
- 분석: dev에서 72개 중 풀링 F1 최대를 고르고, test에서 튜닝값 − 기본값 쌍 비교(블록 부트스트랩 10k).
- 결정 규칙: test 점추정 ≥ 0이면 채택(튜닝값을 `defaults.toml` 과 `amt_bp` manifest에 기록), 아니면 기본값 유지.
- 판정 설정: `rule=tuning; metric=onset_f1_50; baseline=f4_o0.5_fr0.3; split_col=section`
- 비용: 트랙당 모델 1회(`sweep` 모드) + 후처리 72회.
- 실행 명령: `bandscribe eval exp E23`
- 판정 설정 보완: 2026-10-05(2026-10-05 리뷰, 결정 뒤의 기록 정정. **기록된 결정은 바꾸지 않는다**). 러너의 tuning 규칙은 dev 최대를
  72개 arm 전체에서 골랐다. 그래서 2026-10-03 실행의 계산된 판정은 `f12_o0.6_fr0.3`(dev 65.70)을 골랐지만, 선택지 정의는 12 프레임을
  "비교용으로만 허용"으로 사전 등록했고 config 도 12 를 거부한다. 기록된 결정(`f7_o0.6_fr0.4`, dev 65.48 = 프레임 3–7 의 60개 중
  dev 최대)은 그 등록대로 사람이 고른 것이다. 다만 분석 문구("72개 중 풀링 F1 최대")는 12 프레임을 빼지 않아 등록 자체가 서로
  어긋나 있었다. **결정 뒤에 기계 판독 줄을 고치지 않는다**(결과를 본 뒤 규칙을 바꾸는 것이 되므로): 이 줄은 실행 당시 그대로 두고,
  러너의 계산값(`f12_…`)과 기록된 결정(`f7_…`)이 다른 이유를 여기 남긴다. 2026-10-05 에 생긴 `exclude=<접두어>` 키(보고만 하고
  고르지 않을 arm)를 쓰면 같은 행에서 러너도 `f7_o0.6_fr0.4`(test Δ +6.56)를 고른다 — 앞으로의 튜닝 실험은 등록할 때 이 키를 쓴다.
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전(등록 문구의 구체화). EGDB 는 README 의 공식 분할이 있다: dev = train 클립 1–215, test = 클립 216–240(241 없음). 두 분할 모두 앰프 렌더 5종(Ftwin, JCjazz, Marshall, Mesa, Plexi), DI 제외. 블록 = 클립(한 클립의 렌더 5개 = 한 블록). `scenario` = 앰프 이름으로 짐작한 clean/distorted(귀로 확인 안 함).
- 결과:
  - 2026-10-03 `20261003-182652_1e422dc_exp-e23`: 계산된 판정 = 채택 (튜닝값 f12_o0.6_fr0.3, test Δ +6.04점)
  - 요약(실험자, 2026-10-03): EGDB README 공식 분할, 앰프 렌더 5종(DI 제외). dev = 클립 1–215(1,075 렌더, 215 블록),
    test = 클립 216–240(125 렌더, 25 블록). onset-only F1(50 ms) 풀링, rung `onnx-cpu`. dev 기본값 `f4_o0.5_fr0.3` 61.86.
  - **러너의 계산값(`f12_o0.6_fr0.3`, dev 65.70)은 쓰지 않는다**: 이 항목의 선택지 정의가 12 프레임을 "업스트림 127.7 ms
    기본값, 비교용으로만 허용" 으로 사전 등록했고, config 도 12 를 거부한다(빠른 피킹을 지움, DESIGN S6). 그래서 선택 대상은
    프레임 3–7 의 60개다. 그중 dev 최대 = **`f7_o0.6_fr0.4`(dev 65.48)**. test 확인(쌍 비교, 클립 블록 10k):
    기본값 66.04 → **72.60, Δ +6.56 [+3.77, +9.86]**. clean Δ +6.60 [+4.34, +9.46], distorted Δ +6.53 [+3.31, +10.17].
    결정 규칙(test 점추정 ≥ 0) 충족 → 채택. 참고: test 에서 f12_o0.6_fr0.3 − f7_o0.6_fr0.4 = −0.52 [−2.04, +0.73] 이다.
    12 프레임이 더 낫다는 근거는 없다.
  - 반영: `amt.bp_onset_threshold = 0.6`, `amt.bp_frame_threshold = 0.4`, `amt.bp_min_note_frames = 7`(81 ms 이상 유지).
    defaults.toml, config.py, AMT_DEFAULTS, test_config, test_amt_bp_params, test_amt_stages, M1_M2_SPEC §1.3 을 고쳤다.
    `amt_bp` manifest 에는 다음 `bandscribe run` 때 기록된다.
  - dev 주변 효과(풀링 F1): 프레임 3/4/5/6/7/12 = 51.4/52.8/54.2/55.4/56.2/57.2, onset 0.3/0.4/0.5/0.6 =
    45.0/54.9/60.4/61.3, frame 0.2/0.3/0.4 = 49.8/56.7/58.2. **최적값이 격자의 끝(onset 0.6, frame 0.4, 프레임 7)에 있다.**
    격자 밖이 더 나을 수 있으므로, 넓히려면 새 id 로 사전 등록한다. 한계: EGDB 는 단독 기타(앰프 시뮬)라 밴드 분리 스템의
    잔여물 조건은 아니다. clean/distorted 는 앰프 이름으로 짐작했다. 근거: `data/runs/20261003-182652_1e422dc_exp-e23/`
    (`e23_eligible.json`).

## E24-sep
- 제목: SW dtype — 업스트림(fp32 가중치 + AMP) vs fp16 가중치
- 상태: 결정: 판단 보류 (M2 범위에서 미실행: GPU 40분 이상 비용. 업스트림 dtype 유지)
- 질문: SW 가중치를 fp16으로 바꿔도 하류 전사 품질이 유지되고 자원이 줄어드는가?
- 선택지: `sep.dtype` = `upstream`(기준) | `fp16`
- 주 지표: SW 스템에 MuScriptor(medium 고정)를 돌린 onset F1(50 ms), reftx 대비.
- MDE: 비열등 한계 −0.5점.
- 데이터(세트, 분할, 창 정의): Tier B/C 부분집합(E1과 같은 리믹스). 창 = 곡.
- 분석: 곡 블록 부트스트랩 10k, fp16 − upstream. 자원: `reserved_peak_mb`, `rtf` 행(같은 칸).
- 결정 규칙: CI 하한 > −0.5 **그리고** (reserved 피크 −20 % 이상 **또는** 실시간 배수 +20 % 이상)일 때만 fp16 채택. 기본은 upstream 유지.
- 판정 설정: `rule=noninferiority; metric=onset_f1_50; baseline=upstream; margin=-0.5; resource=reserved_peak_mb:-20%|rtf:+20%`
- 비용: SW 곡당 2회.
- 실행 명령: `bandscribe eval exp E24-sep`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전(지표·MDE·결정 규칙은 그대로). (1) 지연 상수 `amt.latency_s` 는 추정 전사와 reftx 참조에 **똑같이** 뺀다. 둘 다 MuScriptor 원시 출력이라 같은 지연을 가지므로, 추정에만 빼면 모든 arm 이 상수만큼 참조와 어긋난다(실행 전에 `_refs_by_kind` 를 고침). (2) Tier B = Cambridge-MT 6곡(ButterflyEffect, JetB, Mistrusted, YoungGriffo, Secretariat, Woodfire). `lines.yaml` 은 사람이 검토하기 전(`reviewed: false`)이고 MedleyDB 는 없다.
- 사전 등록 보완 2: 2026-10-05, 첫 실행 전(2026-10-05 리뷰). **참조를 기타 Line 들에 걸쳐 병합한다**(reference merged across
  guitar lines): kind(기타·베이스)마다 Line 들의 reftx 를 합친 뒤 같은 음높이·onset 50 ms 안의 음을 하나로 세고, 추정도 같은 규칙으로
  병합한다(`eval.suites._merged_arrays`, `bandscribe.amt.experiments.merged`). 이유는 E3b 의 같은 날 보완과 같다(두 Line 이 함께 친
  음이 두 번 세어져 재현율 상한·arm 간 편향). 지표·MDE·결정 규칙은 그대로다.
- 결과: (아직 없음)

## E24-amt
- 제목: MuScriptor dtype — 업스트림(fp32 가중치 + 내장 fp16 autocast) vs fp16 가중치(조건화기 fp32)
- 상태: 결정: 판단 보류 (M2 범위에서 미실행: GPU 40분 이상 비용. 업스트림 dtype 유지)
- 질문: MuScriptor 가중치를 fp16으로 바꿔도 전사 품질이 유지되고 자원이 줄어드는가? (bf16은 금지, sm_75)
- 선택지: `amt.dtype` = `upstream`(기준) | `fp16`
- 주 지표: onset F1(50 ms), reftx(Tier B)·GT(Tier C) 대비.
- MDE: 비열등 한계 −0.5점.
- 데이터(세트, 분할, 창 정의): Tier B/C 부분집합. 창 = 곡/트랙.
- 분석: 블록 부트스트랩 10k, fp16 − upstream.
- 결정 규칙: E24-sep와 같다. 기본은 upstream 유지.
- 판정 설정: `rule=noninferiority; metric=onset_f1_50; baseline=upstream; margin=-0.5; resource=reserved_peak_mb:-20%|rtf:+20%`
- 비용: MuScriptor 뷰당 2회.
- 실행 명령: `bandscribe eval exp E24-amt`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전. 데이터 = GuitarSet 연주자마다 5트랙(항목 id 의 seed 20260930 해시 순, 모두 30트랙). 코드가 앞 30트랙(= 연주자 00 하나 = 블록 1개)을 고르던 것을 실행 전에 고쳤다. 블록 = 연주자(6블록, 기본 분석 규칙). Tier B reftx 부분은 비용 때문에 뺀다. GuitarSet × MuScriptor 오염 = 불명 → 두 dtype 의 상대 비교만 하고 절대 성능은 주장하지 않는다.
- 결과: (아직 없음)

## gate-m2
- 제목: 주 전사기 점검 게이트 (MuScriptor vs Basic Pitch, 디스토션)
- 상태: 결정: 판단 보류 (오염 불명; 수치는 디스토션 Δ +8.1점 [+2.1, +13.5], 문턱 +5 이상이나 통과로 쓰지 않음)
- 질문: 디스토션 기타에서 MuScriptor가 Basic Pitch보다 충분히 나은가?
- 선택지: `muscriptor`(medium, 튜닝값) vs `basicpitch`(E23 튜닝값)
- 주 지표: onset F1(50 ms), 디스토션 하위 세트(`scenario=distortion`), 두 전사기 모두 튜닝한 설정.
- MDE: 게이트 문턱 +5점(MuScriptor − Basic Pitch).
- 데이터(세트, 분할, 창 정의): EGDB 앰프 렌더 test 트랙(공식 test 분할이 있으면 그것; EGDB 논문의 80/5/15 분할은 트랙 id를 공개하지
  않는다). EGDB는 Basic Pitch 학습 밖이지만 MuScriptor 학습 여부는 불명. **Tier B는 제외**(참조가 MuScriptor 파생이라 MuScriptor에 유리).
- 분석: 트랙 그룹 블록 부트스트랩 10k, muscriptor − basicpitch.
- 결정 규칙: 점추정 ≥ +5점이면 통과, 아니면 "주 전사기 재검토"를 기록. **사전 등록한 오염 규칙:** contamination.md에서
  MuScriptor-medium × EGDB가 `학습 포함` 이면 `판단 보류 (오염)`, `불명` 이면 `판단 보류 (오염 불명)` 으로 기록하고 통과로 쓰지 않는다
  (수치와 CI는 그대로 보고).
- 판정 설정: `rule=gate; metric=onset_f1_50; candidate=muscriptor; baseline=basicpitch; threshold=5.0; subset=distortion; contamination=muscriptor-medium:egdb`
- 비용: EGDB test 트랙에 두 전사기.
- 실행 명령: `bandscribe eval exp gate-m2`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전. test = EGDB 공식 test 클립 216–240 의 앰프 렌더(README 분할). 디스토션 하위 세트 = Marshall, Mesa, Plexi(앰프 이름으로 짐작, 귀로 확인 안 함). 판정과 무관한 진단 행을 더한다: 두 전사기의 onset 중앙 잔차(`median_residual_ms`). Basic Pitch 는 설계상 지연 보정을 하지 않으므로, 점수 차이가 지연에서 오는지 확인하려는 것이다.
- 결과:
  - 2026-10-04 `20261004-010229_1e422dc_exp-gate-m2`: 계산된 판정 = 판단 보류 (오염 불명)
  - 요약(실험자, 2026-10-04): EGDB 공식 test 클립 216–240 × 앰프 렌더 5종 = 125 렌더(25 클립 블록). MuScriptor =
    medium-lean, `latency_s = -0.003`. Basic Pitch = E23 튜닝값 `f7_o0.6_fr0.4`, `onnx-cpu`. onset F1(50 ms):
    **디스토션 하위 세트(Marshall/Mesa/Plexi, 75 렌더): MuScriptor 80.3 [75.5, 84.7] vs Basic Pitch 72.2 [68.6, 76.2],
    Δ +8.1 [+2.1, +13.5]** — 수치로는 게이트 문턱 +5점 이상이다. clean(Ftwin/JCjazz) Δ +8.0 [+2.6, +12.9]. 앰프별 Δ: Plexi +15.1,
    Ftwin +10.4, JCjazz +5.5, Marshall +5.3, Mesa +4.8.
  - 진단(사전 등록 보완의 행): onset 중앙 잔차(렌더별 중앙값의 중앙값)는 MuScriptor +2.8 ms(보정 후), Basic Pitch −4.8 ms 다.
    두 전사기 모두 50 ms 허용폭에 비해 작다. 그래서 점수 차이는 지연에서 오지 않는다.
  - **판정: 사전 등록한 오염 규칙에 따라 `판단 보류 (오염 불명)`.** `contamination.md` 에서 MuScriptor-medium × EGDB 는 `불명` 이다
    (MuScriptor 실제 오디오 학습 데이터 목록 비공개). 위 수치는 "통과" 로 쓰지 않는다. 주 전사기는 MuScriptor 를 유지한다.
    재검토 사유가 아니라 판정 불가다. 오염이 없는 디스토션 기타 세트(예: 사용자 GP 정답이 있는 Tier A)가 생기면 새 id 로 다시 본다.
    근거: `data/runs/20261004-010229_1e422dc_exp-gate-m2/`.

## latency
- 제목: MuScriptor 지연 상수 (서술)
- 상태: 결정: 채택 (amt.latency_s = -0.003, 2026-10-03; 서술 실험 — b10 미달: held-out 10.9 ms)
- 질문: MuScriptor onset에 일정한 지연이 있는가, 보정하면 잔차가 얼마인가?
- 선택지: 없음(측정). `amt.latency_s` 를 dev 추정값으로 둔다.
- 주 지표: 보정 후 onset 잔차 절댓값의 중앙값(ms), **held-out** 세트에서. b10 기준 10 ms(참고).
- MDE: 없음(서술).
- 데이터(세트, 분할, 창 정의): 상수 **추정** = GuitarSet 연주자 00–02(+ EGDB dev가 있으면); **보고** = 연주자 03–05.
  두 분할은 겹치지 않는다(`section` = dev/test). 스펙트럴 플럭스 onset은 교차 확인.
- 분석: dev에서 GT 기반 중앙값 = 상수; test 잔차 중앙값에 트랙 블록 부트스트랩 CI. GuitarSet은 MuScriptor 학습 여부 불명이므로
  절대 정확도 주장은 하지 않고 상수 추정에만 쓴다.
- 결정 규칙: 서술(판정 없음). 상수는 통합 요청으로 `defaults.toml` 에 반영.
- 판정 설정: `rule=descriptive; metric=median_abs_residual_ms; split_col=section`
- 비용: GuitarSet 트랙당 MuScriptor 1회.
- 실행 명령: `bandscribe eval exp latency`
- 사전 등록 보완: 2026-10-03 17시 45분 KST, 첫 실행 전. (1) EGDB dev 는 넣지 않는다. 실행 시점에 EGDB 가 덜 받아졌고, 다 받아도 train 클립 앰프 렌더 1,075개(약 9시간 오디오, MuScriptor 약 7시간)라 비용이 맞지 않는다. 추정 = GuitarSet 연주자 00–02 전체(180트랙), 보고 = 03–05 전체(180트랙). (2) 부트스트랩 블록 = 트랙(등록대로). 코드가 연주자를 블록으로 쓰던 것(분할당 3블록)을 실행 전에 고쳤다. 같은 연주자의 트랙끼리는 독립이 아니므로 연주자 블록 CI 도 참고로 적는다.
- 결과:
  - 2026-10-03 `20261003-190418_1e422dc_exp-latency`: 계산된 판정 = 서술(판정 없음)
  - 요약(실험자, 2026-10-03): 상수 = dev(GuitarSet 연주자 00–02, 180트랙, 짝지은 onset 29,970개)의 잔차 중앙값 **−3.0 ms**
    (MuScriptor onset 이 GT 보다 약 3 ms 이르다) → `amt.latency_s = -0.003` 반영(defaults.toml, config.py, AMT_DEFAULTS,
    test_config, M1_M2_SPEC §1.3). 보정 후 |잔차| 중앙값(트랙별 중앙값의 평균, 트랙 블록 10k):
    **held-out test(연주자 03–05, 180트랙) 10.87 ms [10.47, 11.28]**, dev 11.54 [11.11, 11.98]. 연주자 블록(분할당 3블록) CI:
    test [10.42, 11.25]. test 풀링(짝 25,541개): |r| 중앙값 11.26 ms, 부호 중앙값 −0.57 ms(상수가 held-out 에서도 중심을 맞춘다),
    |r| ≤ 10 ms 45 %, ≤ 25 ms 82 %, ≤ 50 ms 96 %. 교차 확인(전사 onset − 스펙트럴 플럭스 onset, 트랙 중앙값): dev +7.8 ms,
    test +6.3 ms(플럭스 자체의 편향이 섞여 참고만).
  - **b10(보정 후 < 10 ms) 미달**: held-out 10.9 ms, CI 하한도 10 ms 위. 상수로는 더 줄지 않는다(보정 후 부호 중앙값 ≈ 0).
    남은 것은 음마다 흩어지는 타이밍이다: MuScriptor 출력 시각이 10 ms 격자로 양자화돼 있고(캐시 11,156개 onset 전부
    10 ms 배수, 확인함), GuitarSet GT onset 자체의 오차도 섞인다. GuitarSet × MuScriptor 오염 = 불명 → 절대 정확도
    주장 아님. EGDB dev 는 비용 때문에 뺐다(사전 등록 보완). 근거: `data/runs/20261003-190418_1e422dc_exp-latency/`
    (`latency_constant.json`, `latency_extra.json`).

## stereo-preservation
- 제목: SW 기타 스템의 스테레오 보존 (M6 조기판, 서술)
- 상태: 결정: 판단 보류 (서술: FLAC 보관 오차 −137~−148 dB 로 b7 충족; 합성 4장면 중 3장면은 SW 기타 스템이 비어 정보 없음, A 장면만 보고; M6 에서 실제 멀티트랙으로 다시)
- 질문: SW가 기타 스템의 bin별 coherence·ILD를 얼마나 보존하는가? float 스템 24-bit FLAC 보관 오차는?
- 선택지: 없음(측정).
- 주 지표: 장면별 에너지 가중 |Δcoherence|, |ΔILD| dB(참 기타 합 vs SW 기타 스템), `window_features` 차이;
  실제 스템 FLAC 왕복 오차(믹스 RMS 대비 dB, −60 dB 미만 희박 스템은 `skipped_sparse`).
- MDE: 없음(서술).
- 데이터(세트, 분할, 창 정의): 합성 `full_band` 장면(A, P70, B, twin 등, SW 최상위 칸 고정), 작업 1개 이상의 실제 스템.
- 분석: 장면별 값과 합성 세트 평균(블록 부트스트랩 CI).
- 결정 규칙: 서술(판정 없음). M6 go/no-go 참고 자료.
- 판정 설정: `rule=descriptive; metric=stereo_delta_coh`
- 비용: SW 장면당 1회.
- 실행 명령: `bandscribe eval exp stereo-preservation`
- 결과:
  - 2026-10-05 `20261005-143016_b50ea18_exp-stereo-preservation`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-05): (1) **FLAC 보관 오차**(실제 작업의 float 스템을 24-bit FLAC 으로 보관 후 복원, 게인 되돌림): 믹스 대비
    −136.9 ~ −148.1 dB → M2 b7 기준(< −100 dB) 충족. (2) **합성 장면 스테레오 보존**: A·B·P70·twin 4장면 중 B·P70·twin 은
    SW 기타 스템 에너지가 참 기타 대비 −75 ~ −80 dB 로 사실상 비어 있어(E3 와 같은 현상: 합성 기타를 SW 가 기타로 보내지 않음)
    비교할 정보가 없다. A 장면만 값이 있다: 창 특징 Δcoherence +0.005, Δ고-coherence 비율 −0.02, Δhard +0.012, Δcentre −0.001
    (라우터 특징은 거의 보존), 반면 bin 단위 Δcoherence 0.26·ΔILD 5.3 dB·S/M +4.7 dB·지연 특징 −29 로 bin 단위 큐는 흔들린다.
    결론(서술): 창 단위 라우터 특징은 A 장면에서 보존됐지만 근거가 1장면뿐이다. M6 go/no-go 는 합성 장면이 아니라 실제 멀티트랙
    (distractors 의 Cambridge-MT 곡)으로 다시 잰다.

## E12
- 제목: 운지(줄/프렛 배정) — 자체 Viterbi(DP) vs tuttut (M3 운지 [판정])
- 상태: 결정: 판단 보류 (목표 미달: dev 최고 dp_s1.2_h0.05 의 test 일치 57.4점 < 75점; tuttut 대비 Δ +10.65 [+8.91, +12.47]점으로 기준 이상; 기본 가중치 유지)
- 질문: M3 의 자체 Viterbi 운지기(`bandscribe.tab.fretting`)가 GuitarSet 줄 정답에서 줄/프렛 일치 75 % 이상이고 tuttut 이상인가
  (DESIGN 10 M3 "운지 [판정]")? 손 이동(`shift`)과 높이(`height`) 가중치는 dev 에서 고른다.
- 선택지: `tuttut`(0.0.6 기본 가중치, `envs/tuttut`, MIT; 기준) | `dp_s{shift}_h{height}`, shift ∈ {0.3, 0.6, 1.2} ×
  height ∈ {0.05, 0.15, 0.4} (9 arm). 나머지 가중치는 이 등록 커밋의 `fretting.DEFAULT_WEIGHTS`(`dp_s0.6_h0.15` = 기본값).
- 주 지표: `string_agreement` = 정답 음 중 예측 (줄, 프렛)이 정답과 같은 비율(점, ×100), 트랙 평균(트랙마다 같은 무게).
  입력은 정답 음(음높이·onset·offset; "음높이가 맞은 음에서")이고 튜닝은 표준(E2–E4)이다. 이벤트 = onset 이 그 이벤트 첫 음에서
  50 ms 안인 음들(스트럼 화음)이며 두 시스템에 같은 이벤트를 준다(tuttut 에는 이벤트의 모든 음을 첫 onset 에 둔 MIDI). tuttut 가 낸
  이벤트에 그 음높이 위치가 없으면(범위 밖 옥타브 이동, 같은 음높이 중복) 그 음은 틀린 것으로 센다. tuttut 가 예외로 실패한
  트랙은 두 arm 모두에서 빼고 수를 보고한다.
- MDE: 해당 없음(dev 선택 + test 에서 기준 이상 확인).
- 데이터(세트, 분할, 창 정의): GuitarSet 1.1.0 전 360트랙(6명 × 30곡 × comp/solo). 분할(결과 보기 전 고정): **dev = 진행 1**
  (트랙 id `NN_<스타일>1-…`, 120트랙), **test = 진행 2·3**(240트랙). 블록 = 곡(스타일+진행+템포+조, 예 `BN2-131-B`; test 20블록).
  같은 곡을 6명이 쳤으므로 블록은 곡이다.
- 분석: dev 에서 10 arm 의 트랙 평균. dev 최고 arm 하나만 test 에서 tuttut 와 비교(곡 블록 부트스트랩 10k, seed 20260930,
  쌍 Δ 95 % CI). 함께 보고: comp/solo 별 값, 연주 가능성 불변식(프렛 0–24, 스팬 ≤ 5, 줄 중복 없음) 위반 수(0 이어야 함), 트랙당 시간.
- 결정 규칙: (러너 `tuning` 규칙) dev 최고가 tuttut 이면 기각(자체 DP 를 쓰되 M3 운지 [판정] 실패로 기록). 아니면 test 에서
  Δ(dev 최고 − tuttut) ≥ 0 이면 채택 후보이고, **그 arm 의 test 일치가 75.0점 이상이어야** 채택(M3 운지 [판정] 통과, 기본 가중치를
  그 arm 으로). Δ ≥ 0 이지만 75점 미만이면 `판단 보류 (목표 미달)` 로 기록하고 M3 한계로 남긴다.
- 판정 설정: `rule=tuning; metric=string_agreement; baseline=tuttut; split_col=section`
- 비용: CPU 약 10–20분(tuttut 360트랙, 트랙당 1–5 s), dp 9 arm 수 분.
- 실행 명령: `bandscribe eval exp E12`
- 등록 메모(2026-10-05, 실행 전): 개발은 합성 예(음계·개방 코드·분산화음)로만 했고 GuitarSet 결과는 아직 보지 않았다. tuttut 는
  `envs/tuttut`(Python 3.10, 고정 의존성) 에서 `bandscribe/tab/tuttut_runner.py` 로 돌린다.
- 결과:
  - 2026-10-05 `20261005-165232_7a53f8c_exp-e12`: 계산된 판정 = 채택 (튜닝값 dp_s1.2_h0.05, test Δ +10.65점)
  - 요약(2026-10-05): dev(진행 1, 120트랙) 최고 = `dp_s1.2_h0.05` 59.9점(tuttut 48.4, 기본값 `dp_s0.6_h0.15` 51.7). test(진행 2·3, 240트랙, 20곡 블록): dp_s1.2_h0.05 **57.4** [53.9, 60.7] vs tuttut 46.7 [42.9, 50.7], Δ **+10.65 [+8.91, +12.47]**점(comp +10.7, solo +10.6). tuttut 실패 0, 연주 가능성 위반 0. 러너의 tuning 규칙은 Δ ≥ 0 이라 '채택'을 계산했지만, 사전 등록한 추가 조건(test ≥ 75.0점)을 못 넘어 **판단 보류 (목표 미달)** 이다: 기본 가중치는 바꾸지 않고, M3 운지 [판정](≥ 75 %)은 미달로 남긴다.
  - 해석: dev 최고가 격자의 모서리(shift 최대, height 최소)다 — 선수들은 한 포지션에 머물고 높은 프렛을 꺼리지 않는다. 등록 모델은 단음의 손 위치를 그 음의 프렛으로 보므로 같은 포지션 안의 프렛 이동도 손 이동으로 센다(`tests/test_tab_fretting.py` 의 xfail 사례). 손 위치 창(한 손가락 한 프렛) 상태를 넣은 모델은 새 id(E12b)로 사전 등록한다. 한계: GuitarSet 은 어쿠스틱 6명의 반주·솔로이고 같은 곡을 여럿이 친다.

## E12b
- 제목: 운지 모델 v2(손 위치 창) — E12 의 dev 최고 v1 대비
- 상태: 결정: 채택 (win_s0.5_h0.0_o0.0, test Δ +11.96 [+8.41, +15.34]점; M3 운지 [판정] 미달: test 69.3점, 음 단위 74.6 % < 75)
- 질문: 손 위치를 "한 손가락 한 프렛" 창(프렛 p..p+3)으로 상태에 넣은 모델(`fretting.assign(model="window")`)이 E12 의 dev 최고
  v1 설정보다 GuitarSet 줄/프렛 일치가 높은가? 75 % 목표(M3 운지 [판정])에 닿는가?
- 선택지: `v1_s1.2_h0.05`(E12 dev 최고 = 기준) | `tuttut`(비교) | `win_s{shift}_h{height}_o{open}`, shift ∈ {0.5, 1.0, 2.0} ×
  height ∈ {0.0, 0.05, 0.15} × open ∈ {0.0, 0.3} (18 arm). 나머지 가중치는 이 등록 커밋의 `fretting.DEFAULT_WEIGHTS`.
- 주 지표: E12 와 같은 `string_agreement`(트랙 평균, 점). 같은 입력·이벤트(50 ms)·채점. 함께 보고(서술, 판정 없음): 음 단위로
  모은 일치(DESIGN 6.3 의 75 % 와 문헌 수치는 음 단위다; E12 등록은 트랙 평균이었고 이 실험도 판정은 트랙 평균으로 한다).
- MDE: 해당 없음(튜닝).
- 데이터: E12 와 같다(GuitarSet 360트랙; dev = 진행 1, test = 진행 2·3; 블록 = 곡). **한계: test 분할은 E12 에서 한 번
  (arm 하나) 썼다** — 이 실험의 선택은 dev 만 보고, test 는 고른 arm 하나만 본다.
- 분석: E12 와 같다(dev 에서 고르고 test 에서 기준과 쌍 비교, 곡 블록 부트스트랩 10k, seed 20260930).
- 결정 규칙: (러너 `tuning`) dev 최고가 기준(v1)이면 기각. 아니면 test Δ(dev 최고 − v1) ≥ 0 이면 채택 — 기본 운지 모델·가중치를
  그 arm 으로 바꾼다. M3 운지 [판정](test ≥ 75점, tuttut 이상)은 그 arm 의 test 값으로 따로 적는다(미달이어도 채택은 유효).
- 판정 설정: `rule=tuning; metric=string_agreement; baseline=v1_s1.2_h0.05; split_col=section`
- 비용: CPU 약 10분.
- 실행 명령: `bandscribe eval exp E12b`
- 등록 메모(2026-10-05, 실행 전): E12 의 dev 오류 분석(dev 만): 틀린 음의 85 % 는 예측이 정답보다 낮은 프렛, 개방현 비율 예측
  14.9 % vs 정답 8.6 % — 모델이 낮은 포지션과 개방현을 지나치게 좋아한다. 창 모델은 같은 포지션 안의 프렛 이동을 손 이동으로
  세지 않는다. 개방현 보너스 0 도 선택지에 넣었다.
- 결과:
  - 2026-10-05 `20261005-170507_228868d_exp-e12b`: 계산된 판정 = 채택 (튜닝값 win_s0.5_h0.0_o0.0, test Δ +11.96점)
  - 요약(2026-10-05): dev 최고 = `win_s0.5_h0.0_o0.0` 70.6점(창 모델, 높이·개방현 보너스 0; s2.0 70.2, s1.0 69.8 — shift 는 거의 무관, height·open 0 이 결정적). test(240트랙, 20곡): **69.3** [66.0, 72.6] vs 기준 v1 57.4, Δ **+11.96 [+8.41, +15.34]**점(comp +2.7 [−1.5, +6.7], solo **+21.2** [+16.1, +26.4]). 음 단위로 모은 일치(서술): test **74.6 %**(v1 66.6 %, tuttut 55.9 %), dev 78.7 %. 연주 가능성 위반 0.
  - 결정: 채택 — 기본 운지 모델을 `window`, 가중치 shift 0.5·height 0.0·open 0.0 으로 바꾼다(튜닝 탐색의 운지 비용은 개방현 단서가 필요해 등록 당시 v1 가중치를 그대로 쓴다). M3 운지 [판정](≥ 75)은 등록 지표(트랙 평균) 69.3점으로 **미달**, 음 단위 74.6 % 도 문턱 바로 아래다. tuttut 이상 조건은 충족(트랙 평균 46.7 대비).

## E13
- 제목: 베이스 정리 — 3-트래커 옥타브·피치 앵커, 조각·슬라이드 합치기, 킥 거부, 투표자 2개 정책(F0 채우기) vs AND
- 상태: 결정: 채택 (합치기 단계만 켬: fragments·slides, 비열등 Δ +1.29 [−0.06, +3.14]; anchor·kick·gap_fill 은 판단 보류라 끔)
- 질문: MuScriptor 베이스 패스(M2 원시)를 F0 트래커 셋(CREPE full·PESTO mir-1k_g7·pyin, `bandscribe.bass.f0`)으로 정리하면
  onset F1 이 오르는가? 어느 단계를 기본으로 켤까(DESIGN 5 "도움이 된 것만 켠다")? 투표자 2개 정책(MuScriptor 주 소스 + F0 로
  빈 곳 채우기)이 AND(둘 다 있는 음만)보다 재현율에서 낫고 정밀도 손실이 작은가(M5 [판정])?
- 선택지(`bandscribe.bass.clean.arm_params`, 문턱은 이 등록 커밋의 `clean.DEFAULT_PARAMS`·`f0.DEFAULT_PARAMS`, 주법 표시는 끔):
  `raw`(기준 = M2 원시) | `anchor`(2-of-3 투표로 옥타브·반음 이동 + 화성 중복 제거) | `anchor_merge`(+ 같은 음 조각·감쇠 꼬리
  합치기 + 슬라이드 사슬 합치기) | `anchor_merge_kick`(+ 킥 거부, 드럼 스템이 있는 Tier B 에서만 달라짐) | `two_voter`
  (anchor_merge + 두 트래커가 같은 음을 100 ms 이상 낸 빈 곳을 저신뢰 음으로) | `and`(anchor_merge + F0 투표가 그 음높이를 댄
  음만, 비교용으로만).
- 주 지표: 베이스 onset F1(50 ms, 음높이 일치, offset 무시), 참조·추정 모두 같은 음 50 ms 안 합침(E1–E3b 와 같음).
  함께 보고(판정 없음): chroma F1, 옥타브 오류율(= chroma F1 − pitch F1, 점), tick F1(정답 박 격자, IDMT·FiloBass).
- MDE: 1.0점(우월성 단계). 비열등 여유 −1.0점(합치기 단계).
- 데이터(세트, 분할, 창 정의):
  - IDMT-SMT-Bass-Single-Track 17트랙 전체(DI 일렉 베이스, 연주자 1명; 블록 = 트랙). 입력 = 녹음 그대로.
  - FiloBass 12곡(곡 이름의 `sha256("20260930:filobass:<곡>")` 순서 앞 12곡; 블록 = 곡), 창 30–90 s, 입력 = 세트의 분리된 베이스
    스템, 정답 = `midi_fully_aligned` 중 창 안에서 시작하는 음.
  - Tier B: Cambridge-MT 의 E1 6곡(ButterflyEffect, JetB, Mistrusted, YoungGriffo, Secretariat, Woodfire)을 `tierb_mix`
    (master −8 LUFS)로 다시 섞어 SW 로 분리한 베이스 스템, 참조 = 참 베이스 트랙의 reftx(arm 비교용, 절대 성능 주장 금지).
  - 분할 없음(선택지가 미리 정해져 있고 튜닝하지 않는다). `scenario` = 데이터셋.
  - 오염: MuScriptor × IDMT·FiloBass = 불명(`docs/eval/contamination.md`) — 절대값은 서술만, arm 사이 비교는 유효.
- 분석: 블록 부트스트랩 10k(seed 20260930). 단계마다 바로 앞 arm 과 쌍 비교(`bandscribe.bass.experiments.STEP_RULES`),
  하위 세트 = 데이터셋.
- 결정 규칙(단계별, 러너가 `steps.json` 에 계산; 사람이 `[bass]` 기본값에 반영):
  - anchor(`anchor` − `raw`): DESIGN 8.1 우월성(CI 가 0 제외, 점추정 ≥ 1.0, 어느 데이터셋 CI 상한도 −2 미만 아님).
  - 합치기(`anchor_merge` − `anchor`): 비열등(전체 CI 하한 > −1.0 이고 어느 데이터셋 CI 상한도 −2 미만 아님). 목적이 기보
    (계단으로 적힌 슬라이드를 한 음 + 슬라이드로)라 F1 이 오를 필요는 없고 떨어지지만 않으면 켠다.
  - 킥 거부(`anchor_merge_kick` − `anchor_merge`, Tier B 항목만): 우월성(MDE 1.0).
  - F0 채우기(`two_voter` − `anchor_merge`): 우월성(MDE 1.0).
  - 단계는 정해진 순서로 쌓은 arm 끼리 비교한다(앞 단계가 꺼져도 다시 돌리지 않는다; 효과가 거의 더해진다고 본다).
  - 기각·보류된 단계는 끈다. 러너의 계산된 판정(`판정 설정`, raw 대비 가장 큰 채택 arm)은 참고로 함께 적는다.
  - M5 [판정] "투표자 2개가 AND 보다 재현율에서 낫고 정밀도 손실이 CI상 작다": `policy_check.json` 의 `two_voter` − `and` 재현율
    Δ CI 가 0 초과, 정밀도 Δ CI 하한 ≥ −2점이면 충족.
- 판정 설정: `rule=superiority; metric=onset_f1_50; baseline=raw; mde=1.0; subset_col=scenario; exclude=and`
- 비용: GPU 약 25–30분(SW 6곡 약 7분, MuScriptor 베이스 패스 IDMT 6분·FiloBass 12분·Tier B 24분 분량 오디오, CREPE·PESTO),
  CPU pyin·정리 약 10분. MuScriptor 는 eval 전사 캐시, F0 는 `data/eval/f0_cache` 를 쓴다(다시 돌리면 GPU 거의 없음).
- 실행 명령: `bandscribe eval exp E13`
- 등록 메모(2026-10-08, 실행 전):
  - 미리 본 것: seisyun complex(Tier A, 정답 악보 PDF) 한 곡만 서술로 봤다 — 마디별 음 묶음 F1 raw 90.1, anchor 90.1(옥타브 이동 0,
    반음 6), anchor_merge 90.7(FP 98 → 87, 인트로 슬라이드 41·39·38·37·36·35 → 한 음 + 슬라이드), two_voter = anchor_merge(채운 음
    0), and 72.2. E13 데이터(IDMT·FiloBass·Tier B)는 등록 전에 러너 동작만 IDMT 2트랙으로 확인했고 지표는 보지 않았다.
  - 문턱은 seisyun complex 의 슬라이드(3.0–5.6 s, F0 42.5 → 31.2 반음)와 그 곡의 다른 구간에서 손으로 정했다: 글라이드 = 10 ms
    걸음 ≤ 0.8반음, 음정 가운데 절반을 40 ms 이상 걸쳐 지남; 정체 = 250 ms 동안 기울기 ≤ 1.5반음/s, 흩어짐 ≤ 0.15. 등록 뒤에는
    바꾸지 않는다.
  - 트래커 신뢰도 문턱은 셋 다 0.5(정하지 않고 관례값). seisyun complex 에서 프레임 유효 비율 CREPE 60 %, PESTO 67 %(대부분 한
    옥타브 위), pyin 16 % — 투표가 자주 "split"(셋이 서로 다름, 노트의 31 %)이다.
  - torchcrepe 의 Viterbi 는 시그모이드 출력에 softmax 를 해 관측이 거의 평평하고, 베이스 스템에서 경로가 최저 bin 에 붙어
    프레이즈 전체를 놓쳤다(seisyun complex 34–36 s). 원래 CREPE 처럼 salience 를 합 1 로 정규화해 한 번에 디코딩한다.
  - SCNet 앙상블(DESIGN E13 의 네 번째 항목)과 YourMT3+ 투표자 3개(2-of-3)는 이번에 없다(설치 안 함) — 다음 등록으로 미룬다.
- 결과:
  - 2026-10-08 `20261008-184108_55fe1df_exp-e13`: 계산된 판정 = 판단 보류 (기본값 raw 유지)
  - 요약(2026-10-08): 항목 35개(IDMT 17, FiloBass 12, Tier B 6), rung `medium-lean|crepe-full`, GPU 약 14분(Tier B 6곡의
    MuScriptor 는 E1 의 eval 캐시를 그대로 맞혔다: 같은 분리 결과). onset F1(raw → anchor → anchor_merge → +kick → two_voter;
    and): IDMT 87.74 → 87.74 → 87.78 → 87.78 → 87.97; 72.40. FiloBass 91.50 → 91.30 → 91.23 → 91.23 → 90.97; 81.65. Tier B 75.88 →
    75.64 → 77.81 → 78.24 → 77.79; 77.52.
  - 단계(`steps.json`): anchor Δ −0.18 [−0.42, +0.04] → 판단 보류(옥타브 이동 IDMT 3·FiloBass 16·Tier B 66, 반음 0·24·27; Tier B
    옥타브 오류율 1.12 → 1.50 으로 오히려 늘었다). 합치기 Δ +1.29 [−0.06, +3.14](Tier B +2.17 [+0.01, +3.92], FiloBass −0.07,
    IDMT +0.05) → 비열등 채택. 킥 거부(Tier B) Δ +0.43 [−0.25, +1.04] → 판단 보류(3,545음을 뺐지만 대부분 50 ms 안 중복이라 합친
    채점에서 거의 안 보인다). F0 채우기 Δ −0.06 [−0.14, +0.01] → 판단 보류(채운 음 IDMT 8·FiloBass 16·Tier B 5).
  - 투표자 2개 vs AND(`policy_check.json`): 재현율 Δ +20.9 [+17.2, +25.3], 정밀도 Δ −15.4 [−36.4, −2.6] → CI 하한이 −2 보다 낮아
    M5 [판정] **미달**(데이터셋별 정밀도 Δ: IDMT −1.2 [−4.3, +1.7], FiloBass −7.2 [−10.7, −4.0], Tier B −23.2 [−56.7, −0.9]).
  - 서술(같은 실행): 트래커 30–100 Hz 원 피치 정확도(신뢰도 ≥ 0.5, 50 cents) IDMT CREPE 85.9 % · pyin 56.0 % · PESTO 0.9 %,
    FiloBass CREPE 82.7 % · pyin 52.9 % · PESTO 34.4 %(신뢰도를 무시하면 PESTO 68.6 / 71.2 %, 옥타브 오류 FiloBass 7.1 %). IDMT 운지
    (정답 음, 하모닉스 뺌) 줄 = 줄·프렛 82.1 %(916음). 기본 설정의 주법 제안(IDMT): 데드 노트 재현율 17/28, 정밀도 17/31; 비브라토
    1/31, 1/4; 슬라이드 4/20, 4/8.
  - 서술(채택 조합 = 합치기만, 등록 arm 에 없는 조합이라 같은 항목으로 따로 잼, `final_config_descriptive.json`): raw 대비
    Δ +1.28 [−0.08, +3.12](IDMT +0.05, FiloBass −0.03, Tier B +2.13), 옥타브 오류율 IDMT 0.52 · FiloBass 0.19 · Tier B 1.10(raw 와
    같음).
  - 결정: 채택 — `[bass]` 기본값을 fragments·slides 켬, anchor·kick·gap_fill 끔, policy muscriptor 로 둔다. 트래커는 슬라이드
    판단에 쓰므로 `bass_f0` 단계는 그대로 돈다. SCNet 앙상블·투표자 3개는 다음 등록으로.
  - 리뷰 뒤 수정(2026-10-08, 결정은 그대로): 코드가 등록한 정의와 달랐던 두 곳을 맞췄다. ① 글라이드는 "가운데 절반을 40 ms 이상
    걸쳐 지남"인데, 코드는 그 구간 안의 프레임 수만 셌다. 그래서 A440 에서 25 cents 이상 어긋난 녹음이나 비브라토가 있는 음이
    가운데에 머물러 슬라이드로 읽혔고, +35 cents 의 반음계 피킹 40-41-42-43-45 는 41·42 가 지워졌다. 이제 `a` 쪽 1/4 의 마지막
    프레임에서 `b` 쪽 1/4 의 첫 프레임까지 건너가는 시간을 재고, 그 들머리·날머리 걸음은 0.4반음 이하여야 한다. 윤곽선에서는 곡의
    튜닝 어긋남(정체 구간의 윤곽선 − 음높이 중앙값)을 뺀다. ② "다시 친 같은 음은 남긴다"인데 80 ms 보다 짧은 같은 음은 어택이
    있어도 합쳐졌다(170 BPM 16분음표 8개 → 1개). 이제 어택이 없을 때만 합친다. 같은 항목으로 다시 잰 채택 조합(서술,
    `post_review_descriptive.json`): raw 대비 Δ +0.92 [−0.01, +2.16](IDMT 0.00, FiloBass +0.03, Tier B +1.50 [+0.01, +2.65]),
    여전히 비열등. 옥타브 오류율 그대로(0.52 / 0.19 / 1.12). IDMT 슬라이드 제안 2/20(표시 3개 중 2개 맞음, 전에는 4/20·4/8).

## E14
- 제목: 양자화 설정 — 타이밍 잡음 σ, 수준 벌점, 24 vs 48 tick, 이분/삼분 전환 벌점, 셔플·12/8 판정 문턱
- 상태: 사전 등록
- 질문: `bandscribe.tab.quantize` 의 잠정 설정(σ 30 ms, `LEVEL_PEN`, switch 1.0, swing 문턱)이 실제 곡에서 기보 일치(8.4 #2)와
  MUSTER식 onset 오류를 가장 좋게 하는가? 32분음표·16분 셋잇단(48 tick)을 허용하는 것이 24 tick 격자보다 나은가?
- 선택지: σ ∈ {22, 26, 30, 35} ms × 격자 {48, 24} (24 = 32분음표·16분 셋잇단 금지) × switch ∈ {0.5, 1.0, 2.0}. 기준 = 잠정 기본값
  (σ 30, 48, 1.0).
- 주 지표: Tier A 기보 일치(정답 템포맵으로 마디를 맞춘 뒤 (마디, tick, 음높이) 정확 일치 비율, 8.4 #2), 점.
- MDE: 1.0점.
- 데이터: Tier A 곡(사용자 Guitar Pro 정답, M1b). dev/test = gt.yaml 구간 분할.
- 분석·결정 규칙: `select=dev; split=test` 우월성(DESIGN 8.1).
- 판정 설정: `rule=superiority; metric=notation_exact; baseline=s30_t48_w1.0; mde=1.0; select=dev; split=test`
- 비용: CPU 수 분(양자화만).
- 실행 명령: `bandscribe eval exp E14`
- 등록 메모(2026-10-05): σ 30 ms 잠정 기본값은 정답 없이 정했다 — 합성(±20 ms) 정확 (마디, tick) 0.990(22 ms: 0.994)이고, 실제
  5곡의 p90 onset 잔차가 23–36 ms 라 22 ms 는 기타 음의 6–18 % 를 32분음표로 보냈다(30 ms: 0–11 %). Tier A 정답이 생기면 이 실험이
  정한다(데이터 대기).
- 결과: (아직 없음)

## E16
- 제목: A4 varispeed — 기준음이 어긋난 녹음을 분석 전에 A440 으로 되돌릴 문턱
- 상태: 사전 등록
- 질문: 녹음의 기준음 오프셋(`a4.json`)이 클 때 분석용 사본을 varispeed 로 A440 에 맞춘 뒤 전사하면 기타·베이스 음표 F1 이
  오르는가? 어느 문턱부터 켜야 하는가?
- 선택지: `off`(기준, 현재) | `t15`, `t25`, `t35`(|cents| ≥ 문턱이면 분리·전사 전에 varispeed, 음 시각은 원래 시간축으로 되돌림).
- 주 지표: 기타 onset F1(`note_f1_onset50`), 정답 대비.
- MDE: 1.0점.
- 데이터: GuitarSet 트랙(음 정답)을 +25, −30, +45 cents 로 재렌더(resample)한 사본 + 원본(0 cents), 곡 블록.
- 분석·결정 규칙: 우월성(DESIGN 8.1), 오프셋 크기별 하위 세트 보고. 0 cents 하위 세트에서 손해(CI 상한 < −2)면 기각.
- 판정 설정: `rule=superiority; metric=note_f1_onset50; baseline=off; mde=1.0`
- 비용: GPU 약 20–30분(MuScriptor, 4 오프셋 × 트랙 표본).
- 실행 명령: `bandscribe eval exp E16`
- 등록 메모(2026-10-05): M3 는 오프셋을 재고(`a4` 단계) 25 cents 부터 경고만 한다. 사용자 5곡은 +2–4 cents 로 적용 대상이 아니다.
- 결과: (아직 없음)

## E6
- 제목: 라우터 판정 — 참 기타 합 vs 분리(SW) 스템, 합성 문턱 vs 분리 스템에서 다시 맞춘 문턱 (M6 공간 경로 go/no-go)
- 상태: 결정: 채택 (GO: sw|refit test macro 0.994 [0.985, 1.000], R2 재현율 1.00; 다시 맞춘 문턱을 [router.thresholds] 기본값으로)
- 질문: 실제 멀티트랙으로 만든 리믹스에서 라우터 특징(DESIGN S4)이 SW 분리 뒤에도 보존되는가? 창마다 라우터 라벨이
  리믹스를 만든 방식(정답)과 얼마나 맞는가 — 합성에서 나온 설계 문턱으로, 그리고 분리 스템 기준으로 다시 맞춘 문턱으로?
  공간 경로를 설계대로 써도 되는가(DESIGN 10 M6 의 go/no-go)?
- 선택지(arm): `true|design`(참 기타 합의 특징 + 설계 문턱), `sw|design`(SW 기타 스템의 특징 + 설계 문턱),
  `sw|refit`(SW 특징 + dev 곡에서 다시 맞춘 문턱).
- 주 지표: `sw|refit` 의 test 곡 창 macro 정확도(정답 클래스 R0/R1/R2/R3 마다 맞힌 비율의 평균)와 R2(넓은 같은 파트 더블)
  재현율. 보조: arm × 분할마다 macro·클래스별 정확도, 혼동 행렬, 특징 보존 |참 − SW|(SM, coh, cen, hard, fcoh, ildcoh, xcoff).
- MDE: 없음(go/no-go 판정; 점추정으로 판정하고 곡 블록 부트스트랩 CI 를 함께 적는다).
- 데이터(세트, 분할, 창 정의): Cambridge-MT 13곡 전부(로컬). 곡마다 만들 수 있는 장면 B·A·P·ADT(`bandscribe.router.tierb`:
  A 는 더블이 있는 곡만, P 는 서로 다른 테이크 둘이 함께 연주하는 곡만), 60 s 발췌, 8 s 창(겹치지 않음). 정답 = 리믹스 구성에서
  나온 창 라벨(`tierb.window_truth`); `mixed`·`silent` 창은 정확도에서 빼고 개수만 적는다. 테이크·더블 관계는 기계 라벨이다
  (onset 상관·chroma, `tierb` 모듈 문서; 사람이 귀로 확인하지 않았다). 분할: 곡 id 의 sha256 순서로 더블이 있는 곡끼리
  dev/test 를 번갈아 나누고 나머지 곡이 그 번갈음을 이어 받는다(`split_songs`). SW 는 최상위 칸 고정, 리믹스를 묶어 분리한다
  (디스트랙터 스위트와 같은 배치: 각 리믹스가 혼자일 때와 같은 청크 격자).
- 분석: dev 곡의 SW 특징으로 문턱을 라벨 단위로 함께 격자 탐색(후보 = 설계값 + dev 특징의 11개 분위수, 최대 3회 반복,
  목적 = dev macro, 동률은 설계값에 가까운 쪽)해 다시 맞추고 test 곡에서 잰다. 곡 블록 부트스트랩 10,000회(seed 20260930).
- 결정 규칙: `sw|refit` test macro ≥ 0.80 그리고 R2 재현율 ≥ 0.70 → GO(공간 경로를 설계대로 쓰고, 다시 맞춘 문턱을
  `[router.thresholds]` 기본값으로); macro ≥ 0.60 → 부분(여유가 큰 창만 라우터 판정을 쓰고 나머지는 가능한 경로를 모두;
  여유 문턱은 M6b 에서 dev 로 정한다); 그 밖 → NO-GO(N 경로와 기호 분할로 대신). 0.80 은 M6 완료 기준 "마디 단위 80 % 일치".
  test 곡에 R2 창이 없으면 GO 가 될 수 없다(부분이 최대).
- 판정 설정: `rule=descriptive; metric=router_acc; split_col=section` (go/no-go 는 위 규칙으로 `go_nogo.json` 에 계산)
- 비용: SW 리믹스 약 45개 × 60 s(약 45분 오디오) → GPU 약 11–15분(시험 실행 실측 0.23 s/s). CPU: 트랙 관계(곡당 30–60 s,
  캐시), 특징.
- 미리 본 것(등록 전): 2곡(mistrusted_polemic, secretariat_overthetop) 시험 실행으로 두 가지를 고쳤다. (1) ADT 규칙에
  "지연 상관이 지연 0 상관보다 크다"(`adt_over_r0`)를 더했다: 가운데 모노 기타 14창 중 5창이 ADT 로 잡혔다(지속음이 주기적이라
  9–36 ms 에서 상관 0.5–0.75, 지연 0 상관은 1). (2) 다시 맞추기를 문턱 하나씩에서 라벨 단위 묶음 탐색으로 바꿨다: 실제 더블
  (hard 0.14–0.23, cen 0.24–0.45)이 설계 R2 상자(hard < 0.1, cen > 0.4)와 R3 사이에 있어 한 문턱씩으로는 움직일 수 없었다.
  규칙의 문턱 값·결정 규칙은 이 등록 뒤 바꾸지 않는다. R1 자기 일치 보정(DESIGN R1 거부 문턱·변주 경계)은 L/R 전사가 필요해
  M6b 에서 따로 등록한다(E6b).
- 실행 명령: `bandscribe eval exp E6`
- 결과:
  - 2026-10-10 `20261010-180608_6bed725_exp-e6`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-10): 리믹스 45개(B 13, ADT 13, P 12, A 7; SW 39개 새로 분리, GPU 11.5분), 창 284개(정확도 대상; `mixed` 12·
    `silent` 19 제외). 사전 등록 규칙으로 계산한 판정 = **GO** (`go_nogo.json`).

    | arm | dev macro [CI] | dev R2 | test macro [CI] | test R2 |
    |---|---|---|---|---|
    | `true\|design` | 0.807 [0.713, 0.900] | 0.29 | 0.969 [0.958, 1.000] | 0.90 |
    | `sw\|design` | 0.816 [0.709, 0.925] | 0.35 | 0.969 [0.958, 1.000] | 0.90 |
    | `sw\|refit` | 0.978 [0.957, 0.995] | 1.00 | **0.994 [0.985, 1.000]** | **1.00** |

    - 분리 뒤에도 특징이 보존됐다: `true` 와 `sw` 의 정확도가 거의 같고(창 단위 |Δcoh|·|Δcen| 0.01 수준, `d_*` 행), M2 조기
      점검의 "창 특징은 보존" 이 실제 멀티트랙에서 확인됐다. 설계 문턱이 놓친 것은 분리가 아니라 **실제 더블**이다(dev R2
      0.29: 합성 더블보다 hard 비율이 높고 centre 비율이 낮다). 그래서 다시 맞춘 R2 상자가 크게 넓어졌다.
    - 다시 맞춘 문턱(설계 → dev): `r0_sm_db` −12 → −4.14, `r0_coh` 0.95 → 0.84, `r2_coh` 0.85 → 0.994, `r2_hard` 0.10 → 0.846,
      `r2_cen` 0.40 → 0.117, `r3_ild_db` 6 → 1.90. R2 상자는 사실상 "R0·R1·R3 이 아니고 centre ≥ 0.12" 가 됐다(점수 순서가
      나머지를 가른다). dev 7곡, test 6곡.
    - 한계: 리믹스는 깨끗하다(장면 하나, 추가 리버브 없음). 테이크·더블은 기계 라벨이고 귀로 확인하지 않았다. 창은 8 s
      고정(파이프라인은 4마디).
    - 사용자 5곡(서술, Tier A): 다시 맞춘 문턱으로 `mixed` 창이 대부분 R3 로 정리된다(seisyun complex 8 → 0, AZ 7 → 0, 青と夏
      3 → 0; AIZO 11 → 0 이 R2 7·R3 6·R0 4 로). 설계 문턱에서 R3 이던 창은 그대로 R3.
    - 리뷰 뒤 메모(2026-10-10): `tierb.pair_relation` 이 onset 프레임을 활동 프레임에 정수 비(20, 실제 20.045)로 대응시켜
      200 s 에서 약 0.45 s 앞당겨진 활동 마스크로 상관을 쟀다. 고쳤고(`tierb` 코드 버전 2) 관계 캐시는 다음 실행에서 다시
      계산된다. 이 실행의 테이크·더블 라벨은 옛 대응으로 만든 것이다. 2026-10-11 다시 계산해 보니 더블 라벨이 바뀐 것은 1쌍뿐이다
      (zeno G3/G4: r 0.31 → 0.30 미만, 더블 → 다른 파트). 이 실행의 리믹스를 다시 쓰는 실험(E22·E22b·E6b·E7)은 E6 의 관계(tierb
      코드 1 캐시)를 그대로 읽는다(`split_eval.E6_REL_VERSION`).

## E22
- 제목: 음표별 스테레오 서명으로 기타 두 Line 나누기 — P(좌우 다른 파트)·A(넓은 더블 + 가운데 파트) (M6b)
- 상태: 결정: 판단 보류 (P: note_sig − majority +18.4 [7.9, 29.6]점 < 20점, 창 단위 팬보다 +15.8 [5.2, 27.3]; A: 리드 F1 −35.1 [−52.7, −16.6], 누설 0.30)
- 질문: 분리된 기타 스템에서 음표마다 잰 스테레오 서명(`bandscribe.router.notesig`: 그 음의 배음 bin 에서 ILD, coherence,
  mid/side)으로 음을 Line 에 나누면(`bandscribe.router.split`), P 리믹스에서 다수 클래스 기준선보다 20점 넘게 맞히는가
  (DESIGN 10 M6 [판정] P)? 창 단위 팬보다 나은가(E22 원래 질문: 0 을 제외하지 못하면 창 단위가 기본)? A 리믹스에서 가운데
  파트(리드) Line 이 합친 기준선보다 나아지고 리듬(더블) Line 은 거의 잃지 않는가(DESIGN 10 M6 [판정] A)?
- 선택지(arm): P — `majority`(모든 음을 가장 큰 Line 에), `window_pan`(8 s 창의 음을 그 창에서 에너지가 큰 쪽에),
  `note_sig|oracle`(구성에서 나온 창 라벨 + 음표별 서명), `note_sig|router`(라우터가 SW 특징으로 붙인 라벨, 설계 문턱 +
  음표별 서명; 끝에서 끝까지, 서술). A — `majority`, `merged`(나누지 않음: 모든 음이 두 Line 에), `note_sig|oracle`,
  `note_sig|router`. 서명 문턱은 합성 값으로 미리 고정한다(`split.DEFAULT_PARAMS`: P 는 |ILD| ≥ 6 dB, A 는 coherence ≥ 0.8
  그리고 mid/side ≥ 10 dB 면 가운데). Tier B 로 고르지 않는다.
- 주 지표: P — 리믹스마다 macro 할당 정확도(참조 Line 마다 그 Line 에만 속한 음 중 정확히 그 Line 으로 간 비율의 평균; 한 Line
  음을 두 Line 에 쓰면 오답, 매칭 안 된 시스템 음은 제외), `note_sig|oracle` − `majority` 를 리믹스 쌍으로. A — 가운데 Line
  음표 F1(`lead_f1`) 의 `note_sig|oracle` − `merged`, 리듬 Line F1 하락(`merged` − `note_sig|oracle`), 가운데 Line 누설
  (`leak_lead`: 가운데로 간 음 중 참조가 더블뿐인 비율). 보조: 유니즌 음이 두 Line 에 모두 간 비율, `note_sig|router`, 
  `window_pan` 과의 차이.
- MDE: P 20점(DESIGN 기준), A 리드 10점(DESIGN 기준).
- 데이터: E6 실행(`20261010-180608_6bed725_exp-e6`)의 P·A 리믹스 전부, 앞 32 s(8 s 창 4개). 참조 = 리믹스에 들어간 각 테이크의
  대표 파트 같은 구간을 모노로 MuScriptor(기타 3클래스 마스크, 최상위 칸) 전사; 시스템 = 같은 리믹스의 SW 기타 스템(E6 캐시)
  모노 다운믹스를 같은 설정으로 전사. 매칭은 같은 음높이, onset ±50 ms, 가까운 것부터 한 번씩. 테이크·더블 라벨은 E6 와 같은
  기계 라벨.
- 분석: 리믹스 쌍 차이의 평균, 곡 블록 부트스트랩 10,000회(seed 20260930) 95 % CI.
- 결정 규칙: P — `note_sig|oracle` − `majority` ≥ 20점 그리고 CI 하한 > 0 → P 창을 음표별 서명으로 나눈다(채택). 아니면 판단
  보류(P 창도 합친 트랙). `window_pan` 대비 CI 가 0 을 제외하지 못하면 그 사실을 적는다(창 단위 팬은 P 에서 두 파트가 함께
  치는 창을 나눌 수 없어 채택 후보가 아니다). A — 리드 F1 이득 ≥ 10점 그리고 `leak_lead` 평균 ≤ 0.25 그리고 리듬 F1 하락 ≤ 2점
  → A 창을 나눈다(채택). 아니면 판단 보류(A 창은 합친 트랙). `note_sig|router` 는 판정에 쓰지 않는다(서술).
- 판정 설정: `rule=descriptive; metric=assign_macro; subset_col=scenario` (규칙은 `e22_decision.json` 에 계산)
- 비용: MuScriptor 뷰 약 (P 리믹스 × 3 + A 리믹스 × 4–5) × 32 s, 약 35분 오디오 → GPU 약 25분(실측 0.65 s/s 기준). SW 는 E6 캐시.
- 미리 본 것(등록 전): 합성 음(왼쪽 끝 E4 + 오른쪽 끝 A4 동시, 가운데 C5, 8 ms·6 cents 떨어진 두 테이크 G4)으로 서명만
  확인했다(ILD +30 / −24 dB, 가운데 coherence 1.0·mid/side > 30 dB, 더블 coherence 0.56·mid/side 2.8 dB). Tier B 음은 보지 않았다.
- 실행 명령: `bandscribe eval exp E22`
- 결과:
  - 2026-10-10 `20261010-183611_fc4ac94_exp-e22`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-10): P 11 리믹스(11곡; mu 는 앞 32 s 에 기타가 없어 빠짐), A 7 리믹스. `e22_decision.json`: P `note_sig|oracle`
    − `majority` **+18.4 [7.9, 29.6]**점(20점 미달), − `window_pan` +15.8 [5.2, 27.3]; A 리드 F1 **−35.1 [−52.7, −16.6]**, 리듬
    하락 +2.8, 누설 0.30. 리믹스별 P 는 0.40(jetb)–0.99(zeno): 6 dB 공유 구간에 든 음이 양쪽에 쓰여 오답이 된 곳이 낮다. 라우터 라벨
    (`note_sig|router`)은 P 에서 oracle 과 같은 값이다(P 리믹스 창을 라우터가 모두 R3 로 봤다).

## E22b
- 제목: 음표별 스테레오 서명으로 두 Line 나누기 — 문턱을 dev 곡에서 고르고 test 곡에서 판정 (M6b, E22 의 후속)
- 상태: 결정: 채택 (P 만: p_ild_db 0 = ILD 부호, test − majority +33.6 [16.2, 44.6]점, 창 단위 팬보다 +32.7 [15.3, 44.5]; A 는 판단 보류: 기준을 통과한 조합은 가운데 Line 을 비워 리드 F1 −51.5 [−66.0, −24.1])
- 질문: E22 와 같다(P: 다수 기준선 + 20점, A: 리드 F1 + 10점·누설 ≤ 25 %·리듬 하락 ≤ 2점). 다른 점은 서명 문턱을 합성 값으로
  고정하지 않고 dev 곡의 리믹스로 고른다는 것뿐이다.
- 등록 이유(E22 결과를 보기 전, 2026-10-10): E22 시험 실행(P 리믹스 2개: butterflyeffect, jetb; 등록된 실행 아님)에서 왼쪽·
  오른쪽 테이크 음의 ILD 가 부호로는 갈렸지만(jetb 중앙값 +6.2 / −4.8 dB) 합성 값 ±11.8 dB 보다 훨씬 작았다. 다른 쪽 기타와
  분리 잔여물이 섞이기 때문이다. E22 의 고정 문턱 6 dB 로는 많은 음이 "양쪽 공유" 로 가서 오답이 된다. E22 는 등록대로 돌리고,
  이 등록은 그 결과와 무관하게 문턱 선택 규칙을 미리 정해 둔다.
- 선택지(arm): E22 와 같다(`majority`, `window_pan`/`merged`, `note_sig|oracle`, `note_sig|router`). `note_sig` 의 문턱은 dev 에서
  고른 하나뿐이다.
- 문턱 선택(dev 곡만): P — `p_ild_db` ∈ {0, 1, 2, 3, 4, 6} 중 `note_sig|oracle` 의 평균 macro 정확도가 가장 높은 것(동률은 큰 값).
  A — (`a_coh` ∈ {0.6, 0.7, 0.8, 0.9}) × (`a_ms_db` ∈ {3, 6, 10, 15}) 중 평균 `leak_lead` ≤ 0.25 이고 평균 리듬 하락 ≤ 2점인 것 가운데
  리드 F1 이득이 가장 큰 것(동률은 격자 순서 앞), 통과하는 조합이 없으면 합성 값(0.8, 10).
- 주 지표·MDE·결정 규칙: E22 와 같은 규칙을 **test 곡의 리믹스에만** 적용한다(`e22_decision.json` 의 `test`). dev 값은 참고.
- 데이터: E22 와 같다(E6 실행의 P·A 리믹스, 앞 32 s, 같은 참조·시스템 전사 — eval 캐시에서 다시 쓴다). 분할 = E6 의 곡 분할
  (`relations.json` 의 `split`: dev 7곡, test 6곡). P 리믹스는 dev·test 모두에, A 리믹스는 dev 4곡(mistrusted, moosmusic,
  jokers, mu)·test 3곡(zeno, signejakobsen, rememberdecember)에 있다.
- 분석: E22 와 같다(리믹스 쌍 차이, 곡 블록 부트스트랩 10,000회, seed 20260930).
- 판정 설정: `rule=descriptive; metric=assign_macro; split_col=section` (규칙은 `e22_decision.json` 의 `test` 에 계산)
- 비용: GPU 없음(E22 의 전사 캐시), CPU 몇 분.
- 실행 명령: `bandscribe eval exp E22b`
- 결과:
  - 2026-10-10 `20261010-184905_bd19625_exp-e22b`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-10): dev 에서 고른 문턱 `p_ild_db` **0**(0 / 1 / 2 / 3 / 4 / 6 dB 의 dev macro 0.813 / 0.800 / 0.774 / 0.760 /
    0.736 / 0.666), A 는 누설·하락 제한을 지키는 조합이 `a_ms_db` 15 뿐이고 그 조합은 가운데 Line 을 비운다(리드 이득 −24.7).
    test: P **+33.6 [16.2, 44.6]**점(5 리믹스: rememberdecember 0.88, signejakobsen 0.88, woodfire 1.00, younggriffo 0.93, zeno
    0.99), 창 단위 팬보다 +32.7 [15.3, 44.5] → **채택**; A 리드 **−51.5 [−66.0, −24.1]** → 판단 보류. dev 값도 같은 방향(P +31.3
    [20.8, 41.2]).
    - 리뷰 뒤 메모(2026-10-10): `choose_params` 가 A 조합의 리드 이득·리듬 하락을 쌍 없이 평균끼리 뺐다(리드 F1 이 한쪽만 있는
    리믹스 mu 가 한 평균에만 들어감). `decide()` 처럼 리믹스 쌍 차이로 고쳤다. 이 실행의 A 선택에는 영향이 없다(어떤 조합도 리드
    이득이 양수가 아니다).

## E6b
- 제목: R1 자기 일치 보정 — 실제 더블의 좌·우 채널 전사 일치도 분포 (M6 미달 g2, M7a 파트 수 추정 준비)
- 상태: 결정: 채택 (서술: R1 거부 문턱 0.61, 변주 경계 0.61/0.80 — 더블 창 16개 7곡이라 잠정; 다른 파트 창의 60 % 가 문턱 아래)
- 질문: 같은 파트를 두 번 친 더블(좌우 끝)을 SW 로 분리한 뒤 왼쪽·오른쪽 채널을 따로 전사하면, 8 s 창마다 두 전사의
  chroma-onset F1(같은 음이름, onset ±50 ms)이 어떻게 분포하는가? DESIGN 4.2 R1: 그 하위 5 % 가 거부 문턱(이보다 낮으면
  더블이 아님), 5–25 % 가 변주 경계다. 서로 다른 두 파트(P)의 창은 그 문턱 아래로 얼마나 떨어지는가(거부력)?
- 선택지: 없음(측정). 행 `system` = `double`(A 리믹스의 R2 창) / `different`(P 리믹스의 R3 창).
- 주 지표: 창 단위 L/R chroma-onset F1 의 분위수(5/25/50/75/95 %).
- MDE: 없음(서술).
- 데이터: E6 실행(`20261010-180608_6bed725_exp-e6`)의 A·P 리믹스, 앞 32 s(E22 와 같은 구간), 8 s 창, 정답 라벨이 R2(A)·R3(P)인
  창 중 양쪽 전사가 각각 4음 이상인 창. MuScriptor 기타 3클래스 마스크, 최상위 칸. DESIGN 은 합성 더블을 썼지만 SW 가 합성
  기타를 기타 스템으로 보내지 않아(M2 stereo-preservation) 실제 멀티트랙 더블을 쓴다.
- 분석: 분위수와 `different` 창 중 거부 문턱(`double` 의 5 %) 아래인 비율.
- 결정 규칙: 서술. `double` 의 5 % 분위수를 R1 거부 문턱, 5 %·25 % 분위수를 변주 경계로 기록하고 M7a 의 기본값으로 쓴다.
  `different` 창의 중앙값이 그 문턱보다 높으면(거부력이 없다는 뜻) 그 사실을 적고 M7a 에서 R1 거부를 끈다.
- 판정 설정: `rule=descriptive; metric=lr_f1; subset_col=scenario`
- 비용: MuScriptor 뷰 약 38개 × 32 s(약 20분 오디오) → GPU 약 13분. SW 는 E6 캐시.
- 실행 명령: `bandscribe eval exp E6b`
- 결과:
  - 2026-10-11 `20261011-150735_812853b_exp-e6b`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-11): 창 단위 L/R chroma-onset F1. `double`(A 리믹스 R2 창 16개, 7곡): 5 / 25 / 50 / 75 / 95 % = **0.61** /
    **0.80** / 0.86 / 0.94 / 0.98. `different`(P 리믹스 R3 창 40개, 11곡): 0.05 / 0.28 / 0.45 / 0.73 / 0.99. → R1 거부 문턱 **0.61**,
    변주 경계 **0.61 / 0.80**(M7a 기본값). 거부력: 다른 파트 창의 60 % 가 문턱 아래, 더블 창은 6 %(1/16, mistrusted 0.35).
    `different` 의 위쪽 꼬리는 기계 라벨이 같은 파트를 다른 파트로 본 경우다(woodfire 4창 0.98–1.00, zeno 1창 1.00: chroma 0.79 로
    "different" 가 된 쌍). 더블 창이 적어(양쪽 4음 이상인 창만) 문턱은 잠정이다.
    - **정정(2026-10-11, M7a 리뷰):** 위 꼬리 설명이 틀렸다. F1 이 높은 `different` 창은 그 8 s 에 쌍의 둘째 테이크가 치지 않는
      창이다(woodfire 4창, zeno P:2, jokers P:3, younggriffo P:1 — 둘째 테이크의 활동 0 s): 한 테이크가 두 채널에 다 있어 L/R 이
      같다. 두 테이크가 다 치는 창 32개만 보면 **72 %(23/32)** 가 거부 문턱 아래다(한 테이크만 치는 8창은 1/8). 또 "M7a 의 기본값"
      은 결과적으로 쓰이지 않았다: M7a v0 는 L/R 채널을 따로 전사하지 않아 R1 거부권이 없다(DESIGN 「M7a 구현」).

## E7
- 제목: 센터 추출기 — A 경로(넓은 더블 + 가운데 파트)의 리드 Line (M6 미달 g4 다시 시도)
- 상태: 결정: 판단 보류 (모든 센터 뷰가 합친 기준선보다 리드 F1 이 낮음: cw_bs_v3beta −13.8 [−29.9, +0.5], cw_bs_v2 −14.1, ipd_ild −14.5, mid −16.4점; A 창은 합친 트랙 유지)
- 질문: DESIGN 4.5 의 A 규칙(리드 = 센터 뷰 음 중 S = (L−R)/2 뷰에 ±1 반음·±50 ms 안의 음이 없는 것, 리듬 = S 뷰 전사)으로
  리드 Line F1 이 합친 기준선보다 10점 넘게 오르고 누설 25 % 이하, 리듬 하락 2점 이하를 지키는 센터 뷰가 있는가(DESIGN 10 M6
  [판정] A)? 있다면 어느 추출기인가(E7 원래 질문)?
- 선택지(arm): `mid`((L+R)/2), `ipd_ild`(bin 마스크: exp(−(ILD/3 dB)²)·cos²(IPD/2) 를 mid 에), `cw_bs_v2`, `cw_bs_v3beta`
  (gilliaan CenterWide BS-RoFormer, `bandscribe.router.centre`); 기준 `merged`(mid 뷰의 음을 두 Line 모두에, E22 와 같음).
- 주 지표: A 리믹스마다 리드 F1(가운데 테이크 참조 대비) − `merged` 리드 F1. 보조: 리듬 F1 하락, `leak_lead`, `leak_rhythm`.
- MDE: 10점(DESIGN 기준).
- 데이터: E6 실행의 A 리믹스 7개 전부, 앞 32 s. 참조 = E22 와 같은 테이크 전사(더블 두 테이크 = 리듬, 가운데 테이크 = 리드),
  매칭도 E22 와 같다. SW 는 E6 캐시, 센터 추출은 각 모델의 yaml 청크·겹침으로 같은 워커(`sep_msst`)에서.
- 분석: 리믹스 쌍 차이, 곡 블록 부트스트랩 10,000회(seed 20260930).
- 결정 규칙: 누설(평균 `leak_lead` ≤ 0.25)·리듬 하락(≤ 2점)을 지키는 arm 중 리드 이득이 가장 큰 것을 고르고, 그 이득이 10점
  이상이고 CI 하한 > 0 이면 A 경로를 채택한다(파이프라인: 라우터 R2 창에 한해 센터 추출 + 센터·S 뷰 전사, 이 규칙으로 리드·리듬
  Line). 아니면 판단 보류(A 창은 합친 트랙). 고르는 것과 판정을 같은 7 리믹스로 하므로 낙관적이다(그렇게 적는다).
- 판정 설정: `rule=descriptive; metric=lead_f1; subset_col=scenario` (규칙은 `e7_decision.json` 에 계산)
- 비용: 센터 추출 GPU 약 2분, MuScriptor 뷰 약 28개 새로(측면·마스크·모델 2개 × 7) × 32 s → GPU 약 10분. 참조 테이크는 캐시.
- 등록 메모(2026-10-11): DESIGN 의 Mel-Roformer v3 대신 같은 저장소의 BS-Roformer v3beta 를 쓴다(Mel 구조는 librosa 의 mel
  필터가 필요한데 gpu env 에 없다; v2 는 DESIGN 그대로). Bertom VST 는 하지 않는다. 모델 카드 라이선스는 CC BY-NC-SA 4.0
  (2026-10-11 저장소에 새로 붙음; DESIGN 작성 때는 표기 없음): 개인용, 재배포 안 함. 미리 본 것: mu A 리믹스 1개에서 두 모델이
  워커에서 도는 것만 확인했다(reserved 570–582 MB, 실시간 2.6–3.6배; 그 32 s 는 기타가 거의 없어 품질은 보지 않았다).
- 실행 명령: `bandscribe eval exp E7`
- 결과:
  - 2026-10-11 `20261011-152240_812853b_exp-e7`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-11): 규칙으로 고른 arm = `cw_bs_v3beta`(누설·하락 제한은 네 arm 모두 통과), 리드 이득 **−13.8 [−29.9, +0.5]**점 →
    판단 보류. 다른 arm: `cw_bs_v2` −14.1 [−29.5, −1.5], `ipd_ild` −14.5 [−31.0, −0.6], `mid` −16.4 [−36.0, −2.3]. 리듬(S 뷰) F1 은 합친
    것보다 +6.1 높다(하락 −6.1). 누설은 v3beta 가 가장 작다(0.009). 리믹스별 리드 F1 은 moosmusic 0.68–0.71·rememberdecember
    0.59–0.71 이 높고 jokers·mistrusted·signejakobsen·zeno 가 낮다(0.02–0.31): "센터 음 중 S 에 ±1 반음 안의 음이 없는 것" 규칙이
    리드 음을 너무 많이 지운다. 한계: 우리 A 리믹스의 '가운데' 참조는 리드만이 아니라 가운데로 보낸 다른 파트 전부다(signejakobsen
    가운데 참조 1,704음). DESIGN 이 노린 "넓은 리듬 더블 + 가운데 리드 하나" 와 다르다. 센터 모델 자원: reserved 570–582 MB,
    실시간 2.6–3.6배(첫 시험 실행) — M6 g10(4.5 GB 이하) 충족.
    - 실행 메모: 첫 실행(`20261011-151802`)은 리뷰 수정(tierb 코드 2)으로 다시 계산된 관계 때문에 zeno 의 A 계획이 사라져 멈췄다.
      E6 의 관계(코드 1)를 읽게 고친 뒤 다시 돌렸다(센터 추출은 캐시).

## E25
- 제목: Part 뷰 v0 — 파트 수 추정과 음표의 Line 할당 (M7a 인수 판정)
- 상태: 결정: 채택 (dev 에서 고른 설정을 기본값으로: n_min 6, 창 2, 좌우 3 dB, 음색·기호 근거 끔 — 사전값은 dev 기타 1대 7/10·파트 수 0.36. M7a 판정은 미달: test 파트 수 0.72·A1+P 0.50·MAE 0.28, 할당 P +15.2 [0.0, 30.5]·A1 0.0·B2 0.0점; 기타 1대 12/12·미배정 0 %·퇴행 없음)
- 질문: S8 v0(`bandscribe.parts`: 라우터 창, 음표별 스테레오 서명, MuScriptor 기타 클래스, 기호 층으로 창마다 동시 스트림을 세어
  강건 하한으로 파트 수를 정하고, 음마다 손 가중치 사후확률로 Line 을 고름)이 DESIGN 10 M7a 의 판정 기준을 통과하는가?
  ① 파트 수 정확 70 % 이상, A·P 90 % 이상, MAE 0.4 이하 ② macro 할당 정확도(Unassigned = 오류)가 다수 기준선보다 A·P +20점,
  B +10점, A·P 는 기호 분할 기준선보다도 +10점 ③ 기타 1대 픽스처에서 파트 수 1 유지 ④ Unassigned 15 % 이하(+ coverage–accuracy).
- 선택지(arm): `majority`(모든 음을 한 Line 에, 파트 수 1 — macro 의 바닥), `symbolic`(기호 분할만, DESIGN 8.5: TabForge
  `split_lead_rhythm` 의 공개 설명대로 자체 구현한 `layers.tabforge_split` — 1 s 창, 코드 음 가중 2, 리듬 문턱 0.6, 10반음 넘게
  떨어진 윗음 떼기; 파트 수 2), `s8v0`(dev 에서 고른 설정), `s8v0|oracle_k`(정답 파트 수를 강제 — 수 오류와 할당 오류를 가르는
  진단, 판정에 안 씀).
- 데이터: E6 실행 `20261010-180608_6bed725_exp-e6` 의 Tier B 13곡, E6 의 곡 분할(dev 7 / test 6). 리믹스 시나리오:
  - `P`, `ADT`: E6 리믹스 그대로(믹스·SW 스템 캐시).
  - `A1`, `B2`, `N1`(`tierb.VARIANTS`, 여기서 새로 렌더링, E6 와 같은 방식·SW 단): A1 = E6 A 의 더블(좌우 끝) + 다른 테이크 하나
    (평가 구간에서 가장 오래 치는 것)를 가운데, 나머지 테이크는 뺌 — DESIGN 의 A; B2 = E6 P 의 두 테이크를 둘 다 가운데 — DESIGN 의
    B; N1 = E6 ADT 의 테이크 하나만 가운데(지연 복사본 없음) — 모노 기타 1대. 변형은 바탕 시나리오의 시작점과 테이크를 그대로 써서
    테이크 참조 전사를 공유한다.
  - E6 의 A·B 리믹스(나머지 테이크를 모두 가운데에, 한 곡 최대 10 테이크)는 쓰지 않는다: 정답 파트 수가 DESIGN 의 시나리오
    정의를 벗어나고(signejakobsen A 는 32 s 안에 활성 테이크 8개), 기계 관계 라벨이 그 수를 정한다.
  - 평가 구간은 앞 32 s(E22 와 같음): E6 의 8 s 창 4개, 구간(section) 1개.
- 정답:
  - 참조 Line = 리믹스에 넣은 테이크를 E6 관계(tierb 코드 1)의 더블 링크로 묶은 것(R1: 더블은 Line 하나). 참조 음 = 각 테이크
    발췌를 따로 MuScriptor 로 전사한 것(E22 와 같은 뷰, 기타 3클래스).
  - 참조 음이 16개 이상인 Line 이 활성(DESIGN 4.4: 16음 미만 Line 은 해체)이고, 파트 수 정답 = 활성 Line 수. 0 인 리믹스는 뺀다.
  - 시스템 음 = SW 스템 모노 다운믹스의 MuScriptor 전사(기타 클래스)와 음표별 스테레오 서명. 음의 정답 Line 집합 = E22 의
    `match`(같은 음높이, onset ±50 ms)로 맞은 활성 Line.
  - 라우터 라벨 = SW 창 특징 + 파이프라인 문턱(`[router.thresholds]`, E6 채택값).
- 주 지표:
  - 파트 수: 리믹스마다 예측 = 정답(정확), |예측 − 정답|(MAE).
  - `assign_macro`: 예측 Line 을 활성 Line 에 헝가리안으로 짝짓고(배타 음 수 최대), 활성 Line 마다 그 Line 만 정답인 음 중 짝지은
    예측에 간 비율을 평균. Unassigned 와 짝 없는 예측의 음은 오답. 유니즌 음(정답 Line 둘 이상)은 따로(`unison_hit`).
  - Unassigned 비율, coverage–accuracy(τ 0.4–0.9, 짝짓기는 τ 0 기준).
- dev 선택(dev 곡만): `s8v0` 설정 격자 48개 = `n_min` {12, 8, 6} × `min_windows` {3, 2} × `side_db` {3, 0} × 클래스 근거
  {끔, 켬} × 기호 근거 {끔, 켬}. 목적(사전식): ① 기타 1대 리믹스(ADT·N1, 정답 1) 중 1 로 센 수 ② 파트 수 정확 비율(dev 리믹스 전부)
  ③ A1·P·B2 의 평균 macro. 동률은 격자 순서 앞(보수적인 쪽: 큰 `n_min`, 많은 창, 큰 `side_db`, 근거 끔). 고정: 가중치
  (`arbitration.DEFAULT_WEIGHTS` — pan 6, ILD 척도 1 dB, 역할 1, 클래스 1, 사전 1), τ 0.4, `min_sections` 2(평가 구간은 구간이
  1개라 실제로는 1). 고른 설정은 파이프라인 `parts` 단계의 기본값이 된다.
- MDE: DESIGN 의 문턱 그대로(+20 / +10점).
- 분석: 리믹스 쌍 차이, 곡 블록 부트스트랩 10,000회(seed 20260930). 판정은 test 곡만, dev 값은 참고.
- 결정 규칙(test; 항목마다 PASS / FAIL 을 M7a 인수 문서에 적는다):
  1. 파트 수: 정확 비율 ≥ 0.70(모든 시나리오), A1·P 합쳐 ≥ 0.90, MAE ≤ 0.4.
  2. 할당: `s8v0` − `majority` 가 A1·P 각각 +20점 이상, B2 +10점 이상, CI 하한 > 0. A1·P 는 `s8v0` − `symbolic` 도 +10점 이상,
     CI 하한 > 0.
  3. 기타 1대: ADT·N1 의 test 리믹스(정답 1)를 모두 1 로 센다.
  4. Unassigned: test 시스템 음 전체의 15 % 이하.
  5. 퇴행 없음(8.1): 어느 시나리오도 `s8v0` − `majority` 의 CI 상한이 −2점 아래로 내려가지 않는다.
  - 여기서 재지 않는 M7a 기준: D·트윈·유니즌 픽스처(데이터 없음), Tier A(서술, 따로), 합친 기타 tick F1 하락(구성상 0 — 파트
    트랙은 합친 기타의 음을 나눠 가질 뿐이고 합친 오선보는 그대로라 코드 테스트로 확인), `--parts` 전환 재실행 2 s(파이프라인에서
    잰다).
- 판정 설정: `rule=descriptive; metric=assign_macro; split_col=section` (규칙은 `e25_decision.json` 의 `test` 에 계산)
- 등록 메모(2026-10-11, 결과 전): dev 곡의 E22 캐시(E6 의 P·A 리믹스, 설계 문턱 라벨)로 v0 를 미리 돌려 봤다(등록된 실행 아님,
  test 곡은 보지 않음). 이 세션들의 두 기타는 대부분 코드를 쳐서 기호 층(코드 위의 리드) 근거가 한 창에서도 2 가 되지 않았고,
  한 손으로 못 잡는 화음(R9)도 창당 0–4개라 근거에 넣지 않았다. 그래서 v0 는 같은 위치의 두 코드 파트(A1·B2)를 가르지 못할
  공산이 크다. P 는 창 3개 기준이면 dev 6곡 중 2곡만 2 로 셌다(창 2개면 5곡) — 32 s 발췌는 창이 4개뿐이라 `min_windows` 를 격자에
  넣었다.
- 비용: 변형 리믹스 31개 SW 분리(60 s, E6 속도로 약 7분), MuScriptor 새 뷰 약 57개 × 32 s(약 30분 오디오, GPU 약 20분). 한 번에
  30분을 넘지 않게 dev 곡·test 곡으로 나눠 캐시만 먼저 채운다(채점 없음). 등록된 실행은 그 캐시로 CPU 몇 분.
- 실행 명령: `bandscribe eval exp E25`
- 결과:
  - 2026-10-11 `20261011-182127_2436e55_exp-e25`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-11): dev 선택 = `n_min` 6, `min_windows` 2, `side_db` 3, 음색 근거 끔, 기호 근거 끔(dev: 기타 1대 10/10, 파트 수
    0.60, macro 0.618). 음색 근거를 켜면 dev 기타 1대 리믹스 3개가 2 파트로 갈렸고(MuScriptor 가 한 테이크를 acoustic/clean 으로
    나눔), 기호 근거(코드 위 리드 층)는 한 창에서도 2 가 되지 않아 켜나 끄나 같았다. 등록 전 사전값(8, 3, 0 dB, 둘 다 켬)은 dev 기타
    1대 7/10, 파트 수 0.36. **test 판정**(리믹스 25개, 6곡): ① 파트 수 정확 **0.72**(기준 0.70 통과), A1·P **0.50**(기준 0.90 미달),
    MAE 0.28(통과) → 미달. 시나리오별 P 0.8, ADT 1.0, A1 0.0, B2 0.4, N1 1.0. ② 할당: P `s8v0` − `majority` **+15.2 [0.0, 30.5]**점,
    기호 분할 대비 +9.8 [−20.2, +28.2] → 미달; A1 0.0 (기호 분할 대비 **−24.5 [−48.9, −9.1]**), B2 0.0 → 미달. ③ 기타 1대 **12/12**
    통과. ④ 미배정 0 %(파트 수가 2 이하라 0.5 미만 사후확률이 없다). ⑤ 퇴행 없음.
    - P 가 +20 에 못 미친 까닭: test P 5개 중 2개(woodfire, younggriffo)는 둘째 테이크가 32 s 안에 거의 없어 정답 파트 수가 1 이라
      올릴 여지가 0 이고, zeno 는 둘째 테이크가 7 s 뿐이라 2 스트림 창이 1개(기준 2) → 1 파트로 셌다(정답 파트 수를 주면 0.99).
      파트 수를 맞힌 두 리믹스는 +37.9·+38.2점. 정답 파트 수를 강제한 진단 팔: P +25.0 [7.6, 42.4], A1 +11.5 [0.0, 34.5], B2 +10.0
      [0.0, 30.0]점.
    - A1·B2 는 예상대로 모두 1 파트(라우터 R2·R0 창이라 좌우 근거 없음). 기호 분할만으로도 zeno(A1 0.99, B2 1.00)·signejakobsen
      (0.65, 0.74)·rememberdecember(0.59, 0.64) 가 다수 기준선보다 나았다 — 이 곡들은 두 파트의 코드/홑음 결이 다르다. 다만 기호
      분할은 기타 1대도 늘 둘로 가른다(test ADT·N1 macro 0.54–1.00).
    - 등록과 다른 점(M7a 리뷰, 판정에는 영향 없음): `symbolic` 팔의 파트 수는 등록처럼 늘 2 가 아니라 실제로 나온 라벨 수로 적혔다
      (50개 중 8개가 1 — 서술 행 `k_correct`·`k_abs_err` 만 달라짐). coverage–accuracy 의 짝짓기는 τ 0 이 아니라 τ 0.4 를 적용한
      예측으로 했다(파트 수 2 이하라 0.4 미만 사후확률이 없어 같다).
    - 실행 메모: `판정 설정` 줄의 백틱 뒤 괄호까지 `split_col` 로 읽혀 `summary.json` 의 서술 값은 dev/test 를 가르지 않았다(E22b 와
      같은 형식). 판정은 등록대로 `e25_decision.json` 의 `test` 로 했다. GPU 캐시는 같은 PC 에서 게임이 GPU 를 쓰는 동안을 피해
      곡마다 나눠 채웠다(`data/tmp/m7a/orchestrate.sh`, 게임 중엔 대기).

## E26
- 제목: 편집 앵커 재적용률 — 상위 단계가 바뀐 뒤 다시 붙는 편집의 비율 (M4b 인수 판정)
- 상태: 결정: 통과 (합친 0.94 [0.82, 1.00] ≥ 0.90; 기타 0.86·베이스 1.00·마디선 0.99 — 근거는 약하다: 기타 음이 실제로 바뀐 곡은 青と夏 하나)
- 질문: M4a 편집 앵커(`bandscribe.tab.edits`: 음은 (트랙, 전사 onset, 음높이) + 같은 음높이 ±30 ms(`MATCH_TOL_S`), 마디선은
  박 시각 ±⅓박(`BEAT_TOL`) 과 그 박에 마디선이 있는지)가, M5·M6·M7a 로 상위 단계(notes·bass·grid·parts)가 실제로 바뀐 뒤의 결과에
  DESIGN 10 M4b 기준대로 **90 % 이상** 다시 붙는가? 나머지가 목록으로 보이는지는 뷰어 검토 큐에서 확인한다(코드 테스트·M4b 인수
  문서).
- 데이터: `data/jobs` 의 작업 중 (가) 지금 `export/` 가 있고 `edits.load_base` 가 열리며(M7a 뒤 실행) (나) 2026-10-05(M4a 날, UTC)
  에 만든 `quant` 단계 결과가 있는 것 — 2026-10-17 기준 Tier A 5곡(seisyun complex, AIZO, AZ, 怪獣の花唄, 青と夏). 옛 결과 = 그날 마지막으로 만든 `quant` 출력(과 그 `grid` 의존), 새 결과 = 지금 export 의 base.
- 합성 편집(곡마다, `random.Random(20260930 + 곡 순번)`, 곡 순번 = 작업 폴더 이름 순):
  - 음: 옛 `quant` 의 기타(`guitar_all`) 음 50개, 베이스(`bass`; 옛 id `bass_raw`) 음 50개를 비복원 균등 추출 → 각 음에 대한
    `pitch` 편집(앵커 = 옛 onset·음높이). 사람이 고칠 음은 균등하지 않지만, 상위 변화로 앵커가 떨어지는 비율의 바닥을 본다.
  - 마디선: 옛 `grid` 의 마디선(첫 마디 제외) 20개 → 한 박 뒤로 옮기는 `barline` 편집(`at_s` = 옛 박 시각, `to_s` = 다음 박 시각).
  - 편집마다 따로 붙여 본다(서로 영향 없음): 음은 `edits.find` 를 새 base 의 양자화 음에(편집 0개 replay), 마디선은
    `edits._beat_of` + 그 박에 새 마디선이 있는지(`edited_grid` 와 같은 조건).
- 주 지표: 재적용률 = 붙은 편집 / 전체 편집, 세 종류(기타 음·베이스 음·마디선)를 합친 것(곡 5 × 120 = 600개). 서술: 종류별·곡별
  비율, 곡 블록 부트스트랩 CI(10,000회, seed 20260930), 정확(≤0.1 ms) / 허용 안(≤30 ms) / 놓침의 분류 — 놓친 음은 같은 음높이가
  30–100 ms 에 있음(허용 폭 문제) / ±12반음이 30 ms 안에 있음(옥타브 변경) / 없음(상위가 음을 지움)으로, 놓친 마디선은 박이 ⅓박
  안에 있으나 마디선이 아님 / 박이 없음으로 나눈다. 허용 폭 60 ms 였다면의 비율과 그때 같은 음높이 후보가 둘 이상인 앵커 수(오적용
  위험)도 서술로만 적는다.
- 결정 규칙: 합친 재적용률 ≥ 0.90 이면 PASS, 아니면 FAIL(M4b 인수 문서에 적는다). 종류별 값은 판정에 안 쓴다. 허용 폭·앵커
  규칙은 이 실험 결과를 보고 바꾸지 않는다(바꾸려면 새 실험).
- 판정 설정: `rule=descriptive; metric=reapply_rate`
- 비용: CPU 몇 초(GPU 없음).
- 실행 명령: `bandscribe eval exp E26`
- 결과:
  - 2026-10-17 `20261017-130024_740d6fd_exp-e26`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-17): **합친 재적용률 0.94 [0.82, 1.00]**(564/600) → 기준 0.90 **PASS**. 기타 0.86 [0.58, 1.00], 베이스 1.00, 마디선
    0.99 [0.97, 1.00].
    - 근거가 약하다: M4a 날 이후 기타 음이 실제로 바뀐 곡은 青と夏 하나뿐이다(M4a 뒤 넣은 템포 힌트 185 BPM 으로 `grid` 가 바뀌고, 격자에
      기대는 `amt_gtr`·`amt_ms1` 과 `notes` 가 다시 돌아 기타 음 4978 → 3009). 나머지 4곡은 기타 앵커 50/50 이 모두 정확히(≤0.1 ms) 붙었다
      — 그 사이 notes 의 출력이 그대로였다. 베이스(M5 정리 단계)도 표본 250개가 모두 정확히 붙었다(SC 833 → 827음처럼 몇 개만 빠짐).
    - 青と夏 기타: 50개 중 15개(정확 8, 허용 안 7). 놓친 35개는 음이 없어짐 31, 옥타브가 바뀜 4, 허용 폭(30–100 ms) 문제 0 — 허용 폭을
      60 ms 로 넓혀도 15개 그대로다. 즉 전사가 크게 바뀌면 앵커 대부분이 떨어지고, 그건 허용 폭으로 풀 문제가 아니다(목록으로 보여
      다시 고치게 하는 것이 맞다 — M4b 검토 큐). 마디선 1개는 박은 있으나 그 자리의 마디선이 없어졌다.
    - 등록 정정(M4b 리뷰, 판정에는 영향 없음): 마디선 허용은 "±⅓박" 이 아니라 `edits.BEAT_TOL` = **0.3박**이다(하네스는 edits 의 값을
      그대로 썼다). 분류 `beat_not_bar` 는 "옮길 박(`to`)이 없음" 도 포함한다(이번 1건은 마디선이 없어진 경우). "허용 폭 60 ms" 는
      같은 음높이 음이 60 ms 안에 있는지로 쟀다 — `find` 를 60 ms 로 부른 것과 같다(같은 음높이 음이 있으면 늘 붙는다). "M7a 뒤 실행"
      조건은 `load_base` 가 M7a 전 실행을 거부해서 지켜진다.

## E27
- 제목: 정체성 추적 v1 — 곡 전체 리믹스에서 파트 뒤바뀜과 할당 (M7b 인수 판정)
- 상태: 결정: 기각 (추적기 끔 유지: test 전환 중앙값 v0 0 = track 0 → ① 미달, macro 차 0.0 [0.0, 0.0] → ② 통과; dev 에서 고른 조합은 dev·test 어디서도 음을 바꾸지 않았다)
- 질문: M7b 추적기(`bandscribe.parts.track`: 창마다 v0 가 Line 별로 모은 음을 층으로 보고 서명 — 음역 중앙값, 코드·리드 층 비율,
  음 밀도, R3 창의 ILD, 클래스 — 을 만들어, 이웃 창의 층과 헝가리안으로 이은 뒤 창마다 Line 순열을 Viterbi 로 고르고 — 방출 =
  v0 사후확률, 전이 = 이어진 층이 다른 Line 으로 가는 벌점, 구간 경계에서는 `boundary` 배 — 반복 지도가 짝지은 창끼리는 같은
  Line 쪽으로 반복 조건부 최빈 2회)가 DESIGN 10 M7b 의 판정 기준을 통과하는가? ① 정체성 전환 중앙값 2 이하이고 M7a(v0)보다
  적다 ② macro 할당 정확도가 M7a 보다 낮아지지 않는다(곡 블록 부트스트랩 CI 상한이 −2점 아래로 내려가지 않음) ③ 마디 DER 를 곡별로
  보고 ④ 1절 수정이 2절로 퍼진다(단위 테스트 — `tests/test_parts_track.py::test_a_part_edit_in_verse_1_reaches_verse_2`).
- 선택지(arm): `v0`(M7a, 파이프라인 `[parts]` 설정 그대로), `track`(v0 + 추적기, dev 에서 고른 값, 리믹스의 반복 지도),
  진단 `v0|oracle_k`·`track|oracle_k`(정답 파트 수 강제, 판정에 안 씀).
- 데이터: E6 실행(`split_eval.E6_RUN`)의 Cambridge-MT 곡 중 32 s 발췌(E25)에서 기타 Line 이 둘 이상이었던 9곡(butterflyeffect,
  jetb, jokers, mistrusted, moosmusic, rememberdecember, secretariat, signejakobsen, zeno), E6 의 곡 분할. 시나리오 `P`(E6 계획)·
  `A1`·`B2`(`tierb.VARIANTS`)를 **곡 전체**(시작 0, 길이 = 곡 길이 내림)로 새로 렌더링한다(`bandscribe.parts.eval_track`).
  - 리믹스마다: SW 기타 스템(E6 의 rung, eval 캐시), 믹스의 박(Beat This!, 파이프라인 `beats` 단계)·격자·구간·반복 지도
    (파이프라인 CPU 단계), 라우터 창 = 구간 안 4마디(파이프라인 창 크기), 라벨 = 스템 창 특징 + 파이프라인 문턱(E25 와 같음).
  - 정답: E25 와 같다 — 포함한 테이크를 더블 링크로 묶은 Line, 참조 음이 16개 이상인 Line 이 활성, 시스템 음(MuScriptor, 스템
    모노, 기타 클래스)의 정답 = 일치(같은 음높이, onset ±50 ms)한 활성 Line. 테이크 뷰는 곡 전체라 시나리오끼리 같은 pcm(한 번
    전사). 활성 Line 이 둘 이상인 리믹스만 채점한다.
- 지표(리믹스마다, `bandscribe.parts.idmetrics`): 예측 Line 을 E25 의 헝가리안(배타 음 수 최대)으로 정답 Line 에 짝지은 뒤
  - `id_switches`: 정답 Line 마다 배타 음을 2 s 블록으로 묶어 블록마다 다수 정체성(짝 없는 예측·미배정도 정체성)을 정하고,
    이웃한 활성 블록 사이에 정체성이 바뀐 횟수를 Line 들에 걸쳐 합한다.
  - `assign_macro`: E25 와 같다. `block_der`: 2 s 블록마다 음 2개 이상인 정답 Line 집합과 예측 정체성 집합의 누락·오경보·혼동
    합 / 정답 Line-블록 수.
- dev 선택(dev 곡만): 격자 12개 = `w_link` {1, 2, 4} × `boundary` {0.5, 0.25} × `w_rep` {0, 1}(그 밖은 `track.DEFAULT_PARAMS`:
  `min_layer` 4, `w_post` 1, `link_max` 3, `sweeps` 2), 보수적인 것부터. dev 평균 macro 가 v0 보다 0.01 넘게 낮지 않은 조합 중
  dev 정체성 전환 합이 가장 적은 것(동률: 평균 macro 가 높은 것, 그다음 격자 순서). 고른 값은 파이프라인 `[parts] track` 의
  설정이 된다(채택될 때만 켬).
- 분석: 리믹스 쌍 차이(track − v0), 곡 블록 부트스트랩 10,000회(seed 20260930). 판정은 test 곡만, dev 값은 참고.
- 결정 규칙(test; M7b 인수 문서에 PASS / FAIL 로):
  1. 정체성 전환 중앙값(활성 Line 이 둘 이상인 test 리믹스): `track` ≤ 2 이고 `track` < `v0`(엄격히).
  2. macro: `track` − `v0` 의 평균(점) CI 상한 ≥ −2. 시나리오별로 같은 값을 내고, CI 상한이 −2 아래인 시나리오는 "퇴행 시나리오"
     로 적는다(런타임에는 시나리오를 모르므로 적기만 한다).
  3. 마디(블록) DER 를 곡별·팔별로 적는다(서술).
  - ①·② 를 모두 통과하면 `[parts] track = true` 로 채택, 아니면 끈 채로 둔다.
- 판정 설정: `rule=descriptive; metric=id_switches`
- 비용: GPU — 9곡 × (리믹스 2–3개 분리 + 박) + 뷰(리믹스 + 테이크) 전사, 약 1시간 30분을 곡마다 나눠(`data/tmp/m7b/fill_e27.py`,
  게임 중엔 대기). 채점은 CPU 몇 분.
- 등록 메모(2026-10-17, 결과 전): E25 의 32 s dev 리믹스로 v0 의 전환을 재 보니 중앙값 P 1, 나머지 0 이었다(A1·B2 는 1 파트라 전환이
  없다). 추적기도 32 s 에서는 하나도 바꾸지 않았다. 그래서 곡 전체로 잰다. 기준 ① 은 v0 의 중앙값이 0 이면 통과할 수 없다(등록한
  그대로 둔다).
- 등록 수정(2026-10-17, 채점 전 — test 곡은 보지 않음): seisyun complex(Tier A, 서술용)에서 추적기를 켜 보니 반복 창 짝이 0개였다 —
  반복 지도는 158마디 중 48마디만 짝을 지어 "창의 마디 대부분" 이 늘 성립하지 않았다. 그래서 짝 규칙을 "짝이 있는 마디 중 대부분"
  으로 고쳤다(11쌍). 그래도 바뀐 음이 0개였다: v0 사후확률이 거의 1 이라(M7a k3 의 과신) 층 하나를 바꾸는 방출 비용이 약 14 로, 연결
  ·반복 항이 이길 수 없다. 그래서 dev 격자에 사후확률 바닥 `p_floor` {0(끔), 0.1}(사후확률을 [p_floor, 1 − p_floor] 로 자름)을
  더해 격자 24개 = `p_floor` × `w_link` × `boundary` × `w_rep`, 순서는 `p_floor` 0 부터(보수적인 것 먼저). 나머지 규칙은 그대로.
- 실행 명령: `bandscribe eval exp E27`
- 결과:
  - 2026-10-17 `20261017-160506_91ed416_exp-e27`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-17): 데이터 9곡 24 리믹스(dev 15, test 9), 모두 활성 Line 2. GPU 캐시 채우기 1시간 52분(곡마다, 게임 없음).
    dev 선택 = 격자 첫 조합(`p_floor` 0, `w_link` 1, `boundary` 0.5, `w_rep` 0) — 24개 모두 dev 전환 합 79 로 같았고, 고른 조합은
    음을 하나도 바꾸지 않았다. `w_link` 2 이상인 조합은 dev jetb P 에서 19음을 바꿨다(dev macro 0.6118 → 0.6115, 전환은 그대로). **test 판정**: ① 전환 중앙값 v0 **0** = track **0** → 미달
    (등록 메모대로 v0 가 0 이면 통과할 수 없다), ② macro 차 **0.0 [0.0, 0.0]**점 → 통과, 퇴행 시나리오 없음 → **FAIL**, 추적기는
    끈 채로 둔다. 블록 DER(test, 곡별 v0 = track): rememberdecember 0.378, signejakobsen 0.350, zeno 0.347(dev 0.207–0.667).
    - 중앙값이 0 인 까닭: v0 가 A1·B2 리믹스 15개 중 15개, P 9개 중 1개(jokers, 곡 전체)를 1 파트로 세어 바뀔 정체성이 없다. 2 파트로
      센 P 리믹스 8개의 전환은 27·20·18·10·4(dev), 23·2·0(test) — 곡 전체에서는 많다. dev 곡을 창별로 보면(mistrusted P) 모든 창이
      R3 이고 사후확률이 ~1.0 이며, 틀린 것은 창 전체가 아니라 ILD 가 반대로 읽힌 음 하나하나다(둘째 기타가 성기어 블록 하나에 1–3음).
      창 단위 순열은 고칠 것이 없어서 바꾸지 않았다 — 맞는 판단이지만 전환은 줄지 않는다.
    - 진단(정답 파트 수 강제): A1·B2 에서 v0 가 같은 위치의 두 파트를 이미 가르는 곡이 있다 — zeno A1 macro 0.967, B2 1.000,
      signejakobsen A1 0.649. mistrusted A1·moosmusic A1 은 2 파트가 되지만 가르지 못하고(0.500, 0.449), 나머지는 강제해도 한 파트가
      비어 1 파트다. 같은 위치 파트의 병목은 정체성 추적보다 **파트 수 세기**다.
    - 실행 메모(M7b 리뷰, 판정에는 영향 없음): E27 의 창(`eval_track.bar_windows`)은 파이프라인 라우터와 달리 구간 끝의 1마디 창을
      앞 창에 합치지 않았고 `end_bar` 가 없어, 반복 창 짝이 4마디로 가정되어 24개 중 20개에서 라우터 창으로 짝지은 것과 달랐다.
      `w_rep` 0 과 1 의 결과가 같아 판정은 바뀌지 않는다. 고친 코드는 다음 데이터부터 라우터와 같다.
    - 다음 길: 음 단위 평활(같은 스트럼·선율 안의 ILD 이상치)과 사후확률 보정(M10), 그리고 좌우 근거가 없을 때의 파트 수(기호·음색
      근거를 게이트해서; E25 에서는 기타 1대를 갈랐다).

## E28
- 제목: 같은 위치 파트 수 — 넓은 더블 옆 가운데 파트의 "폭" 근거 (M7a k1·k2 의 A 후속)
- 상태: 결정: 판단 보류 (기본값 끔 유지: ① 기타 1대 15/15 통과, ③ 퇴행 없음 통과, ② A1 파트 수 v0 0/6 → 3/6·macro +17.0 [0.0, 26.6]점 — CI 하한이 0 이라 미달; test 곡 3개 중 rememberdecember 가 바뀌지 않음)
- 질문: 라우터 `R2`(넓은 더블) 창에서 뚜렷이 가운데인 음(`ms_db ≥ width_ms_hi`, `coh ≥ width_coh_hi`: 한 소스가 가운데)과 뚜렷이
  넓은 음(`ms_db ≤ width_ms_lo`: 더블의 두 테이크)이 각각 `width_n_min` 개 이상이면 두 스트림으로 세고(`reid` 의 `width`), 그 근거로
  파트를 나누면(가운데·넓은 음을 씨앗으로, 사후확률에 `R2` 창의 폭 항 = 6 × tanh((ms_db − 4 dB) / 3 dB) × 파트의 평균 폭 —
  좌우 항과 같은 꼴, 폭 근거로 만든 파트일 때만) DESIGN 4.5 의 A(넓은 리듬 더블 + 가운데 파트)를 세고 가르면서 기타 1대는 1 파트로
  남는가? 근거: E25(A1 전부 1 파트), E27 진단(정답 파트 수를 주면 가르는 곡이 있다).
- 선택지(arm): `v0`(파이프라인 `[parts]` 그대로), `width`(v0 + `use_width`, dev 에서 고른 문턱).
- 데이터(곡 분할은 E6):
  - E25 의 32 s 리믹스(P, ADT, A1, B2, N1 — 캐시에서만 다시 만듦, GPU 거부),
  - E27 의 곡 전체 리믹스(P, A1, B2),
  - 새 기타 1대 대조군 `D1`(`tierb.VARIANTS`: A 의 더블만 좌우 끝, 나머지 테이크 끔 → `R2`, 정답 1 파트), 곡 전체, 더블이 있는
    곡마다(`eval_track` `<song>.d1.json`; 등록 전에 GPU 로 채우기 시작했고 결과는 보지 않았다).
- 지표(리믹스마다): `k_pred`, `k_correct`, `assign_macro`(E25 와 같음; 정답 1 파트 리믹스는 파트 수만).
- dev 선택(dev 곡만): 격자 16개 = `width_ms_hi` {10, 6} × `width_coh_hi` {0.95, 0.9} × `width_ms_lo` {0, 2} × `width_n_min`
  {6, 4}, 이 순서(보수적인 것 먼저). 목적(사전식): ① 기타 1대 리믹스(ADT, N1, D1)를 1 로 센 수 ② 모든 dev 리믹스의 파트 수 정확
  비율 ③ A1 평균 macro. 동률은 격자 순서. 고른 값은 채택될 때 `[parts]` 기본값이 된다.
- 분석: 곡 블록 부트스트랩 10,000회(seed 20260930), 리믹스 쌍 차이(width − v0, 점). 판정은 test 곡만.
- 결정 규칙(test):
  1. 기타 1대: test 의 ADT·N1(32 s)·D1(곡 전체) 리믹스를 `width` 가 모두 1 로 센다.
  2. A1: `width` 가 파트 수를 맞힌 A1 리믹스(32 s + 곡 전체)가 `v0` 보다 많고, A1 macro 차 평균 ≥ +10점이며 CI 하한 > 0.
  3. 퇴행 없음: A1·기타 1대가 아닌 시나리오(P, B2)마다 macro 차 CI 상한 ≥ −2점.
  - 셋 다 통과하면 `[parts] use_width = true`(고른 문턱)로 채택, 아니면 끈 채로.
- 판정 설정: `rule=descriptive; metric=k_correct`
- 비용: D1 GPU 약 20분(곡마다, 게임 중엔 대기 — `data/tmp/m7b/fill_d1.py`), 채점 CPU 몇 분.
- 등록 메모(2026-10-17, 결과 전): dev 곡에서 가운데(`ms` ≥ 6, `coh` ≥ 0.9)·넓은(`ms` ≤ 2) 음을 재 보니 A1 의 가운데 파트는 coh 0.72–0.999·
  ms 6.6–33 dB, 더블은 coh 0.27–0.56·ms −1–0 dB 였다. 그 값(문턱 6/0.9/2/6)으로 dev 곡 전체 A1 jokers·mistrusted 가 2 파트(macro 0.5
  → 0.72, 0.75)가 되고 dev ADT·N1 10개는 1 파트로 남았다(moosmusic A1 은 가운데 파트가 R0 창에 있어 0, moosmusic B2 는 R2 창 3개로
  2 파트가 되나 macro 0.5 그대로). 32 s 발췌에서는 어느 A1 도 바뀌지 않았다(가운데 파트 음이 창마다 적다). ADT 는 `R1` 이라 이 근거를
  보지 않는다(R1 창으로 재면 jokers ADT 가 4창 모두 두 스트림이 됐다 — 그래서 R2 만).
- 실행 명령: `bandscribe eval exp E28`
- 결과:
  - 2026-10-17 `20261017-170948_0d5774a_exp-e28`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-17): D1 대조군 6곡(더블이 없는 3곡은 없음), GPU 22분. 항목: 32 s 리믹스(E25 캐시) + 곡 전체 E27 24개 + D1 6개.
    dev 선택 = `width_ms_hi` 6, `width_coh_hi` 0.9, `width_ms_lo` 0, `width_n_min` 6(dev 기타 1대 12/13 — 놓친 1개는 moosmusic D1 로
    v0 도 2 로 센다; dev 파트 수 정확 0.581, A1 macro 0.582). **test 판정**: ① 기타 1대(ADT·N1 32 s, D1 곡 전체) **15/15** 를 1 로
    셈(v0 도 15/15) → 통과. ② A1 파트 수를 맞힌 리믹스 v0 **0/6 → 3/6**, macro 차 **+17.0 [0.0, 26.6]**점 — 평균 ≥ +10 은 넘었지만
    CI 하한이 0 이라(> 0 이어야) 미달. ③ P·B2 차 0.0 → 통과. **FAIL → 기본값은 끈 채로**(`[parts] use_width = false`).
    - 바뀐 A1: zeno 곡 전체 0.50 → **0.99**(R2 창 15개), signejakobsen 32 s 0.50 → **0.99**, 곡 전체 0.50 → 0.55; rememberdecember 는
      32 s·곡 전체 모두 근거 0 창(가운데 파트가 R2 창에 보이지 않음) — test 곡이 3개뿐이라 한 곡이 그대로면 곡 블록 CI 하한이 0 이 된다.
      dev: jokers 0.74, mistrusted 0.75(곡 전체), moosmusic 그대로(가운데 파트가 R0 창에), moosmusic B2 는 2 파트가 되나 macro 0.5.
    - 기타 1대를 잘못 둘로 센 것은 dev·test 어디에도 없다(dev moosmusic D1 은 v0 부터 2 파트 — 좌우 창이 있음).
    - Tier A(서술): 5곡 모두 좌우 근거로 이미 2 파트, 폭 근거 창 0–2개, 배정 그대로.
    - 다음 길: A1 이 더 있는 곡으로 다시 재면(새 실험) 판정이 바뀔 수 있다. 그전까지는 곡마다 `--set parts.use_width=true` 나
      `hints.toml` `[parts] use_width = true` 로 켤 수 있다(기타 1대 리믹스 28개에서 폭 근거가 새로 만든 오검출 0개).

## E10
- 제목: B/F 반복 차감 리드 뷰 on/off (M8, DESIGN S5 R0 ①)
- 상태: 결정: 기각 (실행 안 함 — Tier B 상한이 기준 아래. 반복 차감 뷰는 만들지 않고 S3.5 반복 잔차 1차는 솔로 게이트 근거로만 남는다)
- 질문: 좁은 믹스(B/F, 라우터 R0)에서 구간 정렬 중앙값 차감(repet3) 잔차를 전사해 리드 Line 을 만들면 리드 Line tick F1 이 M7b 보다
  0.05 이상 오르는가(DESIGN §10 M8 완료 기준, "합성·Tier B B/F 리믹스")?
- 데이터: E27 의 곡 전체 Tier B `B2`(두 테이크 모두 가운데 = 설계의 B; F 는 기타에는 B 와 같아 E6 에서 뺐다) 9곡 중 **한 테이크가
  단선율(onset 40 ms 묶음 중 한 음짜리 비율 ≥ 0.6), 다른 테이크가 코드(≤ 0.3)인 4곡**: jetb_tothewolves(g1 0.72 / g0 0.27),
  jokers_seaofleaves(g5 0.97 / g2 0.15), secretariat_overthetop(g0 0.63 / g1 0.28), zeno_signs(g6 1.00 / g3 0.00). 나머지 5곡은 두
  테이크가 모두 코드이거나 모두 단선율이라 리드 정답이 없다. Tier B 테이크에는 역할 라벨이 없어(`lines.yaml` `role: null`) 이 기계
  라벨을 썼다.
- 실행 전 상한 분석(2026-10-17, `docs/research/experiments/e10_ceiling.py`, CPU, 캐시만): 일부러 후하게 —
  모든 창이 R0 경로를 탄다고 보고, 출현 2회 묶음도 넣고(중앙값 대신 최소), 리드가 묶음 출현의 소수(2회면 1회)에서만 치는 마디마다
  참 리드 음을 모두 리드 Line 에, 나머지를 모두 밖으로, 섞인 믹스 전사가 놓친 리드 참조 음까지 더한다. 그 밖은 M7b 기본 배정 그대로.
  리드 Line F1 = Line 중 최대(음 단위, 재현율 분모 = 리드 테이크의 참조 음).
  - 결과: jetb 0.672 → 0.672(+0.000, 소수 출현 마디의 리드 참조 음 0개), jokers 0.114 → 0.117(+0.003, 9개), secretariat 0.487 → 0.500
    (+0.013, 25개), zeno 0.216 → 0.218(+0.002, 8개). **평균 +0.004** — 기준 +0.05 의 1/10 아래. 4곡 모두 M7b 는 1 파트(K=1).
  - 까닭: 실제 곡의 리드(단선율 테이크)도 반복 구간마다 같이 친다. 중앙값 차감은 "반복되는 소리"를 지우므로 반복되는 리드도 지운다.
    반복 묶음 마디에 든 리드 음 중 소수 출현에 있는 것은 곡마다 0–25개뿐이고, 솔로처럼 한 번만 나오는 구간은 정의상 반복이 없다.
    zeno·jokers 는 3회 이상 출현 묶음이 아예 없어 S3.5 기본(`resid_min_occurrences = 3`)으로는 잔차가 계산되지도 않는다.
- 합성 장면은 돌리지 않았다: repet3 식 장면은 리드를 반복의 소수 출현에만 두도록 만들어져 기제만 재고, 완료 기준은 합성과 Tier B
  둘 다다. Tier B 상한이 기준 아래면 어떤 합성 결과로도 기본값을 켤 수 없다.
- 한계: 리드가 있는 곡 4개(dev 3, test 1), 음 단위 F1(tick F1 이 아님 — 상한의 크기 차이가 커서 판정이 바뀌지 않는다).
- 다음 길: B 의 리드는 반복이 아니라 다른 근거가 필요하다 — 솔로 게이트 안의 기호 분할(단선율·음역; E25 에서 기호만으로 가르면
  기타 1대도 갈랐다 → 게이트·대조군을 둔 새 실험), 또는 MVSEP 101(클라우드, 사용자 계정·업로드 동의, E11).
- 독립 리뷰(2026-10-18): 상한 계산은 후하고 리드 Line F1 도 바르게 셌다. 다만 **수치는 "소수 출현 마디"의 정의에 달려 있다.**
  - 가장 후한 다른 정의(리드가 어느 한 출현에라도 빠진 반복 마디 전부, 어디서나 최소 배경)로는 평균 **+0.037**이다(jetb +0.063, secretariat
    +0.081). 여전히 +0.05 아래라 결론은 그대로다.
  - 그 정의에서는 리드 음이 jetb 267·secretariat 141개다. 그래서 위의 "곡마다 0–25개"·"기준의 1/10 아래"는 등록한 정의에서만 맞다.
  - 참조 음으로 리드 위치를 정하면 +0.014 다.

## E29
- 제목: 반복 합의 — 반복 구간끼리 음을 맞대어 한 출현에서만 빠진 음을 채우기 (M8, DESIGN S7 "반복 합의")
- 상태: 결정: 기각 (기본값 끔 — test F1 차 −0.02 [−0.17, +0.06]점, 기준 +2.0 미달; Tier A 5곡은 투표한 칸 0)
- 질문: 반복 지도(`repeat_map.json`)의 같은 묶음 출현들을 마디·박 위치로 맞대어, 다른 출현 대부분에 있는데 이 출현에만 없는 기타 음을
  저신뢰로 더하면 합친 기타(`guitar_all`) 음 F1 이 2점 이상 오르는가(DESIGN §10 M8 "반복 합의가 합친 기타 F1을 2점 이상 올린다. 아니면
  끈다")? 변주(한 출현에만 있는 음)는 DESIGN 대로 지우지 않고 표시만 한다(`repeat_lone`) — 이 실험의 F1 에는 영향이 없다.
- 방법(`bandscribe/amt/consensus.py`): 출현 3회 이상 묶음, 출현들의 공통 마디 수만큼 마디 k 를 맞댄다(박 수가 다른 마디는 건너뜀).
  음의 위치 = (k, 마디 안 박 위치 — 그 출현의 박 격자로, 출현의 `offset_s` 를 뺀 시각), 같은 음높이·`tol` 박 이내면 같은 음.
  - 투표자: 대상 출현 o 의 마디 k 에 음이 `min_notes`(3) 개 이상일 때, 다른 출현 o′ 중 o 의 그 마디 음의 `sim_min` 비율 이상이 o′ 에서
    맞는 출현(마디가 실제로 같은 음악인지 확인 — 반복 지도는 믹스 전체로 만든 것이라).
  - 더하기: 투표자가 2명 이상이고, o 에 없는 음 자리를 투표자 `ceil(add_frac × 투표자 수)` 명 이상(최소 2)이 가지면 그 자리(투표자 위치의
    중앙값, 길이도 중앙값, 악기는 최빈값)에 음을 더한다(`added: "repeat"`, 지지 비율 기록).
- 선택지(arm): `off`(지금 그대로), `consensus`(dev 에서 고른 값).
- 데이터: E27·E28 의 곡 전체 Tier B 리믹스 — P·A1·B2(9곡) + D1(6곡), `data/eval/parts_track`(GPU 없음, 캐시만). 기타 음 = 그 리믹스의
  분리 기타 스템 모노의 MuScriptor 기타 음(`sys`), 정답 = 리믹스에 든 테이크들의 단독 전사 합(같은 음높이 50 ms 안은 하나로).
  격자·구간·반복 지도는 그 리믹스에 파이프라인 단계를 돌린 것. 곡 분할은 E6(test: rememberdecember, signejakobsen, zeno).
- 지표(리믹스마다): 합친 기타 음 F1(onset 50 ms, ±50 cents, 1:1 짝짓기 `eval.metrics.note_prf`), 점수는 ×100. 보조: 더한 음 수, 그중
  정답과 맞는 비율(정밀도), `repeat_lone` 수, 합의가 닿은 마디 비율.
- dev 선택(dev 곡만): 격자 8개 = `add_frac` {1.0, 2/3} × `sim_min` {0.7, 0.5} × `tol` {0.125, 0.0625}박, 이 순서(보수적인 것 먼저).
  목적: dev 리믹스 F1 차 평균, 동률은 격자 순서.
  - 등록 정정(2026-10-17, 구현 중·결과 전): `0.67` 은 투표자 3명에서 ceil(2.01) = 3 이 되어 1.0 과 같아지므로 의도한 2/3 으로 적는다.
- 분석: 곡 블록 부트스트랩 10,000회(seed 20260930), 리믹스 쌍 차이(consensus − off, 점). 판정은 test 곡만.
- 결정 규칙(test):
  1. F1 차 평균 ≥ +2.0점이고 CI 하한 > 0.
  2. 퇴행 없음: 시나리오(P, A1, B2, D1)마다 F1 차 평균 ≥ −1.0점.
  - 둘 다 통과하면 `notes` 단계에 켜고(기본값), 아니면 끈 채로 둔다(코드는 남기고 음성 결과로 기록).
- 서술(판정 없음): Tier A seisyun complex 를 사용자 Guitar Pro 악보와 대조한 합친 기타 F1 전후(`data/tmp/sheets_sc` 의 대조 방식), Tier A
  5곡의 더한 음 수.
- 판정 설정: `rule=descriptive; metric=f1`
- 비용: CPU 몇 분.
- 실행 명령: `bandscribe eval exp E29`
- 결과:
  - 2026-10-17 `20261017-184442_9a6cfa9_exp-e29`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-17): dev 선택 = `add_frac` 1.0, `sim_min` 0.5, `tol` 0.0625박(dev F1 차 +0.11점 — 격자 8개 모두 +0.03…+0.11,
    `add_frac` 은 차이 없음: 투표자 3명 이상인 칸이 거의 없다). **test 판정**(리믹스 12개, 곡 3개): F1 차 **−0.02 [−0.17, +0.06]**점 →
    ① 평균 ≥ +2.0 미달 → **FAIL, 기본값 끔**(합의는 `notes` 단계에 연결하지 않았다; 코드 `amt/consensus.py` 와 실험만 남김).
    ② 시나리오별 평균 P +0.03, A1 −0.20, B2 +0.10, D1 0.00 → 퇴행 규칙은 통과.
    - 닿은 범위가 작다: test 12개 중 6개, dev 18개 중 9개 리믹스에서만 투표한 칸이 있고(칸의 0–75 %), 더한 음은 test 275개·dev 275개.
      더한 음의 정밀도 0.65(test)·0.67(dev) — 원래 음들의 정밀도(대략 0.7–0.8)보다 낮아 더할수록 F1 이 거의 그대로이거나 내려간다.
    - zeno·jokers·rememberdecember B2 는 출현 3회 이상 묶음이 없어 0.
    - Tier A(서술): 5곡 모두 투표한 칸 **0개**, 더한 음 0 — seisyun complex 의 합친 기타 마디 F1(사용자 악보 대조) 73.95 → 73.95.
      원인을 보니 반복 지도가 같은 묶음으로 묶은 마디들(seisyun complex 32·73·117 마디)의 기타 코드가 서로 달랐다(믹스 전체로 만든
      반복 지도는 마디 단위 기타 음을 맞대기에는 거칠다). "같은 음악인지" 확인(`sim_min`)이 이를 걸러 잘못 더한 음은 없었다.
    - 다음 길: 합의는 반복 지도가 아니라 기타 음 자체로 마디 짝을 찾아야 쓸 만해진다(예: 마디별 음 집합 유사도로 짝, M7b 의
      `repeat_pairs` 와 같은 생각). 새 실험으로 남긴다.
  - 독립 리뷰(2026-10-18) — 결론(기각)은 그대로, 설명 셋을 바로잡는다:
    1. **더한 음이 이미 있는 음과 겹쳤다.** "빠졌다"를 박 허용치(0.0625 박 ≈ 30 ms)로만 판정해, 더한 음 550개 중 131개(24 %)가 이미 있는 같은
       음 50 ms 안에 들어갔다(F1 짝짓기는 50 ms). 고쳤다(`consensus.py` `dup_s` 0.05 s, 테스트). 겹침을 빼면 test 는 **+0.11 [+0.00, +0.19]**점으로
       여전히 기준 +2.0 미달이다.
       - 위의 "정밀도 0.65" 는 더한 음만 따로 맞댄 값이다. 실제로 새로 맞힌 음은 550개 중 248개(0.45)였다.
       - "원래 음보다 정밀도가 낮아 더할수록 손해"라는 설명은 **틀렸다**. 정밀도 q 인 음을 더하면 q > F1/2(여기서 약 0.4)일 때 F1 이 오른다.
         F1 을 떨어뜨린 것은 겹친 음이다.
       - 상한: 다른 출현 둘 이상이 가진 놓친 참조 음을 모두 더하는 오라클도 +0.49 이고, 출현을 다시 맞추면 +0.67 이다.
    2. **Tier A 투표 0 의 원인은 출현 시작 마디의 어긋남(구간 위상)이었다.** "반복끼리 전사가 반 넘게 다르다"가 아니다.
       - seisyun complex 의 S5 를 +4 마디 옮기면 마디 유사도가 0.14 → 0.69 가 된다. 기타 마디는 실제로 반복된다(마디의 96 % 가 다른 마디와 0.5
         이상 맞고, 중앙값 0.72).
       - 출현을 다시 맞춰도 Tier A 는 투표 0, Tier B test 는 +0.007 이었다. 위 "다음 길"(기타 음으로 마디 짝 찾기)도 리뷰가 시험했고, 격자 6개
         모두 test 음수(−0.04…−1.73)였다.
    3. 수치 정정:
       - zeno D1 은 투표 36칸·더한 음 91개로 test 최다다("zeno 0" 은 P·A1·B2 만의 이야기).
       - 투표한 칸 비율은 최대 81 %(signejakobsen P)다.
       - jokers·rememberdecember·zeno "3회 이상 묶음 없음"은 B2 리믹스 이야기다. rememberdecember 의 P·A1 에는 있다.

## E30
- 제목: 블리드 감점 — 보컬·베이스를 따라가는 가짜 기타 음 표시 (M8, DESIGN S7 "감점")
- 상태: 결정: 판단 보류 (vocal: test 18음 < 20, 정밀도 0.56) / 기각 (bass: 정밀도 0.70 < 0.80) — 둘 다 기본값 끔
- 질문: 아래 두 규칙이 표시한 기타 음은 실제로 가짜인가(정밀도 ≥ 0.80, DESIGN §10 M8 "보컬 블리드 감점의 정밀도가 80 % 이상"), 그리고
  표시한 음을 탭에서 빼도(뷰어 후보로 남김 — 지우지 않음) 합친 기타 F1 이 떨어지지 않는가?
- 규칙(`bandscribe/amt/bleed.py`, 규칙마다 따로 판정):
  - `vocal`: 보컬 f0(pyin 16 kHz, 80–1000 Hz, hop 10 ms, 유성 확률 > 0.5)로 — 음의 앞 300 ms(더 짧으면 음 길이) 프레임의 90 % 이상이 음높이
    ±50 cents, 음 길이 전체로도 80 % 이상, 그리고 보컬이 그 음에 **도착**한다(시작 뒤 60 ms 안에 그 음이고, 시작 전 150–60 ms 에는 그 음이
    아님). 기타가 노래를 같은 음으로 치는 경우(코드음이 노래 음과 같음)를 피하려는 마지막 조건이다.
  - `bass`: E2(MIDI 40) 아래 기타 음 중, 시작부터 200 ms 의 베이스 f0 유성 프레임(신뢰 > 0.5)의 50 % 이상이 같은 음이름(DESIGN 규칙;
    베이스 음 대신 베이스 f0 — Tier B 리믹스에는 원시 베이스 전사가 없다. 파이프라인에서는 `bass_f0` 단계의 f0).
- 등록 전 dev 탐색(2026-10-17, dev 곡만, `data/tmp/m8/vox_explore*.py`): dev 기타 음 40,080개 중 가짜 42 %. 보컬 따라감(앞 300 ms 90 %)
  단독 382음 정밀도 0.70, +도착 107음 0.82, +도착+길이 104음 0.84, 리드 음역(≥ C4)만 0.42, 다른 음과 같이 시작하지 않는 음은 9음뿐.
  → `vocal` 규칙은 "+도착+길이". `bass` 는 dev 에서 E2 아래 음 560개 중 80 % 가 가짜(베이스 조건 전, 탐색은 이것뿐).
- 데이터: E29 의 곡 전체 Tier B 리믹스(P·A1·B2·D1, 캐시만). 보컬 f0 = 곡의 보컬 파트 합(리믹스는 분리 기타 스템만 캐시해 두어 분리 보컬이
  없다; 파이프라인은 분리 보컬 스템), 베이스 f0 = 베이스 파트 합(`bass.f0.pyin_track`). **가짜 = 그 음에 맞는 테이크 정답이 없음**(E25 의
  truth 가 빈 집합). 곡 분할은 E6(test: rememberdecember, signejakobsen, zeno).
- 지표(규칙마다): 표시한 음 수, 정밀도(test 곡의 표시한 음을 모두 모아), 곡 블록 부트스트랩 10,000회(seed 20260930) CI, 합친 기타 F1 차(표시한
  음을 뺀 것 − 그대로, 점; E29 의 참조·짝짓기).
- 결정 규칙(test, 규칙마다):
  1. 정밀도 ≥ 0.80. 표시한 음이 20개 미만이면 판단 보류.
  2. F1 차 평균 ≥ −0.5점.
  - 둘 다 통과한 규칙만 `notes` 단계에서 기본으로 켠다(표시한 음은 `guitar_all` 에서 빼고 뷰어 후보로 남김). 아니면 끈 채로.
- 서술(판정 없음): seisyun complex — 표시한 음 중 그 마디의 사용자 악보(기타 1·2 합)에 없는 음높이 비율(사람이 들어 보는 "50음 수동 점검"의
  대용; 귀 점검은 사용자 몫으로 남긴다), 나머지 Tier A 곡의 표시 수.
- 판정 설정: `rule=descriptive; metric=precision`
- 비용: CPU — 보컬 pyin 곡당 약 1분, 채점 몇 초.
- 실행 명령: `bandscribe eval exp E30`
- 결과:
  - 2026-10-17 `20261017-191338_888f213_exp-e30`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-17): f0 는 등록 전 탐색 때 같은 알고리즘으로 만든 것을 캐시로 옮겨 썼다(`data/tmp/m8/seed_bleed_cache.py`).
    - **`vocal` → 판단 보류(기본값 끔)**: test 표시 **18음**(기준 20음 미만), 그중 가짜 10(**0.56**, rememberdecember 4/0, signejakobsen
      6/6, zeno 8/4). F1 차 +0.01 [−0.02, +0.05]. dev 의 0.84(104음)는 moosmusic 한 곡(82/82)이 끌어올린 값이었다 — 다른 dev 곡은
      14음 중 1, 7음 중 3. 곡마다 들쭉날쭉해 기준을 넘는다고 볼 근거가 없다.
    - **`bass` → 기각(기본값 끔)**: test 표시 **198음**, 가짜 139(**0.70** [0.64, 0.71] < 0.80). rememberdecember 170음 중 가짜 121 —
      E2 아래로 실제로 치는 기타가 베이스와 같은 음을 많이 친다(DESIGN 이 걱정한 드롭 튜닝의 경우). F1 차 +0.05 [−0.00, +0.14]. dev 는
      54음 0.93.
    - Tier A(서술): 걸리는 음이 곡당 0–5개(`vocal` seisyun complex 2·KH 4, `bass` AZ 4·KH 5 — 2026-10-18 정정: 처음에 KH 와 AZ 의 작업 번호를 바꿔 적었다. KH 의 입력 wav 는 기타를 뺀
      음원이었다) — 무음 게이트와 `guitar+present` 마스크가
      이미 거르고 있다. seisyun complex 의 `vocal` 2음은 **둘 다 사용자 악보의 그 마디에 있는 음**이었다(가짜 아님).
    - 남는 것: `amt/bleed.py`(규칙·보컬 f0)와 실험. 파이프라인에는 연결하지 않았다. 사람 귀 50음 점검은 걸리는 음이 너무 적어 하지 않았다.
  - 독립 리뷰(2026-10-18): **`bass` 의 결과는 거짓 음성이었다.** test 에서 "진짜 기타 음을 잘못 걸렀다"로 셈한 42음은 정답(단독 테이크 전사)의
    옥타브 아래 유령이었다(위의 "드롭 튜닝" 설명은 틀렸다). 정답을 고쳐 E30b 로 다시 쟀고 채택됐다(아래). `vocal` 은 그대로 판단 보류.

## E31
- 제목: 코드 차트 라벨 정확도 — seisyun complex 사용자 악보의 코드 이름 대조 (M9, DESIGN 6.5)
- 상태: 결정: 기각 (근음 + 부류 0.546 < 0.70 — 리뷰가 고친 채점기 기준, 처음 0.543; 코드는 chords/chords.txt 로만, 악보에는 넣지 않음)
- 질문: 코드 인식(`bandscribe/tab/chords.py`: 박마다 기타 음의 길이 가중 음이름 분포 × 템플릿 10종 코사인, 베이스 근음 가산점, 코드가
  바뀔 때 벌점을 주는 Viterbi)이 사용자 악보의 코드 이름과 **근음 + 부류**로 70 % 이상 맞는가(DESIGN §10 M9 "코드 라벨(근음 +
  maj/min/5/7) 70 % 이상")?
- 매개변수: 등록 시 기본값 그대로 — `change_cost` 0.15, `bass_bonus` 0.1, `size_prior` 0.02, `min_fill` 0.1, `empty_cost` 0.25. 맞출
  데이터가 없어(코드 정답은 이 곡 하나) 튜닝하지 않고 정했다. 결과를 본 뒤 바꾸면 새 실험이다.
- 입력: seisyun complex 작업의 양자화된 합친 기타 음(`quant.json` `guitar_all`)과 베이스 음(`bass`), 박 격자(`grid.json`).
- 정답(`data/tmp/sheets_sc/pdf_marks.py` → `gtr1_marks.json`; 악보는 사용자 것이라 저장소 밖): Ele. Guitar 1 PDF 의 코드 이름(보표 위 8–20 pt,
  글자 높이 6 pt 이상, 이름이 A–G 로 시작), 갈린 이름은 붙인다("A7(" + "b13)", 마디선에 걸친 "C" + "#dim7").
  - 위치: 이름의 바로 오른쪽 음표 열(이름 x 이상, 마디 폭의 0.2 안)의 박. 열의 박 = (열 위치 − 첫 열 위치) / (1 − 첫 열 위치) × 박 수,
    반 박으로 반올림(열이 없으면 이름 위치에 같은 식). Guitar Pro 는 박을 길이에 비례해 띄우므로 근사다.
  - 코드는 다음 코드 이름까지 이어진다. **채점하지 않는 곳**: 기타 1 이 쉬는 마디(`rest_only`), "N.C.", 첫 코드 이름 전.
  - 마디 대응은 `eval_sc.py` 와 같다(악보 박 j = 우리 전역 박 j − 3).
- 부류 줄이기(정답·우리 같은 규칙): "sus" 가 있으면 sus, "dim" 이면 dim, "aug"/"+" 면 aug, 근음 뒤가 "m"(maj 아님)으로 시작하면 min,
  "5" 만이면 5, "7"·"9"·"11"·"13" 이 있고 "maj" 가 없으면 7, 그 밖은 maj(maj7, 6, add9 등). 슬래시 베이스("C/Bb")는 버린다. 근음은 음이름
  (이명동음 같음).
- 지표: 반 박 단위 정확도(근음 + 부류 일치 비율, 채점하는 반 박 전부). 보조: 근음만 정확도, 그리고 **정답 음**(PDF 의 기타 1·2·베이스 음을 열
  위치의 박으로 놓고 다음 열까지 울린다고 본 것)에 같은 인식기를 돌린 정확도 — 인식기 자체의 한계와 전사 오류를 가르려고.
- 결정 규칙: 정확도 ≥ 0.70 이면 M9 의 코드 기준 통과이고, 악보(alphaTex·GP5)에 코드 이름을 기본으로 넣는다. 아래면 기준 미달이고,
  코드는 `chords/chords.txt`(자동 추정 표시)로만 내보내고 악보에는 넣지 않는다(설정으로 켤 수 있게).
- 한계: 곡 1개(편곡자 Rika 의 악보), 정답 위치는 반 박 근사.
- 판정 설정: `rule=descriptive; metric=accuracy`
- 비용: CPU 몇 초.
- 실행 명령: `data/tmp/m9/e31_sc.py`(사용자 악보가 필요해 저장소 밖 스크립트) — 결과는 아래에 손으로 적는다.
- 결과:
  - 2026-10-18 `data/tmp/m9/e31_sc.py`(커밋 위 작업 트리, 등록한 기본값): 채점한 반 박 1,164개, 정답 코드 이름 189개(min 94, maj 57,
    7 26, dim 5, sus 5).
    - **우리 음: 근음 + 부류 0.543, 근음만 0.623 → 기준 0.70 미달(FAIL).** 코드는 `chords/chords.txt`(자동 추정 표시)로만 내보내고
      악보에는 넣지 않는다.
    - 진단 — 악보의 음(기타 1·2 + 베이스)에 같은 인식기: 0.601, 근음만 0.706. 전사 오류보다 **인식기 자체가 이 곡의 화성에 약한 것**이
      더 크다. 많이 틀린 것: 정답 Dm9 → 우리 Cadd9(기타가 근음 D 를 빼고 C·E·F·A 를 짚는다), Dm7 → F5, Am7 → C5, A7(b13) → Gsus2,
      Bbmaj7 → Gsus2. 베이스 근음은 박마다 0.1 가산점일 뿐이라 근음 없는 보이싱을 바로잡지 못한다.
    - 다음 길(새 실험): 베이스 음을 분포에 넣기(사람이 화성을 읽는 방식), 텐션 템플릿(m9, maj9, 9, 7sus4, 6, m6, dim7) 더하기. 정답 곡이
      하나뿐이라 이 곡으로 맞추면 과적합이다 — 다른 곡의 코드 정답(사용자가 가진 코드 악보)이 있어야 판정할 수 있다.
  - 독립 리뷰(2026-10-18): 채점기가 슬래시 코드(C/Bb ×2, Gm/C, D/F)를 이름으로 못 읽어 버렸다(등록은 "베이스만 버린다"). 고쳐서 다시
    재니 **0.546**(근음만 0.639), 진단 0.601(근음만 0.720), 정답 코드 이름 193개 — 판정 그대로. 확인된 것: 악보 박 대응 오프셋 3 이 가장 잘
    맞는다(1–5: .277/.393/.543/.448/.317). 위치를 ±1 박 봐줘도 0.614. 6add9 → maj, "C"+"#dim7" 붙임 3회, "A7("+"b13)" 18회. 참고로
    dim7 은 네 근음 어느 것으로도 부를 수 있는데 등록 규칙은 근음 하나만 맞다고 봐(Gdim ↔ C#dim7 16 반 박, 약 1.4점) 조금 짜다.

## E32
- 제목: 벤딩 검출(f0 규칙) — seisyun complex 사용자 악보의 벤딩 표시 대조 (M9, DESIGN 6.4 1층)
- 상태: 결정: 기각 (seisyun complex 벤딩 16곳 검출 0 — 벤딩이 모두 화음 속이라 단선율 f0 규칙이 닿지 않음; 표시는 제안만)
- 질문: f0 규칙(`bandscribe/tab/techniques.py`, 등록 시 `BEND_DEFAULTS`: 상승 80–450 cents, 시작 40 ms 뒤부터, 꼭대기는 시작 50 ms 뒤 이후,
  시작이나 꼭대기가 음 자신의 음높이 ±70 cents, 한 파트에서 음 길이의 절반 넘게 겹치는 음이 없는 음만)이 찾은 벤딩이 사용자 악보의
  벤딩과 맞는가? DESIGN §10 M9 의 [판정](벤딩 F1 ≥ 0.6, 깊이 클래스 ≥ 70 %)은 GOAT/Guitar-TECHS(다운로드 필요)로 재게 되어 있어 여기서는
  **서술**이고, DESIGN 6.4 의 표시 규칙("측정 정밀도 ≥ 0.7 인 클래스만 기본으로 보인다")을 이 측정으로 정한다.
- 입력: seisyun complex 의 합친 기타 음(`quant.json` `guitar_all`) 중 파트 1(왼쪽) 음은 기타 스템 왼쪽 채널, 파트 2(오른쪽) 음은 오른쪽 채널의
  pyin f0(16 kHz, 75–1600 Hz, hop 10 ms; `data/tmp/m9/gtr_f0.py`). 파트가 없는 음(Unassigned)은 보지 않는다.
- 정답: 두 기타 PDF 의 "full" 표시(기타 2 에 20개, 기타 1 에 0개 — 모두 온음 벤딩). 위치는 E31 과 같은 규칙(표시의 오른쪽 음표 열의 박, 반 박
  반올림), 시각은 `eval_sc.py` 의 마디 대응.
- 짝짓기: 검출한 음의 시작이 아직 짝이 없는 정답 벤딩의 ±1 박 안이면 가장 가까운 것과 짝(탐욕, 시각 순). 지표: 정밀도, 재현율, F1, 짝지은
  것의 깊이 클래스 정확도("full").
- 결정 규칙(표시): 정밀도 ≥ 0.70 이면 벤딩을 악보(alphaTex `b`, GP5 벤딩)에 기본으로 쓴다. 아래면 `score.json` 의 제안 표시만.
- 한계: 곡 1개, 정답 20개, 위치는 박 근사. 분리 스템 채널의 f0 라 다른 기타·보컬이 섞인다.
- 판정 설정: `rule=descriptive; metric=f1`
- 비용: CPU — pyin 채널당 약 1.5분.
- 실행 명령: `data/tmp/m9/e32_sc.py`(사용자 악보가 필요해 저장소 밖 스크립트) — 결과는 아래에 손으로 적는다.
- 결과:
  - 2026-10-18 `data/tmp/m9/e32_sc.py`(등록한 기본값): **검출 0개 → 재현율 0, 정밀도 정의 안 됨 → 표시 규칙상 제안만**(그나마 제안도 0).
    - 정답 정정(결과와 무관한 셈): "full" 표시 20개 중 4개는 같은 벤딩이 "full" 과 "full↗" 두 단어로 뽑힌 것 — 서로 다른 벤딩은 **16곳**
      (기타 2: 39·40·77·78·89·101·124·125·152 마디).
    - 까닭(`data/tmp/m9/e32_diag.py`): 16곳 모두 **두 기타가 8분음표로 코드를 치는 구간**이다 — 그 순간 우리 음은 4–6음 화음이고, 기타 스템
      양 채널의 pyin 은 거의 다 무성(오른쪽 채널 곡 전체 유성 8 %, 왼쪽 1 %). 설계서의 f0 규칙은 단선율 Line 오디오를 전제로 하는데, 이 곡의
      벤딩은 화음 속에서 나온다. 한 파트에서 겹치지 않는 음도 거의 없다(파트 1·2 가 같은 순간에 섞여 있다).
    - 다음 길(새 실험): ① 화음 속 벤딩 — 음마다 그 음높이 둘레의 좁은 대역에서 배음 봉우리의 주파수를 따라가기(f0 대신 부분음 추적).
      ② 판정용 정답 — Tier B 의 단독 리드 테이크(jetb g1, secretariat g0, zeno g6)에 pyin 을 돌리면 깨끗한 벤딩 정답이 된다(분리·믹스 속
      검출을 그것과 대조). DESIGN 의 GOAT/Guitar-TECHS 는 다운로드 승인이 필요하다.

## E30b
- 제목: 베이스 블리드 표시 다시 재기 — 정답(단독 테이크 전사)의 저음 유령 걸러내기 (E30 리뷰 후속)
- 상태: 결정: 채택 (bass: test 정밀도 0.975 [0.857, 0.994], F1 차 +0.195 — notes 단계 기본으로 켬)
- 왜: E30 의 정답은 단독 테이크를 MuScriptor 로 전사한 것인데, 그 전사에도 믹스 전사와 같은 **옥타브 아래 유령 음**이 있었다. 독립
  리뷰(2026-10-18)에 따르면 test 에서 "진짜 기타 음을 잘못 걸렀다"로 셈한 42음이 모두, 같은 테이크에 같은 순간 정확히 한 옥타브 위 음을
  갖는다. rememberdecember 테이크 오디오에서 그 기본음은 옥타브 위 음보다 −32 dB(A1)·−28 dB(C#2) 약했다. 같은 테이크의 진짜 D2/E2 는
  −14 dB 였다. A1(MIDI 33)은 드롭 D 보다도 낮다. **리뷰가 test 의 표시를 들여다보고 찾은 문제**라, 이 실험은 같은 test 곡을 다시 재는 것이다.
  바꾸는 것은 정답 쪽 규칙 하나뿐이고, 검출 규칙과 문턱은 E30 의 `bass` 그대로다.
- 정답 정정 규칙: 테이크 참조 음 중 E2(MIDI 40) 아래 음을 다음 둘이 모두 맞으면 참조에서 뺀다. 그 뒤 truth 를 E25 방식(테이크 → 줄 묶음,
  `split_eval.match`)으로 다시 짓는다.
  - 같은 테이크에 시작 ±50 ms 안에 정확히 12반음 위 음이 있다.
  - 테이크 모노 오디오에서 그 음의 기본음 대역 에너지가 위 음의 기본음 대역보다 20 dB 넘게 약하다. 기본음 대역은 음높이 ±50 cents(가장
    가까운 bin 은 늘 포함)이고, 시작부터 150 ms 동안 Hann 8192 창·hop 1024 의 파워 합으로 잰다.
- 지표·결정 규칙: E30 의 `bass` 와 같다(test 정밀도 ≥ 0.80, 표시 ≥ 20음 아니면 판단 보류, F1 차 평균 ≥ −0.5점; F1 은 정정한 참조로).
  통과하면 `notes` 단계에서 기본으로 켠다(표시한 음은 `guitar_all` 에서 빼고 뷰어 후보로).
- 함께 보고: 정정으로 빠진 참조 음 수(곡별), dev 곡도 같은 방식으로.
- 판정 설정: `rule=descriptive; metric=precision`
- 비용: CPU 몇 분.
- 실행 명령: `bandscribe eval exp E30b`
- 결과:
  - 2026-10-18 `20261018-134043_fba5ffc_exp-e30b`: 계산된 판정 = 서술(판정 없음)
  - 요약(2026-10-18): 정정으로 빠진 참조 음 — butterflyeffect 테이크 0: 165, rememberdecember 0: 29·1: 10, zeno 2: 12, jetb 0: 2,
    jokers 1·3: 1씩(나머지 0). **test: 표시 198음 중 가짜 193 — 정밀도 0.975 [0.857, 0.994]**(E30 0.70), F1 차 +0.195 [0.00, +0.52]점
    → ① ② 통과 → **PASS, `notes` 단계에 기본으로 켠다**(`[amt] bass_bleed = true`). dev: 54음 0.926(E30 과 같음 — dev 표시는 원래
    대부분 가짜로 셌다). 곡별 test: rememberdecember 170음 중 169, zeno 28음 중 24.
    - E30 의 "드롭 튜닝 기타가 베이스와 같은 음을 친다"는 설명은 틀렸다: 그 음들은 정답 전사의 유령이었다.

## E33
- 제목: 코드 인식 v1 — 베이스를 분포에 넣고 텐션 템플릿을 더하기, 새 두 곡으로 판정 (M9 후속, E31 다음 길)
- 상태: 결정: 기각 (test 두 곡 v0 0.359, v1 0.343 — 둘 다 0.70 한참 아래, v1 이 v0 를 못 넘음; v0 유지)
- 질문: E31 이 본 실패(기타가 근음을 뺀 보이싱, m9·maj7 같은 텐션)를 겨냥한 두 옵션(`bandscribe/tab/chords.py`)이 **처음 보는 곡**에서
  근음 + 부류 정확도를 0.70 위로 올리는가?
  - `extended`: 템플릿 m9, maj9, 9, 7sus4, 6, m6, dim7, m7b5, madd9 를 더한다. 평가 부류는 E31 의 줄이기 규칙과 같게 둔다(m9·m6·m7b5·madd9 →
    min, maj9·6 → maj, 9 → 7, 7sus4 → sus, dim7 → dim).
  - `bass_weight`: 베이스 음의 울리는 시간을 그 배수로 기타 분포에 더한다.
- 데이터:
  - **dev = seisyun complex**(E31 에서 이미 본 곡): 격자 16개(`extended` {False, True} × `bass_weight` {0, 0.5, 1.0, 2.0} × `change_cost`
    {0.15, 0.3}, 이 순서 — 단순한 것 먼저)에서 정확도가 가장 높은 것을 v1 로 고른다. 동률이면 격자 순서.
  - **test = 「ギターと孤独と蒼い惑星」·「あのバンド」**(사용자가 2026-10-18 에 준 음원과 Guitar Pro 악보 PDF; 저장소 밖). 이 두 곡의 코드
    결과는 등록 전에 보지 않았다(악보에서 뽑은 단어 목록만 봤다).
- 정답·채점: E31 과 같은 규칙(리뷰가 고친 채점기: 슬래시 코드는 베이스만 버림, 갈린 이름 붙이기, 오른쪽 음표 열의 박을 반 박으로, 기타 1 이
  쉬는 마디·N.C.·첫 코드 전은 채점 안 함). 곡마다 악보 박과 우리 박의 오프셋은 **코드와 상관없이** 정한다: 오프셋 −8…+8 박 중 마디별 음
  bag F1(우리 합친 기타 vs 악보 기타 1·2, `eval_sc.py` 방식)이 가장 높은 것.
- 지표: 반 박 단위 근음 + 부류 정확도. test 두 곡은 반 박을 모두 모아 하나로 낸다(곡별로도 보고). v0(E31 기본값)도 같은 곡에서 잰다.
- 결정 규칙(test):
  1. v1 이 test 에서 0.70 이상이면 v1 을 기본값으로 하고 악보 안 코드 이름도 기본으로 켠다(`tab.chords_in_score = true`).
  2. 아니어도 v1 이 test 두 곡 모두에서 v0 보다 높으면 v1 을 코드 차트의 기본 인식기로 바꾼다(악보 안 이름은 끈 채로).
  3. 둘 다 아니면 v0 그대로.
- 판정 설정: `rule=descriptive; metric=accuracy`
- 비용: CPU 몇 분(두 곡 파이프라인의 GPU 단계는 이 실험과 별개로 돌린다).
- 실행 명령: `data/tmp/m9/e33.py`(사용자 악보가 필요해 저장소 밖) — 결과는 아래에 손으로 적는다.
- 결과:
  - 2026-10-18 `data/tmp/m9/e33.py dev` → v1 = `extended` False, `bass_weight` 1.0, `change_cost` 0.15(seisyun complex 0.607, 근음 0.711;
    v0 0.546). 텐션 템플릿은 어느 조합에서도 도움이 되지 않았다(최고 0.591).
  - 두 새 곡을 돌리다 찾은 것(코드와 무관): 「ギターと孤独と蒼い惑星」의 박 격자가 **반 박자**였다(약 97 BPM·368박, 악보 ♩=193·732박) — 악보의
    템포 표시를 작업 `hints.toml` `[grid] tempo_bpm = 193` 로 넣고 다시 돌렸다(192.5 BPM, 730박). 「あのバンド」는 188.8 BPM·656박(악보 ♩=190·
    672박)으로 그대로. 오프셋(음 bag F1 최대): ギター −3 박(0.606; −2 박 0.601 과 거의 같다), あのバンド +3 박(0.641).
  - **test**(`e33.py test`, 반 박 2,492개):
    - ギターと孤独と蒼い惑星: v0 0.281(근음 0.399), v1 0.252(근음 0.414).
    - あのバンド: v0 0.452(근음 0.585), v1 0.450(근음 0.627).
    - 모아서 **v0 0.359, v1 0.343** → ① v1 ≥ 0.70 아님, ② v1 이 두 곡 모두에서 v0 보다 높지 않음 → **v0 그대로**.
  - 해석: 베이스를 분포에 넣으면 근음은 조금 더 맞히지만(+0.02–0.04) 부류를 더 틀린다. seisyun complex 에서만 좋아진 것은 그 곡에 맞춘 것이었다.
    seisyun complex 의 0.546 자체도 이 인식기의 일반 성능보다 높았다 — 새 두 곡에서는 0.36. 코드 차트는 지금 수준으로는 "참고용"이다.

## E32b
- 제목: 벤딩 f0 규칙 다시 재기 — 새 두 곡 (E32 후속)
- 상태: 결정: 기각 (새 두 곡 벤딩 99개 중 1개 — 채널 f0 가 리드를 단선율로 못 봄; 표시 그대로 제안만)
- 질문: E32 의 규칙(`BEND_DEFAULTS` 그대로, 바꾸지 않음)이 단선율 리드가 있는 곡에서는 벤딩을 찾는가? E32 의 seisyun complex 는 벤딩이 모두
  화음 속이라 0개였다.
- 데이터: 「ギターと孤独と蒼い惑星」(기타 2 "full" 9개)·「あのバンド」(기타 2 "full" 88개; 같은 벤딩이 두 단어로 뽑힌 것은 0.25 박 안에서
  하나로). f0 는 E32 와 같다 — 파트 1 은 기타 스템 왼쪽, 파트 2 는 오른쪽 채널의 pyin.
- 짝짓기·지표: E32 와 같다(±1 박, 정밀도·재현율·F1, 깊이 클래스 "full"). 두 곡을 모아서 낸다(곡별로도).
- 결정 규칙(표시): 모은 정밀도 ≥ 0.70 이고 짝지은 것이 10개 이상이면 벤딩을 `score.json` 제안에서 악보 표시로 올리는 일을 다음 일로 잡는다
  (지금 파이프라인에는 벤딩 규칙이 연결돼 있지 않다). 아니면 그대로.
- 판정 설정: `rule=descriptive; metric=f1`
- 비용: CPU — pyin 채널당 약 1.5분 × 4.
- 실행 명령: `data/tmp/m9/e32b.py`(저장소 밖) — 결과는 아래에 손으로 적는다.
- 결과:
  - 2026-10-18 `data/tmp/m9/e32b.py`(E32 규칙 그대로, 오프셋은 E33 과 같은 음 bag 정렬):
    - ギターと孤独と蒼い惑星: 정답 9, 검출 1(짝 1, 깊이는 ½ 로 틀림). 기타 스템 유성 비율 오른쪽 20 %·왼쪽 3 %.
    - あのバンド: 정답 90, 검출 0. 유성 비율 오른쪽 3 %·왼쪽 0 %.
    - 모아서 정답 99 중 짝 1 → **재현율 0.01**, 정밀도 1.0(1개) → 표시 규칙(정밀도 ≥ 0.70 이고 짝 ≥ 10) 미달 → **그대로**.
  - 해석: 리드가 따로 있는 곡에서도 기타 스템 채널에는 디스토션 리듬이 같이 있어, pyin 이 단선율로 보지 않는다. 채널 f0 방식은 이 곡들에서
    닿지 않는다.
  - 다음 길(새 실험): 음마다 그 음높이 둘레의 좁은 대역에서 배음 봉우리 주파수를 따라가는 **부분음 추적**. 이제 정답 벤딩이 세 곡에 115개
    있다 — dev 를 seisyun complex + ギターと孤独と蒼い惑星(25개), test 를 あのバンド(90개)로 둘 수 있다.

## E35
- 제목: 음 확신 — 기타 음마다 "악보에 있는 음일 확률", 보정과 검토 큐 (M10, DESIGN §10 M10 "피치 신뢰도 ECE 0.10 이하", "검토 큐 상위 20 %에 실제 오류의 60 % 이상")
- 상태: 결정: 기각 (포착률 0.234 — 무작위 0.20·파트 확률 0.235 와 비슷, AUC 0.557; ECE 0.045 는 변별력 없이 낮음)
- 질문: 음마다 특징을 넣은 로지스틱 회귀가 처음 보는 곡에서 ① 보정되고(ECE ≤ 0.10) ② 확률이 낮은 순서 상위 20 % 가 틀린 음의 60 % 이상을
  잡는가?
- 데이터: Tier A 악보 곡 여섯(seisyun complex, ギターと孤独と蒼い惑星, あのバンド, 星座になれたら, ひとりぼっち東京, 光の中へ)의 합친 기타 음
  (`quant.json` `guitar_all`, 지금 기본값으로 돌린 결과).
- 정답(음마다 맞음/틀림): 그 음이 든 악보 마디(오프셋 = 음 bag 이 가장 잘 맞는 박, M9 표와 같음)의 기타 1·2 음높이 묶음(악보 머리의 튜닝·카포
  반영)에 그 음높이가 남아 있으면 맞음 — 마디 안에서 시작 순서대로 하나씩 소비한다(`eval_sc.py` 의 bag 과 같은 규칙). 악보 마디 밖(앞·뒤)의 음은
  뺀다.
- 특징(음마다, 고정):
  1. log 길이.
  2. 같은 순간(±30 ms) 시작하는 기타 음 수.
  3. 곡의 기타 음높이 중앙값과의 차(반음).
  4. 같은 순간 옥타브 짝 유무.
  5. Basic Pitch 기타 전사(`amt_bp`)에 같은 음높이·50 ms 안의 음 유무, 옥타브만 맞는 음 유무.
  6. 악기 클래스(clean / distorted / acoustic).
  7. 파트 확률 `p`, 미배정 여부.
  8. 운지 확신 `conf`(파트 탭의 그 음; 탭에서 빠진 음은 0, 빠짐 표시 따로).
  9. 양자화 잔차 |`resid_ms`|.
  10. 주변 1초의 음 밀도.
  11. E2 아래 여부.
- 모델: 표준화 + 로지스틱 회귀(scikit-learn, C = 1.0). **곡 하나씩 빼고** 다섯 곡으로 학습해 뺀 곡을 예측한다.
- 지표(뺀 곡 예측을 모아서):
  - ECE: 10개 등간격 구간, 음 수 가중.
  - AUC.
  - 검토 큐 포착률: 곡마다 확률 낮은 순 상위 20 % 음이 그 곡 틀린 음의 몇 %를 잡는지, 여섯 곡 평균.
  - 기준선: 무작위(0.20), 파트 확률 낮은 순, 운지 확신 낮은 순(지금 뷰어 큐의 두 정렬).
- 결정 규칙:
  1. ECE ≤ 0.10 이고 포착률 ≥ 0.60 이면 기준 통과. 음 확신을 `score.json` `Confidence.pitch` 로 쓰고 뷰어 검토 큐의 기본 정렬로 한다.
  2. 기준 미달이어도 포착률이 두 기준선보다 모두 높으면 검토 큐의 정렬 선택지로만 더한다(기본 아님).
  3. 아니면 넣지 않는다.
- 한계: 곡 여섯, 모두 같은 밴드·같은 채보자(Rika). 마디 단위 정답이라 마디 안 시점은 보지 않는다.
- 판정 설정: `rule=descriptive; metric=capture20`
- 비용: CPU 몇 분.
- 실행 명령: `data/tmp/m10/e35.py`(사용자 악보가 필요해 저장소 밖; 특징 계산은 저장소 `bandscribe/tab/confidence.py`) — 결과는 아래에 손으로 적는다.
- 결과:
  - 2026-10-18 `data/tmp/m10/e35.py`: 음 29,679개(곡마다 틀린 음 33–45 %).
    - 뺀 곡 ECE 0.04–0.11, 모아서 **ECE 0.045**. AUC **0.557**. 포착률 곡별 0.21–0.26, 평균 **0.234**(무작위 0.20, 파트 확률 0.235,
      운지 확신 0.203).
    - → ① ECE 는 넘지만 포착률 0.60 미달, ② 두 기준선을 모두 넘지 못함(파트 확률과 같음) → **넣지 않는다**.
    - ECE 가 낮은 것은 모든 음에 비슷한 확률(틀린 비율 근처)을 줘서다 — AUC 0.56 이라 가려내는 힘이 없다. 계수(여섯 곡 전체 학습) 중 가장 큰 것은
      Basic Pitch 도 같은 음을 들었는가(+0.33), 다음이 E2 아래(−0.14), 음역(+0.10).
  - 진단(`data/tmp/m10/wrong_kinds.py`): 마디 bag 기준 "틀린 음" 39 % 중 **30 %p 는 그 마디 악보에 같은 음이름이 있는 음**(옥타브가 다르거나
    같은 코드음이 더 있음)이고, 아예 다른 음이름은 9 %, 악보가 쉬는 마디 1 %. 우리 특징으로는 이 옥타브·중복을 가려낼 수 없다 — 옥타브·중복을
    고르는 일은 전사 단계(또는 음표 정리) 몫이다.

## E36
- 제목: 파트 신뢰도 — 과신한 파트 사후확률 대신 음마다의 좌우 근거로 "그 파트가 맞을 확률" (M10, DESIGN §10 M10 "파트 신뢰도 ECE 0.10 이하")
- 상태: 결정: 기각 (ECE 0.108 > 0.10, AUC 0.477 — 보정은 나아졌으나 곡이 바뀌면 틀린 파트를 못 가려냄)
- 질문: 지금 파트 단계의 사후확률 `p` 는 E27 곡 전체 Tier B 리믹스에서 평균 0.998 인데 맞는 비율은 0.838(ECE 0.161, 등록 전 서술 측정
  `data/tmp/m10/part_ece.py`)이다. 음마다의 좌우 근거로 만든 로지스틱 모델이 처음 보는 곡에서 ECE ≤ 0.10 으로 보정되고 틀린 파트를 가려내는가?
- 데이터: E27 의 곡 전체 Tier B 리믹스 중 정답 파트가 둘 이상인 것(P, A1, B2; 9곡). 파트 = 파이프라인 `[parts]` 그대로(v0). 맞음 = 그 음의 파트가
  (다수결로 짝지은) 정답 Line 이 그 음의 truth 에 있음. truth 가 빈 음(어느 테이크에도 없는 음)과 파트가 없는 음은 뺀다.
- 특징(음마다, 고정): |ILD|, ILD 가 그 파트 음들의 평균 ILD 와 같은 쪽인가, (ILD − 파트 평균) / (파트 표준편차 + 1), coherence, M/S dB,
  창 라벨(R0–R3, 없음), 사후확률 `p`, 파트 수 K, 같은 순간 시작하는 음 수.
- 모델: 표준화 + 로지스틱 회귀(C = 1.0), **곡 하나씩 빼고** 학습(9겹).
- 지표(뺀 곡 예측 모아서): ECE(10 구간), AUC, 틀린 파트 포착률(곡마다 낮은 순 상위 20 %, 평균). 기준선: 지금 `p`(ECE 0.161).
- 결정 규칙: ECE ≤ 0.10 이고 AUC ≥ 0.70 이면 채택 — 파트 단계가 음마다 보정된 파트 신뢰도(`p_cal`)를 내고 뷰어 "파트 확률" 정렬과 `score.json`
  `Confidence.part` 가 그것을 쓴다. 아니면 그대로.
- 한계: 9곡(Cambridge-MT, 리믹스는 우리가 만든 배치), 한 테이크 = 한 파트라는 정답 정의.
- 판정 설정: `rule=descriptive; metric=ece`
- 비용: CPU 몇 분.
- 실행 명령: `data/tmp/m10/e36.py` — 결과는 아래에 손으로 적는다.
- 결과:
  - 2026-10-18 `data/tmp/m10/e36.py`: 음 37,227개(맞는 비율 0.84). 뺀 곡마다 ECE 0.03–0.18, AUC 0.43–0.71, 포착률 0.06–0.43.
  - 모아서 **ECE 0.108**(지금 `p` 0.161), **AUC 0.477**(`p` 0.519), 포착률 0.226(`p` 0.146) → ECE 0.10 미달, AUC 0.70 한참 미달 → **그대로**.
  - 해석: 보정은 나아졌지만(평균 확률을 맞는 비율 근처로 내림), 곡이 바뀌면 틀린 파트를 가려내지 못한다(AUC 0.5 근처). 가장 큰 계수는 |ILD|(+1.07)·
    창 라벨 없음(−1.09)·파트 수(−0.40) — 좌우가 뚜렷한 음이 맞기 쉽다는 당연한 것뿐이고, 틀린 파트의 원인(같은 위치·더블·유니즌)은 이 특징에
    없다. 사후확률 과신 문제는 남는다.

## E37
- 제목: 소리 종류별·음량별 기타 탭 오염 — 피아노·전자 건반·오르간·신스·현악·관악을 기타 대비 −6 / 0 / +6 dB 로 섞었을 때 (M10 진단 후속)
- 상태: 결정: 판단 보류 (서술 측정: 현악 −3.2·기타 신스 −14.1 점이 규칙 충족 → E38, 마스크 변경은 절대 F1 로 손해)
- 질문: 결속밴드 곡과 달리 J-pop 곡(青と夏·lulu.·AIZO)은 분리된 건반+기타 외 소리가 기타와 비슷한 크기다(기타 대비 −7 ~ +1 dB,
  `data/tmp/m10/arrangement.py`). `distractors` 측정(2026-10-04)은 멀티트랙 자체 밸런스였고, 그 구간에서 피아노는 −30 ~ −10 dB, 현악은
  −32 ~ −16 dB 로 작았으며 관악은 곡이 없었다(`data/tmp/m10/distr_levels.py`). 각 소리를 J-pop 수준까지 키우면 기타 탭이 얼마나 나빠지고,
  전사기(MuScriptor)는 그 소리를 기타가 아닌 악기로 얼마나 가려내는가? 현악·관악을 마스크에 허용하면 나아지는가?
- 선택지: 없음(측정). 조건 = `base`(기타·드럼·베이스·보컬) / `+<범주>@<단계>`(base + 그 범주 하나를 기타 합 대비 단계 dB 로).
  범주 = 피아노(keys_piano), 일렉 피아노(keys_epiano), 오르간, 신스 패드·리드·기타 신스, 현악, 관악(brass_winds; 하모니카 포함).
  효과음(sfx)·미분류(other)는 조건을 만들지 않는다. 경로 = `production`(현재 기본 마스크 guitar+present), 진단 경로 `strings`(+ 현악 클래스,
  E3b 의 guitar+present+strings), `strings_winds`(+ 현악·금관·목관 클래스; 지금은 어떤 마스크도 관악을 허용하지 않는다).
- 음량 정의: 그 범주 스템 합의 기타 합 대비 dB = 둘 다 소리 나는 100 ms 프레임(각자 가장 큰 프레임의 −30 dB 이내, −60 dBFS 이상)의 평균
  전력 비(겹치는 프레임이 10개 미만이면 각자 소리 나는 프레임끼리). 0 dB 이득 = −(그 값), 상한 +30 dB(넘으면 기록). 구간마다 마스터링 게인
  곡선 하나(−8 LUFS, −1 dBTP)를 **가장 큰 혼합**(base + 있는 범주 모두 +6 dB)에서 계산해 모든 조건에 쓴다 — 기타 신호는 모든 조건에서 같다.
- 주 지표: 범주·단계마다 `+범주@단계` − `base` 의 기타 onset F1(점), FP/분, FN/분, `production` 경로, 곡 블록 부트스트랩 95 % CI
  (10,000회, seed 20260930). 참조 = 참 기타 스템 합의 MuScriptor 전사(기타만 마스크) — `distractors` 와 같다. 함께 보고: **구분률**(참조와
  맞지 않는 출력 음 중 그 음높이 대역 1위 소리가 그 범주인 음에서, 비기타로 표시돼 탭에서 빠진 비율), 참 기타 음을 비기타로 잘못 뺀 수,
  FP 원인 라벨, 분리 누출(dB), 진단 경로의 같은 지표(0 dB 조건과 base 만), 디코더 잡음 기준(`base_null`, ‑90 dBFS).
- 데이터(세트, 분할, 창 정의): Tier B `cambridge_mt` 20곡 중 위 범주가 있는 17곡. 기존 13곡에 이 등록과 함께 받은 7곡(곡 고르기 규칙은
  결과·오디오를 보기 전 `data/tmp/m10/ms21_select.py` 에 고정: MS21 주석 CSV 에서 Clean = 1, 팝·록·얼터너티브·일렉트로니카, 일렉 기타
  있음, 바로 받을 주소가 있는 곡 중 현악 파일이 가장 많은 4곡과 관악 파일이 가장 많은 4곡 — Ben Carrigan 2곡, Dunning Kruger, E J Rios,
  Flags, Tim Taler, Pablo Martin; registry.toml). 곡당 75 s 창 1개(`distractors` 의 규칙, 단 "있음"·창 고르기는 위 범주만 센다).
  계획(GPU 전, 2026-10-18 `data/tmp/m10/e37_plan.py`): 창 17개, 조건 122개. 범주별 창 수: 피아노 7, 일렉 피아노 2, 오르간 3, 신스 패드 3,
  신스 리드 3, 기타 신스 5, 현악 7, 관악 5. 분할 없음(서술).
- 결정 규칙: 서술 측정이지만 다음 일을 미리 정한다. **0 dB 조건에서 ΔF1 ≤ −3점이고 95 % CI 상한 < 0 이며 2곡 이상**인 범주는 다음
  실험(E38)에서 그 소리 전용 대책(마스크 변경 또는 S7 블리드 감점)을 사전 등록해 고칠 대상이다. 진단 경로가 그 범주의 0 dB ΔF1 을
  production 보다 3점 이상 덜 깎으면 그 마스크 변경이 대책 후보 1순위다. 그런 범주가 없으면 다음 순서는 쪼개짐·옥타브 정리(M10 진단)다.
  곡이 1개뿐인 범주(CI 없음)는 서술만 한다.
- 한계: 합성 믹스(실제 녹음의 밸런스·마스터링이 아님), 곡당 구간 1개, 범주마다 2–7곡, 참조가 MuScriptor 파생(절대 성능 주장 없음),
  스템 이름으로 범주를 나눈다(마림바는 피아노로, 첼로가 베이스 라인을 치면 베이스로 셈), 디코더가 입력의 작은 변화에도 흔들린다
  (`distractors` null) — 잡음 기준과 함께 읽는다.
- 판정 설정: `rule=descriptive; metric=note_f1_onset50; split_col=scenario`
- 비용: GPU 약 3–4시간(계획의 추정 226분; 10월 실측으로 보정하면 약 190분). 곡 묶음마다 30분 안팎으로 캐시를 채우고(롤이 켜져 있으면
  기다림), 마지막 `bandscribe eval exp E37` 은 캐시만 읽는다(CPU).
- 실행 명령: `bandscribe eval exp E37` (캐시 채우기: 같은 인자로 `bandscribe.eval.distractors.run` 에 `songs=` 를 나눠 준다)
- 등록 메모(2026-10-18, GPU 실행 전): 함께 고친 것 — `stem_taxonomy` 가 "Sample(s)" 가 든 이름을 무조건 드럼으로 보내던 규칙(SampleBrass,
  TrumpetSamples, ViolinSamples, SampleChoir)을 다른 악기 이름이 없을 때만으로 좁혔다. 기존 13곡의 분류는 하나도 바뀌지 않았다(확인함).
  `amt/instruments.py` 주석의 "怪獣の花唄 / 青と夏 의 SW 기타 스템은 대부분 현악·패드" 는 기타를 뺀 음원(2026-10-18 확인)에서 본 것이라
  근거가 되지 않는다.
- 결과:
  - 2026-10-19 `20261019-211741_fb7a7c2_exp-e37`: 계산된 판정 = 서술(판정 없음)
  - 실행 기록: GPU 캐시 채우기 9묶음(2026-10-18 1–5, 사용자 요청으로 멈춘 뒤 2026-10-19 6–9; 합계 약 3시간 6분,
    `data/tmp/m10/e37_chunks.log`), 공식 실행은 캐시만 읽음. 창 17개(17곡), 조건 122개.
  - 요약(실험자, 2026-10-19; 표 `data/tmp/m10/e37_table.py`). `production` 의 기본 조건 F1 78.9(풀링). 0 dB 조건 ΔF1(점, 곡 블록 95 % CI):
    - **현악(7구간·6곡) −3.2 [−6.0, −1.1] → 등록 규칙 충족(E38 대상).** 단계별 −0.6 / −3.2 / −6.6 으로 음량에 비례하고, FP(+28/분)와 FN(+23/분)이
      함께 는다. 디코더 잡음 기준은 현악 구간 모두 0.
    - **기타 신스(synth_other, 5곡) −14.1 [−24.4, −0.8] → 규칙 충족(E38 대상).** 곡마다 크게 다르다: Pablo Martin −32.0·Flags −11.3(잡음 기준
      작음, 진짜 손해), James Elder −15.6(잡음 기준 FP +90/분, 불안정), E J Rios −1.0, Mu +3.6. +6 dB 에서는 −52.9.
    - 피아노(7구간·6곡) −4.3 [−12.9, +1.1]: 규칙 미충족(CI 가 0 포함). 곡마다 갈린다 — Hey Carrie Anne −32.8(참조 음 62개뿐),
      JetB −19.1(E3c 와 같은 곡), Tim Taler −11.1, 나머지 넷은 −5.4 ~ +2.8.
    - 관악(5구간·4곡) −2.4 [−7.1, +0.1](Pablo Martin −9.7, 나머지 ±2), 오르간 +3.0, 신스 패드 +1.9(+6 dB 에서 −8.7), 신스 리드 −0.2.
    - 일렉 피아노(2곡) +12.2 [+2.0, +23.5] 는 개선이 아니라 Mu 한 곡(+23.5)의 디코더 불안정이다(그 구간 잡음 기준 FP −242/분;
      `distractors` 때와 같은 곡·같은 현상).
    - **구분률: 제품 경로는 모든 범주에서 0 %** — 그 소리가 가장 큰 가짜 음 중 MuScriptor 가 비기타로 표시해 뺀 음이 하나도 없다
      (피아노 183, 기타 신스 359, 현악 119, 관악 15개 모두 기타로 남음). 지금 탭에서 다른 악기를 막는 것은 분리(SW)뿐이다.
    - **마스크 경로는 대책이 아니다.** 등록 문구("진단 경로가 0 dB ΔF1 을 production 보다 3점 이상 덜 깎으면")를 글자대로 적용하면
      `strings_winds` 가 현악에서 +6.1 [−0.8, +16.4] 로 충족하지만, 이것은 자기 기본 조건 대비 차이다. 절대 F1 은 기본 조건 58.9(production
      78.9), 현악 0 dB 62.4(production 75.0)로 오히려 크게 나쁘다 — 관악·목관 클래스를 허용하자 진짜 기타 음을 관악으로 표시해 뺐다
      (현악 조건 296개, 관악 조건 87개). `strings` 경로도 기본 75.5·현악 72.5 로 production 보다 낮다. **등록 규칙의 결함**(같은 조건의 절대
      F1 이 아니라 경로별 자기 기준 대비 차이를 비교)을 기록해 두고, 마스크 변경은 대책 후보에서 뺀다.
  - 결론: E38 은 현악과 기타 신스를 대상으로 하되, 전사기 라벨(마스크)이 아니라 **분리된 스템의 대역 에너지로 하는 S7 블리드 감점**
    (그 음 대역에서 piano+other 스템이 기타 스템보다 크면 감점, 베이스 블리드 E30b 와 같은 방식)을 사전 등록해 잰다. 피아노는 곡에 따라
    큰 손해(−11 ~ −33점)가 나므로 같은 실험에 함께 넣는다. 한계: 합성 믹스, 곡당 75 s 한 구간, 참조가 MuScriptor 파생, 디코더 불안정 곡
    (Mu, James Elder) 포함.

## E38
- 제목: 스템 에너지 블리드 감점 — 기타 음의 음높이 대역에서 분리된 piano+other 스템이 기타 스템보다 크면 그 음을 뺀다 (E37 후속)
- 상태: 결정: 기각 (모든 문턱에서 지운 진짜 기타 음이 가짜보다 많음 — 분리된 piano+other 스템에 기타 에너지가 커서 에너지 비로 못 가름)
- 질문: E37 에서 현악(0 dB −3.2점)·기타 신스(−14.1점)가 기타 탭을 망쳤고, 전사기는 그 음을 한 번도 비기타로 표시하지 않았으며(구분률 0 %)
  마스크를 넓히면 진짜 기타 음을 잃었다. 분리된 두 스템의 대역 에너지 비로 그런 음을 거르면 F1 이 오르는가, 진짜 기타 음을 잃지 않는가?
- 선택지: `production`(지금) | `stem<T>`(production 음 중 기타 클래스 음에서 10·log10(E_other / E_guitar) ≥ T dB 인 음을 뺀다;
  T ∈ {−3, 0, +3, +6, +10}). E = `eval.attribution` 의 대역 에너지(±50 cents f0 대역, MIDI 52 미만은 2배음 대역 더함, STFT 8192/1024,
  [onset, onset + clip(길이, 50 ms, 500 ms)] 평균), 신호 = 그 조건의 분리 결과 `guitar_mono` 와 `piano_other_mono`
  (파이프라인 `stems` 단계의 보기와 같은 것). 코드 `bandscribe/amt/bleed.py` `stem_ratio_db` / `stem_flags`.
- 주 지표: 기타 onset F1(`note_f1_onset50`, 참조 = 참 기타 스템 합의 MuScriptor 전사), E37 과 같은 창·같은 전사(캐시, GPU 0).
  판정 조건 = `base` 와 범주마다 0 dB 조건(J-pop 수준), 창 17개.
- 데이터 분할(결과 보기 전 고정): 곡 이름 순으로 번갈아 dev 9곡(Hey Carrie Anne, Dunning Kruger, Flags, JetB, Mistrusted, Mu, Remember
  December, Signe Jakobsen, Zeno) / test 8곡(We'll Talk About It All Tonight, E J Rios, James Elder, Jokers, Moosmusic, Pablo Martin,
  Secretariat, Tim Taler). 현악 dev 4·test 3, 기타 신스 dev 2·test 3, 피아노 dev 4·test 3.
- 분석·결정 규칙: dev 에서 풀링 F1 이 가장 높은 문턱 하나를 고르고(동점이면 production), test 에서 그 하나만 production 과 곡 블록 쌍
  부트스트랩(10,000회, seed 20260930)으로 비교한다. **채택** = CI 하한 > 0 이고 Δ ≥ +1.0점이며 조건별(시나리오) 부분집합 중 CI 상한이
  −2점 아래인 것이 없음(DESIGN 8.1, `stats.decide`). 추가 가드(실제 곡): 고른 문턱에서 정답 악보 6곡(봇치)의 합친 기타 마디 F1(M9 Tier A
  측정, `data/tmp/m10/e38_tiera.py`)이 평균 1점 넘게 떨어지거나 한 곡이라도 3점 넘게 떨어지면 채택하지 않고 판단 보류(기본 끔, 곡별 옵션).
  채택되면 `[amt] stem_bleed` 를 켜고 `notes` 단계가 표시된 음을 `bleed.json` 으로 옮긴다(베이스 블리드와 같은 방식, 뷰어 후보로 되살림).
- 함께 보고: 문턱별 dev/test F1, 조건별 Δ, 지운 음 수, J-pop 원곡 7곡(정답 없음)에서 문턱별로 지워질 기타 음 비율(서술).
- 한계: 합성 믹스, 곡당 75 s 한 구간, 참조가 MuScriptor 파생, 디코더가 흔들리는 곡(Mu, James Elder) 포함, 정답 악보 6곡은 건반·현악이
  거의 없어 "진짜 음을 잃지 않는가"만 본다.
- 판정 설정: `rule=superiority; metric=note_f1_onset50; baseline=production; mde=1.0; split_col=section; select=dev; split=test; subset_col=scenario`
- 비용: CPU 수 분(E37 캐시). GPU 0.
- 실행 명령: `bandscribe eval exp E38`, 그다음 `data/tmp/m10/e38_tiera.py`
- 결과:
  - 2026-10-21 `20261021-193533_e5ba017_exp-e38`: 계산된 판정 = 기각 (dev 최적이 기본값 production). CPU 약 8분(E37 캐시, GPU 0).
  - 요약(실험자, 2026-10-21): 풀링 F1(base + 0 dB 조건) — dev: production 74.3, stem−3 65.5, stem+0 67.2, stem+3 68.9, stem+6 70.1, stem+10 71.1;
    test: production 77.0, 가장 나은 stem+10 76.0. **모든 문턱에서 지운 진짜 기타 음이 지운 가짜 음보다 많다.** 0 dB 조건(전 곡)에서 stem+0 이
    현악은 진짜 1,524개를 잃고 가짜 410개를 지웠고, 피아노는 1,086 대 433, 오르간 550 대 140, 아무 소리도 더하지 않은 base 에서도 475 대 253.
    가짜 음이 진짜보다 많이 지워진 것은 기타 신스(131 대 163)뿐이고 그것도 F1 은 −1.6점.
  - 해석: SW 분리는 기타 소리의 상당 부분을 piano+other 스템에 넣는다(다른 소리가 없는 base 에서도 진짜 기타 음의 대역 에너지가 other 쪽이
    큰 경우가 흔하다). 그래서 "그 음 대역에서 other 가 더 크다"는 블리드의 증거가 되지 못한다. 현악·건반이 기타와 같은 음을 겹쳐 칠 때도
    마찬가지다.
  - 실제 곡 가드(`data/tmp/m10/e38_tiera.py`, 사전 등록한 보조 측정): 봇치 6곡은 어느 문턱에서도 지워지는 음이 0.2 % 이하, 마디 F1 변화
    0.13점 이하(건반·현악이 거의 없으니 당연하다). J-pop 원곡에서는 0 dB 문턱이 青と夏 37.6 %, lulu. 26.5 %, 狭心症 18.8 %, 怪獣の花唄 6.7 %,
    絶対零度 7.0 %, AIZO 2.3 %, 逆光 0.0 % 의 기타 음을 지웠을 것이다 — Tier B 결과로 보면 대부분 진짜 기타 음이다.
  - 다음: 스템 에너지 비 하나로는 안 된다. 남은 후보는 (1) piano+other 스템을 건반·현악 마스크로 전사해 같은 시각·같은 음높이의 비기타 음이
    있을 때만 감점(곡마다 GPU 전사가 필요; 파이프라인의 표본 존재 패스를 전곡으로) 또는 (2) 분리 모델 교체. 어느 쪽이든 새 E# 로 등록한다.
  - 2026-10-21 `20261021-193533_e5ba017_exp-e38`: 계산된 판정 = 기각 (dev 최적이 기본값 production)

## E39
- 제목: 파트 안 같은 음 중복 정리 — 같은 파트에서 같은 음높이가 30 ms 안에 두 번 시작하면 하나로 (M10 음 정리 1)
- 상태: 결정: 채택 (파트별 F1 +1.9, 지운 음 정확도 1.000, 악보 6곡 평균 +0.29 — dedupe_s 0.03 기본 켬)
- 질문: MuScriptor 는 한 기타 음을 두 클래스(어쿠스틱 + 클린 일렉 등)로 겹쳐 내는 일이 잦다. 지금 탭에서 같은 파트 안 같은 박에 같은 음이
  두 번 적힌 곳이 逆光 305개·青と夏 389개(SC 4개)다. 한 연주자가 같은 음을 동시에 두 번 낼 수 없으니 합치면 파트별 탭이 나아지고 진짜 음은
  잃지 않는가?
- 등록 전에 본 것(개발 데이터만): E27 dev 리믹스(moosmusic 제외 14개)에서 같은 음 재시작 간격 20 ms 미만 2,074개 중 93 % 가 참조와 안 맞음
  (`data/tmp/m10/cleanup_diag.py`, `cleanup_diag2.py`) — 단 이 참조는 테이크 합집합이라 두 기타가 겹친 음을 하나로 센다. 정답 악보 곡과
  test 리믹스는 보지 않았다.
- 선택지: `off` | `e39`(`bandscribe/tab/cleanup.py` `dedupe_within_line`, 창 30 ms 고정: 같은 `line` 같은 음높이에서 앞 음을 남기고 끝을
  뒤 음의 끝까지 늘린다; 파트가 없는 음끼리도 같은 규칙). 조정할 값 없음 → dev/test 선택 없이 모든 항목으로 판정.
- 측정(`data/tmp/m10/e39_e40.py`, CPU): (1) Tier B 곡 전체 리믹스 전부(E27 캐시: P·A1·B2·D1): 파이프라인 v0 파트 배정 뒤 적용, 파트별 매크로
  F1(각 정답 Line 의 참조 대 겹침이 가장 큰 예측 파트, 50 ms·50 cents) 과 합친 F1, 지운 음 수와 그 때문에 잃은 참조 일치 수.
  (2) 정답 악보 6곡: 파이프라인 quant 음에 적용한 파트별 마디 F1(gtr_1/gtr_2 대 악보 두 기타, 두 배정 중 나은 것), 합친 마디 F1.
  (3) J-pop 원곡 7곡에서 지워질 음 수(서술).
- 결정 규칙: **채택** = (a) Tier B 파트별 F1 Δ 의 곡 블록 부트스트랩(10,000회) 점추정 ≥ 0 이고 95 % CI 하한 ≥ −0.2점, (b) 지운 음의
  정확도(1 − 잃은 파트 참조 일치 / 지운 음) ≥ 0.90, (c) 정답 악보 6곡의 파트별 마디 F1 이 평균 −0.2점 이상이고 어느 곡도 −0.5점보다 나빠지지
  않음. 채택되면 `[tab] dedupe_s = 0.03` 을 기본으로 켜 `quant` 단계 입구에서 적용한다. 아니면 그대로.
- 판정 설정: `rule=descriptive; metric=part_f1` (결과는 손으로 적는다)
- 비용: CPU 수십 분. GPU 0.
- 실행 명령: `data/tmp/m10/e39_e40.py`
- 결과:
  - 2026-10-21 `data/tmp/m10/e39_e40.py`(결과 `e39_e40.json`, 로그 `e39_e40.log`): Tier B 곡 전체 리믹스 24개(9곡), 정답 악보 6곡, J-pop 7곡.
  - (a) Tier B 파트별 F1 Δ **+1.90 [+0.14, +4.68]**점, 합친 F1 Δ +3.07 [+0.21, +7.89] — 충족. 큰 곳: Butterfly Effect P +5.6·B2 +2.6(파트),
    Secretariat P +1.8, Mistrusted A1 +1.1, moosmusic P·B2(폭주 구간의 같은 음 반복) +17.9·+15.7.
  - (b) 지운 음 7,270개 중 잃은 파트 참조 일치 2개 → 정확도 **1.000** — 충족(합친 기준으로는 135개, 0.981: 두 파트가 같은 음을 겹쳐 칠 때
    합집합 참조가 하나로 세는 것).
  - (c) 정답 악보 6곡 파트별 마디 F1 Δ: SC +0.04, GKA +0.04, ANB +0.05, SNN +0.26, HBT **+1.38**(465음 정리), HNN −0.04 → 평균 +0.29,
    가장 나쁜 곡 −0.04 — 충족.
  - **채택:** `[tab] dedupe_s = 0.03` 기본 켬, `quant` 단계 입구(`tab/stage.py` `quant_tracks`)에서 파트별로 적용.
  - J-pop 원곡에서 지워질 음: 怪獣の花唄 2,130(기타 음의 34 % — 이 곡 Gtr 1 의 "한 손으로 칠 수 없는 음 2,080개"가 이것으로 보인다),
    青と夏 389, 逆光 305, lulu. 120, 狭心症 18, 絶対零度 5, AIZO 0.

## E40
- 제목: 초단음 폭주 막기 — 50 ms 보다 짧은 기타·베이스 음을 뺀다 (M10 음 정리 2)
- 상태: 결정: 채택 (합친 F1 +3.5, 지운 음 정확도 0.997, 악보 6곡 ±0.1 — min_dur_s 0.05 기본 켬)
- 질문: MuScriptor 가 곡의 한 구간에서 퇴화해 10 ms 짜리 음을 쏟아내는 일이 있다(lulu. 베이스 2:30 이후 10,965개 = 베이스 음의 83 %;
  Tier B moosmusic 기타 53–83 %). 그 밖에서는 이렇게 짧은 음이 드물다(파이프라인 기타 출력 0.0–0.4 %, dev 리믹스에서 moosmusic 밖 8개).
  50 ms 미만을 빼면 폭주 구간은 깨끗해지고 정상 곡은 손해가 없는가?
- 선택지: `off` | `e40`(`cleanup.drop_short`, 50 ms 고정, 기타는 파트 배정 전·베이스는 그대로). 조정할 값 없음 → 모든 항목으로 판정.
- 측정: E39 와 같은 스크립트. Tier B 합친 F1·파트별 F1·지운 음의 정확도, 정답 악보 6곡의 합친 기타 마디 F1 과 베이스 마디 F1, J-pop 원곡의
  지워질 기타·베이스 음 수(lulu. 베이스 포함, 서술).
- 결정 규칙: **채택** = (a) Tier B 합친 F1 Δ 점추정 ≥ 0 이고 95 % CI 하한 ≥ −0.2점, (b) 지운 음의 정확도(1 − 잃은 합친 참조 일치 / 지운 음)
  ≥ 0.90, (c) 정답 악보 6곡의 합친 기타 마디 F1 과 베이스 마디 F1 이 각각 평균 −0.2점 이상, 어느 곡도 −0.5점보다 나빠지지 않음. 채택되면
  `[tab] min_dur_s = 0.05` 를 기본으로 켜 `quant` 단계 입구(기타·베이스)에서 적용한다.
- 판정 설정: `rule=descriptive; metric=merged_f1` (결과는 손으로 적는다)
- 비용·실행: E39 와 같다.
- 결과:
  - 2026-10-21 같은 실행.
  - (a) Tier B 합친 F1 Δ **+3.46 [−0.03, +10.38]**점(점추정 ≥ 0, 하한 ≥ −0.2) — 충족. 효과는 거의 모두 moosmusic 폭주 구간(P 12.7 → 51.5,
    B2 17.4 → 61.7); 나머지 곡은 ±0.2점 안.
  - (b) 지운 음 5,918개 중 잃은 합친 참조 일치 17개 → 정확도 **0.997** — 충족.
  - (c) 정답 악보 6곡: 합친 기타 마디 F1 Δ −0.01 ~ +0.04, 베이스 마디 F1 Δ −0.03 ~ +0.09 — 충족.
  - **채택:** `[tab] min_dur_s = 0.05` 기본 켬(기타·베이스, `quant` 단계 입구).
  - J-pop 원곡: **lulu. 베이스 12,027음 중 10,289음 제거**(2:30 이후 폭주), 기타는 곡마다 0–18음.

## E41
- 제목: 건반·현악 블리드 — piano+other 스템을 따로 전사해 같은 시각·같은 음높이의 비기타 음이 있으면 기타 음을 뺀다 (E38 후속)
- 상태: 결정: 기각 (모든 규칙이 진짜 기타 음을 더 많이 지움 — 건반·현악이 기타와 같은 음을 함께 치는 일이 흔해 겹침이 블리드 증거가 못 됨)
- 질문: E38 의 에너지 비는 SW 가 기타를 piano+other 에도 넣어서 실패했다. 그 스템을 건반·신스·현악(+기타) 마스크로 전사하면 새어 든 기타는
  기타 클래스로 남고 건반·현악은 비기타 클래스로 나올 것이다. 기타 패스의 음 중 그 비기타 음과 같은 시각·같은 음높이인 음을 빼면 F1 이 오르는가?
- 선택지: `production` | `vote30` | `vote50` | `vote50o` (`bandscribe/amt/bleed.py` `vote_flags`: piano+other 전체 구간 전사에서 기타 클래스가 아닌
  음 중 같은 음높이(`o` 는 ±12 도) 이고 onset 이 30 / 50 ms 안인 것이 있으면 production 의 그 기타 음을 뺀다). piano+other 전사 = 파이프라인의
  보기(`piano_other_mono`)와 마스크(`amt.piano_other_instruments` = keys+guitar: 건반·신스·현악 + 기타 클래스), 단 표본 구간이 아니라 창 전체.
- 주 지표·데이터·분할·분석: E38 과 같다(E37 창 17개, base 와 0 dB 조건, dev 9곡 / test 8곡, `note_f1_onset50`, dev 최적 하나를 test 에서 확인,
  곡 블록 쌍 부트스트랩 10,000회).
- 결정 규칙: E38 과 같다 — test 에서 CI 하한 > 0, Δ ≥ +1.0점, 조건별 부분집합 CI 상한 < −2 없음이면 채택 후보. 추가 가드(실제 곡,
  `data/tmp/m10/e41_real.py`, 같은 piano+other 전체 전사): 고른 규칙에서 정답 악보 6곡의 합친 기타 마디 F1 이 평균 1점 넘게 떨어지거나 한 곡이라도
  3점 넘게 떨어지면 판단 보류(기본 끔). 채택되면 파이프라인에 piano+other 전곡 전사(곡당 GPU 약 2분)와 `notes` 단계의 표시(`bleed.json`, 뷰어
  후보)를 넣는다 — 그 구현은 별도 작업.
- 함께 보고: 지운 음 수와 그중 진짜 음, 조건별 Δ, 범주별(현악·기타 신스·피아노) Δ, J-pop 원곡 7곡에서 지워질 기타 음 비율(서술).
- 한계: E38 과 같다. 전사기가 piano+other 의 기타 소리를 비기타로 부를 때 진짜 음을 잃는다(그게 측정 대상).
- 판정 설정: `rule=superiority; metric=note_f1_onset50; baseline=production; mde=1.0; split_col=section; select=dev; split=test; subset_col=scenario`
- 비용: GPU 약 30분(Tier B 52개 조건 × 75 s piano+other 전사) + 약 25분(가드: 정답 악보 6곡·J-pop 7곡의 piano+other 전곡 전사). 사용자 승인
  2026-10-23.
- 실행 명령: `bandscribe eval exp E41`, 그다음 `data/tmp/m10/e41_real.py`
- 결과:
  - 2026-10-23 `20261023-205003_3b771bd_exp-e41`: 계산된 판정 = 기각 (dev 최적이 기본값 production). GPU 27분(piano+other 창 전체 전사).
    가드 `data/tmp/m10/e41_real.py`(GPU 17분, 13곡 전곡).
  - 풀링 F1(base + 0 dB) — dev: production 74.3, vote30 73.4, vote50 73.2, vote50o 72.3; test: production 77.0, vote30 75.2. **모든 규칙이 지운
    가짜 음보다 잃은 진짜 기타 음이 많다.** vote50, 0 dB 조건(전 곡): 피아노 잃은 진짜 523 / 지운 가짜 141, 오르간 185 / 3, 현악 188 / 37, 기타
    신스 123 / 22, base 70 / 20.
  - 해석: 편곡에서 건반·현악은 기타와 같은 음을 같은 시각에 자주 친다(코드 반주를 함께 깐다). 그래서 "piano+other 에도 그 음이 있다"는 블리드의
    증거가 아니라 흔한 겹침이다. E38(에너지)과 E41(전사 투표) 모두 같은 이유로 실패 — 섞인 건반·현악 음을 기타 음과 가르려면 음 단위 규칙이
    아니라 기타를 더 잘 떼어 내는 분리(또는 음색을 보는 전사기)가 필요하다.
  - 가드(서술): 봇치 6곡은 piano+other 전사에서 비기타 음이 0–1개라 영향 없음. J-pop 원곡에서 vote50 이 지웠을 기타 음: 青と夏 17.8 %,
    lulu. 17.4 %, 狭心症 11.7 %, 怪獣の花唄 4.1 %, AIZO 3.4 %, 絶対零度 2.3 %, 逆光 0.1 % — Tier B 결과로 보면 대부분 진짜 음.
  - 2026-10-23 `20261023-205003_3b771bd_exp-e41`: 계산된 판정 = 기각 (dev 최적이 기본값 production)
