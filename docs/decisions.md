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
- 상태: 사전 등록
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
- 결과: (아직 없음)

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
