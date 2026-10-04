# 실험 등록부 (DESIGN §8.1, §8.6)

모든 선택지는 여기의 실험으로 관리한다. **실험 전에** 주 지표, 최소 검출 효과(MDE), 데이터, 분석 방법, 결정 규칙을 적는다.
**첫 실행 뒤에는 지표·MDE·데이터·규칙을 바꾸지 않는다.** 바꿔야 하면 새 id로 다시 사전 등록한다.

- 제목 줄 `## <id>` 의 id가 곧 `bandscribe eval exp <id>` 의 id다(정확히 같아야 한다).
- 상태: `사전 등록` → (첫 결과가 나오면 러너가 바꿈) `실행 중` → (사람이 확인 후) `결정: 채택|기각|판단 보류 (이유)`.
  러너는 `사전 등록`·`실행 중` 만 실행하고, `결정:` 항목은 거부한다(다시 보려면 새 id로 등록).
  데이터·모델이 없어 결과 없이 끝난 실행은 상태를 바꾸지 않는다. 데이터를 기다리기로 했으면
  `bandscribe eval exp <id> --record "판단 보류 (데이터 대기)"` 로 사람이 기록한다.
- `판정 설정` 의 `select=dev; split=test`: dev 에서 선택지 하나를 고르고 그 하나만 test 에서 확인한다(test 로 고르지 않음).
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
- 결과: (아직 없음)

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
- 상태: 사전 등록
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
- 결과: (아직 없음)

## E24-amt
- 제목: MuScriptor dtype — 업스트림(fp32 가중치 + 내장 fp16 autocast) vs fp16 가중치(조건화기 fp32)
- 상태: 사전 등록
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
- 상태: 사전 등록
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
- 결과: (아직 없음)
