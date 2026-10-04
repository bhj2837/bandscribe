# bandscribe

> **English summary.** bandscribe is a personal, local Windows tool whose goal is to turn one band recording into
> one guitar tab per part plus a bass tab (drum and piano scores come later). **It does not produce tabs yet.**
> Today (milestone M2 of M0–M12) it separates 6 stems, transcribes guitar and bass to MIDI with a tempo map, and
> builds practice backing tracks. Splitting several guitars into parts, the core of the project, is planned for
> M6–M8. Engineering highlights: a one-model-at-a-time GPU worker inside Windows Job Objects, a content-addressed
> stage cache, pre-registered experiments judged with song-block bootstrap CIs, and a sampled presence pass (plus
> moving a baseline pass out of the default profile) that cut a full run from 12–25 min to 6–8 min per song on an
> RTX 2060. The documentation is in Korean.

밴드 녹음 한 곡을 넣으면 **기타 파트마다 탭 하나와 베이스 탭**을 만드는 것이 목표인 Windows 로컬 도구다.
드럼·피아노 악보는 그 다음 계획이다. 기타가 두 대 이상인 곡에서 "누가 어떤 음을 쳤는가"는 오디오 분리만으로는
거의 가를 수 없다. 그래서 bandscribe는 분리와 전사로 음표 후보와 증거(스테레오 위치, 음색 클래스, 반복, 기호 규칙)를
모은 뒤 **음표 하나하나를 파트에 배정**하도록 설계했다. 지금은 M2까지 구현되어 6 스템 분리, 기타·베이스 MIDI,
연습용 반주를 만든다. **탭과 기타 파트 분리는 아직 없다.**

- 설계서: [`docs/DESIGN.md`](docs/DESIGN.md) · 구현 명세: [`docs/M0_SPEC.md`](docs/M0_SPEC.md), [`docs/M1_M2_SPEC.md`](docs/M1_M2_SPEC.md)
- 실험 등록부: [`docs/decisions.md`](docs/decisions.md) · 현재 상태와 측정 기록: [`docs/PROGRESS.md`](docs/PROGRESS.md)

## 목차

1. [지금 되는 것과 아직 안 되는 것](#지금-되는-것과-아직-안-되는-것)
2. [마일스톤](#마일스톤)
3. [파이프라인](#파이프라인)
4. [설계 결정과 이유](#설계-결정과-이유)
5. [측정 결과](#측정-결과)
6. [실행 방법](#실행-방법)
7. [법적 고지와 개인 사용](#법적-고지와-개인-사용)
8. [폴더 구조](#폴더-구조)
9. [테스트](#테스트)

## 지금 되는 것과 아직 안 되는 것

`bandscribe run <파일|URL> --until notes`(기본 프로필 `quality`)가 만드는 것:

| 결과 | 설명 |
|---|---|
| 6 스템 | vocals, drums, bass, guitar, piano, other (+ `nonvox` = 믹스 − 보컬·드럼·베이스, `leftover` = 믹스 − 6 스템 합) |
| 박·마디 격자 | Beat This! 비트·다운비트에 자체 후처리(곡 안의 반/배 박 전환 정리, 동적 계획법 마디선)를 더한 템포맵·박자표 |
| 편성 추정 | 악기 클래스별 존재 확률(`instrumentation.json`, 건반·신스·현악 등). 기타 전사의 악기 마스크를 정하는 데 쓴다 |
| `guitar_all.mid` | 기타 스템 전사(MuScriptor). **파트별이 아니다.** 모든 기타를 음색 클래스(어쿠스틱·클린·디스토션)별 트랙으로만 나눈다 |
| `bass_raw.mid` | 베이스 스템 원시 전사. 옥타브 보정·주법은 M5 |
| 연습용 반주 | 기타 뺀 믹스, 베이스 뺀 믹스, 베이스 스템. 분리 오차가 그대로 남는다 |

MIDI는 오디오 시간축을 그대로 두고 템포맵만 격자에 맞춘다. DAW에 넣으면 마디선이 곡과 맞는다.

함께 있는 도구:

- `bandscribe doctor`: GPU 셀프테스트, ffmpeg 빌드 옵션, Node, 디스크, NVML 등 환경 점검. 필수 항목이 실패하면 종료 코드 1.
- `bandscribe bench gpu`: 백엔드·설정 칸마다 VRAM과 실시간 배수를 재서 VRAM 사전 점검 표를 만든다.
- `bandscribe eval ...`: 지표, 곡 블록 부트스트랩 비교, 사전 등록 실험 러너, 외부 결과 가져오기.
- `bandscribe gt ...`: Guitar Pro 정답 가져오기(반복·다른 엔딩 펼침), 참조 렌더, DTW 정렬, 다운비트 탭 보정.
- `bandscribe data ...`: 평가용 데이터셋(Cambridge-MT, EGDB, GuitarSet, IDMT-SMT-Bass, FiloBass, MedleyDB) 받기·검증·로더.

**아직 없는 것:** 탭(.gp5, alphaTex)과 양자화·운지(M3), 기타 파트(Line) 분리(M6–M8), 편집 UI(M4a), 베이스 완성(M5),
주법·코드 차트(M9). 드럼·피아노 악보는 v1 범위 밖이다.

## 마일스톤

각 마일스톤이 끝날 때마다 그 시점에 쓸 수 있는 도구가 남도록 나눴다. 순서는 M0 → M1a → M2 → M3 → M4a → M5 → M6 →
M7a → M4b → M7b → M7c → M8 → M9 → M10 → M11 → (M12)이고, M1b는 M1a 뒤로 계속 병행한다.

| | 내용 | 상태 |
|---|---|---|
| **M0** | 기반: 환경 점검, 입력(파일·YouTube), 콘텐츠 주소 캐시, DAG 실행기, Job Object GPU 워커 | **완료** |
| **M1a** | 측정 도구: 지표, 곡 블록 부트스트랩, 사전 등록 실험 러너, 정답 가져오기·DTW 정렬, 합성 리믹스, 데이터셋 로더 | **구현됨**, 리뷰·인수 전 |
| **M1b** | Tier A: 실제로 치는 곡의 구간 정답(Guitar Pro) 구축 | 대기: 후보 5곡 등록, 정답 파일 필요 |
| **M2** | 첫 수직 슬라이스: 분리, 전사, 편성 추정, MIDI, 연습용 반주, GPU 벤치 | **구현·통합 완료**(상용곡 5곡 + Cambridge-MT 1곡 end-to-end), 리뷰·인수 전 |
| M3 | 첫 탭: 양자화, 튜닝·Viterbi 운지, .gp5/alphaTex 내보내기(기타는 합친 트랙, 베이스는 원시 탭) | 다음 |
| M4a | 로컬 웹 뷰어(alphaTab)와 기본 편집, 수정 로그 | 계획 |
| M5 | 베이스 완성: 옥타브 앵커, 튜닝 후보, 주법 | 계획 |
| **M6** | 스테레오 라우터와 공간 경로: **첫 자동 기타 파트 분리** | 계획 (핵심) |
| **M7a** | Part 뷰: 파트 수(K) 추정, 음표별 배정, 미배정(Unassigned) 트랙 | 계획 (핵심) |
| M4b | A/B 믹서, 검토 큐 | 계획 |
| **M7b** | 정체성 추적: 트랙렛, 재식별, HMM 평활 | 계획 (핵심) |
| M7c | Role/Player 뷰(연주자 수에 맞춘 트랙 배치) | 계획 |
| M8 | 좁은 믹스(모노·같은 위치) 경로, 반복 차감 | 계획 |
| M9 | 주법, 코드 차트, GP7 백킹 | 계획 |
| M10 | 수정 로그로 배정 가중치·신뢰도 보정 | 계획 |
| M11 | YouTube 편의 기능(앨범판 찾기, 트리밍) | 계획 |
| M12 | (선택) 자체 모델 학습. 평가가 한계를 보일 때만 | 조건부 |

각 마일스톤의 산출물과 완료 기준은 [`docs/DESIGN.md`](docs/DESIGN.md) §10에 있다.

## 파이프라인

### 지금 (M2, 13단계)

입력 처리(ingest) 뒤에 13단계가 돈다. 주황색이 GPU 단계다. 주요 의존만 그렸고, 정확한 의존 관계는
[`bandscribe/pipeline/graph.py`](bandscribe/pipeline/graph.py)에 있다.

```mermaid
flowchart LR
    src["입력<br/>로컬 파일 · YouTube"] --> ingest["ingest<br/>float32 44.1 kHz 디코드<br/>LUFS · true peak"]
    ingest --> beats["beats<br/>Beat This!"]
    beats --> grid["grid<br/>박 · 마디 · 박자표"]
    grid --> sections["sections<br/>구간 · 반복 지도"]
    ingest --> sep["sep<br/>BS-RoFormer SW<br/>6 스템"]
    sep --> stems["stems<br/>nonvox · leftover<br/>연습용 반주 · 모노 뷰"]
    stems --> vocal["vocal<br/>보컬 활동"]
    stems --> resid1["resid1<br/>반복 잔차"]
    sections --> ms1
    stems --> ms1["amt_ms1<br/>MuScriptor<br/>베이스 전곡 + piano/other 표본 구간"]
    stems --> bp["amt_bp<br/>Basic Pitch (CPU)"]
    ms1 --> instr["instr<br/>편성 추정"]
    instr --> gtr["amt_gtr<br/>MuScriptor 기타 스템<br/>편성 기반 마스크"]
    gtr --> s35["s35<br/>1차 분석 색인"]
    vocal --> s35
    resid1 --> s35
    bp --> s35
    s35 --> notes["notes<br/>guitar_all.mid<br/>bass_raw.mid"]
    grid -. 템포맵 .-> notes
    classDef gpu fill:#fde2c8,stroke:#c46a1b,color:#000
    class beats,sep,ms1,gtr gpu
```

### 계획 (M3–M8)

```mermaid
flowchart LR
    cand["M2 음표 후보<br/>스템 · 편성"] --> router["S4 라우터 (M6)<br/>4마디 창마다 스테레오 경로 판정"]
    router --> views["S5 뷰 (M6, M8)<br/>centre · S · L · R · 반복 잔차"]
    views --> tx["S6 뷰별 전사<br/>MuScriptor + Basic Pitch"]
    tx --> fuse["S7 음표 융합<br/>블리드 감점 · 반복 합의"]
    fuse --> diar["S8 파트 다이어라이저 (M7)<br/>파트 수 K · 음표별 배정"]
    fuse --> bass["S11 베이스 (M5)"]
    diar --> quant["S12 양자화 (M3)"]
    bass --> quant
    quant --> fret["S13 튜닝 · 운지 Viterbi (M3)"]
    fret --> export["S15 .gp5 · alphaTex · MIDI"]
```

단계별 입력·출력·위험은 [`docs/DESIGN.md`](docs/DESIGN.md) §2–§3에 있다.

## 설계 결정과 이유

### 1. 오디오로 리드·리듬을 가르지 않고, 분리 → 전사 → 음표 단위 배정

- **결정:** 기타 스템을 파트별 오디오로 쪼개려 하지 않는다. 여러 "뷰"(centre, S, L, R, 반복 잔차)에서 전사한 음표에
  증거를 모아 **음표마다** 어느 파트(Line)의 것인지 늦게 정한다. 파트별 오디오는 연습용 부산물로만 둔다.
- **이유:**
  - 설계 조사에서 같은 음색 기타 2대의 실녹음 분리는 두 번째 기타가 0.2–1.4 dB SDR(GuitarDuets)에 그쳤다. 추정 음표로
    조건화해도 1.01 → 1.04 dB였다.
  - 검증된 공개 lead/rhythm 분리 모델이 없다. 커뮤니티 모델은 지표가 없고, 상용 서비스는 비공개 세트 수치뿐이다.
  - 산출물은 오디오가 아니라 탭이다. "이 음을 어느 파트가 쳤나"만 맞히면 된다.
- **먼저 정한 것:** 출력 계약 R1–R9. 예: 더블 트래킹은 파트 하나, 유니즌 음은 양쪽 파트에 모두 적기, 비기타는 절대 파트가
  아님, 한 파트는 한 사람이 칠 수 있어야 함. 정답 세트도 같은 규칙으로 만든다.
- 기타 파트 수(K)를 세는 발표된 방법은 없다(관련 연구는 모두 K를 입력으로 받는다). 그래서 강건한 하한과 사전분포로
  K를 추정하는 부분은 자체 구현이다(M7a). 자체 lead/rhythm 분리기 학습은 평가가 필요를 보일 때만 한다(M12).
- 근거: [`docs/DESIGN.md`](docs/DESIGN.md) §1.3, §1.4, §4.

### 2. 스테레오: 팬 위치가 아니라 L/R coherence로 경로를 고른다 (M6, 설계 단계)

- **결정:** 4마디 창마다 S/M 에너지비, 에너지 가중 L/R coherence, |ILD| 비율, 1–40 ms 지연 상호상관으로 경로를 고른다.
  경로는 R0 좁음, R1 ADT, R2 넓은 더블, R3 다른 위치의 다른 파트이고, 경로마다 다른 뷰와 분할 규칙을 쓴다.
- **이유:** 합성 실험([`router_exp.py`](docs/research/experiments/router_exp.py))에서 bin별 ILD(팬)는 리듬 더블 구간
  에너지의 56 %를 '센터'로 잘못 봤다. 좌우로 벌린 리듬 더블은 양쪽 에너지가 비슷해서 레벨 차이로는 가운데 있는
  것처럼 보인다. 하지만 서로 다른 두 테이크라 좌우 coherence가 낮고, 가운데 놓인 모노 음원은 coherence가 높다.
- **지금 상태:** 특징 추출([`analysis/stereo.py`](bandscribe/analysis/stereo.py))은 연구 스크립트를 그대로 옮겼고,
  원본과 ±0.02 안에서 같은지 테스트한다. 라우터 자체는 M6이다. 분리기가 스테레오 정보를 보존하는지가 이 설계의
  전제라서, 그 측정(`stereo-preservation`)을 먼저 사전 등록해 뒀다(아직 실행 전).
- 근거: [`docs/DESIGN.md`](docs/DESIGN.md) §4.5, S4.

### 3. GPU에는 모델 하나: Job Object 워커, 프로세스가 모두 끝날 때까지 락 유지

- **결정:** CLI(core 환경)는 torch를 import하지 않는다(테스트로 강제). GPU 단계는 gpu 환경의 자식 프로세스가 모델 하나를
  로드해 그 단계의 입력을 모두 처리하고 종료한다. 종료하면 VRAM이 통째로 반환된다.
- **이유:** Windows의 venv `python.exe`는 런처다. 실제 인터프리터를 자식으로 띄운다. 런처만 끝내면 VRAM을 쥔 자식이 남고,
  다음 워커는 OOM 대신 NVIDIA Sysmem Fallback에 빠져 조용히 느려진다(설계 조사상 약 10배).
- **구현:**
  - ctypes로 만든 Job Object(`KILL_ON_JOB_CLOSE`)에 워커를 넣는다. `CREATE_SUSPENDED`로 띄워, 런처가 자식을 만들기 전에
    Job에 들어가게 한다.
  - `data/gpu.lock`은 **Job 안의 프로세스가 모두 끝날 때까지** 쥔다. 타임아웃·Ctrl+C에서도 트리 전체를 끝낸 뒤에 푼다.
    예외는 하나다: 드라이버에 걸려 끝나지 않는 프로세스는 70초 뒤 포기하고 락을 풀되, 그 실행을 실패로 기록하고 오류를 남긴다.
  - 이전 보유자가 강제 종료됐으면 pid 기록을 보고 그 프로세스들이 사라질 때까지 기다린다(종료에 9–38 ms 걸리는 것을 쟀다).
  - 시작 전에 NVML로 여유 VRAM을 재서 설정 사다리(예: 분리 청크 588,800 → 352,256 → 262,144)에서 맞는 칸을 고르거나
    기다린다. Windows 성능 카운터의 Shared Usage 증가로 Sysmem Fallback을 감지한다.
  - MuScriptor 업스트림 로더는 가중치를 GPU에 두 벌 올려 reserved 2.39 GB가 들었다. state dict를 CPU에서 읽는 로더로
    1.74 GB로 줄였고, 업스트림과 같은 음표를 내는지 패리티 테스트를 통과한 뒤에 기본값으로 바꿨다.
- 근거: [`docs/DESIGN.md`](docs/DESIGN.md) §9.5, [`gpu.py`](bandscribe/gpu.py), [`winjob.py`](bandscribe/winjob.py),
  [`tests/test_gpu_runner.py`](tests/test_gpu_runner.py).

### 4. 콘텐츠 주소 단계 캐시

- **결정:** 단계 키 = sha256(단계 이름, 코드 버전, 해석된 파라미터, 입력 해시, 모델 SHA256). 결과는
  `jobs/<작업 키>/stages/<단계>/<키 앞 12자>/`에 둔다. 작업 키는 파일이면 PCM 해시, YouTube면 동영상 ID다. 같은 오디오를
  다른 이름이나 다른 컨테이너로 넣어도 같은 작업이 된다.
- **이유:** GPU 단계 하나가 곡당 몇 분씩 걸린다. 파라미터만 바꾼 재실행이 앞 단계를 다시 계산하면 안 되고, 반쯤 쓴
  결과를 완료로 착각해서도 안 된다.
- **구현:** 임시 폴더에 manifest까지 다 쓴 뒤 폴더 rename 한 번으로 공개한다. 파라미터 변형은 나란히 남아 A/B 비교가 쉽다.
  내보낸 WAV는 캐시 파일에 하드링크하고 읽기 전용으로 둔다. MIDI는 복사해서 DAW에서 고쳐도 캐시가 바뀌지 않는다.
- **효과:**
  - 13단계가 모두 캐시면 `run --until notes`가 약 0.8초에 끝난다.
  - 패키지 이름 변경과 폴더 이동 뒤에도 단계 키 1,755개 중 바뀐 것이 0개였다. 몇 시간 분량의 GPU 결과를 그대로 다시 썼다.
  - GPU 전사가 캐시된 상태에서 하류 단계를 강제로 다시 계산하면 결과 파일이 byte 단위로 같았다(1곡에서 확인).
    GPU 단계 자체(fp16 autocast, 자기회귀 디코딩)는 비트 단위로 결정적이지 않다.
- 근거: [`docs/DESIGN.md`](docs/DESIGN.md) §9.3, [`jobs/store.py`](bandscribe/jobs/store.py),
  [`docs/RENAME.md`](docs/RENAME.md).

### 5. 사전 등록 실험과 곡 블록 부트스트랩

- **결정:** 기본값은 모두 [`docs/decisions.md`](docs/decisions.md)의 실험(E#)으로 정한다. **첫 실행 전에** 주 지표,
  최소 검출 효과(MDE), 데이터, 분할, 분석 방법, 결정 규칙을 적는다. 실행한 뒤에는 바꾸지 않고, 바꾸려면 새 id로 다시
  등록한다.
- **판정 규칙:**
  - 95 % CI는 곡(또는 클립·연주자) 단위 블록 부트스트랩 10,000회로 낸다. 같은 곡의 음표끼리는 독립이 아니기 때문이다.
  - 채택하려면 CI가 0을 제외하고, 점추정이 MDE 이상이고, 어떤 시나리오 하위 세트에서도 명백히 나빠지지 않아야 한다.
  - CI가 0을 포함하면 "판단 보류"로 적고 기본값을 유지한다. 선택은 dev에서 하고, test는 확인에만 쓴다.
  - [학습 데이터 오염 표](docs/eval/contamination.md)에서 모델이 학습했을 수 있는 세트(불명 포함)로는 절대 성능을
    주장하지 않는다.
- **지금까지:** 6건을 실행해 채택 2건(E23, latency), 판단 보류 4건(E1, E2, E3, gate-m2)이다. 4건은 사전 등록만 했다.
  gate-m2는 수치상 문턱을 넘었지만 오염 규칙 때문에 통과로 쓰지 않았다(아래 측정 결과).
- `bandscribe eval exp <id>`가 등록부를 읽어 실행하고, 이미 결정된 항목은 다시 실행하지 않는다.

### 6. 표본 존재 패스: 편성은 "있는지"만 알면 된다

- **문제:** 기타 전사의 악기 마스크를 정하려고 piano+other 스템 전곡을 MuScriptor로 전사했다. 패드가 많은 곡은 오디오
  1초에 약 3초가 걸려(4.6분 곡에 864초) 파이프라인 대부분을 차지했다. 전체 믹스 전사(기준선 B0)도 기본 경로에서 돌았다.
- **결정:**
  - B0는 기본 프로필에서 빼 `eval` 프로필의 추가 출력으로 옮겼다. 평가용 출력이 평가 대상을 바꾸지 않도록, `eval`도
    편성과 기타 전사는 `quality`와 똑같이 정한다.
  - piano+other는 마디 시작에 맞춘 10초 창을 최대 6개(곡의 20 %, 60초 이하)만 전사한다. 창은 결정적으로 고른다:
    활성 스템을 먼저 덮고, 서로 떨어뜨리고, 다른 구간을 우선한다.
  - 오검출을 막는 조건: 한 클래스가 8음 이상, 두 구간 이상에서 나오고, 그 클래스의 스템이 마디의 10 % 이상 활성이어야
    "있음"으로 본다.
- **효과(RTX 2060, `quality`, 처음부터 실행한 단계 시간 합):** 곡당 12.4–24.6분 → 5.7–7.7분(1.8–3.2배).
  해당 단계(`amt_ms1`)만 보면 394–1,205초 → 10–46초다(베이스 패스가 캐시된 경우).
- **한계:** 창 밖에서만 연주되는 파트는 놓친다. 기타 마스크에는 안전한 쪽의 실패다(마스크가 기타만으로 남는다). M6의
  블리드 감점은 전곡 음표가 필요해서 전곡 패스를 따로 요청해야 한다.
- 근거: [`amt/presence.py`](bandscribe/amt/presence.py), [`docs/PROGRESS.md`](docs/PROGRESS.md) "Speed and presence fix".

### 7. 발견: 악기 마스크를 바꾸면 블리드가 줄기보다 출력이 뒤섞인다

MuScriptor의 `instruments` 인자는 조건이면서 동시에 나머지 악기를 하드 마스크한다. 기타 스템에 샌 건반 소리를 건반
클래스로 받아 내려고 마스크를 넓혀 봤다. 정답 악보가 없어서, 원곡의 기타 음 중 **기타를 뺀 반주에서도 나오는 음**
(같은 음높이, onset ±50 ms)을 블리드로 보고 비교했다.

| 곡 | 마스크 변경 | 기타 음표 수 | 블리드 비율 | 예전 출력에서 남은 음: 진짜 음 / 블리드 |
|---|---|---|---|---|
| KH | 기타만 → + electric_piano | 3,543 → 2,307 (−35 %) | 17.7 % → 11.9 % | 1,050/2,916 (36 %) / 239/627 (38 %) |
| aotonat | 건반 4종 → 3종 (chromatic_percussion 제외) | 3,649 → 4,796 (+31 %) | 17.0 % → 17.7 % | 1,372/3,029 (45 %) / 476/620 (77 %) |

- 마스크를 바꾸면 블리드만 골라 지워지지 않고 출력 전체가 바뀐다. 진짜 음과 블리드가 비슷한 비율로 남거나(KH),
  블리드가 더 많이 남는다(aotonat). KH에서 블리드 비율이 줄어든 대가로 진짜 기타 음이 최대 약 530개 사라졌을 수 있다.
- 그래서 마스크를 넓히는 대신 전사 **뒤에** 음표 단위로 블리드를 감점하는 쪽(S7)을 택했다. 기본 마스크는 E3에서 정한
  `guitar+present`로 두고, 정답 데이터로 판정할 E3b를 사전 등록했다.
- 한계: 비교에 쓴 반주는 다른 도구로 기타를 지운 것이라 불완전하다. 곡은 개발 PC의 상용곡이고 문서의 곡 코드로 적었다.
- 근거: [`docs/PROGRESS.md`](docs/PROGRESS.md) "Instrumentation and guitar mask", [`docs/decisions.md`](docs/decisions.md) E3, E3b.

## 측정 결과

개발 PC: Windows 10, Ryzen 5 3600, RAM 16 GB, RTX 2060 6 GB. 다른 데스크톱 앱이 VRAM 1.5–2 GB를 쓰는 평소 상태에서 쟀다.

| 항목 | 결과 | 조건과 주의 |
|---|---|---|
| 분리 SDR (BS-RoFormer SW vs 참 스템) | 기타 7.6 dB · 베이스 10.0 · 드럼 7.9 · 보컬 9.0 | Cambridge-MT의 Secretariat "Over The Top" 1곡. 멀티트랙 합을 −8 LUFS로 리미터 마스터링한 믹스이고, 참 스템은 최소제곱 게인으로 맞춘 근사다. 해먼드 오르간은 piano가 아니라 other 스템으로 갔다. SW의 학습 데이터가 불명이라 진단용이다 |
| 곡당 실행 시간 | 5.7–7.7분 (이전 12.4–24.6분) | `quality`, 3.4–4.6분 상용곡 3곡, 처음부터 실행한 단계 시간 합. 가장 큰 단계는 기타 전사(141–197초) |
| 분리 단계 | 52–99초 (실시간의 3.6–5.6배) | 상용곡 5곡 + Cambridge-MT 1곡 |
| VRAM (reserved 피크) | 분리 1.69 GB · MuScriptor 1.74 GB · Beat This! 0.3–0.4 GB | 한 번에 하나만 올라간다 |
| 캐시된 재실행 | 약 0.8초 | 13단계 모두 캐시, `run --until notes` |
| E23 Basic Pitch 튜닝 | onset F1 66.0 → 72.6, **Δ +6.6 [+3.8, +9.9]** | EGDB 공식 test 25클립 × 앰프 렌더 5종(Basic Pitch 학습 밖). dev에서 60개 조합 중 골랐다. 업스트림 기본 최소 음 길이(127.7 ms)는 빠른 16분음표를 지운다. 최적값이 탐색 격자의 끝에 있다 |
| MuScriptor onset 지연 | 상수 −3 ms, 보정 후 \|잔차\| 중앙값 **10.9 ms** [10.5, 11.3] | GuitarSet 연주자 00–02로 추정, 03–05로 보고. 목표 10 ms에 못 미친다. 출력 시각이 10 ms 격자로 양자화돼 있고 정답 onset의 오차도 섞인다. 오염 불명이라 절대 정확도는 주장하지 않는다 |
| gate-m2 (디스토션, MuScriptor − Basic Pitch) | Δ +8.1 [+2.1, +13.5] → **판단 보류** | 문턱 +5를 넘지만, MuScriptor 학습 데이터에 EGDB가 있는지 알 수 없어 사전 등록 규칙대로 통과로 쓰지 않았다 |
| E1, E2, E3 | 판단 보류 | 라우드니스 정규화, 전사 입력 뷰, 악기 마스크. 모두 CI가 0을 포함해 기본값 유지 |

E# 실험은 각각 [`docs/decisions.md`](docs/decisions.md)에 등록 문구와 결과가 있다. 원시 측정값(작업 폴더, 실행 로그)은
원곡 오디오 분석이라 저장소에 넣지 않았다.

## 실행 방법

**요구 사항**

- Windows 10/11 x64, NVIDIA GPU. 개발과 측정은 RTX 2060 6 GB에서만 했다. gpu 환경은 torch 2.11.0+cu128을 쓴다.
- ffmpeg(libsoxr, chromaprint, ebur128 포함 빌드)와 Node.js 22 이상이 PATH에 있어야 한다.
- [uv](https://docs.astral.sh/uv/) 실행 파일을 `tools\uv.exe`에 둔다. `tools\uvw.cmd`가 Python 설치와 캐시를 저장소 폴더
  안에 둔다(`%APPDATA%`를 쓰지 않는다).
- 디스크 여유 60 GB 이상(doctor 기준). 저장소 경로는 ASCII로 둔다.

**설치**

```bat
git clone https://github.com/bhj2837/bandscribe
cd bandscribe
:: core 환경 (Python 3.12, CLI·분석, torch 없음)
tools\uvw.cmd sync --locked --extra core --group dev
:: gpu 환경 (torch 2.11.0+cu128, MuScriptor, Beat This!)
cd envs\gpu && ..\..\tools\uvw.cmd sync --locked && cd ..\..
:: bp310 환경 (Python 3.10, Basic Pitch ONNX; 없으면 Basic Pitch 단계만 건너뛴다)
cd envs\bp310 && ..\..\tools\uvw.cmd sync --locked && cd ..\..
:: (선택) Guitar Pro 정답 가져오기·렌더용 alphaTab
cd node && npm ci && cd ..
```

**모델은 직접 받는다.** 이 저장소는 모델 가중치를 포함하지도, 대신 받지도 않는다. 라이선스는
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)를 본다.

1. **MuScriptor medium** (전사, CC BY-NC 4.0, Hugging Face 게이트): 본인 계정으로 모델 페이지의 사용 조건을 직접 수락한다.
   조건에는 입력 음악에 대한 권리 보증과 면책이 들어 있다. 그다음 로그인하고 고정된 리비전을 받는다.

   ```bat
   tools\hf-login.cmd
   set HF_HOME=%CD%\data\hf
   tools\uvw.cmd tool run --from huggingface_hub hf download MuScriptor/muscriptor-medium --revision f32236969308476e01fd3aae67357de5feb05a2d
   ```

2. **Beat This! final0** (비트, MIT): `bandscribe.cmd models fetch beat_this.final0`. URL·크기·라이선스를 보여 주고 y/N을 받는다.
3. **BS-RoFormer SW** (분리): 원 업로더 계정이 삭제되어 공식 배포처가 없고 라이선스도 불명이다. 이 저장소는 배포하지
   않는다. 직접 구한 사본을 `data\models\bs_roformer_sw\`에 `BS-Rofo-SW-Fixed.ckpt`와 `.yaml`로 둔다. 기대하는 SHA256은
   [`bandscribe/defaults.toml`](bandscribe/defaults.toml)의 `sep.ckpt_sha256`이다.
4. **Basic Pitch** (보조 전사, Apache-2.0): bp310 환경의 휠에 들어 있다.

**실행**

```bat
bandscribe.cmd doctor                              :: 환경 점검 (GPU 셀프테스트 포함)
bandscribe.cmd run "C:\music\song.flac"            :: 기본: --until notes, --profile quality
bandscribe.cmd run "C:\music\song.flac" --dry-run  :: 단계별 캐시 여부만 보기
bandscribe.cmd status                              :: 작업 목록
bandscribe.cmd config show                         :: 지금 적용되는 설정
```

- 결과는 `data\jobs\<작업 키>\export\`에 모인다: `midi\guitar_all.mid`, `midi\bass_raw.mid`, `practice\*.wav`,
  `instrumentation.json`. 스템은 `stages\sep\<키>\stems\`에 있다.
- 프로필: `fast`(표본 창 절반), `quality`(기본), `eval`(B0 믹스 전사와 piano+other 전곡 전사를 추가 출력으로 더함, 훨씬 느림).
- 설정 우선순위: `bandscribe/defaults.toml` → `data/config.toml` → 작업 `hints.toml` → `--set 키=값`. 틀린 키는
  무시하지 않고 오류로 멈춘다.
- GPU 작업 전에는 VRAM을 많이 쓰는 앱(브라우저, 게임)을 닫는 편이 좋다. 여유가 모자라면 기본 정책은 기다리기다.
- 평가 데이터셋은 `bandscribe.cmd data list`로 라이선스를 확인하고 `data fetch <이름>`으로 받는다.

## 법적 고지와 개인 사용

법률 자문이 아니다. 자세한 위험 검토는 [`docs/DESIGN.md`](docs/DESIGN.md) S0에 있다.

- **개인용 도구다.** 분리한 스템, 연습용 반주, MIDI, 앞으로 만들 탭은 모두 원곡의 파생물이다. 공유·재배포·상업적 이용을
  하지 않는다. 이 저장소에는 오디오, 스템, 작업 폴더, 모델 가중치가 없다(`data/`는 git에 올리지 않는다).
- **입력은 권리가 있는 로컬 파일을 권한다.** CD·구입 음원·직접 녹음한 합주가 그렇다.
- **YouTube 입력은 기본으로 켜져 있다.** 개발자 본인이 아래 위험을 검토하고 정한 기본값이다. 처음 실행하면
  `data/config.toml`이 만들어지면서 개발자의 동의 문구(2026-09-30)가 `[consent]`에 그대로 적힌다. 다른 사람에게는 이것이 그
  사람의 동의가 아니다. 위험을 직접 판단하고, 쓰지 않으려면 `data/config.toml`에 `[youtube]` `enabled = false`를 넣는다.
  - YouTube 약관은 허락 없는 다운로드와 자동 접근을 금지한다.
  - yt-dlp가 YouTube의 JS 챌린지를 푸는 방식이 해외 판례(독일 함부르크, youtube-dl 호스팅 사건)에서 기술적 보호조치
    우회로 판단됐다는 보도가 있다. 한국 저작권법에 어떻게 적용되는지는 확인하지 않았다.
  - 쿠키와 PO 토큰은 쓰지 않는다. 동영상 하나만 받고, 다운로드 사이에 5초 이상 쉰다.
- **모델 라이선스:** MuScriptor 가중치는 CC BY-NC 4.0이고 게이트 조건(입력 음악 권리 보증·면책)을 사용자가 직접
  수락한다. BS-RoFormer SW는 출처·라이선스가 불명이라 개인용으로만 쓰고 배포하지 않는다. Beat This!는 MIT, Basic Pitch는
  Apache-2.0이다. 벤더링한 MSST 모델 코드는 MIT다([`bandscribe/third_party/msst/LICENSE`](bandscribe/third_party/msst/LICENSE),
  이 코드의 원본인 lucidrains/BS-RoFormer 의 고지는 [`LICENSE.BS-RoFormer`](bandscribe/third_party/msst/LICENSE.BS-RoFormer)).
  전체 목록은 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **데이터셋:** 각 데이터셋의 약관을 따른다(예: Cambridge-MT는 교육용 약관). `data fetch`는 받기 전에 라이선스를 보여 준다.
- **이 저장소의 코드:** [`LICENSE`](LICENSE)를 따른다. 위의 모델·데이터·벤더링 코드에는 각자의 라이선스가 적용된다.

## 폴더 구조

```
bandscribe/            core 패키지 (torch import 금지)
  cli.py, commands/      CLI (typer, 한국어 출력)
  ingest/                디코드, 라우드니스, 입력 플래그, YouTube(yt-dlp.exe 래퍼)
  jobs/                  콘텐츠 주소 저장소, DAG 실행기
  pipeline/              13단계 그래프, 실행기, export 게시
  analysis/              박·마디 격자, 구간·반복, 보컬 활동, 반복 잔차, 스테레오 특징
  sep/                   분리 단계, 파생 스템·모노 뷰·연습용 반주
  amt/                   전사 단계, 표본 존재 패스, 편성 추정, MIDI 쓰기
  workers/               GPU·bp310 워커: sep_msst, amt_muscriptor, beats_beatthis, amt_basicpitch
  gpu.py, winjob.py      GPU 락, Job Object (ctypes)
  vram.py, sysmon.py     VRAM 사다리·대기, NVML·성능 카운터
  eval/                  지표, 블록 부트스트랩, 사전 등록 실험 러너, GP 가져오기, DTW 정렬, 합성 리믹스
  datasets/              Tier B/C 데이터셋 받기·검증·로더
  schema/                pydantic 스키마
  third_party/msst/      MSST BS-RoFormer 모델 코드 (MIT, 커밋 고정)
envs/gpu/              GPU 워커 환경 (Python 3.12, torch 2.11.0+cu128)
envs/bp310/            Basic Pitch 워커 환경 (Python 3.10, ONNX)
node/                  alphaTab (Guitar Pro 정답 가져오기·렌더)
tools/                 uvw.cmd, hf-login.cmd, yt-dlp.conf
docs/                  설계서, 명세, 진행 상황, 실험 등록부, 오염 표, 조사 자료
tests/                 pytest
data/                  (git 밖) 작업, 모델, HF·uv 캐시, 데이터셋, 실험 실행 결과
```

## 테스트

```bat
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

- 테스트 1,199개. 개발 PC에서 1,198개 통과, 1개 건너뜀, 약 7분.
- GPU, gpu·bp310 환경, Node, 데이터셋, 모델, 네트워크가 필요한 테스트는 마커로 표시되어 있고, 없으면 자동으로 건너뛴다.
- 주요 대상:
  - GPU 워커 정리: 타임아웃·Ctrl+C·남은 자식 프로세스가 있을 때 트리 전체가 끝난 **뒤에** 락이 풀리는지,
    강제 종료된 이전 보유자를 기다리는지.
  - 캐시: 원자적 공개, 임시 폴더 정리, 이름 변경 전에 쓴 결과 파일 호환.
  - 지표 항등식: 항등이면 F1 1.0, 무작위 20 % 삭제면 재현율 0.80, +12반음이면 pitch F1 0·chroma F1 1.0, 라벨 교환은
    헝가리안 뒤 1.0, 시스템 격자 오류가 음표 지표에 새지 않음.
  - 업스트림·연구 코드와의 패리티: MSST 분리, MuScriptor 로더, 라우터 특징, 반복 차감.
  - core 환경이 torch를 import하지 않는지, 평가 실행이 두 번 모두 byte 단위로 같은 결과를 내는지.
