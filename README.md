# gtab

밴드 녹음 한 곡을 넣으면 기타 파트별 탭과 베이스 탭을 만드는 **개인용** 로컬 도구다.
Windows 10 + RTX 2060(6 GB)에서 돌도록 만들었다. 결과물과 중간 산출물은 모두 비공개로 쓴다.

- 설계: [`docs/DESIGN.md`](docs/DESIGN.md) · M0 인터페이스: [`docs/M0_SPEC.md`](docs/M0_SPEC.md)
- 지금 단계: **M0(기반과 입력)**. 환경 점검, 입력 처리(파일·YouTube), 작업 캐시까지 된다. 분리·전사·탭은 M2 이후다.

## 빠른 시작

```bat
D:\gtab\gtab.cmd doctor                                   :: 환경 점검 (GPU 셀프테스트 포함, 필수 항목 실패 시 종료 코드 1)
D:\gtab\gtab.cmd ingest "D:\녹음\합주 2026-09-30.wav"      :: 로컬 파일 입력
D:\gtab\gtab.cmd ingest "https://www.youtube.com/watch?v=..."  :: YouTube 링크 입력 (기본 켜짐, 아래 고지 참고)
D:\gtab\gtab.cmd status                                   :: 작업 목록 / gtab status <작업 키> 로 하나 자세히
D:\gtab\gtab.cmd config show                              :: 지금 적용되는 설정 전체
D:\gtab\gtab.cmd gc                                       :: 중단된 작업이 남긴 임시 폴더 정리
```

설정 우선순위: `gtab/defaults.toml` → `data/config.toml`(사용자) → 작업 `hints.toml` → 명령줄 `--set 키=값`.
`--set`의 섹션·키 이름이 틀리면 조용히 무시하지 않고 오류(종료 코드 2)로 멈춘다.

## 입력: 로컬 파일이 1순위

- **로컬 파일을 권한다**(CD·구입 음원, 합주 녹음 원본). FLAC, WAV, MP3, M4A, OGG, Opus, WebM을 받는다.
- 원본은 `jobs/<작업 키>/input/source.<확장자>`로 **바이트 그대로** 보관한다. 재인코딩하지 않는다.
- 분석용 믹스 `mix_44k_f32.wav`는 스테레오 float32, soxr 44.1 kHz다(샘플레이트는 44100 고정). 크게 마스터링된 곡의
  0 dBFS 넘는 샘플을 자르지 않고, float 게인으로 true peak를 −1 dBFS 아래로 내린 뒤 그 게인을 `meta.json`에 적는다.
- 작업 키는 파일이면 `f-<PCM 해시 앞 16자>`, YouTube면 `yt-<동영상 ID>`다. 같은 오디오를 다른 이름·다른 컨테이너로
  넣어도 같은 작업이 되고, 이미 처리한 입력은 1초 안에 캐시에서 돌려준다.
- 입력 설정(true peak 천장, 플래그 문턱 등)을 바꾸면 캐시를 그대로 쓰지 않고 `input/`을 새 설정으로 다시 만든다.

## YouTube 입력: 기본 켜짐과 위험 고지

**2026-09-30 사용자 요청으로 YouTube 링크 입력은 기본으로 켜져 있다.** 사용자는 아래 위험을 고지받고 기본 켜기를 요청했으며,
그 동의 문구가 처음 실행할 때 `data/config.toml`의 `[consent] youtube`에 기록된다. 위험은 사용자 본인 책임이다.

위험 (설계서 S0, 법률 자문 아님):

- **약관:** YouTube 약관(한국어판, 2022-01-05 시행)은 허락 없는 다운로드와 자동 접근을 금지한다.
- **사적 복제:** 저작권법 제30조 사적 복제에는, 불법 복제물임을 알고 한 복제는 제외된다는 판결이 있다(2008카합968).
- **기술적 보호조치:** yt-dlp는 `--js-runtimes node`로 YouTube의 JS 챌린지(signature, n 파라미터)를 푼다. 이 **기본 경로를 포함한**
  YouTube 다운로드가 해외 판례에서 기술적 보호조치 우회로 판단된 바 있다(독일 함부르크 법원, youtube-dl 호스팅 사건:
  지방법원 2023-03 말, 고등법원 항소 기각 2024-11-27 보도). 한국 저작권법 제104조의2가 적용되는지는 확인되지 않았다.
- **추가 위험:** 쿠키와 PO 토큰은 계정 정지와 추가 우회 해석의 위험이 더해진다.

gtab이 하는 것과 하지 않는 것:

- **쿠키·PO 토큰은 쓰지 않는다.** YouTube가 켜져 있어도 따로 꺼져 있고(`youtube.cookies = false`, `youtube.po_token = false`),
  yt-dlp에 `--no-cookies --no-cookies-from-browser --ignore-config`를 넘긴다. 켜려면 `[consent]`에 날짜가 적힌 별도 동의가
  있어야 하며, M0 코드는 그래도 쓰지 않고 거부한다.
- 동영상 하나만 받는다(재생목록·채널 주소는 거부). 가장 좋은 오디오 스트림을 원본 컨테이너 그대로 받고, DRC 스트림은 피하며,
  format_id·abr·코덱을 `meta.json`에 적는다. 다운로드 사이에는 5초 이상 쉰다(명령을 따로 실행해도 적용).
- `youtube.timeout_s`(기본 600초)가 지나면 yt-dlp 프로세스 전체를 끝낸다.
- 이미 받은 동영상은 네트워크 없이 캐시에서 돌려준다.
- 실패하면 `D:\gtab\tools\yt-dlp.exe --update-to nightly` 후 다시 시도하고, 그래도 안 되면 **로컬 파일을 넣는다.**

**YouTube 입력 끄기:** `data/config.toml`에 아래를 넣는다. 그러면 URL 입력을 거부한다.

```toml
[youtube]
enabled = false
```

한 번만 끄려면 `gtab ingest ... --set youtube.enabled=false`. 확인은 `gtab config show`.

## MuScriptor 권리 보증 (M2부터)

전사 모델 MuScriptor는 Hugging Face 게이트 모델(CC BY-NC 4.0)이다. 게이트 문구가 **입력 음악에 대한 권리 보증과 면책**을
요구한다. 사용자가 **본인 계정으로 직접** 게이트를 수락하고 `hf auth login`을 실행해야 하며, gtab이나 Claude가 대신 수락하지
않는다. 넣는 음악에 그런 권리가 있는지는 사용자가 판단한다. 수락하지 않으면 Basic Pitch·YourMT3+로 내려가고 품질이 떨어진다.
비상업(NC) 조건이므로 결과를 상업적으로 쓰지 않는다.

## 재배포 금지

- **모델 가중치**(라이선스 불명·비상업 포함)와 **분리된 스템**, **작업 폴더**(`data/jobs/`: 원곡 오디오 사본이 들어 있다),
  **다운로드**(`data/downloads/`)는 공유·재배포하지 않는다. 탭 결과도 개인 비공개로 쓴다.
- `data/`, `tools/*.exe`는 git에 올리지 않는다(`.gitignore`). 모델 출처는 기록만 하고 파일은 저장소 밖에 둔다.

## 폴더와 환경

- 코드와 데이터는 모두 `D:\gtab` 아래에 있다. `data\`에 작업(`jobs`), 다운로드, 로그(`logs\gtab.log`), 사용자 설정,
  모델 캐시(`hf`, `torch`), uv 캐시가 있다. `%APPDATA%`/`%LOCALAPPDATA%`에는 아무것도 두지 않는다.
- 두 가상환경: core `D:\gtab\.venv`(Python 3.12, CLI·분석, torch 없음)와 gpu `D:\gtab\envs\gpu\.venv`(torch 2.11.0+cu128,
  GPU 워커 전용). GPU 작업은 Job Object 안의 자식 프로세스로 한 번에 하나만 돌고, `data\gpu.lock`이 이를 지킨다.
- 동기화: `D:\gtab\tools\uvw.cmd sync --extra core --group dev`(`D:\gtab`에서). 전역 Python 3.10은 건드리지 않는다.
- 테스트: `D:\gtab\.venv\Scripts\python.exe -m pytest -q`. GPU·네트워크가 필요한 테스트는 없으면 자동으로 건너뛴다.
