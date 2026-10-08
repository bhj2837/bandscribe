"""Layered configuration.

Layers, later wins: package ``defaults.toml`` -> user ``data/config.toml`` -> job ``hints.toml``
-> ``--set a.b=value`` overrides. The merged dict is validated by pydantic models; known sections
forbid unknown keys (typos fail loudly). Unknown top-level sections in *files* are kept in
``Config.extra`` so a later milestone can add sections to the user file before its model exists;
on the command line (``--set``) an unknown section is rejected, because there it is a typo.

``defaults.toml`` is the source of truth for values. The model defaults below mirror it so that a bare
``Config()`` also works (handy in tests); ``tests/test_config.py`` fails if the two drift apart.

M1a/M2 sections (M1_M2_SPEC 1.3): ``run``, ``hints``, ``sep``, ``amt``, ``instr``, ``grid``, ``sections``, ``s35``,
``eval``, ``datasets`` and the new ``[gpu]`` VRAM keys. Stages read only their params (cache keys, DESIGN 9.3);
``[gpu]`` keys are non-semantic and never enter a stage key.

Core env only (the GPU worker env has no tomli_w; workers receive plain dicts, not Config).
"""

from __future__ import annotations

import datetime as _dt
import difflib
import logging
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bandscribe import atomic, paths

log = logging.getLogger(__name__)

DEFAULTS_PATH: Path = Path(__file__).with_name("defaults.toml")

# Recorded verbatim in the user config created by ensure_user_config(). Do not reword: it documents what
# the user was told and agreed to (DESIGN S0 risk notice) when asking for YouTube input to be on by default.
YOUTUBE_CONSENT_TEXT = "2026-09-30 사용자 동의: YouTube 약관(다운로드 금지)과 JS 챌린지 해결의 기술적 보호조치 우회 위험(해외 판례, 한국법 미검증)을 고지받고 기본 켜기를 요청함. 개인 비공개 사용."

# ingest.sample_rate is fixed: the mix file is named mix_44k_f32.wav and the f-<PCM sha256> job key is
# defined on a 44.1 kHz decode (DESIGN S0), so another rate would give the same audio another job.
SAMPLE_RATE = 44100

_MISSING: Any = object()


class ConfigError(ValueError):
    """Invalid config file, --set syntax or value. The message is user-facing (Korean)."""


# --------------------------------------------------------------------------------------------- models


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class YoutubeCfg(_Section):
    enabled: bool = True
    format: str = "ba[format_id!*=drc][acodec=opus]/ba[format_id!*=drc]/ba"
    cookies: bool = False
    po_token: bool = False
    timeout_s: int = Field(600, gt=0)


class IngestCfg(_Section):
    sample_rate: int = SAMPLE_RATE
    true_peak_ceiling_dbfs: float = Field(-1.0, le=0.0)
    quasi_mono_side_mid_db: float = -30.0
    lowpass_detect_hz: int = Field(16000, gt=0)

    @field_validator("sample_rate")
    @classmethod
    def _fixed_sample_rate(cls, v: int) -> int:
        if v != SAMPLE_RATE:
            raise ValueError(
                f"{SAMPLE_RATE} 으로 고정입니다 (mix_44k_f32.wav 이름과 f-<PCM 해시> 작업 키가 "
                "44.1 kHz 디코드를 전제로 합니다)"
            )
        return v


GPU_BACKENDS: tuple[str, ...] = ("sep_msst", "amt_muscriptor", "beats_beatthis", "f0_bass")


class GpuCfg(_Section):
    lock_timeout_s: int = Field(7200, gt=0)
    worker_timeout_s: int = Field(7200, gt=0)
    # Headroom for other apps' fluctuation and fragmentation only; the CUDA context is counted separately
    # (ctx_mb_default / the bench's measured ctx_mb, M1_M2_SPEC 8.1).
    vram_margin_gb: float = Field(0.7, ge=0.0)
    target_reserved_gb: float = Field(4.5, gt=0.0)
    insufficient_vram_policy: Literal["wait", "error", "cpu"] = "wait"
    # M2 (M1_M2_SPEC 1.3, 8): wait policy, worker-side check, Sysmem Fallback detection.
    wait_poll_s: float = Field(30.0, gt=0.0)
    wait_max_s: float = Field(3600.0, ge=0.0)
    # All the waiting of one GPU stage together (pre-check waits + retries after the worker's insufficient_vram);
    # without it, wait_max_s applied to each of the 1 + max_worker_retries pre-checks (up to ~4 h; review 2026-10-05).
    wait_total_max_s: float = Field(3600.0, ge=0.0)
    max_worker_retries: int = Field(3, ge=0)
    worker_headroom_mb: float = Field(256.0, ge=0.0)
    # CUDA context estimate until the bench measures it; workers measured 79-91 MB on this PC (2026-10-03).
    ctx_mb_default: float = Field(150.0, ge=0.0)
    shared_growth_fail_mb: float = Field(64.0, gt=0.0)  # calibrated from fallback-off bench runs
    # [잠정] this pid's shared usage with the model resident minus before the upload; above it = a spill at load
    # time (bench 2026-09-30: normal loads +0-4 MB, MuScriptor upstream loader +32 MB, spills +64-70 MB). 0 = off.
    load_spill_fail_mb: float = Field(48.0, ge=0.0)
    rtf_fail_ratio: float = Field(0.5, gt=0.0, le=1.0)
    # Backends that may use a CPU rung, and only under policy "cpu" (DESIGN 12: only small models on CPU).
    cpu_backends: list[Literal["sep_msst", "amt_muscriptor", "beats_beatthis", "f0_bass"]] = Field(
        default_factory=lambda: ["beats_beatthis"]
    )


# ------------------------------------------------------------------------------------- M1a/M2 sections

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
STFT_HOP = 512  # BS-RoFormer SW stft_hop_length: chunk sizes must be multiples of it (DESIGN S3)
_LUFS_MSG = '"off" 이거나 -30 ~ -6 사이의 LUFS 숫자여야 합니다'


RUN_PROFILES: tuple[str, ...] = ("fast", "quality", "eval")


class RunCfg(_Section):
    # fast / quality: no full-mix B0 pass, sampled piano+other presence pass (fast: half the windows).
    # eval: quality + the full-mix B0 pass and the full-length piano+other pass as extra outputs (baselines,
    # evaluation); its instrumentation / guitar pass equal quality's. max = later.
    profile: Literal["fast", "quality", "eval"] = "quality"


class HintsCfg(_Section):
    """User hints; CLI flags write these through --set (--instruments -> hints.instruments)."""

    # "auto" or a comma list of families (guitar,keys,synth,bass,strings,brass,winds,vocals); parsed by AMT
    # (bandscribe.amt.instruments), which also rejects unknown family names.
    instruments: str = "auto"
    parts: str = "auto"  # used from M6 on; stored now
    # The song's title in the score, the viewer and `status`; empty = the input's file name (YouTube: video title).
    title: str = ""


class SepCfg(_Section):
    backend: Literal["sw_msst"] = "sw_msst"
    model_dir: str = "data/models/bs_roformer_sw"  # relative to BANDSCRIBE_ROOT
    ckpt: str = "BS-Rofo-SW-Fixed.ckpt"
    config: str = "BS-Rofo-SW-Fixed.yaml"
    ckpt_sha256: str = Field(
        "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e", pattern=r"^[0-9a-f]{64}$"
    )
    msst_commit: str = Field("84b1eac0887756b4f1a9d7a1ff49105939749ed2", pattern=r"^[0-9a-f]{40}$")
    chunk_ladder: list[int] = Field(default_factory=lambda: [588800, 352256, 262144])
    num_overlap: int = Field(2, ge=1)
    batch_size: int = Field(1, ge=1)
    dtype: Literal["upstream", "fp16"] = "upstream"  # upstream = fp32 weights + AMP; fp16 = E24-sep only
    # E1: "off", or a target LUFS in [-30, -6] (gain before SW, inverse gain after).
    input_lufs: Literal["off"] | float = "off"
    leftover_warn_db: float = -20.0  # used by the `stems` stage only

    @field_validator("chunk_ladder")
    @classmethod
    def _ladder(cls, v: list[int]) -> list[int]:
        # The worker tries the rungs in list order and steps down: largest first, whole STFT hops each.
        if not v:
            raise ValueError("비어 있으면 안 됩니다 (예: [588800, 352256, 262144])")
        for c in v:
            if c <= 0 or c % STFT_HOP:
                raise ValueError(f"{c}: {STFT_HOP} 의 양의 배수여야 합니다 (SW 의 stft_hop_length)")
        if any(a <= b for a, b in zip(v, v[1:])):
            raise ValueError("큰 칸부터 작은 칸 순서(엄격히 감소)로 적어야 합니다")
        return v

    @field_validator("input_lufs", mode="before")
    @classmethod
    def _input_lufs(cls, v: Any) -> Any:
        if isinstance(v, str):
            if v.strip().lower() == "off":
                return "off"
            raise ValueError(_LUFS_MSG)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(_LUFS_MSG)
        if not -30.0 <= float(v) <= -6.0:
            raise ValueError(f"{v}: {_LUFS_MSG}")
        return float(v)


class AmtCfg(_Section):
    muscriptor_model: Literal["small", "medium", "large"] = "medium"
    muscriptor_revision: str = Field("f32236969308476e01fd3aae67357de5feb05a2d", pattern=r"^[0-9a-f]{40}$")
    # Used only if that size's model.safetensors is already in the HF cache (never downloaded).
    muscriptor_fallback: list[Literal["small", "medium", "large"]] = Field(default_factory=lambda: ["small"])
    # lean = build on CUDA + load weights on the CPU (no second GPU copy): ~0.67 GB less reserved, identical notes
    # (AMT test_lean_loader_parity on a synthetic clip and a real 60 s excerpt, 2026-10-03).
    muscriptor_loader: Literal["upstream", "lean"] = "lean"
    # upstream = fp32 weights + MuScriptor's built-in fp16 autocast; fp16 = E24-amt only. Never bf16 (sm_75).
    dtype: Literal["upstream", "fp16"] = "upstream"
    prelude_forcing: bool = True
    beam_size: int = Field(1, ge=1)
    cfg_coef: float = 1.0
    batch_size: int = Field(1, ge=1)
    latency_s: float = Field(-0.003, ge=-1.0, le=1.0)  # latency experiment 2026-10-03; subtracted from MuScriptor times
    piano_other_instruments: Literal["keys", "keys+guitar"] = "keys+guitar"  # M1_M2_SPEC A5
    # piano+other presence pass (S3.5, bandscribe.amt.presence): auto = sampled windows in every profile (eval runs
    # the full-length view as an extra output only); sampled | full force the presence source. The sampled budget: up to presence_max_windows windows of
    # presence_window_s, at most presence_max_total_s and presence_max_fraction of the song (fast: half the
    # windows); a window qualifies if >= presence_min_active of its frames have piano+other above
    # instr.active_rel_db, or its power-mean level is above it.
    presence_pass: Literal["auto", "sampled", "full"] = "auto"
    presence_window_s: float = Field(10.0, ge=2.0, le=60.0)
    presence_max_windows: int = Field(6, ge=1, le=50)
    presence_max_total_s: float = Field(60.0, gt=0.0)
    presence_max_fraction: float = Field(0.2, gt=0.0, le=1.0)
    presence_min_active: float = Field(0.5, ge=0.0, le=1.0)
    guitar_view: Literal["guitar_mono", "mix_mono", "nonvox_mono", "guitar_other_mono"] = "guitar_mono"  # E2
    guitar_mask: Literal["guitar_only", "guitar+present", "guitar+present+strings", "all"] = "guitar+present"  # E3
    bp_onset_threshold: float = Field(0.6, gt=0.0, lt=1.0)  # E23 (2026-10-03) tuned these three
    bp_frame_threshold: float = Field(0.4, gt=0.0, lt=1.0)
    # Shortest KEPT note in frames (1 frame = 256/22050 s = 11.61 ms); the worker passes frames - 1 to Basic
    # Pitch, which drops notes of length <= its threshold (M1_M2_SPEC A11).
    bp_min_note_frames: int = Field(7, ge=1, le=30)
    bp_min_freq_hz: float = Field(0.0, ge=0.0)  # 0 = backend default
    bp_max_freq_hz: float = Field(0.0, ge=0.0)
    bp_melodia_trick: bool = True
    bp_threads: int = Field(4, ge=1, le=64)  # onnxruntime intra-op threads, pinned (stable NoteSets)
    # notes stage: drop guitar/bass notes that start where their stem is this far below the mix's loud level
    # (p95 of its 10 Hz frames); MuScriptor writes notes over silent stems (bandscribe.amt.silence). -200 = off.
    silence_gate_db: float = Field(-50.0, ge=-200.0, le=0.0)

    @field_validator("bp_min_note_frames", mode="before")
    @classmethod
    def _bp_min_note_frames(cls, v: Any) -> Any:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("1 ~ 30 사이의 정수(프레임 수)여야 합니다")
        if v >= 12:
            # 12 frames = upstream's 127.7 ms default, which deletes fast picking (DESIGN S6, M1_M2_SPEC A11);
            # longer minimums delete even more (review 2026-10-05: the guard used to stop only at exactly 12).
            raise ValueError(
                f"{v} 프레임은 Basic Pitch 기본값(12 프레임 = 127.7 ms) 이상이라 빠른 음을 지웁니다(DESIGN S6). "
                "11 이하를 쓰세요"
            )
        return v


class InstrCfg(_Section):
    threshold: Prob = 0.5  # E3
    guitar_threshold: Prob = 0.15  # low: omitting a guitar class deletes its notes
    guitar_stem_active_ratio: Prob = 0.05  # guitar stem active in >= 5 % of bars -> all 3 guitar classes
    active_rel_db: float = Field(-40.0, le=0.0)  # a stem/bar is "active" above this RMS relative to the mix
    mix_class_min_notes: int = Field(8, ge=0)
    presence_min_notes: int = Field(8, ge=0)  # notes a class needs in the piano+other presence pass to count


class GridCfg(_Section):
    checkpoint: str = Field("final0", pattern=r"^[A-Za-z0-9_.-]+$")  # model "beat_this.<checkpoint>"
    dbn: bool = False  # true is rejected (needs madmom, not installed)
    refine_window_ms: float = Field(35.0, ge=0.0)  # Beat This! times are quantised to 20 ms
    half_double_check: bool = True
    compound_triple_ratio: Prob = 0.6  # E14
    compound_margin: Prob = 0.1
    downbeat_snap_ms: float = Field(70.0, ge=0.0)
    meter_smooth_bars: int = Field(8, ge=1)

    @field_validator("dbn")
    @classmethod
    def _no_dbn(cls, v: bool) -> bool:
        if v:
            raise ValueError("grid.dbn=true 는 madmom 이 필요해 지원하지 않습니다")
        return v


class SectionsCfg(_Section):
    min_section_bars: int = Field(4, ge=1)
    max_sections: int = Field(16, ge=2)
    repeat_min_score: Prob = 0.6
    align_max_offset_ms: float = Field(50.0, ge=0.0)


class S35Cfg(_Section):
    vocal_active_rel_db: float = -30.0
    resid_min_occurrences: int = Field(3, ge=2)


class TabCfg(_Section):
    # M3: 0 = auto (the merged guitar is written as staff + MIDI only, DESIGN 4.2; part splitting comes in M7a),
    # 1 = the guitars are one part: fret the merged guitar into a tab track (`bandscribe run --parts 1`)
    guitar_parts: int = Field(0, ge=0, le=1)
    # "auto" = tuning search; else a tuning the user names ("E standard", "Eb standard"/"하프다운", "Drop D",
    # "Open G, capo 2", ...; checked by the tab stage's params, Korean ConfigError). Usually set per song in the
    # job's hints.toml.
    guitar_tuning: str = Field("auto", min_length=1)
    bass_tuning: str = Field("auto", min_length=1)


class BassCfg(_Section):
    # M5 (DESIGN 5): the S11 clean-up of the pass-1 bass with the F0 tracks (bandscribe.bass.clean); E13
    # (docs/decisions.md, 2026-10-08) turned on only the merges (fragments, slides); the other steps were on hold.
    anchor: bool = False  # octave / semitone moves from the 2-of-3 tracker vote + harmonic duplicates dropped
    kick: bool = False  # notes on a kick without any F0 dropped (needs the drums stem)
    fragments: bool = True  # same-pitch fragments and decaying tails merged
    slides: bool = True  # glide chains -> one note sliding into the note it stops on
    gap_fill: bool = False  # stable two-tracker F0 where MuScriptor wrote nothing -> low-confidence notes
    policy: Literal["muscriptor", "and"] = "muscriptor"  # "and" = keep only F0-confirmed notes (E13 comparison)
    techniques: bool = True  # vibrato / dead-note suggestions on the notes


class EvalCfg(_Section):
    tpb: int = Field(48, ge=1)
    onset_tol_s: float = Field(0.05, gt=0.0)
    bootstrap_iterations: int = Field(10000, ge=1)
    seed: int = 20260930
    mde_points: float = Field(1.0, gt=0.0)  # default MDE (F1 points); costly elements use 1.5 (DESIGN 8.1)


class DatasetsCfg(_Section):
    root: str = "data/datasets"  # relative to BANDSCRIBE_ROOT


class DoctorCfg(_Section):
    min_free_disk_gb: int = Field(60, ge=0)
    min_node_major: int = Field(22, ge=0)


class ConsentCfg(_Section):
    """Explicit, dated risk acknowledgements. Only the user config writes these, never the defaults."""

    youtube: str | None = None
    cookies: str | None = None
    po_token: str | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    youtube: YoutubeCfg = Field(default_factory=YoutubeCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    gpu: GpuCfg = Field(default_factory=GpuCfg)
    doctor: DoctorCfg = Field(default_factory=DoctorCfg)
    consent: ConsentCfg = Field(default_factory=ConsentCfg)
    # M1a/M2 (M1_M2_SPEC 1.3)
    run: RunCfg = Field(default_factory=RunCfg)
    hints: HintsCfg = Field(default_factory=HintsCfg)
    sep: SepCfg = Field(default_factory=SepCfg)
    amt: AmtCfg = Field(default_factory=AmtCfg)
    instr: InstrCfg = Field(default_factory=InstrCfg)
    grid: GridCfg = Field(default_factory=GridCfg)
    sections: SectionsCfg = Field(default_factory=SectionsCfg)
    s35: S35Cfg = Field(default_factory=S35Cfg)
    tab: TabCfg = Field(default_factory=TabCfg)
    bass: BassCfg = Field(default_factory=BassCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    datasets: DatasetsCfg = Field(default_factory=DatasetsCfg)
    # Unknown top-level sections, kept verbatim (see module docstring).
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_unknown_sections(cls, data: Any) -> Any:
        # Runs for every construction path (load_config, model_validate, tests), so callers never have
        # to split known and unknown sections themselves.
        if not isinstance(data, Mapping):
            return data
        known = set(cls.model_fields) - {"extra"}
        data = dict(data)
        extra = dict(data.pop("extra", None) or {})
        for key in list(data):
            if key not in known:
                extra[key] = data.pop(key)
        data["extra"] = extra
        return data

    @model_validator(mode="after")
    def _risky_options_need_consent(self) -> Config:
        # Cookies / PO tokens add risk on top of the YouTube path itself (DESIGN S0), so they are only
        # accepted together with a dated consent entry, like the YouTube default was.
        for opt in ("cookies", "po_token"):
            if getattr(self.youtube, opt) and not getattr(self.consent, opt):
                raise ValueError(
                    f"youtube.{opt}=true 는 추가 위험(계정 정지, 추가 우회 해석)이 있어 "
                    f"data/config.toml 의 [consent] {opt} = \"날짜 + 동의 내용\" 이 있어야 켤 수 있습니다"
                )
        return self

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """``cfg.get("youtube.enabled")``; also reaches into unknown sections (``cfg.get("web.port")``).

        Raises KeyError when the key is missing and no default is given.
        """
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Config) and part not in Config.model_fields:
                node = node.extra
            if isinstance(node, BaseModel) and part in type(node).model_fields:
                node = getattr(node, part)
            elif isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise KeyError(dotted)
                return default
        return node


KNOWN_SECTIONS: tuple[str, ...] = tuple(k for k in Config.model_fields if k != "extra")


# ---------------------------------------------------------------------------------------------- merge


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: tables merge recursively, anything else in ``over`` replaces ``base``."""
    out: dict[str, Any] = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        elif isinstance(value, Mapping):
            out[key] = deep_merge({}, value)
        else:
            out[key] = value
    return out


def parse_value(raw: str) -> Any:
    """Parse a --set value as a TOML literal (true, 600, 0.5, "x", [1, 2]); fall back to the raw string."""
    if "\n" in raw or "\r" in raw:
        return raw  # a newline could smuggle extra keys into the one-line TOML document below
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw
    # TOML would turn 2026-09-30 into a date object; no setting wants that, the text is more useful.
    if isinstance(value, (_dt.date, _dt.time)):
        return raw
    return value


def parse_override(item: str) -> tuple[list[str], Any]:
    """``"youtube.enabled=false"`` -> (["youtube", "enabled"], False)."""
    key, sep, raw = item.partition("=")
    parts = [p.strip() for p in key.split(".")]
    if not sep or not key.strip() or any(not p for p in parts):
        raise ConfigError(f"--set 형식이 잘못되었습니다: {item!r} (예: --set youtube.enabled=false)")
    return parts, parse_value(raw.strip())


def _override_dict(parts: list[str], value: Any) -> dict[str, Any]:
    node: Any = value
    for part in reversed(parts):
        node = {part: node}
    return node


def _read_toml(path: Path, what: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ConfigError(f"{what}을(를) 읽지 못했습니다: {path} ({e})") from e
    try:
        # utf-8-sig: Notepad's "UTF-8 with BOM" is common on Windows and tomllib rejects the BOM.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ConfigError(f"{what}이(가) UTF-8이 아닙니다: {path} (UTF-8로 다시 저장해 주세요)") from e
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{what}의 TOML 문법 오류: {path} ({e})") from e


def _format_validation_error(err: ValidationError) -> str:
    lines = ["설정 값이 올바르지 않습니다:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(전체)"
        if e["type"] == "extra_forbidden":
            lines.append(f"  - {loc}: 알 수 없는 키입니다 (오타인지 확인해 주세요)")
        else:
            msg = e["msg"].removeprefix("Value error, ")
            scalar = isinstance(e.get("input"), (str, int, float, bool))
            given = f" (입력값: {e['input']!r})" if scalar else ""
            lines.append(f"  - {loc}: {msg}{given}")
    lines.append("설정 파일: bandscribe/defaults.toml, data/config.toml, 작업 hints.toml, 그리고 --set 값을 확인해 주세요.")
    return "\n".join(lines)


def _unknown_set_section(section: str, item: str) -> ConfigError:
    # Config files may carry sections for later milestones, but on the command line an unknown section
    # is almost always a typo, and silently ignoring e.g. `--set youtub.enabled=false` would download.
    close = difflib.get_close_matches(section, KNOWN_SECTIONS, n=1)
    hint = f" '{close[0]}'을(를) 뜻했나요?" if close else ""
    return ConfigError(
        f"--set {item}: 알 수 없는 섹션 '{section}' 입니다.{hint} "
        f"(--set 으로 바꿀 수 있는 섹션: {', '.join(KNOWN_SECTIONS)})"
    )


# ----------------------------------------------------------------------------------------------- load


def load_config(
    job_hints: Path | None = None,
    overrides: Iterable[str] = (),
    *,
    user_config: Path | None = None,
) -> Config:
    """Merge all layers and validate. Missing user config / hints files are simply skipped.

    ``user_config`` defaults to ``paths.USER_CONFIG`` (looked up at call time so tests can repoint it).
    Raises ConfigError with a user-facing message.
    """
    data = _read_toml(DEFAULTS_PATH, "기본 설정")

    user_path = Path(paths.USER_CONFIG if user_config is None else user_config)
    if user_path.is_file():
        data = deep_merge(data, _read_toml(user_path, "사용자 설정"))
        log.debug("config layer: user %s", user_path)

    if job_hints is not None:
        hints_path = Path(job_hints)
        if hints_path.is_file():
            data = deep_merge(data, _read_toml(hints_path, "작업 hints"))
            log.debug("config layer: hints %s", hints_path)
        else:
            log.debug("config layer: hints %s not found, skipped", hints_path)

    for item in overrides:
        parts, value = parse_override(item)
        if parts[0] not in KNOWN_SECTIONS:
            raise _unknown_set_section(parts[0], item)
        data = deep_merge(data, _override_dict(parts, value))
        log.debug("config layer: --set %s", item)

    try:
        return Config.model_validate(data)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(e)) from e


def dump_effective(cfg: Config) -> str:
    """The effective config as TOML (unknown sections back at top level, where they were written)."""
    import tomli_w  # core-only dependency; imported lazily so `import bandscribe.config` stays light

    data = cfg.model_dump(mode="json", exclude_none=True)
    data.update(data.pop("extra", {}))
    # An all-None section (e.g. [consent] with no user config) would print as a bare, confusing header.
    return tomli_w.dumps({k: v for k, v in data.items() if v != {}})


# ---------------------------------------------------------------------------------------- user config


def _user_config_template(path: Path) -> str:
    import tomli_w

    header = [
        "# bandscribe 사용자 설정",
        f"# 위치: {path}",
        "# 여기 적은 값이 패키지 기본값(bandscribe/defaults.toml)을 덮어쓴다.",
        "# 우선순위: 기본값 -> 이 파일 -> 작업 hints.toml -> 명령줄 --set 키=값",
        "# 지금 적용되는 값 전체 보기: bandscribe config show",
        "#",
        "# 예) YouTube 링크 입력 끄기 (기본은 켜짐, 아래 [consent] 참고):",
        "# [youtube]",
        "# enabled = false",
        "#",
        "# 쿠키와 PO 토큰은 추가 위험(계정 정지 등)이라 YouTube를 켜도 따로 꺼 둔다.",
        "# 켜려면 [consent] 에 cookies / po_token 동의 문구(날짜 포함)를 함께 적어야 한다.",
        "",
    ]
    consent = tomli_w.dumps({"consent": {"youtube": YOUTUBE_CONSENT_TEXT}})
    return "\n".join(header) + "\n" + consent


def ensure_user_config(path: Path | None = None) -> Path:
    """Create ``data/config.toml`` (commented header + YouTube consent) if missing. Never overwrites."""
    target = Path(paths.USER_CONFIG if path is None else path)
    if target.exists():
        return target
    # Two processes racing here both write identical content via os.replace, which is harmless.
    atomic.write_text(target, _user_config_template(target))
    log.info("사용자 설정 파일을 만들었습니다: %s", target)
    return target
