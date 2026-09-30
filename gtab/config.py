"""Layered configuration.

Layers, later wins: package ``defaults.toml`` -> user ``data/config.toml`` -> job ``hints.toml``
-> ``--set a.b=value`` overrides. The merged dict is validated by pydantic models; known sections
forbid unknown keys (typos fail loudly). Unknown top-level sections in *files* are kept in
``Config.extra`` so a later milestone can add sections to the user file before its model exists;
on the command line (``--set``) an unknown section is rejected, because there it is a typo.

``defaults.toml`` is the source of truth for values. The model defaults below mirror it so that a bare
``Config()`` also works (handy in tests); ``tests/test_config.py`` fails if the two drift apart.

Core env only (the GPU worker env has no tomli_w; workers receive plain dicts, not Config).
"""

from __future__ import annotations

import datetime as _dt
import difflib
import logging
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from gtab import atomic, paths

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


class GpuCfg(_Section):
    lock_timeout_s: int = Field(7200, gt=0)
    worker_timeout_s: int = Field(7200, gt=0)
    vram_margin_gb: float = Field(0.7, ge=0.0)
    target_reserved_gb: float = Field(4.5, gt=0.0)
    insufficient_vram_policy: Literal["wait", "error", "cpu"] = "wait"


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
        """``cfg.get("youtube.enabled")``; also reaches into unknown sections (``cfg.get("sep.model")``).

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
    lines.append("설정 파일: gtab/defaults.toml, data/config.toml, 작업 hints.toml, 그리고 --set 값을 확인해 주세요.")
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
    import tomli_w  # core-only dependency; imported lazily so `import gtab.config` stays light

    data = cfg.model_dump(mode="json", exclude_none=True)
    data.update(data.pop("extra", {}))
    # An all-None section (e.g. [consent] with no user config) would print as a bare, confusing header.
    return tomli_w.dumps({k: v for k, v in data.items() if v != {}})


# ---------------------------------------------------------------------------------------- user config


def _user_config_template(path: Path) -> str:
    import tomli_w

    header = [
        "# gtab 사용자 설정",
        f"# 위치: {path}",
        "# 여기 적은 값이 패키지 기본값(gtab/defaults.toml)을 덮어쓴다.",
        "# 우선순위: 기본값 -> 이 파일 -> 작업 hints.toml -> 명령줄 --set 키=값",
        "# 지금 적용되는 값 전체 보기: gtab config show",
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
