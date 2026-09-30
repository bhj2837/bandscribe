from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from gtab import config, paths
from gtab.config import Config, ConfigError, load_config

# The exact text the user was shown and agreed to (M0_SPEC 3). Duplicated here on purpose: if someone
# rewords the constant in config.py, this test must fail.
SPEC_CONSENT = (
    "2026-09-30 사용자 동의: YouTube 약관(다운로드 금지)과 JS 챌린지 해결의 기술적 보호조치 우회 위험"
    "(해외 판례, 한국법 미검증)을 고지받고 기본 켜기를 요청함. 개인 비공개 사용."
)


@pytest.fixture
def no_user(tmp_path: Path) -> Path:
    return tmp_path / "missing-config.toml"


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


# --------------------------------------------------------------------------------------- defaults


def test_defaults_match_spec(no_user: Path) -> None:
    cfg = load_config(user_config=no_user)
    assert cfg.youtube.enabled is True  # user request 2026-09-30: on by default
    assert cfg.youtube.format == "ba[format_id!*=drc][acodec=opus]/ba[format_id!*=drc]/ba"
    assert cfg.youtube.cookies is False and cfg.youtube.po_token is False
    assert cfg.youtube.timeout_s == 600
    assert cfg.ingest.sample_rate == 44100
    assert cfg.ingest.true_peak_ceiling_dbfs == -1.0
    assert cfg.ingest.quasi_mono_side_mid_db == -30.0
    assert cfg.ingest.lowpass_detect_hz == 16000
    assert cfg.gpu.lock_timeout_s == 7200 and cfg.gpu.worker_timeout_s == 7200
    assert cfg.gpu.vram_margin_gb == 0.7 and cfg.gpu.target_reserved_gb == 4.5
    assert cfg.gpu.insufficient_vram_policy == "wait"
    assert cfg.doctor.min_free_disk_gb == 60 and cfg.doctor.min_node_major == 22
    assert cfg.consent.youtube is None
    assert cfg.extra == {}


def test_model_defaults_do_not_drift_from_defaults_toml(no_user: Path) -> None:
    raw = tomllib.loads(config.DEFAULTS_PATH.read_text(encoding="utf-8"))
    assert Config().model_dump() == load_config(user_config=no_user).model_dump()
    # every key in defaults.toml is a modelled field (nothing lands in extra by accident)
    assert set(raw) <= set(config.KNOWN_SECTIONS)
    for section, values in raw.items():
        assert set(values) <= set(type(getattr(Config(), section)).model_fields), section


# --------------------------------------------------------------------------------------- layering


def test_layer_order_defaults_user_hints_set(tmp_path: Path, no_user: Path) -> None:
    user = write(
        tmp_path / "config.toml",
        "[youtube]\nenabled = false\n[ingest]\nquasi_mono_side_mid_db = -25.0\nlowpass_detect_hz = 15000\n",
    )
    hints = write(tmp_path / "hints.toml", "[ingest]\nquasi_mono_side_mid_db = -20.0\n")

    assert load_config(user_config=no_user).ingest.quasi_mono_side_mid_db == -30.0
    c1 = load_config(user_config=user)
    assert (c1.youtube.enabled, c1.ingest.quasi_mono_side_mid_db) == (False, -25.0)
    c2 = load_config(hints, user_config=user)
    assert c2.ingest.quasi_mono_side_mid_db == -20.0
    assert c2.ingest.lowpass_detect_hz == 15000  # user value survives: tables merge, not replace
    assert c2.youtube.format == Config().youtube.format  # untouched default survives
    c3 = load_config(hints, ["ingest.quasi_mono_side_mid_db=-35.0", "youtube.enabled=true"], user_config=user)
    assert (c3.ingest.quasi_mono_side_mid_db, c3.youtube.enabled) == (-35.0, True)


def test_default_user_config_path_is_looked_up_at_call_time(tmp_path: Path, monkeypatch) -> None:
    user = write(tmp_path / "config.toml", "[doctor]\nmin_node_major = 24\n")
    monkeypatch.setattr(paths, "USER_CONFIG", user)
    assert load_config().doctor.min_node_major == 24


def test_missing_hints_file_is_skipped(tmp_path: Path, no_user: Path) -> None:
    assert load_config(tmp_path / "nope.toml", user_config=no_user) == Config()


def test_later_set_wins(no_user: Path) -> None:
    cfg = load_config(overrides=["gpu.vram_margin_gb=1", "gpu.vram_margin_gb=0.5"], user_config=no_user)
    assert cfg.gpu.vram_margin_gb == 0.5


# ------------------------------------------------------------------------------------------ --set


@pytest.mark.parametrize(
    ("item", "path", "value"),
    [
        ("youtube.enabled=false", ["youtube", "enabled"], False),
        ("ingest.sample_rate=48000", ["ingest", "sample_rate"], 48000),
        ("gpu.vram_margin_gb=0.5", ["gpu", "vram_margin_gb"], 0.5),
        ('a.b="quoted string"', ["a", "b"], "quoted string"),
        ("a.b=plain words", ["a", "b"], "plain words"),  # not TOML -> raw string
        ("youtube.format=ba[acodec=opus]/ba", ["youtube", "format"], "ba[acodec=opus]/ba"),
        ("a.b=[1, 2]", ["a", "b"], [1, 2]),
        ("a.b=2026-09-30", ["a", "b"], "2026-09-30"),  # TOML date kept as text
        ("a.b=", ["a", "b"], ""),
        (" a . b = 1 ", ["a", "b"], 1),
        ("x=1=2", ["x"], "1=2"),  # only the first '=' splits
    ],
)
def test_parse_override(item: str, path: list[str], value: object) -> None:
    assert config.parse_override(item) == (path, value)


@pytest.mark.parametrize("item", ["youtube.enabled", "=1", "a..b=1", ".a=1", "a.=1", ""])
def test_parse_override_rejects_bad_syntax(item: str) -> None:
    with pytest.raises(ConfigError, match="--set"):
        config.parse_override(item)


def test_newline_in_value_cannot_inject_keys() -> None:
    assert config.parse_value("1\nother = 2") == "1\nother = 2"


def test_set_coerces_and_validates(no_user: Path) -> None:
    cfg = load_config(overrides=["youtube.timeout_s=120"], user_config=no_user)
    assert cfg.youtube.timeout_s == 120
    with pytest.raises(ConfigError, match="youtube.enabled"):
        load_config(overrides=["youtube.enabled=maybe"], user_config=no_user)
    with pytest.raises(ConfigError, match="gpu.insufficient_vram_policy"):
        load_config(overrides=["gpu.insufficient_vram_policy=explode"], user_config=no_user)


def test_unknown_key_in_known_section_fails(no_user: Path) -> None:
    with pytest.raises(ConfigError, match="youtube.enabeld: 알 수 없는 키"):
        load_config(overrides=["youtube.enabeld=false"], user_config=no_user)


def test_unknown_section_via_set_is_an_error_with_suggestion(no_user: Path) -> None:
    """Regression: `--set youtub.enabled=false` used to warn and then download anyway."""
    with pytest.raises(ConfigError, match=r"'youtub'.*'youtube'을\(를\) 뜻했나요") as ei:
        load_config(overrides=["youtub.enabled=false"], user_config=no_user)
    assert "youtube, ingest" in str(ei.value)  # lists what can be set
    with pytest.raises(ConfigError, match="알 수 없는 섹션 'separator'"):
        load_config(overrides=["separator.backend=sw"], user_config=no_user)


def test_sample_rate_is_fixed_at_44100(tmp_path: Path, no_user: Path) -> None:
    """mix_44k_f32.wav and the f-<PCM sha> key assume a 44.1 kHz decode (review: overrides were ignored)."""
    assert load_config(overrides=["ingest.sample_rate=44100"], user_config=no_user).ingest.sample_rate == 44100
    with pytest.raises(ConfigError, match="ingest.sample_rate: 44100 으로 고정입니다"):
        load_config(overrides=["ingest.sample_rate=48000"], user_config=no_user)
    user = write(tmp_path / "config.toml", "[ingest]\nsample_rate = 48000\n")
    with pytest.raises(ConfigError, match="44100"):
        load_config(user_config=user)


# ------------------------------------------------------------------------------- unknown sections


def test_unknown_top_level_sections_kept_in_extra(tmp_path: Path) -> None:
    user = write(tmp_path / "config.toml", '[separator]\nbackend = "sw_msst"\n[separator.sw]\nchunk = 352256\n')
    cfg = load_config(user_config=user)
    assert cfg.extra["separator"] == {"backend": "sw_msst", "sw": {"chunk": 352256}}
    assert cfg.get("separator.backend") == "sw_msst"
    assert cfg.get("separator.sw.chunk") == 352256
    # direct construction splits them the same way
    assert Config.model_validate({"separator": {"backend": "x"}}).extra == {"separator": {"backend": "x"}}


def test_get() -> None:
    cfg = Config()
    assert cfg.get("youtube.enabled") is True
    assert cfg.get("gpu") is cfg.gpu
    assert cfg.get("consent.youtube") is None
    assert cfg.get("nope.x", 5) == 5
    assert cfg.get("youtube.nope", None) is None
    with pytest.raises(KeyError):
        cfg.get("youtube.nope")
    with pytest.raises(KeyError):
        cfg.get("youtube.enabled.deeper")


# ----------------------------------------------------------------------------------- file errors


def test_user_config_with_bom_is_read(tmp_path: Path) -> None:
    user = write(tmp_path / "config.toml", "[youtube]\nenabled = false\n", encoding="utf-8-sig")
    assert load_config(user_config=user).youtube.enabled is False


def test_non_utf8_user_config_is_a_clear_error(tmp_path: Path) -> None:
    user = write(tmp_path / "config.toml", '[consent]\nyoutube = "동의"\n', encoding="cp949")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(user_config=user)


def test_bad_toml_is_a_clear_error(tmp_path: Path) -> None:
    user = write(tmp_path / "config.toml", "[youtube\nenabled = false\n")
    with pytest.raises(ConfigError, match="TOML"):
        load_config(user_config=user)


# ----------------------------------------------------------------------------------------- consent


def test_cookies_and_po_token_need_their_own_consent(tmp_path: Path, no_user: Path) -> None:
    for opt in ("cookies", "po_token"):
        with pytest.raises(ConfigError, match=f"youtube.{opt}"):
            load_config(overrides=[f"youtube.{opt}=true"], user_config=no_user)
    user = write(tmp_path / "config.toml", '[consent]\ncookies = "2026-10-01 동의"\n')
    assert load_config(overrides=["youtube.cookies=true"], user_config=user).youtube.cookies is True


# ------------------------------------------------------------------------------------ user config


def test_consent_constant_is_the_spec_text() -> None:
    assert config.YOUTUBE_CONSENT_TEXT == SPEC_CONSENT


def test_ensure_user_config_creates_consent_file(tmp_path: Path) -> None:
    target = tmp_path / "data" / "config.toml"
    assert config.ensure_user_config(target) == target
    text = target.read_text(encoding="utf-8")
    assert text.startswith("#")  # commented header
    assert tomllib.loads(text) == {"consent": {"youtube": SPEC_CONSENT}}
    cfg = load_config(user_config=target)
    assert cfg.consent.youtube == SPEC_CONSENT
    assert cfg.youtube.enabled is True


def test_ensure_user_config_never_overwrites(tmp_path: Path) -> None:
    target = write(tmp_path / "config.toml", "[youtube]\nenabled = false\n")
    config.ensure_user_config(target)
    assert target.read_text(encoding="utf-8") == "[youtube]\nenabled = false\n"


def test_ensure_user_config_default_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "USER_CONFIG", tmp_path / "d" / "config.toml")
    assert config.ensure_user_config() == tmp_path / "d" / "config.toml"
    assert (tmp_path / "d" / "config.toml").is_file()


# ---------------------------------------------------------------------------------------- dumping


def test_dump_effective_round_trips(tmp_path: Path) -> None:
    user = write(
        tmp_path / "config.toml",
        f'[consent]\nyoutube = "{SPEC_CONSENT}"\n[separator]\nbackend = "sw_msst"\n',
    )
    cfg = load_config(overrides=["ingest.lowpass_detect_hz=15000"], user_config=user)
    text = config.dump_effective(cfg)
    parsed = tomllib.loads(text)
    assert parsed["separator"] == {"backend": "sw_msst"}  # unknown section back at top level
    assert "extra" not in parsed
    assert Config.model_validate(parsed) == cfg


def test_dump_effective_omits_empty_sections() -> None:
    parsed = tomllib.loads(config.dump_effective(Config()))
    assert "consent" not in parsed  # all-None section
    assert parsed["youtube"]["enabled"] is True
