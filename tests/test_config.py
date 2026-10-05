from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from bandscribe import config, paths
from bandscribe.config import Config, ConfigError, load_config

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


# ------------------------------------------------------------------------------- M1a/M2 sections


def test_m2_defaults_match_spec(no_user: Path) -> None:
    """M1_M2_SPEC 1.3 values (the stage keys of M2 depend on them)."""
    cfg = load_config(user_config=no_user)
    assert cfg.run.profile == "quality"
    assert (cfg.hints.instruments, cfg.hints.parts) == ("auto", "auto")
    s = cfg.sep
    assert (s.backend, s.model_dir, s.ckpt, s.config) == (
        "sw_msst", "data/models/bs_roformer_sw", "BS-Rofo-SW-Fixed.ckpt", "BS-Rofo-SW-Fixed.yaml")
    assert s.ckpt_sha256 == "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e"
    assert s.msst_commit == "84b1eac0887756b4f1a9d7a1ff49105939749ed2"
    assert s.chunk_ladder == [588800, 352256, 262144]
    assert (s.num_overlap, s.batch_size, s.dtype, s.input_lufs, s.leftover_warn_db) == (2, 1, "upstream", "off", -20.0)
    a = cfg.amt
    assert (a.muscriptor_model, a.muscriptor_fallback, a.muscriptor_loader, a.dtype) == (
        "medium", ["small"], "lean", "upstream")  # lean: AMT parity gate passed (integration request)
    assert a.muscriptor_revision == "f32236969308476e01fd3aae67357de5feb05a2d"
    assert (a.prelude_forcing, a.beam_size, a.cfg_coef, a.batch_size, a.latency_s) == (True, 1, 1.0, 1, -0.003)  # latency exp
    assert (a.piano_other_instruments, a.guitar_view, a.guitar_mask) == ("keys+guitar", "guitar_mono", "guitar+present")
    assert (a.presence_pass, a.presence_window_s, a.presence_max_windows, a.presence_max_total_s,
            a.presence_max_fraction, a.presence_min_active) == ("auto", 10.0, 6, 60.0, 0.2, 0.5)
    assert (a.bp_onset_threshold, a.bp_frame_threshold, a.bp_min_note_frames) == (0.6, 0.4, 7)  # E23
    assert (a.bp_min_freq_hz, a.bp_max_freq_hz, a.bp_melodia_trick, a.bp_threads) == (0.0, 0.0, True, 4)
    i = cfg.instr
    assert (i.threshold, i.guitar_threshold, i.guitar_stem_active_ratio, i.active_rel_db, i.mix_class_min_notes) == (
        0.5, 0.15, 0.05, -40.0, 8)
    assert i.presence_min_notes == 8
    g = cfg.grid
    assert (g.checkpoint, g.dbn, g.refine_window_ms, g.half_double_check) == ("final0", False, 35.0, True)
    assert (g.compound_triple_ratio, g.compound_margin, g.downbeat_snap_ms, g.meter_smooth_bars) == (0.6, 0.1, 70.0, 8)
    sc = cfg.sections
    assert (sc.min_section_bars, sc.max_sections, sc.repeat_min_score, sc.align_max_offset_ms) == (4, 16, 0.6, 50.0)
    assert (cfg.s35.vocal_active_rel_db, cfg.s35.resid_min_occurrences) == (-30.0, 3)
    e = cfg.eval
    assert (e.tpb, e.onset_tol_s, e.bootstrap_iterations, e.seed, e.mde_points) == (48, 0.05, 10000, 20260930, 1.0)
    assert cfg.datasets.root == "data/datasets"
    gp = cfg.gpu
    assert (gp.wait_poll_s, gp.wait_max_s, gp.max_worker_retries, gp.worker_headroom_mb) == (30, 3600, 3, 256)
    assert (gp.ctx_mb_default, gp.shared_growth_fail_mb, gp.rtf_fail_ratio) == (150, 64, 0.5)
    assert gp.cpu_backends == ["beats_beatthis"]
    assert cfg.get("sep.chunk_ladder") == [588800, 352256, 262144]  # owners read through Config.get too


def test_grid_dbn_true_is_rejected(no_user: Path) -> None:
    with pytest.raises(ConfigError, match="grid.dbn=true 는 madmom 이 필요해 지원하지 않습니다"):
        load_config(overrides=["grid.dbn=true"], user_config=no_user)


@pytest.mark.parametrize("key", ["sep.dtype", "amt.dtype"])
def test_dtype_accepts_only_upstream_or_fp16(no_user: Path, key: str) -> None:
    assert load_config(overrides=[f"{key}=fp16"], user_config=no_user).get(key) == "fp16"
    for bad in ("bf16", "bfloat16", "fp32"):
        with pytest.raises(ConfigError, match=key):
            load_config(overrides=[f"{key}={bad}"], user_config=no_user)


def test_sep_input_lufs(no_user: Path) -> None:
    assert load_config(overrides=["sep.input_lufs=-14"], user_config=no_user).sep.input_lufs == -14.0
    assert load_config(overrides=["sep.input_lufs=-16.0"], user_config=no_user).sep.input_lufs == -16.0
    assert load_config(overrides=['sep.input_lufs="off"'], user_config=no_user).sep.input_lufs == "off"
    for bad in ("-40", "-5", "loud", "true"):
        with pytest.raises(ConfigError, match="sep.input_lufs"):
            load_config(overrides=[f"sep.input_lufs={bad}"], user_config=no_user)


def test_bp_min_note_frames(no_user: Path) -> None:
    assert load_config(overrides=["amt.bp_min_note_frames=5"], user_config=no_user).amt.bp_min_note_frames == 5
    for bad in ("0", "31", "4.5", "true"):
        with pytest.raises(ConfigError, match="amt.bp_min_note_frames"):
            load_config(overrides=[f"amt.bp_min_note_frames={bad}"], user_config=no_user)
    for at_or_above in ("12", "13", "30"):  # >= upstream default (127.7 ms), refused (DESIGN S6; review 2026-10-05)
        with pytest.raises(ConfigError, match="127.7 ms"):
            load_config(overrides=[f"amt.bp_min_note_frames={at_or_above}"], user_config=no_user)
    assert load_config(overrides=["amt.bp_min_note_frames=11"], user_config=no_user).amt.bp_min_note_frames == 11


def test_chunk_ladder_validation(no_user: Path) -> None:
    assert load_config(overrides=["sep.chunk_ladder=[352256]"], user_config=no_user).sep.chunk_ladder == [352256]
    for bad, match in (("[]", "비어"), ("[588801]", "512"), ("[262144, 588800]", "감소")):
        with pytest.raises(ConfigError, match=match):
            load_config(overrides=[f"sep.chunk_ladder={bad}"], user_config=no_user)


def test_m2_literals_and_unknown_keys(no_user: Path) -> None:
    for item in ("run.profile=max", "amt.guitar_view=bass_mono", "amt.guitar_mask=none",
                 "amt.presence_pass=some", "amt.presence_max_fraction=0", "amt.presence_window_s=1",
                 "amt.piano_other_instruments=guitar", "gpu.cpu_backends=[\"amt_basicpitch\"]",
                 "sep.allow_cpu=true", "amt.muscriptor_model=huge"):
        with pytest.raises(ConfigError):
            load_config(overrides=[item], user_config=no_user)
    cfg = load_config(overrides=["run.profile=fast", "hints.instruments=guitar,keys", "amt.guitar_view=mix_mono"],
                      user_config=no_user)
    assert (cfg.run.profile, cfg.hints.instruments, cfg.amt.guitar_view) == ("fast", "guitar,keys", "mix_mono")
    cfg = load_config(overrides=["run.profile=eval", "amt.presence_pass=full"], user_config=no_user)
    assert (cfg.run.profile, cfg.amt.presence_pass) == ("eval", "full")


def test_m2_sections_are_settable_and_listed(no_user: Path) -> None:
    for sec in ("run", "hints", "sep", "amt", "instr", "grid", "sections", "s35", "eval", "datasets"):
        assert sec in config.KNOWN_SECTIONS
    # known now, so a typo inside them fails loudly instead of landing in Config.extra
    with pytest.raises(ConfigError, match="알 수 없는 키"):
        load_config(overrides=["eval.sede=1"], user_config=no_user)
