"""Stage graph names/deps/order (M1_M2_SPEC 3.2; GRID)."""
from __future__ import annotations

import pytest

from gtab.jobs.dag import topo_order
from gtab.pipeline import graph

# M1_M2_SPEC 3.2 table (+ vocal -> sep: vocal reads the sep stage's vocals stem, see gtab.pipeline.graph)
EXPECTED = {
    "beats": set(),
    "grid": {"beats"},
    "sections": {"grid"},
    "sep": set(),
    "stems": {"sep"},
    "vocal": {"stems", "grid", "sep"},
    "resid1": {"stems", "grid", "sections"},
    "amt_ms1": {"stems"},
    "amt_bp": {"stems"},
    "instr": {"amt_ms1", "stems", "sections", "grid"},
    "amt_gtr": {"stems", "instr"},
    "s35": {"instr", "vocal", "resid1", "amt_ms1", "amt_gtr", "amt_bp"},
    "notes": {"amt_gtr", "amt_ms1", "amt_bp", "grid", "s35"},
}
GPU = {"beats", "sep", "amt_ms1", "amt_gtr"}


def fake_impls(monkeypatch, missing_module: str | None = None):
    def provider(module):
        if module == missing_module:
            return {}
        return {n: {"run": lambda ctx: None, "code_version": "1", "params": lambda cfg: {},
                    "device": "gpu" if n in GPU else "cpu", "models": lambda cfg: {}}
                for n in graph.NAMES if graph.PROVIDER[n] == module}

    monkeypatch.setattr(graph, "_provider", provider)


def test_names_and_deps_match_spec():
    assert list(graph.NAMES) == list(EXPECTED)
    assert {n: set(d) for n, d in graph.DEPS.items()} == EXPECTED
    assert graph.PROVIDER["beats"] == "gtab.analysis.stage"
    assert graph.PROVIDER["sep"] == "gtab.sep.stage"
    assert graph.PROVIDER["notes"] == "gtab.amt.stage"


def test_acyclic_and_gpu_order(monkeypatch):
    fake_impls(monkeypatch)
    stages = graph.build_stages(None, "notes")
    order = topo_order(stages)  # raises CycleError on a cycle
    assert set(order) == set(EXPECTED)
    gpu = [n for n in order if n in GPU]
    assert gpu == ["beats", "sep", "amt_ms1", "amt_gtr"]
    for st in stages:
        for d in st.deps:
            assert order.index(d) < order.index(st.name)


def test_until_notes_includes_s35_parts():
    sel = graph.selected("notes")
    assert "vocal" in sel and "resid1" in sel and "s35" in sel
    assert graph.selected("grid") == ["beats", "grid"]
    assert graph.selected("sections") == ["beats", "grid", "sections"]
    assert graph.descendants({"sections"}) == {"sections", "resid1", "instr", "amt_gtr", "s35", "notes"}


def test_missing_stage_module_korean_error(monkeypatch):
    fake_impls(monkeypatch, missing_module="gtab.sep.stage")
    with pytest.raises(graph.StageNotImplemented) as ei:
        graph.build_stages(None, "notes")
    msg = str(ei.value)
    assert "단계 'sep' 가 아직 구현되지 않았습니다 (gtab.sep.stage)" in msg
    assert "단계 'stems' 가 아직 구현되지 않았습니다" in msg
    # stages that do not need the missing provider still build
    assert [s.name for s in graph.build_stages(None, "sections")] == ["beats", "grid", "sections"]


def test_unknown_stage_name():
    with pytest.raises(graph.UnknownStage, match="알 수 없는"):
        graph.selected("tabs")


def test_grid_stages_are_registered():
    from gtab.analysis import stage

    assert set(stage.STAGES) == {"beats", "grid", "sections", "vocal", "resid1"}
    assert stage.STAGES["beats"]["device"] == "gpu"
    assert all(stage.STAGES[n]["device"] == "cpu" for n in ("grid", "sections", "vocal", "resid1"))
    assert all(stage.STAGES[n]["models"](None) == {} for n in ("grid", "sections", "vocal", "resid1"))
    assert stage.beats_params(None) == {"checkpoint": "final0", "dbn": False, "float16": False}
    assert stage.beats_model_name(None) == "beat_this.final0"
    with pytest.raises(ValueError, match="madmom"):  # refused by the config itself
        stage.beats_params({"grid": {"dbn": True}})
    assert stage.grid_params({"grid": {"refine_window_ms": 30.0}})["refine_window_ms"] == 30.0
    assert stage.sections_params(None) == {"min_section_bars": 4, "max_sections": 16, "repeat_min_score": 0.6,
                                           "align_max_offset_ms": 50.0}
    assert stage.vocal_params(None) == {"vocal_active_rel_db": -30.0}
    assert stage.resid1_params(None) == {"resid_min_occurrences": 3}


def test_grid_params_come_from_the_validated_config():
    """No private copy of the defaults: params equal the Config sections, and a misspelled key fails."""
    from gtab.analysis import grid as gridmod
    from gtab.analysis import sections as secmod
    from gtab.analysis import stage
    from gtab.config import Config

    cfg = Config()
    assert stage.grid_params(cfg) == cfg.grid.model_dump(mode="json")
    assert stage.sections_params(cfg) == cfg.sections.model_dump(mode="json")
    # the analysis modules' fallbacks (used by direct callers) match the configured defaults
    assert {k: stage.grid_params(cfg)[k] for k in gridmod.DEFAULT_PARAMS} == gridmod.DEFAULT_PARAMS
    assert stage.sections_params(cfg) == secmod.DEFAULT_PARAMS
    with pytest.raises(ValueError):
        stage.grid_params({"grid": {"refine_windw_ms": 30.0}})
    with pytest.raises(TypeError):
        stage.grid_params(42)


def test_fast_profile_drops_mix_view():
    amt = pytest.importorskip("gtab.amt.stage")
    impl = getattr(amt, "STAGES", {}).get("amt_ms1")
    if impl is None:
        pytest.skip("amt_ms1 not implemented yet")
    quality = impl["params"]({"run": {"profile": "quality"}})
    fast = impl["params"]({"run": {"profile": "fast"}})
    assert "mix_mono" in quality["views"] and "mix_mono" not in fast["views"]
