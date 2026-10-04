"""Stage DAG: topo order, cycle detection, caching, force, until (M0_SPEC §7, §15; DESIGN §9.7)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from bandscribe.jobs import CycleError, JobStore, Stage, StageContext, run_dag, topo_order

SONG = "f-00000000000000aa"
INPUTS = {"pcm_sha256": "1" * 64}


def _noop(ctx: StageContext) -> None:
    pass


def S(name: str, *deps: str, **kw) -> Stage:
    return Stage(name=name, deps=list(deps), run=kw.pop("run", _noop), **kw)


# ---------------------------------------------------------------------------- topo_order


def test_topo_order_puts_deps_first_and_is_deterministic():
    stages = [S("tab", "notes", "grid"), S("notes", "sep"), S("grid", "input"), S("sep", "input"), S("input")]
    order = topo_order(stages)
    assert sorted(order) == sorted(s.name for s in stages)
    pos = {n: i for i, n in enumerate(order)}
    for s in stages:
        for d in s.deps:
            assert pos[d] < pos[s.name]
    assert topo_order(stages) == order
    assert order == ["input", "sep", "notes", "grid", "tab"]  # DFS in list order


def test_topo_order_unknown_dep_is_keyerror():
    with pytest.raises(KeyError, match="nope"):
        topo_order([S("a", "nope")])


def test_topo_order_duplicate_name():
    with pytest.raises(ValueError, match="duplicate"):
        topo_order([S("a"), S("a")])


def test_cycle_error_carries_the_cycle_path():
    # input -> a -> b -> c -> a ; d hangs off the cycle
    stages = [S("input"), S("d", "c"), S("a", "input", "c"), S("b", "a"), S("c", "b")]
    with pytest.raises(CycleError) as ei:
        topo_order(stages)
    cyc = ei.value.cycle
    assert cyc[0] == cyc[-1] and len(cyc) == 4
    assert set(cyc) == {"a", "b", "c"}
    by = {s.name: s for s in stages}
    for x, y in zip(cyc, cyc[1:]):
        assert y in by[x].deps  # every hop is a real dependency edge
    assert isinstance(ei.value, ValueError)
    assert " -> ".join(cyc) in str(ei.value)


def test_self_loop_is_a_cycle():
    with pytest.raises(CycleError) as ei:
        topo_order([S("a", "a")])
    assert ei.value.cycle == ["a", "a"]


# ---------------------------------------------------------------------------- run_dag


class Pipeline:
    """input -> sep -> notes ; input -> grid ; (notes, grid) -> tab. Counts executions."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.params = {n: {} for n in ("input", "sep", "grid", "notes", "tab")}
        self.versions = {n: "1" for n in self.params}
        self.fail: set[str] = set()
        self.seen: dict[str, StageContext] = {}

    def _run(self, name: str):
        def run(ctx: StageContext) -> None:
            self.calls[name] += 1
            self.seen[name] = ctx
            if name in self.fail:
                (ctx.out_dir / "partial.txt").write_text("half", encoding="utf-8")
                raise RuntimeError(f"{name} exploded")
            # each stage reads its deps' outputs, proving dep_dirs point at published results
            upstream = "".join(
                (ctx.dep_dirs[d] / "out.txt").read_text(encoding="utf-8") for d in sorted(ctx.dep_dirs)
            )
            (ctx.out_dir / "out.txt").write_text(f"{upstream}[{name}{self.calls[name]}]", encoding="utf-8")

        return run

    def stages(self) -> list[Stage]:
        deps = {"input": [], "sep": ["input"], "grid": ["input"], "notes": ["sep"], "tab": ["notes", "grid"]}
        return [
            Stage(n, deps[n], self._run(n), code_version=self.versions[n], params=self.params[n],
                  device="gpu" if n == "sep" else "cpu")
            for n in ("tab", "notes", "grid", "sep", "input")  # deliberately not in dependency order
        ]


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs")


def _manifest(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text(encoding="utf-8"))


def _run(p: Pipeline, store: JobStore, **kw) -> dict[str, Path]:
    return run_dag(p.stages(), SONG, store, config={"cfg": 1}, input_hashes=INPUTS, **kw)


def test_first_run_executes_all_and_publishes(store: JobStore):
    p = Pipeline()
    res = _run(p, store)
    assert set(res) == {"input", "sep", "grid", "notes", "tab"}
    assert p.calls == Counter(input=1, sep=1, grid=1, notes=1, tab=1)
    for name, d in res.items():
        m = _manifest(d)
        assert m["status"] == "complete" and m["stage"] == name and m["song_key"] == SONG
        assert store.is_complete(SONG, name, m["key"]) and d == store.stage_dir(SONG, name, m["key"])
    assert list(store.root.rglob(".tmp-*")) == []


def test_outputs_flow_through_dep_dirs(store: JobStore):
    p = Pipeline()
    res = _run(p, store)
    assert (res["notes"] / "out.txt").read_text(encoding="utf-8") == "[input1][sep1][notes1]"
    assert (res["tab"] / "out.txt").read_text(encoding="utf-8") == "[input1][grid1][input1][sep1][notes1][tab1]"
    ctx = p.seen["tab"]
    assert ctx.song_key == SONG and ctx.stage == "tab" and ctx.config == {"cfg": 1} and ctx.store is store
    assert ctx.dep_dirs == {"notes": res["notes"], "grid": res["grid"]}
    assert ctx.out_dir.name.startswith(".tmp-")  # stages write into the temp dir, not the final one
    assert res["tab"].name == ctx.key[:12]


def test_manifest_records_recipe(store: JobStore):
    p = Pipeline()
    p.params["sep"] = {"model": "bs_roformer", "overlap": 4}
    res = _run(p, store)
    m = _manifest(res["sep"])
    assert m["params"] == {"model": "bs_roformer", "overlap": 4}
    assert m["device"] == "gpu" and m["code_version"] == "1"
    assert m["deps"] == {"input": _manifest(res["input"])["key"]}
    assert m["inputs"] == INPUTS
    assert "bandscribe_version" in m


def test_rerun_is_fully_cached(store: JobStore):
    p = Pipeline()
    first = _run(p, store)
    second = _run(p, store)
    assert first == second
    assert all(v == 1 for v in p.calls.values())


def test_param_change_reruns_stage_and_downstream_only(store: JobStore):
    p = Pipeline()
    first = _run(p, store)
    p.params["sep"] = {"model": "other"}
    second = _run(p, store)
    assert p.calls == Counter(input=1, grid=1, sep=2, notes=2, tab=2)
    assert second["input"] == first["input"] and second["grid"] == first["grid"]
    for n in ("sep", "notes", "tab"):
        assert second[n] != first[n] and first[n].is_dir()  # variants sit side by side (§9.3)
    # switching back is a pure cache hit
    p.params["sep"] = {}
    assert _run(p, store) == first
    assert p.calls == Counter(input=1, grid=1, sep=2, notes=2, tab=2)


def test_code_version_and_input_hash_changes(store: JobStore):
    p = Pipeline()
    _run(p, store)
    p.versions["grid"] = "2"
    _run(p, store)
    assert p.calls == Counter(input=1, sep=1, notes=1, grid=2, tab=2)
    run_dag(p.stages(), SONG, store, config=None, input_hashes={"pcm_sha256": "2" * 64})
    assert all(p.calls[n] >= 2 for n in p.calls) and p.calls["input"] == 2


def test_force_reruns_stage_and_downstream(store: JobStore):
    p = Pipeline()
    first = _run(p, store)
    second = _run(p, store, force={"sep"})
    assert p.calls == Counter(input=1, grid=1, sep=2, notes=2, tab=2)
    assert second == first  # same keys -> same dirs, contents replaced in place
    assert (second["sep"] / "out.txt").read_text(encoding="utf-8") == "[input1][sep2]"
    assert (second["tab"] / "out.txt").read_text(encoding="utf-8").endswith("[notes2][tab2]")
    assert list(store.root.rglob(".tmp-*")) == []


def test_force_leaf_only(store: JobStore):
    p = Pipeline()
    _run(p, store)
    _run(p, store, force={"tab"})
    assert p.calls == Counter(input=1, sep=1, grid=1, notes=1, tab=2)


def test_until_runs_only_ancestors(store: JobStore):
    p = Pipeline()
    res = _run(p, store, until="notes")
    assert set(res) == {"input", "sep", "notes"}
    assert p.calls == Counter(input=1, sep=1, notes=1)
    full = _run(p, store)
    assert p.calls == Counter(input=1, sep=1, notes=1, grid=1, tab=1)
    assert full["notes"] == res["notes"]


def test_force_outside_until_is_ignored(store: JobStore):
    p = Pipeline()
    _run(p, store)
    _run(p, store, until="grid", force={"tab"})
    assert p.calls["tab"] == 1


def test_unknown_until_or_force_is_keyerror(store: JobStore):
    p = Pipeline()
    with pytest.raises(KeyError):
        _run(p, store, until="nope")
    with pytest.raises(KeyError):
        _run(p, store, force={"nope"})
    assert not p.calls


def test_input_hash_name_clashing_with_dep_is_rejected(store: JobStore):
    p = Pipeline()
    with pytest.raises(ValueError, match="collide"):
        run_dag(p.stages(), SONG, store, config=None, input_hashes={"sep": "x"})
    assert not p.calls


def test_cycle_detected_before_anything_runs(store: JobStore):
    calls: list[str] = []
    rec = lambda ctx: calls.append(ctx.stage)  # noqa: E731
    stages = [S("input", run=rec), S("a", "input", "c", run=rec), S("b", "a", run=rec), S("c", "b", run=rec)]
    with pytest.raises(CycleError):
        run_dag(stages, SONG, store, config=None)
    assert calls == []
    assert not store.root.exists() or list(store.root.rglob("manifest.json")) == []


def test_failure_publishes_nothing_and_resume_uses_cache(store: JobStore):
    p = Pipeline()
    p.fail = {"notes"}
    with pytest.raises(RuntimeError, match="notes exploded") as ei:
        _run(p, store)
    assert any("stage 'notes'" in n for n in getattr(ei.value, "__notes__", []))
    assert p.calls == Counter(input=1, sep=1, notes=1)  # order is input, sep, notes, grid, tab
    notes_dir = store.root / SONG / "stages" / "notes"
    assert not notes_dir.exists() or list(notes_dir.iterdir()) == []  # no final, no temp
    assert list(store.root.rglob(".tmp-*")) == []

    p.fail = set()
    _run(p, store)
    assert p.calls == Counter(input=1, sep=1, notes=2, grid=1, tab=1)  # completed stages reused


def test_models_are_part_of_the_key(store: JobStore):
    calls = Counter()

    def run(ctx):
        calls[ctx.stage] += 1

    a = [Stage("sep", [], run, models={"bs_roformer": "a" * 64})]
    b = [Stage("sep", [], run, models={"bs_roformer": "b" * 64})]
    ra = run_dag(a, SONG, store, config=None)
    rb = run_dag(b, SONG, store, config=None)
    assert ra["sep"] != rb["sep"] and calls["sep"] == 2
    assert _manifest(rb["sep"])["models"] == {"bs_roformer": "b" * 64}
