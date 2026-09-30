"""Stage DAG runner with content-addressed caching (DESIGN §2 "단계 의존", §9.3, M0_SPEC §7).

A stage's key hashes its name, code version, params, model hashes, the keys of its direct deps
(which transitively cover everything upstream) and the job's input hashes. A stage whose key
already has a complete manifest is skipped; otherwise it runs inside JobStore.stage_writer, so a
crash or exception never leaves a half-written result behind.

Stages run strictly one after another in topological order. That matches the "one model on the
GPU at a time" rule (DESIGN §9.5); CPU parallelism can come later without changing the keys.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import gtab
from gtab.jobs.store import JobStore

log = logging.getLogger(__name__)


@dataclass
class Stage:
    name: str
    deps: list[str]
    run: Callable[[StageContext], None]  # writes outputs into ctx.out_dir
    code_version: str = "0"
    params: dict = field(default_factory=dict)
    device: Literal["cpu", "gpu"] = "cpu"
    # model name -> sha256 of the weights actually loaded (DESIGN §9.3 includes these in the key)
    models: dict[str, str] = field(default_factory=dict)


class CycleError(ValueError):
    """The stage graph has a cycle. ``cycle`` is the closed path, e.g. ["a", "b", "a"]."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__("stage dependency cycle: " + " -> ".join(cycle))


@dataclass
class StageContext:
    song_key: str
    stage: str
    key: str
    out_dir: Path  # temp dir; becomes the final stage dir on success
    dep_dirs: dict[str, Path]  # dep name -> its final (read-only) stage dir
    config: Any
    store: JobStore
    params: dict = field(default_factory=dict)


def _by_name(stages: Iterable[Stage]) -> dict[str, Stage]:
    out: dict[str, Stage] = {}
    for s in stages:
        if s.name in out:
            raise ValueError(f"duplicate stage name {s.name!r}")
        out[s.name] = s
    return out


def topo_order(stages: list[Stage]) -> list[str]:
    """Dependencies-first order, deterministic for a given stage list (DFS in list order).

    Raises KeyError for a dep that names no stage and CycleError (with the path) for a cycle,
    both before anything runs."""
    graph = _by_name(stages)
    for s in stages:
        for d in s.deps:
            if d not in graph:
                raise KeyError(f"stage {s.name!r} depends on unknown stage {d!r}")

    done: set[str] = set()
    order: list[str] = []
    for root in graph:
        if root in done:
            continue
        # Iterative DFS; `path` mirrors the grey (in-progress) nodes so a back edge yields the cycle.
        path: list[str] = [root]
        on_path: set[str] = {root}
        iters = [iter(graph[root].deps)]
        while iters:
            nxt = next(iters[-1], None)
            if nxt is None:
                node = path.pop()
                on_path.discard(node)
                iters.pop()
                done.add(node)
                order.append(node)
            elif nxt in on_path:
                raise CycleError(path[path.index(nxt):] + [nxt])
            elif nxt not in done:
                path.append(nxt)
                on_path.add(nxt)
                iters.append(iter(graph[nxt].deps))
    return order


def _ancestors(graph: dict[str, Stage], name: str) -> set[str]:
    seen: set[str] = set()
    todo = [name]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(graph[n].deps)
    return seen


def run_dag(
    stages: list[Stage],
    song_key: str,
    store: JobStore,
    config: Any,
    until: str | None = None,
    force: set[str] | frozenset[str] = frozenset(),
    input_hashes: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Run (or reuse) every stage needed, returning {stage: final_dir}.

    until: stop after this stage (only it and its ancestors are considered).
    force: rerun these stages even if cached, and every selected stage downstream of them:
      keys are derived from recipes, not output bytes, so a regenerated upstream output could
      otherwise sit under an old downstream result without any key changing. (A plain cache
      miss does not cascade: same recipe, same key, downstream results stay valid.)
    """
    graph = _by_name(stages)
    order = topo_order(stages)  # validates the whole spec up front (DESIGN §9.7)
    inputs = dict(input_hashes or {})

    for name in force:
        if name not in graph:
            raise KeyError(f"force: unknown stage {name!r}")
    if until is not None:
        if until not in graph:
            raise KeyError(f"until: unknown stage {until!r}")
        wanted = _ancestors(graph, until)
        order = [n for n in order if n in wanted]
    for name in order:
        clash = set(graph[name].deps) & inputs.keys()
        if clash:
            raise ValueError(f"input_hashes names {sorted(clash)} collide with deps of stage {name!r}")

    keys: dict[str, str] = {}
    results: dict[str, Path] = {}
    forced_set: set[str] = set()  # explicitly forced stages plus everything downstream of them
    for name in order:
        st = graph[name]
        dep_keys = {d: keys[d] for d in st.deps}
        key = JobStore.stage_key(name, st.code_version, st.params, dep_keys | inputs, st.models)
        keys[name] = key
        final = store.stage_dir(song_key, name, key)
        results[name] = final

        if name in force or any(d in forced_set for d in st.deps):
            forced_set.add(name)
        complete = store.is_complete(song_key, name, key)
        if complete and name not in forced_set:
            log.info("stage %s: cached (%s)", name, final.name)
            continue

        log.info("stage %s: running%s -> %s", name, " (forced)" if complete else "", final.name)
        meta = {
            "code_version": st.code_version,
            "params": st.params,
            "device": st.device,
            "models": st.models,
            "deps": dep_keys,
            "inputs": inputs,
            "gtab_version": gtab.__version__,
        }
        try:
            with store.stage_writer(song_key, name, key, meta, replace=complete) as out_dir:
                ctx = StageContext(
                    song_key=song_key,
                    stage=name,
                    key=key,
                    out_dir=out_dir,
                    dep_dirs={d: results[d] for d in st.deps},
                    config=config,
                    store=store,
                    params=copy.deepcopy(st.params),  # a stage mutating it must not skew its manifest
                )
                st.run(ctx)
        except BaseException as e:
            e.add_note(f"gtab: stage {name!r} (key {key[:12]}) of {song_key} failed; nothing was published")
            raise
    return results
