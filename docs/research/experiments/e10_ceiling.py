"""M8 feasibility (before any E10 run): the ceiling of the repetition-subtraction lead view on the Tier B B2
remixes whose two takes are one lead (single-note) and one rhythm (chords).

Oracle, generous on purpose: every window takes the R0 path; groups with 2 occurrences count (a min background
instead of the median); in every bar where the lead plays in a minority of the group's occurrences (1 of 2 for
pairs), the oracle puts every true lead note into the best lead Line, takes every other note out of it, and adds
every lead reference note the mix transcription missed there. Elsewhere the M7b default assignment stays.
Lead Line F1 = max over Lines (note level; recall over the lead take's reference notes)."""
import bisect
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from bandscribe import audio as AU, config  # noqa: E402
from bandscribe.amt import experiments as AE, instruments as I, stage as S  # noqa: E402
from bandscribe.parts import diarize as DZ, eval_track as ET  # noqa: E402
from bandscribe.parts.stage import parts_params  # noqa: E402

LEAD = {"jetb_tothewolves": ("g1", "1"), "jokers_seaofleaves": ("g5", "5"), "secretariat_overthetop": ("g0", "0"),
        "zeno_signs": ("g6", "6")}
cfg = config.load_config()
params = S.ms_params(cfg)
count = parts_params(cfg)["count"]
G = set(I.GUITAR_CLASSES)


def no_gpu(*a, **k):
    raise RuntimeError("not cached")


def f1(tp, n_sys, n_ref):
    p = tp / n_sys if n_sys else 0.0
    r = tp / n_ref if n_ref else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


rows = []
for it in ET.load_items(list(LEAD)):
    if it["scenario"] != "B2":
        continue
    lead, take = LEAD[it["song"]]
    work = ET.cache_dir() / "work" / it["song"]
    tw = work / f"take{take}_mono.wav"
    raws, _ = S.transcribe_views_ms([{"id": "t", "wav": tw, "pcm_sha256": AU.pcm_sha256(tw),
                                      "instruments": I.mask_guitar_only()}], params, cfg, work_dir=work / "_m8",
                                    cache=AE._eval_cache(), force_rung=AE.ms_rung(params), run_gpu_stage=no_gpu)
    ref = [n for n in raws["t"].get("notes") or [] if n.get("instrument") in G]
    bars = it["grid"]["bars"]
    starts = [float(b["start_s"]) for b in bars]
    barno = [int(b["bar"]) for b in bars]

    def bar_of(t):
        return barno[max(0, bisect.bisect_right(starts, float(t)) - 1)]

    is_lead = [lead in t for t in it["truth"]]
    lead_bar = {}
    for n, L in zip(it["sys"], is_lead):
        if L:
            lead_bar[bar_of(n["onset_s"])] = lead_bar.get(bar_of(n["onset_s"]), 0) + 1
    minority = set()
    for g in it["repeat"]["groups"]:
        occs = g["occurrences"]
        if len(occs) < 2:
            continue
        for k in range(min(int(o["n_bars"]) for o in occs)):
            bs = [int(o["start_bar"]) + k for o in occs]
            on = sum(1 for b in bs if lead_bar.get(b, 0) > 0)
            if 0 < on and (on * 2 < len(occs) or (len(occs) == 2 and on == 1)):
                minority.update(b for b in bs if lead_bar.get(b, 0) > 0)
    d = DZ.diarize(it["sys"], it["sigs"], it["windows"], params=count)
    line = [x["line"] for x in d["note"]]
    n_ref = len(ref)
    best = None
    for l in sorted({x for x in line if x is not None}):
        tp = sum(1 for x, L in zip(line, is_lead) if x == l and L)
        n = sum(1 for x in line if x == l)
        f = f1(tp, n, n_ref)
        if best is None or f > best[0]:
            best = (f, l)
    base, lstar = best if best else (0.0, None)
    inM = [bar_of(n["onset_s"]) in minority for n in it["sys"]]
    tp_o = sum(1 for x, L, m in zip(line, is_lead, inM) if (L if m else (x == lstar and L)))
    n_o = sum(1 for x, L, m in zip(line, is_lead, inM) if (L if m else x == lstar))
    ref_M = sum(1 for n in ref if bar_of(n["onset_s"]) in minority)
    sys_M = sum(1 for L, m in zip(is_lead, inM) if L and m)
    add = max(0, ref_M - sys_M)
    oracle = max(f1(tp_o + add, n_o + add, n_ref), f1(sys_M + add, sys_M + add, n_ref))
    rows.append((it["song"], it["split"], d["count"]["chosen"], base, oracle, ref_M, n_ref))
    print(f"{it['song'][:22]:22s} {it['split']:4s} K={d['count']['chosen']} lead ref {n_ref:4d}, in minority bars"
          f" {ref_M:3d} | lead Line F1 M7b {base:.3f} -> oracle ceiling {oracle:.3f} (+{oracle - base:.3f})")
print("mean gain ceiling", round(sum(r[4] - r[3] for r in rows) / len(rows), 3))
