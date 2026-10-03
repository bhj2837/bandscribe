"""S3.5 v0 instrumentation estimator (M1_M2_SPEC 9.2 item 5, DESIGN S3.5). Hand rules, **[잠정]**; E3 calibrates.

Why it exists: MuScriptor's ``instruments`` hard-masks every other class. Leave keys/synth out of the guitar-view
mask and keyboard notes are forced into guitar classes; leave clean guitar out and that guitar disappears. So the
guitar mask needs to know which classes actually play in the song, with evidence a person can check and fix.

Evidence per MT3 class, combined with a noisy-OR ``p = 1 - prod(1 - w_s * e_s)``:

- **mix transcription** (``quality`` profile only; the all-class MuScriptor pass on the full mix, also baseline
  B0): ``active_ratio`` = share of bars with >= 2 notes of that class; ``e = clip(active_ratio / 0.1, 0, 1)`` if
  the class has >= ``mix_class_min_notes`` notes, else 0. Weight 0.9.
- **stem energy** (SW stems, ``energy.npz`` at 10 Hz): share of bars whose stem RMS is above ``active_rel_db``
  relative to the mix. piano -> keys classes, guitar -> guitar classes, other -> synth/organ/strings/brass/winds
  (weak, x0.5); additionally bass -> bass classes, vocals -> voice, drums -> drums (so that the per-song family
  table of b13 covers every family). Weight 0.6.
- **hints** (``hints.instruments``): listed families ``p = max(p, 0.99)``; with an explicit list, unlisted
  families ``p = min(p, 0.05)`` - guitar excepted (a wrong hint must not delete a guitar).

``present = p >= instr.threshold`` (``instr.guitar_threshold`` for guitar classes, deliberately low). If the
guitar stem is active in >= ``instr.guitar_stem_active_ratio`` of the bars, all three guitar classes are present
(DESIGN S3.5 start value). The masks derived from this (``gtab.amt.instruments``) never drop a guitar class.

Deterministic: pure numpy on the inputs, floats rounded to 1e-4, dict order fixed (MT3 order / FAMILIES order).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gtab.amt import instruments as I

log = logging.getLogger(__name__)

W_MIX = 0.9
W_STEM = 0.6
OTHER_STEM_SCALE = 0.5
MIX_SATURATION_RATIO = 0.1
HINT_PRESENT_P = 0.99
HINT_ABSENT_P = 0.05
KEYS_WARN_STEM_RATIO = 0.3

# stem -> (classes it evidences, scale)
STEM_EVIDENCE: dict[str, tuple[tuple[str, ...], float]] = {
    "piano": (tuple(I.FAMILIES["keys"]), 1.0),
    "guitar": (tuple(I.FAMILIES["guitar"]), 1.0),
    "other": (tuple(I.FAMILIES["synth"]) + ("organ",) + tuple(I.FAMILIES["strings"]) + tuple(I.FAMILIES["brass"])
              + tuple(I.FAMILIES["winds"]), OTHER_STEM_SCALE),
    "bass": (tuple(I.FAMILIES["bass"]), 1.0),
    "vocals": (("voice",), 1.0),
    "drums": (("drums",), 1.0),
}

DEFAULT_PARAMS: dict[str, Any] = {
    "threshold": 0.5, "guitar_threshold": 0.15, "guitar_stem_active_ratio": 0.05, "active_rel_db": -40.0,
    "mix_class_min_notes": 8,
}


def _r4(x: float) -> float:
    return round(float(x), 4)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def bars_from_grid(grid: Any) -> list[tuple[int, float, float]]:
    """[(bar, start_s, end_s)] from a Grid model or grid.json dict (musical bar numbers)."""
    out = []
    for b in _get(grid, "bars") or []:
        out.append((int(_get(b, "bar")), float(_get(b, "start_s")), float(_get(b, "end_s"))))
    out.sort()
    return out


def mix_evidence(notes: Sequence[Any], bars: Sequence[tuple[int, float, float]], *,
                 min_notes: int) -> dict[str, dict[str, Any]]:
    """Per class: n_notes, bars with >= 2 notes, active_ratio, e (clip(ratio / 0.1))."""
    starts = np.array([b[1] for b in bars], dtype=float)
    ends = np.array([b[2] for b in bars], dtype=float)
    per_class: dict[str, np.ndarray] = {}
    n_notes: dict[str, int] = {}
    for n in notes:
        cls = _get(n, "instrument")
        if cls is None or cls not in I.MT3_GROUPS:
            continue
        on = float(_get(n, "onset_s"))
        n_notes[cls] = n_notes.get(cls, 0) + 1
        if len(bars) == 0:
            continue
        k = int(np.searchsorted(starts, on, side="right") - 1)
        if 0 <= k < len(bars) and on < ends[k]:
            per_class.setdefault(cls, np.zeros(len(bars), dtype=np.int64))[k] += 1
    out: dict[str, dict[str, Any]] = {}
    for cls in I.MT3_GROUPS:
        cnt = per_class.get(cls)
        active = int((cnt >= 2).sum()) if cnt is not None else 0
        ratio = active / len(bars) if len(bars) else 0.0
        nn = n_notes.get(cls, 0)
        e = min(1.0, ratio / MIX_SATURATION_RATIO) if nn >= min_notes else 0.0
        out[cls] = {"n_notes": nn, "active_bars": active, "active_ratio": _r4(ratio), "e": _r4(e)}
    return out


def stem_bar_activity(energy: Mapping[str, np.ndarray], bars: Sequence[tuple[int, float, float]], *,
                      active_rel_db: float, fps: float = 10.0) -> dict[str, np.ndarray]:
    """Per stem: bool per bar, True if the stem's bar RMS is above ``active_rel_db`` relative to the mix's.

    Bar RMS comes from the 10 Hz dB series: mean power over the frames inside the bar.
    """
    mix = np.asarray(energy.get("mix"), dtype=np.float64) if "mix" in energy else None
    out: dict[str, np.ndarray] = {}
    if mix is None or len(bars) == 0:
        return out

    def bar_db(series: np.ndarray) -> np.ndarray:
        p = 10.0 ** (series / 10.0)
        vals = np.full(len(bars), -200.0)
        for i, (_b, s, e) in enumerate(bars):
            a, z = int(np.floor(s * fps)), int(np.ceil(e * fps))
            a, z = max(0, a), min(len(series), max(z, a + 1))
            if z > a:
                vals[i] = 10.0 * np.log10(np.mean(p[a:z]) + 1e-20)
        return vals

    mix_bar = bar_db(mix)
    for stem in STEM_EVIDENCE:
        if stem in energy:
            out[stem] = (bar_db(np.asarray(energy[stem], dtype=np.float64)) - mix_bar) > active_rel_db
    return out


def estimate(*, profile: str, bars: Sequence[tuple[int, float, float]], mix_notes: Sequence[Any] | None,
             energy: Mapping[str, np.ndarray] | None, sections: Sequence[Any] | None, hint: Any,
             params: Mapping[str, Any], guitar_view: str = "guitar_mono", guitar_mask: str = "guitar+present",
             piano_other_mode: str = "keys+guitar", energy_fps: float = 10.0) -> dict[str, Any]:
    """The ``instrumentation.json`` document (schema §6.5 ``Instrumentation``), built deterministically."""
    prm = dict(DEFAULT_PARAMS)
    prm.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS})
    families_hint = I.parse_hint(hint)
    use_mix = profile != "fast" and mix_notes is not None
    mixe = mix_evidence(mix_notes or [], bars, min_notes=int(prm["mix_class_min_notes"])) if use_mix else {}
    stem_act = stem_bar_activity(energy or {}, bars, active_rel_db=float(prm["active_rel_db"]), fps=energy_fps)
    stem_ratio = {s: (float(a.mean()) if len(a) else 0.0) for s, a in stem_act.items()}

    classes: dict[str, Any] = {}
    warnings: list[str] = []
    guitar_stem_ratio = stem_ratio.get("guitar", 0.0)
    guitar_forced = guitar_stem_ratio >= float(prm["guitar_stem_active_ratio"])
    for cls in I.MT3_GROUPS:
        fam = I.FAMILY_OF.get(cls, "")
        sources: dict[str, dict[str, Any]] = {}
        terms: list[float] = []
        if use_mix:
            m = mixe[cls]
            sources["mix_transcription"] = dict(m, w=W_MIX)
            terms.append(W_MIX * m["e"])
        stems_ev: dict[str, dict[str, Any]] = {}
        for stem, (evidenced, scale) in STEM_EVIDENCE.items():
            if cls in evidenced and stem in stem_ratio:
                # organ is evidenced by both the piano and the other stem: each is its own noisy-OR term
                e = scale * stem_ratio[stem]
                stems_ev[stem] = {"active_ratio": _r4(stem_ratio[stem]), "scale": scale, "e": _r4(e)}
                terms.append(W_STEM * e)
        if stems_ev:
            sources["stem_energy"] = {"w": W_STEM, "stems": stems_ev}
        p = 1.0 - float(np.prod([1.0 - t for t in terms])) if terms else 0.0
        if families_hint is not None:
            if fam in families_hint:
                p = max(p, HINT_PRESENT_P)
                sources["hint"] = {"families": list(families_hint), "effect": "listed"}
            elif fam != "guitar":
                p = min(p, HINT_ABSENT_P)
                sources["hint"] = {"families": list(families_hint), "effect": "not_listed"}
        thr = float(prm["guitar_threshold"] if fam == "guitar" else prm["threshold"])
        present = p >= thr
        if fam == "guitar" and guitar_forced:
            present = True
            sources["guitar_stem_rule"] = {"active_ratio": _r4(guitar_stem_ratio),
                                           "min_ratio": float(prm["guitar_stem_active_ratio"])}
        classes[cls] = {"p": _r4(min(1.0, max(0.0, p))), "present": bool(present), "sources": sources}

    families = {fam: _r4(max(classes[c]["p"] for c in members)) for fam, members in I.FAMILIES.items()}

    # per-section family activity: a family is active in a bar if its stem is active there or the mix
    # transcription has >= 2 notes of one of its classes in that bar.
    fam_bar = _family_bar_activity(mix_notes if use_mix else None, stem_act, bars)
    per_section = []
    bar_index = {b[0]: i for i, b in enumerate(bars)}
    for sec in sections or []:
        sid = str(_get(sec, "id"))
        sb, eb = int(_get(sec, "start_bar")), int(_get(sec, "end_bar"))
        idx = [bar_index[b] for b in range(sb, eb) if b in bar_index]  # half-open bar range
        fa = {fam: (_r4(float(np.mean(act[idx]))) if idx else 0.0) for fam, act in fam_bar.items()}
        per_section.append({"section": sid, "family_active": fa})

    if stem_ratio.get("piano", 0.0) >= KEYS_WARN_STEM_RATIO and use_mix:
        keys_notes = sum(mixe[c]["n_notes"] for c in I.FAMILIES["keys"])
        if keys_notes < int(prm["mix_class_min_notes"]):
            warnings.append("건반(piano) 스템 에너지는 있는데 믹스 전사에 건반 음이 거의 없음: 건반이 있는지 확인하세요.")
    if not use_mix and profile != "fast":
        warnings.append("믹스 전사가 없어 스템 에너지와 힌트로만 편성을 정했습니다.")
    if not stem_ratio:
        warnings.append("스템 에너지(energy.npz)가 없어 믹스 전사와 힌트로만 편성을 정했습니다.")

    doc_present = {c: classes[c]["present"] for c in classes}
    inst = {"classes": {c: {"present": v} for c, v in doc_present.items()}}
    guitar_pass = I.mask_guitar_view(inst, guitar_mask)
    # Pass-1 views keep their own masks; the guitar pass (amt_gtr) reads "guitar_pass". The guitar view id is
    # listed too (spec example) unless it is a pass-1 view itself (E2: mix_mono), whose pass-1 mask stays.
    masks: dict[str, list[str] | None] = {}
    if guitar_view not in I.pass1_masks(piano_other_mode):
        masks[guitar_view] = guitar_pass
    masks.update(I.pass1_masks(piano_other_mode))
    masks["guitar_pass"] = guitar_pass
    thresholds = {k: prm[k] for k in DEFAULT_PARAMS}
    thresholds.update({"w_mix": W_MIX, "w_stem": W_STEM, "mix_saturation_ratio": MIX_SATURATION_RATIO,
                       "guitar_mask": guitar_mask, "piano_other_instruments": piano_other_mode})
    return {
        "format": "gtab.instrumentation/1",
        "profile": profile,
        "classes": classes,
        "families": families,
        "masks": masks,
        "per_section": per_section,
        "thresholds": thresholds,
        "hints": {"instruments": hint if isinstance(hint, (str, type(None))) else list(hint),
                  "families": families_hint},
        "warnings": warnings,
    }


def _family_bar_activity(mix_notes: Sequence[Any] | None, stem_act: Mapping[str, np.ndarray],
                         bars: Sequence[tuple[int, float, float]]) -> dict[str, np.ndarray]:
    n = len(bars)
    fam_act = {fam: np.zeros(n, dtype=bool) for fam in I.FAMILIES}
    for stem, (evidenced, _scale) in STEM_EVIDENCE.items():
        act = stem_act.get(stem)
        if act is None:
            continue
        fams = {I.FAMILY_OF[c] for c in evidenced}
        if stem == "other":
            continue  # too ambiguous for a family-level activity map (weak evidence only)
        for fam in fams:
            fam_act[fam] |= act
    if mix_notes and n:
        starts = np.array([b[1] for b in bars])
        ends = np.array([b[2] for b in bars])
        counts: dict[str, np.ndarray] = {}
        for note in mix_notes:
            cls = _get(note, "instrument")
            fam = I.FAMILY_OF.get(cls) if cls else None
            if fam is None:
                continue
            on = float(_get(note, "onset_s"))
            k = int(np.searchsorted(starts, on, side="right") - 1)
            if 0 <= k < n and on < ends[k]:
                counts.setdefault(cls, np.zeros(n, dtype=np.int64))[k] += 1
        for cls, c in counts.items():
            fam_act[I.FAMILY_OF[cls]] |= c >= 2
    return fam_act


def estimated_families(instrumentation: Mapping[str, Any]) -> list[str]:
    """FAMILIES keys with at least one present class (the b13 comparison with ``families_present``)."""
    present = {c for c, ev in (instrumentation.get("classes") or {}).items()
               if (ev.get("present") if isinstance(ev, Mapping) else getattr(ev, "present", False))}
    return [fam for fam, members in I.FAMILIES.items() if any(c in present for c in members)]


def guitar_class_omissions(instrumentation: Mapping[str, Any]) -> int:
    """Guitar classes missing from the guitar-pass mask (must be 0 in every case; b13 / E3 constraint)."""
    masks = instrumentation.get("masks") or {}
    return I.guitar_omissions(masks.get("guitar_pass", I.mask_guitar_only()))
