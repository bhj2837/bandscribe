"""Identity metrics of a part assignment (DESIGN 4.11, M7b): identity switches and the block diarization error.

Both work on time blocks of ``BLOCK_S`` seconds (about a bar at 120 BPM; the Tier B remixes have no bar grid) and on
predicted labels mapped to the true Lines by the macro accuracy's Hungarian match (``eval.macro_hungarian``).
A predicted label the match leaves out, and Unassigned, are identities of their own (``x:<label>``, ``u``).

- ``identity_switches``: per true Line, its exclusive notes in time order; each block where it has notes takes the
  identity most of them got (ties: the earlier note's); a switch is a change between two consecutive such blocks.
- ``block_der``: per block, the true Lines with at least ``MIN_BLOCK_NOTES`` notes (a unison note counts for each of
  its Lines) and the predicted identities with as many; missed = more true than predicted, false alarm = more
  predicted than true, confusion = the smaller count minus the identities both sides share; DER = their sum over all
  blocks / the true Line-blocks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

BLOCK_S = 2.0
MIN_BLOCK_NOTES = 2


def _ident(p: Any, match: Mapping[str, str]) -> str:
    if p is None:
        return "u"
    return match.get(str(p), f"x:{p}")


def identity_switches(truth: Sequence[set[str]], pred: Sequence[Any], onsets: Sequence[float],
                      match: Mapping[str, str], block_s: float = BLOCK_S) -> dict[str, Any]:
    """{"total": switches over the true Lines, "per_line": {line: switches}, "blocks": {line: active blocks}}."""
    lines = sorted({ln for t in truth if len(t) == 1 for ln in t})
    per: dict[str, int] = {}
    nblk: dict[str, int] = {}
    for ln in lines:
        idx = sorted((i for i, t in enumerate(truth) if t == {ln}), key=lambda i: (float(onsets[i]), i))
        by_block: dict[int, list[str]] = {}
        for i in idx:
            by_block.setdefault(int(float(onsets[i]) // block_s), []).append(_ident(pred[i], match))
        seq = []
        for b in sorted(by_block):
            ids = by_block[b]
            cnt = Counter(ids)
            top = max(cnt.values())
            seq.append(next(x for x in ids if cnt[x] == top))  # ties: the earliest note's identity
        per[ln] = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        nblk[ln] = len(seq)
    return {"total": sum(per.values()), "per_line": per, "blocks": nblk}


def block_der(truth: Sequence[set[str]], pred: Sequence[Any], onsets: Sequence[float], match: Mapping[str, str],
              block_s: float = BLOCK_S, min_notes: int = MIN_BLOCK_NOTES) -> dict[str, Any]:
    """{"der", "missed", "false_alarm", "confusion", "true_blocks"} (``der`` None without any true Line-block)."""
    true_n: dict[int, Counter] = {}
    pred_n: dict[int, Counter] = {}
    for t, p, on in zip(truth, pred, onsets):
        b = int(float(on) // block_s)
        for ln in t:
            true_n.setdefault(b, Counter())[ln] += 1
        if t or p is not None:
            pred_n.setdefault(b, Counter())[_ident(p, match)] += 1
    miss = fa = conf = tot = 0
    for b in sorted(set(true_n) | set(pred_n)):
        tl = {ln for ln, n in (true_n.get(b) or {}).items() if n >= min_notes}
        pl = {x for x, n in (pred_n.get(b) or {}).items() if n >= min_notes and x != "u"}
        tot += len(tl)
        miss += max(0, len(tl) - len(pl))
        fa += max(0, len(pl) - len(tl))
        conf += min(len(tl), len(pl)) - len(tl & pl)
    return {"der": (miss + fa + conf) / tot if tot else None, "missed": miss, "false_alarm": fa,
            "confusion": conf, "true_blocks": tot}
