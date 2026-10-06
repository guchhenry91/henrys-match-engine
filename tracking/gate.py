"""Each kind of pick EARNS its place from its own graded record (NFL / NBA / MLB props).

A group is a market and a side ("hits|over", "rbi|under"). With MIN_N graded picks
or more, it is HELD BACK when its record says, at Z standard errors, either

  * it wins less often than its own stated chances (calibration): the picks said
    27 of 42 hits overs should land and 19 did -- the selection is choosing the
    side the book had right; or
  * it loses money at the prices actually taken (profit).

Losing alone is not enough. Home runs went 5-40 on 2026-10-06 against 6.1 expected
and lost about the bookmaker's margin -- long shots doing what long shots do, so
they are NOT held back. A hand-picked list would have dropped them.

Held-back picks are still published, frozen and graded -- the record keeps growing,
so a group that recovers comes back on its own -- but their value verdict is
withdrawn, so they never reach the Bet list.
"""
from __future__ import annotations

import math

from tracking import performance

MIN_N = 20
Z = 2.0


def _side(e: dict) -> str:
    return str(e.get("side") or ("yes" if e.get("market") == "anytime_touchdown" else "over"))


def report(entries) -> dict:
    """{"market|side": {n, won, expected, calib_z, roi, roi_z, held, reason}}."""
    groups = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("kind") == "winner" or not e.get("market"):
            continue
        groups.setdefault(f"{e['market']}|{_side(e)}", []).append(e)
    out = {}
    for key, es in groups.items():
        graded = [e for e in es if e.get("graded") in ("correct", "wrong")]
        probs = []
        for e in graded:
            try:
                probs.append(float(e.get("probability", e.get("p_pick"))))
            except (TypeError, ValueError):
                probs.append(None)
        pairs = [(e, p) for e, p in zip(graded, probs) if p is not None and 0 < p < 1]
        n = len(pairs)
        won = sum(e["graded"] == "correct" for e, _ in pairs)
        expected = sum(p for _, p in pairs)
        var = sum(p * (1 - p) for _, p in pairs)
        calib_z = (won - expected) / math.sqrt(var) if var > 0 else 0.0
        s = performance.summarize(es)
        roi, se = s.get("roi"), s.get("roi_se")
        roi_z = roi / se if roi is not None and se else None
        reason = None
        if n >= MIN_N and calib_z <= -Z:
            reason = f"won {won} of {n} against {expected:.1f} its own chances promised"
        elif s.get("bets", 0) >= MIN_N and roi_z is not None and roi_z <= -Z:
            reason = f"{roi:+.0%} at the prices taken over {s['bets']} bets"
        out[key] = {"n": n, "won": won, "expected": round(expected, 1),
                    "calib_z": round(calib_z, 2), "roi": roi,
                    "roi_z": round(roi_z, 2) if roi_z is not None else None,
                    "held": reason is not None, "reason": reason}
    return out


def _withhold(p: dict, why: str) -> None:
    p["held_back"] = why
    v = p.get("value")
    if isinstance(v, dict):
        v.update(value=False, stake=0.0, note=f"held back: {why}")


def apply(payload: dict, rep: dict) -> int:
    """Withdraw the value verdict from every pick in a held-back group. Returns count."""
    n = 0
    lists = [b.get("picks") or [] for b in (payload.get("props") or {}).values()]
    lists += [ps for markets in (payload.get("game_props") or {}).values()
              for ps in (markets or {}).values()]
    for picks in lists:
        for p in picks:
            g = rep.get(f"{p.get('market')}|{_side(p)}")
            if g and g["held"]:
                _withhold(p, g["reason"])
                n += 1
    payload["gate"] = {"min_n": MIN_N, "z": Z, "groups": rep,
                       "held_back": sorted(k for k, g in rep.items() if g["held"])}
    return n
