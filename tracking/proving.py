"""Is the rebuilt model doing what it says? One panel per market (audit, 2026-10-12).

For the REBUILT engine's graded picks only:
  n            picks settled (won or lost)
  said         average stated chance
  landed       share that won, with a 95% Wilson interval
  voids        picks voided (did not play, push) and their share
  clv          closing-line value where a closing price is on record
  status       "unproven" below MIN_PROVEN graded; then "on target" when landed is at
               least said - SLACK, else "behind". "behind (early)" is raised before
               MIN_PROVEN when even the top of the interval sits below said -- a
               market clearly missing does not need 300 picks to show it.
Real money waits for "on target" (the four reviewers' bar).
"""
from __future__ import annotations

import math

MIN_PROVEN = 300
SLACK = 0.02
ENGINE = "forecast-v1"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def _p(e):
    for k in ("p_pick", "probability"):
        try:
            v = float(e.get(k))
            if 0 < v < 1:
                return v
        except (TypeError, ValueError):
            continue
    return None


def summary(entries, market_of=lambda e: e.get("market"), engine: str = ENGINE) -> dict:
    from tracking import performance
    by = {}
    for e in entries:
        if not isinstance(e, dict) or e.get("engine") != engine:
            continue
        by.setdefault(market_of(e) or "?", []).append(e)
    out = {}
    for market, es in sorted(by.items()):
        settled = [e for e in es if e.get("graded") in ("correct", "wrong") and _p(e) is not None]
        voids = sum(1 for e in es if e.get("graded") == "void")
        n = len(settled)
        won = sum(1 for e in settled if e["graded"] == "correct")
        said = sum(_p(e) for e in settled) / n if n else None
        landed = won / n if n else None
        lo, hi = _wilson(won, n)
        clv = [performance.clv(e) for e in settled]
        clv = [c for c in clv if c[0] is not None]
        if n < MIN_PROVEN:
            status = "behind (early)" if (n >= 30 and hi is not None and hi < said) else "unproven"
        else:
            status = "on target" if landed >= said - SLACK else "behind"
        out[market] = {
            "n": n, "pending": sum(1 for e in es if not e.get("graded")),
            "said": round(said, 4) if said is not None else None,
            "landed": round(landed, 4) if landed is not None else None,
            "landed_95": [lo, hi], "voids": voids,
            "void_rate": round(voids / (n + voids), 4) if (n + voids) else None,
            "clv": ({"n": len(clv), "beat_close": round(sum(c[0] == "ours" for c in clv) / len(clv), 4),
                     "avg": round(sum(c[1] for c in clv) / len(clv), 4)} if clv else None),
            "status": status, "needed": max(MIN_PROVEN - n, 0),
        }
    return out
