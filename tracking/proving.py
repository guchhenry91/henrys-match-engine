"""Is the rebuilt model doing what it says? One panel per market (audit, 2026-10-12).

For the REBUILT engine's graded picks only:
  n            picks settled (won or lost)
  said         average stated chance
  landed       share that won, with a 95% Wilson interval
  voids        picks voided (did not play, push) and their share
  clv          closing-line value where a closing price is on record (same line)
  line_move    where the closing line moved off ours: the share that moved our way
  roi          profit per unit at the price each pick was approved at (bet365's, or
               the bet365 estimate the price check used), with a 95% lower bound
  status       "unproven" below MIN_PROVEN graded; then "on target" when landed is at
               least said - SLACK AND the top of its interval reaches said, else
               "behind". "behind (early)" is raised before MIN_PROVEN when even the
               top of the interval sits below said.
  bettable     THIRD REVIEW, 2026-10-12: "on target" proves the hit rate, not profit
               (a market landing 2 points under at break-even prices loses ~3%). A
               market is bettable only when it is on target, its ROI at the approved
               prices is >= 0, AND either the ROI's 95% lower bound is above 0 or it
               beats the closing price significantly (CLV z >= 2 on MIN_CLV pairs).
Real money waits for "bettable".
"""
from __future__ import annotations

import math

MIN_PROVEN = 300
SLACK = 0.02
MIN_CLV = 100
CLV_Z = 2.0
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
        rets = []
        for e in settled:
            price = performance.price_of(e)
            if price:
                rets.append(price - 1.0 if e["graded"] == "correct" else -1.0)
        roi = sum(rets) / len(rets) if rets else None
        roi_lo = None
        if len(rets) >= 2:
            sd = math.sqrt(sum((r - roi) ** 2 for r in rets) / (len(rets) - 1))
            roi_lo = roi - 1.96 * sd / math.sqrt(len(rets))
        sizes = [c[1] for c in clv]
        clv_z = None
        if len(sizes) >= 2:
            m_ = sum(sizes) / len(sizes)
            sd_ = math.sqrt(sum((x - m_) ** 2 for x in sizes) / (len(sizes) - 1))
            clv_z = m_ / (sd_ / math.sqrt(len(sizes))) if sd_ > 0 else None
        # Where the closing line MOVED off ours (scripts/close_props.py), the price
        # cannot be compared; which way it moved still says who the market agreed with.
        moves = [(e.get("closing") or {}).get("moved") for e in settled]
        moves = [m for m in moves if m in ("ours", "against")]
        if n < MIN_PROVEN:
            status = "behind (early)" if (n >= 30 and hi is not None and hi < said) else "unproven"
        else:
            status = "on target" if (landed >= said - SLACK and hi is not None and hi >= said) else "behind"
        bettable = bool(status == "on target" and roi is not None and roi >= 0 and (
            (roi_lo is not None and roi_lo > 0)
            or (len(sizes) >= MIN_CLV and clv_z is not None and clv_z >= CLV_Z)))
        out[market] = {
            "n": n, "pending": sum(1 for e in es if not e.get("graded")),
            "said": round(said, 4) if said is not None else None,
            "landed": round(landed, 4) if landed is not None else None,
            "landed_95": [lo, hi], "voids": voids,
            "void_rate": round(voids / (n + voids), 4) if (n + voids) else None,
            "clv": ({"n": len(clv), "beat_close": round(sum(c[0] == "ours" for c in clv) / len(clv), 4),
                     "avg": round(sum(c[1] for c in clv) / len(clv), 4)} if clv else None),
            "line_move": ({"n": len(moves), "ours": round(moves.count("ours") / len(moves), 4)}
                          if moves else None),
            "roi": None if roi is None else round(roi, 4),
            "roi_lo95": None if roi_lo is None else round(roi_lo, 4),
            "priced": len(rets),
            "clv_z": None if clv_z is None else round(clv_z, 2),
            "status": status, "bettable": bettable, "needed": max(MIN_PROVEN - n, 0),
        }
    return out
