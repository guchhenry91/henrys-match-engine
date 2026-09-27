"""Pull the NFL yardage probabilities toward the bookmaker's price.

WHY. The live record says the model is OVERCONFIDENT on yardage, and more so the
surer it is: picks it rated 60-65% hit 41%, 65-70% hit 38% (203 graded,
2026-09-27). The walk-forward backtest says calibrated, so the error lives in
what the backtest does not see -- chiefly publishing only each game's top picks,
which selects exactly the players the model is most wrong about. A sharp book's
price already knows most of what the model knows.

HOW.  p = anchor + w * (p_model - anchor)
  anchor = the book's de-vigged over probability at the line the pick stands on.
  w      = how much of the model's disagreement with the book to believe, FITTED on
           the graded record by maximum likelihood and bounded to [0, 1]:
           w = 0 publishes the book's own view, w = 1 the raw model.

FITTING ON THE RECORD WE HAVE. Most graded picks predate bookmaker lines and
stood on the player's OWN median, where the fair over probability is 1/2 by
construction -- so there the anchor is 0.5. Picks that stood on a book line use
that book's de-vigged price. As book-line picks accumulate they dominate the fit,
which is the point: `scripts/fit_nfl_blend.py` re-fits every run.

Anytime TD is NOT blended: its record runs the other way (stated 53%, hit 63%)
and its book price is one-sided with the margin in it, so the same formula would
pull it toward a number that is not a fair probability.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "data-raw" / "nfl" / "blend_weights.json"
YARDAGE = ("receiving_yards", "rushing_yards", "passing_yards")
GRID = [i / 20 for i in range(21)]          # 0.00, 0.05, ... 1.00
# Until a market has this many graded picks of its own, it uses the pooled
# yardage weight: 28 passing picks cannot carry a weight of their own.
MIN_PER_MARKET = 100


def blend(p_model: float, anchor: float, w: float) -> float:
    p = anchor + w * (p_model - anchor)
    return min(max(p, 0.01), 0.99)


def _loglik(rows, w) -> float:
    total = 0.0
    for p_model, anchor, outcome in rows:
        p = blend(p_model, anchor, w)
        total += math.log(p if outcome else 1.0 - p)
    return total


def fit(rows) -> dict:
    """rows: [(p_model, anchor, outcome 0/1)]. The w in GRID maximising log-lik."""
    if not rows:
        return {"w": None, "n": 0}
    scored = [(w, _loglik(rows, w)) for w in GRID]
    best_w, best_ll = max(scored, key=lambda t: t[1])
    n = len(rows)
    return {"w": best_w, "n": n,
            "loglik_per_pick": round(best_ll / n, 4),
            "raw_model_loglik_per_pick": round(_loglik(rows, 1.0) / n, 4),
            "anchor_only_loglik_per_pick": round(_loglik(rows, 0.0) / n, 4),
            "hit_rate": round(sum(r[2] for r in rows) / n, 4),
            "raw_model_mean": round(sum(r[0] for r in rows) / n, 4)}


def rows_from_log(props_log: dict, market=None) -> list:
    """Graded yardage picks as (p_model, anchor, outcome)."""
    out = []
    for e in props_log.values():
        if e.get("market") not in YARDAGE or e.get("graded") not in ("correct", "wrong"):
            continue
        if market and e["market"] != market:
            continue
        # The model's OWN number: newer entries store it separately once blending
        # is live; before that the frozen p_pick was the raw model.
        p_model = e.get("p_model", e.get("p_pick"))
        if p_model is None:
            continue
        book = e.get("book_p")
        anchor = float(book) if book is not None and e.get("line_source") not in (None, "model") else 0.5
        out.append((float(p_model), anchor, 1 if e["graded"] == "correct" else 0))
    return out


def fit_all(props_log: dict) -> dict:
    pooled = fit(rows_from_log(props_log))
    out = {"pooled": pooled}
    for market in YARDAGE:
        own = fit(rows_from_log(props_log, market))
        use = own["w"] if own["n"] >= MIN_PER_MARKET and own["w"] is not None else pooled["w"]
        out[market] = {**own, "w_used": use,
                       "source": "own" if own["n"] >= MIN_PER_MARKET else "pooled"}
    return out


def weights() -> dict:
    """market -> w in force. Missing file = trust the book fully (w = 0): the safe
    default for a model whose record says it is overconfident."""
    try:
        raw = json.loads(WEIGHTS.read_text(encoding="utf-8"))
    except Exception:
        return {m: 0.0 for m in YARDAGE}
    return {m: float((raw.get(m) or {}).get("w_used") or 0.0) for m in YARDAGE}


# A ladder is over-only, so each rung's 1/price still carries the book's margin.
# It is removed with the overround of the player's main line (both sides quoted);
# failing that, a typical ladder margin. Never below MIN_OVERROUND, so a rung can
# only be made to look LESS likely than its raw price, not more.
DEFAULT_OVERROUND = 1.06
MIN_OVERROUND = 1.03


# THE LADDER UP: each rung ABOVE the main line, with its chance of hitting, so a
# reader can see how high a player is likely to go -- "55.5+ 41%, 65.5+ 30%, ...".
# Same arithmetic as the main pick: the rung's price, de-margined, then pulled by
# the fitted share w of the model's disagreement. Real, bettable rungs only.
LADDER_UP_MAX = 5
LADDER_UP_MIN_P = 0.05


def ladder_up(ladder, pick: dict, w: float) -> list:
    """[{"line", "price", "book", "p"}] for rungs above the pick's line, lowest first."""
    if not ladder or not ladder.get("rungs") or pick.get("line") is None:
        return []
    main = DEFAULT_OVERROUND
    if pick.get("book_p") and pick.get("book_price"):
        main = (1.0 / float(pick["book_price"])) / float(pick["book_p"])
    overround = max(main, MIN_OVERROUND)
    out = []
    for r in sorted(ladder["rungs"], key=lambda r: r["line"]):
        if r["line"] <= float(pick["line"]):
            continue
        anchor = min((1.0 / r["price"]) / overround, 0.99)
        p = anchor if r.get("raw") is None else blend(r["raw"], anchor, w)
        # Never above the main line's own chance: a higher line cannot be likelier.
        p = min(p, float(pick.get("probability") or 1.0))
        if p < LADDER_UP_MIN_P:
            break
        out.append({"line": r["line"], "price": r["price"], "book": ladder["book"],
                    "p": round(p, 4)})
        if len(out) == LADDER_UP_MAX:
            break
    return out
