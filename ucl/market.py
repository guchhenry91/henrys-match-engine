"""bet365's 1X2 prices for the Champions League board, and the check against them.

WHY (owner's request, 2026-10-13). The model is fitted on European results only, so a
club that beat minnows in qualifying looks stronger than it is: the board picked
Slovan Bratislava over Stuttgart (bet365 ~1.30 Stuttgart) and Sabah over Slavia
Prague. Every other board in the app trusts the book until real results prove the
model (tracking/combine_fit); this does the same for the Champions League.

  - The book's chance is de-vigged from bet365's three prices (proportional).
  - The published chance is the book's moved a fraction A_CAP of the way toward the
    model's (geometric pooling over the three outcomes, the 3-way form of
    combine_fit.shrink). The model's own numbers are kept as p_model_*.
  - Where the MODEL'S pick sits more than MAX_GAP from the book's chance of that
    same outcome, the card says "model disagrees with the market" -- the raw model
    is not trusted there, and the published pick comes from the combined chance.
  - A fixture bet365 has not priced yet stays on the model alone, labelled, and is
    never a Best pick: an unchecked number is not a recommendation.

Prices come from API-Football (bookmaker 8 = bet365, bet 1 = Match Winner), written
by scripts/sync_ucl_odds.py to data-raw/ucl/odds.json. Free under the existing
API-Football guards; no Odds API credits.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data-raw" / "ucl" / "odds.json"
BET365 = 8
MATCH_WINNER = 1
A_CAP = 0.2            # tracking/combine_fit.A_CAP: the model may only nudge the book
MAX_GAP = 0.15         # tracking/pricecheck.MAX_GAP
SIDES = ("home", "draw", "away")
LABEL = {"Home": "home", "Draw": "draw", "Away": "away"}


def parse(rows) -> dict:
    """{fixture id: {"home","draw","away"} decimal odds} from API-Football /odds rows."""
    out = {}
    for r in rows or []:
        fid = (r.get("fixture") or {}).get("id")
        if fid is None:
            continue
        for bm in r.get("bookmakers") or []:
            if int(bm.get("id", -1)) != BET365:
                continue
            for bet in bm.get("bets") or []:
                if int(bet.get("id", -1)) != MATCH_WINNER:
                    continue
                prices = {}
                for v in bet.get("values") or []:
                    side = LABEL.get(str(v.get("value")))
                    try:
                        odd = float(v.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    if side and odd > 1.0:
                        prices[side] = odd
                if len(prices) == 3:
                    out[str(fid)] = prices
    return out


def load(path: Path | None = None) -> dict:
    try:
        return json.loads(Path(path or STORE).read_text(encoding="utf-8"))
    except Exception:
        return {}


def devig(odds: dict) -> dict:
    raw = {s: 1.0 / float(odds[s]) for s in SIDES}
    total = sum(raw.values())
    return {s: raw[s] / total for s in SIDES}


def combine(model: dict, book: dict, a: float = A_CAP) -> dict:
    """book^(1-a) * model^a, renormalised: the book moved a fraction a toward the model."""
    w = {s: math.exp((1 - a) * math.log(max(book[s], 1e-6)) + a * math.log(max(model[s], 1e-6)))
         for s in SIDES}
    total = sum(w.values())
    return {s: w[s] / total for s in SIDES}


def apply(match: dict, quote: dict | None, confidence) -> None:
    """Rewrite one board match in place: market numbers, combined chance, pick, flags.

    `confidence` maps a probability to the board's 1-5 confidence."""
    model = {"home": match["p_home"], "draw": match["p_draw"], "away": match["p_away"]}
    name = {"home": match["home"], "draw": "Draw", "away": match["away"]}
    model_side = max(model, key=model.get)
    match.update({f"p_model_{s}": round(model[s], 3) for s in SIDES})
    match["model_pick"] = name[model_side]
    odds = (quote or {}).get("odds") if quote else None
    if not odds or any(not odds.get(s) for s in SIDES):
        match["basis"] = "model only"
        match["market"] = None
        match["best_pick"] = False
        match["market_note"] = "no bet365 price yet -- model only, not a pick to back"
        return
    book = devig(odds)
    both = combine(model, book)
    side = max(both, key=both.get)
    gap = abs(model[model_side] - book[model_side])
    match.update({f"p_{s}": round(both[s], 3) for s in SIDES})
    match["pick"] = name[side]
    match["p_pick"] = round(both[side], 4)
    match["confidence"] = confidence(both[side])
    match["best_pick"] = both[side] >= 0.65
    match["basis"] = "with market"
    match["market"] = {"book": "bet365", "odds": {s: float(odds[s]) for s in SIDES},
                       "p_book": {s: round(book[s], 3) for s in SIDES},
                       "pick_odds": float(odds[side]),
                       "checked_at": quote.get("checked_at")}
    match["model_gap"] = round(gap, 3)
    match["disagrees"] = gap > MAX_GAP
    if match["disagrees"]:
        match["market_note"] = (f"model alone said {name[model_side]} {model[model_side]:.0%}; "
                                f"bet365 makes it {book[model_side]:.0%} -- the model "
                                f"overrates sides it knows from weak European opposition, "
                                f"so the pick follows the market")
    else:
        match["market_note"] = None


def annotate(matches: list, confidence, store: dict | None = None) -> int:
    """Apply bet365's prices to every board match. Returns how many were priced."""
    fixtures = (store if store is not None else load()).get("fixtures") or {}
    priced = 0
    for m in matches:
        quote = fixtures.get(str(m.get("id")))
        apply(m, quote, confidence)
        priced += m.get("basis") == "with market"
    return priced
