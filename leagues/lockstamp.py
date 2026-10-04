"""What is frozen WITH a pick, at the moment it locks -- shared by both lockers.

THE PROBLEM THIS FIXES (review, 2026-10-04). leagues.publish stamped bet365's price
and a value verdict onto a pick on the run that locked it; scripts/lock_picks.py, the
fast locker that freezes most picks, stamped neither. tracking.release.mark then ran
straight afterwards, found no value verdict, and labelled the pick "tracked" for
good -- so no soccer pick frozen by the fast locker could ever be an official bet,
and "value_plays" stayed empty. Both lockers now call these functions BEFORE
release.mark, so the label is decided on the same evidence whichever froze it.

Only on the run that locks the pick (`just_locked`): a price stamped later would be
one nobody was offered at the moment of the decision.
"""
from __future__ import annotations

import pandas as pd

from leagues import odds, prop_odds, shot_odds
from tracking import trust, value


def just_locked(entry: dict, now, minutes: float = 30.0) -> bool:
    """True when `entry` was locked on THIS run (or within the last half hour)."""
    try:
        locked = pd.Timestamp(entry.get("locked_at"))
        now = pd.Timestamp(now)
        if locked.tzinfo is None:
            locked = locked.tz_localize("UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        return abs((now - locked).total_seconds()) <= minutes * 60
    except Exception:
        return False


def pick_side(pick: str, home: str, away: str) -> str:
    return "home" if pick == home else "away" if pick == away else "draw"


def stamp_match(entry: dict, market_odds, home: str, away: str, now) -> None:
    """bet365's 1X2 price for the frozen side, and the value verdict at that price."""
    if entry.get("odds") is not None or not just_locked(entry, now):
        return
    side = pick_side(entry.get("pick"), home, away)
    mk = (odds.market_for(market_odds, home, away) if market_odds is not None else None) or {}
    price, fair = (mk.get("odds") or {}).get(side), mk.get(f"p_{side}")
    if price:
        entry["odds"] = float(price)
        entry["odds_book"] = str(mk.get("book") or "")     # whose price it was
    if fair is not None:
        entry["book_p_pick"] = round(float(fair), 4)     # what tracking/trust fits on
    # Judged on the model pulled toward bet365 by the weight the record supports.
    entry["value"] = value.assess(trust.soccer_value_p(entry.get("p_pick"), fair, "winner"), fair,
                                  price, str(mk.get("book") or "").lower() == "bet365")


def stamp_prop(entry: dict, league: str, match_id, now, prop_store=None, shots_store=None) -> None:
    """The price at lock for a player pick: bet365's anytime price for a goal pick;
    a US book's price on the same line for shots / on target (bet365 has no such
    market in any feed), with the bet365 price ESTIMATED from it."""
    if entry.get("book_price") is not None or not just_locked(entry, now):
        return
    market, prob = entry.get("market"), entry.get("p_pick")
    if market == "goal":
        price = prop_odds.price_for(league, match_id, entry.get("player"), store=prop_store)
        if price:
            entry["book_price"] = float(price)
        entry["value"] = value.assess(prob, None, price, bool(price))
    elif market in shot_odds.LINE:
        shot_odds.freeze(entry, league, match_id, store=shots_store)
        entry["value"] = value.assess(trust.soccer_value_p(prob, entry.get("book_p"), "prop"),
                                      entry.get("book_p"), None, False)
