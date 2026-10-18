"""THE PRICE CHECK: a pick is published only if it is worth betting at bet365.

Owner's decision, 2026-10-12, after the four-reviewer audit: "winning picks" and
"making money" only meet when the price is right. A pick that wins 70% of the time at
1.30 loses money in the long run (it needs 77% to break even). So every pick, in every
sport, now carries:

  ref_price         the price it is judged at: bet365's own quote where we have it,
                    otherwise an ESTIMATE of bet365's price from another book's fair
                    chance less a typical bet365 margin (tracking/value.py) -- marked
                    "estimate" on the page;
  break_even_price  the odds at which the pick's stated chance breaks even;
  min_price         the lowest odds worth taking: the stated chance returns MIN_EV;
  ev                the expected return per unit staked at ref_price;
  ok                ev >= MIN_EV and the forecast within MAX_GAP of the book.

A pick with ok=False can still be published as a PROVING pick when the model backs it
(`qualifies`); it is never offered as a bet. Otherwise it stays on the game page,
flagged short_price. A pick with no price at all cannot be checked and is not
published either -- except where the caller lets it wait for a price (soccer shots,
whose price is fetched only for picks already on the board) without ever locking it.
"""
from __future__ import annotations

from tracking import value

# THIRD REVIEW, 2026-10-12 -- all four reviewers: "EV >= 0" was passing picks whose
# edge came from the combining formula, not from skill, at an optimistic bet365 price.
MIN_EV = 0.03            # at least +3% expected return at the reference price
# bet365's margin where only another book's fair chance is known. Two-way player
# props run ~7.5% over the book; one-sided markets (anytime TD, home runs, soccer
# shots / on target as bet builders) carry far more.
PROP_MARGIN = 0.075
ONE_SIDED_MARGIN = 0.12
ONE_SIDED = {"anytime_touchdown", "hr", "shots", "sot"}
# A forecast far from the book is more often wrong than clever: the 2026-10-01 audit
# found the biggest disagreements landed 49%. Beyond this gap (in chance, same side)
# no pick is published, as WINNER_GAP does for team winners (tracking/value.py).
MAX_GAP = 0.15
# A proving pick needs the model GENUINELY behind it: at least this much surer than the
# book's fair chance. A model merely equal to the book adds nothing -- that is the
# book's pick, and the live record showed those land below the book (owner, 2026-10-12).
MIN_BACKING = 0.02


def margin_for(pick: dict) -> float:
    return ONE_SIDED_MARGIN if (pick.get("market") in ONE_SIDED or pick.get("fair") is False) \
        else PROP_MARGIN


def gap(pick: dict, fair_key: str = "book_p"):
    """|forecast - book| for the side taken, or None when either is missing."""
    try:
        return abs(float(pick["p_model"]) - float(pick[fair_key]))
    except (KeyError, TypeError, ValueError):
        return None


def check(pick: dict, prob_key: str = "probability", fair_key: str = "book_p",
          price_key: str = "book_price", is_bet365: bool | None = None,
          margin: float | None = None) -> dict:
    """Annotate `pick` with price_check and return it."""
    p = pick.get(prob_key)
    b365 = value._is_bet365(pick) if is_bet365 is None else is_bet365
    v = value.assess(p, None if b365 else pick.get(fair_key), pick.get(price_key), b365,
                     margin=margin_for(pick) if margin is None else margin)
    try:
        be = round(1.0 / float(p), 2)
        mp = round((1.0 + MIN_EV) / float(p), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        be = mp = None
    g = gap(pick, fair_key)
    if not v or v.get("ev") is None:
        out = {"ok": False, "reason": "no price", "break_even_price": be, "min_price": mp}
    else:
        reason = (None if v["ev"] >= MIN_EV else "price too short")
        if reason is None and g is not None and g > MAX_GAP:
            reason = "forecast too far from the book"
        out = {"ok": reason is None, "ref_price": v["ref_price"], "basis": v["basis"],
               "ev": v["ev"], "break_even_price": be, "min_price": mp,
               "gap": None if g is None else round(g, 4),
               "reason": reason}
    pick["price_check"] = out
    if not out["ok"]:
        pick["short_price"] = out.get("reason") == "price too short"
    return out


def passes(pick: dict, **kw) -> bool:
    return check(pick, **kw)["ok"]


def model_backs(pick: dict, fair_key: str = "book_p") -> bool:
    """The forecast is at least MIN_BACKING surer than the book's fair chance of the same
    side, and within MAX_GAP of it: book and model both favour it, the model genuinely
    more, without the wild disagreements that did worst in the 2026-10-01 audit."""
    g = gap(pick, fair_key)
    try:
        return (g is not None and g <= MAX_GAP
                and float(pick["p_model"]) >= float(pick[fair_key]) + MIN_BACKING)
    except (KeyError, TypeError, ValueError):
        return False


def qualifies(pick: dict, **kw) -> bool:
    """Owner's decision, 2026-10-12: the boards carry two kinds of pick.
      value    -- passes the price check: +3% at bet365. Eligible for the Bet list once
                  its market is proven.
      proving  -- not value at this price, but a real contender: the caller's chance bar
                  is met AND the model backs it (model_backs). Frozen and graded so the
                  market can prove itself; never offered as a bet.
    Anything else (a forecast weaker than the book, or far from it) is not published.
    Sets pick["pick_kind"]; returns True for either kind."""
    ok = check(pick, **kw)["ok"]
    if ok:
        pick["pick_kind"] = "value"
        return True
    if model_backs(pick, kw.get("fair_key", "book_p")):
        pick["pick_kind"] = "proving"
        return True
    return False


def untag_unpublished(priced: list, kept: list) -> None:
    """qualifies() tags pick_kind on every row it passes; a row the board's own bar
    then drops is NOT a proving pick, and must not be shown as one on a game page
    (MLB audit, 2026-10-18: seven unpublished rows read "Proving pick")."""
    keep = {id(p) for p in kept}
    for p in priced:
        if id(p) not in keep:
            p.pop("pick_kind", None)


def one_per_player(board: dict) -> dict:
    """ONE PICK PER PLAYER PER GAME across markets (audit, 2026-10-12): the same
    striker on scorer, shots and on target is one bet three times. Keeps the highest
    stated chance, then expected return -- ranking on expected return first would
    favour the pick where the forecast disagrees with the book most (third review).
    `board` is {market: [picks]}."""
    best = {}
    for market, picks in board.items():
        for p in picks or []:
            key = (str(p.get("game_id")), str(p.get("player_id") or p.get("player")))
            score = (p.get("probability") or 0, (p.get("price_check") or {}).get("ev") or -9)
            if key not in best or score > best[key][0]:
                best[key] = (score, market, id(p))
    keep = {v[2] for v in best.values()}
    for ps in board.values():
        for p in ps or []:
            if id(p) not in keep:
                p["on_board"] = False
                p["dropped"] = "another pick on the same player in this game is better"
    return {m: [p for p in ps or [] if id(p) in keep] for m, ps in board.items()}
