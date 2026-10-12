"""THE PRICE CHECK: a pick is published only if it is worth betting at bet365.

Owner's decision, 2026-10-12, after the four-reviewer audit: "winning picks" and
"making money" only meet when the price is right. A pick that wins 70% of the time at
1.30 loses money in the long run (it needs 77% to break even). So every pick, in every
sport, now carries:

  ref_price         the price it is judged at: bet365's own quote where we have it,
                    otherwise an ESTIMATE of bet365's price from another book's fair
                    chance less a typical bet365 margin (tracking/value.py) -- marked
                    "estimate" on the page;
  break_even_price  the lowest odds at which the pick's stated chance breaks even;
  ev                the expected return per unit staked at ref_price;
  ok                ev >= MIN_EV.

A pick with ok=False is NOT published on the board (it stays on the game page,
flagged short_price). A pick with no price at all cannot be checked and is not
published either -- except where the caller lets it wait for a price (soccer shots,
whose price is fetched only for picks already on the board) without ever locking it.
"""
from __future__ import annotations

from tracking import value

MIN_EV = 0.0             # break-even or better at the reference price


def check(pick: dict, prob_key: str = "probability", fair_key: str = "book_p",
          price_key: str = "book_price", is_bet365: bool | None = None) -> dict:
    """Annotate `pick` with price_check and return it."""
    p = pick.get(prob_key)
    b365 = value._is_bet365(pick) if is_bet365 is None else is_bet365
    v = value.assess(p, None if b365 else pick.get(fair_key), pick.get(price_key), b365)
    try:
        be = round(1.0 / float(p), 2)
    except (TypeError, ValueError, ZeroDivisionError):
        be = None
    if not v or v.get("ev") is None:
        out = {"ok": False, "reason": "no price", "break_even_price": be}
    else:
        out = {"ok": v["ev"] >= MIN_EV, "ref_price": v["ref_price"], "basis": v["basis"],
               "ev": v["ev"], "break_even_price": be,
               "reason": None if v["ev"] >= MIN_EV else "price too short"}
    pick["price_check"] = out
    if not out["ok"]:
        pick["short_price"] = out.get("reason") == "price too short"
    return out


def passes(pick: dict, **kw) -> bool:
    return check(pick, **kw)["ok"]


def one_per_player(board: dict) -> dict:
    """ONE PICK PER PLAYER PER GAME across markets (audit, 2026-10-12): the same
    striker on scorer, shots and on target is one bet three times. Keeps the highest
    expected return, then the highest chance. `board` is {market: [picks]}."""
    best = {}
    for market, picks in board.items():
        for p in picks or []:
            key = (str(p.get("game_id")), str(p.get("player_id") or p.get("player")))
            score = ((p.get("price_check") or {}).get("ev") or -9, p.get("probability") or 0)
            if key not in best or score > best[key][0]:
                best[key] = (score, market, id(p))
    keep = {v[2] for v in best.values()}
    for ps in board.values():
        for p in ps or []:
            if id(p) not in keep:
                p["on_board"] = False
                p["dropped"] = "another pick on the same player in this game is better"
    return {m: [p for p in ps or [] if id(p) in keep] for m, ps in board.items()}
