"""Is this pick worth betting AT BET365, and how much?

The person using this app bets at bet365 (UK). So a pick is judged against the
price bet365 offers, never against a US sportsbook nobody here can use.

Every pick gets:
  min_price  the lowest bet365 price at which the bet is worth taking: the price
             where the pick's probability returns at least MIN_EV per unit.
             Check bet365; bet only at this price or better.
  ref_price  the price the verdict is judged at -- bet365's own quote where we
             have it ("bet365"), otherwise an ESTIMATE of bet365's price built from
             the sharp books' fair probability less a typical bet365 margin
             ("estimate"). Estimated verdicts are marked as such on the page.
  value      True when the pick clears MIN_EV at ref_price.
  stake      suggested stake, % of bankroll: a quarter of the Kelly fraction at
             ref_price, capped at MAX_STAKE. Zero when it is not value.

Why quarter-Kelly and a cap: full Kelly assumes the probability is exactly right.
It is not -- Phase 1 showed the props running behind their stated chances -- and
over-staking on an overstated edge is how a bankroll goes to zero. A quarter of
Kelly keeps most of the growth with a fraction of the swings.
"""
from __future__ import annotations

MIN_EV = 0.03            # at least +3% expected return per unit staked
BET365_MARGIN = 0.05     # assumed per-side margin where bet365's price is unknown
KELLY_FRACTION = 0.25
MAX_STAKE = 0.02         # never more than 2% of the bankroll on one pick
MIN_STAKE = 0.0025       # below a quarter of a percent, round to "not worth it"


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x else None          # NaN -> None


def stake(p: float, price: float) -> float:
    """Quarter-Kelly stake as a fraction of bankroll, capped, in 0.25% steps."""
    if price is None or price <= 1.0:
        return 0.0
    kelly = (p * price - 1.0) / (price - 1.0)
    s = min(MAX_STAKE, max(0.0, KELLY_FRACTION * kelly))
    s = round(s / MIN_STAKE) * MIN_STAKE
    return round(s, 4) if s >= MIN_STAKE else 0.0


def assess(p, fair_p=None, price=None, price_is_bet365=False) -> dict | None:
    """The value verdict for one pick. None when it has no probability."""
    p, fair_p, price = _f(p), _f(fair_p), _f(price)
    if p is None or not 0.0 < p < 1.0:
        return None
    out = {"min_price": round((1.0 + MIN_EV) / p, 2)}
    if price_is_bet365 and price and price > 1.0:
        ref, basis = price, "bet365"
    elif fair_p and 0.0 < fair_p < 1.0:
        ref, basis = 1.0 / (fair_p * (1.0 + BET365_MARGIN)), "estimate"
    else:
        out.update(basis=None, value=None, stake=0.0)
        return out
    ev = p * ref - 1.0
    good = ev >= MIN_EV
    out.update(basis=basis, ref_price=round(ref, 2), ev=round(ev, 4), value=good,
               stake=stake(p, ref) if good else 0.0)
    return out


def _is_bet365(pick: dict) -> bool:
    return any(str(pick.get(k) or "").lower() == "bet365" for k in ("line_source", "book"))


def annotate_line_board(payload: dict) -> int:
    """NFL / NBA / MLB boards: props carry probability, book_p (the fair chance of
    the side taken) and book_price; winners carry p_pick and book_p_pick."""
    n = 0
    for g in payload.get("games") or []:
        v = assess(g.get("p_pick"), g.get("book_p_pick"),
                   g.get("book_price"), _is_bet365(g))
        g["value"] = v
        n += bool(v and v.get("value"))
    for block in (payload.get("props") or {}).values():
        for p in (block or {}).get("picks") or []:
            v = assess(p.get("probability"), p.get("book_p"), p.get("book_price"), _is_bet365(p))
            p["value"] = v
            n += bool(v and v.get("value"))
    return n


def annotate_soccer_match(item: dict) -> dict | None:
    """A Best Picks entry: the price shown is bet365's where `book` says so."""
    side = item.get("pick_type")
    fair = ((item.get("market") or {}).get(f"p_{side}")) if side else None
    return assess(item.get("p_pick"), fair, item.get("odds"),
                  str(item.get("book") or "").lower() == "bet365")


def annotate_soccer_player(pick: dict) -> dict | None:
    """A player pick: bet365's anytime price where fetched. Shots and shots on
    target have no bet365 market in any feed; they carry a US book's fair chance
    (leagues/shot_odds.py), from which the bet365 price is ESTIMATED."""
    b365 = str(pick.get("book") or "").lower() == "bet365"
    return assess(pick.get("p_pick"), None if b365 else pick.get("book_p"),
                  pick.get("book_price"), b365)


def annotate_game_props(game_props: dict) -> int:
    """Value verdicts for every pick on the game pages."""
    return sum(annotate_line_board({"props": {m: {"picks": ps} for m, ps in (markets or {}).items()}})
               for markets in (game_props or {}).values())
