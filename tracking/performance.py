"""Does a market make money? Profit, return and closing-line value per market.

A hit rate cannot answer that. 64% on favourites priced at 70% loses money, and 12%
on home runs priced at 10% wins it. This module scores every graded pick at the
PRICE IT WAS FROZEN AT, one unit staked, and -- where the first price seen is on
record -- whether the market then moved toward the pick (closing-line value).

Closing-line value is the faster signal. Profit needs about a thousand bets before
luck washes out; whether the line keeps moving your way shows up in a few hundred,
and it is the measure professional bettors use to know an edge is real.

What the two prices are here: OPEN is the first quote this app saw for that
player and line (the board check, when the pick first appears). CLOSE is the quote
the pick froze on (the lock check, the last run before the start). The app checks
each game at most twice to protect the Odds API budget, so CLOSE is the latest
price on record rather than the literal final price at first pitch.

A market's STATUS is earned, never assumed:
  proven    -- profitable AND the line moves its way more often than chance
               allows, over enough bets (or a profit too large to be luck)
  promising -- ahead on profit or on closing-line value, not yet enough bets
  losing    -- enough bets to say it is behind, and no closing-line edge
  testing   -- too few priced bets to judge either way
"""
from __future__ import annotations

import math

MIN_JUDGE = 50          # fewer priced bets than this: "testing", nothing more
MIN_PROVEN = 200        # a "proven" market needs at least this many
MIN_CLV_PROVEN = 100    # ...and this many open/close pairs behind its CLV
MIN_LOSING = 100        # "losing" needs this many before it is called
FLAT = 0.005            # a move under half a point is noise, not a move
Z = 2.0                 # ~95%: how far from chance a result must be to count


def side_quote(quote: dict | None, side: str):
    """(fair probability, decimal price) of `side` in a stored over/under quote."""
    if not quote:
        return None, None
    side = side or "over"
    p = quote.get(side)
    price = quote.get(f"odd_{side}")
    return (float(p) if p is not None else None,
            float(price) if price else None)


def _checked_at_or_before(checks, when):
    """The latest check time at or before `when` (ISO strings sort as times)."""
    when = str(when or "")
    done = [c for c in (checks or []) if str(c) <= when]
    return max(done) if done else None


def stamp_prop(entry: dict, game: dict | None, quote_for) -> bool:
    """Record the OPEN quote on a frozen prop, once. True if it stamped.

    `quote_for(block)` finds this pick's quote inside one {player: quote} block,
    so name matching stays each sport's own (book names differ from box scores).
    Stamped only while the entry has no "open" yet; a frozen record is never
    rewritten afterwards.
    """
    if "open" in entry or not game:
        return False
    market = entry.get("market")
    block = ((game.get("open") or {}).get(market)) or {}
    q = quote_for(block)
    if q is None:
        entry["open"] = None              # looked, nothing first-seen on record
        return True
    p, price = side_quote(q, entry.get("side"))
    entry["open"] = {"line": q.get("line"), "p": p, "price": price,
                     "book": q.get("book"), "at": q.get("at")}
    entry["close_at"] = _checked_at_or_before(game.get("checks"), entry.get("locked_at"))
    return True


def stamp_winner(entry: dict, game: dict | None, picked_side: str) -> bool:
    """Price and OPEN quote for a frozen team-winner pick, from the match odds."""
    if "open" in entry or not game or picked_side not in ("home", "away"):
        return False
    close = game.get("moneyline") or {}
    if entry.get("book_price") is None:
        raw = close.get(f"odd_{picked_side}")
        if raw:
            entry["book_price"] = float(raw)
        elif close.get(picked_side) and close.get("overround") is not None:
            # Older odds files kept only the fair probability and the margin; the
            # price is rebuilt by spreading the margin evenly. Marked as estimated.
            entry["book_price"] = round(
                1.0 / (float(close[picked_side]) * (1.0 + float(close["overround"]))), 3)
            entry["price_estimated"] = True
    op = (game.get("open") or {}).get("moneyline")
    entry["open"] = ({"p": op.get(picked_side), "price": op.get(f"odd_{picked_side}"),
                      "book": op.get("book"), "at": op.get("at")} if op else None)
    entry["close_p"] = close.get(picked_side)
    entry["close_at"] = close.get("at") or entry.get("locked_at")
    return True


def clv(entry: dict):
    """(direction, size) of the move from OPEN to CLOSE, from the pick's side.

    direction: "ours" (the market moved toward the pick), "against", "flat", or
    None when there is no usable pair. size: the change in the fair probability
    of the pick's side at an unchanged line, else None (a moved line is scored by
    direction only -- converting between lines would need the model, and this is
    meant to be the market's verdict, not the model's).
    """
    op = entry.get("open")
    if not op or op.get("p") is None:
        return None, None
    if op.get("at") and entry.get("close_at") and str(op["at"]) >= str(entry["close_at"]):
        return None, None                 # one check only: open IS close
    close_p = entry.get("close_p", entry.get("book_p"))
    if entry.get("kind") == "winner" or entry.get("market") is None:
        if close_p is None:
            return None, None
        d = float(close_p) - float(op["p"])
        return ("flat" if abs(d) < FLAT else "ours" if d > 0 else "against"), round(d, 4)
    line_o, line_c = op.get("line"), entry.get("line")
    if line_o is not None and line_c is not None and float(line_o) != float(line_c):
        up = float(line_c) > float(line_o)
        under = entry.get("side") == "under"
        return ("ours" if up != under else "against"), None
    if close_p is None:
        return None, None
    d = float(close_p) - float(op["p"])
    return ("flat" if abs(d) < FLAT else "ours" if d > 0 else "against"), round(d, 4)


def _won(entry: dict):
    g = entry.get("graded")
    if entry.get("void") or g == "void":
        return None
    if g == "correct":
        return True
    if g == "wrong":
        return False
    return None


def price_of(entry: dict):
    for key in ("book_price", "odds"):
        v = entry.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v > 1.0:
            return v
    return None


def summarize(entries) -> dict:
    """Profit at the frozen price, one unit a bet, plus closing-line value."""
    returns, prices = [], []
    wins = voids = 0
    moves = {"ours": 0, "against": 0, "flat": 0}
    sizes = []
    for e in entries:
        direction, size = clv(e)
        if direction:
            moves[direction] += 1
        if size is not None:
            sizes.append(size)
        won = _won(e)
        price = price_of(e)
        if won is None:
            if e.get("graded"):
                voids += 1
            continue
        if price is None:
            continue
        returns.append(price - 1.0 if won else -1.0)
        prices.append(price)
        wins += int(won)
    n = len(returns)
    profit = sum(returns)
    roi = profit / n if n else None
    se = (math.sqrt(sum((r - roi) ** 2 for r in returns) / (n - 1)) / math.sqrt(n)
          if n > 1 else None)
    moved = moves["ours"] + moves["against"]
    beat = moves["ours"] / moved if moved else None
    clv_z = ((moves["ours"] - moved / 2) / math.sqrt(moved / 4)) if moved else None
    out = {
        "bets": n, "wins": wins, "losses": n - wins, "voids": voids,
        "profit": round(profit, 2), "roi": round(roi, 4) if roi is not None else None,
        "roi_se": round(se, 4) if se is not None else None,
        "hit_rate": round(wins / n, 4) if n else None,
        # The hit rate these prices needed just to break even.
        "break_even": round(sum(1.0 / p for p in prices) / n, 4) if n else None,
        "avg_price": round(sum(prices) / n, 3) if n else None,
        "clv": {"pairs": moved + moves["flat"], "ours": moves["ours"],
                "against": moves["against"], "flat": moves["flat"],
                "beat_rate": round(beat, 4) if beat is not None else None,
                "avg_move": round(sum(sizes) / len(sizes), 4) if sizes else None},
    }
    out["status"], out["status_note"] = status(out, clv_z)
    return out


def status(s: dict, clv_z=None):
    n, roi, se = s["bets"], s["roi"], s["roi_se"]
    pairs = s["clv"]["ours"] + s["clv"]["against"]
    beat = s["clv"]["beat_rate"]
    roi_z = (roi / se) if (roi is not None and se) else None
    if n < MIN_JUDGE:
        return "testing", f"{n} priced bets so far; {MIN_JUDGE} needed before any verdict"
    if n >= MIN_PROVEN and roi > 0 and (
            (pairs >= MIN_CLV_PROVEN and clv_z is not None and clv_z >= Z)
            or (roi_z is not None and roi_z >= Z)):
        return "proven", "Profitable, and the evidence is beyond what luck explains"
    if n >= MIN_LOSING and roi < 0 and (beat is None or pairs < 30 or beat <= 0.5):
        return "losing", "Behind at the prices taken, with no closing-line edge: do not bet"
    if roi > 0 or (beat is not None and pairs >= 30 and beat > 0.5):
        return "promising", "Ahead so far, but not yet enough evidence to trust"
    return "testing", "Behind so far, but not yet enough bets to call it"


def by_market(entries, market_of) -> dict:
    """{market: summary} plus "all", for one sport's graded entries."""
    groups = {}
    for e in entries:
        groups.setdefault(market_of(e), []).append(e)
    out = {m: summarize(rows) for m, rows in groups.items() if m}
    out["all"] = summarize(entries)
    return out


# --- per-sport wiring ----------------------------------------------------------------

def _read_games(path) -> dict:
    import json
    from pathlib import Path
    try:
        return (json.loads(Path(path).read_text(encoding="utf-8")).get("games")) or {}
    except Exception:
        return {}


def stamp_line_log(log: dict, odds_path) -> int:
    """NBA and MLB (one pick-log shape, nba/picks.py): stamp OPEN quotes and
    winner prices from that sport's Odds API file. Returns how many stamped."""
    from nfl.odds import match_player
    games = _read_games(odds_path)
    n = 0
    for entry in log.values():
        if not isinstance(entry, dict) or "open" in entry:
            continue
        game = games.get(str(entry.get("game_id")))
        if game is None:
            continue
        if entry.get("kind") == "winner":
            side = ("home" if entry.get("pick") == entry.get("home")
                    else "away" if entry.get("pick") == entry.get("away") else None)
            n += stamp_winner(entry, game, side)
        elif entry.get("market") == "team_runs":
            side = "home" if entry.get("team") == game.get("home") else "away"
            # Team totals are kept per side ("home"/"away"), not per player.
            totals = {"team_runs": (game.get("open") or {}).get("team_totals") or {}}
            n += stamp_prop(entry, dict(game, open=totals), lambda block, s=side: block.get(s))
        elif entry.get("kind") == "prop":
            n += stamp_prop(entry, game, lambda block, e=entry: match_player(block, e.get("player")))
    return n


def stamp_nfl_log(props: dict, odds_path) -> int:
    """NFL props: the Odds API file is keyed "HOME|AWAY"; the pick knows its team
    and opponent, so the game is found by that pair."""
    from nfl.odds import match_player
    games = _read_games(odds_path)
    by_pair = {frozenset(k.split("|")): g for k, g in games.items() if "|" in k}
    n = 0
    for key, entry in props.items():
        if key.startswith("_") or not isinstance(entry, dict) or "open" in entry:
            continue
        game = by_pair.get(frozenset((entry.get("team"), entry.get("away"))))
        if game is None:
            continue
        n += stamp_prop(entry, game, lambda block, e=entry: match_player(block, e.get("player")))
    return n


def market_of(entry: dict) -> str:
    if entry.get("kind") == "winner" or (entry.get("pick") and not entry.get("market")):
        return "team_winner"
    return entry.get("market")
