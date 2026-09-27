"""Sport-agnostic helpers for player-prop LINES from The Odds API.

Shared by the NFL gap-fill (nfl/book_lines.py) and the NBA board
(nba/book_lines.py), so the two-checks-per-game rule and the parsing are one
tested implementation rather than two that drift apart.

THE SPEND RULE (CLAUDE.md, "API credit budget"): a game is checked at most
`max_checks` times -- first when it comes within `board_hours`, then once more
on the LAST scheduled run before it starts, whose line is the one a pick freezes
on. A reply with no lines costs nothing and is not counted by the callers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nfl import odds


def utc(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def next_scheduled_run(now, run_slots) -> datetime:
    """The next scheduled start strictly after `now`.

    `run_slots` is ((weekday Mon=0 or None, hour, minute), ...) in UTC."""
    now = utc(now)
    best = None
    for day in range(0, 9):
        base = (now + timedelta(days=day)).replace(second=0, microsecond=0)
        for weekday, hour, minute in run_slots:
            slot = base.replace(hour=hour, minute=minute)
            if weekday is not None and slot.weekday() != weekday:
                continue
            if slot > now and (best is None or slot < best):
                best = slot
    return best


def is_last_run_before(start, now, run_slots, lead_hours=0.5) -> bool:
    """True when no later scheduled run comes before `start` (less a lead)."""
    return next_scheduled_run(now, run_slots) >= utc(start) - timedelta(hours=lead_hours)


def due(checks: list, start, now, run_slots, board_hours: float,
        min_lead_hours: float = 0.5, max_checks: int = 2) -> str | None:
    """"board", "lock" or None: whether this run should spend on this game."""
    start, now = utc(start), utc(now)
    hours = (start - now).total_seconds() / 3600.0
    if hours <= min_lead_hours or hours > board_hours or len(checks) >= max_checks:
        return None
    if not checks:
        return "board"
    if (now - utc(checks[-1])) < timedelta(hours=1):
        return None
    if is_last_run_before(start, now, run_slots, min_lead_hours):
        return "lock"
    return None


def parse_event(payload, market_map: dict, book_order, book_label: dict) -> dict:
    """{our market: {player: quote}} from one /events/{id}/odds reply.

    Both sides at one line are required, so the pair de-vigs into the book's fair
    over probability; a one-sided quote is refused. Where a book lists alternates
    the line nearest an even price is its main line. Books are read in
    `book_order` and the first to quote a player wins."""
    out = {}
    books = {b.get("key"): b for b in (payload or {}).get("bookmakers") or []}
    for key in book_order:
        book = books.get(key)
        if not book:
            continue
        for market_row in book.get("markets") or []:
            market = market_map.get(market_row.get("key"))
            if not market:
                continue
            sides = {}
            for o in market_row.get("outcomes") or []:
                name = str(o.get("description") or "").strip()
                side = str(o.get("name") or "").lower()
                prob = odds.decimal_to_prob(o.get("price"))
                if not name or side not in ("over", "under") or prob is None \
                        or o.get("point") is None:
                    continue
                sides.setdefault((name, float(o["point"])), {})[side] = (prob, float(o["price"]))
            best = {}
            for (name, line), pair in sides.items():
                if "over" not in pair or "under" not in pair:
                    continue
                fair = odds.devig({"over": pair["over"][0], "under": pair["under"][0]})
                quote = {"_name": name, "line": line, "over": round(fair["over"], 4),
                         "under": round(fair["under"], 4), "odd_over": pair["over"][1],
                         "odd_under": pair["under"][1], "book": book_label.get(key, key),
                         "source": key}
                held = best.get(name)
                if held is None or abs(quote["over"] - 0.5) < abs(held["over"] - 0.5):
                    best[name] = quote
            quotes = out.setdefault(market, {})
            for name, quote in best.items():
                if odds.match_player(quotes, name) is None:
                    quotes[name] = quote
    return {m: q for m, q in out.items() if q}


def parse_moneyline(event, book_order, home_name, away_name) -> dict | None:
    """De-vigged home/away win probability from the first book that prices both."""
    books = {b.get("key"): b for b in (event or {}).get("bookmakers") or []}
    for key in book_order:
        for market in (books.get(key) or {}).get("markets") or []:
            if market.get("key") != "h2h":
                continue
            prices = {o.get("name"): odds.decimal_to_prob(o.get("price"))
                      for o in market.get("outcomes") or []}
            h, a = prices.get(home_name), prices.get(away_name)
            if h is None or a is None:
                continue
            fair = odds.devig({"home": h, "away": a})
            return {"book": key, "home": round(fair["home"], 4),
                    "away": round(fair["away"], 4),
                    "overround": round(h + a - 1.0, 4)}
    return None
