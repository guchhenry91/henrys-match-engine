"""NFL prop LINES from The Odds API, filling what bet365 (via API-NFL) leaves out.

WHY. The board asks the model about the bookmaker's line wherever one exists
(see nfl/odds.py). API-NFL carries bet365's passing and rushing lines for some
players and NO book's receiving yards at all -- on 2026-09-26 only 12 of 73
yardage picks had a book line. The coverage probe (scripts/probe_odds_api.py)
found DraftKings, FanDuel and Pinnacle quoting all three yardage markets for
every NFL game, receiving yards included.

PRIORITY: bet365 first (it is free, on the API-NFL key we already pay for), then
Pinnacle (the sharpest price, so the most honest line to be measured against),
then DraftKings, then FanDuel. A player bet365 quotes is never overwritten.

THE BUDGET (CLAUDE.md, "API credit budget"): three markets from three named books
is ONE region-equivalent, so a game costs 3 credits per check, and each game is
checked AT MOST TWICE -- once when it comes within BOARD_HOURS, and once on the
last scheduled NFL run before kickoff, whose line is the one a pick freezes on.
About 16 games x 2 x 3 = ~100 credits a week. Anytime TD is not asked for: it has
no line to replace, and bet365 already prices it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nfl import odds

SPORT_KEY = "americanfootball_nfl"
MARKETS = {"player_pass_yds": "passing_yards",
           "player_rush_yds": "rushing_yards",
           "player_reception_yds": "receiving_yards"}
BOOK_ORDER = ("pinnacle", "draftkings", "fanduel")
BOOK_LABEL = {"pinnacle": "Pinnacle", "draftkings": "DraftKings", "fanduel": "FanDuel"}

# First check once a game is this close. Wide enough that a Thursday game's
# props reach the board days ahead -- the board now publishes ONLY book lines,
# so an unchecked game shows no yardage picks at all. Widening costs nothing:
# the cap is two PAID checks a game wherever they fall, and a check that comes
# back empty (lines not posted yet) is free and is not counted.
BOARD_HOURS = 72.0
MIN_LEAD_HOURS = 0.5        # never spend on a game about to kick off
MAX_CHECKS = 2

# When .github/workflows/nfl.yml is SCHEDULED to run (UTC): daily 09:00 and
# 16:00, plus Tuesday 08:30. Used to recognise the LAST run before a kickoff,
# which takes the second (lock) check. Keep in step with the workflow's crons.
RUN_SLOTS = ((None, 9, 0), (None, 16, 0), (1, 8, 30))   # (weekday Mon=0 or None, h, m)

# The Odds API's team names -> the nflverse codes the board uses.
TEAM_CODES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO",
    "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def _utc(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def next_scheduled_run(now: datetime) -> datetime:
    """The next time nfl.yml is scheduled to start, strictly after `now`."""
    now = _utc(now)
    best = None
    for day in range(0, 9):
        base = (now + timedelta(days=day)).replace(second=0, microsecond=0)
        for weekday, hour, minute in RUN_SLOTS:
            slot = base.replace(hour=hour, minute=minute)
            if weekday is not None and slot.weekday() != weekday:
                continue
            if slot > now and (best is None or slot < best):
                best = slot
    return best


def due(checks: list, kickoff, now) -> str | None:
    """"board", "lock" or None: whether this run should spend on this game.

    At most MAX_CHECKS per game, ever. The second is taken only on the last
    scheduled run before kickoff, so the line a pick freezes against is the
    freshest one the budget allows."""
    kickoff, now = _utc(kickoff), _utc(now)
    hours = (kickoff - now).total_seconds() / 3600.0
    if hours <= MIN_LEAD_HOURS or hours > BOARD_HOURS or len(checks) >= MAX_CHECKS:
        return None
    if not checks:
        return "board"
    last = _utc(checks[-1])
    if (now - last) < timedelta(hours=1):
        return None
    if next_scheduled_run(now) >= kickoff - timedelta(hours=MIN_LEAD_HOURS):
        return "lock"
    return None


def parse_event(payload) -> dict:
    """{market: {player: quote}} in the same shape as odds.player_props.

    Both sides at the same line are required, so the pair de-vigs into the
    book's fair over probability; a one-sided quote is refused. Where a book
    lists alternates, the line nearest an even price is its main line. Books are
    read in BOOK_ORDER and the first to quote a player wins."""
    out = {}
    books = {b.get("key"): b for b in (payload or {}).get("bookmakers") or []}
    for key in BOOK_ORDER:
        book = books.get(key)
        if not book:
            continue
        for market_row in book.get("markets") or []:
            market = MARKETS.get(market_row.get("key"))
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
                         "odd_under": pair["under"][1], "book": BOOK_LABEL[key],
                         "source": key}
                held = best.get(name)
                if held is None or abs(quote["over"] - 0.5) < abs(held["over"] - 0.5):
                    best[name] = quote
            quotes = out.setdefault(market, {})
            for name, quote in best.items():
                if odds.match_player(quotes, name) is None:
                    quotes[name] = quote
    return {m: q for m, q in out.items() if q}


def merge(primary: dict, extra: dict) -> dict:
    """bet365's quotes, with a gap-fill book added ONLY for players bet365 does
    not quote. Both are {"HOME|AWAY": {market: {player: quote}}}."""
    merged = {g: {m: dict(q) for m, q in (markets or {}).items()}
              for g, markets in (primary or {}).items()}
    for game, markets in (extra or {}).items():
        for market, quotes in (markets or {}).items():
            have = merged.setdefault(game, {}).setdefault(market, {})
            for name, quote in quotes.items():
                if odds.match_player(have, name) is None:
                    have[name] = quote
    return merged
