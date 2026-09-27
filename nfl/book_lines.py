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

from datetime import datetime

from nfl import odds
from oddsapi import props as shared

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
    return shared.utc(value)


def next_scheduled_run(now: datetime) -> datetime:
    """The next time nfl.yml is scheduled to start, strictly after `now`."""
    return shared.next_scheduled_run(now, RUN_SLOTS)


def due(checks: list, kickoff, now) -> str | None:
    """"board", "lock" or None -- see oddsapi.props.due. At most MAX_CHECKS per
    game; the second only on the last scheduled run before kickoff."""
    return shared.due(checks, kickoff, now, RUN_SLOTS, BOARD_HOURS,
                      MIN_LEAD_HOURS, MAX_CHECKS)


def parse_event(payload) -> dict:
    """{market: {player: quote}} in the same shape as odds.player_props."""
    return shared.parse_event(payload, MARKETS, BOOK_ORDER, BOOK_LABEL)


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


# ALTERNATE LINES: the bookmaker's own ladder for each player ("40+ rushing
# yards at 1.17, 50+ at 1.32 ..."). Fetched in the same call as the main lines
# (3 more markets, so a check costs 6 credits), and used for one thing: the
# highest rung a player is still likely to clear -- a real, bettable line rather
# than one the model made up. Only DraftKings and FanDuel quote ladders; Pinnacle
# does not. Over-only, so each rung's price still carries the book's margin.
ALT_MARKETS = {"player_pass_yds_alternate": "passing_yards",
               "player_rush_yds_alternate": "rushing_yards",
               "player_reception_yds_alternate": "receiving_yards"}
ALT_BOOKS = ("draftkings", "fanduel")


def parse_alternates(payload) -> dict:
    return shared.parse_alternates(payload, ALT_MARKETS, ALT_BOOKS, BOOK_LABEL)
