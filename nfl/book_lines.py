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
checked AT MOST THREE TIMES (owner, 2026-10-22): about 30 hours out, about 15
hours out, and on the pre-lock run ~2h50 before kickoff that the Worker starts
(cloudflare/worker.js nflPreLock), whose line is the one a pick freezes on. With
alternates a check is 7 credits: about 16 games x 3 x 7 = ~340 credits a week. Anytime TD is not asked for: it has
no line to replace, and bet365 already prices it.
"""
from __future__ import annotations

from datetime import datetime

from nfl import odds
from oddsapi import props as shared

SPORT_KEY = "americanfootball_nfl"
MARKETS = {"player_pass_yds": "passing_yards",
           "player_rush_yds": "rushing_yards",
           "player_reception_yds": "receiving_yards",
           # Pinnacle prices BOTH sides of anytime TD (checked 2026-10-04), so its
           # fair chance is the reference for bet365's one-sided TD price
           # (nfl/mispricing.py). +1 credit a check.
           "player_anytime_td": "anytime_touchdown"}
BOOK_ORDER = ("pinnacle", "draftkings", "fanduel")
BOOK_LABEL = {"pinnacle": "Pinnacle", "draftkings": "DraftKings", "fanduel": "FanDuel"}

# THREE CHECKS A GAME (owner, 2026-10-22). At 72 hours the first check caught 15
# lines for a Thursday game (books had posted almost nothing) and the board sat on
# them for two days. Check n is due once the game is within CHECK_HOURS[n]; the
# last is the PRE-LOCK check, taken on the run the Worker starts ~2h50 before
# kickoff (PRE_LOCK_MINUTES), so its line is on the board before the 2-hour lock
# window freezes it. A check that comes back empty is free and is not counted.
CHECK_HOURS = (30.0, 15.0, 3.0)
BOARD_HOURS = CHECK_HOURS[0]
PRE_LOCK_MINUTES = 170      # cloudflare/worker.js NFL_PRE_LOCK_MIN -- keep in step
MIN_LEAD_HOURS = 0.5        # never spend on a game about to kick off
MAX_CHECKS = len(CHECK_HOURS)

# When .github/workflows/nfl.yml is SCHEDULED to run (UTC): daily 05:00, 09:00
# and 16:00, plus Tuesday 08:30. 05:00 is the grading pass after the night games;
# no NFL game kicks off between 05:00 and 09:00, so it never takes a lock check. Used to recognise the LAST run before a kickoff,
# which takes the second (lock) check. Keep in step with the workflow's crons.
RUN_SLOTS = ((None, 5, 0), (None, 9, 0), (None, 16, 0), (1, 8, 30))   # (weekday Mon=0 or None, h, m)

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
    """"board", "lock" or None. Check n once the game is within CHECK_HOURS[n], an
    hour or more after the last; inside the final window the check is "lock", and
    only one is ever taken there (a later scheduled run must not spend again)."""
    kickoff, now = _utc(kickoff), _utc(now)
    hours = (kickoff - now).total_seconds() / 3600.0
    if hours <= MIN_LEAD_HOURS or len(checks) >= MAX_CHECKS:
        return None
    if any((kickoff - _utc(c)).total_seconds() / 3600.0 <= CHECK_HOURS[-1] for c in checks):
        return None
    if checks and (now - _utc(checks[-1])).total_seconds() < 3600:
        return None
    if hours <= CHECK_HOURS[-1]:
        return "lock"
    if hours <= CHECK_HOURS[len(checks)] and len(checks) < MAX_CHECKS - 1:
        return "board"
    return None


def kickoffs(games, now) -> list:
    """Upcoming kickoffs (ISO UTC) for the Worker's pre-lock run, soonest first."""
    now = _utc(now)
    out = set()
    for g in games or []:
        ko = g.get("kickoff")
        if ko and _utc(ko) > now:
            out.add(_utc(ko).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return sorted(out)


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
