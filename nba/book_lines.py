"""NBA bookmaker lines and match odds from The Odds API.

EVERY NBA PROP STANDS ON A BOOKMAKER'S LINE (owner's instruction, 2026-09-27).
The model is trained across a spread of lines (nba.features.augment_lines) so it
can be asked about the book's number and mean it; a player no book quotes is not
published at all.

BUDGET (CLAUDE.md, "API credit budget"):
  * Props: points, rebounds, assists, threes from Pinnacle / DraftKings / FanDuel
    -- three named books are one region -- so 4 credits a check, at most TWO
    checks a game (once within BOARD_HOURS, once on the last scheduled run before
    tip-off). ~7 games a night -> about 55 credits a day in season.
  * Match odds: ONE /odds call per run covers every game (h2h, 1 credit).
"""
from __future__ import annotations

from oddsapi import props as shared

SPORT_KEY = "basketball_nba"
MARKETS = {"player_points": "points", "player_rebounds": "rebounds",
           "player_assists": "assists", "player_threes": "threes"}
BOOK_ORDER = ("pinnacle", "draftkings", "fanduel")
BOOK_LABEL = {"pinnacle": "Pinnacle", "draftkings": "DraftKings", "fanduel": "FanDuel"}

# NBA props post on the day of the game, so the first check opens 30h out.
BOARD_HOURS = 30.0
MIN_LEAD_HOURS = 0.5
MAX_CHECKS = 2

# When .github/workflows/nba.yml is SCHEDULED (UTC). 21:30 is the last run before
# the usual 23:00-03:30 UTC tip-offs; 15:00 covers weekend matinees. Keep in step
# with the workflow's crons.
RUN_SLOTS = ((None, 15, 0), (None, 21, 30))

# The Odds API's team names -> the tricodes the NBA's own feeds use.
TEAM_CODES = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def due(checks: list, tipoff, now) -> str | None:
    return shared.due(checks, tipoff, now, RUN_SLOTS, BOARD_HOURS,
                      MIN_LEAD_HOURS, MAX_CHECKS)


def is_last_run_before(tipoff, now) -> bool:
    return shared.is_last_run_before(tipoff, now, RUN_SLOTS, MIN_LEAD_HOURS)


def parse_event(payload) -> dict:
    return shared.parse_event(payload, MARKETS, BOOK_ORDER, BOOK_LABEL)
