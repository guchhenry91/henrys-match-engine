"""MLB bookmaker lines, team totals and match odds from The Odds API.

Every MLB prop stands on a BOOKMAKER'S line (Pinnacle > DraftKings > FanDuel);
a player no book quotes is not priced. Budget (CLAUDE.md): one call per check
carries all six markets from three named books (one region-equivalent), so a
check costs at most 6 credits, at most two checks a game; match odds are one
1-credit /odds call per run for the whole slate.
"""
from __future__ import annotations

from nfl import odds
from oddsapi import props as shared

SPORT_KEY = "baseball_mlb"
MARKETS = {"batter_hits": "hits", "batter_hits_runs_rbis": "hrr",
           "batter_home_runs": "hr", "batter_rbis": "rbi",
           "pitcher_strikeouts": "strikeouts"}
TEAM_TOTALS = "team_totals"
ALL_MARKETS = list(MARKETS) + [TEAM_TOTALS]
BOOK_ORDER = ("pinnacle", "draftkings", "fanduel")
BOOK_LABEL = {"pinnacle": "Pinnacle", "draftkings": "DraftKings", "fanduel": "FanDuel"}

BOARD_HOURS = 30.0
MIN_LEAD_HOURS = 0.5
MAX_CHECKS = 2
# When .github/workflows/mlb.yml is SCHEDULED (UTC). Keep in step with its crons.
RUN_SLOTS = ((None, 14, 0), (None, 17, 30), (None, 22, 0))
ONE_SIDED_OVERROUND = 1.07     # a one-sided "Yes" price (home runs) carries its margin

TEAM_CODES = {
    "Los Angeles Angels": "ANA", "Arizona Diamondbacks": "ARI", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHN", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCA", "Los Angeles Dodgers": "LAN",
    "Washington Nationals": "WAS", "New York Mets": "NYN", "Athletics": "ATH",
    "Oakland Athletics": "ATH", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDN",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SFN", "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Minnesota Twins": "MIN", "Philadelphia Phillies": "PHI", "Atlanta Braves": "ATL",
    "Chicago White Sox": "CHA", "Miami Marlins": "MIA", "New York Yankees": "NYA",
    "Milwaukee Brewers": "MIL",
}


def due(checks, start, now):
    return shared.due(checks, start, now, RUN_SLOTS, BOARD_HOURS, MIN_LEAD_HOURS, MAX_CHECKS)


def is_last_run_before(start, now):
    return shared.is_last_run_before(start, now, RUN_SLOTS, MIN_LEAD_HOURS)


def parse_event(payload) -> dict:
    """{market: {player: quote}}. Two-sided markets are de-vigged by the shared
    parser; home runs are often quoted over-only, so a lone over price is kept
    with its margin estimated (ONE_SIDED_OVERROUND) and flagged `fair: False`."""
    out = shared.parse_event(payload, MARKETS, BOOK_ORDER, BOOK_LABEL)
    books = {b.get("key"): b for b in (payload or {}).get("bookmakers") or []}
    hr = out.setdefault("hr", {})
    for key in BOOK_ORDER:
        for m in (books.get(key) or {}).get("markets") or []:
            if m.get("key") != "batter_home_runs":
                continue
            for o in m.get("outcomes") or []:
                name = str(o.get("description") or "").strip()
                prob = odds.decimal_to_prob(o.get("price"))
                if (str(o.get("name") or "").lower() != "over" or not name or prob is None
                        or float(o.get("point") or 0.5) != 0.5
                        or odds.match_player(hr, name) is not None):
                    continue
                hr[name] = {"_name": name, "line": 0.5,
                            "over": round(min(prob / ONE_SIDED_OVERROUND, 0.99), 4),
                            "odd_over": float(o["price"]), "book": BOOK_LABEL[key],
                            "source": key, "fair": False}
    return {m: q for m, q in out.items() if q}


def parse_team_totals(payload, home_name, away_name) -> dict:
    """{"home"|"away": quote} from the team_totals market, de-vigged per team."""
    books = {b.get("key"): b for b in (payload or {}).get("bookmakers") or []}
    out = {}
    for key in BOOK_ORDER:
        for m in (books.get(key) or {}).get("markets") or []:
            if m.get("key") != TEAM_TOTALS:
                continue
            sides = {}
            for o in m.get("outcomes") or []:
                team = o.get("description")
                prob = odds.decimal_to_prob(o.get("price"))
                if prob is None or o.get("point") is None:
                    continue
                sides.setdefault((team, float(o["point"])), {})[str(o.get("name")).lower()] = (
                    prob, float(o["price"]))
            for (team, line), pair in sides.items():
                side = "home" if team == home_name else "away" if team == away_name else None
                if side is None or side in out or "over" not in pair or "under" not in pair:
                    continue
                fair = odds.devig({"over": pair["over"][0], "under": pair["under"][0]})
                out[side] = {"line": line, "over": round(fair["over"], 4),
                             "under": round(fair["under"], 4),
                             "odd_over": pair["over"][1], "odd_under": pair["under"][1],
                             "book": BOOK_LABEL[key], "source": key}
    return out
