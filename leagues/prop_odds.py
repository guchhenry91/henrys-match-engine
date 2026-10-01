"""bet365's anytime-goalscorer prices for the soccer player picks.

Fetched from API-Football's own odds (bookmaker 8 = bet365, bet id 92 "Anytime
Goal Scorer"), so it costs nothing on The Odds API and is priced by the UK book
the site already uses for match odds. Written by scripts/sync_soccer_prop_odds.py
to data-raw/leagues/prop_odds.json and attached to the upcoming picks here.

ONLY THE ANYTIME SCORER MARKET. Checked on 2026-09-27 (scripts/
probe_soccer_player_odds.py): bet365 prices every player in both squads for id
92. Its "Away Player Shots On Target Total" (id 275) is NOT usable -- it lists
both teams and priced Gyokeres at 9.50, which cannot be "1+ shot on target" --
and there is no player shot-attempts market at all. Those two markets stay
unpriced rather than carrying a number that means something else.

THE IMPLIED PROBABILITY IS RAW (1/odds). A scorer market is one-sided, so its
margin cannot be removed by pairing sides; the raw figure overstates the book's
true view, which makes the edge UNDERSTATED rather than flattered.
"""
from __future__ import annotations

import json
from pathlib import Path

from nfl.odds import norm_name

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data-raw" / "leagues" / "prop_odds.json"
BET365 = 8
ANYTIME_BET = 92
# A price this long is a defender's lottery ticket; the model's picks never live
# out there, and a bad name join would land exactly there.
MAX_ODDS = 21.0


def fixture_key(league_key: str, match_id) -> str:
    return f"{league_key}#{int(match_id)}"


def parse_anytime(rows) -> dict:
    """{bet365 player name: decimal odds} from one API-Football /odds response."""
    out = {}
    for r in rows or []:
        for bm in r.get("bookmakers") or []:
            if int(bm.get("id", -1)) != BET365:
                continue
            for bet in bm.get("bets") or []:
                if int(bet.get("id", -1)) != ANYTIME_BET:
                    continue
                for v in bet.get("values") or []:
                    try:
                        odd = float(v.get("odd"))
                    except (TypeError, ValueError):
                        continue
                    name = str(v.get("value") or "").strip()
                    if name and odd > 1.0 and name.lower() != "no goalscorer":
                        out[name] = odd
    return out


def load() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def price_for(league_key: str, match_id, player: str, store: dict | None = None):
    """bet365's anytime-scorer price for one player in one fixture, or None."""
    fixtures = (store if store is not None else load()).get("fixtures") or {}
    entry = fixtures.get(fixture_key(league_key, match_id)) or {}
    want = norm_name(player)
    hits = [odd for name, odd in (entry.get("anytime") or {}).items() if norm_name(name) == want]
    return hits[0] if len(hits) == 1 and hits[0] <= MAX_ODDS else None


def attach(upcoming: list, store: dict | None = None) -> int:
    """Put bet365's price on each upcoming anytime-scorer pick. Returns count."""
    fixtures = (store if store is not None else load()).get("fixtures") or {}
    priced = 0
    for pick in upcoming:
        if pick.get("market") != "goal":
            continue
        entry = fixtures.get(fixture_key(pick["league_key"], pick["id"])) or {}
        prices = entry.get("anytime") or {}
        want = norm_name(pick.get("player"))
        hits = [odd for name, odd in prices.items() if norm_name(name) == want]
        if len(hits) != 1 or hits[0] > MAX_ODDS:
            continue                      # absent, ambiguous or implausible: unpriced
        odd = hits[0]
        pick["book"] = "bet365"
        pick["book_price"] = odd
        pick["book_p"] = round(1.0 / odd, 4)
        pick["edge"] = round(float(pick["p_pick"]) - 1.0 / odd, 4)
        pick["book_checked_at"] = entry.get("checked_at")
        priced += 1
    return priced
