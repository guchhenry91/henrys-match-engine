"""Bookmaker prices for the soccer SHOTS and SHOTS-ON-TARGET player picks.

bet365's own shots markets are not in any feed this app reads (leagues/prop_odds.py
records why API-Football's are unusable), so these come from The Odds API, whose
soccer player props cover the five leagues from US books only. They are a real
bookmaker's price on the SAME line the pick is on -- 2+ shots is "over 1.5"
player_shots, 1+ on target is "over 0.5" player_shots_on_target -- which is enough
to score the record at a price and for closing-line value. At bet365 the value
verdict is an ESTIMATE from this fair price (tracking/value.assess), labelled so.

Written by scripts/sync_soccer_shot_odds.py to data-raw/leagues/shot_odds.json.
Pure stdlib, so the fast lock path can read it without the model stack.
"""
from __future__ import annotations

import json
from pathlib import Path

from nfl.odds import decimal_to_prob, devig, norm_name

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "data-raw" / "leagues" / "shot_odds.json"

SPORT_KEYS = {"PL": "soccer_epl", "LALIGA": "soccer_spain_la_liga",
              "BUNDESLIGA": "soccer_germany_bundesliga", "LIGUE1": "soccer_france_ligue_one",
              "SERIEA": "soccer_italy_serie_a"}
MARKETS = {"player_shots": "shots", "player_shots_on_target": "sot"}
LINE = {"shots": 1.5, "sot": 0.5}            # 2+ shots, 1+ on target
BOOK_ORDER = ("draftkings", "fanduel", "betmgm", "betrivers", "williamhill_us")
BOOK_LABEL = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
              "betrivers": "BetRivers", "williamhill_us": "Caesars"}
# A lone "over" price carries the whole margin; spread it as the MLB home-run
# board does (mlb/book_lines.ONE_SIDED_OVERROUND) and flag the quote fair: False.
ONE_SIDED_OVERROUND = 1.07


def fixture_key(league_key: str, match_id) -> str:
    return f"{league_key}#{int(match_id)}"


def parse(payload) -> dict:
    """{"shots"|"sot": {player: quote}} from one /events/{id}/odds reply, at the
    pick's own line only. A book quoting both sides wins over a one-sided one;
    otherwise books are read in BOOK_ORDER."""
    books = {b.get("key"): b for b in (payload or {}).get("bookmakers") or []}
    found = {}                    # (market, player) -> [(two_sided, rank, quote)]
    for rank, key in enumerate(BOOK_ORDER):
        for m in (books.get(key) or {}).get("markets") or []:
            market = MARKETS.get(m.get("key"))
            if market is None:
                continue
            sides = {}
            for o in m.get("outcomes") or []:
                name = str(o.get("description") or "").strip()
                prob = decimal_to_prob(o.get("price"))
                try:
                    point = float(o.get("point"))
                except (TypeError, ValueError):
                    continue
                if not name or prob is None or point != LINE[market]:
                    continue
                sides.setdefault(name, {})[str(o.get("name") or "").lower()] = (
                    prob, float(o["price"]))
            for name, pair in sides.items():
                if "over" not in pair:
                    continue
                q = {"line": LINE[market], "odd_over": pair["over"][1],
                     "book": BOOK_LABEL[key], "source": key}
                if "under" in pair:
                    fair = devig({"over": pair["over"][0], "under": pair["under"][0]})
                    q.update(over=round(fair["over"], 4), under=round(fair["under"], 4),
                             odd_under=pair["under"][1])
                else:
                    q.update(over=round(min(pair["over"][0] / ONE_SIDED_OVERROUND, 0.99), 4),
                             fair=False)
                found.setdefault((market, name), []).append(("under" in pair, -rank, q))
    out = {}
    for (market, name), cands in found.items():
        out.setdefault(market, {})[name] = max(cands, key=lambda c: (c[0], c[1]))[2]
    return out


def load() -> dict:
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _match(block: dict, player: str):
    want = norm_name(player)
    hits = [q for name, q in (block or {}).items() if norm_name(name) == want]
    return hits[0] if len(hits) == 1 else None    # absent or ambiguous: unpriced


def quote_for(league_key: str, match_id, market: str, player: str, store=None, opening=False):
    """The latest (or, with opening=True, the first-seen) quote, or None."""
    fx = ((store if store is not None else load()).get("fixtures") or {}).get(
        fixture_key(league_key, match_id)) or {}
    return _match(((fx.get("open") if opening else fx.get("props")) or {}).get(market), player)


def checked_at(league_key: str, match_id, store=None):
    fx = ((store if store is not None else load()).get("fixtures") or {}).get(
        fixture_key(league_key, match_id)) or {}
    return (fx.get("checks") or [None])[-1]


def attach(upcoming: list, store=None) -> int:
    """Put the book's price on each upcoming shots / on-target pick. Returns count."""
    store = store if store is not None else load()
    n = 0
    for pick in upcoming:
        if pick.get("market") not in LINE or pick.get("book_price") is not None:
            continue
        q = quote_for(pick["league_key"], pick["id"], pick["market"], pick.get("player"), store)
        if q is None:
            continue
        pick["book"] = q["book"]
        pick["book_price"] = q["odd_over"]
        pick["book_p"] = q["over"]
        pick["edge"] = round(float(pick["p_pick"]) - q["over"], 4)
        pick["book_checked_at"] = checked_at(pick["league_key"], pick["id"], store)
        n += 1
    return n


def freeze(entry: dict, league_key: str, match_id, store=None) -> bool:
    """Stamp price, fair chance and the first-seen quote on a pick at lock.

    The open carries no line: a settled soccer pick's "line" is its label ("2+ shot
    attempts"), and both quotes are at the pick's fixed line anyway."""
    store = store if store is not None else load()
    market, player = entry.get("market"), entry.get("player")
    q = quote_for(league_key, match_id, market, player, store)
    if q is None:
        return False
    entry["book"], entry["book_price"], entry["book_p"] = q["book"], q["odd_over"], q["over"]
    op = quote_for(league_key, match_id, market, player, store, opening=True)
    fx = (store.get("fixtures") or {}).get(fixture_key(league_key, match_id)) or {}
    entry["open"] = ({"p": op["over"], "price": op["odd_over"], "book": op["book"],
                      "at": (fx.get("checks") or [None])[0]} if op else None)
    entry["close_at"] = (fx.get("checks") or [None])[-1]
    return True
