"""Fetch bet365 anytime-goalscorer prices for fixtures with an upcoming scorer pick.

API-Football, under its existing guards (per-run limit, and the 1,500-a-day
account floor). Only fixtures that carry an anytime-scorer pick on the published
board, kick off within WINDOW_HOURS, and were not checked in the last
RECHECK_HOURS -- so a normal day is a handful of requests. Writes
data-raw/leagues/prop_odds.json; leagues.prop_odds attaches it at publish.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from leagues import prop_odds
from leagues.api_football import Client
from leagues.names import UnknownTeam, canonical
from scripts.sync_rosters import API_LEAGUES

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "data" / "leagues" / "player_picks.json"
WINDOW_HOURS = 72.0
RECHECK_HOURS = 12.0
RUN_BUDGET = 30


def due_fixtures(board: dict, store: dict, now) -> list:
    """[(league_key, match_id, date, home, away)] needing a price check."""
    seen, out = set(), []
    fixtures = store.get("fixtures") or {}
    for p in board.get("upcoming") or []:
        if p.get("market") != "goal":
            continue
        key = prop_odds.fixture_key(p["league_key"], p["id"])
        if key in seen:
            continue
        seen.add(key)
        kick = pd.Timestamp(p["date"])
        kick = kick.tz_localize("UTC") if kick.tzinfo is None else kick
        hours = (kick - pd.Timestamp(now)).total_seconds() / 3600.0
        if not 0 < hours <= WINDOW_HOURS:
            continue
        last = (fixtures.get(key) or {}).get("checked_at")
        if last and (pd.Timestamp(now) - pd.Timestamp(last)) < pd.Timedelta(hours=RECHECK_HOURS):
            continue
        out.append((p["league_key"], p["id"], kick, p["home"], p["away"]))
    return out


def main(now=None) -> int:
    if not os.environ.get("API_FOOTBALL_KEY"):
        print("API_FOOTBALL_KEY is not set; no scorer prices this run")
        return 0
    now = now or datetime.now(timezone.utc)
    try:
        board = json.loads(BOARD.read_text(encoding="utf-8"))
    except Exception:
        print("no published player board yet; nothing to price")
        return 0
    store = prop_odds.load()
    fixtures = store.setdefault("fixtures", {})
    # Drop fixtures that kicked off more than two days ago.
    for key in [k for k, v in fixtures.items()
                if v.get("kickoff") and pd.Timestamp(v["kickoff"]) < pd.Timestamp(now) - pd.Timedelta(days=2)]:
        del fixtures[key]
    todo = due_fixtures(board, store, now)
    if not todo:
        print("no scorer pick inside the price window needs a check; no quota used")
        return 0

    client = Client(limit=RUN_BUDGET)
    listings, priced = {}, 0
    for lk, mid, kick, home, away in todo:
        league_id = API_LEAGUES.get(lk)
        day = kick.date().isoformat()
        try:
            if (lk, day) not in listings:
                listings[(lk, day)] = client.get("fixtures", league=league_id, date=day,
                                                 season=kick.year if kick.month >= 7 else kick.year - 1,
                                                 timezone="UTC")
            fid = None
            for c in listings[(lk, day)]:
                try:
                    h = canonical(c["teams"]["home"]["name"], lk)
                    a = canonical(c["teams"]["away"]["name"], lk)
                except UnknownTeam:
                    continue
                if (h, a) == (home, away):
                    fid = c["fixture"]["id"]
                    break
            if fid is None:
                print(f"  {home} v {away}: no API-Football fixture found; unpriced")
                continue
            prices = prop_odds.parse_anytime(client.get("odds", fixture=fid,
                                                        bookmaker=prop_odds.BET365))
        except RuntimeError as exc:
            print(f"  stopped early: {exc}")
            break
        fixtures[prop_odds.fixture_key(lk, mid)] = {
            "api_fixture_id": fid, "home": home, "away": away,
            "kickoff": kick.isoformat(), "checked_at": now.isoformat(timespec="seconds"),
            "anytime": prices}
        priced += 1
        print(f"  {home} v {away}: {len(prices)} bet365 scorer prices")
    store["_note"] = ("bet365 anytime-goalscorer prices from API-Football (bookmaker 8, "
                      "bet 92) for fixtures with a scorer pick. Written by "
                      "scripts/sync_soccer_prop_odds.py -- never by hand.")
    store["updated"] = now.isoformat(timespec="seconds")
    tmp = prop_odds.STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(prop_odds.STORE)
    print(f"priced {priced} fixture(s)")
    print(client.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
