"""Fetch bet365's Champions League 1X2 prices from API-Football (ucl/market.py).

One paged /odds call for the whole competition (10 fixtures a page; bet365 and the
match-winner bet only), at most every RECHECK_HOURS -- a handful of requests a run,
under the API-Football per-run limit and 1,500-a-day account floor. Writes
data-raw/ucl/odds.json; ucl.publish applies it.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from leagues.api_football import Client
from ucl import config, market

RECHECK_HOURS = 2.0
MAX_PAGES = 8


def due(store: dict, now: datetime) -> bool:
    last = store.get("checked_at")
    if not last:
        return True
    try:
        then = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        return True
    return now - then >= timedelta(hours=RECHECK_HOURS) or then > now


def main(now=None, client=None) -> int:
    now = now or datetime.now(timezone.utc)
    store = market.load()
    if not due(store, now):
        print(f"Champions League prices checked under {RECHECK_HOURS:.0f}h ago; no quota used")
        return 0
    if client is None:
        if not os.environ.get("API_FOOTBALL_KEY"):
            print("API_FOOTBALL_KEY is not set; no Champions League prices this run")
            return 0
        client = Client(limit=MAX_PAGES + 2)
    found = {}
    try:
        for page in range(1, MAX_PAGES + 1):
            rows = client.get("odds", league=config.API_LEAGUE_ID, season=config.CURRENT_SEASON,
                              bookmaker=market.BET365, bet=market.MATCH_WINNER, page=page)
            found.update(market.parse(rows))
            if len(rows) < 10:
                break
    except Exception as exc:                             # noqa: BLE001
        print(f"Champions League prices: fetch failed ({exc}); keeping the last ones")
        if not found:
            return 0
    at = now.isoformat(timespec="seconds")
    fixtures = store.get("fixtures") or {}
    for fid, odds in found.items():
        prev = fixtures.get(fid) or {}
        fixtures[fid] = {"odds": odds, "checked_at": at,
                         # the first price seen, kept for a closing-line comparison
                         "open": prev.get("open") or odds}
    store.update({"checked_at": at, "fixtures": fixtures})
    market.STORE.parent.mkdir(parents=True, exist_ok=True)
    market.STORE.write_text(json.dumps(store, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Champions League prices: {len(found)} fixture(s) priced by bet365")
    return 0


if __name__ == "__main__":
    sys.exit(main())
