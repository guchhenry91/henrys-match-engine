"""Fetch DraftKings / FanDuel / Pinnacle yardage lines for the NFL slate.

Phase 2 of the odds budget. Runs in nfl.yml after the bet365 sync. Spends only
on games nfl.book_lines.due() says are due -- at most two checks per game, 3
credits each -- through oddsapi.Client, which enforces every agreed cap (off
switch, 200/run, 1,000/day, 40,000/month). Listing events is free.

Writes data-raw/nfl/odds_api_props.json:
  {"games": {"HOME|AWAY": {"event_id", "kickoff", "checks": [iso, ...],
                           "props": {market: {player: quote}}}}}
Kept between runs, since most runs spend nothing; games already played are
pruned. publish.book_props() merges it under bet365.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nfl import book_lines as bl
from oddsapi import client as oc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "nfl" / "odds_api_props.json"


def load() -> dict:
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def run(client, now=None, store=None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = store if store is not None else load()
    games = store.setdefault("games", {})
    # Prune anything that kicked off more than a day ago.
    for key in [k for k, g in games.items()
                if bl._utc(g.get("kickoff", "1970-01-01T00:00:00+00:00"))
                < now - timedelta(days=1)]:
        del games[key]

    events = client.get(f"sports/{bl.SPORT_KEY}/events", sport="nfl",
                        purpose="nfl: list events") or []
    spent_on = []
    for ev in sorted(events, key=lambda e: e.get("commence_time", "")):
        home, away = bl.TEAM_CODES.get(ev.get("home_team")), bl.TEAM_CODES.get(ev.get("away_team"))
        if not home or not away:
            print(f"  unknown team name in {ev.get('away_team')} @ {ev.get('home_team')}; skipped")
            continue
        key = f"{home}|{away}"
        entry = games.setdefault(key, {"checks": [], "props": {}})
        entry["event_id"], entry["kickoff"] = ev["id"], ev["commence_time"]
        why = bl.due(entry["checks"], ev["commence_time"], now)
        if why is None:
            continue
        est = oc.cost(len(bl.MARKETS), bookmakers=len(bl.BOOK_ORDER))
        try:
            payload = client.get(f"sports/{bl.SPORT_KEY}/events/{ev['id']}/odds",
                                 sport="nfl", purpose=f"nfl {why} check {key}", est=est,
                                 bookmakers=",".join(bl.BOOK_ORDER),
                                 markets=",".join(bl.MARKETS), oddsFormat="decimal")
        except oc.BudgetExceeded as exc:
            print(f"  budget stop: {exc}")
            break
        except RuntimeError as exc:
            print(f"  {key}: {exc}")
            continue
        props = bl.parse_event(payload)
        if not props:
            # Nothing posted yet. An empty reply costs no credits, so it is not
            # counted against the game's two checks.
            print(f"  {key} ({why}): no lines posted yet (free)")
            continue
        entry["checks"].append(now.isoformat(timespec="seconds"))
        if props:
            # A later check replaces the earlier lines wholesale: the freshest
            # quote is the one a pick should freeze on.
            entry["props"] = props
        counts = {m: len(q) for m, q in props.items()}
        spent_on.append(key)
        print(f"  {key} ({why}): {counts or 'no lines quoted'}")
    store["updated"] = now.isoformat(timespec="seconds")
    store["_note"] = ("NFL yardage lines from The Odds API (Pinnacle, DraftKings, "
                      "FanDuel) for players bet365 does not quote. Written by "
                      "scripts/sync_nfl_book_lines.py -- never by hand. At most two "
                      "checks per game (see nfl/book_lines.py).")
    print(f"checked {len(spent_on)} game(s) this run")
    return store


def main():
    if not oc.enabled():
        print("The Odds API is switched off (ODDS_API_ENABLED); NFL lines stay "
              "bet365 / model only this run")
        return 0
    client = oc.Client()
    try:
        store = run(client)
    except (oc.Disabled, oc.BudgetExceeded, RuntimeError) as exc:
        print(f"NFL line sync stopped: {exc}")
        print(client.report())
        return 0
    save(store)
    print(client.report())
    for warning in client.warnings:
        print(f"ALERT: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
