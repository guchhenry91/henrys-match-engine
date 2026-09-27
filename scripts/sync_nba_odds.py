"""NBA: box scores from the NBA's CDN, then bookmaker lines from The Odds API.

1. nba.current.sync -- fetch box scores for finished games not yet kept (free).
2. The Odds API, only when switched on (ODDS_API_ENABLED) and only for games
   inside the window: one h2h /odds call for the whole slate (1 credit), then
   props for games nba.book_lines.due() says are due (4 credits, <= 2 per game).

Writes data-raw/nba/odds_api.json:
  {"games": {game_id: {"tipoff", "home", "away", "event_id", "checks": [...],
                       "props": {market: {player: quote}}, "moneyline": {...}}}}
keyed by the NBA's own game id, so a game two teams play three times a season
can never be confused with another. Every cap is enforced by oddsapi.Client.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nba import book_lines as bl
from nba import current
from oddsapi import client as oc
from oddsapi import props as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "nba" / "odds_api.json"
MATCH_HOURS = 6.0          # an Odds API event and a CDN game this close are the same game


def load() -> dict:
    try:
        return json.loads(OUT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(store: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)


def match_event(event, sched_rows) -> dict | None:
    """The CDN game this Odds API event is, by both teams and tip-off time."""
    home, away = bl.TEAM_CODES.get(event.get("home_team")), bl.TEAM_CODES.get(event.get("away_team"))
    if not home or not away:
        return None
    start = shared.utc(event["commence_time"])
    for g in sched_rows:
        if g["home_team"] == home and g["away_team"] == away and \
                abs((shared.utc(g["tipoff"]) - start).total_seconds()) <= MATCH_HOURS * 3600:
            return g
    return None


def run(client, sched_rows, now=None, store=None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = store if store is not None else load()
    games = store.setdefault("games", {})
    for gid in [g for g, v in games.items()
                if shared.utc(v.get("tipoff", "1970-01-01T00:00:00Z")) < now - timedelta(days=2)]:
        del games[gid]

    window = [g for g in sched_rows
              if g["stage"] == current.REGULAR and g["status"] == 1
              and 0 < (shared.utc(g["tipoff"]) - now).total_seconds() / 3600 <= bl.BOARD_HOURS]
    if not window:
        print("no NBA regular-season game inside the odds window; no credits spent")
        return store

    events = client.get(f"sports/{bl.SPORT_KEY}/events", sport="nba",
                        purpose="nba: list events") or []
    matched = []
    for ev in events:
        g = match_event(ev, window)
        if g is None:
            continue
        entry = games.setdefault(g["game_id"], {"checks": [], "props": {}})
        entry.update({"tipoff": g["tipoff"], "home": g["home_team"],
                      "away": g["away_team"], "event_id": ev["id"]})
        matched.append((ev, g, entry))

    # MATCH ODDS: one call, every game, 1 credit.
    try:
        rows = client.get(f"sports/{bl.SPORT_KEY}/odds", sport="nba",
                          purpose="nba: match odds", est=oc.cost(1, bookmakers=3),
                          bookmakers=",".join(bl.BOOK_ORDER), markets="h2h",
                          oddsFormat="decimal") or []
        by_id = {r.get("id"): r for r in rows}
        for ev, g, entry in matched:
            line = shared.parse_moneyline(by_id.get(ev["id"]), bl.BOOK_ORDER,
                                          ev["home_team"], ev["away_team"])
            if line:
                entry["moneyline"] = line
    except oc.BudgetExceeded as exc:
        print(f"  budget stop before match odds: {exc}")
        return store

    for ev, g, entry in matched:
        why = bl.due(entry["checks"], g["tipoff"], now)
        if why is None:
            continue
        try:
            payload = client.get(
                f"sports/{bl.SPORT_KEY}/events/{ev['id']}/odds", sport="nba",
                purpose=f"nba {why} check {g['away_team']}@{g['home_team']}",
                est=oc.cost(len(bl.MARKETS), bookmakers=len(bl.BOOK_ORDER)),
                bookmakers=",".join(bl.BOOK_ORDER), markets=",".join(bl.MARKETS),
                oddsFormat="decimal")
        except oc.BudgetExceeded as exc:
            print(f"  budget stop: {exc}")
            break
        except RuntimeError as exc:
            print(f"  {g['game_id']}: {exc}")
            continue
        props = bl.parse_event(payload)
        if not props:
            print(f"  {g['away_team']}@{g['home_team']} ({why}): no lines posted yet (free)")
            continue
        entry["checks"].append(now.isoformat(timespec="seconds"))
        entry["props"] = props
        print(f"  {g['away_team']}@{g['home_team']} ({why}): "
              f"{ {m: len(q) for m, q in props.items()} }")
    store["updated"] = now.isoformat(timespec="seconds")
    store["_note"] = ("NBA bookmaker lines (Pinnacle, DraftKings, FanDuel) and match "
                      "odds from The Odds API, keyed by the NBA game id. Written by "
                      "scripts/sync_nba_odds.py -- never by hand.")
    return store


def main() -> int:
    try:
        sched = current.schedule()
    except Exception as exc:
        print(f"NBA schedule unavailable from the CDN ({exc}); nothing synced")
        return 0
    counts = current.sync(sched)
    print(f"box scores: {counts}")
    if not oc.enabled():
        print("The Odds API is switched off (ODDS_API_ENABLED); no NBA lines this run")
        return 0
    client = oc.Client()
    try:
        store = run(client, sched.to_dict("records"))
    except (oc.Disabled, oc.BudgetExceeded, RuntimeError) as exc:
        print(f"NBA odds sync stopped: {exc}")
        print(client.report())
        return 0
    save(store)
    print(client.report())
    for warning in client.warnings:
        print(f"ALERT: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
