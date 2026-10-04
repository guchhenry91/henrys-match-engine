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
from oddsapi import props as shared

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


def frozen_games(path=None) -> set:
    """"HOME|AWAY" for every game with a prop already frozen in the NFL picks log.

    A "lock" check exists to price a pick at the moment it freezes. On a Sunday the
    1pm ET games are frozen by the 15:05 lock run, so the 16:00 lock check bought
    ~6 credits a game for picks that could no longer change (review, 2026-10-04)."""
    from nfl import picks as nfl_picks
    try:
        log = json.loads((path or nfl_picks.PICKS_LOG).read_text(encoding="utf-8"))
    except Exception:
        return set()
    out = set()
    for k, e in (log.get(nfl_picks.PROPS_KEY) or {}).items():
        parts = str((e or {}).get("game_id") or "").split("_")   # season_week_AWAY_HOME
        if not str(k).startswith("_") and len(parts) == 4:
            out.add(f"{parts[3]}|{parts[2]}")
    return out


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
    frozen = frozen_games()
    for ev in sorted(events, key=lambda e: e.get("commence_time", "")):
        home, away = bl.TEAM_CODES.get(ev.get("home_team")), bl.TEAM_CODES.get(ev.get("away_team"))
        if not home or not away:
            print(f"  unknown team name in {ev.get('away_team')} @ {ev.get('home_team')}; skipped")
            continue
        key = f"{home}|{away}"
        entry = games.setdefault(key, {"checks": [], "props": {}})
        entry["event_id"], entry["kickoff"] = ev["id"], ev["commence_time"]
        why = bl.due(entry["checks"], ev["commence_time"], now)
        if why == "lock" and key in frozen:
            print(f"  {key}: picks already frozen; lock check skipped (no credits)")
            continue
        if why is None:
            # LADDER BACKFILL: a game checked before alternate lines were fetched
            # has main lines but no ladder, so its cards cannot show a 70%+ line.
            # One alternates-only call (3 credits), once per game, inside the
            # same window a normal check would use.
            hours = (bl._utc(ev["commence_time"]) - now).total_seconds() / 3600.0
            if entry["checks"] and not entry.get("alt") and not entry.get("alt_checked") \
                    and bl.MIN_LEAD_HOURS < hours <= bl.BOARD_HOURS:
                try:
                    payload = client.get(
                        f"sports/{bl.SPORT_KEY}/events/{ev['id']}/odds", sport="nfl",
                        purpose=f"nfl ladder backfill {key}",
                        est=oc.cost(len(bl.ALT_MARKETS), bookmakers=len(bl.ALT_BOOKS)),
                        bookmakers=",".join(bl.ALT_BOOKS),
                        markets=",".join(bl.ALT_MARKETS), oddsFormat="decimal")
                except oc.BudgetExceeded as exc:
                    print(f"  budget stop: {exc}")
                    break
                except RuntimeError as exc:
                    print(f"  {key}: {exc}")
                    continue
                entry["alt_checked"] = now.isoformat(timespec="seconds")
                alts = bl.parse_alternates(payload)
                if alts:
                    entry["alt"] = alts
                print(f"  {key} (ladder backfill): "
                      f"{ {m: len(q) for m, q in alts.items()} or 'no ladders quoted'}")
            continue
        markets = list(bl.MARKETS) + list(bl.ALT_MARKETS)
        est = oc.cost(len(markets), bookmakers=len(bl.BOOK_ORDER))
        try:
            payload = client.get(f"sports/{bl.SPORT_KEY}/events/{ev['id']}/odds",
                                 sport="nfl", purpose=f"nfl {why} check {key}", est=est,
                                 bookmakers=",".join(bl.BOOK_ORDER),
                                 markets=",".join(markets), oddsFormat="decimal")
        except oc.BudgetExceeded as exc:
            print(f"  budget stop: {exc}")
            break
        except RuntimeError as exc:
            print(f"  {key}: {exc}")
            continue
        props = bl.parse_event(payload)
        alts = bl.parse_alternates(payload)
        # This check asked for the ladders too, so no backfill is ever owed for it.
        entry["alt_checked"] = now.isoformat(timespec="seconds")
        if alts:
            entry["alt"] = alts
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
            shared.remember_open(entry, now.isoformat(timespec="seconds"))
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
