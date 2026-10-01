"""Book prices for the soccer shots / shots-on-target picks (leagues/shot_odds.py).

The Odds API, US books only (its soccer player props are US-only). Only fixtures
that carry an upcoming shots or on-target pick on the published board are asked
about, at most TWICE each (CLAUDE.md, "API credit budget"):

  board -- the first run within BOARD_HOURS of kickoff
  lock  -- once more inside LOCK_CHECK_HOURS, before the pick freezes

A check is 2 markets from <= 10 named books = 2 credits. A reply with no shots
lines costs nothing and is not counted, so a fixture the books have not posted yet
is simply asked again next run. The events list is a free call.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from leagues import shot_odds
from leagues.names import UnknownTeam, canonical
from oddsapi import client as oc

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "data" / "leagues" / "player_picks.json"
BOARD_HOURS = 36.0
LOCK_CHECK_HOURS = 4.0     # the second look, ahead of the 2h lock window
MIN_LEAD_HOURS = 0.5
MATCH_HOURS = 3.0          # an event's kickoff must sit this close to the fixture's


def _utc(text) -> datetime:
    t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def due(checks: list, start, now):
    hours = (_utc(start) - now).total_seconds() / 3600.0
    if hours <= MIN_LEAD_HOURS or hours > BOARD_HOURS or len(checks) >= 2:
        return None
    if not checks:
        return "board"
    if hours <= LOCK_CHECK_HOURS and now - _utc(checks[-1]) >= timedelta(hours=1):
        return "lock"
    return None


def wanted(board: dict) -> dict:
    """{fixture key: (league_key, match_id, kickoff, home, away)} with a shots pick."""
    out = {}
    for p in board.get("upcoming") or []:
        if p.get("market") in shot_odds.LINE and p.get("league_key") in shot_odds.SPORT_KEYS:
            out[shot_odds.fixture_key(p["league_key"], p["id"])] = (
                p["league_key"], p["id"], p["date"], p["home"], p["away"])
    return out


def match_event(events, league_key, kickoff, home, away):
    """The event for this fixture: kickoff within MATCH_HOURS and BOTH clubs
    resolving to ours. A club the alias map does not know is reported, never
    guessed -- a wrong join prices one fixture's players from another's."""
    best = None
    for ev in events or []:
        try:
            if (canonical(ev["home_team"], league_key) != home
                    or canonical(ev["away_team"], league_key) != away):
                continue
        except (UnknownTeam, KeyError):
            continue
        gap = abs((_utc(ev["commence_time"]) - _utc(kickoff)).total_seconds())
        if gap <= MATCH_HOURS * 3600 and (best is None or gap < best[0]):
            best = (gap, ev)
    return best[1] if best else None


def run(client, board, store, now) -> dict:
    fixtures = store.setdefault("fixtures", {})
    for k in [k for k, v in fixtures.items()
              if _utc(v.get("start", "1970-01-01T00:00:00Z")) < now - timedelta(days=3)]:
        del fixtures[k]
    todo = {k: v for k, v in wanted(board).items()
            if due((fixtures.get(k) or {}).get("checks") or [], v[2], now)}
    if not todo:
        print("no shots pick due a price check; no credits spent")
        return store
    events = {}
    for key, (league_key, mid, kickoff, home, away) in todo.items():
        sport = shot_odds.SPORT_KEYS[league_key]
        if sport not in events:
            events[sport] = client.get(f"sports/{sport}/events", sport=league_key.lower(),
                                       purpose=f"{league_key}: events") or []
        ev = match_event(events[sport], league_key, kickoff, home, away)
        entry = fixtures.setdefault(key, {"checks": []})
        entry.update({"start": str(kickoff), "home": home, "away": away})
        if ev is None:
            names = sorted({e.get("home_team") for e in events[sport]}
                           | {e.get("away_team") for e in events[sport]})
            print(f"  {home} v {away}: no matching event (book names: {names[:6]}...)")
            continue
        why = due(entry["checks"], kickoff, now)
        try:
            payload = client.get(
                f"sports/{sport}/events/{ev['id']}/odds", sport=league_key.lower(),
                purpose=f"{league_key} shots {why} {home} v {away}",
                est=oc.cost(len(shot_odds.MARKETS), bookmakers=len(shot_odds.BOOK_ORDER)),
                bookmakers=",".join(shot_odds.BOOK_ORDER),
                markets=",".join(shot_odds.MARKETS), oddsFormat="decimal")
        except oc.BudgetExceeded as exc:
            print(f"  budget stop: {exc}")
            break
        except RuntimeError as exc:
            print(f"  {home} v {away}: {exc}")
            continue
        props = shot_odds.parse(payload)
        if not props:
            print(f"  {home} v {away} ({why}): no shots lines posted yet (free)")
            continue
        entry["checks"].append(now.isoformat(timespec="seconds"))
        entry["event_id"] = ev["id"]
        entry["props"] = props
        entry.setdefault("open", props)          # the first quote seen, kept
        print(f"  {home} v {away} ({why}): "
              f"{ {m: len(q) for m, q in props.items()} } from "
              f"{sorted({q['book'] for b in props.values() for q in b.values()})}")
    store["updated"] = now.isoformat(timespec="seconds")
    store["_note"] = ("Shots / on-target player prices (US books via The Odds API), keyed "
                      "LEAGUE#match_id. Written by scripts/sync_soccer_shot_odds.py -- "
                      "never by hand.")
    return store


def main() -> int:
    if not oc.enabled():
        print("The Odds API is switched off (ODDS_API_ENABLED); no shots prices this run")
        return 0
    try:
        board = json.loads(BOARD.read_text(encoding="utf-8"))
    except Exception:
        print("no published player board yet; nothing to price")
        return 0
    client = oc.Client()
    try:
        store = run(client, board, shot_odds.load(), datetime.now(timezone.utc))
    except (oc.Disabled, oc.BudgetExceeded, RuntimeError) as exc:
        print(f"shots odds sync stopped: {exc}")
        print(client.report())
        return 0
    shot_odds.STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = shot_odds.STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(shot_odds.STORE)
    print(client.report())
    for w in client.warnings:
        print(f"ALERT: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
