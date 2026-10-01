"""MLB: box scores (StatsAPI, free), then bookmaker lines (The Odds API).

1. mlb.current.sync -- box scores for finished games not yet kept.
2. When ODDS_API_ENABLED: for games inside the window, ONE h2h /odds call for the
   whole slate (1 credit), then player props + team totals for games
   mlb.book_lines.due() says are due (<= 6 credits a check, <= 2 checks a game).

Writes data-raw/mlb/odds_api.json keyed by MLB gamePk.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mlb import book_lines as bl
from mlb import current
from oddsapi import client as oc
from oddsapi import props as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "mlb" / "odds_api.json"
MATCH_HOURS = 6.0


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


def match_event(event, games) -> dict | None:
    home, away = bl.TEAM_CODES.get(event.get("home_team")), bl.TEAM_CODES.get(event.get("away_team"))
    if not home or not away:
        return None
    start = shared.utc(event["commence_time"])
    best = None
    for g in games:
        if g["home_team"] == home and g["away_team"] == away:
            gap = abs((shared.utc(g["start"]) - start).total_seconds())
            if gap <= MATCH_HOURS * 3600 and (best is None or gap < best[0]):
                best = (gap, g)
    return best[1] if best else None


def run(client, games, now=None, store=None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = store if store is not None else load()
    kept = store.setdefault("games", {})
    for k in [k for k, v in kept.items()
              if shared.utc(v.get("start", "1970-01-01T00:00:00Z")) < now - timedelta(days=2)]:
        del kept[k]
    window = [g for g in games if g["status"] in ("Scheduled", "Pre-Game", "Warmup")
              and 0 < (shared.utc(g["start"]) - now).total_seconds() / 3600 <= bl.BOARD_HOURS]
    if not window:
        print("no MLB game inside the odds window; no credits spent")
        return store
    events = client.get(f"sports/{bl.SPORT_KEY}/events", sport="mlb", purpose="mlb: events") or []
    matched = []
    for ev in events:
        g = match_event(ev, window)
        if g is None:
            continue
        entry = kept.setdefault(str(g["game_pk"]), {"checks": [], "props": {}})
        entry.update({"start": g["start"], "home": g["home_team"], "away": g["away_team"],
                      "event_id": ev["id"], "home_name": ev["home_team"], "away_name": ev["away_team"]})
        matched.append((ev, g, entry))
    try:
        rows = client.get(f"sports/{bl.SPORT_KEY}/odds", sport="mlb", purpose="mlb: match odds",
                          est=oc.cost(1, bookmakers=3), bookmakers=",".join(bl.BOOK_ORDER),
                          markets="h2h", oddsFormat="decimal") or []
        by_id = {r.get("id"): r for r in rows}
        for ev, g, entry in matched:
            line = shared.parse_moneyline(by_id.get(ev["id"]), bl.BOOK_ORDER,
                                          ev["home_team"], ev["away_team"])
            if line:
                entry["moneyline"] = {**line, "at": now.isoformat(timespec="seconds")}
                shared.remember_open(entry, now.isoformat(timespec="seconds"))
    except oc.BudgetExceeded as exc:
        print(f"  budget stop before match odds: {exc}")
        return store
    for ev, g, entry in matched:
        why = bl.due(entry["checks"], g["start"], now)
        if why is None:
            continue
        try:
            payload = client.get(
                f"sports/{bl.SPORT_KEY}/events/{ev['id']}/odds", sport="mlb",
                purpose=f"mlb {why} check {g['away_team']}@{g['home_team']}",
                est=oc.cost(len(bl.ALL_MARKETS), bookmakers=len(bl.BOOK_ORDER)),
                bookmakers=",".join(bl.BOOK_ORDER), markets=",".join(bl.ALL_MARKETS),
                oddsFormat="decimal")
        except oc.BudgetExceeded as exc:
            print(f"  budget stop: {exc}")
            break
        except RuntimeError as exc:
            print(f"  {g['game_pk']}: {exc}")
            continue
        props = bl.parse_event(payload)
        totals = bl.parse_team_totals(payload, ev["home_team"], ev["away_team"])
        if not props and not totals:
            print(f"  {g['away_team']}@{g['home_team']} ({why}): nothing posted yet (free)")
            continue
        entry["checks"].append(now.isoformat(timespec="seconds"))
        entry["props"], entry["team_totals"] = props, totals
        entry["alt"] = bl.parse_alternates(payload)
        shared.remember_open(entry, now.isoformat(timespec="seconds"))
        print(f"  {g['away_team']}@{g['home_team']} ({why}): "
              f"{ {m: len(q) for m, q in props.items()} } totals={list(totals)}")
    store["updated"] = now.isoformat(timespec="seconds")
    store["_note"] = ("MLB bookmaker lines, team totals and match odds (Pinnacle, DraftKings, "
                      "FanDuel via The Odds API), keyed by MLB gamePk. Written by "
                      "scripts/sync_mlb_odds.py -- never by hand.")
    return store


def main() -> int:
    today = datetime.now(timezone.utc).date()
    try:
        sched = current.schedule((today - timedelta(days=4)).isoformat(),
                                 (today + timedelta(days=3)).isoformat())
    except Exception as exc:
        print(f"MLB schedule unavailable ({exc}); nothing synced")
        return 0
    print(f"box scores: {current.sync(sched)}")
    if not oc.enabled():
        print("The Odds API is switched off (ODDS_API_ENABLED); no MLB lines this run")
        return 0
    client = oc.Client()
    try:
        store = run(client, sched.to_dict("records"))
    except (oc.Disabled, oc.BudgetExceeded, RuntimeError) as exc:
        print(f"MLB odds sync stopped: {exc}")
        print(client.report())
        return 0
    save(store)
    print(client.report())
    for w in client.warnings:
        print(f"ALERT: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
