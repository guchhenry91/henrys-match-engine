"""Phase 1 of the odds plan: WHICH books and regions actually carry the props we
need, per sport -- measured once, for at most PROBE_CAP credits, before any
scheduled job spends a thing.

For each sport: /events (free) finds the soonest upcoming game, then one
/events/{id}/odds call per region asks for exactly the markets the site
publishes. An empty reply costs nothing (the API charges for markets RETURNED),
so asking a region that has no props is free. The Champions League is asked for
match odds only (one /odds call, 1 credit), since the API lists no UCL props.

Also re-tests API-Football's own pre-match odds for a Premier League fixture:
bet types for anytime scorer (92) and player shots / on target (240-243) exist
and bet365 is listed, but an earlier check found pre-match odds EMPTY. If they
are there now, soccer props can come from bet365 through the quota we already
pay for, and cost no Odds API credits at all.

Manual only (.github/workflows/odds-probe.yml). Prints no URL -- the Odds API
key travels in one. Writes data-raw/odds_api/coverage_probe.json.
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from oddsapi import client as oc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data-raw" / "odds_api" / "coverage_probe.json"
PROBE_CAP = 60

SOCCER_PROPS = ["player_shots", "player_shots_on_target", "player_goal_scorer_anytime"]
# name -> (sport key, markets we publish, regions to ask)
SPORTS = {
    "nfl": ("americanfootball_nfl",
            ["player_reception_yds", "player_rush_yds", "player_pass_yds",
             "player_anytime_td"], ["us", "uk", "eu"]),
    "pl": ("soccer_epl", SOCCER_PROPS, ["us", "uk", "eu"]),
    "laliga": ("soccer_spain_la_liga", SOCCER_PROPS, ["us", "uk"]),
    "bundesliga": ("soccer_germany_bundesliga", SOCCER_PROPS, ["us", "uk"]),
    "seriea": ("soccer_italy_serie_a", SOCCER_PROPS, ["us", "uk"]),
    "ligue1": ("soccer_france_ligue_one", SOCCER_PROPS, ["us", "uk"]),
    "nba": ("basketball_nba",
            ["player_points", "player_rebounds", "player_assists", "player_threes"],
            ["us"]),
}


def _soonest(events, now):
    ahead = [e for e in events or []
             if datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
             > now + timedelta(hours=1)]
    return min(ahead, key=lambda e: e["commence_time"]) if ahead else None


def _summarise(payload):
    """bookmaker -> market -> number of distinct players quoted."""
    out = defaultdict(dict)
    for book in (payload or {}).get("bookmakers", []):
        for market in book.get("markets", []):
            players = {o.get("description") or o.get("name")
                       for o in market.get("outcomes", [])}
            out[book["key"]][market["key"]] = len(players)
    return dict(out)


def probe_odds_api(now):
    c = oc.Client()
    result = {"sports_active": [], "by_sport": {}}
    active = c.get("sports", sport="-", purpose="probe: sport list") or []
    result["sports_active"] = sorted(s["key"] for s in active if s.get("active"))
    print(f"active sport keys: {len(result['sports_active'])}")

    for name, (key, markets, regions) in SPORTS.items():
        entry = result["by_sport"].setdefault(name, {"sport_key": key, "regions": {}})
        if key not in result["sports_active"]:
            entry["note"] = "not active right now"
            print(f"\n{name}: {key} not active right now -- skipped, 0 credits")
            continue
        events = c.get(f"sports/{key}/events", sport=name, purpose="probe: events")
        ev = _soonest(events, now)
        if ev is None:
            entry["note"] = "no upcoming event"
            print(f"\n{name}: no upcoming event -- skipped")
            continue
        entry["event"] = f"{ev['away_team']} @ {ev['home_team']} {ev['commence_time']}"
        print(f"\n{name}: {entry['event']}")
        for region in regions:
            est = oc.cost(len(markets))
            if c.run_used + est > PROBE_CAP:
                entry["regions"][region] = "skipped: probe cap"
                print(f"  {region}: skipped, probe cap of {PROBE_CAP} reached")
                continue
            try:
                payload = c.get(f"sports/{key}/events/{ev['id']}/odds", sport=name,
                                purpose=f"probe: props {region}", est=est,
                                regions=region, markets=",".join(markets),
                                oddsFormat="decimal")
            except RuntimeError as exc:
                entry["regions"][region] = f"error: {exc}"
                print(f"  {region}: {exc}")
                continue
            summary = _summarise(payload)
            entry["regions"][region] = summary
            if not summary:
                print(f"  {region}: no props quoted (free)")
            for book, mk in sorted(summary.items()):
                print(f"  {region}: {book:<22} " + ", ".join(
                    f"{m}={n}" for m, n in sorted(mk.items())))

    # Champions League match odds: one call, one market, one region.
    if "soccer_uefa_champs_league" in result["sports_active"] and \
            c.run_used + 1 <= PROBE_CAP:
        rows = c.get("sports/soccer_uefa_champs_league/odds", sport="ucl",
                     purpose="probe: ucl h2h", est=oc.cost(1), regions="uk",
                     markets="h2h", oddsFormat="decimal") or []
        books = sorted({b["key"] for r in rows for b in r.get("bookmakers", [])})
        result["by_sport"]["ucl"] = {"events_priced": len(rows), "uk_books": books}
        print(f"\nucl: {len(rows)} events priced h2h in uk by {books}")

    result["credits_spent"] = c.run_used
    result["account_used"] = c.month_used()
    print("\n" + c.report())
    return result


def probe_api_football():
    """Does API-Football now return bet365 pre-match player odds? ~5 requests."""
    if not os.environ.get("API_FOOTBALL_KEY"):
        return {"skipped": "no API_FOOTBALL_KEY"}
    from leagues.api_football import Client
    client = Client(limit=6)
    out = {"fixtures": []}
    try:
        fixtures = client.get("fixtures", league=39, season=2026, next=2)
        for fx in fixtures:
            fid = fx["fixture"]["id"]
            label = f"{fx['teams']['home']['name']} v {fx['teams']['away']['name']}"
            odds = client.get("odds", fixture=fid, bookmaker=8)      # 8 = bet365
            bets = [b["name"] for r in odds for bm in r.get("bookmakers", [])
                    for b in bm.get("bets", [])]
            player = [b for b in bets if any(w in b.lower() for w in
                                             ("scorer", "shot", "player"))]
            out["fixtures"].append({"fixture": label, "date": fx["fixture"]["date"],
                                    "bet365_bet_types": len(bets),
                                    "player_markets": player})
            print(f"API-Football {label}: {len(bets)} bet365 bet types; "
                  f"player markets: {player or 'none'}")
    except RuntimeError as exc:
        out["error"] = str(exc)
        print(f"API-Football probe stopped: {exc}")
    print(client.report())
    return out


def main():
    now = datetime.now(timezone.utc)
    report = {"probed_at": now.isoformat(timespec="seconds"),
              "probe_cap": PROBE_CAP,
              "_note": ("Phase 1 coverage probe (scripts/probe_odds_api.py). "
                        "Which books/regions quote the props the site publishes. "
                        "Generated -- never hand-edited.")}
    try:
        report["odds_api"] = probe_odds_api(now)
    except (oc.Disabled, oc.BudgetExceeded, RuntimeError) as exc:
        report["odds_api"] = {"error": str(exc)}
        print(f"The Odds API probe stopped: {exc}")
    report["api_football"] = probe_api_football()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote data-raw/odds_api/{OUT.name}")


if __name__ == "__main__":
    main()
